import importlib
import logging
import os.path as osp
from collections import OrderedDict

import mmcv
import torch
import torch.nn.functional as F

import datsr.models.networks as networks
import datsr.utils.metrics as metrics
from datsr.utils import ProgressBar, tensor2img, img2tensor
from inference.tiling import tile_lr, get_ref_tile
from inference.blending import reconstruct_hr

from .sr_model import SRModel

loss_module = importlib.import_module('datsr.models.losses')
logger = logging.getLogger('base')
psnr_list = []


class RefRestorationModel(SRModel):

    def __init__(self, opt):
        super(RefRestorationModel, self).__init__(opt)

        # net_map does not have any trainable parameters.
        self.net_map = networks.define_net_map(opt)
        self.net_map = self.model_to_device(self.net_map)

        # define network for feature extraction
        self.net_extractor = networks.define_net_extractor(opt)
        self.net_extractor = self.model_to_device(self.net_extractor)
        self.print_network(self.net_extractor)

        # load pretrained feature extractor
        load_path = self.opt['path'].get('pretrain_model_feature_extractor',
                                         None)
        if load_path is not None:
            self.load_network(self.net_extractor, load_path,
                              self.opt['path']['strict_load'])

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_model_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path']['strict_load'])
        if self.is_train:
            self.net_g.train()

            # optimizers
            train_opt = self.opt['train']
            weight_decay_g = train_opt.get('weight_decay_g', 0)
            optim_params_g = []
            optim_params_offset = []
            optim_params_relu2_offset = []
            optim_params_relu3_offset = []
            if train_opt.get('lr_relu3_offset', None):
                optim_params_relu3_offset = []
            for name, v in self.net_g.named_parameters():
                if v.requires_grad:
                    if 'offset' in name:
                        if 'small' in name:
                            logger.info(name)
                            optim_params_relu3_offset.append(v)
                        elif 'medium' in name:
                            logger.info(name)
                            optim_params_relu2_offset.append(v)
                        else:
                            optim_params_offset.append(v)
                    else:
                        optim_params_g.append(v)

            self.optimizer_g = torch.optim.Adam(
                [{
                    'params': optim_params_g
                }, {
                    'params': optim_params_offset,
                    'lr': train_opt['lr_offset']
                }, {
                    'params': optim_params_relu3_offset,
                    'lr': train_opt['lr_relu3_offset']
                }, {
                    'params': optim_params_relu2_offset,
                    'lr': train_opt['lr_relu2_offset']
                }],
                lr=train_opt['lr_g'],
                weight_decay=weight_decay_g,
                betas=train_opt['beta_g'])

            self.optimizers.append(self.optimizer_g)

    def init_training_settings(self):
        train_opt = self.opt['train']

        if self.opt.get('network_d', None):
            # define network net_d
            self.net_d = networks.define_net_d(self.opt)
            self.net_d = self.model_to_device(self.net_d)
            self.print_network(self.net_d)
            # load pretrained models
            load_path = self.opt['path'].get('pretrain_model_d', None)
            if load_path is not None:
                self.load_network(self.net_d, load_path,
                                  self.opt['path']['strict_load'])
        else:
            logger.info('No discriminator.')
            self.net_d = None

        if self.net_d:
            self.net_d.train()

        # define losses
        if train_opt['pixel_weight'] > 0:
            cri_pix_cls = getattr(loss_module, train_opt['pixel_criterion'])
            self.cri_pix = cri_pix_cls(
                loss_weight=train_opt['pixel_weight'],
                reduction='mean').to(self.device)
        else:
            logger.info('Remove pixel loss.')
            self.cri_pix = None

        if train_opt.get('perceptual_opt', None):
            cri_perceptual_cls = getattr(loss_module, 'PerceptualLoss')
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            logger.info('Remove perceptual loss.')
            self.cri_perceptual = None

        if train_opt.get('style_opt', None):
            cri_style_cls = getattr(loss_module, 'PerceptualLoss')
            self.cri_style = cri_style_cls(**train_opt['style_opt']).to(
                self.device)
        else:
            logger.info('Remove style loss.')
            self.cri_style = None

        if train_opt.get('texture_opt', None):
            cri_texture_cls = getattr(loss_module, 'TextureLoss')
            self.cri_texture = cri_texture_cls(**train_opt['texture_opt']).to(
                self.device)
        else:
            logger.info('Remove texture loss.')
            self.cri_texture = None

        if train_opt.get('gan_type', None):
            cri_gan_cls = getattr(loss_module, 'GANLoss')
            self.cri_gan = cri_gan_cls(
                train_opt['gan_type'],
                real_label_val=1.0,
                fake_label_val=0.0,
                loss_weight=train_opt['gan_weight']).to(self.device)

            if train_opt['grad_penalty_weight'] > 0:
                cri_grad_penalty_cls = getattr(loss_module,
                                               'GradientPenaltyLoss')
                self.cri_grad_penalty = cri_grad_penalty_cls(
                    loss_weight=train_opt['grad_penalty_weight']).to(
                        self.device)
            else:
                logger.info('Remove gradient penalty.')
                self.cri_grad_penalty = None
        else:
            logger.info('Remove GAN loss.')
            self.cri_gan = None

        # we need to train the net_g with only pixel loss for several steps
        self.net_g_pretrain_steps = train_opt['net_g_pretrain_steps']
        self.net_d_steps = train_opt['net_d_steps'] if train_opt[
            'net_d_steps'] else 1
        self.net_d_init_steps = train_opt['net_d_init_steps'] if train_opt[
            'net_d_init_steps'] else 0

        # optimizers
        if self.net_d:
            weight_decay_d = train_opt.get('weight_decay_d', 0)
            self.optimizer_d = torch.optim.Adam(
                self.net_d.parameters(),
                lr=train_opt['lr_d'],
                weight_decay=weight_decay_d,
                betas=train_opt['beta_d'])
            self.optimizers.append(self.optimizer_d)

        # check the schedulers
        self.setup_schedulers()

        self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.img_in_lq = data['img_in_lq'].to(self.device)
        self.img_ref = data['img_ref'].to(self.device)
        self.gt = data['img_in'].to(self.device)  # gt
        self.match_img_in = data['img_in_up'].to(self.device)
        if 'img_in_ori' in data:
            self.gt_ori = data['img_in_ori'].to(self.device)


    def optimize_parameters(self, step):

        self.features = self.net_extractor(self.match_img_in, self.img_ref)
        self.pre_offset, self.img_ref_feat = self.net_map(
            self.features, self.img_ref)
        self.output = self.net_g(self.img_in_lq, self.pre_offset,
                                 self.img_ref_feat)

        if step <= self.net_g_pretrain_steps:
            # pretrain the net_g with pixel Loss
            self.optimizer_g.zero_grad()
            l_pix = self.cri_pix(self.output, self.gt)
            l_pix.backward()
            self.optimizer_g.step()

            # set log
            self.log_dict['l_pix'] = l_pix.item()
        else:
            if self.net_d:
                # train net_d
                self.optimizer_d.zero_grad()
                for p in self.net_d.parameters():
                    p.requires_grad = True

                # compute WGAN loss
                real_d_pred = self.net_d(self.gt)
                l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                self.log_dict['l_d_real'] = l_d_real.item()
                self.log_dict['out_d_real'] = torch.mean(real_d_pred.detach())
                # fake
                fake_d_pred = self.net_d(self.output.detach())
                l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                self.log_dict['l_d_fake'] = l_d_fake.item()
                self.log_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
                l_d_total = l_d_real + l_d_fake
                if self.cri_grad_penalty:
                    l_grad_penalty = self.cri_grad_penalty(
                        self.net_d, self.gt, self.output)
                    self.log_dict['l_grad_penalty'] = l_grad_penalty.item()
                    l_d_total += l_grad_penalty
                l_d_total.backward()
                self.optimizer_d.step()

            # train net_g
            self.optimizer_g.zero_grad()
            if self.net_d:
                for p in self.net_d.parameters():
                    p.requires_grad = False

            l_g_total = 0
            if (step - self.net_g_pretrain_steps) % self.net_d_steps == 0 and (
                    step - self.net_g_pretrain_steps) > self.net_d_init_steps:
                if self.cri_pix:
                    l_g_pix = self.cri_pix(self.output, self.gt)
                    l_g_total += l_g_pix
                    self.log_dict['l_g_pix'] = l_g_pix.item()
                if self.cri_perceptual:
                    l_g_percep, _ = self.cri_perceptual(self.output, self.gt)
                    l_g_total += l_g_percep
                    self.log_dict['l_g_percep'] = l_g_percep.item()
                if self.cri_style:
                    _, l_g_style = self.cri_style(self.output, self.gt)
                    l_g_total += l_g_style
                    self.log_dict['l_g_style'] = l_g_style.item()
                if self.cri_texture:
                    l_g_texture = self.cri_texture(self.output, self.maps,
                                                   self.weights)
                    l_g_total += l_g_texture
                    self.log_dict['l_g_texture'] = l_g_texture.item()

                if self.net_d:
                    # gan loss
                    fake_g_pred = self.net_d(self.output)
                    l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                    l_g_total += l_g_gan
                    self.log_dict['l_g_gan'] = l_g_gan.item()

                l_g_total.backward()
                self.optimizer_g.step()

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.features = self.net_extractor(self.match_img_in, self.img_ref)
            self.pre_offset, self.img_ref_feat = self.net_map(self.features, self.img_ref)
            self.output = self.net_g(self.img_in_lq, self.pre_offset, self.img_ref_feat)

        self.net_g.train()

    def test_tiled(self):
        """
        Max-Scale Square Margin Tiling + Overlap Blending 기반 추론.

        YAML의 'tiling' 섹션이 활성화(enable: true)되어 있을 때 호출된다.
        LR을 overlap 타일로 분할하고, 각 타일에 대응하는 Ref 크롭을 잘라낸 뒤
        DATSR 파이프라인(net_extractor → net_map → net_g)을 개별 실행한다.
        이후 Gaussian/Linear 가중치 맵으로 SR 타일들을 블렌딩하여 최종 HR을 복원한다.

        Batch size = 1 을 가정한다.
        """
        tiling_opt  = self.opt['tiling']
        tile_size   = int(tiling_opt['lr_tile_size'])
        overlap     = int(tiling_opt['lr_overlap_pixels'])
        stride      = tile_size - overlap
        margin      = int(tiling_opt['ref_search_margin'])
        pad_mode    = tiling_opt.get('padding_mode') or 'reflect'
        blend_meth  = tiling_opt.get('blending_method') or 'gaussian'
        sigma       = float(tiling_opt.get('gaussian_sigma') or 0.5)
        scale       = self.opt['scale']  # 4

        # (C, H, W) — batch dim 제거
        lr         = self.img_in_lq[0].cpu()
        ref        = self.img_ref[0].cpu()
        match_full = self.match_img_in[0].cpu()   # (C, H*scale, W*scale)

        # ── 1. LR 타일링 ──────────────────────────────────────
        lr_tiles, positions, original_shape = tile_lr(
            lr, tile_size, stride, pad_mode
        )

        sr_tiles = []
        self.net_g.eval()
        self.net_extractor.eval()
        self.net_map.eval()

        with torch.no_grad():
            for lr_tile, (r, c) in zip(lr_tiles, positions):

                # ── 2. Ref 타일 크롭 (Max-Scale Margin Tiling) ──
                ref_tile = get_ref_tile(
                    ref, original_shape, r, c, tile_size, margin, pad_mode
                )  # (C, final_size, final_size)

                # ── 3. match_img_in 타일 크롭 (HR 스케일) ────────
                r_hr = r * scale
                c_hr = c * scale
                ms   = tile_size * scale          # match tile size at HR scale
                _, Mh, Mw = match_full.shape
                pb = max(0, r_hr + ms - Mh)
                pr = max(0, c_hr + ms - Mw)
                mwork = (
                    F.pad(match_full.unsqueeze(0), (0, pr, 0, pb), mode=pad_mode).squeeze(0)
                    if pb > 0 or pr > 0 else match_full
                )
                match_tile = mwork[:, r_hr:r_hr + ms, c_hr:c_hr + ms]

                # ref_tile 크기에 맞춰 match_tile 을 리사이즈
                # → net_extractor(ContrasExtractorSep) 의 두 입력이 동일 공간 스케일을 공유
                _, rh, rw = ref_tile.shape
                if match_tile.shape[1] != rh or match_tile.shape[2] != rw:
                    match_tile = F.interpolate(
                        match_tile.unsqueeze(0).float(),
                        size=(rh, rw),
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(0)

                # ── 4. 디바이스로 이동 + 배치 차원 추가 ─────────
                lr_b    = lr_tile.unsqueeze(0).to(self.device)
                ref_b   = ref_tile.unsqueeze(0).to(self.device)
                match_b = match_tile.unsqueeze(0).to(self.device)

                # ── 5. DATSR 파이프라인 실행 ──────────────────────
                features              = self.net_extractor(match_b, ref_b)
                pre_offset, ref_feat  = self.net_map(features, ref_b)
                sr_tile               = self.net_g(lr_b, pre_offset, ref_feat)

                sr_tiles.append(sr_tile[0].cpu())

        self.net_g.train()

        # ── 6. Overlap Blending → self.output ─────────────────
        hr = reconstruct_hr(
            sr_tiles, positions, original_shape, scale, blend_meth, sigma
        )
        self.output = hr.unsqueeze(0).to(self.device)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['img_in_lq'] = self.img_in_lq.detach().cpu()
        out_dict['rlt'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()

        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        if self.net_d:
            self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        pbar = ProgressBar(len(dataloader))
        avg_psnr = 0.
        avg_psnr_y = 0.
        avg_ssim_y = 0.
        avg_lpips = 0.
        dataset_name = dataloader.dataset.opt['name']
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            tiling_cfg = self.opt.get('tiling')
            if tiling_cfg and tiling_cfg.get('enable'):
                self.test_tiled()
            else:
                self.test()

            visuals = self.get_current_visuals()
            sr_img, gt_img = tensor2img([visuals['rlt'], visuals['gt']])

            if 'multi' in dataset_name:  
                _, h, w, _ = self.gt_ori.shape
                sr_img = sr_img[:h, :w, :]
                gt_img = gt_img[:h, :w, :]

            if 'padding' in val_data.keys():
                padding = val_data['padding']
                original_size = val_data['original_size']
                if padding:
                    sr_img = sr_img[:original_size[0], :original_size[1]]

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f"{img_name}_{self.opt['name']}.png")
                    if self.opt['suffix']:
                        save_img_path = save_img_path.replace(
                            '.png', f'_{self.opt["suffix"]}.png')
                mmcv.imwrite(sr_img, save_img_path)

            # tentative for out of GPU memory
            del self.img_in_lq
            del self.output
            del self.gt
            torch.cuda.empty_cache()

            # calculate PSNR
            psnr = metrics.psnr(
                sr_img, gt_img, crop_border=self.opt['crop_border'])
            psnr_list.append(psnr)
            avg_psnr += psnr
            sr_img_y = metrics.bgr2ycbcr(sr_img / 255., only_y=True)
            gt_img_y = metrics.bgr2ycbcr(gt_img / 255., only_y=True)
            psnr_y = metrics.psnr(
                sr_img_y * 255,
                gt_img_y * 255,
                crop_border=self.opt['crop_border'])
            avg_psnr_y += psnr_y
            ssim_y = metrics.ssim(
                sr_img_y * 255,
                gt_img_y * 255,
                crop_border=self.opt['crop_border'])
            avg_ssim_y += ssim_y

            if not self.is_train:
                logger.info(f'# img {img_name} # PSNR: {psnr:.4e} '
                            f'# PSNR_Y: {psnr_y:.4e} # SSIM_Y: {ssim_y:.4e}.')

            pbar.update(f'Test {img_name}')

        avg_psnr = avg_psnr / (idx + 1)
        avg_psnr_y = avg_psnr_y / (idx + 1)
        avg_ssim_y = avg_ssim_y / (idx + 1)
        
        # log
        logger.info(f'# Validation {dataset_name} # PSNR: {avg_psnr:.4e} '
                    f'# PSNR_Y: {avg_psnr_y:.4e} # SSIM_Y: {avg_ssim_y:.4e}.')
                   
        if tb_logger:
            tb_logger.add_scalar('psnr', avg_psnr, current_iter)
            tb_logger.add_scalar('psnr_y', avg_psnr_y, current_iter)
            tb_logger.add_scalar('ssim_y', avg_ssim_y, current_iter)
