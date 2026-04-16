# ── Compatibility Patch: MMCV 2.x + torchvision 0.16+ ───────────────────────
# 반드시 다른 모든 import 보다 먼저 실행되어야 합니다.
import os as _os, sys as _sys, types as _types

# Fix 1: mmcv.scandir — MMCV 2.x 에서 최상위 네임스페이스에서 제거됨
#   datsr/{data,models,models/archs}/__init__.py 가 모듈 로드 시점에 즉시
#   mmcv.scandir 을 호출하므로, datsr 패키지 import 전에 패치 완료 필요.
import mmcv as _mmcv
if not hasattr(_mmcv, 'scandir'):
    def _mmcv_scandir(dir_path, suffix=None, recursive=False,
                      case_sensitive=True):
        """MMCV 1.x mmcv.scandir 드롭인 대체 구현."""
        if isinstance(suffix, str):
            suffix = (suffix,)
        for _root, _dirs, _files in _os.walk(str(dir_path)):
            _dirs[:] = sorted(d for d in _dirs if not d.startswith('.'))
            for _fname in sorted(_files):
                if _fname.startswith('.'):
                    continue
                _rel = _os.path.relpath(
                    _os.path.join(_root, _fname), str(dir_path))
                if suffix is None:
                    yield _rel
                else:
                    _chk = _rel if case_sensitive else _rel.lower()
                    _suf = (suffix if case_sensitive
                            else tuple(s.lower() for s in suffix))
                    if _chk.endswith(_suf):
                        yield _rel
            if not recursive:
                break
    _mmcv.scandir = _mmcv_scandir

# Fix 2: torchvision 0.16+ 에서 functional_tensor 서브모듈 자체가 제거됨.
#   해당 모듈을 import 하는 코드를 위해 rgb_to_grayscale 을 포함한
#   스텁 모듈을 sys.modules 에 선제적으로 주입.
_ft_name = 'torchvision.transforms.functional_tensor'
try:
    import torchvision.transforms.functional_tensor as _ft
    if not hasattr(_ft, 'rgb_to_grayscale'):
        import torchvision.transforms.functional as _tvf
        _ft.rgb_to_grayscale = _tvf.rgb_to_grayscale
except ImportError:
    import torchvision.transforms.functional as _tvf
    _stub = _types.ModuleType(_ft_name)
    _stub.rgb_to_grayscale = _tvf.rgb_to_grayscale
    _sys.modules[_ft_name] = _stub
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import logging
import math
import os.path as osp
import random
import time

import torch
from mmcv.runner import get_time_str, init_dist

from mmsr.data import create_dataloader, create_dataset
from mmsr.data.data_sampler import DistIterSampler
from mmsr.models import create_model
from mmsr.utils import (MessageLogger, get_root_logger, init_tb_logger,
                        make_exp_dirs, set_random_seed)
from mmsr.utils.options import dict2str, dict_to_nonedict, parse
from mmsr.utils.util import check_resume


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=True)

    # distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    # load resume states if exists
    if opt['path'].get('resume_state', None):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    else:
        resume_state = None

    # mkdir and loggers
    if resume_state is None:
        make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='base', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))
    # initialize tensorboard logger
    tb_logger = None
    if opt['use_tb_logger'] and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir='./tb_logger/' + opt['name'])

    # convert to NoneDict, which returns None for missing keys
    opt = dict_to_nonedict(opt)

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info(f'Random seed: {seed}')
    set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # create train and val dataloaders
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            # dataset_ratio: enlarge the size of datasets for each epoch
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_size = int(
                math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank,
                                                dataset_enlarge_ratio)
                total_epochs = total_iters / (
                    train_size * dataset_enlarge_ratio)
                total_epochs = int(math.ceil(total_epochs))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt,
                                             train_sampler)
            logger.info(
                f'Number of train images: {len(train_set)}, iters: {train_size}'
            )
            logger.info(
                f'Total epochs needed: {total_epochs} for iters {total_iters}')
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            logger.info(
                f"Number of val images/folders in {dataset_opt['name']}: "
                f'{len(val_set)}')
        else:
            raise NotImplementedError(f'Phase {phase} is not recognized.')
    assert train_loader is not None

    # create model
    model = create_model(opt)

    # resume training
    if resume_state:
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_iter = 0
        start_epoch = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = 0, 0

    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train']['warmup_iter'])
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_time = time.time() - iter_time
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # validation
            if opt['datasets'][
                    'val'] and current_iter % opt['val']['val_freq'] == 0:
                model.validation(val_loader, current_iter, tb_logger,
                                 opt['val']['save_img'])

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            data_time = time.time()
            iter_time = time.time()
        # end of iter
    # end of epoch

    logger.info('End of training.')
    logger.info('Saving the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 for the latest

    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()
