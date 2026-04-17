"""
DATSR Standalone Inference Pipeline
────────────────────────────────────────────────────────────────
GT(정답) 이미지 없이 실제 LR + Ref 쌍만으로 SR 추론을 수행한다.
벤치마크 평가용 test.py 와 완전히 분리된 독립 스크립트.

특징:
  - 이름 매칭(Name Matching): LR/Ref 폴더에서 파일명이 동일한 쌍만 추론
  - Max-Scale Square Margin Tiling + Overlap Blending
  - Robust Weight Loader (DCNv2 키 자동 변환 포함)
  - Dataset 클래스 불필요 — 이미지 직접 로드

사용법:
    python inference.py -opt options/inference/inference_datsr_tiling.yml
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import compat  # noqa: F401 — MMCV 2.x / torchvision 0.16+ 호환 패치 (반드시 최상단)

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

# ── 프로젝트 루트를 sys.path 에 추가 ──────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from inference.tiling import tile_lr, get_ref_tile
from inference.blending import reconstruct_hr
from datsr.models.weight_loader import load_robust_state_dict
import datsr.models.networks as networks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("inference")

SCALE = 4   # DATSR 고정 업스케일 배율


# ══════════════════════════════════════════════════════════════
# Phase 1 헬퍼: YAML 파서
# ══════════════════════════════════════════════════════════════

def load_opt(opt_path: str) -> dict:
    with open(opt_path, "r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)
    return opt


# ══════════════════════════════════════════════════════════════
# Phase 2: 이름 매칭 I/O
# ══════════════════════════════════════════════════════════════

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect_matched_pairs(lr_dir: str, ref_dir: str):
    """
    LR 폴더와 Ref 폴더를 탐색하여 파일명(확장자 무시)이 동일한 쌍을 반환한다.

    Returns:
        pairs: list of (lr_path, ref_path)
        skipped: list of lr_path that had no matching ref
    """
    lr_root  = Path(lr_dir)
    ref_root = Path(ref_dir)

    if not lr_root.exists():
        raise FileNotFoundError(f"LR 디렉토리를 찾을 수 없습니다: {lr_root}")
    if not ref_root.exists():
        raise FileNotFoundError(f"Ref 디렉토리를 찾을 수 없습니다: {ref_root}")

    # Ref 폴더: stem → path 인덱스 구축
    ref_index: dict = {}
    for p in sorted(ref_root.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            ref_index[p.stem] = p

    pairs, skipped = [], []
    for lr_path in sorted(lr_root.iterdir()):
        if lr_path.suffix.lower() not in IMG_EXTS:
            continue
        stem = lr_path.stem
        if stem in ref_index:
            pairs.append((lr_path, ref_index[stem]))
        else:
            skipped.append(lr_path)
            logger.warning(f"[SKIP] Ref not found for: {lr_path.name}")

    logger.info(
        f"[Matching] {len(pairs)} pair(s) matched, {len(skipped)} skipped"
    )
    return pairs, skipped


# ══════════════════════════════════════════════════════════════
# Phase 3-A: 이미지 로드 유틸
# ══════════════════════════════════════════════════════════════

def load_image_tensor(path) -> torch.Tensor:
    """
    이미지를 (C, H, W) float32 [0,1] RGB 텐서로 로드한다.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"이미지를 읽을 수 없습니다: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)   # (C, H, W)


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """
    (C, H, W) float32 [0,1] 텐서 → uint8 BGR numpy 이미지.
    """
    arr = t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ══════════════════════════════════════════════════════════════
# Phase 3-B: 모델 로드
# ══════════════════════════════════════════════════════════════

def build_models(opt: dict, device: torch.device):
    """
    YAML 설정에 따라 세 네트워크를 생성하고 가중치를 로드한다.
    Robust Loader 를 통해 DCNv2 키 불일치를 자동 해결한다.
    """
    # net_g (SwinUnetv3)
    net_g = networks.define_net_g(opt).to(device).eval()

    # net_extractor (ContrasExtractorSep)
    net_extractor = networks.define_net_extractor(opt).to(device).eval()

    # net_map (FlowSimCorrespondenceGenerationArch)
    net_map = networks.define_net_map(opt).to(device).eval()

    # 가중치 로드
    ckpt_g   = opt.get("pretrain_model_g")
    ckpt_ext = opt.get("pretrain_model_extractor")

    if ckpt_g:
        logger.info(f"Loading net_g from: {ckpt_g}")
        load_robust_state_dict(net_g, ckpt_g)
    else:
        logger.warning("pretrain_model_g 가 설정되지 않았습니다.")

    if ckpt_ext:
        logger.info(f"Loading net_extractor from: {ckpt_ext}")
        load_robust_state_dict(net_extractor, ckpt_ext)
    else:
        logger.warning("pretrain_model_extractor 가 설정되지 않았습니다.")

    return net_g, net_extractor, net_map


# ══════════════════════════════════════════════════════════════
# Phase 3-C: forward_tiling — 핵심 추론 함수
# ══════════════════════════════════════════════════════════════

def forward_tiling(
    lr: torch.Tensor,
    ref: torch.Tensor,
    net_g,
    net_extractor,
    net_map,
    tiling_opt: dict,
    device: torch.device,
) -> torch.Tensor:
    """
    Tiling + Overlap Blending 기반 DATSR 추론.

    tiling.mode 에 따라 두 가지 방식 중 하나를 선택한다:
      "margin" (기본): Max-Scale Square Margin Tiling
                       ref:lr 비율이 비정수여도 원본 ref 해상도를 보존.
      "resize"        : ref 를 lr×4 크기로 리사이즈한 뒤 1:4 비율로 타일링.
                       항상 tile_size×4 정방형 ref 타일 → margin/crop 불필요.

    Args:
        lr          : (C, H_lr, W_lr) float32 [0,1] LR 텐서 (CPU)
        ref         : (C, H_ref, W_ref) float32 [0,1] Ref 텐서 (CPU)
        net_g       : SwinUnetv3 SR 네트워크
        net_extractor: ContrasExtractorSep
        net_map     : FlowSimCorrespondenceGenerationArch
        tiling_opt  : YAML 의 'tiling' 섹션 dict
        device      : 추론 디바이스

    Returns:
        hr : (C, H_lr*4, W_lr*4) float32 [0,1] HR 텐서 (CPU)
    """
    mode       = (tiling_opt.get("mode") or "margin").lower()
    tile_size  = int(tiling_opt["lr_tile_size"])
    overlap    = int(tiling_opt["lr_overlap_pixels"])
    stride     = tile_size - overlap
    pad_mode   = tiling_opt.get("padding_mode") or "reflect"
    blend_meth = tiling_opt.get("blending_method") or "gaussian"
    sigma      = float(tiling_opt.get("gaussian_sigma") or 0.5)
    batch_size = int(tiling_opt.get("batch_size") or 1)

    if mode == "margin":
        margin = int(tiling_opt["ref_search_margin"])

    # 네트워크 dtype 자동 감지 (fp16 / fp32)
    net_dtype = next(net_g.parameters()).dtype

    # ── Phase 1: CPU 에서 전체 타일 사전 계산 ────────────────────
    lr_tiles, positions, original_shape = tile_lr(
        lr, tile_size, stride, pad_mode
    )

    # match_img_in: LR 를 4× bicubic 업샘플 (float32 고정 — CPU 연산 안정성)
    match_full = F.interpolate(
        lr.float().unsqueeze(0),
        scale_factor=SCALE,
        mode="bicubic",
        align_corners=False,
    ).squeeze(0)  # (C, H_lr*4, W_lr*4)

    _, Mh, Mw = match_full.shape

    # "resize" 모드: ref 를 lr×4 크기로 한 번만 리사이즈
    if mode == "resize":
        ref_resized = F.interpolate(
            ref.float().unsqueeze(0),
            size=(Mh, Mw),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)  # (C, H_lr*4, W_lr*4)
        logger.info(f"[resize mode] ref resized: {tuple(ref.shape[1:])} → ({Mh},{Mw})")

    all_lr: list    = []
    all_ref: list   = []
    all_match: list = []

    for lr_tile, (r, c) in zip(lr_tiles, positions):
        r_hr, c_hr = r * SCALE, c * SCALE
        ms         = tile_size * SCALE

        if mode == "resize":
            # ref 타일과 match 타일을 동일 위치·동일 크기로 크롭 (경계 패딩 공유)
            pb = max(0, r_hr + ms - Mh)
            pr = max(0, c_hr + ms - Mw)
            if pb > 0 or pr > 0:
                pad_kwargs = dict(mode=pad_mode)
                match_work = F.pad(match_full.unsqueeze(0),    (0, pr, 0, pb), **pad_kwargs).squeeze(0)
                ref_work   = F.pad(ref_resized.unsqueeze(0),   (0, pr, 0, pb), **pad_kwargs).squeeze(0)
            else:
                match_work, ref_work = match_full, ref_resized

            ref_tile   = ref_work  [:, r_hr:r_hr + ms, c_hr:c_hr + ms]
            match_tile = match_work[:, r_hr:r_hr + ms, c_hr:c_hr + ms]

        else:  # "margin"
            # Ref 타일 크롭 (Max-Scale Margin Tiling)
            ref_tile = get_ref_tile(
                ref, original_shape, r, c, tile_size, margin, pad_mode
            )  # (C, final_size, final_size)

            # match 타일 크롭 (HR 스케일)
            pb = max(0, r_hr + ms - Mh)
            pr = max(0, c_hr + ms - Mw)
            mwork = (
                F.pad(match_full.unsqueeze(0), (0, pr, 0, pb), mode=pad_mode).squeeze(0)
                if pb > 0 or pr > 0 else match_full
            )
            match_tile = mwork[:, r_hr:r_hr + ms, c_hr:c_hr + ms]

            # match 타일을 ref 타일 크기로 리사이즈
            _, rh, rw = ref_tile.shape
            if match_tile.shape[1] != rh or match_tile.shape[2] != rw:
                match_tile = F.interpolate(
                    match_tile.unsqueeze(0),
                    size=(rh, rw),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

        all_lr.append(lr_tile.float())
        all_ref.append(ref_tile.float())
        all_match.append(match_tile)   # already float32

    # ── Phase 2: GPU 배치 추론 ───────────────────────────────────
    total    = len(all_lr)
    sr_tiles = []

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)

            # 배치 스택 → GPU 이동 → 네트워크 dtype 캐스팅
            lr_b    = torch.stack(all_lr[start:end]).to(device=device, dtype=net_dtype)
            ref_b   = torch.stack(all_ref[start:end]).to(device=device, dtype=net_dtype)
            match_b = torch.stack(all_match[start:end]).to(device=device, dtype=net_dtype)

            features             = net_extractor(match_b, ref_b)
            pre_offset, ref_feat = net_map(features, ref_b)
            sr_batch             = net_g(lr_b, pre_offset, ref_feat)

            # float32 로 변환해 CPU 에 저장 (blending 정밀도 확보)
            sr_tiles.extend(t.float().cpu() for t in sr_batch)

            if end % max(1, total // 10) == 0 or end == total:
                logger.info(f"  Tile {end}/{total}")

    # ── Phase 3: Overlap Blending ────────────────────────────────
    hr = reconstruct_hr(
        sr_tiles, positions, original_shape, SCALE, blend_meth, sigma
    )
    return hr


# ══════════════════════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DATSR Standalone Inference (no GT required)"
    )
    parser.add_argument(
        "-opt", type=str, required=True,
        help="Path to inference YAML config"
    )
    args = parser.parse_args()

    opt = load_opt(args.opt)

    # 디바이스 설정
    gpu_ids = opt.get("gpu_ids") or []
    if gpu_ids and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # 출력 디렉토리 생성
    out_dir = Path(opt["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    # ── Phase 2: 이름 매칭 ─────────────────────────────────────
    pairs, _ = collect_matched_pairs(opt["dataroot_lr"], opt["dataroot_ref"])
    if not pairs:
        logger.error("매칭된 LR-Ref 쌍이 없습니다. 종료합니다.")
        return

    # ── Phase 3-B: 모델 로드 ──────────────────────────────────
    opt["scale"] = opt.get("scale", SCALE)
    net_g, net_extractor, net_map = build_models(opt, device)

    # FP16 (선택) — 네트워크만 half 로 변환; 입력 캐스팅은 forward_tiling 내부에서 처리
    if opt.get("fp16") and device.type == "cuda":
        net_g = net_g.half()
        net_extractor = net_extractor.half()
        net_map = net_map.half()
        logger.info("FP16 mode enabled")

    tiling_opt = opt["tiling"]

    # ── Phase 3-C: 이미지별 추론 ──────────────────────────────
    for idx, (lr_path, ref_path) in enumerate(pairs, 1):
        logger.info(f"[{idx}/{len(pairs)}] {lr_path.name}")

        lr  = load_image_tensor(lr_path)
        ref = load_image_tensor(ref_path)

        hr = forward_tiling(
            lr, ref, net_g, net_extractor, net_map, tiling_opt, device
        )

        save_path = out_dir / lr_path.name
        cv2.imwrite(str(save_path), tensor_to_image(hr))
        logger.info(f"  Saved -> {save_path}")

    logger.info(f"Done. {len(pairs)} image(s) saved to {out_dir}")


if __name__ == "__main__":
    main()
