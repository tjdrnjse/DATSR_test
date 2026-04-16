"""
Full Inference Pipeline
Phase 1~3 를 통합한 end-to-end 타일드 추론 함수.

사용법:
    from inference.pipeline import run_inference
    hr = run_inference(lr, ref, config, model_fn)
"""

from typing import Callable

import torch

from .config_loader import InferenceConfig
from .tiling import tile_lr, get_ref_tile
from .blending import reconstruct_hr

SCALE_FACTOR = 4   # DATSR: 4× super-resolution


def run_inference(
    lr: torch.Tensor,
    ref: torch.Tensor,
    config: InferenceConfig,
    model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    DATSR 타일드 추론 파이프라인.

    Args:
        lr       : (C, H_lr, W_lr) LR 입력 이미지 텐서
        ref      : (C, H_ref, W_ref) 참조 이미지 텐서
        config   : load_config() 로 읽어온 InferenceConfig
        model_fn : (lr_tile, ref_tile) → sr_tile 형태의 callable.
                   lr_tile  shape : (C, T, T)
                   sr_tile  shape : (C, T*4, T*4)

    Returns:
        hr : (C, H_lr*4, W_lr*4) 최종 SR 이미지 텐서
    """
    # ── Phase 2-A: LR 타일링 ──────────────────────────────────────────
    lr_tiles, positions, original_shape = tile_lr(
        lr,
        tile_size    = config.lr_tile_size,
        stride       = config.lr_stride,
        padding_mode = config.padding_mode,
    )

    # ── Phase 2-B / Model: 각 타일에 대해 Ref 크롭 + SR 추론 ─────────
    sr_tiles = []
    for lr_tile, (row, col) in zip(lr_tiles, positions):

        ref_tile = get_ref_tile(
            ref               = ref,
            lr_original_shape = original_shape,
            lr_tile_top       = row,
            lr_tile_left      = col,
            lr_tile_size      = config.lr_tile_size,
            ref_search_margin = config.ref_search_margin,
            padding_mode      = config.padding_mode,
        )

        sr_tile = model_fn(lr_tile, ref_tile)
        sr_tiles.append(sr_tile)

    # ── Phase 3: Overlap Blending 으로 HR 재구성 ─────────────────────
    hr = reconstruct_hr(
        sr_tiles          = sr_tiles,
        lr_positions      = positions,
        lr_original_shape = original_shape,
        scale             = SCALE_FACTOR,
        blending_method   = config.blending_method,
        gaussian_sigma    = config.gaussian_sigma,
    )

    return hr
