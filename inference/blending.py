"""
Phase 3: Blending & Reconstruction

SR 타일들을 원본 LR 해상도의 4배 크기(HR)로 합친다.
단순 덮어쓰기/평균이 아닌 가중치 누적 블렌딩을 사용한다.

  1. 타일 중심에서 거리가 멀수록 가중치가 낮아지는 Weight Window 생성
     - Gaussian  : 2D Gaussian, sigma = gaussian_sigma * (tile_size / 2)
     - Linear    : 2D Tent (triangular) window
  2. canvas      += tile  * weight_window
     weight_acc  += weight_window
  3. hr           = canvas / weight_acc
"""

from typing import List, Tuple

import torch


# ────────────────────────────────────────────────────────────────
# Weight-window factories
# ────────────────────────────────────────────────────────────────

def make_gaussian_window(size: int, sigma: float = 0.5) -> torch.Tensor:
    """
    (size, size) 2D Gaussian weight window.

    sigma 는 half-width 에 대한 비율.  예: sigma=0.5 → std = size/4.
    피크값은 1 로 정규화된다.
    """
    if size == 1:
        return torch.ones(1, 1)

    center = (size - 1) / 2.0
    std    = sigma * center          # effective std (pixels)
    if std == 0:
        std = 1.0

    coords = torch.arange(size, dtype=torch.float32) - center
    g1d    = torch.exp(-0.5 * (coords / std) ** 2)
    window = torch.outer(g1d, g1d)
    window = window / window.max()   # peak → 1
    return window


def make_linear_window(size: int) -> torch.Tensor:
    """
    (size, size) 2D Linear (tent / triangular) weight window.
    중심에서 1, 가장자리에서 0.
    """
    if size == 1:
        return torch.ones(1, 1)

    half   = (size - 1) / 2.0
    coords = torch.arange(size, dtype=torch.float32)
    w1d    = (1.0 - torch.abs(coords - half) / (half if half > 0 else 1.0)).clamp(min=0.0)
    return torch.outer(w1d, w1d)


def make_weight_window(
    tile_size: int,
    method: str = "gaussian",
    sigma: float = 0.5,
) -> torch.Tensor:
    """(tile_size, tile_size) 가중치 윈도우를 반환한다."""
    if method == "gaussian":
        return make_gaussian_window(tile_size, sigma)
    elif method == "linear":
        return make_linear_window(tile_size)
    else:
        raise ValueError(
            f"알 수 없는 blending_method: '{method}'. "
            f"'gaussian' 또는 'linear' 를 사용하세요."
        )


# ────────────────────────────────────────────────────────────────
# Reconstruction
# ────────────────────────────────────────────────────────────────

def reconstruct_hr(
    sr_tiles: List[torch.Tensor],
    lr_positions: List[Tuple[int, int]],
    lr_original_shape: Tuple[int, int],
    scale: int,
    blending_method: str = "gaussian",
    gaussian_sigma: float = 0.5,
) -> torch.Tensor:
    """
    SR 타일들을 Overlap Blending 으로 합쳐 최종 HR 이미지를 생성한다.

    Args:
        sr_tiles           : List of (C, T*scale, T*scale) SR 타일 텐서
        lr_positions       : tile_lr() 가 반환한 (row_start, col_start) 목록
                             — 패딩된 LR 좌표계
        lr_original_shape  : (H_lr, W_lr) — 패딩 전 원본 LR 크기
        scale              : 업스케일 배율 (DATSR = 4)
        blending_method    : 'gaussian' | 'linear'
        gaussian_sigma     : Gaussian window 의 sigma 비율

    Returns:
        hr : (C, H_lr * scale, W_lr * scale) 블렌딩된 HR 텐서
    """
    assert len(sr_tiles) == len(lr_positions), (
        f"sr_tiles({len(sr_tiles)}) 와 lr_positions({len(lr_positions)}) 개수가 다릅니다."
    )

    C, sr_h, sr_w = sr_tiles[0].shape
    assert sr_h == sr_w, f"SR 타일이 정방형이어야 합니다. 실제 shape: {sr_tiles[0].shape}"
    sr_tile_size = sr_h

    H_lr, W_lr = lr_original_shape
    H_hr = H_lr * scale
    W_hr = W_lr * scale

    device = sr_tiles[0].device
    dtype  = sr_tiles[0].dtype

    canvas     = torch.zeros(C, H_hr, W_hr, device=device, dtype=dtype)
    weight_acc = torch.zeros(1, H_hr, W_hr, device=device, dtype=dtype)

    # 모든 타일에 동일한 weight window 사용
    window = make_weight_window(sr_tile_size, blending_method, gaussian_sigma)
    window = window.to(device=device, dtype=dtype)   # (sr_tile_size, sr_tile_size)

    for tile, (r_lr, c_lr) in zip(sr_tiles, lr_positions):
        # LR 좌표 → HR 좌표
        r_hr = r_lr * scale
        c_hr = c_lr * scale

        # HR 캔버스 경계를 벗어나면 스킵
        if r_hr >= H_hr or c_hr >= W_hr:
            continue

        # 캔버스 끝을 넘어가는 경우 크롭
        r_end = min(r_hr + sr_tile_size, H_hr)
        c_end = min(c_hr + sr_tile_size, W_hr)

        th = r_end - r_hr   # 실제 누적 높이
        tw = c_end - c_hr   # 실제 누적 너비

        tile_crop   = tile  [:, :th, :tw]           # (C, th, tw)
        window_crop = window[    :th, :tw]           # (th, tw)

        canvas    [:, r_hr:r_end, c_hr:c_end] += tile_crop * window_crop.unsqueeze(0)
        weight_acc[:, r_hr:r_end, c_hr:c_end] += window_crop.unsqueeze(0)

    # 0 나눗셈 방지
    weight_acc = weight_acc.clamp(min=1e-8)
    hr = canvas / weight_acc

    return hr
