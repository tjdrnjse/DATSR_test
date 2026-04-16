"""
Phase 2: Tiling & Padding Logic

LR 타일링:
  - Overlap 을 두고 정방형(square) 패치로 분할
  - 이미지가 tile_size 보다 작으면 패딩

Ref 타일링 (Max-Scale Square Margin Tiling):
  1. ref/lr 양 축의 배율을 구하고 더 큰 값(max_scale) 사용
  2. base_ref_tile_size = lr_tile_size * max_scale
  3. LR 타일 중심의 상대 좌표(u, v) → Ref 이미지 위의 중심 좌표
  4. 상하좌우 ref_search_margin 만큼 확장
  5. 범위 초과 시 Reflection Padding
"""

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────
# 내부 유틸
# ────────────────────────────────────────────────────────────────

def _tile_starts(total: int, tile_size: int, stride: int) -> List[int]:
    """
    한 축에서의 타일 시작 위치 목록을 반환한다.
    마지막 타일이 이미지 끝을 반드시 포함하도록 보정한다.
    """
    if total <= tile_size:
        return [0]
    positions = list(range(0, total - tile_size + 1, stride))
    if positions[-1] + tile_size < total:
        positions.append(total - tile_size)
    return positions


def _pad_to_min(
    tensor: torch.Tensor,
    min_size: int,
    padding_mode: str = "reflect",
) -> torch.Tensor:
    """텐서 (C, H, W) 의 H, W 가 min_size 미만이면 오른쪽/아래쪽에 패딩."""
    _, H, W = tensor.shape
    ph = max(0, min_size - H)
    pw = max(0, min_size - W)
    if ph > 0 or pw > 0:
        tensor = F.pad(
            tensor.unsqueeze(0),
            (0, pw, 0, ph),
            mode=padding_mode,
        ).squeeze(0)
    return tensor


# ────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────

def tile_lr(
    lr: torch.Tensor,
    tile_size: int,
    stride: int,
    padding_mode: str = "reflect",
) -> Tuple[List[torch.Tensor], List[Tuple[int, int]], Tuple[int, int]]:
    """
    LR 이미지를 Overlap 정방형 패치로 분할한다.

    Args:
        lr            : (C, H, W) float tensor
        tile_size     : 정방형 타일 한 변의 크기 (픽셀)
        stride        : 타일 간 간격 = tile_size - overlap
        padding_mode  : 경계 패딩 방식

    Returns:
        tiles          : List of (C, tile_size, tile_size) tensors
        positions      : 각 타일의 (row_start, col_start) — 패딩된 LR 좌표계
        original_shape : (H, W) — 패딩 전 원본 크기
    """
    C, H, W = lr.shape
    original_shape = (H, W)

    # 최소 tile_size 확보
    lr_padded = _pad_to_min(lr, tile_size, padding_mode)
    _, Hp, Wp = lr_padded.shape

    rows = _tile_starts(Hp, tile_size, stride)
    cols = _tile_starts(Wp, tile_size, stride)

    tiles: List[torch.Tensor] = []
    positions: List[Tuple[int, int]] = []

    for r in rows:
        for c in cols:
            tiles.append(lr_padded[:, r : r + tile_size, c : c + tile_size])
            positions.append((r, c))

    return tiles, positions, original_shape


def get_ref_tile(
    ref: torch.Tensor,
    lr_original_shape: Tuple[int, int],
    lr_tile_top: int,
    lr_tile_left: int,
    lr_tile_size: int,
    ref_search_margin: int,
    padding_mode: str = "reflect",
) -> torch.Tensor:
    """
    Max-Scale Square Margin Tiling 알고리즘으로 Ref 타일을 크롭한다.

    알고리즘:
        1. scale_x = W_ref / W_lr,  scale_y = H_ref / H_lr
        2. max_scale = max(scale_x, scale_y)
        3. base_ref_tile_size = ceil(lr_tile_size * max_scale)   (정방형 유지)
        4. final_size = base_ref_tile_size + 2 * ref_search_margin
        5. LR 타일 중심 → 상대 좌표 (u, v) → Ref 이미지 위 중심 (ref_cx, ref_cy)
        6. final_size × final_size 크롭
        7. 범위 초과 시 Reflection Padding

    Args:
        ref                : (C, H_ref, W_ref) float tensor
        lr_original_shape  : (H_lr, W_lr) — 패딩 전 원본 LR 크기
        lr_tile_top        : LR 타일의 row 시작 (패딩된 LR 좌표)
        lr_tile_left       : LR 타일의 col 시작 (패딩된 LR 좌표)
        lr_tile_size       : LR 타일 한 변 크기
        ref_search_margin  : base Ref 타일 크기에 상하좌우 추가할 마진(픽셀)
        padding_mode       : 범위 초과 시 패딩 방식

    Returns:
        ref_tile : (C, final_size, final_size) tensor
    """
    C, H_ref, W_ref = ref.shape
    H_lr, W_lr = lr_original_shape

    # ── 1. Max Scale ─────────────────────────────────────────────
    max_scale = max(W_ref / W_lr, H_ref / H_lr)

    # ── 2. Base / Final 크기 ─────────────────────────────────────
    base_size  = int(math.ceil(lr_tile_size * max_scale))
    final_size = base_size + 2 * ref_search_margin

    # ── 3. LR 타일 중심 → 상대 좌표 → Ref 중심 ──────────────────
    lr_cy = lr_tile_top  + lr_tile_size / 2.0
    lr_cx = lr_tile_left + lr_tile_size / 2.0
    u = lr_cx / W_lr        # [0, 1]
    v = lr_cy / H_lr        # [0, 1]

    ref_cx = u * W_ref
    ref_cy = v * H_ref

    # ── 4. 크롭 경계 ─────────────────────────────────────────────
    half   = final_size / 2.0
    left   = int(math.floor(ref_cx - half))
    top    = int(math.floor(ref_cy - half))
    right  = left + final_size
    bottom = top  + final_size

    # ── 5. Reflection Padding (범위 초과 시) ─────────────────────
    pl = max(0, -left)
    pt = max(0, -top)
    pr = max(0, right  - W_ref)
    pb = max(0, bottom - H_ref)

    if pl or pt or pr or pb:
        ref_work = F.pad(
            ref.unsqueeze(0),
            (pl, pr, pt, pb),
            mode=padding_mode,
        ).squeeze(0)
        left  += pl;  right  += pl
        top   += pt;  bottom += pt
    else:
        ref_work = ref

    return ref_work[:, top:bottom, left:right]
