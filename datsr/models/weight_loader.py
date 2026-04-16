"""
Phase 2: Robust Weight Loader
────────────────────────────────────────────────────────────────
load_robust_state_dict(model, checkpoint_path)

기능:
  1. torch.load(..., weights_only=False) 로 체크포인트 로드
  2. 체크포인트 내 'params' / 'state_dict' / 'model' 계층 자동 탐색
  3. DCN 관련 구형 키 자동 변환
       - .conv_offset.{weight|bias}  →  .conv_offset_mask.{weight|bias}
         (구형 C++ DCNv2 는 offset 만 있고 mask 없음)
       - .{any}.conv.{weight|bias}   →  .{any}.{weight|bias}
         (weight/bias 가 nn.Conv2d 'conv' 안에 래핑된 경우)
  4. strict=False 로 주입 후 missing / unexpected 키를 로그에 출력
  5. 결과 딕셔너리 반환
"""

import logging
import re
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("base")


# ──────────────────────────────────────────────────────────────
# 키 변환 규칙 (순서대로 적용)
# ──────────────────────────────────────────────────────────────

# 각 항목: (컴파일된 패턴, 치환 문자열, 설명)
_KEY_RULES = [
    (
        re.compile(r"\.conv_offset\.(weight|bias)$"),
        r".conv_offset_mask.\1",
        "conv_offset -> conv_offset_mask  (구형 DCN: mask 없음)",
    ),
    # NOTE: .conv.(weight|bias) → .(weight|bias) 규칙은 제거됨.
    # 체크포인트의 DCN 레이어는 이미 .weight/.bias 직접 저장이며,
    # 해당 규칙이 합법적인 nn.Conv2d 자식 레이어(예: layers.N.conv.weight)를
    # 잘못 변환하여 missing/unexpected 경고를 유발함.
]


def _remap_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    구형 키 이름을 현재 모델 구조에 맞게 변환한다.
    변환이 발생한 경우에만 로그에 기록한다.
    """
    new_sd: Dict[str, torch.Tensor] = {}
    renamed: list = []

    for key, value in state_dict.items():
        new_key = key
        for pattern, repl, desc in _KEY_RULES:
            candidate = pattern.sub(repl, new_key)
            if candidate != new_key:
                renamed.append((key, candidate, desc))
                new_key = candidate
                # 하나의 규칙이 적용되면 다음 규칙도 계속 적용 가능
        new_sd[new_key] = value

    if renamed:
        logger.info(f"[RobustLoader] {len(renamed)} key(s) renamed:")
        for old, new, reason in renamed:
            logger.info(f"  {old!r:60s}  ->  {new!r}  ({reason})")

    return new_sd


def _extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    """
    다양한 형식의 체크포인트에서 state_dict 를 추출한다.
    지원 형식:
      - 순수 state_dict (dict[str, Tensor])
      - {'params': state_dict}
      - {'state_dict': state_dict}
      - {'model': state_dict}
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(f"체크포인트가 dict 타입이 아닙니다: {type(checkpoint)}")

    # Tensor 값을 직접 담은 dict 인지 확인
    first_val = next(iter(checkpoint.values()), None)
    if isinstance(first_val, torch.Tensor):
        return checkpoint

    for key in ("params", "state_dict", "model", "params_ema"):
        if key in checkpoint:
            logger.info(f"[RobustLoader] '{key}' 계층에서 state_dict 추출")
            inner = checkpoint[key]
            if isinstance(inner, dict):
                return inner

    raise KeyError(
        f"체크포인트에서 state_dict 를 찾지 못했습니다. "
        f"최상위 키: {list(checkpoint.keys())}"
    )


def load_robust_state_dict(
    model: nn.Module,
    checkpoint_path: str,
    key_remapping: bool = True,
    strip_module_prefix: bool = True,
) -> Dict:
    """
    체크포인트를 로드하고 DCN 키 이름을 자동 변환한 뒤 모델에 주입한다.

    Args:
        model            : 가중치를 받을 nn.Module
        checkpoint_path  : .pth / .pt 체크포인트 경로
        key_remapping    : DCN 관련 키 자동 변환 여부 (기본 True)
        strip_module_prefix : 'module.' 접두어 제거 여부 (기본 True)

    Returns:
        report dict:
            'missing_keys'   : 모델에는 있지만 체크포인트에 없는 키
            'unexpected_keys': 체크포인트에는 있지만 모델에 없는 키
            'renamed_count'  : 이름 변환된 키 수
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {path.resolve()}")

    logger.info(f"[RobustLoader] Loading checkpoint: {path}")
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)

    # ── 1. state_dict 추출 ─────────────────────────────────────
    state_dict = _extract_state_dict(checkpoint)

    # ── 2. 'module.' 접두어 제거 (DataParallel 저장 형식) ─────────
    if strip_module_prefix:
        stripped = {}
        for k, v in state_dict.items():
            stripped[k[7:] if k.startswith("module.") else k] = v
        state_dict = stripped

    # ── 3. DCN 키 자동 변환 ────────────────────────────────────
    renamed_count = 0
    if key_remapping:
        original_keys = set(state_dict.keys())
        state_dict = _remap_keys(state_dict)
        renamed_count = sum(
            1 for k in state_dict if k not in original_keys
        )

    # ── 4. strict=False 로드 ───────────────────────────────────
    # DataParallel 래핑 해제
    net = model.module if hasattr(model, "module") else model

    result = net.load_state_dict(state_dict, strict=False)
    missing    = result.missing_keys
    unexpected = result.unexpected_keys

    # ── 5. 결과 리포트 ─────────────────────────────────────────
    logger.info(
        f"[RobustLoader] Load complete: "
        f"{len(missing)} missing, {len(unexpected)} unexpected, "
        f"{renamed_count} renamed"
    )

    if missing:
        logger.warning("[RobustLoader] Missing keys (model has, checkpoint lacks):")
        for k in missing:
            logger.warning(f"  MISSING    {k}")

    if unexpected:
        logger.warning("[RobustLoader] Unexpected keys (checkpoint has, model lacks):")
        for k in unexpected:
            logger.warning(f"  UNEXPECTED {k}")

    return {
        "missing_keys":   missing,
        "unexpected_keys": unexpected,
        "renamed_count":  renamed_count,
    }
