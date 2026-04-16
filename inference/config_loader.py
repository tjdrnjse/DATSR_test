"""
Phase 1: Configuration Setup
config.yml 을 읽고 InferenceConfig 객체를 반환한다.
lr_stride 는 lr_tile_size - lr_overlap_pixels 로 자동 계산된다.
"""
import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class InferenceConfig:
    # LR 타일링
    lr_tile_size: int
    lr_overlap_pixels: int
    lr_stride: int          # computed: lr_tile_size - lr_overlap_pixels

    # Ref 타일링
    ref_search_margin: int
    padding_mode: str       # 'reflect' | 'replicate' | 'constant'

    # 블렌딩
    blending_method: str    # 'gaussian' | 'linear'
    gaussian_sigma: float


def load_config(config_path: str) -> InferenceConfig:
    """YAML 파일을 로드하고 InferenceConfig 를 반환한다."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if "inference" not in raw:
        raise KeyError("Config YAML 최상위에 'inference' 섹션이 필요합니다.")

    cfg = raw["inference"]

    lr_tile_size      = int(cfg["lr_tile_size"])
    lr_overlap_pixels = int(cfg["lr_overlap_pixels"])

    if lr_overlap_pixels >= lr_tile_size:
        raise ValueError(
            f"lr_overlap_pixels({lr_overlap_pixels}) 는 "
            f"lr_tile_size({lr_tile_size}) 보다 작아야 합니다."
        )

    return InferenceConfig(
        lr_tile_size      = lr_tile_size,
        lr_overlap_pixels = lr_overlap_pixels,
        lr_stride         = lr_tile_size - lr_overlap_pixels,
        ref_search_margin = int(cfg["ref_search_margin"]),
        padding_mode      = str(cfg.get("padding_mode", "reflect")),
        blending_method   = str(cfg.get("blending_method", "gaussian")),
        gaussian_sigma    = float(cfg.get("gaussian_sigma", 0.5)),
    )
