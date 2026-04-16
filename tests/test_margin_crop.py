"""
test_margin_crop.py
───────────────────────────────────────────────────────────────────
DynamicAggregationRestoration._center_crop 및 use_margin_crop 플래그 검증.

테스트 설계:
  LR tile  : 40×40  →  base (bicubic ×4) = 160×160
  margin   : 32 px  (HR 스케일 기준)
  Ref tile : 160 + 2*32 = 224×224

VGG 피처 크기 (stride 고려):
  relu1_1 (stride=1): ref=224, content(x0)=160  → crop 64px (32 per side)
  relu2_1 (stride=2): ref=112, content(x1)=80   → crop 32px (16 per side)
  relu3_1 (stride=4): ref=56,  content(x2)=40   → crop 16px  (8 per side)

실행:
    python tests/test_margin_crop.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import compat  # noqa: F401

import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────
# 테스트 파라미터
# ──────────────────────────────────────────────────────────────────
B      = 1          # batch size
NGF    = 64         # ngf (forward 에서 64 채널이 x0 channel)
MARGIN = 32         # HR 스케일 마진 (px)
SCALE  = 4

# Content feature 크기 (LR tile 40×40 → base 160×160)
H_LR   = 40
H_BASE = H_LR * SCALE          # 160
H_MED  = H_BASE // 2           # 80
H_SML  = H_BASE // 4           # 40

# Ref feature 크기 (margin 포함)
H_REF1 = H_BASE + 2 * MARGIN           # 224  (stride 1)
H_REF2 = H_BASE // 2 + 2 * (MARGIN//2) # 96  — wait: (H_BASE + 2*MARGIN)//2
H_REF3 = H_BASE // 4 + 2 * (MARGIN//4) # 48

# 정확하게: VGG stride 2,4 후 공간 크기
H_REF2 = (H_BASE + 2 * MARGIN) // 2    # 112
H_REF3 = (H_BASE + 2 * MARGIN) // 4    # 56

# VGG feature 채널 (하드코딩)
C_RELU1, C_RELU2, C_RELU3 = 64, 128, 256
# deformable_groups=8, kernel=3×3 → channels_ = 8*3*3 = 72
DGROUPS = 8
N_PATCH = 9   # pre_offset 의 patch 수


def make_feat(B, C, H):
    """[B, C, H, H] 랜덤 float32 텐서."""
    return torch.randn(B, C, H, H)


def make_offset_sim(B, H):
    """
    pre_offset : [B, 9, H, H, 2]
    pre_sim    : [B, 72, H, H]
    """
    pre_offset = torch.randn(B, N_PATCH, H, H, 2)
    pre_sim    = torch.randn(B, DGROUPS * 3 * 3, H, H)
    return pre_offset, pre_sim


# ──────────────────────────────────────────────────────────────────
# Test 1: _center_crop 정확도 검증
# ──────────────────────────────────────────────────────────────────
def test_center_crop():
    """_center_crop 이 올바른 중앙 영역을 반환하는지 확인."""
    from datsr.models.archs.swin_unetv3_ref_restoration_arch import DynamicAggregationRestoration

    cases = [
        # (src_h, tgt_h, expected_dh)
        (H_REF1, H_BASE, MARGIN),       # 224 → 160, dh=32
        (H_REF2, H_MED,  MARGIN // 2),  # 112 → 80,  dh=16
        (H_REF3, H_SML,  MARGIN // 4),  # 56  → 40,  dh=8
    ]

    for src_h, tgt_h, exp_dh in cases:
        src = torch.arange(src_h * src_h, dtype=torch.float32).reshape(1, 1, src_h, src_h)
        tgt = torch.zeros(1, 1, tgt_h, tgt_h)
        out = DynamicAggregationRestoration._center_crop(src, tgt)

        assert out.shape == (1, 1, tgt_h, tgt_h), \
            f"shape mismatch: expected (1,1,{tgt_h},{tgt_h}), got {out.shape}"

        # 실제로 중앙 영역인지 확인
        expected = src[:, :, exp_dh:exp_dh + tgt_h, exp_dh:exp_dh + tgt_h]
        assert torch.equal(out, expected), \
            f"center-crop content mismatch at src={src_h}, tgt={tgt_h}"

        print(f"  [PASS] _center_crop: {src_h}×{src_h} → {tgt_h}×{tgt_h}  (dh={exp_dh})")

    # 크기가 이미 같으면 동일 텐서 반환 (메모리 절약)
    same = torch.randn(1, 64, 160, 160)
    out  = DynamicAggregationRestoration._center_crop(same, same)
    assert out is same, "_center_crop: same-size 입력 시 동일 객체를 반환해야 함"
    print(f"  [PASS] _center_crop: same-size → identity")


# ──────────────────────────────────────────────────────────────────
# Test 2: use_margin_crop=False — 크기 일치 시 정상 작동
# ──────────────────────────────────────────────────────────────────
def test_no_margin_no_mismatch():
    """마진 없을 때(ref == content 크기) use_margin_crop=False 가 정상 작동해야 함."""
    from datsr.models.archs.swin_unetv3_ref_restoration_arch import DynamicAggregationRestoration

    model = _build_tiny_model(use_margin_crop=False)
    base, img_ref_feat, pre_offset_flow_sim = _build_inputs(margin=0)

    out = model(base, None, pre_offset_flow_sim, img_ref_feat)
    assert out.shape == (B, NGF, H_BASE, H_BASE), f"unexpected output shape: {out.shape}"
    print(f"  [PASS] use_margin_crop=False, margin=0: output {out.shape}")


# ──────────────────────────────────────────────────────────────────
# Test 3: use_margin_crop=False — 크기 불일치 시 RuntimeError 발생 확인
# ──────────────────────────────────────────────────────────────────
def test_no_margin_crop_raises_on_mismatch():
    """마진이 있을 때 use_margin_crop=False 면 torch.cat 에서 오류가 나야 함."""
    from datsr.models.archs.swin_unetv3_ref_restoration_arch import DynamicAggregationRestoration

    model = _build_tiny_model(use_margin_crop=False)
    base, img_ref_feat, pre_offset_flow_sim = _build_inputs(margin=MARGIN)

    try:
        _ = model(base, None, pre_offset_flow_sim, img_ref_feat)
        raise AssertionError("RuntimeError 가 발생하지 않았습니다 — 테스트 실패!")
    except RuntimeError:
        print(f"  [PASS] use_margin_crop=False, margin={MARGIN}: 예상대로 RuntimeError 발생")


# ──────────────────────────────────────────────────────────────────
# Test 4: use_margin_crop=True — 마진 있어도 정상 작동
# ──────────────────────────────────────────────────────────────────
def test_margin_crop_fixes_mismatch():
    """use_margin_crop=True 면 margin 포함 ref 피처도 크기 맞춰 처리돼야 함."""
    from datsr.models.archs.swin_unetv3_ref_restoration_arch import DynamicAggregationRestoration

    model = _build_tiny_model(use_margin_crop=True)
    base, img_ref_feat, pre_offset_flow_sim = _build_inputs(margin=MARGIN)

    out = model(base, None, pre_offset_flow_sim, img_ref_feat)
    assert out.shape == (B, NGF, H_BASE, H_BASE), f"unexpected output shape: {out.shape}"
    print(f"  [PASS] use_margin_crop=True,  margin={MARGIN}: output {out.shape}")


# ──────────────────────────────────────────────────────────────────
# 헬퍼: 경량 DynamicAggregationRestoration (SwinBlock 없는 패치 버전)
# ──────────────────────────────────────────────────────────────────
def _build_tiny_model(use_margin_crop: bool):
    """
    SwinBlock + DynAgg 를 Identity 로 교체한 테스트용 경량 모델.
    forward 에서의 torch.cat 크기 정합성만 검증한다.
    """
    from datsr.models.archs.swin_unetv3_ref_restoration_arch import DynamicAggregationRestoration

    model = DynamicAggregationRestoration(
        ngf=NGF,
        groups=DGROUPS,
        depths=(2, 2),
        num_heads=(2, 2),
        window_size=8,
        use_margin_crop=use_margin_crop,
    )

    # ── SwinBlock 을 Identity 로 교체 (비용 절감) ──
    model.down_body_large  = nn.Identity()
    model.down_body_medium = nn.Identity()
    model.up_body_small    = nn.Identity()
    model.up_body_medium   = nn.Identity()
    model.up_body_large    = nn.Identity()

    # ── DynAgg 를 입력 텐서 그대로 반환하는 람다로 교체 ──
    # DynAgg.forward(x_list, pre_offset, pre_sim) → x_list[0] 을 그대로 반환
    class _FakeDynAgg(nn.Module):
        def __init__(self, out_c): super().__init__(); self.out_c = out_c
        def forward(self, x, pre_offset, pre_sim):
            inp = x[0] if isinstance(x, (list, tuple)) else x
            # 채널 수가 맞지 않을 수 있으므로 강제 조정
            B_, _, H_, W_ = inp.shape
            return torch.zeros(B_, self.out_c, H_, W_)

    model.down_large_dyn_agg  = _FakeDynAgg(64)
    model.down_medium_dyn_agg = _FakeDynAgg(128)
    model.up_small_dyn_agg    = _FakeDynAgg(256)
    model.up_medium_dyn_agg   = _FakeDynAgg(128)
    model.up_large_dyn_agg    = _FakeDynAgg(64)

    model.eval()
    return model


def _build_inputs(margin: int):
    """
    margin 크기에 따라 ref 피처와 offset/sim 생성.

    Returns:
        base                : [B, 3, H_BASE, H_BASE]
        img_ref_feat        : dict
        pre_offset_flow_sim : (pre_offset, pre_flow, pre_similarity) 각각 dict
    """
    # base = 4× 업스케일된 LR
    base = torch.randn(B, 3, H_BASE, H_BASE)

    # Ref 피처 크기 (margin 반영)
    r1 = H_BASE + 2 * margin            # relu1_1
    r2 = (H_BASE + 2 * margin) // 2     # relu2_1
    r3 = (H_BASE + 2 * margin) // 4     # relu3_1

    img_ref_feat = {
        'relu1_1': make_feat(B, C_RELU1, r1),
        'relu2_1': make_feat(B, C_RELU2, r2),
        'relu3_1': make_feat(B, C_RELU3, r3),
    }

    # pre_offset / pre_flow / pre_similarity
    # flow_warp 은 (feature, flow) 크기가 같아야 함 → flow 도 ref 크기로 생성
    pre_offset, pre_sim = {}, {}
    pre_flow = {}
    for key, h in [('relu1_1', r1), ('relu2_1', r2), ('relu3_1', r3)]:
        po, ps = make_offset_sim(B, h)
        pre_offset[key] = po
        pre_sim[key]    = ps
        pre_flow[key]   = torch.zeros(B, h, h, 2)  # zero flow → no warp

    return base, img_ref_feat, (pre_offset, pre_flow, pre_sim)


# ──────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("Test 1: _center_crop 정확도")
    print("=" * 60)
    test_center_crop()

    print()
    print("=" * 60)
    print("Test 2: use_margin_crop=False, margin=0 (크기 일치)")
    print("=" * 60)
    test_no_margin_no_mismatch()

    print()
    print("=" * 60)
    print("Test 3: use_margin_crop=False, margin=32 (크기 불일치 → 오류 예상)")
    print("=" * 60)
    test_no_margin_crop_raises_on_mismatch()

    print()
    print("=" * 60)
    print("Test 4: use_margin_crop=True, margin=32 (크기 불일치 → 크롭으로 해결)")
    print("=" * 60)
    test_margin_crop_fixes_mismatch()

    print()
    print("All tests passed.")
