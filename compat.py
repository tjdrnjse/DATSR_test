"""
compat.py — Unified Monkey Patch for MMCV 2.x + torchvision 0.16+
──────────────────────────────────────────────────────────────────
메인 스크립트(test.py / train.py / inference.py) 최상단에서 import 하세요.
다른 모든 datsr / mmcv import 보다 먼저 실행되어야 합니다.

Fixes:
  1. mmcv.scandir         — MMCV 2.x 에서 top-level 네임스페이스에서 제거됨
  2. mmcv.runner.*        — MMCV 2.x 에서 mmengine 으로 이동됨
                            대상: get_time_str, init_dist, master_only, get_dist_info
  3. rgb_to_grayscale     — torchvision 0.16+ 에서 functional_tensor 모듈 제거됨
"""

import datetime
import os
import sys
import types

import mmcv


# ══════════════════════════════════════════════════════════════════════════════
# Fix 1: mmcv.scandir
#   datsr/{data,models,models/archs}/__init__.py 가 모듈 로드 즉시 호출함.
# ══════════════════════════════════════════════════════════════════════════════
if not hasattr(mmcv, 'scandir'):
    def _mmcv_scandir(dir_path, suffix=None, recursive=False,
                      case_sensitive=True):
        """MMCV 1.x mmcv.scandir 드롭인 대체 구현."""
        if isinstance(suffix, str):
            suffix = (suffix,)
        for _root, _dirs, _files in os.walk(str(dir_path)):
            _dirs[:] = sorted(d for d in _dirs if not d.startswith('.'))
            for _fname in sorted(_files):
                if _fname.startswith('.'):
                    continue
                _rel = os.path.relpath(
                    os.path.join(_root, _fname), str(dir_path))
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
    mmcv.scandir = _mmcv_scandir


# ══════════════════════════════════════════════════════════════════════════════
# Fix 2: mmcv.runner
#   MMCV 2.x 에서 runner 서브모듈 전체가 mmengine 으로 이동됨.
#   sys.modules 에 스텁을 선제 주입하여 `from mmcv.runner import X` 가
#   동작하게 한다. mmengine 이 설치된 환경이면 실제 구현을 사용하고,
#   없으면 최소 폴백 구현을 제공한다.
# ══════════════════════════════════════════════════════════════════════════════
_RUNNER = 'mmcv.runner'

if _RUNNER not in sys.modules:
    _runner = types.ModuleType(_RUNNER)
    sys.modules[_RUNNER] = _runner
else:
    _runner = sys.modules[_RUNNER]

# mmcv 패키지 객체에도 속성으로 등록 (mmcv.runner.X 접근 방식 대응)
if not hasattr(mmcv, 'runner'):
    mmcv.runner = _runner


def _patch_runner(attr, factory):
    """_runner 에 attr 이 없으면 factory() 결과를 주입."""
    if not hasattr(_runner, attr):
        setattr(_runner, attr, factory())


def _make_get_time_str():
    try:
        from mmengine.utils import get_time_str
        return get_time_str
    except ImportError:
        def get_time_str():
            return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        return get_time_str


def _make_init_dist():
    try:
        from mmengine.dist import init_dist
        return init_dist
    except ImportError:
        def init_dist(launcher, **kwargs):
            pass
        return init_dist


def _make_master_only():
    try:
        from mmengine.dist import master_only
        return master_only
    except ImportError:
        def master_only(func):
            """단일 프로세스 환경에서는 그냥 원본 함수를 반환."""
            return func
        return master_only


def _make_get_dist_info():
    try:
        from mmengine.dist import get_dist_info
        return get_dist_info
    except ImportError:
        def get_dist_info():
            return 0, 1
        return get_dist_info


_patch_runner('get_time_str',  _make_get_time_str)
_patch_runner('init_dist',     _make_init_dist)
_patch_runner('master_only',   _make_master_only)
_patch_runner('get_dist_info', _make_get_dist_info)


# ══════════════════════════════════════════════════════════════════════════════
# Fix 3: torchvision.transforms.functional_tensor.rgb_to_grayscale
#   torchvision 0.16+ 에서 functional_tensor 서브모듈 자체가 제거됨.
#   스텁 모듈을 sys.modules 에 선제 주입.
# ══════════════════════════════════════════════════════════════════════════════
_FT = 'torchvision.transforms.functional_tensor'
try:
    import torchvision.transforms.functional_tensor as _ft
    if not hasattr(_ft, 'rgb_to_grayscale'):
        import torchvision.transforms.functional as _tvf
        _ft.rgb_to_grayscale = _tvf.rgb_to_grayscale
except ImportError:
    import torchvision.transforms.functional as _tvf
    _stub = types.ModuleType(_FT)
    _stub.rgb_to_grayscale = _tvf.rgb_to_grayscale
    sys.modules[_FT] = _stub
