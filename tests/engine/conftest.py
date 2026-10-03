import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import torch
torch.set_num_threads(2)   # 이 컨테이너에서 4 스레드는 원소별 연산이 1000 배 느리다

# ---------------------------------------------------------------------------------------------
# **모듈 전역 오염 방지** (MEASUREMENTS §52.180).
#
# 엔진은 설정을 모듈 전역(대문자 상수)으로 받는다. `copyfit.main()` 을 같은 프로세스에서 돌리는
# 시험(`test_calibration.py` 의 왕복 시험)은 `_fit.BAL_W[k] = …`, `MOTION_SLOW.update(…)`,
# `tract.HF_POLES = True` 따위를 **되돌리지 않고** 남긴다. 그 뒤에 도는 시험이 조용히 다른 엔진을
# 보게 된다 — 실측: `test_hfsep` 3 개와 `test_fitter_recovers_a_detuned_tilt` 가 단독으로는 통과하고
# 전체에서만 깨졌다(기울기 시험은 단독 0.005 dB, 전체 0.167 dB).
#
# 시험마다 `try/finally` 로 되돌리는 것은 새 시험이 늘 때마다 또 뚫린다. 그래서 **모든 시험 전후로**
# 엔진 모듈의 대문자 전역을 떠 두었다가 되돌린다. dict/list/set 은 **같은 객체의 내용을** 되돌린다 —
# 다른 모듈이 그 객체를 참조하고 있을 수 있기 때문이다.
# ---------------------------------------------------------------------------------------------
import copy as _copy
import importlib as _importlib

import numpy as _np
import pytest as _pytest

_ENGINE_MODULES = ("fit", "tract", "glottis", "noise", "voice", "tviir", "rng", "waveform",
                   "analyze", "calibration", "harmonic", "loaded_source", "turbulence", "tube",
                   "room", "residual", "events", "bands", "denoise", "segment",
                   "control", "profile")
_SCALAR = (int, float, str, bool, bytes, tuple, frozenset, type(None))
_MUTABLE = (dict, list, set, _np.ndarray)


def _snapshot():
    snap = []
    for name in _ENGINE_MODULES:
        try:
            m = _importlib.import_module(f"formant_ml.engine.{name}")
        except Exception:
            continue
        keep = {}
        for k, v in vars(m).items():
            if not k.isupper() or k.startswith("_"):
                continue
            if isinstance(v, _SCALAR):
                keep[k] = (v, None)
            elif isinstance(v, _MUTABLE):
                try:
                    keep[k] = (v, _copy.deepcopy(v))
                except Exception:
                    pass
        snap.append((m, keep))
    return snap


def _restore(snap):
    for m, keep in snap:
        for k, (obj, content) in keep.items():
            if content is not None:
                if isinstance(obj, dict):
                    obj.clear()
                    obj.update(content)
                elif isinstance(obj, list):
                    obj[:] = content
                elif isinstance(obj, set):
                    obj.clear()
                    obj.update(content)
                elif isinstance(obj, _np.ndarray) and obj.shape == content.shape:
                    obj[...] = content
            if getattr(m, k, None) is not obj:
                setattr(m, k, obj)


@_pytest.fixture(autouse=True)
def _engine_globals_are_restored():
    snap = _snapshot()
    try:
        yield
    finally:
        _restore(snap)
