"""분석기가 **재현되는가** — 판정 지표로 쓰기 전에 통과해야 하는 검사.

    PYTHONPATH=src python3 -m pytest tests/test_analysis_reproducibility.py -q

같은 소리를 전체 발화 안에서 보는 것과 잘라내서 보는 것이 다르면, 그 분석기로
잰 어떤 값도 재현되지 않는다. 실제로 그 상태에서 상수를 고르다가 발췌와 전체의
결과가 어긋나는 것을 오래 못 알아챘다(docs/PLAN_TONGUE2D.md §0 의 실수 #8).

**측정 결과(2026-09-06): `track_formants` 는 슬롯 2 이상에서 최대 52 % 어긋난다.**
전역 씨 프레임 + 양방향 추적이라 분석 구간 길이에 결과가 딸려 오기 때문이다.
그래서 이 추적기는 **진단 전용**이고, 합격 판정에는 쓰지 않는다(§3.1).
여기서는 **우리가 실제로 의존하는 F1/F2 만** 재현되는지 지킨다.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from formant_ml.analysis.acoustic import load
from formant_ml.analysis.track import track_formants
from formant_ml.config import Config

FS = 24000
REC = os.path.join(os.path.dirname(__file__), "..", "reference", "recordings",
                   "ko_liquid_ra-eulla-ara_male_44k.wav")
FULL = (0.40, 6.20)
TOKENS = {"ra": (0.50, 1.02), "eulla": (1.58, 2.55),
          "ara1": (3.20, 3.86), "ara2": (5.36, 6.08)}


def _drift():
    """전체 발화 안에서 본 값 vs 잘라내서 본 값의 슬롯별 차이 [%]."""
    cfg = Config()
    hop = cfg.audio.hop_size
    raw, _ = load(REC, FS)
    y = raw[int(FULL[0] * FS):int(FULL[1] * FS)]
    y = y / max(np.abs(y).max(), 1e-9)
    f_full, _ = track_formants(y, FS, hop, 1024, n=cfg.filt.n_formants)
    out = {}
    for name, (t0, t1) in TOKENS.items():
        seg = raw[int(t0 * FS):int(t1 * FS)]
        seg = seg / max(np.abs(seg).max(), 1e-9)
        f_cut, _ = track_formants(seg, FS, hop, 1024, n=cfg.filt.n_formants)
        i0 = int(round((t0 - FULL[0]) * FS / hop))
        a = f_full[i0:i0 + len(f_cut)]
        m = min(len(a), len(f_cut))
        out[name] = 100 * np.abs(np.median(f_cut[:m], 0)
                                 - np.median(a[:m], 0)) / np.maximum(
                                     np.median(a[:m], 0), 1e-9)
    return out


def test_f1_f2_are_reproducible_across_analysis_windows():
    """F1·F2 는 분석 구간 길이와 무관해야 한다 — 여기까지만 의존한다."""
    if not os.path.exists(REC):
        return
    for name, d in _drift().items():
        assert d[0] < 3.0 and d[1] < 3.0, (
            f"{name}: F1 {d[0]:.1f} %, F2 {d[1]:.1f} % 어긋난다")


def test_upper_slots_are_documented_as_unreliable():
    """상위 슬롯이 **여전히** 못 미더운지 확인한다 (문서와 코드가 안 어긋나게).

    이 검사가 실패한다면 추적기가 좋아진 것이므로 축하할 일이다. 그때는
    docs/PLAN_TONGUE2D.md §3.3 과 이 파일의 설명을 같이 고쳐라 — 지금은
    '상위 슬롯은 판정에 쓰지 않는다' 가 문서와 코드 양쪽의 전제다.
    """
    if not os.path.exists(REC):
        return
    worst = max(float(d[2:10].max()) for d in _drift().values())
    assert worst > 5.0, (
        f"상위 슬롯 최대 어긋남이 {worst:.1f} % 로 줄었다 — 추적기가 좋아졌다면 "
        "§3.3 과 이 파일의 전제를 갱신하라")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1; print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} 통과")
    sys.exit(1 if failed else 0)
