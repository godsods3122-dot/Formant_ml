"""음원 판정 척도(`scripts/diag/source_ab.py`) 자체의 점검 — 알려진 지터·가산 잡음에 단조여야 한다 (§52.17)."""
import importlib.util
from pathlib import Path

import numpy as np


def _load():
    path = Path(__file__).resolve().parents[2] / "scripts" / "diag" / "source_ab.py"
    spec = importlib.util.spec_from_file_location("source_ab", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selfcheck_passes():
    assert _load().selfcheck() == 0


def test_detrend_keeps_harmonic_ripple():
    # 예전 21 칸 추세 제거는 F0 240 Hz(20 칸) 의 배음 무늬를 지웠다. 완전 주기 신호의 배음성은 높아야 한다.
    m = _load()
    fs = 48000
    f0 = np.full(1500, 240.0)
    ok = np.zeros(1500, bool)
    ok[100:1400] = True
    harm, hnr = m.band_metrics(m._synth(fs, 0.0, None), fs, f0, ok)
    assert min(harm) > 0.5
    assert min(hnr) > 40.0


def test_own_f0_contour_ignores_invalid_frames():
    m = _load()
    f = np.array([200.0, 0.0, 200.0, 400.0, 200.0])
    ok = np.array([True, False, True, False, True])
    c = m.contour_log2(f, ok, width=5)
    assert np.isnan(c[1]) and np.isnan(c[3])
    assert np.allclose(c[[0, 2, 4]], np.log2(200.0))


def test_band_metrics_high_f0_does_not_break_detrend():
    # F0 470 Hz 면 추세 제거 창(4·lag+1 = 161 칸) 이 0.3-2 kHz 대역(145 칸) 보다 길어져 길이 불일치로 죽었다.
    m = _load()
    fs = 48000
    f0 = np.full(1500, 470.0)
    ok = np.zeros(1500, bool)
    ok[100:1400] = True
    harm, hnr = m.band_metrics(m._synth(fs, 0.0, None, f0=470.0), fs, f0, ok)
    assert np.all(np.isfinite(hnr))
