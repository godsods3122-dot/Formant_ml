"""파형 일치도 — 조화 분해, 분절 SNR, 하모닉별 위상 오차."""
import numpy as np
import pytest

from formant_ml.engine.waveform import (decompose, harmonic_fit, phase_error,
                                        report, seg_snr)

FS = 48000


def _tone(f0=200.0, dur=0.2, n_h=40, phase=0.0, fs=FS):
    n = int(fs * dur)
    t = np.arange(n) / fs
    return sum(np.cos(2 * np.pi * f0 * k * t + phase * k) / k for k in range(1, n_h + 1))


def test_harmonic_model_captures_a_periodic_signal():
    x = _tone()
    f0 = np.full(int(0.2 * 1000), 200.0)
    _, h, r = decompose(x, FS, f0, FS // 1000)
    assert (r ** 2).sum() / (x ** 2).sum() < 0.01          # 잔차가 1 % 미만
    assert seg_snr(x, h, FS) > 30.0


def test_noise_is_not_captured_by_the_harmonic_model():
    """난류는 조화 모형에 안 잡힌다 — 그래서 파형으로 비교하면 안 된다."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(int(FS * 0.2))
    f0 = np.full(200, 200.0)
    _, h, r = decompose(x, FS, f0, FS // 1000)
    assert (h ** 2).sum() / (x ** 2).sum() < 0.5           # 절반도 못 설명한다


def test_seg_snr_is_high_for_identity_and_low_for_noise():
    x = _tone()
    rng = np.random.default_rng(1)
    assert seg_snr(x, x, FS) > 90.0
    assert seg_snr(x, rng.standard_normal(len(x)) * x.std(), FS) < 3.0


def test_pure_time_shift_is_not_counted_as_shape_error():
    """순수한 시간 이동은 파형의 **모양**을 안 바꾼다 — 차수 비례 성분으로 빠져야 한다."""
    k = np.arange(1, 41)
    at = (1.0 / k) * np.exp(1j * 0.3 * k)
    shift = 0.11
    as_ = (1.0 / k) * np.exp(1j * (0.3 * k - shift * k))   # 순수 지연
    shape, slope, nk = phase_error(at, as_)
    assert shape < 1.0, shape                              # 모양 오차 거의 0
    assert abs(slope - np.degrees(shift)) < 1.0            # 기울기로 잡힌다
    assert nk > 10


def test_shape_error_is_detected():
    """차수에 비례하지 않는 위상 교란은 모양 오차로 잡힌다."""
    k = np.arange(1, 41)
    at = (1.0 / k) * np.exp(1j * 0.3 * k)
    rng = np.random.default_rng(2)
    as_ = at * np.exp(1j * rng.standard_normal(len(k)) * 0.5)
    shape, _, _ = phase_error(at, as_)
    assert shape > 10.0, shape


def test_report_on_identity_is_perfect():
    x = _tone()
    f0 = np.full(200, 200.0)
    r = report(x, x, FS, f0, FS // 1000)
    assert r["snr_harmonic"] > 60.0
    assert r["phase_shape_deg"] < 1.0
    assert abs(r["residual_level_db"]) < 3.0
