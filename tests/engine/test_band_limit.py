"""녹음 사슬의 고정 대역 상한 (`fit.BAND_LIMIT_HZ`, MEASUREMENTS §52.291)."""
import numpy as np
import pytest
import torch

from formant_ml.engine import fit as F


def _fir(fc, fs=48000.0, taps=513):
    return F._codec_lowpass(fc, fs, taps)


def test_lowpass_is_identity_above_nyquist():
    h = _fir(24000.0)
    assert h[len(h) // 2] == pytest.approx(1.0)
    assert np.allclose(np.delete(h, len(h) // 2), 0.0)


def test_lowpass_is_symmetric_and_unit_dc():
    h = _fir(20050.0)
    assert np.allclose(h, h[::-1])
    assert h.sum() == pytest.approx(1.0)


def test_wall_drops_by_20_db_within_300_hz():
    """실측한 벽(20.0 → 20.2 kHz 에서 21 dB)과 같은 기울기여야 한다."""
    h = _fir(20050.0)
    f = np.fft.rfftfreq(8192, 1.0 / 48000.0)
    H = 20.0 * np.log10(np.abs(np.fft.rfft(h, 8192)) + 1e-30)
    lo = H[np.argmin(np.abs(f - 19800.0))]
    hi = H[np.argmin(np.abs(f - 20300.0))]
    assert lo > -1.0
    assert lo - hi > 20.0


def test_band_limit_does_not_shift_time(monkeypatch):
    """선형 위상 + 대칭 채움 = 군지연 상쇄. 펄스 위상이 표본 단위라 한 표본도 밀리면 안 된다."""
    monkeypatch.setattr(F, "BAND_LIMIT_HZ", 20050.0)
    fs = 48000.0
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4096)
    x[2000] += 20.0                                   # 뾰족한 표시
    class _Stub:
        pass
    st = _Stub()
    st.fs = fs
    st.target = torch.zeros(1, 4096)
    y = F.CopySynthFitter._band_limit(st, torch.as_tensor(x, dtype=torch.float32).reshape(1, -1))
    assert y.shape == (1, 4096)
    assert int(torch.argmax(y.abs(), dim=-1)) == 2000


def test_band_limit_removes_energy_above_the_wall(monkeypatch):
    monkeypatch.setattr(F, "BAND_LIMIT_HZ", 20050.0)
    fs = 48000.0
    n = 48000
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 21000.0 * t) + np.sin(2 * np.pi * 10000.0 * t)
    class _Stub:
        pass
    st = _Stub()
    st.fs = fs
    st.target = torch.zeros(1, n)
    y = F.CopySynthFitter._band_limit(st, torch.as_tensor(x, dtype=torch.float32).reshape(1, -1))
    Y = np.abs(np.fft.rfft(y[0].numpy()))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    keep = Y[np.argmin(np.abs(f - 10000.0))]
    cut = Y[np.argmin(np.abs(f - 21000.0))]
    assert 20.0 * np.log10(keep / max(cut, 1e-12)) > 40.0


def test_default_is_off():
    assert F.BAND_LIMIT_HZ == 0.0
