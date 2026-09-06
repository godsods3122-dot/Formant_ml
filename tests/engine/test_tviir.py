"""시변 IIR 원시연산 — 세 구현의 일치, 안정성, 위상 연속성."""
import math

import numpy as np
import pytest
import torch
from scipy.signal import lfilter

from formant_ml.engine import tviir as T

FS = 44100.0


def _rand(b, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, n, generator=g, dtype=torch.float64)


def test_scan_matches_sequential_time_varying():
    x = _rand(2, 3000)
    n = x.shape[1]
    f = torch.linspace(300.0, 2500.0, n, dtype=torch.float64).expand(2, n)
    bw = torch.linspace(60.0, 400.0, n, dtype=torch.float64).expand(2, n)
    co = T.resonator_coeffs(f, bw, FS)
    y1, z1 = T.tv_biquad(x, *co)
    y2, z2 = T.tv_biquad_seq(x, *co)
    assert torch.allclose(y1, y2, atol=1e-9, rtol=1e-7)
    assert torch.allclose(z1, z2, atol=1e-9)


def test_scan_matches_scipy_constant_coeffs_and_state_convention():
    x = _rand(1, 2048)
    co = [c if not torch.is_tensor(c) else c for c in
          T.resonator_coeffs(torch.tensor(800.0, dtype=torch.float64),
                             torch.tensor(90.0, dtype=torch.float64), FS)]
    b = [float(co[0]), float(co[1]), float(co[2])]
    a = [1.0, float(co[3]), float(co[4])]
    zi = np.array([0.3, -0.2])
    y_ref, zf_ref = lfilter(b, a, x[0].numpy(), zi=zi)
    y, zf = T.tv_biquad(x, *co, zi=torch.tensor(zi).unsqueeze(0))
    assert np.allclose(y[0].numpy(), y_ref, atol=1e-10)
    assert np.allclose(zf[0].numpy(), zf_ref, atol=1e-10)


def test_numpy_realtime_path_matches_scan():
    x = _rand(1, 5000)
    n = x.shape[1]
    f = torch.linspace(500.0, 900.0, n, dtype=torch.float64).expand(1, n)
    bw = torch.full((1, n), 120.0, dtype=torch.float64)
    co = T.resonator_coeffs(f, bw, FS)
    y, _ = T.tv_biquad(x, *co)
    y_np, _ = T.tv_biquad_np(x[0].numpy(), *[c[0].numpy() for c in co])
    tol = 1e-9 if T.HAVE_NUMBA else 2e-2
    assert np.abs(y[0].numpy() - y_np).max() < tol


def test_streaming_chunks_equal_offline():
    x = _rand(1, 4000)
    n = x.shape[1]
    f = torch.linspace(400.0, 1200.0, n, dtype=torch.float64).expand(1, n)
    co = T.resonator_coeffs(f, torch.full((1, n), 80.0, dtype=torch.float64), FS)
    y_off, _ = T.tv_biquad(x, *co)
    zi, outs = None, []
    for i in range(0, n, 333):
        j = min(n, i + 333)
        y, zi = T.tv_biquad(x[:, i:j], *[c[:, i:j] for c in co], zi=zi)
        outs.append(y)
    assert torch.allclose(torch.cat(outs, -1), y_off, atol=1e-9)


def test_resonator_peak_and_dc_gain():
    n = 1 << 15
    x = torch.zeros(1, n, dtype=torch.float64); x[0, 0] = 1.0
    co = T.resonator_coeffs(torch.tensor(1000.0, dtype=torch.float64),
                            torch.tensor(100.0, dtype=torch.float64), FS)
    y, _ = T.tv_biquad(x, *co)
    H = np.abs(np.fft.rfft(y[0].numpy()))
    f = np.fft.rfftfreq(n, 1 / FS)
    assert abs(f[H.argmax()] - 1000.0) < 5.0
    assert abs(H[0] - 1.0) < 1e-6                       # DC 이득 1


def test_antiresonator_notch_and_allpass_flat():
    n = 1 << 14
    x = torch.zeros(1, n, dtype=torch.float64); x[0, 0] = 1.0
    f = np.fft.rfftfreq(n, 1 / FS)
    co = T.antiresonator_coeffs(torch.tensor(3000.0, dtype=torch.float64),
                                torch.tensor(200.0, dtype=torch.float64), FS)
    y, _ = T.tv_biquad(x, *co)
    H = np.abs(np.fft.rfft(y[0].numpy()))
    assert abs(f[H[(f > 500)].argmin() + (f > 500).argmax()] - 3000.0) < 20.0
    assert abs(H[0] - 1.0) < 1e-6
    co = T.allpass_coeffs(torch.tensor(2000.0, dtype=torch.float64), 0.9, FS)
    y, _ = T.tv_biquad(x, *co)
    H = np.abs(np.fft.rfft(y[0].numpy()))
    assert H.max() < 1.0 + 1e-6 and H.min() > 1.0 - 1e-6


def test_stability_under_extreme_sweep():
    n = 44100
    x = _rand(1, n, 3)
    f = torch.linspace(100.0, 20000.0, n, dtype=torch.float64).expand(1, n)
    bw = torch.full((1, n), 25.0, dtype=torch.float64)
    y, _ = T.tv_biquad(x, *T.resonator_coeffs(f, bw, FS))
    assert torch.isfinite(y).all() and y.abs().max() < 1e3


def test_phase_continuity_at_abrupt_parameter_jump():
    """파라미터가 한 샘플에 700→300 Hz 로 뛰어도 출력 파형은 연속이다.

    v1 의 블록 OLA 는 프레임 경계에서 파형 자체가 찢어졌다(RIEUL §6.5).
    여기서는 상태가 이어지므로 샘플 간 차분이 정상 구간과 같은 크기다.
    """
    n = 8000
    t = torch.arange(n, dtype=torch.float64) / FS
    x = torch.sin(2 * math.pi * 150.0 * t).unsqueeze(0)
    f = torch.full((1, n), 700.0, dtype=torch.float64); f[:, n // 2:] = 300.0
    bw = torch.full((1, n), 80.0, dtype=torch.float64)
    y, _ = T.tv_biquad(x, *T.resonator_coeffs(f, bw, FS))
    d = (y[0, 1:] - y[0, :-1]).abs()
    around = d[n // 2 - 5: n // 2 + 5].max()
    typical = d[1000:3000].max()
    assert around < 2.0 * typical


def test_gradient_flows_through_scan():
    x = _rand(1, 2000)
    f = torch.full((1, 2000), 900.0, dtype=torch.float64, requires_grad=True)
    bw = torch.full((1, 2000), 100.0, dtype=torch.float64)
    y, _ = T.tv_biquad(x, *T.resonator_coeffs(f, bw, FS))
    (y ** 2).sum().backward()
    assert torch.isfinite(f.grad).all() and f.grad.abs().sum() > 0


def test_peaking_eq_gain_at_center():
    n = 1 << 15
    x = torch.zeros(1, n, dtype=torch.float64); x[0, 0] = 1.0
    co = T.peaking_eq_coeffs(torch.tensor(2000.0, dtype=torch.float64), 2.0,
                             torch.tensor(6.0, dtype=torch.float64), FS)
    y, _ = T.tv_biquad(x, *co)
    H = 20 * np.log10(np.abs(np.fft.rfft(y[0].numpy())))
    f = np.fft.rfftfreq(n, 1 / FS)
    assert abs(H[np.argmin(np.abs(f - 2000.0))] - 6.0) < 0.1
    assert abs(H[0]) < 0.05
