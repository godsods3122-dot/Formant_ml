"""성도 — 포먼트 위치, 노치, 비강, 고차 극 보정, 스트리밍 상태."""
import numpy as np
import torch

from formant_ml.engine.tract import VocalTract
from formant_ml.engine.control import PARAM_NAMES, INDEX, default_vector

FS, HOP = 44100, 44


def _ctrl(t, **kw):
    v = np.tile(default_vector(), (t, 1))
    for k, x in kw.items():
        v[:, INDEX[k]] = x
    ten = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
    return {n: ten[..., INDEX[n]] for n in PARAM_NAMES}


def _impulse_response(tr, c, n_frames):
    n = n_frames * HOP
    x = torch.zeros(1, n); x[0, 10] = 1.0
    z = torch.zeros(1, n)
    with torch.no_grad():
        y = tr(x, z, z, z, c)["audio"][0].numpy()
    f = np.fft.rfftfreq(n, 1 / FS)
    return f, 20 * np.log10(np.abs(np.fft.rfft(y)) + 1e-12)


def test_formant_peaks_at_targets():
    tr = VocalTract(FS, HOP)
    f, H = _impulse_response(tr, _ctrl(60, f1=945, f2=1590, f3=2850), 60)
    for target in (945, 1590, 2850):
        m = (f > target * 0.85) & (f < target * 1.15)
        assert abs(f[m][H[m].argmax()] - target) < 40


def test_higher_pole_correction_keeps_highs_alive():
    """K=8 로 자르면 9~13 kHz 가 죽는다(v1 §6.8). 고차 극이 그것을 메운다."""
    a = VocalTract(FS, HOP, n_extra=0)
    b = VocalTract(FS, HOP)
    fa, Ha = _impulse_response(a, _ctrl(60, f1=945, f2=1590, f3=2850), 60)
    fb, Hb = _impulse_response(b, _ctrl(60, f1=945, f2=1590, f3=2850), 60)
    m = (fa > 9000) & (fa < 13000)
    assert Hb[m].mean() > Ha[m].mean() + 6
    assert (fb > 13000).any() and Hb[fb > 15500].max() < Hb[(fb > 800) & (fb < 2000)].max() + 20


def test_lateral_notch_is_local_and_flat_elsewhere():
    tr = VocalTract(FS, HOP)
    f, H0 = _impulse_response(tr, _ctrl(60, f1=459, f2=1914, f3=2907), 60)
    f, H1 = _impulse_response(tr, _ctrl(60, f1=459, f2=1914, f3=2907, lat_z1=3300, lat_bw=350), 60)
    d = H1 - H0
    assert d[(f > 3200) & (f < 3400)].min() < -8
    assert abs(d[(f > 300) & (f < 1500)]).max() < 1.0      # 저역은 그대로
    assert abs(d[(f > 8000) & (f < 12000)]).mean() < 1.5   # 고역도 그대로 (DC 정규화 영점의 +25 dB 없음)


def test_nasal_coupling_adds_pole_and_zero_and_off_is_identity():
    tr = VocalTract(FS, HOP)
    f, H0 = _impulse_response(tr, _ctrl(60, f1=350, f2=1400, f3=2600), 60)
    f, H1 = _impulse_response(tr, _ctrl(60, f1=350, f2=1400, f3=2600, velum=1.0,
                                         nasal_f=280, nasal_z=1700), 60)
    d = H1 - H0
    assert d[(f > 1600) & (f < 1800)].min() < -6
    assert d[(f > 240) & (f < 320)].max() > 0
    f, H2 = _impulse_response(tr, _ctrl(60, f1=350, f2=1400, f3=2600, velum=0.0), 60)
    assert np.abs(H2 - H0).max() < 1e-6


def test_streaming_state_matches_offline():
    tr = VocalTract(FS, HOP)
    t = 80; n = t * HOP
    g = torch.Generator().manual_seed(0)
    du, fr, asp, ev = (torch.randn(1, n, generator=g) for _ in range(4))
    c = _ctrl(t, f1=700, f2=1200, f3=2500, velum=0.3, lat_z1=3000, ap1_f=2000, ap1_r=0.6, a_c=0.2)
    c["f1"][:, 40:] = 400.0                                 # 급변
    torch.set_grad_enabled(False)
    full = tr(du, fr, asp, ev, c)["audio"]
    st = {}
    outs = []
    for i in range(0, t, 17):
        j = min(t, i + 17); s = slice(i * HOP, j * HOP)
        o = tr(du[:, s], fr[:, s], asp[:, s], ev[:, s], {k: v[:, i:j] for k, v in c.items()}, state=st)
        st = o["state"]; outs.append(o["audio"])
    torch.set_grad_enabled(True)
    assert torch.allclose(torch.cat(outs, -1), full, atol=1e-4, rtol=1e-4)
