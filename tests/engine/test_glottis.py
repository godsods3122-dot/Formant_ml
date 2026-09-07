"""성문 소스 — LF 파형 성질, 역치, 기동 모양, 위상 연속."""
import math

import numpy as np
import torch

from formant_ml.engine.glottis import GlottalSource, lf_pulse
from formant_ml.engine.control import PARAM_NAMES, default_vector, INDEX

FS, HOP = 44100, 44


def _ctrl(t, **kw):
    v = np.tile(default_vector(), (t, 1))
    for k, x in kw.items():
        v[:, INDEX[k]] = x
    ten = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
    return {n: ten[..., INDEX[n]] for n in PARAM_NAMES}


def test_lf_pulse_closes_flow_and_is_normalized():
    for rd in (0.3, 0.9, 1.7, 2.7):
        e = lf_pulse(rd)
        assert abs(e.sum()) < 1e-3 * np.abs(e).sum()    # ∫E dt = 0 (유량이 다시 닫힌다)
        assert abs(np.abs(e).max() - 1.0) < 1e-9
        assert e.min() < -0.9                            # 음의 정점(Ee)


def test_threshold_pressure_gates_phonation():
    g = GlottalSource(FS, HOP)
    quiet = g(_ctrl(200, p_sub=1.0, adduction=0.6))     # Pth ≈ 2.2 cmH2O 미만
    loud = g(_ctrl(200, p_sub=8.0, adduction=0.6))
    assert float(quiet["du"].abs().max()) < 1e-6
    assert float(loud["du"].abs().max()) > 0.1
    whisper = g(_ctrl(200, p_sub=8.0, adduction=0.05))  # 벌린 성문: 안 떨고 기식만
    assert float(whisper["du"].abs().max()) < 1e-6
    assert float(whisper["asp_env"].mean()) > float(loud["asp_env"].mean()) * 3


def test_onset_is_convex_logistic_not_first_order():
    """기동 곡선은 t=0 근처가 가장 느려야 한다(볼록 S). v1 HANDOFF §6.0."""
    g = GlottalSource(FS, HOP)
    st = g.physiology(_ctrl(300, p_sub=7.5, adduction=0.6))
    a = st["amp"][0].numpy()
    t90 = int(np.argmax(a >= 0.9 * a[-1]))
    t10 = int(np.argmax(a >= 0.1 * a[-1]))
    t50 = int(np.argmax(a >= 0.5 * a[-1]))
    assert 20 <= t90 <= 150                             # 20~150 ms 안에 90 %
    assert (t50 - t10) < (t90 - t50) * 3 and t10 > 2    # 초반이 늦고, 1 차 지연처럼 급출발 않음


def test_f0_follows_tension_and_pressure():
    g = GlottalSource(FS, HOP)
    lo = g.physiology(_ctrl(10, p_sub=7.0, tension=0.2))["f0"].mean()
    hi = g.physiology(_ctrl(10, p_sub=7.0, tension=0.8))["f0"].mean()
    assert hi > lo * 1.5
    p1 = g.physiology(_ctrl(10, p_sub=5.0, tension=0.5))["f0"].mean()
    p2 = g.physiology(_ctrl(10, p_sub=10.0, tension=0.5))["f0"].mean()
    assert p2 > p1                                       # 압력 ↑ → F0 ↑ (수 Hz/cmH2O)
    direct = g.physiology(_ctrl(10, p_sub=7.0, f0_target=300.0))["f0"].mean()
    assert abs(direct / 300.0 - 1.0) < 0.3


def test_phase_continuity_across_chunks():
    g = GlottalSource(FS, HOP)
    c = _ctrl(200, p_sub=7.5, adduction=0.6)
    from formant_ml.engine.rng import NoiseBank
    nb = NoiseBank(3)
    full = g(c, noise=nb)
    c1 = {k: v[:, :100] for k, v in c.items()}; c2 = {k: v[:, 100:] for k, v in c.items()}
    a = g(c1, noise=nb)
    b = g(c2, phase0=a["phase_last"], noise=nb, frame0=100, amp0=a["amp_last"], state=a["state"])
    d = (torch.cat([a["du"], b["du"]], -1) - full["du"]).abs().max()
    assert float(d) < 1e-2 * float(full["du"].abs().max()), float(d)
    d = torch.remainder(b["phase"][0, 0] - a["phase"][0, -1], 2 * math.pi)
    step = 2 * math.pi * float(a["f0"][0, -1]) / FS
    assert abs(float(d) - step) < 0.05 * step + 1e-4     # 청크 경계에서 위상이 한 스텝만 진행


def test_spectrum_falls_with_frequency_and_tilt_raises_it():
    g = GlottalSource(FS, HOP)
    y0 = g(_ctrl(300, p_sub=7.5, jitter=0.0, shimmer=0.0))["du"][0, 4410:].numpy()
    y1 = g(_ctrl(300, p_sub=7.5, jitter=0.0, shimmer=0.0, tilt=6.0))["du"][0, 4410:].numpy()
    f = np.fft.rfftfreq(len(y0), 1 / FS)
    def ratio(y):
        Y = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
        return 10 * np.log10(Y[(f > 4000) & (f < 8000)].sum() / Y[(f > 200) & (f < 1000)].sum())
    assert ratio(y0) < -20
    assert ratio(y1) > ratio(y0) + 6
