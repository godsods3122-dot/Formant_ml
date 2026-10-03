"""배음 위상 흔들림의 **시간 상관** (glottis.HJIT_TAU_MS, MEASUREMENTS §52.265).

같은 세기 σ 라도 ψ 가 빠르게 뒤섞이면 포락을 훨씬 많이 깎는다(실측 20°: 주기마다 무작위 84.0 대 천천히 92.2).
그래서 σ 는 **보존**하면서 상관 시간만 늘리는 손잡이다.
"""
import numpy as np
import pytest
import torch

from formant_ml.engine import glottis as G


def test_kernel_preserves_the_variance_of_white_noise():
    """평활은 분산을 줄인다 — 커널이 1/√Σh² 로 되돌려야 ψ 의 세기가 그대로다."""
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.standard_normal(20000), dtype=torch.float64)
    for tau in (1.0, 3.0, 10.0):
        h = G._hjit_kernel(tau, 1.0, torch.float64, x.device)
        y = torch.nn.functional.conv1d(x.view(1, 1, -1), h.view(1, 1, -1))[0, 0]
        assert abs(float(y.std()) - 1.0) < 0.05, (tau, float(y.std()))


def test_kernel_actually_slows_the_noise_down():
    """상관 시간이 길수록 이웃 표본끼리 더 닮아야 한다 (지연 1 자기상관)."""
    rng = np.random.default_rng(1)
    x = torch.as_tensor(rng.standard_normal(40000), dtype=torch.float64)
    prev = -1.0
    for tau in (0.5, 2.0, 6.0):
        h = G._hjit_kernel(tau, 1.0, torch.float64, x.device)
        y = torch.nn.functional.conv1d(x.view(1, 1, -1), h.view(1, 1, -1))[0, 0].numpy()
        r1 = float(np.corrcoef(y[:-1], y[1:])[0, 1])
        assert r1 > prev, (tau, r1, prev)
        prev = r1
    assert prev > 0.8, prev          # 6 ms 면 이웃 프레임이 거의 같다


def test_zero_tau_is_the_old_behaviour():
    assert G.HJIT_TAU_MS == 0.0


@pytest.mark.parametrize("tau", [0.0, 4.0])
def test_render_is_finite_and_the_stream_path_matches_the_whole(tau, monkeypatch):
    """청크로 흘려도 통째로 렌더한 것과 같아야 한다 — 평활이 청크 독립성을 깨면 안 된다.

    (`noise.white` 가 위치 기반이라 성립하는 성질이고, 커널 폭만큼 더 받아서 자르는 구현이 그것을 지키는지 본다.)
    """
    from formant_ml.engine.control import ControlTrack, default_vector
    from formant_ml.engine.profile import DEFAULT_PROFILE
    from formant_ml.engine.voice import EngineConfig, VoiceEngine

    monkeypatch.setattr(G, "HARM_PHASE_JIT", 0.02)
    monkeypatch.setattr(G, "HARM_PHASE_ONSET", 1500.0)
    monkeypatch.setattr(G, "HJIT_TAU_MS", tau)
    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, residual=False), DEFAULT_PROFILE)
    v = np.tile(default_vector(), (160, 1))
    tr = ControlTrack(v, 1.0)
    tr["p_sub"] = 7.0; tr["adduction"] = 0.6; tr["f0_target"] = 200.0
    tr["residual_mix"] = 0.0
    tr = tr.clamp()
    whole = np.asarray(eng.render(tr), dtype=np.float64)
    assert np.isfinite(whole).all() and np.abs(whole).max() > 0.0
    streamed = np.concatenate([np.asarray(c, dtype=np.float64) for c in eng.stream(tr, chunk_ms=20.0)])
    n = min(len(whole), len(streamed))
    assert n > 1000, n
    assert np.abs(whole[:n] - streamed[:n]).max() < 1e-9, float(np.abs(whole[:n] - streamed[:n]).max())


def test_a_longer_correlation_makes_neighbouring_cycles_more_alike(monkeypatch):
    """맞바꿈을 못 박는다: ψ 가 빠르면 이웃 주기의 파형이 달라진다(= 선이 번진다).

    상관 시간을 늘리면 **주기 상관**이 올라간다 — 저장소가 판마다 찍는 그 값이다. 즉 이 손잡이는
    포락(위상 비용)과 주기성(선 번짐)을 맞바꾼다.
    """
    from formant_ml.engine.control import ControlTrack, default_vector
    from formant_ml.engine.profile import DEFAULT_PROFILE
    from formant_ml.engine.voice import EngineConfig, VoiceEngine

    F0, FS = 200.0, 48000

    def cycle_corr(tau):
        monkeypatch.setattr(G, "HARM_PHASE_JIT", 0.6)
        monkeypatch.setattr(G, "HARM_PHASE_ONSET", 500.0)
        monkeypatch.setattr(G, "HJIT_TAU_MS", tau)
        eng = VoiceEngine(EngineConfig(sample_rate=FS, frame_ms=1.0, residual=False), DEFAULT_PROFILE)
        v = np.tile(default_vector(), (400, 1))
        tr = ControlTrack(v, 1.0)
        tr["p_sub"] = 7.0; tr["adduction"] = 0.6; tr["f0_target"] = F0
        tr["residual_mix"] = 0.0
        y = np.asarray(eng.render(tr.clamp()), dtype=np.float64)
        lag = int(round(FS / F0))
        a, b = y[FS // 10:-lag], y[FS // 10 + lag:]
        n = min(len(a), len(b))
        return float(np.corrcoef(a[:n], b[:n])[0, 1])

    fast, slow = cycle_corr(0.0), cycle_corr(8.0)
    assert slow > fast + 0.002, (fast, slow)   # 실측 차 0.0039 (지터 0.6)
