"""분석·적합의 성질 테스트. 녹음 파일 없이 도는 것만 여기 둔다."""
import numpy as np
import pytest
import torch

from formant_ml.engine.analyze import (envelope_to_lpc, lpc_formants, smooth_track,
                                       true_envelope)
from formant_ml.engine.control import ControlTrack, INDEX, default_vector
from formant_ml.engine.denoise import denoise, noise_profile, snr_report
from formant_ml.engine.fit import CopySynthFitter, mel_bank
from formant_ml.engine.profile import DEFAULT_PROFILE
from formant_ml.engine.voice import EngineConfig, VoiceEngine


FORMANTS = ((700, 80), (1220, 100), (2600, 150), (3500, 200), (4500, 250))


def _synthetic_vowel(fs=16000, f0=120.0, dur=0.3, formants=FORMANTS):
    """알려진 포먼트를 가진 합성 모음 — 분석기가 그 값을 되찾아야 한다."""
    n = int(fs * dur)
    t = np.arange(n) / fs
    x = np.zeros(n)
    for k in range(1, int(fs / 2 / f0)):
        x += np.cos(2 * np.pi * f0 * k * t + 0.7 * k * k) / k ** 1.5   # −9 dB/oct 소스
    from scipy.signal import lfilter
    for f, bw in formants:
        r = np.exp(-np.pi * bw / fs)
        th = 2 * np.pi * f / fs
        a = [1.0, -2 * r * np.cos(th), r * r]
        x = lfilter([1 - 2 * r * np.cos(th) + r * r], a, x)
    return x / (np.abs(x).max() + 1e-9)


def _envelope(x, fs=16000, f0=120.0, at=2000):
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])            # 프리엠퍼시스
    w = np.hanning(400)
    S = np.abs(np.fft.rfft(x[at:at + 400] * w)) + 1e-9
    n_cep = int(np.clip(0.45 * fs / f0, 14, 45))
    return 20.0 / np.log(10) * true_envelope(np.log(S), n_cep)


def test_true_envelope_removes_harmonic_comb():
    """참 포락선은 하모닉 빗살을 지우고 로그 스펙트럼을 대체로 위에서 감싼다."""
    x = _synthetic_vowel()
    w = np.hanning(400)
    S = np.log(np.abs(np.fft.rfft(x[2000:2400] * w)) + 1e-9)
    env = true_envelope(S, 45)
    # 마지막 켑스트럼 평활 한 번은 max 없이 걸리므로 완전한 상계는 아니다.
    assert np.mean(env >= S - 1e-6) > 0.8
    assert np.max(S - env) < 6.0
    assert np.std(np.diff(env)) < np.std(np.diff(S)) / 3       # 빗살이 크게 줄었다


@pytest.mark.parametrize("f0", [120.0, 200.0])
def test_lpc_formants_recover_known_poles(f0):
    """프리엠퍼시스가 있으면 알려진 포먼트를 8 % 안에서 되찾는다.

    프리엠퍼시스가 없으면 소스의 −9 dB/oct 기울기가 LPC 극 하나를 저역에서 잡아먹고
    배정이 한 칸씩 밀린다. 그 회귀를 아래 test_preemphasis_is_required 가 막는다.
    F0 가 300 Hz 를 넘으면 F1 이 하모닉 사이에 끼어 편향이 커진다(알려진 한계).
    """
    env = _envelope(_synthetic_vowel(f0=f0), f0=f0)
    got = lpc_formants(envelope_to_lpc(env, 14), 16000, 150.0, 5500.0)
    for k, (want, _) in enumerate(FORMANTS[:4]):
        assert abs(got[k][0] - want) < 0.08 * want, (f0, k, got)


def test_preemphasis_is_required():
    """프리엠퍼시스를 빼면 가짜 저역 극이 F1 자리를 차지한다 — 회귀 방지."""
    x = _synthetic_vowel()
    w = np.hanning(400)
    S = np.abs(np.fft.rfft(x[2000:2400] * w)) + 1e-9
    env = 20.0 / np.log(10) * true_envelope(np.log(S), 45)
    got = lpc_formants(envelope_to_lpc(env, 14), 16000, 150.0, 5500.0)
    assert abs(got[0][0] - 700) > 0.10 * 700, got


def test_smooth_track_preserves_length_and_reduces_jitter():
    rng = np.random.default_rng(0)
    x = 900 + rng.standard_normal(200) * 60.0
    s = smooth_track(x, 9, 12)
    assert s.shape == x.shape
    assert np.std(np.diff(s)) < np.std(np.diff(x)) / 4
    assert abs(s.mean() - x.mean()) < 15.0


def test_denoise_lowers_floor_and_keeps_peak():
    fs = 16000
    x = _synthetic_vowel(fs)
    y = np.concatenate([np.zeros(fs // 2), x, np.zeros(fs // 2)])
    rng = np.random.default_rng(1)
    y = y + rng.standard_normal(len(y)) * 0.01
    noise = noise_profile(y, fs)
    d = denoise(y, fs, noise)
    r = snr_report(y, d, fs, noise)
    assert r["noise_db_out"] < r["noise_db_in"] - 5.0          # 바닥이 내려간다
    assert abs(r["peak_db_out"] - r["peak_db_in"]) < 3.0       # 정점은 그대로


def test_mel_bank_rows_sum_to_one():
    m = mel_bank(256, 48000, 48)
    assert m.shape == (48, 129)
    live = m.sum(1) > 0
    # 5.3 ms 창의 빈 간격(187 Hz)보다 좁은 저역 멜 대역은 빈을 하나도 못 잡는다.
    # 그건 정상이다 — 잡은 대역은 정확히 1 로 정규화되어 있어야 한다.
    assert live.sum() >= 40
    assert torch.allclose(m.sum(1)[live], torch.ones(int(live.sum())), atol=1e-5)


@pytest.fixture(scope="module")
def _engine():
    return VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, residual=False),
                       DEFAULT_PROFILE)


def _track(n=60):
    v = np.tile(default_vector(), (n, 1))
    tr = ControlTrack(v, 1.0)
    tr["p_sub"] = 7.0; tr["adduction"] = 0.6; tr["f0_target"] = 200.0
    tr["f1"] = 700; tr["f2"] = 1200; tr["f3"] = 2600; tr["f4"] = 3600
    tr["bw1"] = 80; tr["bw2"] = 100; tr["bw3"] = 150; tr["bw4"] = 200
    tr["residual_mix"] = 0.0
    return tr.clamp()


def test_fitter_recovers_its_own_render(_engine):
    """자기 렌더를 목표로 주면 포락 일치율이 매우 높아야 한다 (동일성 검사)."""
    tr = _track()
    target = _engine.render(tr)
    f = CopySynthFitter(_engine, target, 48000, tr)
    with torch.no_grad():
        _, sc, env_sc, _ = f.loss()
    assert 100 * (1 - float(env_sc)) > 99.0
    assert 100 * (1 - float(sc)) > 95.0


def test_fitter_gradients_are_finite(_engine):
    tr = _track()
    target = _engine.render(tr)
    rng = np.random.default_rng(2)
    target = target + rng.standard_normal(len(target)) * 1e-3 * np.abs(target).max()
    f = CopySynthFitter(_engine, target, 48000, tr)
    l, _, _, _ = f.loss()
    l.backward()
    for name, t in (("w", f.w), ("d", f.d), ("log_gain", f.log_gain)):
        assert t.grad is not None and torch.isfinite(t.grad).all(), name


def test_fitter_recovers_a_detuned_tilt(_engine):
    """소스 기울기가 7 dB/oct 틀린 출발점에서 전역 단계가 원래 값을 되찾는다.

    lr 은 0.05 여야 한다. 0.2 로 하면 30 개 오프셋이 한꺼번에 넘어가 출발점보다
    나빠진 채로 끝난다(실측: 손실 1.3185 에서 한 번도 못 내려왔다).
    """
    tr = _track()
    target = _engine.render(tr)
    bad = ControlTrack(tr.values.copy(), tr.frame_ms)
    bad["tilt"] = 9.0
    f = CopySynthFitter(_engine, target, 48000, bad)
    f.sizes = [256, 512]                 # 손실 비교는 같은 창 집합에서만 뜻이 있다
    with torch.no_grad():
        l0 = float(f.loss()[0])
    rep = f.fit(60, 0.05, 999, verbose=False, params=[f.d, f.log_gain], sizes=(256, 512))
    with torch.no_grad():
        l1 = float(f.loss()[0])
    assert l1 < 0.25 * l0, (l0, l1)
    assert rep.db < 0.6                                   # 평균 스펙트럼 오차 dB
    got = dict((n, b) for n, _, b in f.moved())["tilt"]
    assert abs(got - 2.0) < 1.5, got


def test_best_snapshot_is_taken_before_the_step(_engine):
    """되돌린 해의 손실이 기록된 최선과 같아야 한다 (step 뒤에 사본을 뜨면 어긋난다)."""
    tr = _track()
    bad = ControlTrack(tr.values.copy(), tr.frame_ms)
    bad["tilt"] = 9.0
    f = CopySynthFitter(_engine, _engine.render(tr), 48000, bad)
    rep = f.fit(6, 0.3, 999, verbose=False, params=[f.d, f.log_gain], sizes=(256, 512))
    with torch.no_grad():
        assert abs(float(f.loss()[0]) - rep.loss) < 1e-5


def test_global_offset_never_touches_formants(_engine):
    """전역 오프셋은 포먼트를 건드리지 않는다 (F4 가 F1 아래로 가는 해를 막는다)."""
    tr = _track()
    f = CopySynthFitter(_engine, _engine.render(tr), 48000, tr)
    with torch.no_grad():
        f.d.add_(3.0)                       # 모든 오프셋을 크게 흔들어도
        before = f.control()[0, :, INDEX["f1"]].clone()
    for name in ("f1", "f2", "f3", "f4", "f0_target", "velum"):
        i = f.names.index(name)
        assert float(f.d_mask[i]) == 0.0, name
    with torch.no_grad():
        assert torch.allclose(f.control()[0, :, INDEX["f1"]], before)


def test_penalty_punishes_out_of_order_formants(_engine):
    tr = _track()
    f = CopySynthFitter(_engine, _engine.render(tr), 48000, tr)
    with torch.no_grad():
        p0 = float(f.penalty())
        i2 = f.names.index("f2")
        f.w[:, i2] = -20.0 / float(f.scale[i2])       # F2 를 F1 아래로
        p1 = float(f.penalty())
    assert p0 < 1e-9 < p1
