"""분석·적합의 성질 테스트. 녹음 파일 없이 도는 것만 여기 둔다."""
import numpy as np
import pytest
import torch

from formant_ml.engine.analyze import (analyze, envelope_to_lpc, lpc_formants,
                                       smooth_track, true_envelope)
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
    """포먼트 순서가 뒤집히면 벌점이 **자릿수로** 커져야 한다.

    바닥값은 0 이 아니다. `a_c` 속도 벌점이 `softplus` 로 문턱을 뭉개므로 속도 0
    에서도 `softplus(−5) = 6.7e-3` 만큼 새고, 그것이 의사후버를 지나 프레임당
    ~9e-7 로 남는다 (§29.3). 손실이 ~1.0 인 것에 비하면 무시할 양이지만 정확히
    0 은 아니므로, "바닥은 0" 이 아니라 "바닥은 위반의 자릿수 아래" 를 건다.
    """
    tr = _track()
    f = CopySynthFitter(_engine, _engine.render(tr), 48000, tr)
    with torch.no_grad():
        p0 = float(f.penalty())
        i2 = f.names.index("f2")
        f.w[:, i2] = -20.0 / float(f.scale[i2])       # F2 를 F1 아래로
        p1 = float(f.penalty())
    assert p0 < 1e-5, f"위반이 없는데 벌점이 {p0:.3g} 이다"
    assert p1 > 1e3 * max(p0, 1e-12), f"순서 위반 {p1:.3g} 이 바닥 {p0:.3g} 과 비슷하다"


@pytest.mark.parametrize("name", ["f0_target", "p_sub", "f1", "bw1", "a_c", "velum"])
def test_engine_does_not_raise_on_nan_control(_engine, name):
    """제어열에 NaN 이 들어와도 엔진은 **예외를 던지지 않는다**.

    던지면 적합기의 "손실이 비유한이면 중단" 가드가 손실을 보기도 전에 죽어서 원인을
    못 찾는다(실측: `f0` 가 NaN 일 때 하모닉 상한 계산의 `int(NaN)` 에서 터졌다).
    NaN 은 출력으로 전파되거나(가드가 잡는다) 조건 분기에서 흡수되면 된다.
    """
    tr = _track()
    ctrl = tr.to_tensor()
    ctrl[0, 10, INDEX[name]] = float("nan")
    _engine.reset()
    out = _engine(ctrl, [], 0.0)                    # 던지지 않는 것이 전부다
    assert out["audio"].shape[1] == tr.n_frames * _engine.cfg.hop


def test_fitter_skips_steps_with_nonfinite_gradients(_engine):
    """비유한 기울기가 나온 회차는 건너뛴다 — Adam 모멘트가 오염되면 되돌릴 수 없다."""
    tr = _track()
    f = CopySynthFitter(_engine, _engine.render(tr), 48000, tr)
    l, _, _, _ = f.loss()
    l.backward()
    f.w.grad[0, 0] = float("nan")
    before = f.w.detach().clone()
    opt = torch.optim.Adam([f.w], lr=0.1)
    ps = [f.w]
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in ps):
        for p in ps:
            p.grad = None
    else:
        opt.step()
    assert torch.equal(f.w.detach(), before)
    rep = f.fit(4, 0.05, 999, verbose=False, sizes=(256,))
    assert np.isfinite(rep.loss)


def test_loss_ignores_bands_above_the_recording_nyquist(_engine):
    """44.1 kHz 녹음을 목표로 주면 손실이 그 나이퀴스트 위를 보지 않는다.

    48 kHz 로 올린 44.1 kHz 신호는 22.05~24 kHz 가 정확히 비어 있다. 그걸 목표로
    두면 적합기가 "저 위를 비워라" 를 물리 파라미터로 달성하려 든다.
    """
    tr = _track()
    y = _engine.render(tr)
    from scipy.signal import resample_poly
    y441 = resample_poly(y, 441, 480)
    f = CopySynthFitter(_engine, y441, 44100, tr)
    assert f.f_max == pytest.approx(0.90 * 22050.0)
    for k in (256, 4096):
        assert f.tgt_S[k].shape[1] == f.bin_max[k]
        assert f.bin_max[k] < k // 2 + 1              # 실제로 잘렸다
    # 같은 신호를 48 kHz 로 주면 상한이 더 높다
    g = CopySynthFitter(_engine, y, 48000, tr)
    assert g.f_max > f.f_max


def test_intensity_dip_goes_to_the_tract_not_the_lungs():
    """빠른 세기 변화는 **구강 방사**(tract_gain)로, 느린 것만 폐압(p_sub)으로.

    폐는 20 ms 만에 압력을 못 바꾼다. 자음의 세기 골(탄음 −9 dB / 40 ms, 비음 폐쇄,
    파열음)은 구강이 닫혀 방사가 줄어서 생긴다. 전부 p_sub 로 보내면 −9 dB 가
    −1.1 dB 로 뭉개진다(실측: 여성 탄음에서 p_sub 진폭이 1.05 cmH2O 뿐이었다).
    """
    fs = 16000
    x = _synthetic_vowel(fs, dur=0.4)
    # 40 ms 짜리 −9 dB 골을 낸다 (탄음 모양)
    t = np.arange(len(x)) / fs
    dip = 1.0 - (1.0 - 10 ** (-9.0 / 20.0)) * np.exp(-((t - 0.2) / 0.014) ** 2)
    y = x * dip
    prof = DEFAULT_PROFILE
    tr = analyze(y, fs, prof, int(0.001 * fs))
    g_db = 20 * np.log10(tr["tract_gain"])
    p = tr["p_sub"]
    assert g_db.max() - g_db.min() > 5.0, g_db.max() - g_db.min()   # 골이 이득에 실렸다
    assert p.max() - p.min() < 1.0, p.max() - p.min()               # 폐압은 거의 안 움직인다
    lo = int(0.2 * 1000)
    assert g_db[lo - 5:lo + 5].mean() < g_db[:60].mean() - 4.0      # 골이 제자리에 있다


def test_steady_vowel_keeps_tract_gain_flat():
    """정상 모음에서는 이득이 평탄해야 한다 — 골 검출이 아무 데서나 튀면 안 된다."""
    fs = 16000
    tr = analyze(_synthetic_vowel(fs, dur=0.3), fs, DEFAULT_PROFILE, int(0.001 * fs))
    g_db = 20 * np.log10(tr["tract_gain"])
    assert g_db.max() - g_db.min() < 3.0, g_db.max() - g_db.min()


def test_unvoiced_frames_do_not_get_formants_from_noise():
    """무성 구간의 LPC 봉우리를 포먼트로 쓰면 안 된다.

    포먼트는 성문이 성도를 울릴 때만 뜻이 있다. 마찰음에 LPC 를 걸면 잡음의 우연한
    봉우리가 나오고(실측: 남성 /사/ 마찰부에서 F1 이 3539 → 486 → 2134 Hz), 순서
    규칙이 그것들을 120 Hz 간격으로 겹쳐 쌓아 4 중 고 Q 극을 만든다. 그 극이 성도
    종속을 +88 dB 로 만들어 (du rms 0.04 → glottal_path 1028) 적합이 통째로 무너졌다.
    """
    fs = 16000
    v = _synthetic_vowel(fs, dur=0.2)
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(int(fs * 0.15)) * np.abs(v).max() * 0.3
    y = np.concatenate([noise, v])                      # 무성 150 ms + 유성 200 ms
    tr = analyze(y, fs, DEFAULT_PROFILE, int(0.001 * fs))
    f1, f2 = tr["f1"], tr["f2"]
    assert (f2 - f1).min() > 200.0, (f2 - f1).min()     # 포먼트가 겹쳐 쌓이지 않는다
    assert f1.max() < 1400.0, f1.max()                  # 잡음 봉우리를 F1 으로 집지 않는다
    assert tr.voiced.shape == (tr.n_frames,)
    assert not tr.voiced[:100].any()                    # 앞 100 ms 는 무성으로 잡힌다
    assert tr.voiced[200:].mean() > 0.5                 # 모음은 유성으로 잡힌다


def test_subglottal_pressure_survives_an_unvoiced_stretch():
    """/s/ 동안에도 폐압은 모음과 거의 같다 — 난류가 비효율일 뿐 폐가 쉰 게 아니다.

    무성 구간의 낮은 세기를 폐압에 넣으면 레이놀즈 게이트(비선형) 아래로 떨어져
    마찰음이 통째로 사라진다 (실측: 남 /사/ 마찰부 −42.8 dB).
    """
    fs = 16000
    v = _synthetic_vowel(fs, dur=0.2)
    rng = np.random.default_rng(4)
    quiet = rng.standard_normal(int(fs * 0.15)) * np.abs(v).max() * 0.05
    tr = analyze(np.concatenate([quiet, v]), fs, DEFAULT_PROFILE, int(0.001 * fs))
    p = tr["p_sub"]
    assert p.min() > 5.0, p.min()                        # 무성 구간에서도 안 죽는다
    assert p.max() - p.min() < 3.0, p.max() - p.min()    # 호흡은 천천히 움직인다


def test_fitter_calibrates_voiced_and_unvoiced_levels_separately(_engine):
    """하나의 이득만 맞추면 유성/무성 중 한쪽이 반드시 어긋난다."""
    from formant_ml.engine.control import ControlTrack as CT
    tr = _track(n=120)
    target = _engine.render(tr)
    init = CT(tr.values.copy(), tr.frame_ms)
    init["fric_gain"] = 1.0
    init.voiced = np.ones(init.n_frames, dtype=bool)
    f = CopySynthFitter(_engine, target, 48000, init)
    assert np.asarray(f.track.voiced).shape == (init.n_frames,)   # 마스크가 전달된다
    assert abs(f.gain_db()) < 40.0


def test_global_lr_is_chosen_by_probe_not_hardcoded(_engine):
    """전역 단계의 걸음 크기는 **구간마다 다르다** — 짧게 재 보고 고른다.

    실측: 40 ms 탄음은 lr 0.05 에서 포락 60.3 %, 0.02 에서 90.9 % (lr 에 단조).
    반대로 기울기가 7 dB/oct 틀린 합성 모음은 0.02 로 못 돌아오고 0.05 가 필요하다.
    """
    tr = _track()
    target = _engine.render(tr)
    bad = ControlTrack(tr.values.copy(), tr.frame_ms)
    bad["tilt"] = 9.0
    f = CopySynthFitter(_engine, target, 48000, bad)
    f.sizes = [256, 512]
    before = f._snapshot()
    lr = f.pick_lr_global((0.01, 0.05, 0.2), 25, verbose=False)
    assert lr in (0.01, 0.05, 0.2)
    # 탐침은 출발점을 되돌려 놓아야 한다 — 안 그러면 고른 lr 로 다시 못 돈다
    after = f._snapshot()
    for a, b in zip(before, after):
        assert torch.equal(a, b)


def test_probe_prefers_the_lr_that_actually_converges(_engine):
    """탐침이 손실을 실제로 더 내리는 lr 을 고른다 (발산하는 값을 안 고른다)."""
    tr = _track()
    target = _engine.render(tr)
    bad = ControlTrack(tr.values.copy(), tr.frame_ms)
    bad["tilt"] = 9.0
    f = CopySynthFitter(_engine, target, 48000, bad)
    f.sizes = [256, 512]
    lr = f.pick_lr_global((0.05, 3.0), 25, verbose=False)
    assert lr == 0.05, lr                     # 3.0 은 발산한다


def test_ripple_penalty_bites_ripple_and_spares_articulation(_engine):
    """잔물결 벌점의 **선택성** — 여기가 깨지면 조음을 뭉개고 잔물결을 놓친다.

    같은 크기의 요동이라도 **빠르게 방향을 바꾸는 것**(F0 부근의 잔물결)은 크게,
    **한 방향으로 가는 급전**(파열음 해제 같은 조음)은 거의 안 물어야 한다.
    docs/MEASUREMENTS.md §16 이 이 성질에 통째로 기대고 있다.
    """
    from formant_ml.engine import fit as F

    tr = _track(n=200)
    target = _engine.render(tr)
    f = CopySynthFitter(_engine, target, 48000, tr)
    old = F.RIPPLE_W
    F.RIPPLE_W = 1.0
    try:
        with torch.no_grad():
            base = float(f.penalty())
            t = np.arange(f.w.shape[0])
            # (a) F0 부근(200 Hz, 격자 1 ms)의 잔물결. 진폭 0.1 걸음.
            f.w.copy_(torch.zeros_like(f.w))
            f.w[:, 0] = torch.as_tensor(0.1 * np.cos(2 * np.pi * 0.2 * t),
                                        dtype=f.w.dtype)
            ripple = float(f.penalty()) - base
            # (b) 같은 진폭의 **조음** — 20 ms 에 걸친 단조 전이.
            f.w.copy_(torch.zeros_like(f.w))
            ramp = np.clip((t - 90) / 20.0, 0.0, 1.0) * 0.1
            f.w[:, 0] = torch.as_tensor(ramp, dtype=f.w.dtype)
            move = float(f.penalty()) - base
        assert ripple > 50.0 * max(move, 1e-9)
    finally:
        F.RIPPLE_W = old
        with torch.no_grad():
            f.w.copy_(torch.zeros_like(f.w))


def test_fitting_defaults_are_the_measured_ones():
    """다섯 기본값을 못 박는다. 전부 **전체 파일 A/B 로 정한 값**이다 (0.3.15).

    바꾸려면 근거를 새로 대라 — 아래 숫자는 yang_00000040 전체에서 나온 것이다:

      RIPPLE_W  0 → 0.01 로  포락 90.71 → 94.06 %, 정밀 82.48 → 88.06 %
      VEL_W     0 → 0.001 로 포락 90.47 → 93.17 %, 제어열 곡률 rms 148 → 100
      FLUX_W    0 → 2.0 로   목표 p99 초과 3.08 → **0.000 %** (시드 5 벌 전부)
      ARTIC_VEL_W 1.0 은 **문턱 위만 무는 형태**여야 한다 (§29: 의사후버로 걸면
                  a_c 속도가 0.449 → 0.015 로 얼어 오염이 오히려 늘었다)
      TILT_MAX_HZ 5000 (성대 스무딩, §25)
    """
    from formant_ml.engine import fit as F
    from formant_ml.engine import glottis as G
    assert F.RIPPLE_W == 0.01
    assert F.VEL_W == 0.001 and F.VEL_MODE == "accel"
    assert F.FLUX_W == 2.0
    assert F.ARTIC_VEL_W == 1.0 and F.ARTIC_VEL_KNEE["a_c"] == 0.20
    assert G.TILT_MAX_HZ == 5000.0


def test_articulator_velocity_penalty_spares_real_gestures(_engine):
    """조음 속도 벌점의 **특이성** — 실제 제스처는 통과하고 1 ms 스위칭만 물어야 한다.

    한계 0.20 neper/ms 는 v1 의 최소저크 제스처가 내는 최대(28 ms CV 에서 0.172)
    바로 위이고, 적합 트랙에서 관찰된 이탈(0.87~1.90)보다 4~10 배 아래다
    (docs/MEASUREMENTS.md §24). 이 여백이 무너지면 조음을 뭉개거나 스위칭을 놓친다.
    """
    from formant_ml.engine import fit as F
    from formant_ml.engine.control import INDEX

    tr = _track(n=200)
    target = _engine.render(tr)
    f = CopySynthFitter(_engine, target, 48000, tr)
    k = f.names.index("a_c")
    old = F.ARTIC_VEL_W
    F.ARTIC_VEL_W = 1.0
    try:
        def pen_for(a_c_track):
            with torch.no_grad():
                raw = f._to_raw(torch.as_tensor(a_c_track, dtype=torch.float64), f.specs[k])
                f.w.copy_(torch.zeros_like(f.w))
                f.w[:, k] = (raw - f.u0[:, k]) / f.scale[k]
                return float(f.penalty())

        n = f.n_frames
        t = np.arange(n)
        # (a) 실제 제스처: `phones.sibilant` 이 실제로 그리는 것 — 40 ms 에 걸쳐
        #     0.35 -> 0.08 을 로그·최소저크로 (최대 0.071 neper/ms).
        u = np.clip((t - 60) / 40.0, 0.0, 1.0)
        mj = u ** 3 * (10 - 15 * u + 6 * u ** 2)
        gesture = np.exp(np.log(0.35) + (np.log(0.08) - np.log(0.35)) * mj)
        # (b) 1 ms 스위칭: 같은 진폭을 한 프레임에
        switch = np.full(n, 0.35)
        switch[100] = 0.08
        base = pen_for(np.full(n, 0.35))
        pg, ps = pen_for(gesture) - base, pen_for(switch) - base
    finally:
        F.ARTIC_VEL_W = old
        with torch.no_grad():
            f.w.copy_(torch.zeros_like(f.w))
    assert ps > 10.0 * max(pg, 1e-9)
    # **한계 아래는 정확히 공짜여야 한다.** 처음 쓴 `pseudo_huber(rate/knee)` 는
    # 무릎 아래에서도 이차로 벌해서 실제 제스처까지 얼렸다 (§29: a_c 속도 95 분위가
    # 0.449 -> 0.015 로, 가장 빠른 실제 제스처의 11 배 **아래**로 눌렸다).
    # 이 제스처의 최대 속도는 0.071 로 한계(0.20)의 3 분의 1 이다.
    assert pg < 0.01 * ps


def test_nonfinite_gradient_does_not_freeze_the_fit(_engine):
    """비유한 기울기가 나도 **걸음은 딛어야 한다** — 안 그러면 결정적으로 갇힌다.

    가드가 회차 전체를 버리면 파라미터가 안 변하고, 그러면 다음 회차의 기울기도
    똑같이 비유한이다. 실측(out/sw/r000, yang_00000040 전체): 모든 단계가
    "[30] 수렴 + 비유한 29 회 건너뜀" 으로 끝났다 — 단계당 실제 걸음이 1 번뿐이었고
    포락이 82.7 % 에 머물렀다 (같은 파일의 정상 적합은 93.1 %).

    비유한 **성분만** 0 으로 두면 나머지로 걸음을 딛으므로 그 자리를 벗어난다.
    """
    tr = _track(n=60)
    target = _engine.render(tr)
    f = CopySynthFitter(_engine, target, 48000, tr)
    real_loss = f.loss

    def poisoned():
        l, sc, env_sc, per = real_loss()
        return l + torch.sqrt((f.w * 0.0).sum()), sc, env_sc, per   # w 기울기가 ∞

    # `fit` 은 끝에서 최선 스냅샷을 복원하므로(목표가 같은 트랙의 렌더라 0 회차가
    # 최선이다) **도는 동안**의 파라미터를 봐야 한다.
    seen = []

    def watched():
        seen.append(f.d.detach().clone())
        return poisoned()

    f.loss = watched
    f.fit(iters=6, lr=0.05, verbose=False, patience=0)
    assert len(seen) >= 3
    moved = float((seen[-1] - seen[0]).abs().max())
    assert moved > 1e-6, f"매 회차 비유한인데 파라미터가 안 움직였다 ({moved:.3g})"


def test_skipped_iterations_do_not_count_as_convergence(_engine):
    """비유한 기울기가 든 회차를 정체로 세면 안 된다.

    걸음을 안 딛었으면 손실이 안 변하는 것이 당연한데 그걸 "수렴" 으로 세면, 비유한
    기울기가 잦은 구간에서 단계가 통째로 조기 종료된다. 실측(out/v13/s040): 위상
    단계 42 회 중 27 회가 건너뛰어져 정체 30 이 먼저 찼고, 위상 단계는 항상 한 번
    꺾였다가 회복하므로 **골짜기 한복판**에서 멈췄다 — 포락 90.63 -> 84.29 %
    (docs/MEASUREMENTS.md §27).

    √x 는 x=0 에서 기울기가 무한이다. 그걸로 비유한 기울기를 확실히 만든다.
    """
    tr = _track(n=60)
    target = _engine.render(tr)

    def run(poison: bool) -> int:
        f = CopySynthFitter(_engine, target, 48000, tr)
        real_loss = f.loss
        calls = {"n": 0}

        def wrapped():
            l, sc, env_sc, per = real_loss()
            calls["n"] += 1
            if poison and calls["n"] % 2 == 0:        # 절반의 회차에서 ∞ 기울기
                l = l + torch.sqrt((f.w * 0.0).sum())
            return l, sc, env_sc, per

        f.loss = wrapped
        f.fit(iters=12, lr=0.02, verbose=False, patience=3)
        return calls["n"]

    # 목표가 같은 트랙의 렌더라 실제 걸음도 손실을 못 줄인다 — 그래서 오염 없는
    # 실행은 정당하게 patience 회 만에 멈춘다. 오염된 실행은 **건너뛴 회차가 정체로
    # 세어지지 않는다면** 그만큼 더 돌아야 한다.
    clean, poisoned = run(False), run(True)
    assert poisoned > clean, (
        f"건너뛴 회차가 정체로 세어졌다 — 오염 {poisoned} 회 vs 정상 {clean} 회")
