"""난류를 통계로 재는 자 — 그리고 파형으로 재면 왜 안 되는지의 증거.

이 파일의 절반은 **결함의 회귀 테스트**다. 두 가지를 못 들어오게 막는다:

1. 난류를 프레임별 크기 스펙트럼으로 채점하는 것 (`test_realization_floor`).
   완벽한 모형이라도 34 % 가 상한이므로 그 자로는 "95 %" 를 말할 수 없다.
2. 위상 항이 마찰 진폭을 0 으로 미는 것 (`test_phase_distance_is_blind_to_level`).
   이건 실제로 있던 버그다 — 손실이 800 회 내내 소리를 끄는 쪽으로 기울기를 줬다.
"""
import numpy as np
import pytest
import torch
from scipy.signal import butter, sosfilt

from formant_ml.engine import turbulence as tb
from formant_ml.engine.fit import _stft

FS = 48000.0


def _band(seed, lo, hi, dur=0.25, fs=FS):
    r = np.random.default_rng(seed)
    sos = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
    x = sosfilt(sos, r.standard_normal(int(fs * dur)))
    return x / (x.std() + 1e-12)


def _tone(f0=200.0, dur=0.25, fs=FS):
    t = np.arange(int(fs * dur)) / fs
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 40))
    return x / x.std()


# ------------------------------------------------------- 자 자체의 상한
def test_realization_floor():
    """**같은 스펙트럼의 두 독립 실현이 원래 자로는 34 % 다.**

    이 값이 곧 치찰음에서 도달 가능한 상한이다. 여기가 흔들리면 `turbulence` 머리말과
    `docs/MEASUREMENTS.md` 의 표를 같이 고쳐야 한다.
    """
    a, b = _band(1, 4000, 11000), _band(2, 4000, 11000)
    r = tb.spectral_fidelity(a, b, None, FS)
    assert 25.0 < r["fine"] < 42.0          # 이론 34.5 %
    # 백색 잡음도 같다 — 스펙트럼 모양의 문제가 아니라 실현의 문제다.
    w1 = np.random.default_rng(3).standard_normal(int(FS * 0.25))
    w2 = np.random.default_rng(4).standard_normal(int(FS * 0.25))
    assert 25.0 < tb.spectral_fidelity(w1, w2, None, FS)["fine"] < 42.0


def test_corrected_sc_saturates_for_a_perfect_model():
    """완벽한 모형(= 시드만 다름)이면 보정 일치율이 95 % 를 넘는다."""
    t = _band(1, 4000, 11000)
    p, p2 = _band(2, 4000, 11000), _band(3, 4000, 11000)
    r = tb.spectral_fidelity(t, p, p2, FS)
    assert r["fine"] < 45.0                 # 원래 자는 여전히 낮다
    assert r["fine_corr"] > 95.0            # 보정하면 거의 만점
    # 편향이 실제로 0 이므로 **분해되지 않는다** — 이건 실패가 아니라 이 자의 한계다.
    # 잡음량이 맞는다는 것은 noise_ratio 가 따로 말해 준다.
    assert not r["resolved"]
    assert 0.8 < r["noise_ratio"] < 1.25


def test_noise_ratio_catches_a_level_error_the_correction_cannot():
    """분해가 안 될 때 **좋은 합성과 잡음 과다를 가르는 것은 `noise_ratio` 다.**

    보정 일치율만 보면 둘 다 높게 나올 수 있다. 잡음량은 물리 파라미터이므로 별개로
    보고해야 한다.
    """
    t = _band(1, 4000, 11000)
    good = tb.spectral_fidelity(t, _band(2, 4000, 11000), _band(3, 4000, 11000), FS)
    quiet = tb.spectral_fidelity(t, 0.5 * _band(2, 4000, 11000),
                                 0.5 * _band(3, 4000, 11000), FS)
    assert 0.8 < good["noise_ratio"] < 1.25
    # 진폭 0.5 배 -> 분산비 0.25. 추정이 그 근처를 짚어야 한다.
    assert 0.15 < quiet["noise_ratio"] < 0.40


def test_corrected_sc_still_sees_a_real_error():
    """보정이 '모든 걸 100 % 로 만드는' 자가 아님을 확인한다.

    판별력이 오히려 넓어져야 한다 — 원래 자는 완벽과 어긋남을 몇 %p 로만 갈랐다.
    """
    t = _band(1, 4000, 11000)
    ok = tb.spectral_fidelity(t, _band(2, 4000, 11000), _band(3, 4000, 11000), FS)
    off = tb.spectral_fidelity(t, _band(2, 5000, 12000), _band(3, 5000, 12000), FS)
    bad = tb.spectral_fidelity(t, _band(2, 2000, 6000), _band(3, 2000, 6000), FS)
    assert ok["fine_corr"] > off["fine_corr"] > bad["fine_corr"]
    # 판별 여유가 원래 자보다 넓다
    assert (ok["fine_corr"] - off["fine_corr"]) > 3.0 * (ok["fine"] - off["fine"])


def test_correction_is_harmless_on_deterministic_signals():
    """하모닉에서는 보정이 값을 바꾸지 않아야 한다 (D_pp' ≈ 0)."""
    t = _tone(200.0)
    same = tb.spectral_fidelity(t, _tone(200.0), _tone(200.0), FS)
    assert same["fine"] > 99.0 and same["fine_corr"] > 99.0
    assert same["resolved"]                  # 실현 잡음이 없으니 완전히 분해된다
    off = tb.spectral_fidelity(t, _tone(206.0), _tone(206.0), FS)
    assert abs(off["fine"] - off["fine_corr"]) < 1.0     # 보정이 개입하지 않는다
    assert off["resolved"] and off["trust"] > 0.9


def test_level_error_is_not_forgiven():
    """레벨이 틀린 것은 실현 잡음이 아니다 — 보정해도 낮아야 한다."""
    t = _band(1, 4000, 11000)
    r = tb.spectral_fidelity(t, 0.5 * _band(2, 4000, 11000),
                             0.5 * _band(3, 4000, 11000), FS)
    assert r["fine_corr"] < 70.0
    assert r["resolved"]                     # 레벨 오차는 잡음보다 커서 분해된다
    # 반대로 모양만 보는 자는 레벨을 무시한다 (둘이 다른 것을 잰다)
    assert tb.spectrum_match(t, 0.5 * _band(2, 4000, 11000), FS) > 85.0


# --------------------------------------------------------- 치찰음 특징
def test_spectral_moments_separate_s_from_sh():
    """무게중심이 /s/ 와 /ʃ/ 를 가른다 (Jongman et al. 2000)."""
    f, p = tb.average_psd(_band(1, 5000, 12000), FS)     # /s/ 쪽
    g, q = tb.average_psd(_band(1, 2000, 5000), FS)      # /ʃ/ 쪽
    ms, msh = tb.spectral_moments(f, p), tb.spectral_moments(g, q)
    assert ms["centroid"] > msh["centroid"] + 2000.0
    assert ms["skew"] < msh["skew"]                      # /s/ 는 덜 오른쪽꼬리다
    assert tb.sibilance_db(f, p) > tb.sibilance_db(g, q)


def test_spectral_peak_is_stable_across_realizations():
    """봉우리는 평활 덕에 실현이 바뀌어도 흔들리지 않아야 한다."""
    pk = []
    for s in (1, 2, 3, 4):
        f, p = tb.average_psd(_band(s, 6000, 9000), FS)
        pk.append(tb.spectral_peak(f, p))
    assert np.std(pk) < 800.0


def test_multitaper_has_lower_variance_than_one_window():
    """멀티테이퍼가 단일 창보다 스펙트럼 추정 분산이 낮다 (Reidy 2015 의 권고)."""
    x = _band(1, 4000, 11000, dur=0.05)
    f, p = tb.multitaper_psd(x, FS)
    n = len(x)
    w = np.hanning(n)
    q = np.abs(np.fft.rfft(x * w)) ** 2
    band = (f >= 5000) & (f <= 10000)
    # 대역 안에서 평탄해야 하므로, 변동계수가 작을수록 좋은 추정이다
    cv_mt = p[band].std() / p[band].mean()
    cv_1 = q[band].std() / q[band].mean()
    assert cv_mt < 0.6 * cv_1


def test_gaussian_noise_has_kurtosis_near_three():
    """대조군 — 이 값 없이 '지글거린다' 를 말하면 안 된다."""
    assert 2.7 < tb.amplitude_kurtosis(_band(1, 4000, 11000), FS) < 3.4


def test_modulation_bands_sum_to_100():
    m = tb.modulation_bands(_band(1, 4000, 11000), FS)
    assert np.isfinite(m).all()
    assert abs(m.sum() - 100.0) < 1e-6


# ------------------------------------------ 위상 항의 회귀 테스트 (실제 버그)
def _phase_terms(target, synth):
    """예전(복소 잔차)과 새것(단위 크기)의 위상 거리를 나란히 낸다."""
    T = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
    P = torch.tensor(synth, dtype=torch.float32).unsqueeze(0)
    old = new = 0.0
    for k in (256, 512, 1024):
        w = torch.hann_window(k)
        St, Sp = _stft(T, k, w), _stft(P, k, w)
        ref = St.abs().mean()
        d = St - Sp
        old = old + float((torch.sqrt(d.real ** 2 + d.imag ** 2 + 1e-12)).mean() / (ref + 1e-9))
        at = St.abs()
        dt = 1e-4 * at.mean() + 1e-12
        u = St / torch.sqrt(at ** 2 + dt ** 2) - Sp / torch.sqrt(Sp.abs() ** 2 + dt ** 2)
        e = torch.sqrt(u.real ** 2 + u.imag ** 2 + 1e-6) - 1e-3
        new = new + float((at * e).sum() / (at.sum() + 1e-9))
    return old / 3.0, new / 3.0


def test_old_phase_term_was_minimised_by_silence():
    """**있던 버그의 증거.** 위상이 무상관이면 복소 잔차는 진폭 0 에서 최소다.

    이 테스트는 고쳐진 것을 확인하는 게 아니라, 왜 고쳐야 했는지를 남긴다.
    """
    t, p = _band(1, 4000, 11000), _band(2, 4000, 11000)
    vals = [_phase_terms(t, g * p)[0] for g in (0.0, 0.5, 1.0)]
    assert vals[0] < vals[1] < vals[2]        # 끌수록 좋다고 말한다


def test_phase_distance_is_blind_to_level():
    """새 위상 거리는 합성 **진폭에 무관**해야 한다 — 소리를 끌 이유가 없다."""
    t, p = _band(1, 4000, 11000), _band(2, 4000, 11000)
    v = [_phase_terms(t, g * p)[1] for g in (0.25, 0.5, 1.0, 2.0)]
    assert max(v) - min(v) < 0.02 * np.mean(v)


def test_phase_distance_still_finds_a_real_phase_error():
    """진폭에 무관하되 위상 오차에는 반응해야 한다 — 안 그러면 쓸모가 없다."""
    t = _tone(200.0)
    same, _ = _phase_terms(t, t)[1], None
    shifted = _phase_terms(t, np.roll(t, 37))[1]
    assert same < 0.05
    assert shifted > 10.0 * max(same, 1e-6)


@pytest.mark.parametrize("g", [0.5, 1.0, 2.0])
def test_phase_distance_of_identical_signal_is_zero_at_any_level(g):
    """같은 파형이면 진폭이 달라도 위상 거리는 0 이다 (크기와 완전히 직교)."""
    t = _tone(200.0)
    assert _phase_terms(t, g * t)[1] < 0.05


# ------------------------------------------------- 손실 압축의 대역 상한
def test_effective_bandwidth_leaves_full_band_alone():
    """대역 제한이 없으면 자르지 않는다 — 확실하지 않으면 건드리지 않는 편이 안전하다."""
    x = np.random.default_rng(0).standard_normal(int(FS * 0.4))
    assert tb.effective_bandwidth(x, FS) >= 0.49 * FS


def test_effective_bandwidth_is_conservative_on_an_analog_lowpass():
    """아날로그 저역통과는 **자르지 않거나 보수적으로만** 잡는다.

    이 도구의 목적은 손실 압축의 브릭월이다. 완만한 전이대역을 컷으로 읽으면 진짜
    신호를 버리므로, 애매하면 안 자르는 쪽이 옳다.
    """
    from scipy.signal import butter, sosfilt
    x = np.random.default_rng(0).standard_normal(int(FS * 0.4))
    sos = butter(8, 12000 / (FS / 2), btype="low", output="sos")
    assert tb.effective_bandwidth(sosfilt(sos, x), FS) >= 12000.0


def test_effective_bandwidth_never_returns_less_than_the_real_band():
    """**틀리더라도 높은 쪽으로 틀려야 한다.** 낮게 자르면 진짜 신호를 버린다.

    이 함수는 진단용이고 적합 손실의 상한으로 쓰지 않는다 (`CopySynthFitter` 참조) —
    합성 신호의 자연 롤오프를 컷으로 오인한 전력이 있기 때문이다. 그래서 여기서
    지키는 것은 정확도가 아니라 **한쪽으로만 틀리는 성질**이다.
    """
    n = int(FS * 0.4)
    f = np.fft.rfftfreq(n, 1 / FS)
    for cut in (12000.0, 16000.0, 20000.0):
        X = np.fft.rfft(np.random.default_rng(1).standard_normal(n))
        X[f > cut] = 0.0
        assert tb.effective_bandwidth(np.fft.irfft(X, n), FS) >= cut * 0.95


def test_effective_bandwidth_ignores_a_natural_rolloff():
    """**자연스러운 고역 롤오프를 컷으로 오인하면 안 된다.**

    음성은 소스 기울기(−12 dB/oct) 때문에 고역이 원래 완만히 떨어진다. 레벨만 보고
    자르면 진짜 신호를 버린다 — 실측으로 합성 음성에서 9.96 kHz 를 상한이라고 답한
    적이 있다 (`test_loss_ignores_bands_above_the_recording_nyquist` 가 깨졌다).
    """
    n = int(FS * 0.4)
    f = np.fft.rfftfreq(n, 1 / FS)
    X = np.fft.rfft(np.random.default_rng(2).standard_normal(n))
    X = X * (1.0 / (1.0 + (f / 2000.0) ** 2))       # −12 dB/oct 에 가까운 기울기
    assert tb.effective_bandwidth(np.fft.irfft(X, n), FS) >= 0.49 * FS
