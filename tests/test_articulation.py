"""조음 제약 검증 — 혀가 낼 수 없는 움직임이 표현/학습되지 않는가.

    PYTHONPATH=src python3 tests/test_articulation.py
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from formant_ml.config import Config
from formant_ml.dsp.core import freq_grid
from formant_ml.dsp.gestures_dynamics import (RISE_TIME, apply_dynamics,
                                              apply_to_controls, formant_motion_rank,
                                              gesture_kernel, rise_to_omega)
from formant_ml.models.encoder import ControlEncoder
from formant_ml.models.losses import articulatory_rate_loss, formant_subspace_loss
from formant_ml.models.synth import Controls, PhysicalVoiceSynth
from formant_ml.presets import LOCUS
from formant_ml.score import build_controls
from formant_ml.voice import VoiceProfile

FR = 100.0          # 프레임률 [Hz]


def _step(t=200, lo=1000.0, hi=2000.0):
    x = torch.full((1, t, 1), lo)
    x[:, t // 2:] = hi
    return x


def test_gesture_kernel_is_unit_gain():
    """정상상태에서 목표값을 정확히 따라간다(직류 이득 1)."""
    for tr in (0.03, 0.06, 0.1):
        h = gesture_kernel(tr, FR)
        assert abs(float(h.sum()) - 1.0) < 1e-4


def test_step_response_rise_time_matches_spec():
    """설정한 10~90% 상승시간대로 움직인다 (프레임 양자화 ±1 프레임)."""
    for tr in (0.035, 0.060, 0.080):
        y = apply_dynamics(_step(), tr, FR)[0, :, 0]
        seg = y[100:] - 1000.0
        i10 = int((seg > 100).nonzero()[0])
        i90 = int((seg > 900).nonzero()[0])
        measured = (i90 - i10) / FR
        assert abs(measured - tr) <= 1.5 / FR, (tr, measured)


def test_peak_slew_matches_theory():
    """계단 dF 에 대한 최대 속도 = dF·w/e. 문헌의 F2 전이(10~25 Hz/ms) 범위."""
    tr = 0.060
    y = apply_dynamics(_step(lo=0.0, hi=1000.0), tr, FR)[0, :, 0]
    peak = float((y[1:] - y[:-1]).abs().max() * FR)
    theory = 1000.0 * rise_to_omega(tr) / 2.718281828
    assert abs(peak - theory) / theory < 0.10
    assert 10_000 < peak < 25_000        # Hz/s = 10~25 Hz/ms


def test_dynamics_are_causal():
    """미래의 목표가 현재를 바꾸지 않는다(뒤에서 당겨오는 평활이 아니다)."""
    a = _step()
    b = a.clone()
    b[:, 150:] = 5000.0                     # 150 프레임 이후만 다르게
    ya = apply_dynamics(a, 0.06, FR)
    yb = apply_dynamics(b, 0.06, FR)
    assert (ya[:, :150] - yb[:, :150]).abs().max() < 1e-4


def test_rate_loss_penalizes_only_impossible_speed():
    """상한 아래의 빠른 전이는 벌하지 않고, 프레임 단위 난동만 벌한다."""
    base = torch.tensor([730., 1090, 2440, 3400, 4500, 5500])
    fast = base.repeat(1, 200, 1).clone()
    fast[:, 100:, :2] = torch.tensor([300., 2290.])
    ok = apply_to_controls({"formant_freq": fast}, FR)["formant_freq"]
    wild = base.repeat(1, 200, 1) + torch.randn(1, 200, 6) * 120

    assert float(articulatory_rate_loss(ok, FR)) < 1e-4
    assert float(articulatory_rate_loss(wild, FR)) > 0.1


def test_subspace_loss_separates_coupled_from_independent_motion():
    """함께 움직이는 포먼트는 벌점 0, 제각각 움직이면 큰 벌점."""
    t = 200
    z = torch.sin(torch.linspace(0, 6.0, t))[None, :, None]      # 조음 좌표 1개
    basis = torch.tensor([300., -400., 150., 60., 20., 10.])
    coupled = torch.tensor([730., 1090, 2440, 3400, 4500, 5500]) + z * basis
    independent = torch.tensor([730., 1090, 2440, 3400, 4500, 5500]) \
        + torch.randn(1, t, 6) * 100

    assert float(formant_subspace_loss(coupled, 3)) < 1e-3
    assert float(formant_subspace_loss(independent, 3)) > 0.2


def test_encoder_basis_restricts_formant_motion_dimension():
    """저차원 기저를 켜면 포먼트가 저차원에서만 움직인다 (순서 보장은 유지)."""
    torch.manual_seed(0)
    ranks = {}
    for dim in (0, 4):
        cfg = Config()
        cfg.filt.formant_basis_dim = dim
        enc = ControlEncoder(cfg)
        c = enc(torch.randn(1, 120, 80), torch.full((1, 120), 140.),
                torch.ones(1, 120))
        assert bool((c.formant_freq[..., 1:] > c.formant_freq[..., :-1]).all())
        ranks[dim] = formant_motion_rank(c.formant_freq)
    assert ranks[4] <= 4 and ranks[4] < ranks[0], ranks


def test_aspiration_passes_through_the_whole_tract():
    """기식 노이즈는 성도 전체를 통과한다 — 마찰 노이즈와 다른 경로다.

    이게 무성자음과 모음을 하나의 공명기로 묶는 성분이다. 마찰 노이즈는
    협착 하류만 통과하므로 모음의 F1 을 여기시키지 못한다.
    """
    cfg = Config()
    syn = PhysicalVoiceSynth(cfg)
    K, T = cfg.filt.n_formants, 40
    ff = torch.tensor([730., 1090, 2440, 3400, 4500, 5500, 6500, 7500,
                       8500, 9500, 10500, 11300])[:K]
    kw = dict(
        f0=torch.full((1, T, 1), 120.), harmonic_amp=torch.zeros(1, T, 1),
        rd=torch.full((1, T, 1), 1.2),
        formant_freq=ff.reshape(1, 1, -1).expand(1, T, K).contiguous(),
        formant_bw=torch.full((1, T, K), 90.), formant_gain=torch.ones(1, T, K),
        noise_bands=torch.full((1, T, cfg.noise.n_bands), 1e-6),
        noise_entry=torch.full((1, T, 1), float(K) + 6.0),
        noise_am=torch.zeros(1, T, 1))
    with torch.no_grad():
        out = syn(Controls(**kw, aspiration=torch.full((1, T, 1), 0.1)))
    y = out["aspirated"][0]
    Y = torch.fft.rfft(y * torch.hann_window(len(y))).abs()
    f = freq_grid(len(Y), cfg.audio.sample_rate)
    # F1(730 Hz) 부근 에너지가 그 사이 골(1900 Hz 부근)보다 확실히 커야 한다
    near = Y[(f > 600) & (f < 900)].mean()
    valley = Y[(f > 1700) & (f < 2100)].mean()
    assert float(near / valley.clamp_min(1e-9)) > 2.0


def test_syllable_starts_at_the_consonant_locus():
    """/사/ 의 포먼트가 자음 조음 위치에서 출발해 모음으로 미끄러진다."""
    score = {"timeline": [{"type": "syllable", "onset": "s", "vowel": "a",
                           "dur": 0.6}]}
    c = build_controls(score, VoiceProfile(), Config())
    f2 = c.formant_freq[0, :, 1]
    start, end = float(f2[5]), float(f2[-1])
    assert abs(start - LOCUS["s"][1]) < 150.0, start        # 치경 locus ~1750
    assert abs(end - 1090.0) < 120.0, end                   # /a/ 의 F2
    # 전이가 생리적 시간(30~150 ms) 안에 일어난다
    lo, hi = min(start, end), max(start, end)
    moving = ((f2 > lo + 0.1 * (hi - lo)) & (f2 < lo + 0.9 * (hi - lo))).sum()
    assert 3 <= int(moving) <= 15, int(moving)


def test_syllable_has_aspiration_at_the_release():
    """마찰이 끝나고 유성이 시작되는 사이에 기식 성분이 존재한다."""
    score = {"timeline": [{"type": "syllable", "onset": "s", "vowel": "a",
                           "dur": 0.6}]}
    c = build_controls(score, VoiceProfile(), Config())
    asp = c.aspiration[0, :, 0]
    amp = c.harmonic_amp[0, :, 0]
    peak = int(asp.argmax())
    assert float(asp.max()) > 0.02
    # 기식의 정점이 성대 진동이 완전히 켜지기 전에 온다
    voiced_on = int((amp > 0.5).float().argmax())
    assert peak <= voiced_on + 3, (peak, voiced_on)


def test_nasal_is_identity_when_velum_is_closed():
    """연구개가 닫히면 비강 극-영점이 정확히 상쇄되어 아무 일도 일어나지 않는다."""
    from formant_ml.dsp.nasal import nasal_response
    H = nasal_response(torch.zeros(1, 3, 1), 24000, 513)
    assert float((H.abs() - 1.0).abs().max()) < 1e-5
    assert float(torch.angle(H).abs().max()) < 1e-5


def test_nasal_adds_pole_zero_and_damps_f1():
    """연구개를 열면 (1) 비강 극, (2) 구강 영점, (3) F1 감쇠가 함께 나타난다."""
    from formant_ml.dsp.nasal import f1_bandwidth_factor, nasal_response
    cfg = Config()
    syn = PhysicalVoiceSynth(cfg)
    K, T = cfg.filt.n_formants, 20
    ff = torch.linspace(730, 11300, K)
    kw = dict(f0=torch.full((1, T, 1), 120.), harmonic_amp=torch.ones(1, T, 1),
              rd=torch.full((1, T, 1), 1.2),
              formant_freq=ff.reshape(1, 1, -1).expand(1, T, K).contiguous(),
              formant_bw=torch.full((1, T, K), 90.),
              formant_gain=torch.ones(1, T, K),
              noise_bands=torch.full((1, T, cfg.noise.n_bands), 1e-6),
              noise_entry=torch.zeros(1, T, 1), noise_am=torch.zeros(1, T, 1))
    with torch.no_grad():
        closed = syn(Controls(**kw))["h_harm"]
        open_ = syn(Controls(**kw, velum_open=torch.ones(1, T, 1)))["h_harm"]
    f = freq_grid(closed.shape[-1], cfg.audio.sample_rate)
    # F1 피크가 낮아진다(대역폭이 넓어졌으므로)
    band = (f > 650) & (f < 820)
    assert float(open_.abs()[0, 0][band].max()) < float(closed.abs()[0, 0][band].max())
    # 380~600 Hz 에 없던 스펙트럼 골(비강 영점)이 생긴다
    notch = (f > 380) & (f < 600)
    ratio = float(open_.abs()[0, 0][notch].min() / closed.abs()[0, 0][notch].min())
    assert ratio < 0.5, ratio          # 6 dB 이상 파인다
    assert float(f1_bandwidth_factor(torch.ones(1, 1, 1))) > 1.0


def test_phonation_threshold_pressure_exists():
    """역치압 아래에서는 성대가 아예 진동하지 않는다 (크로스페이드가 아니다)."""
    from formant_ml.dsp.vocalfold import (FoldParams, phonation_threshold,
                                          pressure_sweep)
    p = FoldParams(a01=0.02, a02=0.02)
    ptp = phonation_threshold(p, n_samples=7200)
    assert 0.5 < ptp < 6.0, ptp                     # 생리적 범위 [cmH2O]
    below, above = pressure_sweep(p, (ptp * 0.5, ptp * 3.0), n_samples=7200)
    assert not below["oscillating"] and above["oscillating"]
    assert above["rms"] > below["rms"] * 5


def test_pressure_drives_source_parameters_together():
    """압력 하나가 amp/F0/Rd 를 함께 움직인다 (세 개의 독립 손잡이가 아니다)."""
    from formant_ml.dsp.vocalfold import pressure_to_source
    ps = torch.tensor([1.0, 4.0, 8.0, 16.0])
    r = pressure_to_source(ps, ptp_cm=2.0)
    assert float(r["amp"][0]) == 0.0                      # 역치 아래 = 무성
    assert torch.all(r["amp"][1:].diff() > 0)             # 단조 증가
    assert torch.all(r["f0_shift"][1:].diff() > 0)        # 압력 -> F0 상승
    assert torch.all(r["rd_shift"][1:].diff() < 0)        # 압력 -> pressed
    # 압력 2배당 약 7 dB
    db = 20 * torch.log10(r["amp"][2] / r["amp"][1])
    assert 5.0 < float(db) < 9.0, float(db)


# ---------------------------------------------------------------- 여성 성도 / 곁가지
def test_formant_count_follows_tract_length():
    """성도가 짧으면 나이퀴스트 아래 극의 개수가 준다. 억지로 채우면 응답이 접힌다."""
    from formant_ml.voice import VoiceProfile
    m, f = VoiceProfile(), VoiceProfile.female()
    assert m.n_formants() == 12 and f.n_formants() == 9, (m.n_formants(), f.n_formants())
    assert max(f.formants) < 12000.0
    assert f.n_tract_sections() == 19        # 14.1 cm @ 24 kHz


def test_female_render_is_not_high_frequency_blown_up():
    """여성 프로파일 합성의 스펙트럼 무게중심이 정상 범위여야 한다.

    (포먼트 12 개를 짧은 성도에 억지로 끼우면 나이퀴스트 위에 극이 생겨
     무게중심이 1.3 kHz 에서 10.5 kHz 로 튀었다 — 실제로 겪은 버그.)
    """
    from formant_ml.config import Config as C
    from formant_ml.score import render
    from formant_ml.voice import VoiceProfile
    sc = {"timeline": [{"type": "vowel", "vowel": "a", "dur": 0.5, "f0": 200}],
          "seed": 5}
    y = render(sc, VoiceProfile.female(), C())[0]
    Y = torch.fft.rfft(y * torch.hann_window(len(y))).abs()
    fq = torch.linspace(0, 12000, len(Y))
    centroid = float((fq * Y).sum() / Y.sum())
    assert 400 < centroid < 3000, centroid


def test_side_cavity_notch_is_local():
    """곁가지 노치는 홈만 파고 그 밖에서는 응답이 1 로 돌아와야 한다.

    반공명만 쓰면 영점 위에서 이득이 계속 커져 나이퀴스트에서 +7 dB 가 되고,
    /s/ 의 스펙트럼 피크를 나이퀴스트로 옮겨 버린다.
    """
    from formant_ml.dsp.filters import notch_response
    H = notch_response(torch.full((1, 1, 1), 4500.), torch.full((1, 1, 1), 600.),
                       24000, 1025)
    m = H.abs()[0, 0]
    fq = freq_grid(len(m), 24000)
    notch = float(m[(fq > 4200) & (fq < 4800)].min())
    dc = float(m[fq < 200].mean())
    nyq = float(m[fq > 11000].mean())
    assert notch < 0.6, notch
    assert abs(20 * math.log10(dc)) < 2.0 and abs(20 * math.log10(nyq)) < 2.0


def test_glottal_opening_widens_f1():
    """소스-성도 상호작용 1차: 개방지수가 클수록 F1 대역폭이 넓어진다."""
    from formant_ml.dsp.glottal import glottal_f1_damping, open_quotient
    assert open_quotient(0.4) < open_quotient(1.2) < open_quotient(2.4)
    d = glottal_f1_damping(torch.tensor([[[0.4]], [[1.2]], [[2.4]]]))
    assert float(d[0]) < float(d[1]) < float(d[2])
    assert 30.0 < float(d[1]) < 130.0        # 문헌의 주기평균 50~100 Hz 범위


def test_fricative_noise_has_gaussian_statistics():
    """마찰음은 **가우시안 잡음의 통계**를 가져야 한다 (첨도 3, crest ~4.3).

    실제 난류가 가우시안이다. 포락선을 평평하게 만들면 통계가 사인파 쪽
    (첨도 1.5, crest 1.41)으로 이동하고, 그러면 잡음이 아니라 20~30 Hz 로
    맥동하는 '기계음'처럼 들린다 — 실제로 겪은 실패다(첨도 1.61, crest 1.85).

    스펙트럼만 보는 지표로는 이 실패를 절대 못 잡는다. 그래서 이 테스트가 있다.
    """
    from formant_ml.config import Config as C
    from formant_ml.score import render
    from formant_ml.voice import VoiceProfile
    for prof in (VoiceProfile(), VoiceProfile.female()):
        y = render({"timeline": [{"type": "fricative", "phone": "s", "dur": 0.5}],
                    "seed": 3}, prof, C())[0]
        x = y[int(0.08 * 24000): int(0.45 * 24000)]
        x = x - x.mean()
        sd = x.std()
        kurt = float((x ** 4).mean() / sd ** 4)
        crest = float(x.abs().max() / sd)
        assert 2.5 < kurt < 3.6, kurt          # 가우시안 3.0
        assert 3.3 < crest < 5.5, crest        # 가우시안 ~4.3


def test_low_noise_noise_is_available_but_off_by_default():
    """평탄화 도구는 남겨 두되 기본은 꺼져 있어야 한다."""
    from formant_ml.config import Config as C
    from formant_ml.dsp.noise import low_noise_noise
    from formant_ml.models.synth import PhysicalVoiceSynth
    assert PhysicalVoiceSynth(C()).noise_smoothing == 0.0
    x = low_noise_noise((1, 1 << 14), flatten=1.0)
    e = x[0]
    k = float((e ** 4).mean() / e.std() ** 4)
    assert k < 2.2, k                          # 평탄화하면 사인파 쪽으로 간다


def test_envelope_flattening_preserves_the_spectrum():
    """포락선 평탄화가 치찰음 스펙트럼을 희게 만들면 안 된다.

    (평탄화만 하면 9.2 dB rms 로 희어진다 — 치찰음의 정체가 그 스펙트럼인데.
     넓게 평활한 크기 비로 되돌려 0.4 dB 로 줄인다.)
    """
    from formant_ml.dsp.noise import flatten_fast_envelope
    from formant_ml.dsp.sibilant import SibilantParams, sibilant_response
    n = 1 << 15
    H = sibilant_response(SibilantParams.constant((1, 1, 1), 7000., 900., 3000.,
                                                  900., 0.5, 1.0, 0.), 24000,
                          n // 2 + 1)[0, 0]
    torch.manual_seed(0)
    x = torch.fft.irfft(torch.fft.rfft(torch.randn(n)) * H, n)[None]
    y = flatten_fast_envelope(x, 24000, 1.0)

    def smooth_lts(z):
        P = torch.fft.rfft(z[0] * torch.hann_window(n)).abs()
        k = torch.ones(1, 1, 201) / 201
        return torch.nn.functional.conv1d(P[None, None], k, padding=100)[0, 0]

    f = torch.linspace(0, 12000, n // 2 + 1)
    d = 20 * torch.log10(smooth_lts(y).clamp_min(1e-9) / smooth_lts(x).clamp_min(1e-9))
    err = float(d[(f > 1000) & (f < 12000)].pow(2).mean().sqrt())
    assert err < 1.5, err



# ------------------------------------------------------------------ 마찰음 레벨/모양
def _rms(y):
    return float(y.pow(2).mean().sqrt())


def test_fricative_level_is_below_the_vowel():
    """/s/ 는 모음보다 낮아야 한다. 같거나 크면 자음이 아니라 잡음 버스트로 들린다.

    (측정 이력: 노이즈 게인과 유성 진폭이 한 번도 서로 맞춰진 적이 없어서
     /s/ 가 모음보다 +4 dB 였다. 실제 음성은 -8~-20 dB.)
    """
    from formant_ml.config import Config as C
    from formant_ml.score import render
    from formant_ml.voice import VoiceProfile
    for prof in (VoiceProfile(), VoiceProfile.female()):
        v = render({"timeline": [{"type": "vowel", "vowel": "a", "dur": 0.4}],
                    "seed": 1}, prof, C())[0]
        s = render({"timeline": [{"type": "fricative", "phone": "s", "dur": 0.4}],
                    "seed": 1}, prof, C())[0]
        db = 20 * math.log10(_rms(s) / _rms(v))
        assert -20.0 < db < -6.0, db
        # 비치찰음은 훨씬 더 작다
        fq = render({"timeline": [{"type": "fricative", "phone": "f", "dur": 0.4}],
                     "seed": 1}, prof, C())[0]
        assert 20 * math.log10(_rms(fq) / _rms(s)) < -8.0


def test_sibilant_spectrum_is_broad_not_a_single_hump():
    """/s/ 의 -10 dB 폭이 4 kHz 이상이고 2 kHz 아래는 비어 있어야 한다."""
    from formant_ml.config import Config as C
    from formant_ml.score import render
    from formant_ml.voice import VoiceProfile
    for prof in (VoiceProfile(), VoiceProfile.female()):
        y = render({"timeline": [{"type": "fricative", "phone": "s", "dur": 0.5}],
                    "seed": 3}, prof, C())[0]
        S = torch.stft(y[None], 2048, 512, 2048, torch.hann_window(2048),
                       return_complex=True, center=True).abs()[0]
        P = (S ** 2).mean(-1)
        f = torch.linspace(0, 12000, len(P))
        P = P / P.max()
        idx = (P >= 0.1).nonzero().flatten()
        width = float(f[idx[-1]] - f[idx[0]])
        low = float(P[f < 2000].sum() / P.sum())
        assert width >= 4000.0, width
        assert low < 0.02, low


# ------------------------------------------------------- z축 다질량 성대 모델
def test_multimass_folds_oscillate_and_close_completely():
    """수직 적층 모델이 자가진동하고 **모든 층이 닿는 완전폐쇄**에 도달한다.

    2질량 모델은 질량이 둘뿐이라 '아래 한 번, 위 한 번' 으로만 닿는다.
    접촉면이 자라는 과정도, 지퍼 닫힘도 표현할 수 없다.
    """
    from formant_ml.dsp.vocalfold import (MultiMassParams, contact_progression,
                                          cycle_rate, simulate_multi)
    p = MultiMassParams()
    flow, traj, _ = simulate_multi(p, 7200, 24000, oversample=4)
    seg = slice(2400, None)
    f0 = cycle_rate(flow[seg])
    assert 80.0 < f0 < 400.0, f0
    d = contact_progression(traj[seg], p)
    assert d["max_contact"] > 0.99, d["max_contact"]      # 8개 층이 전부 닿는다
    closed = float((flow[seg] == 0).to(flow.dtype).mean())
    assert closed > 0.25, closed


def test_multimass_f0_does_not_depend_on_layer_count():
    """층 수는 해상도이지 물리가 아니다 — F0 가 N 에 따라 변하면 안 된다.

    질량 M/N, 횡강성 K/N 로 나눠야 고유진동수가 보존된다. (나누지 않으면
    F0 가 sqrt(N) 로 올라간다: 실측 N=2 175 Hz -> N=16 425 Hz.)
    """
    from formant_ml.dsp.vocalfold import MultiMassParams, cycle_rate, simulate_multi
    f0s = []
    for n in (4, 8, 12):
        flow, _, _ = simulate_multi(MultiMassParams(n_masses=n), 6000, 24000,
                                    oversample=4)
        f0s.append(cycle_rate(flow[2400:]))
    assert max(f0s) / min(f0s) < 1.6, f0s


def test_mucosal_wave_speed_controls_self_oscillation():
    """점막파가 빠르면 층들이 한 덩어리로 움직여 진동이 죽는다.

    수렴/발산 형상의 교대가 자가진동의 에너지원이므로, 층간 위상차가 없으면
    에너지가 들어오지 않는다. 이건 2질량 모델에서는 파라미터로 볼 수 없는 것이다.
    """
    from formant_ml.dsp.vocalfold import MultiMassParams, simulate_multi
    def amp(c):
        _, traj, _ = simulate_multi(
            MultiMassParams(mucosal_wave_speed=c), 6000, 24000, oversample=4)
        return float(traj[2400:].std())
    slow, fast = amp(40.0), amp(400.0)
    assert slow > fast * 3.0, (slow, fast)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} 통과")
    sys.exit(1 if failed else 0)
