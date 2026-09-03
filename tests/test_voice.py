"""목소리 제어 검증 — 고역, 위상차, 치찰음, 성구/파사지오, 학습되는 노이즈, 스크립트.

각 테스트는 "손잡이를 돌리면 그 물리량이 그만큼 움직이는가"와
"소리에서 그 손잡이 값을 되찾을 수 있는가"를 본다.

    PYTHONPATH=src python3 tests/test_voice.py
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from formant_ml.analysis.phase import extract as extract_phase
from formant_ml.analysis.registers import (passaggio_candidates, register_features,
                                           rd_from_h1h2)
from formant_ml.analysis.sibilant import fit_sibilant, measure as sib_measure
from formant_ml.config import Config
from formant_ml.data.features import yin_f0
from formant_ml.dsp.noise import TurbulenceSource
from formant_ml.dsp.phase import allpass_phase
from formant_ml.models.losses import (band_energy_loss, periodicity,
                                      relative_phase_loss)
from formant_ml.models.synth import Controls, PhysicalVoiceSynth
from formant_ml.score import SEGMENT_TYPES, render
from formant_ml.voice import VoiceProfile

FS = 24000
CFG = Config()
PROF = VoiceProfile()


def _hf_ratio_db(y, cut=5000.0):
    Y = torch.fft.rfft(y.reshape(-1) * torch.hann_window(y.shape[-1])).abs()
    f = torch.linspace(0, FS / 2, len(Y))
    hi = float((Y * (f > cut)).sum())
    lo = float((Y * (f <= cut)).sum())
    return 20 * math.log10(max(hi, 1e-12) / max(lo, 1e-12))


def _vowel(**kw):
    seg = {"type": "vowel", "vowel": "a", "dur": 1.0, "f0": 120.0}
    seg.update(kw)
    return render({"timeline": [seg], "seed": 3}, PROF, CFG)


# ------------------------------------------------------------------------ 고역
def test_tilt_control_moves_high_frequency_energy():
    """tilt 손잡이가 5 kHz 위 에너지를 단조롭게 올린다 ('고역 부족' 의 직접 손잡이)."""
    r = [_hf_ratio_db(_vowel(tilt=t)) for t in (-8.0, -4.0, 0.0, 4.0, 8.0)]
    assert all(a < b for a, b in zip(r, r[1:])), r
    assert r[-1] - r[0] > 12.0, f"±8 dB/oct 를 돌렸는데 고역이 {r[-1] - r[0]:.1f} dB 밖에"


def test_harmonics_reach_nyquist_for_low_voices():
    """저음 화자(F0=60 Hz)에서도 하모닉이 나이퀴스트까지 닿아야 한다."""
    assert CFG.source.n_harmonics * 60.0 >= FS / 2, \
        "n_harmonics 가 부족하면 저음 화자의 고역이 통째로 빈다"
    y = _vowel(f0=60.0, tilt=6.0)
    Y = torch.fft.rfft(y.reshape(-1) * torch.hann_window(y.shape[-1])).abs()
    f = torch.linspace(0, FS / 2, len(Y))
    band = (f > 8000) & (f < 11000)
    # 포먼트 수가 부족하면 최상단 극 위에서 캐스케이드가 극당 -12 dB/oct 로 겹쳐
    # 떨어져 여기가 통째로 비어 버린다(6 극일 때 -80 dB 수준이었다).
    assert float(Y[band].max()) / float(Y.max()) > 1e-3, "8~11 kHz 가 비어 있다"
    assert CFG.filt.n_formants * 1000.0 >= FS / 2 - 500.0, \
        "나이퀴스트까지 대략 1 kHz 당 포먼트 1 개는 있어야 한다"


def test_band_energy_loss_is_energy_invariant():
    """같은 −6 dB 오차를 고역/저역에 넣었을 때, 대역 손실은 에너지에 휘둘리지 않는다.

    spectral convergence(프로베니우스 노름)는 큰 저역이 지배해서 고역 오차를
    거의 못 본다. 대역 손실은 조용한 대역의 dB 오차도 같은 무게로 센다.
    """
    from formant_ml.data.features import stft
    from formant_ml.models.losses import spectral_convergence
    x = _vowel(tilt=4.0)
    X = torch.fft.rfft(x)
    f = torch.linspace(0, FS / 2, X.shape[-1])
    sh = lambda m, g: torch.fft.irfft(X * torch.where(m, g, 1.0), x.shape[-1])  # noqa: E731
    y_hf, y_lf = sh(f > 5000, 0.5), sh(f < 1000, 0.5)
    sc = lambda a: float(spectral_convergence(stft(a).abs(), stft(x).abs()))    # noqa: E731
    be = lambda a: float(band_energy_loss(a, x, FS))                            # noqa: E731
    assert sc(y_hf) / sc(y_lf) < 0.15, "전제 확인: SC 는 저역 지배적이어야 한다"
    assert be(y_hf) / be(y_lf) > 4 * (sc(y_hf) / sc(y_lf)), \
        f"대역 손실이 SC 만큼이나 저역 지배적이다 ({be(y_hf) / be(y_lf):.2f})"


# ------------------------------------------------------------------------ 위상
def test_phase_dispersion_changes_phase_but_not_magnitude():
    """위상차 파라미터는 크기 스펙트럼을 건드리지 않는다(올패스이므로 구조적으로).

    하모닉 경로에서 잰다. 최종 출력에서 재면 |V+N| 의 간섭 때문에 2% 쯤 차이가
    나는데, 그건 노이즈와의 합성 결과이지 크기응답의 변화가 아니다.
    (창도 반드시 씌운다. 창이 없으면 하모닉 사이 누설이 상대위상에 따라 달라져서
     4% 쯤 차이가 난다 — 역시 분석 아티팩트다.)
    """
    from formant_ml.score import build_controls

    def voiced(**kw):
        seg = {"type": "vowel", "vowel": "a", "dur": 1.0, "f0": 120.0, "rd": 0.6}
        seg.update(kw)
        c = build_controls({"timeline": [seg]}, PROF, CFG)
        g = torch.Generator().manual_seed(3)
        with torch.no_grad():
            return PhysicalVoiceSynth(CFG)(c, generator=g)["voiced"]

    a = voiced()
    b = voiced(disp_freq=[900, 2600, 5200], disp_radius=[0.9, 0.9, 0.88])
    n = min(a.shape[-1], b.shape[-1])
    w = torch.hann_window(n)
    A = torch.fft.rfft(a[0, :n] * w).abs()
    B = torch.fft.rfft(b[0, :n] * w).abs()
    mag_change = float((A - B).abs().sum() / A.sum())
    wave_change = float((a[0, :n] - b[0, :n]).abs().max() / a.abs().max())
    assert mag_change < 0.02, f"크기가 {mag_change:.3f} 만큼 바뀌었다 (올패스가 아니다)"
    assert wave_change > 0.1, "파형이 안 바뀌었다 (위상차가 적용되지 않았다)"


def test_dispersion_parameters_are_recoverable_from_audio():
    """소리에서 위상차 파라미터를 되찾는다 (실제 음성에서 뽑아내는 경로의 검증)."""
    ff, bw = PROF.formants[:CFG.filt.n_formants], PROF.bandwidths[:CFG.filt.n_formants]
    for true_f, true_r in ((1800.0, 0.85), (900.0, 0.80)):
        y = _vowel(rd=1.1, disp_freq=[true_f], disp_radius=[true_r])
        fit = extract_phase(y, rd=PROF.rd_median, n_stages=1,
                            formant_f=ff, formant_bw=bw)
        assert abs(fit["freq"][0] - true_f) / true_f < 0.08, fit
        assert abs(fit["radius"][0] - true_r) < 0.05, fit
        assert fit["residual_rad"] < 0.1, fit


def test_allpass_phase_matches_the_grid_response():
    """임의 주파수 위상 평가가 rfft 격자 응답과 일치해야 한다."""
    from formant_ml.dsp.filters import allpass_response
    apf, apr = torch.full((1, 1, 3), 2000.0), torch.full((1, 1, 3), 0.8)
    H = allpass_response(apf, apr, FS, 513)
    df = (FS / 2) / 512
    for b in (40, 128, 300):
        grid = float(torch.angle(H[0, 0, b]))
        direct = float(allpass_phase(torch.tensor([[[b * df]]]), apf, apr, FS))
        d = (grid - direct + math.pi) % (2 * math.pi) - math.pi
        assert abs(d) < 1e-4, (b, grid, direct)


def test_relative_phase_loss_is_zero_for_identical_signals():
    y = _vowel()
    f0 = torch.full((1, y.shape[-1] // 240 + 1), 120.0)
    assert float(relative_phase_loss(y, y, f0)) < 1e-6


# ---------------------------------------------------------------------- 치찰음
def test_teeth_resonance_controls_the_spectral_peak():
    """실측 모형에서 봉우리를 정하는 것은 **앞니 공명**이다.

    실제 /s/ 에 모형을 맞춰 보면 앞공동 극은 넓고(BW 3800) 낮은 곳(3.75 kHz)에
    있고, 6~8 kHz 의 봉우리는 좁은 앞니 공명(7.2 kHz, BW 1020)이 만든다.
    혀끝과 앞니 틈으로 얕게 빠져나가는 제트의 휘파람 성분이다.
    """
    for teeth in (5000.0, 8000.0):
        y = render({"timeline": [{"type": "fricative", "phone": "s", "dur": 1.5,
                                  "sib_teeth_f": teeth, "sib_teeth_bw": 900.0}],
                    "seed": 5}, PROF, CFG)
        peak = sib_measure(y, smooth_bins=41)["peak_hz"]
        assert abs(peak - teeth) / teeth < 0.12, f"피크 {peak} vs 설정 {teeth}"


def test_sibilant_shape_is_recoverable_from_audio():
    """소리에서 치찰음 모양을 되찾을 수 있어야 한다(적합 오차로 본다).

    개별 파라미터는 서로 축퇴한다(zero_f 와 slope_lo, pole 과 teeth 가 각각
    저역/봉우리를 나눠 만든다). 합성에 쓰는 것은 모양이므로 모양으로 검증한다.
    """
    y = render({"timeline": [{"type": "fricative", "phone": "s", "dur": 1.5}],
                "seed": 5}, PROF, CFG)
    fit = fit_sibilant(y, steps=600)
    assert fit["rmse_db"] < 4.0, fit
    assert 2500.0 < fit["teeth_f"] < 11000.0, fit


def test_fricative_is_not_periodic():
    """치찰음이 주기적 텍스처가 되면 안 된다 (패턴을 외워 반복하는 실패)."""
    fr = render({"timeline": [{"type": "fricative", "phone": "s", "dur": 1.0}],
                 "seed": 5}, PROF, CFG)
    vo = _vowel()
    p_fr = float(periodicity(fr).mean())
    p_vo = float(periodicity(vo).mean())
    assert p_fr < 0.35, f"마찰음 주기성이 {p_fr:.2f} 로 높다"
    assert p_vo > 0.7, f"모음 주기성이 {p_vo:.2f} 로 낮다 (HNR 붕괴)"


def test_noise_can_fully_bypass_the_tract():
    """noise_entry 를 끝까지 올리면 성도 포먼트가 노이즈에 남지 않아야 한다.

    (K 에서 멈추면 마지막 단이 19% 남아 /s/ 의 피크를 그 포먼트가 결정해 버린다.
     K+3 에서도 35% 잔물결이 남는다.)
    """
    K = CFG.filt.n_formants
    T = 60
    syn = PhysicalVoiceSynth(CFG)

    def peak(entry):
        c = Controls(
            f0=torch.full((1, T, 1), 120.0), harmonic_amp=torch.zeros(1, T, 1),
            rd=torch.full((1, T, 1), 1.2),
            formant_freq=torch.tensor([300. + 900. * k for k in range(K)]
                                      ).reshape(1, 1, -1).expand(1, T, K).contiguous(),
            formant_bw=torch.full((1, T, K), 80.0),
            formant_gain=torch.ones(1, T, K),
            noise_bands=torch.ones(1, T, CFG.noise.n_bands),
            noise_entry=torch.full((1, T, 1), entry),
            noise_am=torch.zeros(1, T, 1))
        from formant_ml.dsp.filters import bands_to_response
        H = syn(c)["h_noise"][0, 0].abs().detach()
        # 난류 소스의 색(스펙트럼 사전)은 h_noise 에 정당하게 들어 있다.
        # 여기서 보려는 건 *성도가* 남긴 구조뿐이므로 소스 항을 나눠 준다.
        src = bands_to_response(
            c.noise_bands * syn.noise.spectral_prior(), syn.n_freq,
            min_phase=True)[0, 0].abs().detach()
        R = H / src.clamp_min(1e-9)
        return float(R.max() / R.mean())

    assert peak(float(K) + 6.0) < 1.05, "완전 우회인데 아직 공진이 남아 있다"
    assert peak(0.0) > 3.0, "성문 주입인데 포먼트가 안 보인다"


# ------------------------------------------------------------------ 성대/성구
def test_rd_round_trips_through_h1h2():
    """합성한 Rd -> 소리 -> H1-H2 -> Rd 추정 이 닫혀야 한다."""
    for rd in (0.6, 1.2, 2.0):
        y = _vowel(rd=rd)
        f = register_features(y, FS, 240)
        m = f["voicing"] > 0.5
        got = float(f["rd"][m].median())
        assert abs(got - rd) < 0.2, f"Rd {rd} -> 추정 {got:.2f}"


def test_rd_from_h1h2_is_monotone():
    h = torch.linspace(-3.0, 18.0, 40)
    rd = rd_from_h1h2(h)
    assert bool((rd[1:] >= rd[:-1]).all()), "H1-H2 가 커지면 Rd 도 커져야 한다"


def test_passaggio_is_found_where_it_was_synthesized():
    """글리산도 중간에 성구 전환을 넣으면 그 F0 부근에서 검출되어야 한다."""
    lo_f0, hi_f0, brk = 100.0, 330.0, 0.55
    y = render({"timeline": [{
        "type": "vowel", "vowel": "a", "dur": 3.0,
        "f0": [[0, lo_f0], [1, hi_f0]],
        "rd": [[0, 0.7], [brk - 0.01, 0.8], [brk + 0.05, 1.9], [1, 2.1]],
    }], "seed": 3}, PROF, CFG)
    cands = passaggio_candidates(register_features(y, FS, 240))
    assert cands, "파사지오를 하나도 못 찾았다"
    expected = lo_f0 + (hi_f0 - lo_f0) * brk
    assert abs(cands[0][0] - expected) / expected < 0.12, (cands[0], expected)


def test_no_passaggio_on_a_smooth_glissando():
    """성구 전환이 없는 글리산도에서는 후보가 나오면 안 된다(오검출 방지)."""
    y = render({"timeline": [{"type": "vowel", "vowel": "a", "dur": 3.0,
                              "f0": [[0, 100], [1, 330]], "rd": 1.1}],
                "seed": 3}, PROF, CFG)
    cands = passaggio_candidates(register_features(y, FS, 240))
    assert not cands or cands[0][1] < 2.0, f"매끄러운 글리산도에서 오검출: {cands[:2]}"


# ------------------------------------------------------------------ 학습 노이즈
def test_turbulence_source_is_learnable():
    """난류의 스펙트럼 사전과 변조 스펙트럼이 학습 파라미터여야 한다."""
    ts = TurbulenceSource(FS, 240)
    names = {n for n, _ in ts.named_parameters()}
    assert {"log_prior", "raw_beta", "raw_knee"} <= names, names
    w = ts(50, 1, roughness=torch.full((1, 50, 1), 0.8))
    w.pow(2).mean().backward()
    assert ts.raw_beta.grad is not None and math.isfinite(float(ts.raw_beta.grad))
    assert abs(float(ts.spectral_prior().log().mean().detach())) < 1e-5, \
        "사전은 기하평균 1 로 정규화되어 대역게인과 싸우지 않아야 한다"


def test_roughness_makes_turbulence_non_stationary():
    """roughness 를 올리면 포락선의 시간 변동(비정상성)이 커져야 한다."""
    ts = TurbulenceSource(FS, 240)

    def env_std(r):
        w = ts(100, 1, roughness=torch.full((1, 100, 1), r),
               generator=torch.Generator().manual_seed(0)).detach()
        e = w.abs().reshape(-1, 240).mean(-1)
        return float(e.std() / e.mean())

    assert env_std(0.9) > env_std(0.0) * 1.2, (env_std(0.0), env_std(0.9))


# -------------------------------------------------------------------- 스크립트
def test_every_segment_type_renders():
    """스크립트의 모든 세그먼트 타입이 유한한 오디오를 만들어야 한다."""
    for kind in SEGMENT_TYPES:
        seg = {"type": kind, "dur": 0.25}
        if kind == "glide":
            seg["vowels"] = ["a", "i"]
        y = render({"timeline": [seg], "seed": 1}, PROF, CFG)
        assert y.shape[-1] > 0 and bool(torch.isfinite(y).all()), kind
        if kind != "silence":
            assert float(y.abs().max()) > 1e-5, f"{kind} 가 무음이다"


def _kurtosis(y):
    x = y.reshape(-1)
    x = x - x.mean()
    return float((x ** 4).mean() / (x ** 2).mean().clamp_min(1e-12) ** 2)


def _envelope_cv(y):
    e = y.reshape(-1).abs()
    k = torch.ones(1, 1, 121) / 121
    e = torch.nn.functional.conv1d(e.view(1, 1, -1), k, padding=60).view(-1)
    return float(e.std() / e.mean().clamp_min(1e-9))


def test_fricatives_are_as_smooth_as_pink_noise():
    """치찰음/속삭임이 지글거리면 안 된다.

    '지글거림' 은 진폭 분포의 두꺼운 꼬리다. 백색 소스는 이미 빠른 난류 요동을
    담고 있어서, 그 위에 광대역 곱셈 변조를 얹으면 두 잡음 과정의 곱이 되어
    첨도가 올라간다(측정: 2.97 -> 4.35). 변조를 느린 대역으로 제한해서 고쳤다.
    """
    ref = torch.randn(1, 48000)
    X = torch.fft.rfft(ref)
    f = torch.linspace(0, FS / 2, X.shape[-1])
    pink = torch.fft.irfft(X / (1 + f / 100).sqrt(), 48000)
    k_ref, cv_ref = _kurtosis(pink), _envelope_cv(pink)
    assert 2.5 < k_ref < 3.5, k_ref                      # 전제 확인

    # 첨도가 지글거림의 지표다. 포락선 CV 는 **색칠 정도에도** 반응하므로
    # (대역제한이 강할수록 포락선이 더 요동친다 — 백색 0.07 < 핑크 0.14) 상한을
    # 느슨하게 두고, 꼬리 두께는 첨도로 엄격히 본다.
    for tl in ([{"type": "fricative", "phone": "s", "dur": 2.0}],
               [{"type": "fricative", "phone": "sh", "dur": 2.0}],
               [{"type": "whisper", "dur": 2.0}],
               [{"type": "whisper", "vowel": "i", "dur": 2.0}]):
        y = render({"timeline": tl, "seed": 1}, PROF, CFG)
        k, cv = _kurtosis(y), _envelope_cv(y)
        assert k < k_ref + 0.4, f"{tl[0]['type']} 첨도 {k:.2f} (핑크 {k_ref:.2f})"
        assert cv < cv_ref * 2.0, f"{tl[0]['type']} 포락선 CV {cv:.3f} (핑크 {cv_ref:.3f})"


def _spectral_flatness(y, lo=1000.0, hi=11000.0):
    x = y.reshape(-1) - y.reshape(-1).mean()
    P = torch.fft.rfft(x * torch.hann_window(len(x))).abs() ** 2
    f = torch.linspace(0, FS / 2, len(P))
    P = P[(f > lo) & (f < hi)].clamp_min(1e-20)
    return float(torch.exp(torch.log(P).mean()) / P.mean())


def _peak_width_octaves(y, lo=800.0, hi=11500.0):
    """봉우리의 -6 dB 폭(옥타브). 좁을수록 음조로 들린다. 사람 /s/ 는 0.6~1.0."""
    x = y.reshape(-1)
    P = 20 * torch.log10(torch.fft.rfft(x * torch.hann_window(len(x))).abs()
                         .clamp_min(1e-9))
    k = torch.ones(1, 1, 151) / 151
    P = torch.nn.functional.conv1d(P.view(1, 1, -1), k, padding=75).view(-1)
    f = torch.linspace(0, FS / 2, len(P))
    m = (f > lo) & (f < hi)
    P, f = P[m], f[m]
    pk = int(P.argmax())
    ref = float(P[pk])
    up, dn = P[pk:], P[:pk]
    f_hi = f[pk:][(up < ref - 6).nonzero()[0, 0]] if (up < ref - 6).any() else f[-1]
    f_lo = f[:pk][(dn < ref - 6).nonzero()[-1, 0]] if (dn < ref - 6).any() else f[0]
    return math.log2(float(f_hi) / float(f_lo))


def _band_peakiness(y, lo=2000.0, hi=11000.0, smooth=201):
    x = y.reshape(-1)
    P = 20 * torch.log10(torch.fft.rfft(x * torch.hann_window(len(x))).abs()
                         .clamp_min(1e-9))
    k = torch.ones(1, 1, smooth) / smooth
    P = torch.nn.functional.conv1d(P.view(1, 1, -1), k, padding=smooth // 2).view(-1)
    f = torch.linspace(0, FS / 2, len(P))
    P = P[(f > lo) & (f < hi)]
    return float(P.max() - P.median())


def test_fricative_spectrum_is_not_tonal():
    """마찰음 스펙트럼이 뾰족하면 잡음이 그 공진에서 울려 음조가 들린다.

    이건 시간영역 아티팩트가 아니다 — 위상을 무작위로 돌려도 같은 소리가 난다
    (측정으로 확인). 순전히 스펙트럼이 뾰족해서 생기므로 감쇠로만 고칠 수 있다.
    사람의 /s/ 는 4~10 kHz 의 넓은 고원이고 1~11 kHz 평탄도가 대략 0.2~0.4 다.
    """
    y = render({"timeline": [{"type": "fricative", "phone": "s", "dur": 2.0}],
                "seed": 1}, PROF, CFG)
    # 판단 기준은 평탄도가 아니라 **봉우리의 좁기**다. 마찰음 스펙트럼은 원래
    # 삼각형이라 평탄하지 않다(사람도 그렇다). 음조는 봉우리가 *좁을* 때 들린다.
    width = _peak_width_octaves(y)
    assert width > 0.30, f"봉우리 -6dB 폭이 {width:.2f} 옥타브 (좁아서 음조가 들린다)"
    assert _kurtosis(y) < 3.5, "진폭 분포 꼬리가 두껍다"

    # 위상 무작위화로 '시간영역 원인이 아님' 을 확인 (회귀 방지용 전제)
    X = torch.fft.rfft(y.reshape(-1))
    ph = torch.rand(len(X)) * 2 * math.pi
    ph[0] = 0
    rnd = torch.fft.irfft(X.abs() * torch.exp(1j * ph), y.shape[-1])[None]
    assert abs(_spectral_flatness(rnd) - _spectral_flatness(y)) < 0.03


def test_phone_and_speaker_both_shape_the_sibilant():
    """음소가 범주(s/ʃ)를, 화자 프로파일이 개인차를 정해야 한다.

    프로파일만 쓰면 /s/ 와 /ʃ/ 가 같은 소리가 나고, 프리셋만 쓰면 화자 지문이
    사라진다. 둘 다 반영되어야 한다.
    """
    from formant_ml.analysis.sibilant import measure
    peak = {}
    for ph in ("s", "sh"):
        y = render({"timeline": [{"type": "fricative", "phone": ph, "dur": 1.5}],
                    "seed": 1}, PROF, CFG)
        peak[ph] = measure(y)["peak_hz"]
    assert peak["s"] > peak["sh"] * 1.4, f"/s/ 와 /ʃ/ 가 구분되지 않는다: {peak}"

    bright = VoiceProfile()
    bright.sibilant = dict(PROF.sibilant, teeth_f=PROF.sibilant["teeth_f"] * 1.25)
    y = render({"timeline": [{"type": "fricative", "phone": "s", "dur": 1.5}],
                "seed": 1}, bright, CFG)
    assert measure(y)["peak_hz"] > peak["s"] * 1.08, "화자 지문이 반영되지 않는다"


def _octave_db(y, freqs):
    x = y.reshape(-1)
    P = 20 * torch.log10(torch.fft.rfft(x * torch.hann_window(len(x))).abs()
                         .clamp_min(1e-9))
    k = torch.ones(1, 1, 151) / 151
    P = torch.nn.functional.conv1d(P.view(1, 1, -1), k, padding=75).view(-1)
    return [float(P[int(f / (FS / 2) * (len(P) - 1))]) for f in freqs]


def test_fricative_couples_to_the_oral_cavity():
    """마찰음이 구강과 결합해야 한다.

    협착은 음향적으로 완전한 벽이 아니다. 결합이 0 이면 마찰음이 입 안에서 난
    소리가 아니라 위에 얹은 히스처럼 들리고(저·중역이 통째로 빈다), 뒤따르는
    모음이 무엇이든 똑같은 소리가 난다(동시조음이 없다).
    """
    def lf(leak, formants=None):
        seg = {"type": "fricative", "phone": "s", "dur": 2.0,
               "noise_back_leak": leak}
        if formants:
            seg["formant_freq"] = formants
        y = render({"timeline": [seg], "seed": 1}, PROF, CFG)
        a, b = _octave_db(y, [1000.0, 8000.0])
        return a - b                       # 1 kHz 가 피크 대역 대비 얼마나 낮은가

    # 직접 방사 바닥(floor)이 이미 저역을 깔아 주므로, 결합의 몫은 그 위에
    # 얹히는 **공명 구조**다. 크기가 아니라 모음 의존성으로 확인한다.
    assert lf(0.35) > lf(0.0) + 0.5, "구강 결합이 저역에 아무 영향이 없다"

    # 동시조음: 같은 /s/ 라도 구강 형상이 다르면 마찰음이 달라져야 한다.
    # 바닥(floor)이 저역 레벨을 깔아 주므로, 크기가 아니라 **공명 구조**로 본다.
    def spec(formants):
        y = render({"timeline": [{"type": "fricative", "phone": "s", "dur": 2.0,
                                  "noise_back_leak": 0.35,
                                  "formant_freq": formants}], "seed": 1}, PROF, CFG)
        P = 20 * torch.log10(torch.fft.rfft(
            y.reshape(-1) * torch.hann_window(y.shape[-1])).abs().clamp_min(1e-9))
        k = torch.ones(1, 1, 41) / 41
        P = torch.nn.functional.conv1d(P.view(1, 1, -1), k, padding=20).view(-1)
        f = torch.linspace(0, FS / 2, len(P))
        return P[(f > 300) & (f < 3000)]

    front = [270.0, 2290.0, 3010.0] + [3700.0 + 900.0 * i for i in range(9)]
    back = [730.0, 1090.0, 2440.0] + [3400.0 + 900.0 * i for i in range(9)]
    d = (spec(front) - spec(back))
    assert float((d - d.mean()).abs().mean()) > 1.0, \
        "뒤따르는 모음의 구강 형상이 마찰음에 반영되지 않는다"


def test_turbulence_source_rolls_off_at_high_frequency():
    """난류원은 나이퀴스트까지 평평하지 않다.

    협착부 제트의 에디에는 특징적 크기가 있어 모서리 주파수 위로 떨어지고,
    성도 벽의 점성·열 손실도 주파수를 따라 커진다. 평평하게 두면 6~12 kHz 가
    고원처럼 남아 마찰음이 '쨍하게' 들린다.
    """
    from formant_ml.dsp.noise import TurbulenceSource
    ts = TurbulenceSource(FS, CFG.audio.hop_size, CFG.noise.n_bands)
    prior = ts.spectral_prior().detach()
    f = torch.linspace(0, FS / 2, len(prior))
    lo = float(prior[(f > 2000) & (f < 4000)].mean())
    hi = float(prior[f > 10000].mean())
    assert hi < lo * 0.7, "난류 소스 사전이 고역에서 떨어지지 않는다"

    y = render({"timeline": [{"type": "fricative", "phone": "s", "dur": 2.0}],
                "seed": 1}, PROF, CFG)
    a, b = _octave_db(y, [8000.0, 11000.0])
    assert 2.0 < a - b < 12.0, f"8k->11k 감쇠가 {a - b:.1f} dB (사람은 4~8 dB)"


def test_noise_bandwidth_scale_reduces_ringing():
    """noise_bw_scale 을 올리면 노이즈 경로의 공진 울림이 줄어야 한다."""
    def ring(k):
        y = render({"timeline": [{"type": "whisper", "vowel": "i", "dur": 2.0,
                                  "noise_bw_scale": k}], "seed": 1}, PROF, CFG)
        x = y.reshape(-1) - y.reshape(-1).mean()
        n = len(x)
        S = torch.fft.rfft(x, 2 * n)
        ac = torch.fft.irfft(S.real ** 2 + S.imag ** 2, 2 * n)[:n]
        return float((ac / ac[0])[3:400].abs().max())
    assert ring(6) < ring(1) - 0.05, (ring(1), ring(6))


def test_unknown_segment_option_is_rejected():
    """오타 난 옵션이 조용히 무시되면 안 된다 (예전에는 세그먼트 옵션이 통째로 사라졌다)."""
    from formant_ml.score import build_controls
    try:
        build_controls({"timeline": [{"type": "whisper", "dur": 0.5,
                                      "strenght": 0.5}]}, PROF, CFG)
    except ValueError as e:
        assert "strenght" in str(e)
    else:
        raise AssertionError("오타를 잡지 못했다")


def test_roughness_knob_never_sizzles():
    """난류 거칠기를 끝까지 올려도 진폭 분포가 무너지지 않아야 한다.

    손잡이의 어떤 값도 비물리적 아티팩트를 못 내게 하는 것이 이 레포의 규약이다.
    """
    prev = 0.0
    for r in (0.0, 0.3, 0.6, 1.0):
        y = render({"timeline": [{"type": "fricative", "phone": "s", "dur": 2.0,
                                  "noise_rough": r}], "seed": 1}, PROF, CFG)
        k = _kurtosis(y)
        assert k < 4.6, f"noise_rough={r} 에서 첨도 {k:.2f}"
        assert k >= prev - 0.15, "거칠기를 올렸는데 오히려 매끄러워졌다"
        prev = k


def test_unvoiced_sounds_get_no_glottal_am():
    """성대가 안 떨면 성문동기 변조도 없어야 한다 (속삭임에 F0 주기성 금지)."""
    y = render({"timeline": [{"type": "whisper", "dur": 2.0, "noise_am": 1.0}],
                "seed": 1}, PROF, CFG)
    assert float(periodicity(y, FS, 240).mean()) < 0.3, \
        "무성음인데 F0 주기성이 보인다"


def test_no_transient_at_segment_boundaries():
    """세그먼트 경계에서 클릭이 생기면 안 된다 (평활 정도와 무관해야 한다).

    실제로 겪은 버그 두 개의 회귀 테스트다.
    (1) 모음 프리셋이 5 개뿐인데 12 단까지 마지막 값을 반복해 극이 겹쳤다 (Q^4).
    (2) 무음->마찰음에서 협착 위치를 보간해 '반쪽 캐스케이드' 가 만들어졌다.
    """
    tl = [{"type": "silence", "dur": 0.3},
          {"type": "fricative", "phone": "s", "dur": 0.5},
          {"type": "silence", "dur": 0.2},
          {"type": "vowel", "vowel": "a", "dur": 0.4},
          {"type": "syllable", "onset": "s", "vowel": "a", "dur": 0.5},
          {"type": "laugh", "dur": 0.6}]
    peaks = []
    for sm in (1, 2, 3, 5):
        y = render({"timeline": tl, "smooth_frames": sm, "seed": 0}, PROF, CFG)
        e = y[0].abs().reshape(-1, 240).max(-1).values
        body = e[e > e.median()]
        ratio = float(e.max() / body.median())
        assert ratio < 5.0, f"smooth={sm} 에서 순간 피크가 본체의 {ratio:.1f} 배"
        peaks.append(float(e.max()))
    assert max(peaks) / min(peaks) < 1.2, f"평활 정도가 피크를 바꾼다: {peaks}"


def test_partial_cascade_stays_bounded():
    """부분 캐스케이드(노이즈 경로)는 w=0/1 에서 정확하고 중간에서 폭발하지 않는다."""
    from formant_ml.dsp.filters import (gated_cascade_response,
                                        resonator_stage_responses)
    K = CFG.filt.n_formants
    ff = torch.tensor(PROF.formants[:K]).view(1, 1, -1)
    bw = torch.tensor(PROF.bandwidths[:K]).view(1, 1, -1)
    g = torch.ones(1, 1, K)
    full = resonator_stage_responses(ff, bw, g, FS, 513).prod(dim=2)
    got = gated_cascade_response(ff, bw, g, torch.ones(1, 1, K), FS, 513,
                                 normalize=False)
    assert float((full - got).abs().max() / full.abs().max()) < 1e-5, "w=1 불일치"
    idn = gated_cascade_response(ff, bw, g, torch.zeros(1, 1, K), FS, 513,
                                 normalize=False)
    assert float((idn - 1.0).abs().max()) < 1e-6, "w=0 이 항등이 아니다"
    for entry in torch.linspace(0.0, K + 6.0, 13):
        idx = torch.arange(K, dtype=torch.float32).view(1, 1, K)
        w = torch.sigmoid((idx - entry) / 0.7)
        H = gated_cascade_response(ff, bw, g, w, FS, 513).abs()
        # 정규화된 노이즈 경로 필터는 어떤 협착 위치에서도 증폭하지 않는다.
        assert float(H.max()) <= 1.0 + 1e-5, \
            f"entry={float(entry):.1f} 에서 이득 {float(H.max()):.1f}"
        assert bool(torch.isfinite(H).all())


def test_laugh_pulse_rate_matches_the_request():
    """웃음의 호기 펄스 속도가 지정한 rate_hz 와 일치해야 한다."""
    for rate in (4.0, 7.0):
        y = render({"timeline": [{"type": "laugh", "dur": 2.0, "rate_hz": rate,
                                  "voiced": 0.85}], "seed": 1}, PROF, CFG)
        e = y[0].abs()
        k = torch.ones(1, 1, 241) / 241
        e = torch.nn.functional.conv1d(e.view(1, 1, -1), k, padding=120).view(-1)
        E = torch.fft.rfft((e - e.mean()) * torch.hann_window(len(e))).abs()
        f = torch.linspace(0, FS / 2, len(E))
        m = (f > 2) & (f < 20)
        got = float(f[m][E[m].argmax()])
        assert abs(got - rate) < 0.5, f"{rate} Hz 요청, {got:.2f} Hz 측정"


def test_every_parameter_is_addressable_from_a_script():
    """PARAM_HELP 에 적힌 이름이 실제로 제어에 반영되는지 (문서와 구현의 일치)."""
    from formant_ml.score import PARAM_HELP, build_controls
    seg = {"type": "vowel", "vowel": "a", "dur": 0.3}
    probes = {"f0": 155.0, "rd": 2.0, "tilt": 5.0, "jitter": 0.02,
              "shimmer": 0.2, "noise_am": 0.7, "noise_rough": 0.9,
              "noise_entry": 4.0, "sib_pole_f": 5100.0, "sib_mix": 1.0,
              "f1": 640.0, "bw2": 210.0}
    assert set(probes) <= set(PARAM_HELP), set(probes) - set(PARAM_HELP)
    c = build_controls({"timeline": [seg | probes], "smooth_frames": 1}, PROF, CFG)
    assert abs(float(c.f0.mean()) - 155.0) < 1.0
    assert abs(float(c.rd.mean()) - 2.0) < 0.01
    assert abs(float(c.tilt.mean()) - 5.0) < 0.01
    assert abs(float(c.jitter.mean()) - 0.02) < 1e-4
    assert abs(float(c.noise_entry.mean()) - 4.0) < 0.01
    assert abs(float(c.sib.pole_f.mean()) - 5100.0) < 1.0
    assert abs(float(c.formant_freq[..., 0].mean()) - 640.0) < 1.0
    assert abs(float(c.formant_bw[..., 1].mean()) - 210.0) < 1.0


def test_lpc_formants_resolve_close_formants():
    """/아/ 의 F1(730) 과 F2(1090) 은 켑스트럼으로는 뭉개진다. LPC 는 분리해야 한다."""
    from formant_ml.analysis.extract import lpc_formants
    from formant_ml.data.features import yin_f0
    y = _vowel(vowel="a", dur=2.0, f0=120.0)
    _, v = yin_f0(y, FS, 240)
    f, b = lpc_formants(y, v[0], FS, 240)
    assert len(f) >= 4, f
    for got, want in zip(f[:4], PROF.formants[:4]):
        assert abs(got - want) / want < 0.10, (f[:4], PROF.formants[:4])
    assert all(0.0 < x < 900.0 for x in b[:4]), b[:4]


def test_extracted_profile_recovers_what_was_synthesized():
    """합성 -> 추출 -> 값 비교. 분석 경로 전체가 닫히는지 본다."""
    from formant_ml.analysis.extract import extract_profile
    import tempfile
    from formant_ml.utils import save_wav
    src = VoiceProfile(name="probe", f0_median=140.0, tilt=2.0, rd_median=1.0)
    src.sibilant = dict(src.sibilant, teeth_f=7900.0, pole_f=3500.0)
    d = tempfile.mkdtemp()
    save_wav(f"{d}/v.wav", render({"timeline": [
        {"type": "vowel", "vowel": "a", "dur": 2.5, "f0": 140.0}], "seed": 11},
        src, CFG), FS)
    save_wav(f"{d}/s.wav", render({"timeline": [
        {"type": "fricative", "phone": "s", "dur": 1.5}], "seed": 11}, src, CFG), FS)
    got = extract_profile([f"{d}/v.wav", f"{d}/s.wav"], "probe", FS,
                          vowel_paths=[f"{d}/v.wav"], sibilant_paths=[f"{d}/s.wav"],
                          verbose=False)
    assert abs(got.f0_median - 140.0) < 3.0, got.f0_median
    assert abs(got.rd_median - 1.0) < 0.3, got.rd_median
    assert abs(got.sibilant["pole_f"] - 3500.0) / 3500.0 < 0.15, got.sibilant
    # zero_f 와 slope_lo 는 둘 다 저역 스커트를 만들기 때문에 서로 맞바꿔 가며
    # 같은 모양을 낼 수 있다(부분적 축퇴). 개별 값이 아니라 **재현된 모양**이
    # 맞는지를 본다 — 합성에 쓰는 것도 모양이다.
    assert got.sibilant_moments.get("fit_rmse_db", 99) < 4.5, got.sibilant_moments
    assert abs(got.sibilant["teeth_f"] - 7900.0) / 7900.0 < 0.15, got.sibilant
    for a, b in zip(got.formants[:3], (730.0, 1090.0, 2440.0)):
        assert abs(a - b) / b < 0.12, got.formants[:3]


def test_residual_starts_as_identity_and_stays_bounded():
    """잔차망은 (a) 처음에 아무것도 안 고치고 (b) 어떤 가중치에서도 물리모델을
    대체할 수 없어야 한다. 이게 'AI 가 결국 다 해버리는' 붕괴에 대한 구조적 방어다.
    """
    from formant_ml.models.losses import residual_energy_db
    from formant_ml.models.residual import ResidualCorrector
    torch.manual_seed(0)
    res = ResidualCorrector(CFG)
    mel = torch.randn(2, 50, CFG.audio.n_mels)
    f0, v = torch.full((2, 50), 150.0), torch.ones(2, 50)
    a = torch.randn(2, 50 * CFG.audio.hop_size)

    y = res.apply(a, res(mel, mel, f0, v))
    assert residual_energy_db(y, a) < -30.0, "초기화 상태에서 이미 소리를 바꾼다"

    # 가중치를 크게 흔들어도 보정량이 상한 안에 머문다
    with torch.no_grad():
        for prm in res.parameters():
            prm.mul_(0).add_(torch.randn_like(prm) * 3.0)
    r = res(mel, mel, f0, v)
    assert float(r["filter_db"].abs().max()) <= res.max_db + 1e-4
    assert float(r["noise_gain"].max()) <= res.max_noise + 1e-4
    assert residual_energy_db(res.apply(a, r), a) < 3.0, "잔차가 원신호를 넘어선다"


def test_residual_gradients_flow():
    from formant_ml.models.residual import ResidualCorrector
    res = ResidualCorrector(CFG)
    mel = torch.randn(1, 40, CFG.audio.n_mels)
    a = torch.randn(1, 40 * CFG.audio.hop_size, requires_grad=True)
    res.apply(a, res(mel, mel, torch.full((1, 40), 150.0),
                     torch.ones(1, 40))).pow(2).mean().backward()
    g = sum(float(p.grad.abs().sum()) for p in res.parameters() if p.grad is not None)
    assert g > 0 and math.isfinite(g)


def test_streaming_matches_offline_synthesis():
    """청크로 나눠 만들어도 한 번에 만든 것과 같아야 한다 (실시간 제어의 근거)."""
    from formant_ml.score import build_controls
    from formant_ml.streaming import StreamingSynth
    c = build_controls({"timeline": [{"type": "vowel", "vowel": "a", "dur": 1.5,
                                      "f0": [[0, 150], [0.5, 190], [1, 110]],
                                      "jitter": 0.0, "shimmer": 0.0}]}, PROF, CFG)
    c.noise_bands = c.noise_bands * 0           # 노이즈는 난수라 청크마다 다르다
    syn = PhysicalVoiceSynth(CFG)
    with torch.no_grad():
        full = syn(c)["audio"]
    d = CFG.filt.ir_size // 2
    for chunk in (2, 10, 25):
        st = StreamingSynth(CFG, synth=syn)
        parts = [st.step(c.slice(t0, min(t0 + chunk, c.n_frames)))
                 for t0 in range(0, c.n_frames, chunk)]
        parts.append(st.flush())
        stream = torch.cat([p for p in parts if p.shape[-1]], dim=-1)
        n = min(full.shape[-1], stream.shape[-1] - d)
        err = float((full[:, :n] - stream[:, d:d + n]).abs().max()
                    / full.abs().max())
        assert err < 5e-3, f"청크 {chunk} 프레임에서 오차 {err:.2e}"
    assert StreamingSynth(CFG).latency_ms < 30.0


def test_prosody_rate_and_pitch_are_controllable():
    """조음 속도와 피치 엔벨로프가 의도대로 반영되어야 한다 (LLM 제어면)."""
    from formant_ml.score import build_controls
    tl = [{"type": "syllable", "onset": "s", "vowel": "a", "dur": 0.5},
          {"type": "vowel", "vowel": "i", "dur": 0.5}] * 3
    base = build_controls({"timeline": tl}, PROF, CFG).n_frames
    fast = build_controls({"timeline": tl, "prosody": {"rate": 1.5}},
                          PROF, CFG).n_frames
    slow = build_controls({"timeline": tl, "prosody": {"rate": 0.7}},
                          PROF, CFG).n_frames
    assert fast < base < slow, (fast, base, slow)
    assert 0.6 < (fast / base) / (1 / 1.5) < 1.7, "속도 배율이 길이에 반영되지 않는다"

    c = build_controls({"timeline": tl, "prosody": {
        "pitch_shift": 3.0, "contour": [[0, 0], [0.5, 4], [1, -4]],
        "declination": -2.0, "pitch_range": 1.5}}, PROF, CFG)
    f0 = c.f0[0, :, 0]
    assert CFG.source.f0_min <= float(f0.min()), "음역을 벗어났다"
    assert float(f0.max()) <= CFG.source.f0_max
    flat = build_controls({"timeline": tl, "prosody": {"pitch_range": 0.0,
                                                       "declination": 0.0}},
                          PROF, CFG).f0[0, :, 0]
    assert float(flat.std()) < float(f0.std()), "억양 폭 제어가 안 먹는다"


def test_breath_is_inserted_for_long_utterances():
    """길게 말하면 숨을 쉬어야 한다. 짧으면 안 쉬어야 한다."""
    from formant_ml.prosody import ProsodyPlan, warp_timeline
    long_tl = [{"type": "vowel", "vowel": "a", "dur": 0.6}] * 12
    n = sum(1 for s in warp_timeline(
        long_tl, ProsodyPlan.from_dict({"breath": {"capacity_s": 2.0}}))
        if s["type"] == "breath")
    assert n >= 2, f"7 초 발화에 들숨이 {n} 회"
    short = warp_timeline(long_tl[:2],
                          ProsodyPlan.from_dict({"breath": {"capacity_s": 4.5}}))
    assert all(s["type"] != "breath" for s in short), "1.2 초인데 숨을 쉰다"
    off = warp_timeline(long_tl, ProsodyPlan.from_dict(
        {"breath": {"enabled": False}}))
    assert all(s["type"] != "breath" for s in off)


def test_fricative_level_matches_profile():
    """마찰음이 유성음보다 얼마나 조용한지가 프로파일대로 나와야 한다.

    실측(한국어 "스"): 모음이 5.4 dB 크다. 이걸 안 맞추면 치찰음만 튀어나온다.
    합성 경로를 바꾸면 score.FRICATIVE_CAL_DB 를 다시 재야 하는데, 이 테스트가
    그 드리프트를 잡는다.
    """
    def level(a):
        return 20 * math.log10(float(a.pow(2).mean().sqrt()) + 1e-12)

    v = render({"timeline": [{"type": "vowel", "vowel": "eu", "dur": 1.0}],
                "seed": 1}, PROF, CFG)
    for want in (-8.0, -5.4, -2.0):
        pr = VoiceProfile()
        pr.fricative_level_db = want
        f = render({"timeline": [{"type": "fricative", "phone": "s", "dur": 1.0}],
                    "seed": 1}, pr, CFG)
        assert abs((level(v) - level(f)) - (-want)) < 1.0, \
            f"목표 {-want:+.1f} dB, 실제 {level(v) - level(f):+.1f} dB"


def test_source_tilt_does_not_blow_up_the_level():
    """소스 기울기는 **모양**이지 레벨이 아니다.

    축을 1 kHz 에 두면 F0=120 Hz 의 기본파가 축보다 3.3 옥타브 아래라,
    tilt=-6 dB/oct 만 줘도 기본파가 10 배가 된다(실측: 모음이 20 dB 폭발).
    H1 을 축으로 두면 그런 일이 없다.
    """
    def level(t):
        y = render({"timeline": [{"type": "vowel", "vowel": "eu", "dur": 1.0,
                                  "tilt": t}], "seed": 1}, PROF, CFG)
        return 20 * math.log10(float(y.pow(2).mean().sqrt()) + 1e-12)

    lv = [level(t) for t in (-8.0, -4.0, 0.0, 4.0, 8.0)]
    assert max(lv) - min(lv) < 20.0, f"tilt 로 레벨이 {max(lv) - min(lv):.0f} dB 움직인다"


def test_stacked_folds_show_a_mucosal_wave():
    """성대를 여러 겹 쌓으면 하연이 상연을 앞서는 점막파가 나와야 한다.

    2 겹으로는 성문 통로가 두 점을 잇는 직선뿐이라 모양 변화를 못 담는다.
    n 겹이면 통로가 열릴 때 수렴형·닫힐 때 발산형으로 바뀌는 것이 궤적에 나온다.
    """
    from formant_ml.dsp.vocalfold import (FoldParams, cycle_rate,
                                          mucosal_wave_delay, simulate_stack)
    f0s = []
    for n in (2, 5):
        flow, traj = simulate_stack(FoldParams(a01=0.02, a02=0.02), n_masses=n,
                                    n_samples=7200)
        seg, tr = flow[2400:], traj[2400:]
        assert float(seg.std()) > 1.0, f"n={n} 에서 진동이 죽었다"
        f0 = cycle_rate(seg, 24000)
        assert 60.0 < f0 < 400.0, f"n={n} F0 {f0}"
        f0s.append(f0)
        lag = mucosal_wave_delay(tr)
        assert 0.1 < lag < 2.5, f"n={n} 점막파 {lag:.2f} ms (사람 0.5~1.5, 양수=하연 선행)"
        # 상연이 하연보다 크게 움직인다
        assert float(tr[:, -1].std()) > float(tr[:, 0].std())
    # 겹 수를 바꿔도 F0 가 흐르면 안 된다 (질량/강성 분할이 틀렸다는 뜻)
    assert abs(f0s[0] - f0s[1]) / f0s[0] < 0.25, f0s


def test_stacked_folds_track_tension():
    from formant_ml.dsp.vocalfold import FoldParams, cycle_rate, simulate_stack
    r = []
    for q in (0.7, 1.6):
        flow, _ = simulate_stack(FoldParams(a01=0.02, a02=0.02, q=q), n_masses=5,
                                 n_samples=7200)
        r.append(cycle_rate(flow[2400:], 24000))
    assert r[1] > r[0] * 1.5, r


def test_syllable_uses_a_consonant_locus_not_a_glide():
    """/사/ 의 유성 구간은 자음 로커스에서 출발해야 한다.

    /이/ 포먼트에서 시작해 모음으로 미끄러뜨리면 그건 정의상 /j/ 활음이라
    **"야"** 로 들린다(실제로 그렇게 들린다는 지적을 받았다). 치경 로커스는
    F2 가 1750 Hz 부근이고, /a/(F2≈1090)로 갈 때 F2 가 **내려간다**.
    /이/ 활음이면 반대로 2290 -> 1090 으로 훨씬 크게 내려온다.
    """
    from formant_ml.score import build_controls
    c = build_controls({"timeline": [{"type": "syllable", "onset": "s",
                                      "vowel": "a", "dur": 0.55,
                                      "onset_s": 0.12}]}, PROF, CFG)
    f2 = c.formant_freq[0, :, 1]
    amp = c.harmonic_amp[0, :, 0]
    onset = int((amp > 0.05).float().argmax())
    assert onset > 0, "유성 시작을 못 찾았다"
    start = float(f2[onset])
    assert 1400.0 < start < 2000.0, f"유성 시작 F2 가 {start:.0f} Hz (치경 로커스는 ~1750)"
    assert start < 2100.0, "/이/ 활음이다"
    # 전이는 짧아야 한다 (60 ms 안팎). 길면 활음처럼 들린다.
    tgt = float(f2[-1])
    reached = int((((f2 - tgt).abs() < 40.0).float()).argmax())
    assert (reached - onset) < 12, f"전이가 {(reached - onset) * 10} ms 로 너무 길다"


def test_measured_formant_values_are_used_directly():
    """프로파일에 측정 포먼트가 있으면 프리셋을 스케일하지 않고 그대로 쓴다.

    균일 스케일은 F1 과 F2 를 같은 비율로 옮기는데 실제 화자는 그렇지 않다
    (실측: F1 994 인데 F2 는 1485 가 아니라 1226).
    """
    from formant_ml.score import _vowel_formants
    p = VoiceProfile()
    p.vowel_formants = {"a": [994.0, 1226.0, 3144.0, 4457.0]}
    got = _vowel_formants("a", p, CFG.filt.n_formants)
    assert abs(got[0] - 994.0) < 1.0 and abs(got[1] - 1226.0) < 1.0, got[:3]


def test_voice_profile_round_trips_through_json():
    import tempfile
    p = VoiceProfile(name="t", f0_median=143.0, tilt=2.5)
    p.sibilant["pole_f"] = 7100.0
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    p.save(path)
    q = VoiceProfile.load(path)
    os.unlink(path)
    assert q.name == "t" and q.f0_median == 143.0 and q.sibilant["pole_f"] == 7100.0
    df, dr = q.dispersion_tensors(1, 5)
    assert df is None, "빈 위상차는 None 이어야 한다"


# ---------------------------------------------------------------------- 특징
def test_yin_f0_is_accurate():
    for f0 in (60.0, 150.0, 400.0):
        t = torch.arange(FS) / FS
        x = sum(a * torch.sin(2 * math.pi * k * f0 * t)
                for k, a in ((1, 1.0), (2, 0.5), (3, 0.3)))[None]
        est, voi = yin_f0(x, FS, 240)
        assert abs(float(est[est > 0].median()) - f0) / f0 < 0.01, f0
        assert float(voi.mean()) > 0.8


def test_yin_reports_noise_as_unvoiced():
    _, voi = yin_f0(torch.randn(1, FS), FS, 240)
    assert float(voi.mean()) < 0.1


def _frame_rms(y, win=1200):
    y = y.reshape(-1)
    n = y.shape[0] // win
    return torch.stack([y[i * win:(i + 1) * win].pow(2).mean().sqrt()
                        for i in range(n)])


def test_sibilant_fade_shapes_the_flow_envelope():
    """치찰음이 유량 포락선을 따라 페이드 인/아웃 한다.

    게이트(fade=0)는 상수 히스, 페이드는 시작·끝이 눌리고 가운데가 크다.
    """
    gate = render({"timeline": [{"type": "fricative", "phone": "s", "dur": 0.9,
                                 "fade_in": 0.0, "fade_out": 0.0}], "seed": 1},
                  PROF, CFG)
    fade = render({"timeline": [{"type": "fricative", "phone": "s", "dur": 0.9,
                                 "fade_in": 0.28, "fade_out": 0.28}], "seed": 1},
                  PROF, CFG)
    g, f = _frame_rms(gate), _frame_rms(fade)
    # 게이트: 시작/끝이 가운데와 비슷 (변동 작다)
    assert g[1] > 0.6 * g[len(g) // 2], "게이트인데 시작이 눌렸다"
    assert g[-2] > 0.6 * g[len(g) // 2], "게이트인데 끝이 눌렸다"
    # 페이드: 시작·끝이 가운데보다 훨씬 작다
    mid = f[len(f) // 2]
    assert f[0] < 0.2 * mid, f"fade-in 이 안 눌렸다: {float(f[0])} vs {float(mid)}"
    assert f[-1] < 0.2 * mid, f"fade-out 이 안 눌렸다: {float(f[-1])} vs {float(mid)}"


def test_flow_to_noise_amp_is_superlinear():
    """유량->진폭 매핑이 초선형이다 (파워 ∝ U^n, 진폭 ∝ U^(n/2), n>2).

    반쯤 열린 유량(0.5)에서 진폭이 0.5 보다 작아야 한다 — 즉 낮은 유량에서
    소리가 더 많이 죽어 부드러운 시작을 만든다.
    """
    from formant_ml import aerodynamics as aero
    half = aero.flow_to_noise_amp(torch.tensor(0.5))
    assert float(half) < 0.5, f"초선형이 아니다: {float(half)}"
    # 단조 증가
    u = torch.linspace(0, 1, 20)
    a = aero.flow_to_noise_amp(u)
    assert bool((a[1:] >= a[:-1]).all()), "유량-진폭이 단조가 아니다"


def test_pressure_after_fricative_does_not_break_concat():
    """치찰음(압력 없음) + 압력 실린 모음을 한 타임라인에 섞어도 이어붙는다.

    pressure/adduction 은 파생 손잡이라 제어 dict 에 곡선으로 남으면 안 된다 —
    일부 세그먼트에만 있으면 torch.cat 이 KeyError 로 터졌다(회귀 방지).
    그리고 압력이 실린 모음은 더 세고(amp↑) 성문이 닫혀 pressed(Rd↓) 여야 한다.
    """
    from formant_ml.score import build_controls
    score = {"smooth_frames": 2, "timeline": [
        {"type": "fricative", "phone": "s", "dur": 0.4, "fade_in": 0.05},
        {"type": "glide", "vowels": ["eu", "a"], "dur": 0.5,
         "pressure": [[0, 1.8], [1, 0.85]], "adduction": 0.95,
         "f0": [[0, 200], [1, 180]]}]}
    ctrl = build_controls(score, PROF, CFG)          # 터지면 여기서 예외
    amp = ctrl.harmonic_amp[0, :, 0]
    rd = ctrl.rd[0, :, 0]
    v = (amp > 1e-3).nonzero().squeeze(-1)
    assert len(v) > 0, "유성 구간이 없다"
    onset, tail = int(v[0]), int(v[-1])
    assert float(amp[onset + 1]) > float(amp[tail]), "압력 실린 시작이 더 세야 한다"
    assert float(rd[onset + 1]) < PROF.rd_median + 0.05, "시작이 pressed 여야 한다"


def test_female_profile_sibilant_is_high_frequency():
    """여성 프로파일의 /s/ 는 실측(길게 끈 치찰음)대로 고역에 봉우리가 있다.

    앞니 공명 ~9.9 kHz + 낮은 floor 로 10 kHz 부근이 지배적이고 저역(<3 kHz)은
    거의 비어야 한다. 이걸 안 지키면 '치찰음처럼 안 들린다'.
    """
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "profiles",
                        "female_ko.json")
    if not os.path.exists(path):
        return                                       # 프로파일이 없으면 건너뜀
    prof = VoiceProfile.load(path)
    y = render({"timeline": [{"type": "fricative", "phone": "s", "dur": 0.7,
                              "fade_in": 0.05, "fade_out": 0.05}], "seed": 5},
               prof, CFG)
    a = y.reshape(-1)[int(0.2 * FS):int(0.6 * FS)]
    S = torch.fft.rfft(a * torch.hann_window(a.shape[-1])).abs()
    f = torch.linspace(0, FS / 2, len(S))
    # 평활 포락선의 봉우리가 8 kHz 위
    k = 51
    Ss = torch.nn.functional.avg_pool1d(S.view(1, 1, -1), k, 1, k // 2).view(-1)
    peak = float(f[Ss.argmax()])
    assert peak > 8000.0, f"치찰음 봉우리가 너무 낮다: {peak:.0f} Hz"
    p = (S ** 2)
    low = float(p[f < 3000].sum() / p.sum())
    assert low < 0.12, f"저역(<3k)에 에너지가 너무 많다: {low:.2f}"


def test_custom_flow_curve_makes_two_amplitude_bumps():
    """flow 곡선을 직접 주면 마찰음 세기가 그 곡선을 따라 두 번 부푼다."""
    y = render({"timeline": [{"type": "fricative", "phone": "s", "dur": 1.2,
                              "flow": [[0, 0.0], [0.2, 1.0], [0.5, 0.1],
                                       [0.8, 1.0], [1.0, 0.0]]}], "seed": 1},
               PROF, CFG)
    r = _frame_rms(y, win=1200)
    valley = int(len(r) * 0.5)
    peak_a = r[:valley].max()
    peak_b = r[valley:].max()
    # 두 봉우리가 있고, 그 사이 골이 두 봉우리보다 뚜렷이 낮다
    assert r[valley] < 0.5 * min(float(peak_a), float(peak_b)), \
        f"두 봉우리 사이에 골이 없다: valley={float(r[valley])}"


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
