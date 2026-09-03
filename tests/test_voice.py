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
def test_sibilant_pole_controls_the_spectral_peak_and_is_recoverable():
    """치찰음 필터의 극이 곧 스펙트럼 피크이고, 소리에서 되찾을 수 있다."""
    for pole in (3200.0, 7000.0):
        y = render({"timeline": [{"type": "fricative", "phone": "s", "dur": 1.2,
                                  "sib_pole_f": pole, "sib_pole_bw": 700,
                                  "sib_zero_f": pole * 0.42, "sib_tilt": 0.0}],
                    "seed": 5}, PROF, CFG)
        peak = sib_measure(y)["peak_hz"]
        got = fit_sibilant(y, steps=400)["pole_f"]
        assert abs(peak - pole) / pole < 0.10, f"피크 {peak} vs 설정 {pole}"
        assert abs(got - pole) / pole < 0.15, f"재추출 {got} vs 설정 {pole}"


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
        H = syn(c)["h_noise"][0, 0].abs().detach()
        return float(H.max() / H.mean())

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

    for tl in ([{"type": "fricative", "phone": "s", "dur": 2.0}],
               [{"type": "fricative", "phone": "sh", "dur": 2.0}],
               [{"type": "whisper", "dur": 2.0}],
               [{"type": "whisper", "vowel": "i", "dur": 2.0}]):
        y = render({"timeline": tl, "seed": 1}, PROF, CFG)
        k, cv = _kurtosis(y), _envelope_cv(y)
        assert k < k_ref + 0.6, f"{tl[0]['type']} 첨도 {k:.2f} (핑크 {k_ref:.2f})"
        assert cv < cv_ref * 1.6, f"{tl[0]['type']} 포락선 CV {cv:.3f} (핑크 {cv_ref:.3f})"


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
    src = VoiceProfile(
        name="probe", f0_median=140.0, tilt=2.0, rd_median=1.0,
        sibilant={"pole_f": 7400.0, "pole_bw": 650.0, "zero_f": 3100.0,
                  "zero_bw": 900.0, "tilt": 0.5})
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
    assert abs(got.sibilant["pole_f"] - 7400.0) / 7400.0 < 0.12, got.sibilant
    assert abs(got.sibilant["zero_f"] - 3100.0) / 3100.0 < 0.20, got.sibilant
    for a, b in zip(got.formants[:3], (730.0, 1090.0, 2440.0)):
        assert abs(a - b) / b < 0.12, got.formants[:3]


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
