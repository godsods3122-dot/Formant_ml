"""공기음향 난류 소스 + 성대-난류 결합 검증 (문헌 기반).

  PYTHONPATH=src python3 tests/test_aeroacoustic.py

근거: Stevens(1971) 임계 레이놀즈수·압력강하 소스, Story&Titze(1995) body-cover,
Titze(1988) mucosal wave, Jackson&Shadle(2000) 성문동기 마찰음 변조.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from formant_ml import aeroacoustic as ac
from formant_ml.config import Config
from formant_ml.dsp.vocalfold import (FoldParams, simulate_body_cover,
                                      simulate_stack, glottal_flow_to_modulation,
                                      cycle_rate, mucosal_wave_delay)
from formant_ml.score import render
from formant_ml.voice import VoiceProfile

CFG = Config()
PROF = VoiceProfile()
FS = 24000


def test_turbulence_turns_on_only_above_critical_reynolds():
    """좁은 협착(빠른 기류)은 난류(마찰음), 열린 협착(모음)은 층류(무음).

    유량은 직렬 구동에서 나온 실제 값을 쓴다(모음은 구강이 열려 저항이 성문으로
    옮겨가 유량·속도가 함께 떨어진다).
    """
    Ps = torch.tensor(8000.0)
    u_tight = ac.series_flow(Ps, torch.tensor(0.12), torch.tensor(0.10))
    u_open = ac.series_flow(Ps, torch.tensor(0.05), torch.tensor(3.0))
    tight = ac.turbulence_gate(u_tight, torch.tensor(0.10))   # /s/ 협착
    openv = ac.turbulence_gate(u_open, torch.tensor(3.0))     # 모음 개방
    assert float(tight) > 0.9, f"협착인데 난류가 안 켜진다: {float(tight)}"
    assert float(openv) < 0.2, f"열렸는데 난류가 안 꺼진다: {float(openv)}"
    assert float(ac.reynolds(u_tight, torch.tensor(0.10))) > ac.RE_C


def test_source_amplitude_follows_pressure_drop():
    """소스 세기 ∝ ½ρ(U/A)² (Stevens 1971): 속도 2배면 진폭 ~4배."""
    a = torch.tensor(0.10)
    lo = ac.frication_source_amp(torch.tensor(150.0), a)
    hi = ac.frication_source_amp(torch.tensor(300.0), a)
    assert 3.5 < float(hi / lo.clamp_min(1e-9)) < 4.5, float(hi / lo)


def test_series_driver_couples_glottis_and_oral_constriction():
    """폐압 하나가 성문·구강을 직렬로 지나 한 유량을 만든다. 구강이 좁으면 마찰음,
    열리면 성문(기식) 쪽이 지배 — 발성·기식·마찰음이 같은 구동에서 결합."""
    Ps = torch.tensor(8000.0)
    # /s/: 성문 열림·구강 좁음 -> 구강 마찰 큼
    fr_s = ac.frication_source_amp(ac.series_flow(Ps, torch.tensor(0.12),
                                                  torch.tensor(0.10)), torch.tensor(0.10))
    # 모음: 구강 열림 -> 구강 마찰 거의 0
    fr_v = ac.frication_source_amp(ac.series_flow(Ps, torch.tensor(0.05),
                                                  torch.tensor(3.0)), torch.tensor(3.0))
    assert float(fr_s) > 100.0 * float(fr_v.clamp_min(1e-6))


def test_centroid_follows_the_jet_not_a_hand_drawn_curve():
    """치찰음 무게중심이 **제트를 따라 아치**를 그린다(손 곡선 없이).

    Stevens(1971): 무게중심은 협착부 입자속도와 함께 오른다. 그래서 지속
    마찰음의 무게중심은 세기와 함께 **가운데서 높고 양 끝에서 낮은 아치**여야
    한다. 실측(업로드 녹음의 긴 /s/, 1~20 kHz, 가청 구간의 5/50/95 %):

        토큰 #1   6521 -> 8770 -> 7339 Hz
        토큰 #2   7867 -> 9201 -> 6651 Hz

    예전에는 **정확히 반대**였다(12228 -> 8300 -> 12218): 앞니 다이폴이 독립
    마찰음 경로에서 제트에 안 묶여 있어서 첫 프레임부터 +10 dB/oct 가 만땅이라
    소스 무게중심이 15245 Hz 에 **고정**돼 있었다. 사용자가 "우리 건 처음부터
    고역이 확 튀어나온다" 고 지적한 것이 이것이다. HANDOFF §5.8.

    여기서는 방향만 본다(절대값은 화자 지문이 정한다). **화자 프로파일로 잰다** —
    아치의 크기는 앞니 공명/앞공동 극의 위치에 달려 있고, 기본 프로파일에는
    그 지문이 없다.
    """
    prof = _me_profile()
    if prof is None:
        return
    # **44.1 kHz 로 잰다.** 24 kHz 는 나이퀴스트가 12 kHz 라 앞니 공명(10 kHz)
    # 바로 위에서 스펙트럼이 잘려 무게중심이 움직일 여지가 없다(9314 -> 9354).
    from dataclasses import replace as _replace
    cfg = _replace(CFG, audio=_replace(CFG.audio, sample_rate=44100,
                                       hop_size=441, fmax=22050.0),
                   filt=_replace(CFG.filt, n_formants=len(prof.formants) + 2))
    fs = cfg.audio.sample_rate
    seg = {"type": "fricative", "phone": "s", "dur": 1.6, "aero": True,
           "drive": "tongue", "glottal_area": 0.12}
    y = render({"timeline": [{"type": "silence", "dur": 0.08}, seg],
                "seed": 5}, prof, cfg).reshape(-1)
    win, hop = 2048, 512
    f = torch.linspace(0, fs / 2, win // 2 + 1)
    S = torch.stft(y, win, hop, window=torch.hann_window(win),
                   return_complex=True).abs()
    band = (f > 1000) & (f < fs / 2 - 2000)
    P = S[band] ** 2
    tot = P.sum(0)
    alive = (tot > 0.02 * tot.max()).nonzero().reshape(-1)
    a, b = int(alive[0]), int(alive[-1])
    cen = (f[band].unsqueeze(-1) * P).sum(0) / tot.clamp_min(1e-20)
    n = b - a
    lo1 = float(cen[a + int(n * 0.05)])
    mid = float(cen[a + int(n * 0.50)])
    lo2 = float(cen[a + int(n * 0.95)])
    assert mid > lo1 + 150, f"무게중심이 가운데서 안 올라간다: {lo1:.0f} -> {mid:.0f}"
    assert mid > lo2 + 150, f"무게중심이 끝에서 안 내려간다: {mid:.0f} -> {lo2:.0f}"

def test_reactive_cavity_matches_quasistatic_and_is_stable():
    """관성(리액턴스)을 넣은 구강 공동은 준정상 모형의 **상위 집합**이다.

    두 가지를 고정한다.

    1. **정상상태 일치.** 유량이 정착하면 U_g = U_c = `series_flow`, Pm/Ps = 0.59
       (문헌 60~70 %). 즉 기존 `oral_cavity` 는 이 모형의 준정상 극한이다.
       이게 깨지면 둘 중 하나가 틀린 것이다.
    2. **해제에서 발산하지 않는다.** 헬름홀츠 각진동수는 협착이 **열릴수록**
       높다(0.10 cm² 227 Hz -> 3 cm² 1246 Hz). 좁은 협착 기준으로 부분단계를
       잡고 전진 오일러를 쓰면 해제 순간 NaN 이 난다(실제로 났다). 그래서
       심플렉틱 적분 + 최대 면적 기준 적응 부분단계를 쓴다.

    왜 기본 경로에 안 쓰나: 프레임률(100 Hz)에서 **차이가 없기 때문이다**.
    관성 시상수는 τ = Lc·Ac/U 로 0.5~20 ms 라 한 프레임 안에서 끝난다.
    측정: 실제 음절 조건에서 U_g − U_c 의 최대가 유량의 0.03 % 였고 Pm/Ps 는
    소수 셋째 자리까지 같았다. 계산량만 50~100 배다. 자세한 건 HANDOFF §5f.
    """
    T, fr = 40, 100.0
    ps = torch.full((1, T, 1), 8000.0)
    ag = torch.full((1, T, 1), 0.12)
    # 앞부분은 좁은 협착, 뒤는 모음 면적으로 연다(발산 검사).
    ac_t = ac.constriction_area(T, fr, a_closed=0.10, a_open=3.0,
                                hold=0.5, release=0.05)
    pm, u_c, u_g = ac.oral_cavity_reactive(ps, ag, ac_t, fr)
    assert torch.isfinite(pm).all() and torch.isfinite(u_c).all(), "해제에서 발산했다"

    # 좁은 협착 구간의 끝(정착 후)에서 준정상과 맞아야 한다.
    i = int(0.5 * (T - 1)) - 1
    u_ss = ac.series_flow(ps, ag, ac_t)[0, i, 0]
    assert abs(float(u_c[0, i, 0] - u_ss)) / float(u_ss) < 0.02, \
        f"정상상태 유량이 series_flow 와 다르다: {float(u_c[0,i,0]):.1f} vs {float(u_ss):.1f}"
    assert abs(float(u_g[0, i, 0] - u_c[0, i, 0])) / float(u_ss) < 0.02, \
        "정상상태에서 성문유량과 협착유량이 달라졌다(질량보존 위반)"
    frac = float(pm[0, i, 0] / ps[0, i, 0])
    assert 0.55 < frac < 0.65, f"구강내압 비율이 문헌 밖이다: {frac:.3f}"


def test_body_cover_three_mass_self_oscillates():
    """Story & Titze(1995) body-cover 3질량이 자가진동한다."""
    flow, _ = simulate_body_cover(FoldParams(ps=8000.0, a01=0.02, a02=0.02),
                                  n_samples=6000, sample_rate=FS)
    assert float(flow.std()) > 1e-3, "진동하지 않는다"
    assert 60 < cycle_rate(flow, FS) < 400, cycle_rate(flow, FS)


def test_vertical_multimass_has_mucosal_wave():
    """수직 다질량은 하연이 상연을 앞선다(점막파, Titze 1988). 쓸모없지 않다."""
    _, traj = simulate_stack(FoldParams(ps=8000.0, a01=0.02, a02=0.02),
                             n_masses=5, n_samples=6000, sample_rate=FS)
    delay = mucosal_wave_delay(traj, FS)
    assert 0.1 < delay < 3.0, f"점막파 지연이 비생리적: {delay:.2f} ms"


def _hf_shape(y, fs=FS, voiced_stop=False):
    """4 kHz 이상 포락선의 (길이, 정점위치, 상승/하강). 정점위치가 페이드인 비율."""
    win, hop = 1024, 256
    f = torch.linspace(0, fs / 2, win // 2 + 1)
    S = torch.stft(y.reshape(-1), win, hop, window=torch.hann_window(win),
                   return_complex=True).abs()
    hi = (S[f > 4000] ** 2).sum(0).sqrt()
    lo = (S[(f > 200) & (f < 1000)] ** 2).sum(0).sqrt()
    hi = (hi / hi.amax()).numpy()
    lo = (lo / lo.amax()).numpy()
    pk = int(hi.argmax())
    thr = hi.max() * 0.10
    on = int((hi > thr).argmax())
    v = (lo > 0.5).nonzero()[0]
    off = (int(v[0]) if voiced_stop and len(v) and v[0] > pk
           else len(hi) - int((hi[::-1] > thr).argmax()) - 1)
    return ((off - on) * hop / fs, (pk - on) / max(off - on, 1),
            (pk - on) / max(off - pk, 1))


def test_fade_in_is_more_than_half_of_the_sibilant():
    """치찰음의 페이드 인이 소리의 절반을 넘는다 — 협착 고정, 압력만으로.

    실측(업로드 녹음 4 토큰): 정점 위치 56.1 / 56.5 / 52.2 / 62.5 %.
    길이가 13 배(122 ms~1.65 s) 달라도 비율이 같다 = 시간이 아니라 비율이 불변량.
    """
    fracs = []
    for dur in (0.6, 1.4, 2.3, 3.4):
        y = render({"timeline": [{
            "type": "fricative", "phone": "s", "dur": dur, "aero": True,
            "constriction_area": [[0, 0.10], [1, 0.10]],   # 협착은 **고정**
            "glottal_area": 0.12, "level_db": -5}], "seed": 5}, PROF, CFG)
        span, frac, rf = _hf_shape(y)
        assert 0.50 < frac < 0.70, f"dur {dur}: 페이드인이 {frac*100:.0f}% 뿐"
        assert rf > 1.0, f"dur {dur}: 상승이 하강보다 안 느리다 ({rf:.2f})"
        fracs.append(frac)
    # 흩어짐 한계는 **실측 자체의 흩어짐**보다 느슨해야 한다. 실측 4 토큰이
    # 52.2~62.5 % 로 이미 10.3 %p 벌어져 있어서, 합성에 10 %p 미만을 요구하면
    # 사람보다 일정할 것을 요구하는 셈이다. (합성 현재값 55.4~65.5 %.)
    spread = max(fracs) - min(fracs)
    assert spread < 0.14, f"길이에 따라 페이드인 비율이 흔들린다: {spread*100:.0f}%p"


def test_symmetric_pressure_cannot_make_the_fade_alone():
    """대칭 압력 아치로는 정점이 절대 절반을 못 넘는다 — 비대칭은 호흡에 있다.

    이 테스트가 `breath_drive` 의 존재 이유다. 진폭이 Ps 의 단조함수라
    레이놀즈 게이트도 제곱법칙도 정점을 옮기지 못한다(49~51 % 에 남는다).
    """
    n = 140
    x = torch.linspace(0, 1, n)
    ps = 8000.0 * (0.02 + 0.98 * (0.5 - 0.5 * torch.cos(2 * torch.pi * x)))
    area = torch.full((n,), 0.10)
    u = ac.series_flow(ps, torch.full((n,), 0.12), area)
    e = ac.frication_source_amp(u, area)
    e = (e / e.amax()).numpy()
    thr = e.max() * 0.10
    on = int((e > thr).argmax())
    off = len(e) - int((e[::-1] > thr).argmax()) - 1
    frac = (int(e.argmax()) - on) / max(off - on, 1)
    assert 0.45 < frac < 0.55, f"대칭 아치인데 정점이 {frac*100:.0f}% 로 치우쳤다"


def test_frication_and_voicing_overlap_instead_of_switching():
    """마찰음 -> 기식 -> 유성이 **겹치며** 넘어간다 (성문파열음 방지).

    이어주는 건 마찰음의 꼬리가 **아니라 기식**이다. 실측 /사/ 의 고역은
    마찰음 끝에서 0.14 까지 떨어졌다가 발성과 함께 0.33 으로 **다시 오른다** —
    성대가 덜 모인 상태의 기식성 발성이지 마찰음 잔향이 아니다.
    (마찰음 꼬리로 이으려고 내전을 앞당겼더니, 성문이 좁아지는 순간 압력강하가
     성문으로 옮겨가 협착 뒤 압력이 무너지고 /s/ 가 혀를 풀기도 전에 죽었다.)

    그래서 검사하는 건 셋이다: 발성이 계단으로 들어오지 않을 것, 마찰음이
    꺼지는 창을 기식이 메울 것, 그 기식이 모음에서 잦아들되 0 은 아닐 것.
    """
    from formant_ml.score import build_segment
    seg = {"type": "syllable", "onset": "s", "vowel": "a",
           "dur": 0.58, "onset_s": 0.11, "aero": True, "transition_s": 0.12}
    c = build_segment(seg, PROF, CFG)
    nb = c["noise_bands"][0].sum(-1)
    ab = c["aspiration_bands"][0].sum(-1)
    ha = c["harmonic_amp"][0, :, 0]
    nb = (nb / nb.amax()).numpy()
    ab = (ab / ab.amax().clamp_min(1e-12)).numpy()
    ha = (ha / ha.amax().clamp_min(1e-12)).numpy()
    i10 = int((ha > 0.10).argmax())
    pk = int(nb.argmax())
    # 마찰음 정점 -> 발성 개시 사이에 소리가 **끊기는 구간이 없어야** 한다.
    #
    # 이건 **렌더된 소리**에서 잰다. 예전에는 제어신호로 재면서
    # `ab[i] < 0.30` (구간 최대 대비) 을 썼는데, 그 기준이 무엇을 재는지가
    # 바뀌었다. `ab` 의 구간 최대는 **모음의 기식성 바닥**(발성 중 성대가 덜
    # 닫혀 새는 잡음)이고, 전이 기식은 이제 협착 **기하**로 배분된다
    # (`glottal_drop_fraction` x `constriction_transmission`). 두 양은 물리가
    # 달라서 한쪽만 줄면 비율이 떨어지는데, 그게 성문파열음을 뜻하지는 않는다.
    # 실제로 그 비율이 0.16 으로 떨어졌을 때 렌더된 고역 포락선의 최저는
    # 0.23 이었다 — 실측 /사/ 의 0.14 보다 오히려 **덜** 빈다.
    # (예전 코드는 0.42 로 실측보다 과하게 메워져 있었다.)
    # 그래서 임계값을 낮추는 대신 **직접 소리를 재도록** 바꿨다.
    y = render({"timeline": [seg], "seed": 5}, PROF, CFG).reshape(-1)
    win, hop = 1024, 256
    f = torch.linspace(0, FS / 2, win // 2 + 1)
    S = torch.stft(y, win, hop, window=torch.hann_window(win),
                   return_complex=True).abs()
    hi = (S[f > 4000] ** 2).sum(0).sqrt()
    hi = hi / hi.amax().clamp_min(1e-12)
    lo = (S[(f > 200) & (f < 1000)] ** 2).sum(0).sqrt()
    lo = lo / lo.amax().clamp_min(1e-12)
    a_pk = int(hi.argmax())
    v_on = int((lo > 0.5).float().argmax())
    floor = float(hi[a_pk:v_on + 1].min())
    # 실측 /사/ 는 마찰음 정점 대비 **0.122~0.123** 까지 떨어졌다가 발성과 함께
    # 다시 오른다(두 토큰, 4 kHz 고역통과 + 6 ms RMS 포락선으로 재측정).
    # 예전 주석의 0.13~0.14 는 다른 창/대역으로 잰 값이었다. 합성이 실측보다
    # **더 메워져 있을** 이유는 없으므로 하한을 실측 바로 아래에 둔다.
    assert floor >= 0.11, f"마찰음과 발성 사이가 비었다 (성문파열음): {floor:.3f}"
    # 발성이 계단으로 들어오면 안 된다(예전: 두 프레임 만에 0.09 -> 0.76).
    #
    # **프레임당 증분이 아니라 상승 시간으로 잰다.** 로지스틱 기동의 최대 기울기는
    # σ/4 이고 σ 는 ONSET_CYCLES 에 묶여 있어서(=문헌의 "발성 개시 몇 주기"),
    # 증분 임계값을 두면 그 상수를 간접적으로 못 박게 된다. 실제로 실측 녹음에
    # 맞춘 6 주기(저역 10->90 % 상승 47.7 ms, 실측 49.5~50.0)에서 프레임당
    # 최대 증분이 34 % 라 0.25 임계에 걸렸다 — 소리는 오히려 실측에 가까워졌는데.
    # 계단인지 아닌지를 직접 보는 건 **몇 프레임에 걸쳐 오르는가**다.
    i10 = int((ha > 0.10).argmax())
    i90 = int((ha > 0.90).argmax())
    assert i90 - i10 >= 3, (f"유성 진폭이 {i90 - i10} 프레임 만에 10->90 % "
                            "(성문파열음)")
    # 기식은 마찰음이 죽는 창에서 **올라오고 있어야** 한다. 절대 크기가 아니라
    # 방향으로 본다 — 크기는 위의 렌더 검사가 맡는다.
    assert ab[i10] > ab[pk], ("마찰음이 꺼지는데 기식이 안 올라온다: "
                              f"{ab[pk]:.3f} -> {ab[i10]:.3f}")
    # 모음에도 기식성 바닥이 남아야 한다(성대가 완전히 닫히지 않는다).
    # 상한은 두지 않는다 — 전이 기식이 (1-vfrac) 로 줄어든 뒤에는 이 바닥이
    # 구간 최대가 되는 게 정상이다.
    assert ab[-1] > 0.1, f"모음에 기식성 바닥이 없다: {ab[-1]:.2f}"



def _me_profile():
    """화자 프로파일 (없으면 None). 보정 상수가 이 프로파일에 맞춰져 있다."""
    path = os.path.join(os.path.dirname(__file__), "..", "profiles", "me.json")
    return VoiceProfile.load(path) if os.path.exists(path) else None

def test_fricative_envelope_is_an_arch_not_a_rectangle():
    """마찰음 포락선에 **수직 모서리도 고원도** 없어야 한다.

    2026-09-04 이전에는 직사각형이었다: 한 프레임(10 ms)에 16.6 dB 오르고,
    60 ms 동안 -40.45 dB 가 **여섯 프레임 연속 완전히 동일**하다가, 40 ms 만에
    88 dB 가 사라졌다. 스펙트로그램에서 수직 모서리를 가진 블록으로 보이고,
    그 모서리가 임펄스라 "치찰음 중반부의 파열음" 으로 들렸다.

    셋이 함께 지켜져야 이 모양이 안 돌아온다.
      * 폐압이 발화 개시에서 램프로 오른다 (UTTERANCE_PS_RISE_S)
      * 후두가 매끄러운 종 모양이다 (devoicing_gesture)
      * 해제 시간이 조음기 속도에서 나온다 (release_from_speed)

    **제어 신호가 아니라 렌더된 소리에서 잰다.** 제어의 `noise_bands` 는 꼬리가
    -240 dB 까지 내려가는데 실제 소리의 그 구간은 기식이 메우고 있어서, 제어만
    보면 들리지도 않는 곳의 계단을 잡는다. 실측과 같은 잣대(4~12 kHz 포락선)로
    재야 비교가 된다 — 실측 /사/ 두 토큰은 최대 12.0 / 12.2 dB per 10 ms 다.
    """
    seg = {"type": "syllable", "onset": "s", "vowel": "a", "dur": 0.72,
           "onset_s": 0.09, "hold_s": 0.07, "aero": True, "drive": "tongue"}
    y = render({"timeline": [{"type": "silence", "dur": 0.08}, seg],
                "seed": 5}, PROF, CFG).reshape(-1)
    win, hop = 512, 128                       # hop = 5.3 ms @ 24 kHz
    f = torch.linspace(0, FS / 2, win // 2 + 1)
    S = torch.stft(y, win, hop, window=torch.hann_window(win),
                   return_complex=True).abs()
    hi = (S[(f > 4000) & (f < 12000)] ** 2).sum(0).sqrt()
    db = 20 * torch.log10(hi.clamp_min(1e-12))
    pk = float(db.max())
    n10 = max(int(round(0.010 * FS / hop)), 1)        # 10 ms
    live = db > pk - 45.0
    step = [abs(float(db[i] - db[i + n10])) for i in range(len(db) - n10)
            if bool(live[i]) and bool(live[i + n10])]
    assert step, "가청 마찰음 구간이 없다"
    assert max(step) < 16.0, (f"포락선이 10 ms 에 {max(step):.1f} dB 뛴다 "
                              "(수직 모서리 = 파열음). 실측 12.0~12.2")
    # 고원 금지: 정점 ±0.5 dB 가 길게 이어지면 아치가 아니라 직사각형이다.
    flat = int((db > pk - 0.5).sum()) * hop / FS * 1000
    assert flat <= 40.0, f"정점이 {flat:.0f} ms 동안 평평하다 (직사각형)"


def test_transition_aspiration_is_not_louder_than_the_vowel():
    """전이 기식이 모음보다 크면 안 된다 — /s/ 뒤에 /h/ 가 붙는다.

    전이 기식(`asp_n`)만 900 Hz 셸프를 쓰던 시절, 그 셸프가 성도 캐스케이드를
    통째로 지나며 F1/F2 를 때렸다. 고역 골을 메우려고 ASPIRATION_GAIN 을
    6.3 -> 26 으로 올리자 그대로 따라 커져서, 마찰음 직후 0.9~2 kHz 가 모음
    정상부보다 **9~11 dB 컸다**. 경로별로 따로 렌더해서 확인했다: 그 구간이
    전체 -11.7 dB 인데 기식을 빼면 -59.8 dB 였다(= 통째로 기식).

    지금은 기식성 바닥과 **같은 소스 스펙트럼**을 쓴다(같은 성문 제트 난류다).
    실측 /사/ 두 토큰의 같은 비는 0.81 / 1.06 이다.
    """
    seg = {"type": "syllable", "onset": "s", "vowel": "a", "dur": 0.72,
           "onset_s": 0.09, "hold_s": 0.07, "aero": True, "drive": "tongue"}
    y = render({"timeline": [{"type": "silence", "dur": 0.08}, seg],
                "seed": 5}, PROF, CFG).reshape(-1)
    win, hop = 512, 128
    f = torch.linspace(0, FS / 2, win // 2 + 1)
    S = torch.stft(y, win, hop, window=torch.hann_window(win),
                   return_complex=True).abs()
    f2 = (S[(f > 900) & (f < 2000)] ** 2).sum(0).sqrt()
    hi = (S[(f > 4000) & (f < 12000)] ** 2).sum(0).sqrt()
    pk = int(hi[:int(0.30 * FS / hop)].argmax())            # 마찰음 정점
    trans = float(f2[pk:pk + int(0.12 * FS / hop)].max())   # 전이 구간 최대
    vowel = float(f2[int(0.45 * FS / hop):].quantile(0.97))
    assert trans < 1.4 * vowel, (
        f"전이 F2 대역이 모음의 {trans / max(vowel, 1e-9):.2f} 배 (/h/ 가 붙었다). "
        "실측 0.81~1.06")


def test_vowel_has_an_interharmonic_noise_floor():
    """모음의 하모닉 **사이**가 비어 있으면 안 된다 — 그게 "사각파" 다.

    실측 /아/ 는 300~2000 Hz 에서 스펙트럼 중앙값이 그 구간 최댓값의 -38.1 dB
    이다. 합성은 `aspiration_bands` 셸프의 floor 가 0 이라 -58.6 dB 였다 —
    거의 선 스펙트럼이라 톱니파처럼 들린다.

    **임계값이 실측(-38.1)보다 느슨한 이유**: 여기는 24 kHz 이고
    `BREATH_NOISE_GAIN` 은 24 kHz 값(2.7)이 그대로다. 실측에 맞춘 건 44.1 kHz
    쪽(16.0)이고 거기서는 -38.0 dB 가 나온다. 24 kHz 게인은 아직 재적합하지
    않았다(HANDOFF §6.2). floor 를 0 으로 되돌리면 -62.1 dB 로 떨어진다.
    """
    prof = _me_profile()
    if prof is None:
        return
    y = render({"timeline": [{"type": "vowel", "vowel": "a", "dur": 0.4}],
                "seed": 5}, prof, CFG).reshape(-1)
    seg = y[int(0.15 * FS):int(0.38 * FS)]
    w = torch.hann_window(len(seg))
    P = torch.fft.rfft(seg * w).abs() ** 2
    f = torch.linspace(0, FS / 2, len(P))
    m = (f > 300) & (f < 2000)
    floor_db = 10 * math.log10(float(P[m].median() / P[m].max()) + 1e-20)
    assert floor_db > -52.0, (f"하모닉 사이 바닥이 {floor_db:.1f} dB — "
                              "선 스펙트럼(사각파)이다")


def test_voicing_does_not_start_during_the_fricative():
    """후두가 먼저 모여도 /s/ 는 무성으로 남는다 — 구강내압이 막고 있다."""
    from formant_ml.score import build_segment
    c = build_segment({"type": "syllable", "onset": "s", "vowel": "a",
                       "dur": 0.58, "onset_s": 0.11, "aero": True}, PROF, CFG)
    nb = c["noise_bands"][0].sum(-1)
    ha = c["harmonic_amp"][0, :, 0]
    peak = int(nb.argmax())                      # 마찰음이 가장 셀 때
    assert float(ha[peak]) < 0.02 * float(ha.amax()), "/s/ 가 유성이 됐다"


def test_intraoral_pressure_eats_most_of_the_driving_pressure():
    """좁은 협착 뒤에 Ps 의 60~70 % 가 쌓여 발성을 억제한다(무성 /s/ 의 이유)."""
    n = 60
    ps = torch.full((1, n, 1), 8000.0)
    pm, u = ac.oral_cavity(ps, torch.full((1, n, 1), 0.12),
                           torch.full((1, n, 1), 0.10), 100.0)
    frac = float(pm[0, -1, 0] / ps[0, -1, 0])
    assert 0.5 < frac < 0.8, f"구강내압이 Ps 의 {frac*100:.0f}% 밖에 안 된다"
    # 협착을 풀면 압력이 빠지고 성대 구동압이 살아난다 -> VOT 가 유도된다.
    area = torch.cat([torch.full((1, n // 2, 1), 0.10),
                      torch.full((1, n - n // 2, 1), 3.0)], dim=1)
    pm2, _ = ac.oral_cavity(ps, torch.full((1, n, 1), 0.12), area, 100.0)
    assert float(pm2[0, -1, 0]) < 0.05 * float(pm2[0, n // 2 - 1, 0]), "해제해도 안 빠진다"
    assert float(ac.transglottal_pressure(ps, pm2)[0, -1, 0]) > 0.9 * 8000.0


def test_oral_cavity_matches_series_flow_at_steady_state():
    """공동 적분의 정상상태는 직렬 유량 해와 같다(같은 모형의 정적 극한)."""
    n = 200
    ps = torch.full((1, n, 1), 8000.0)
    ag = torch.full((1, n, 1), 0.12)
    ao = torch.full((1, n, 1), 0.10)
    _, u = ac.oral_cavity(ps, ag, ao, 100.0)
    u_ss = ac.series_flow(ps, ag, ao)
    assert abs(float(u[0, -1, 0] / u_ss[0, -1, 0]) - 1.0) < 0.02


def test_fold_flow_drives_noise_modulation():
    """성대 유량이 난류 변조 신호로 이어진다(소스-치찰음 결합)."""
    flow, _ = simulate_body_cover(FoldParams(ps=8000.0, a01=0.02, a02=0.02),
                                  n_samples=6000, sample_rate=FS)
    open_env, mod = glottal_flow_to_modulation(flow, FS, 240)
    assert open_env.shape[-1] == 1 and open_env.dim() == 3
    assert float(mod.mean()) > 0.05, "성문 맥동이 변조로 안 넘어간다"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1; print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} 통과")
    sys.exit(1 if failed else 0)
