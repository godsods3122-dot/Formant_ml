"""공기음향 난류 소스 + 성대-난류 결합 검증 (문헌 기반).

  PYTHONPATH=src python3 tests/test_aeroacoustic.py

근거: Stevens(1971) 임계 레이놀즈수·압력강하 소스, Story&Titze(1995) body-cover,
Titze(1988) mucosal wave, Jackson&Shadle(2000) 성문동기 마찰음 변조.
"""
from __future__ import annotations

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


def test_centroid_descends_from_constriction_area_alone():
    """치찰음 무게중심이 협착 면적 궤적만으로 내려간다(손 곡선 없이).

    협착이 열리며 입자속도가 떨어져 무게중심이 내려간다(Stevens 1971).
    """
    from formant_ml.score import build_segment
    prof = PROF
    sa = {"type": "syllable", "onset": "s", "vowel": "a", "dur": 0.5,
          "aero": True, "onset_s": 0.14,
          "constriction_area": [[0, 0.13], [0.2, 0.22], [0.26, 1.2],
                                [0.34, 3.0], [1, 3.0]]}
    y = render({"timeline": [sa], "seed": 5}, prof, CFG).reshape(-1)
    f = torch.linspace(0, FS / 2, 361)

    def cen(a, b, win=720):
        cs = []
        for i in range(int(a * FS), int(b * FS) - win, win // 2):
            S = torch.fft.rfft(y[i:i + win] * torch.hann_window(win)).abs()
            m = f > 1500
            cs.append(float((f[m] * S[m] ** 2).sum() / (S[m] ** 2).sum().clamp_min(1e-9)))
        return cs
    # **마찰음이 살아 있는 구간에서** 잰다. 고정 창(0~160 ms)으로 재면 뒤쪽
    # 모음 프레임까지 섞여 들어가 하강 폭이 부풀려진다.
    nbv = build_segment(sa, prof, CFG)["noise_bands"][0].sum(-1)
    nbv = (nbv / nbv.amax()).numpy()
    alive = [i for i, v in enumerate(nbv) if v > 0.10]
    c = cen(alive[0] * 0.01, (alive[-1] + 1) * 0.01)
    # 기준 150 Hz. 예전엔 800 이었는데 그 큰 하강은 상당 부분 **성문이 마찰음
    # 중간에 닫히던** 데서 왔다(그때 유속이 급락한다). 그건 /s/ 의 물리가
    # 아니라서 없앴다(성문은 /s/ 내내 벌어져 있다). 남은 하강은 협착이 열리며
    # 생기는 것뿐이고, 레이놀즈 게이트가 마찰음을 끄기 전까지만 보이므로 작다.
    assert c[0] - c[-1] > 150, f"무게중심이 안 내려간다: {c[0]:.0f}->{c[-1]:.0f}"


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
    # 실측 /사/ 는 0.13~0.14 까지 떨어졌다가 발성과 함께 다시 오른다.
    assert floor >= 0.13, f"마찰음과 발성 사이가 비었다 (성문파열음): {floor:.3f}"
    # 발성이 계단으로 들어오면 안 된다(예전: 두 프레임 만에 0.09 -> 0.76).
    jump = max(ha[i + 1] - ha[i] for i in range(len(ha) - 1))
    assert jump < 0.25, f"유성 진폭이 한 프레임에 {jump*100:.0f}% 뛴다 (성문파열음)"
    # 기식은 마찰음이 죽는 창에서 **올라오고 있어야** 한다. 절대 크기가 아니라
    # 방향으로 본다 — 크기는 위의 렌더 검사가 맡는다.
    assert ab[i10] > ab[pk], ("마찰음이 꺼지는데 기식이 안 올라온다: "
                              f"{ab[pk]:.3f} -> {ab[i10]:.3f}")
    # 모음에도 기식성 바닥이 남아야 한다(성대가 완전히 닫히지 않는다).
    # 상한은 두지 않는다 — 전이 기식이 (1-vfrac) 로 줄어든 뒤에는 이 바닥이
    # 구간 최대가 되는 게 정상이다.
    assert ab[-1] > 0.1, f"모음에 기식성 바닥이 없다: {ab[-1]:.2f}"


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
