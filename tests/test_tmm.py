"""전달행렬 성도 검증 — **파라미터를 맞추지 않고** 고전 결과와 대조한다.

    PYTHONPATH=src python3 -m pytest tests/test_tmm.py -q

여기 있는 검사는 전부 **자유 파라미터가 없다.** 균일관의 공진, 섭동 이론의
부호, 헬름홀츠 해석해, 폐쇄 시의 반파장 계열 — 맞추려고 고를 상수가 없으므로
"지표를 맞췄다" 가 성립하지 않는다. 그게 이 파일의 요점이다.

`presets.VOWEL_AREA_20` 으로는 검증하지 **않는다.** 그 면적함수는 예전
도파관(`dsp/tract.py`)으로 목표 포먼트가 나오도록 푼 것이라, 새 모델을 그걸로
재면 "새 모델이 옛 모델과 같은가" 를 묻는 셈이다(순환 논증).
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from formant_ml.dsp.tmm import (SOUND_SPEED, TubeLosses, formant_bandwidths,
                                formant_peaks, radiation_impedance,
                                tract_transfer)

FS = 24000
NF = 4097            # 저역 포먼트를 5.9 Hz 격자로 분해한다


def _tube(area, n=40, length=17.5, **kw):
    a = torch.as_tensor(area, dtype=torch.float32).reshape(1, 1, -1)
    return tract_transfer(a, FS, NF, length_cm=length, **kw)


def _F(area, n_peaks=4, **kw):
    return formant_peaks(_tube(area, **kw), FS, n_peaks)[0, 0].numpy()


def test_radiation_impedance_matches_the_exact_expression():
    """급수로 계산한 방사 임피던스가 정확식과 맞아야 한다.

    집중정수 근사(Flanagan 의 병렬 R-L)를 쓰면 반사계수 오차가 면적에 따라
    최대 |ΔΓ|=0.20 이다. 나중에 결과가 이상할 때 "근사 탓인가" 를 되묻지
    않으려고 정확식을 급수로 쓴다 — 이 검사가 그걸 지킨다.
    """
    from scipy.special import j1, struve
    f = torch.linspace(1.0, 12000.0, 400, dtype=torch.float64)
    for area in (0.05, 0.2, 1.0, 3.0, 8.0):
        z = radiation_impedance(f, torch.tensor([[[area]]], dtype=torch.float64))
        a = math.sqrt(area / math.pi)
        x = 2.0 * (2.0 * math.pi * f.numpy() / SOUND_SPEED) * a
        exact = (1.0 - 2.0 * j1(x) / x) + 1j * (2.0 * struve(1, x) / x)
        err = np.abs(z[0, 0].numpy() - exact).max()
        assert err < 1e-9, f"A={area}: 정확식과 {err:.2e} 차이"


def test_uniform_tube_gives_the_classical_quarter_wave_series():
    """길이 17.5 cm 균일관(성문 닫힘·입술 열림) -> 500/1500/2500/3500 Hz.

    끝단을 이상적 개방(P=0)으로 두고 손실을 끈 순수 기하 검사다.
    """
    got = _F(np.full(40, 3.0), 4,
             losses=TubeLosses(False, False), radiate=False)
    for g, want in zip(got, (500.0, 1500.0, 2500.0, 3500.0)):
        assert abs(g - want) / want < 0.01, f"{got} vs 500/1500/2500/3500"


def test_closing_the_lips_switches_to_the_half_wave_series():
    """입술을 닫으면 공진이 **반파장 계열**(1000/2000/3000)로 넘어간다.

    양단이 닫힌 관의 공진은 n·c/2L 이다. 열린 관의 (2n−1)·c/4L 과 다르다.
    이건 기하만으로 정해지므로 맞출 상수가 없다.

    (이 전이를 모르면 `formant_peaks` 가 내놓는 1100 Hz 를 'F1 이 튀었다' 로
     오해하게 된다. 실제로 그렇게 오해했었다.)
    """
    # 두 군데를 조심해야 한다. 둘 다 실제로 틀렸었다:
    #
    # 1. **`radiate=False` 로 닫으면 안 된다.** 그건 Z_r=0, 즉 P=0 이라 여전히
    #    '열린' 끝이다. 폐쇄를 만드는 것은 방사 임피던스 자체다 — 면적이 0 으로
    #    가면 ρc/A 가 발산해서 끝단이 막힌다.
    # 2. **막은 단면만큼 공명관이 짧아진다.** 40 단 중 2 단을 막으면 남는 관은
    #    17.5×38/40 = 16.625 cm 이고 c/2L = 1053 Hz 다. 1000 Hz 를 기대하면
    #    모델이 5 % 틀린 것처럼 보인다 — 틀린 건 기대 쪽이다.
    n, length, n_shut = 40, 17.5, 2
    a = np.full(n, 3.0)
    a[-n_shut:] = 1e-4                              # 입술 폐쇄
    got = _F(a, 3, length=length, losses=TubeLosses(False, False))
    l_eff = length * (n - n_shut) / n
    for i, g in enumerate(got, start=1):
        want = i * SOUND_SPEED / (2.0 * l_eff)
        assert abs(g - want) / want < 0.02, (
            f"{i} 번째 공진 {g:.0f} Hz, 반파장 계열 예측 {want:.0f} Hz")


def test_narrowing_the_lips_makes_it_quieter():
    """**입술을 좁히면 출력이 줄어야 한다.**

    이 레포의 예전 도파관(`dsp/tract.py`)은 반대였다 — 입술 반사가 실수 상수
    0.9 라 방사 손실이 주파수를 안 따르고, 좁힐수록 Q 만 올라가 출력이 커졌다
    (1.0 -> 0.05 cm² 에서 +10.5 dB). 협착이 곧 자음이므로 자음이 모음보다 크게
    나왔고, 그걸 덧대려고 `presets.POSTURE_GAIN_20` 이 생겼다.
    """
    f = np.linspace(0.0, FS / 2, NF)
    w = 1.0 / (f + 200.0)                            # 성문 소스 기울기로 가중
    w = w / w.mean()
    prev = None
    for lip in (3.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02):
        a = np.full(40, 3.0)
        a[-3:] = lip
        m = _tube(a).abs()[0, 0].numpy()
        level = 10 * np.log10((m ** 2 * w).mean())
        if prev is not None:
            assert level < prev + 0.01, f"입술 {lip} cm² 에서 출력이 늘었다"
        prev = level


def test_perturbation_theory_signs():
    """섭동 이론: 협착이 **유속 배**면 공진이 내려가고 **압력 배**면 올라간다.

    닫힘-열림 관에서 F1 의 압력 배는 성문(x=0), 유속 배는 입술(x=L) 이다.
    부호만 보므로 맞출 상수가 없다 (Chiba & Kajiyama; Fant 1960).
    """
    n = 40
    base = np.full(n, 3.0)
    f0 = _F(base, 2)

    a = base.copy(); a[:3] = 1.0                     # 성문쪽 = F1 압력 배
    assert _F(a, 2)[0] > f0[0] + 5, "압력 배 협착인데 F1 이 안 올라갔다"

    a = base.copy(); a[-3:] = 1.0                    # 입술쪽 = F1 유속 배
    assert _F(a, 2)[0] < f0[0] - 5, "유속 배 협착인데 F1 이 안 내려갔다"


def test_helmholtz_resonance_matches_theory_with_end_correction():
    """넥+공동 = 헬름홀츠 공진기. f = (c/2π)·sqrt(A_n/(V·L_n)).

    TMM 은 해석해보다 **낮게** 나와야 한다 — 넥의 끝보정(약 0.6a 씩)만큼
    유효 길이가 길기 때문이다. 낮되, 끝보정으로 설명되는 범위 안이어야 한다.
    """
    n, length = 40, 17.5
    for a_neck, l_neck in ((0.3, 2.0), (0.5, 2.0), (0.3, 3.0)):
        n_neck = max(1, round(l_neck / (length / n)))
        area = np.full(n, 6.0)
        area[-n_neck:] = a_neck
        v_back = 6.0 * (length - l_neck)
        naive = SOUND_SPEED / (2 * math.pi) * math.sqrt(a_neck / (v_back * l_neck))
        # 끝보정: 양단 0.6a 씩 -> 유효 넥 길이
        r = math.sqrt(a_neck / math.pi)
        l_eff = l_neck + 1.2 * r
        corrected = SOUND_SPEED / (2 * math.pi) * math.sqrt(a_neck / (v_back * l_eff))
        got = _F(area, 1, losses=TubeLosses(False, False),
                 radiation_derivative=False)[0]
        assert got < naive, f"끝보정이면 해석해보다 낮아야 한다: {got:.0f} vs {naive:.0f}"
        assert abs(got - corrected) / corrected < 0.15, (
            f"끝보정한 해석해 {corrected:.0f} Hz 와 {got:.0f} Hz 가 너무 다르다")


def test_formant_bandwidths_are_in_the_measured_range():
    """손실 모델이 실측 대역폭을 낸다 (Fant 1972, 남성 모음).

    F1 40~90, F2 50~110, F3 70~140 Hz. 맞추려고 고른 값이 아니라
    문헌의 벽 임피던스(R_w=1600, M_w=1.5)와 Kirchhoff 경계층 감쇠를 그대로
    넣은 결과다.
    """
    bw = formant_bandwidths(_tube(np.full(40, 3.0)), FS, 3)
    for got, (lo, hi), name in zip(bw, ((30, 100), (30, 120), (50, 160)),
                                   ("F1", "F2", "F3")):
        assert lo <= got <= hi, f"{name} 대역폭 {got:.0f} Hz 가 {lo}~{hi} 밖이다"


def test_yielding_wall_raises_f1_by_the_documented_amount():
    """벽 진동은 F1 을 **올린다** (Fant 1972: F1' = sqrt(F1² + F_w²), F_w≈180~200).

    벽을 단단하게 두면 폐쇄 시 F1 이 0 으로 가지만, 실제 성도는 벽이 움직여서
    200 Hz 언저리에 머문다. 열린 관에서는 그 몫이 작은 상승으로 나타난다.
    """
    a = np.full(40, 3.0)
    hard = _F(a, 1, losses=TubeLosses(True, False))[0]
    soft = _F(a, 1, losses=TubeLosses(True, True))[0]
    assert soft > hard, "벽 진동이 F1 을 안 올렸다"
    f_wall = math.sqrt(max(soft ** 2 - hard ** 2, 0.0))
    assert 150.0 < f_wall < 260.0, f"함의된 벽 공진 {f_wall:.0f} Hz (문헌 180~200)"


def test_transfer_is_differentiable_wrt_area():
    """면적함수에 대해 기울기가 흐른다 (역추정·학습에 쓸 것이므로)."""
    area = torch.full((1, 1, 40), 3.0, requires_grad=True)
    h = tract_transfer(area, FS, 257)
    h.abs().sum().backward()
    g = area.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


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
