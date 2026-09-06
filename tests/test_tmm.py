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


def test_matches_an_independent_ode_integration():
    """**독립 기준과 대조한다** — 전송선 ODE 를 RK4 로 직접 적분한 값.

        dP/dx = −Z'·U,   dU/dx = −Y'·P

    전달행렬은 이 ODE 의 (구간별 상수 계수에 대한) 해석해다. 그러니 잘게
    적분한 값과 **정확히** 같아야 한다. 이 검사가 필요한 이유는, 예전 판이
    손실 k 와 무손실 특성임피던스 ρc/A 를 섞어 써서 첨두 −40 dB 이내의
    들리는 대역에서 최대 9.5 dB 틀렸는데도 다른 검사(균일관·섭동·대역폭)를
    전부 통과했기 때문이다. **고전 결과만으로는 이 종류의 오류가 안 잡힌다.**
    """
    from scipy.special import j1, struve
    from formant_ml.dsp.tmm import (GAMMA, MU, PRANDTL, RHO, WALL_MASS,
                                    WALL_RESISTANCE)

    length, n_freq, sub = 17.5, 257, 120
    f = np.linspace(0.0, FS / 2, n_freq)
    f[0] = 1e-6
    w = 2 * np.pi * f

    def reference(area):
        n = len(area)
        seg = length / n

        def zy(A):
            a = math.sqrt(A / math.pi)
            circ = 2 * math.pi * a
            zp = (circ / A ** 2) * np.sqrt(w * RHO * MU / 2.0) + 1j * w * RHO / A
            yp = ((circ / (RHO * SOUND_SPEED ** 2)) * (GAMMA - 1.0)
                  * np.sqrt(w * MU / (2 * RHO * PRANDTL))
                  + circ / (WALL_RESISTANCE + 1j * w * WALL_MASS)
                  + 1j * w * A / (RHO * SOUND_SPEED ** 2))
            return zp, yp

        a_l = area[-1]
        x = 2 * (w / SOUND_SPEED) * math.sqrt(a_l / math.pi)
        z_norm = (1 - 2 * j1(x) / x) + 1j * (2 * struve(1, x) / x)
        p = z_norm * (RHO * SOUND_SPEED / a_l)
        u = np.ones_like(p)
        for i in range(n - 1, -1, -1):
            zp, yp = zy(area[i])
            h = seg / sub
            for _ in range(sub):                 # 입술 -> 성문 (부호 반전)
                def der(pp, uu):
                    return zp * uu, yp * pp
                k1 = der(p, u)
                k2 = der(p + h / 2 * k1[0], u + h / 2 * k1[1])
                k3 = der(p + h / 2 * k2[0], u + h / 2 * k2[1])
                k4 = der(p + h * k3[0], u + h * k3[1])
                p = p + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
                u = u + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        return (1.0 / u) * (1j * w)

    rng = np.random.default_rng(0)
    cases = {
        "균일관": np.full(24, 3.0),
        "모음형": np.array([1.0] * 8 + [0.4] * 4 + [3.0] * 6 + [6.0] * 6),
        "무작위": rng.uniform(0.15, 8.0, 32),
    }
    for name, area in cases.items():
        ref = reference(area)
        got = tract_transfer(torch.tensor(area, dtype=torch.float64).reshape(1, 1, -1),
                             FS, n_freq)[0, 0].numpy()
        m_ref = 20 * np.log10(np.abs(ref) + 1e-30)
        audible = (m_ref - m_ref.max()) > -40          # 들리는 대역만 본다
        err = np.abs(20 * np.log10(np.abs(got) + 1e-30) - m_ref)[audible].max()
        assert err < 0.05, f"{name}: ODE 적분과 {err:.3f} dB 차이"


def test_series_argument_stays_in_the_validated_range():
    """cos/sinc 정급수의 인자 |z| 가 항 수로 감당되는 범위 안에 있어야 한다.

    |z| = |k·(단면 길이)|² 다. 단면을 잘게 쪼개거나 손실을 키우면 커진다.
    12 항 급수는 |z| ≤ 25 까지도 항 크기가 1e-10 이므로 넉넉하지만, 그
    가정이 조용히 깨지지 않도록 상한을 검사로 박아 둔다.
    """
    from formant_ml.dsp.tmm import _series_shunt
    f = torch.linspace(0.0, FS / 2, 1025, dtype=torch.float64)
    worst = 0.0
    for n_sections in (20, 40, 64):
        seg = 17.5 / n_sections
        for a in (0.01, 0.1, 1.0, 3.0, 8.0):
            zp, yp = _series_shunt(f, torch.full((1, 1, 1), a, dtype=torch.float64),
                                   TubeLosses(True, True))
            worst = max(worst, float((-(zp * seg) * (yp * seg)).abs().max()))
    assert worst < 25.0, f"|z| 최대 {worst:.2f} — 급수 항 수를 늘려야 한다"


def test_works_for_both_float_precisions():
    """면적의 dtype 이 float32 든 float64 든 돌아가야 한다.

    `freq` 는 `freq_grid` 가 만들고 `area` 는 호출부가 준다. 둘의 dtype 이
    다르면 `torch.complex` 가 그대로 터진다 — 역산 실험에서 numpy 기본값
    (float64) 면적을 넣었다가 실제로 겪었다. 면적이 최적화 변수이므로
    dtype 은 면적 쪽으로 통일한다.
    """
    for dtype, want in ((torch.float32, torch.complex64),
                        (torch.float64, torch.complex128)):
        h = tract_transfer(torch.full((1, 1, 24), 3.0, dtype=dtype), FS, 257)
        assert h.dtype == want
        assert torch.isfinite(h.abs()).all()
    # 두 정밀도의 결과가 서로 맞아야 한다 (수치오차 범위)
    a32 = torch.full((1, 1, 24), 3.0, dtype=torch.float32)
    a64 = torch.full((1, 1, 24), 3.0, dtype=torch.float64)
    d = (20 * torch.log10(tract_transfer(a32, FS, 257).abs() + 1e-30)
         - 20 * torch.log10(tract_transfer(a64, FS, 257).abs() + 1e-30))
    assert float(d.abs().max()) < 0.01, f"정밀도에 따라 {float(d.abs().max()):.3f} dB 다르다"


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
