"""폐 — 호흡 예산과 성문하압의 출처.

사용자: *"호흡을 이 대화에 얼마나 써야 할지에 대한 policy 를 짜고, 어떤 때는 일부러 호흡을
다 쓰게 한다던지, 어떤 때는 발화 전체를 균일한 호흡으로 한다든지 해서 전략으로 나가야
하거든? 그래서 폐에 대한 고려까지 해야 해."*

지금까지 `p_sub` 는 프레임마다 자유로운 0~20 cmH₂O 짜리 숫자였다. 폐용적도, 탄성 반동도,
"이 발화에 공기가 얼마나 남았는가" 도 없었다. 발화 끝에서 압력이 떨어지는 것조차 물리가
아니라 적합이 우연히 맞춘 것이다.

물리
----
폐포압은 **수동 반동 + 능동 근육**이다 (Rahn 도표):

    P_alv(V, m) = P_rel(V) + P_mus(m)

* `P_rel` 은 폐용적만의 함수다. 기능적 잔기용량(FRC)에서 0, 총폐용량(TLC)에서 +30~40,
  잔기용량(RV)에서 −20 cmH₂O 쯤이다.
* 그래서 **발화 초반(용적이 높을 때)에는 반동이 필요 이상으로 세서 들숨근으로 눌러 줘야
  하고, 후반에는 날숨근으로 밀어야 한다.** 그 전환이 음색에 남는다 — 후반부가 기식성·
  프라이로 가고 F0 가 내려가는 것(declination)이 여기서 온다.

용적은 유량으로 준다:  dV/dt = −U.

왜 ASMR 에서 중요한가
---------------------
속삭임은 성문이 열려 있어 **공기를 훨씬 빨리 쓴다**. 같은 폐활량으로 말할 수 있는 시간이
유성음의 1/2~1/3 이다. 호흡 정책(한 호흡에 얼마를 쓸지, 일부러 바닥까지 쓸지)은 그 차이
위에서 짜야 한다.

검증
----
* `P_rel`: FRC 에서 0, TLC 에서 +30 근처, RV 에서 −20 근처.
* 보통 발화는 폐활량의 35~60 % 구간을 쓴다.
* 유성 발화 유량 150~250 cm³/s, 속삭임 350~500 — 한 호흡의 지속이 그만큼 짧아야 한다.
"""
from __future__ import annotations

import numpy as np

CM_H2O = 98.0665

#: 성인 여성의 전형값 [L].
VC = 3.3            # 폐활량
FRC_FRAC = 0.0      # FRC 를 용적 축의 0 으로 둔다 (VC 대비 몫으로 센다)


def relaxation_pressure(v_frac: float) -> float:
    """폐용적(FRC 위 VC 대비 몫, −0.2 ~ 1.0) -> 수동 반동압 [cmH₂O].

    Rahn 이완압 곡선을 세 점(RV −20, FRC 0, TLC +35)에 맞춘 매끄러운 곡선으로 쓴다.
    가슴우리와 폐의 탄성이 합쳐진 것이라 FRC 근처에서 거의 선형이고 끝에서 가팔라진다.
    """
    v = float(np.clip(v_frac, -0.25, 1.0))
    return 35.0 * v ** 1.6 if v >= 0 else -20.0 * (-v / 0.25) ** 1.3


def alveolar_pressure(v_frac: float, p_mus: float) -> float:
    """폐포압 [Pa]. `p_mus` 는 근육이 더하는 압력 [cmH₂O] (음수면 들숨근이 억제)."""
    return (relaxation_pressure(v_frac) + float(p_mus)) * CM_H2O


def muscle_for_target(v_frac: float, p_target_cm: float) -> float:
    """목표 성문하압을 내기 위해 근육이 내야 할 압력 [cmH₂O].

    발화 초반에는 음수(들숨근이 반동을 **억제**), 후반에는 양수(날숨근이 **밀어냄**).
    그 부호가 바뀌는 자리가 음색이 바뀌는 자리다.
    """
    return float(p_target_cm) - relaxation_pressure(v_frac)


def breath_budget(flow_cm3_s: float, v_start: float = 0.60,
                  v_end: float = 0.20) -> float:
    """주어진 유량으로 `v_start` 에서 `v_end` 까지 쓰는 데 걸리는 시간 [s]."""
    dv_l = (float(v_start) - float(v_end)) * VC
    return dv_l * 1e3 / max(float(flow_cm3_s), 1e-9)


def run(flow_series, fs, v_start=0.60, p_target_cm=6.0):
    """유량 계열을 흘려보내며 용적·근육압·성문하압을 낸다."""
    u = np.asarray(flow_series, float)             # cm³/s
    dv = np.cumsum(u) / fs * 1e-3 / VC             # 누적 소비 (VC 대비)
    v = v_start - dv
    p_rel = np.array([relaxation_pressure(x) for x in v])
    p_mus = p_target_cm - p_rel
    return {"v": v, "p_rel": p_rel, "p_mus": p_mus,
            "p_sub": np.full_like(v, p_target_cm)}


def _check() -> int:
    ok = True
    print("폐 모델 검증")
    print("  1 이완압 곡선 [cmH₂O]")
    for v, want, tol in ((-0.25, -20.0, 3.0), (0.0, 0.0, 0.5), (0.5, 11.6, 6.0),
                         (1.0, 35.0, 3.0)):
        got = relaxation_pressure(v)
        print(f"     용적 {v:+5.2f} VC   반동 {got:+7.2f}   기대 {want:+7.2f}")
        ok &= abs(got - want) <= tol

    print("  2 근육압의 부호 전환 (목표 성문하압 6 cmH₂O)")
    flip = None
    for v in (1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0):
        m = muscle_for_target(v, 6.0)
        tag = "들숨근이 억제" if m < 0 else "날숨근이 밀어냄"
        print(f"     용적 {v:4.2f} VC   근육 {m:+7.2f}   {tag}")
        if flip is None and m > 0:
            flip = v
    print(f"     전환 용적 ≈ {flip:.2f} VC")
    ok &= flip is not None and 0.1 < flip < 0.6

    print("  3 호흡 예산 — 같은 폐활량으로 말할 수 있는 시간")
    for u, lab in ((180.0, "유성 보통"), (300.0, "기식성"), (420.0, "속삭임")):
        t = breath_budget(u)
        print(f"     {lab:8s} 유량 {u:5.0f} cm³/s  ->  {t:5.2f} s")
    t_voice, t_whis = breath_budget(180.0), breath_budget(420.0)
    print(f"     속삭임은 유성의 {t_whis/t_voice:.2f} 배 (문헌 0.3~0.5)")
    ok &= 0.25 < t_whis / t_voice < 0.55
    ok &= 2.0 < t_voice < 12.0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_check())
