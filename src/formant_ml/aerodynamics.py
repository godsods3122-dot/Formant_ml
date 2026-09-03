"""공기역학: 성문하압과 내전이 목소리를 만든다.

왜 필요한가
-----------
발성 시작은 입 모양만 바뀌는 사건이 아니다. `/사/` 처럼 기식(마찰)에서 유성음으로
넘어갈 때 실제로 일어나는 일은:

1. 폐에서 밀어 올린 기류가 아직 열린 성문을 지나며 **난류 소음**(기식)을 낸다.
2. 성대가 내전(adduction)하면서 성문 틈이 좁아진다.
3. 성문하압 `Ps` 가 **발성 역치압(PTP)** 을 넘는 순간 자가진동이 시작된다.
   PTP 는 내전이 약할수록 높다 — 벌어져 있으면 더 세게 불어야 떤다.
4. 진동이 시작된 뒤에도 `Ps` 가 계속 오르면서 **세기·F0·성문파 형상**이 함께
   변한다. 이 셋은 독립된 손잡이가 아니라 하나의 압력에서 같이 따라 나온다.

그래서 세기만 램프로 올리면(예전 구현) 시작 부분이 '녹음을 페이드인한 것' 처럼
들린다. F0 가 안 따라 오르고, 성문파가 기식에서 모달로 안 바뀌고, 기식 소음이
성대 진동과 무관하게 사라진다.

여기서는 (Ps, 내전) 두 개를 받아 네 가지를 **일관되게** 만들어 낸다.
Titze 의 발성 역치압 이론을 저차원으로 요약한 것이다(정량적 재현이 아니라
결합 구조를 맞추는 것이 목적이다).
"""
from __future__ import annotations

import torch

from .utils import ramp

# 대략적인 결합 계수. 정밀한 값은 화자마다 다르고, 학습으로 조정할 수 있다.
PTP_AT_FULL_ADDUCTION = 0.35     # 완전 내전 시 발성 역치압 (정규화 Ps, 1.0 = 보통 발화)
F0_PER_PRESSURE = 0.18           # Ps 1.0 변화당 F0 비율 변화 (약 2~5 Hz/cmH2O)
AMP_EXPONENT = 1.4               # 세기는 구동압에 대해 초선형으로 는다

# 협착부 난류 소음의 세기(파워)는 부피속도 U 에 대해 초선형으로 는다.
# Stevens(1971), Shadle(1990)의 마찰음 소스 모형: 난류원 파워 ∝ U^n, n≈2.5~3.
# (제트 운동에너지 ~ρU², 그중 난류로 변환되는 비율까지 곱해져 초선형이 된다.)
# 진폭은 파워의 제곱근이므로 U^(n/2) 로 는다. 즉 유량이 fade-in 으로 서서히
# 오르면 음량은 그보다 **더 가파르게** 오른다 — 이것이 마찰음이 스위치처럼 '탁'
# 켜지는 대신 부드럽게 들고 나는 물리적 이유다. 게이트로 진폭을 직접 램프하면
# (예전 구현: env=ones 로 상수, 또는 선형 램프) 그 곡률이 사라져 '녹음을
# 페이드한 것' 처럼 들린다.
FRICATION_FLOW_EXPONENT = 2.5


def _smoothstep(u: torch.Tensor) -> torch.Tensor:
    """0..1 구간의 부드러운 S 곡선 (raised cosine). 모서리가 없어 클릭이 안 난다."""
    return 0.5 - 0.5 * torch.cos(torch.pi * u.clamp(0.0, 1.0))


def frication_flow(t: int, frame_rate: float, fade_in: float = 0.03,
                   fade_out: float = 0.04, plateau: float = 1.0,
                   floor: float = 0.0, shape=None,
                   device=None, dtype=torch.float32) -> torch.Tensor:
    """협착부 부피속도 U 의 시간 포락선 (1, T, 1), 0~1.

    협착이 형성되며 기류가 붙고(fade_in), 해제되며 빠지는(fade_out) 과정을
    raised-cosine 로 근사한다. `fade_in`/`fade_out` 은 **초 단위** 상승/하강 시간.
    마찰음이 짧아 두 페이드가 겹치면 비례해서 줄인다.

    `shape=[(위치0~1, 값), ...]` 를 주면 그 곡선을 그대로 유량 포락선으로 쓴다
    (예: 유량이 두 번 부풀었다 꺼지는 반복 마찰음).

    이 값 자체가 음량이 아니다 — `flow_to_noise_amp` 로 진폭 배율로 바꿔 쓴다.
    """
    if shape is not None:
        return ramp(t, [(float(p), float(v)) for p, v in shape], device=device)
    x = torch.arange(t, device=device, dtype=dtype) / max(frame_rate, 1e-6)  # 초
    dur = max((t - 1) / max(frame_rate, 1e-6), 1e-6)
    fi, fo = max(float(fade_in), 0.0), max(float(fade_out), 0.0)
    if fi + fo > dur:                       # 짧은 마찰음: 페이드가 안 들어가면 축소
        s = dur / (fi + fo)
        fi, fo = fi * s, fo * s
    up = _smoothstep(x / fi) if fi > 1e-6 else torch.ones_like(x)
    dn = _smoothstep((dur - x) / fo) if fo > 1e-6 else torch.ones_like(x)
    env = floor + (plateau - floor) * (up * dn)
    return env.reshape(1, t, 1)


def flow_to_noise_amp(flow: torch.Tensor,
                      exponent: float = FRICATION_FLOW_EXPONENT) -> torch.Tensor:
    """부피속도 포락선 -> 난류 소음 **진폭** 배율.

    파워 ∝ U^exponent, 진폭 ∝ U^(exponent/2). 유량을 그대로 음량으로 쓰지 않고
    이 초선형 곡률을 거치게 해서, fade-in/out 이 물리적인 마찰음 세기 곡선을
    따르도록 한다.
    """
    return flow.clamp_min(0.0).pow(0.5 * max(float(exponent), 0.0))


def phonation_threshold(adduction: torch.Tensor) -> torch.Tensor:
    """발성 역치압. 내전이 약하면 급격히 커진다 (벌어진 성문은 잘 안 떤다)."""
    a = adduction.clamp(0.02, 1.0)
    return PTP_AT_FULL_ADDUCTION / a


def phonation(pressure: torch.Tensor, adduction: torch.Tensor,
              f0_base: torch.Tensor, rd_modal: float = 1.0,
              rd_breathy: float = 2.4, amp_scale: float = 1.0) -> dict:
    """(Ps, 내전) -> 서로 맞물린 제어값들.

    반환 dict: harmonic_amp, f0, rd, aspiration (0~1 성문 누출 기류)
    모두 입력과 같은 (B, T, 1) 모양이다.
    """
    ps = pressure.clamp_min(0.0)
    ptp = phonation_threshold(adduction)
    drive = (ps - ptp).clamp_min(0.0)                 # 역치를 넘은 만큼만 떤다

    amp = amp_scale * drive.pow(AMP_EXPONENT)
    # 압력이 오르면 성대가 늘어나 F0 가 따라 오른다
    f0 = f0_base * (1.0 + F0_PER_PRESSURE * (ps - 1.0))
    # 구동이 약할 때는 성문이 완전히 닫히지 않아 기식적(Rd 큼)
    rd = rd_modal + (rd_breathy - rd_modal) * torch.exp(-4.0 * drive)
    # 성문 누출: 내전이 덜 됐고 압력이 있으면 난류 소음이 난다
    aspiration = (1.0 - adduction.clamp(0.0, 1.0)) * ps
    return {"harmonic_amp": amp, "f0": f0, "rd": rd, "aspiration": aspiration}


def apply(controls: dict, pressure: torch.Tensor, adduction: torch.Tensor,
          rd_modal: float = 1.0, rd_breathy: float = 2.4,
          noise_scale: float = 1.0, route_noise: bool = True) -> torch.Tensor:
    """제어 dict 를 공기역학적으로 일관되게 덮어쓴다 (제자리).

    `controls` 는 `gestures.base` 가 만든 프레임률 dict 이며, f0 는 이미 들어 있는
    값을 기준선(f0_base)으로 삼는다.
    """
    out = phonation(pressure, adduction, controls["f0"], rd_modal, rd_breathy)
    controls["harmonic_amp"] = out["harmonic_amp"]
    controls["f0"] = out["f0"]
    controls["rd"] = out["rd"]
    asp = out["aspiration"]
    if route_noise:
        # 기식 소음은 성문에서 나므로 성도 전체를 통과한다.
        # (마찰음이 동시에 나는 구간에서는 호출부가 직접 섞는다 — 노이즈 경로가
        #  하나뿐이라 구강 협착과 성문 두 소스를 동시에 낼 수는 없다.
        #  두 경로를 따로 두는 것은 남은 과제다.)
        controls["noise_bands"] = controls["noise_bands"] * (1.0 + noise_scale * 8.0 * asp)
        controls["noise_entry"] = torch.zeros_like(controls["noise_entry"])
    controls["noise_am"] = (controls["noise_am"] + 0.5 * asp).clamp(0.0, 1.0)
    return asp
