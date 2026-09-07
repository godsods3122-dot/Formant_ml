"""운율 제어층 — 발화 계획(무엇을)과 운율(어떻게)을 분리한다.

왜 따로 빼는가
--------------
LLM 과 붙여 대화형으로 쓰려면, 언어 쪽(무슨 음소를 어떤 순서로)과 운율 쪽
(얼마나 빠르게, 어떤 억양으로, 어디서 숨을 쉬는지)이 **다른 시간 규모**에서
결정된다. 문맥은 문장 하나가 아니라 대화 전체를 본다 — 길게 설명할 때는 호흡이
길어지고, 되묻거나 망설일 때는 속도와 억양이 통째로 바뀐다.

그래서 `ProsodyPlan` 은 물리 파라미터가 아니라 **의도**를 담는다. 작은 JSON 이라
LLM 이 바로 뱉을 수 있고, 여기서 물리 파라미터로 번역된다.

    {"rate": 1.15, "pitch_shift": -1.0, "pitch_range": 1.4,
     "contour": [[0, 0], [0.6, 3], [1, -4]], "declination": -2.0,
     "breath": {"capacity_s": 4.0, "depth": 0.6}}

두 지점에서 적용된다.

* `warp_timeline` — 조음 속도(길이)와 **호흡 삽입**. 제어를 만들기 *전*.
* `apply_to_controls` — 피치 엔벨로프와 세기. 제어를 만든 *후*.

호흡 모형
---------
사람은 폐활량이 있어서 길게 말하면 반드시 숨을 쉰다. 그 숨은 잡음이 아니라
발화 구조의 일부이고, 없으면 "AI 가 말한다" 는 인상의 큰 축이 된다.
여기서는 유성 발화 시간을 누적해 용량을 넘기 전에 들숨을 끼워 넣는다.
문장 경계(`silence`)에서 우선 쉬고, 길이가 모자라면 강제로 넣는다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch


@dataclass
class BreathPlan:
    capacity_s: float = 4.5      # 한 번의 들숨으로 이어 말할 수 있는 유성 시간
    depth: float = 0.5           # 들숨 소리 세기
    duration_s: float = 0.32     # 들숨 길이
    min_gap_s: float = 1.2       # 숨과 숨 사이 최소 간격
    enabled: bool = True


@dataclass
class ProsodyPlan:
    """LLM 이 만들어 넣기 좋은 작은 의도 기술자."""
    rate: float = 1.0            # 조음 속도 배율 (1.2 = 20% 빠르게)
    pitch_shift: float = 0.0     # 반음. 전체 음역 이동
    pitch_range: float = 1.0     # 억양 폭 배율 (0 = 단조, 2 = 과장)
    contour: list | None = None  # [[위치 0~1, 반음], ...] 발화 전체에 얹는 억양
    # 발화 전체에 걸친 하강량(반음). 평서문은 -2 쯤이 자연스럽고 의문문은 0 이상.
    # 기본값은 **중립(0)** 이다 — 운율 계획을 안 준 스크립트의 소리가 바뀌면 안 된다.
    # (초당으로 두면 긴 발화에서 음역을 벗어난다: 6 초에 -1.5 반음/초 = -9 반음.
    #  구(phrase) 단위 기술이 음성학의 표준이기도 하다.)
    declination: float = 0.0
    energy: list | None = None   # [[위치, 배율], ...] 세기 엔벨로프
    breathiness: float = 0.0     # 기식성 가산 (Rd 를 breathy 쪽으로)
    breath: BreathPlan = field(default_factory=BreathPlan)

    @property
    def is_neutral(self) -> bool:
        """아무것도 바꾸지 않는 계획인가 (그러면 전부 건너뛴다)."""
        return (self.rate == 1.0 and self.pitch_shift == 0.0
                and self.pitch_range == 1.0 and not self.contour
                and self.declination == 0.0 and not self.energy
                and self.breathiness == 0.0 and not self.breath.enabled)

    # ------------------------------------------------------------------ LLM
    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> "ProsodyPlan":
        if not d:
            return ProsodyPlan()
        d = dict(d)
        b = d.pop("breath", None)
        known = {k: v for k, v in d.items() if k in ProsodyPlan.__annotations__}
        plan = ProsodyPlan(**known)
        if isinstance(b, dict):
            plan.breath = BreathPlan(**{k: v for k, v in b.items()
                                        if k in BreathPlan.__annotations__})
        return plan

    #: LLM 프롬프트에 그대로 넣을 수 있는 스키마 설명
    SCHEMA = (
        "rate: 조음 속도 배율(0.7 느리게 ~ 1.4 빠르게). "
        "pitch_shift: 반음 단위 이동(-6 ~ +6). "
        "pitch_range: 억양 폭(0.3 단조 ~ 2.0 과장). "
        "contour: [[0~1 위치, 반음], ...] 문장 억양. "
        "declination: 발화 전체 하강량(반음, 보통 -1 ~ -4; 평서문은 크게, 의문문은 0 이상). "
        "breathiness: 0~1 (0.5 이상이면 속삭이듯). "
        "breath.capacity_s: 한 숨에 이어 말할 초(짧으면 숨차게 들린다)."
    )


# ------------------------------------------------------------------ 시간 왜곡
def warp_timeline(timeline: list, plan: ProsodyPlan) -> list:
    """조음 속도를 적용하고 필요한 곳에 들숨을 끼워 넣는다.

    무음 구간은 속도의 영향을 절반만 받는다 — 빨리 말한다고 문장 사이 쉼이
    같은 비율로 줄지는 않기 때문이다.
    """
    rate = max(0.25, float(plan.rate))
    out: list = []
    voiced_since_breath = 0.0
    time_since_breath = 0.0
    b = plan.breath

    for seg in timeline:
        seg = dict(seg)
        kind = seg.get("type", "vowel")
        dur = float(seg.get("dur", 0.3))
        seg["dur"] = dur / (rate if kind != "silence" else (1.0 + rate) / 2.0)

        silent = kind in ("silence", "breath")
        need = (b.enabled and not silent
                and voiced_since_breath + seg["dur"] > b.capacity_s
                and time_since_breath > b.min_gap_s)
        if need:
            out.append({"type": "breath", "dur": b.duration_s, "inhale": True,
                        "strength": b.depth})
            voiced_since_breath = 0.0
            time_since_breath = 0.0

        out.append(seg)
        time_since_breath += seg["dur"]
        if silent:
            # 쉼에서도 조금은 회복한다(짧은 도둑숨)
            voiced_since_breath = max(0.0, voiced_since_breath - seg["dur"] * 2.0)
        else:
            voiced_since_breath += seg["dur"]
    return out


# --------------------------------------------------------------- 피치/세기
def _curve(points, t: int) -> torch.Tensor:
    from .utils import ramp
    return ramp(t, [(float(p), float(v)) for p, v in points])


def apply_to_controls(ctrl, plan: ProsodyPlan, frame_rate: float = 100.0,
                      f0_min: float = 55.0, f0_max: float = 880.0):
    """제어 파라미터에 피치 엔벨로프·세기·기식성을 얹는다. (제자리 아님)

    피치는 **로그 영역**에서 다룬다. 반음이 곧 비율이라 화자의 음역이 달라도
    같은 억양이 같은 인상을 준다.
    """
    if plan.is_neutral:
        return ctrl
    t = ctrl.f0.shape[1]
    dev, dt = ctrl.f0.device, ctrl.f0.dtype
    semis = torch.zeros(1, t, 1, device=dev, dtype=dt)

    if plan.contour:
        semis = semis + _curve(plan.contour, t).to(dev)
    if plan.declination:
        pos = torch.linspace(0.0, 1.0, t, device=dev, dtype=dt).view(1, t, 1)
        semis = semis + plan.declination * pos
    semis = semis + plan.pitch_shift

    f0 = ctrl.f0
    if plan.pitch_range != 1.0:
        # 억양 폭: 화자의 중앙 F0 를 축으로 로그 편차를 스케일한다
        voiced = ctrl.harmonic_amp > 1e-4
        ref = f0[voiced].median() if bool(voiced.any()) else f0.median()
        f0 = ref * (f0 / ref.clamp_min(1e-3)) ** float(plan.pitch_range)
    # 화자의 물리 음역을 벗어나지 않게 묶는다 (설계상 발산 불가).
    if float(semis.abs().max()) > 0 or plan.pitch_range != 1.0:
        ctrl.f0 = (f0 * (2.0 ** (semis / 12.0))).clamp(f0_min, f0_max)

    if plan.energy:
        g = _curve(plan.energy, t).to(dev)
        ctrl.harmonic_amp = ctrl.harmonic_amp * g
        ctrl.noise_bands = ctrl.noise_bands * g
    if plan.breathiness:
        k = float(min(max(plan.breathiness, 0.0), 1.0))
        ctrl.rd = ctrl.rd + k * (2.7 - ctrl.rd) * 0.6
        ctrl.noise_am = (ctrl.noise_am + k * 0.5).clamp(0.0, 1.0)
    return ctrl
