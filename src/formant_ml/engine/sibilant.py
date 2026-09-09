"""치찰음 전용 경로 — 협착 **위치** 하나로 /s/·/z/·/ʃ/·/f/·속삭임·/h/ 를 가른다.

왜 따로 빼는가 (docs/MEASUREMENTS.md §27)
------------------------------------------
공용 경로에 정규화를 걸었더니 협착 `a_c` 의 바닥이 0.099 → 0.162 로 올라갔고, 다이폴
기하 효율이 `(A_ref/a_c)^2.5` 라 **앞니 다이폴이 3.8 배 약해졌다**. 적합기는
`fric_gain` 을 올려 레벨로 때웠고, 6 대역 에너지는 1 dB 안에서 맞은 채 소리는
치찰음이 아니게 됐다. 크기 스펙트럼만 보는 손실은 **소스의 종류**를 구별하지 못한다.

그래서 치찰음은 (1) 자기 파라미터화와 (2) 자기 합격 조건을 갖는다.

무엇이 축인가 (docs/MEASUREMENTS.md §19.6)
--------------------------------------------
셋을 묶는 축은 하나가 아니라 셋이고, **그중 둘만 연속**이다.

    (1) 난류 세기   연속 — 국소 협착의 레이놀즈 (Stevens). 위치와 무관한 식.
    (2) 하류 공동   연속 — 협착 위치가 정한다. 앞공동만인가, 성도 전체인가.
    (3) 소스 종류   **범주** — 제트가 모서리를 때리는가(장애물), 벽을 따라 흐르는가.

(3) 을 거리의 매끄러운 함수로 만들면 안 된다 — 앞니는 제트가 **부딪히는** 집중
변동력이고 후두개는 제트가 **따라 흐르는** 분포 압력요동이라, 정도가 아니라 종류가
다르다 (Shadle 1985/1990 의 20 dB 차이). v1 이 `OBSTACLE_SIBILANTS` 집합으로 둔 것이
맞았고, 그것을 연속 함수로 바꾸려던 시도는 §19.6 에서 철회했다.

그래서 유성 치찰음과 속삭임이 어떻게 나오는가
-----------------------------------------------
* **유성 치찰음** (/z/ /ʒ/): (3) 은 /s/ 와 **같다**. 갈리는 것은 성문이다 —
  내전하면 직렬 오리피스에서 `Ac²/Ag²` 가 더 이상 0 이 아니라 제트 속도가 베르누이
  상한에서 내려오고(v/V_REF 0.99 → 0.41), 이미 있는 속도항이 다이폴 몫을 2.4 배
  줄인다 (§19.2). 새 함수가 필요 없다.
* **속삭임**: (1) 의 협착이 구강이 아니라 **성문**이고, (2) 의 하류가 성도 전체이며,
  (3) 이 `wall` 이다. 지금 공용 경로는 협착을 구강 한 곳으로 못박아 두어 이것을
  표현할 수 없었다 (§19.3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .noise import (OBSTACLE_A_REF, OBSTACLE_JET_EXP, OBSTACLE_VEL_EXP,
                    V_REF_DIPOLE, reynolds, series_flow)

C_SOUND = 35000.0                      # cm/s

#: 소스의 **종류**. 범주다 — 사이값을 만들지 말 것 (§19.6).
#:   "obstacle" 제트가 앞니 모서리에 부딪힌다. 집중 변동력, 상승 스펙트럼, 강하다.
#:   "wall"     제트가 벽/자유공간을 따라 흐른다. 분포 압력요동, 평평하고 약하다.
SOURCE_TYPES = ("obstacle", "wall")

#: 성도 길이 대비 협착 **위치** 0(성문) ~ 1(입술) 과 소스 종류.
#: `back_leak` 은 마찰 소스가 앞공동 대신 성도 전체로 새는 비율이다 — 위치가 뒤로
#: 갈수록 1 에 가까워야 한다 (하류에 성도가 다 남으므로).
@dataclass(frozen=True)
class SibilantSpec:
    name: str
    place: float                 # 0 성문 ~ 1 입술
    source: str                  # SOURCE_TYPES
    a_target: float              # 목표 협착 면적 [cm²]
    back_leak: float             # 0 앞공동만 ~ 1 성도 전체
    obstacle: float              # 장애물 결합 세기 (wall 이면 0)
    voiced: bool = False

    def __post_init__(self) -> None:
        if self.source not in SOURCE_TYPES:
            raise ValueError(f"source must be one of {SOURCE_TYPES}")
        if self.source == "wall" and self.obstacle != 0.0:
            raise ValueError("wall 소스에는 장애물 결합이 없다 (§19.6)")


#: 음소별 자세. `place` 와 `a_target` 은 프로파일로 화자마다 덮어쓸 수 있다.
#: /f/ 와 /h/ 가 `wall` 인 것은 v1 `score.py` 의 `OBSTACLE_SIBILANTS` 와 같다 —
#: "장애물이 없어 훨씬 약하고 평평하다".
PRESETS: dict[str, SibilantSpec] = {
    "s":       SibilantSpec("s",       0.90, "obstacle", 0.10, 0.10, 0.15),
    "ss":      SibilantSpec("ss",      0.90, "obstacle", 0.08, 0.10, 0.18),
    "sh":      SibilantSpec("sh",      0.82, "obstacle", 0.14, 0.12, 0.10),
    "z":       SibilantSpec("z",       0.90, "obstacle", 0.10, 0.10, 0.15, voiced=True),
    "f":       SibilantSpec("f",       1.00, "wall",     0.12, 0.60, 0.0),
    "h":       SibilantSpec("h",       0.05, "wall",     0.60, 1.00, 0.0),
    "whisper": SibilantSpec("whisper", 0.05, "wall",     0.08, 1.00, 0.0),
}


def place_from_peak(peak_hz: float, tract_cm: float = 14.6) -> float:
    """앞공동 봉우리 **실측**에서 협착 위치를 역산한다.

    봉우리는 앞공동의 1/4 파장이므로 `L_front = c/(4 f_p)` 이고 위치는
    `1 − L_front/L` 이다. 프로파일의 `sibilant.peak_hz` 가 화자마다 계측돼 있으므로
    (docs/MEASUREMENTS.md), 위치를 상수로 박지 말고 그것에서 받는다 —
    **프로파일이 계측을, 이 모듈이 물리를 맡는다.**
    """
    l_front = C_SOUND / (4.0 * max(float(peak_hz), 1.0))
    return float(min(0.995, max(0.0, 1.0 - l_front / tract_cm)))


def from_profile(name: str, prof, tract_cm: float = 14.6) -> SibilantSpec:
    """프로파일 실측으로 프리셋을 화자에 맞춘다.

    장애물 치찰음만 위치를 옮긴다 — `wall` 소스(속삭임·/h/·/f/)의 위치는 앞공동
    봉우리가 정하는 양이 아니라 **해부학**이다.
    """
    spec = PRESETS[name]
    if spec.source != "obstacle":
        return spec
    sib = getattr(prof, "sibilant", {}) or {}
    place = place_from_peak(sib.get("peak_hz", 8000.0), tract_cm)
    a_min = float(sib.get("a_min", spec.a_target))
    # /ʃ/ 는 /s/ 보다 앞공동이 길다 — 실측 봉우리에서 잰 위치를 그만큼 뒤로 민다.
    place = place - (PRESETS["s"].place - spec.place)
    return SibilantSpec(spec.name, min(0.995, max(0.0, place)), spec.source,
                        a_min if name in ("s", "z") else spec.a_target,
                        spec.back_leak, float(sib.get("obstacle", spec.obstacle))
                        if name == "s" else spec.obstacle, spec.voiced)


def dipole_share(spec: SibilantSpec, a_c, jet_v) -> torch.Tensor:
    """앞니 다이폴이 소스에서 차지하는 몫 `obst_eff`.

    `wall` 소스에서는 **구조적으로 0** 이다 — 거리 함수로 작게 만드는 것이 아니라
    아예 없다 (§19.6). `obstacle` 에서는 기존 두 항을 그대로 쓴다:

        obst_eff = obstacle · (A_ref/A_c)^2.5 · (v/V_REF)^1

    기하항은 "좁은 제트라야 모서리를 때린다" (Shadle), 속도항은 Curle 의 U⁶/U⁴ 에서
    나온 U¹ 이다. 무성 마찰음에서는 속도가 베르누이 상한에 붙어 속도항이 포화하지만
    (§14.2), **유성에서는 성문이 좁아 살아난다** (§19.2).
    """
    a_c = torch.as_tensor(a_c)
    if spec.source == "wall":
        return torch.zeros_like(a_c)
    g = (OBSTACLE_A_REF / a_c.clamp_min(1e-4)).clamp(0.0, 1.0) ** OBSTACLE_JET_EXP
    v = (torch.as_tensor(jet_v) / V_REF_DIPOLE).clamp(0.0, 1.0) ** OBSTACLE_VEL_EXP
    return spec.obstacle * g * v


def front_len_cm(spec: SibilantSpec, tract_cm: float = 14.6) -> float:
    """협착 하류의 앞공동 길이. `wall` + 성문 협착이면 성도 전체가 하류다."""
    return max(0.3, (1.0 - spec.place) * tract_cm)


def front_peak_hz(spec: SibilantSpec, tract_cm: float = 14.6) -> float:
    """앞공동 1/4 파장 공진 — 치찰음 봉우리의 소재."""
    return C_SOUND / (4.0 * front_len_cm(spec, tract_cm))


def jet_state(spec: SibilantSpec, p_sub: float, a_g: float, a_c: float | None = None):
    """직렬 오리피스(성문 → 구강 협착)의 제트 속도와 레이놀즈.

    유성/무성이 갈리는 곳이다. 무성은 성문이 벌어져(Ag ≫ Ac) 속도가 압력에만
    묶이고, 유성은 성문이 좁아 `Ac²/Ag²` 가 살아나 속도가 내려온다.
    """
    ac = spec.a_target if a_c is None else a_c
    t = lambda x: torch.tensor([[float(x)]], dtype=torch.float64)
    u, _ = series_flow(t(p_sub), t(a_g), t(ac))
    re, v, d = reynolds(u, t(ac))
    return dict(reynolds=float(re), jet_v=float(v), diameter=float(d), flow=float(u))


def control_values(spec: SibilantSpec, p_sub: float, a_g: float,
                   tract_cm: float = 14.6, a_c: float | None = None) -> dict:
    """자세 -> 제어열 값. 공용 엔진의 파라미터 이름으로 낸다.

    DSP 를 새로 만들지 않는다 — `FricationNoise`/`VocalTract` 가 이미 이 손잡이를
    받는다. 치찰음 전용인 것은 **이 손잡이들을 무엇으로 채우는가** 이고, 그것이
    위치·소스 종류·성문 자세에서 유도된다.
    """
    ac = spec.a_target if a_c is None else a_c
    js = jet_state(spec, p_sub, a_g, ac)
    obst = float(dipole_share(spec, torch.tensor(ac), js["jet_v"]))
    return dict(a_c=ac, c_place=spec.place, front_len=front_len_cm(spec, tract_cm),
                obstacle=obst, back_leak=spec.back_leak, p_sub=p_sub,
                reynolds=js["reynolds"], jet_v=js["jet_v"],
                front_peak_hz=front_peak_hz(spec, tract_cm))


# ------------------------------------------------------------------ 제스처
# **혀 제스처의 모양은 손으로 그리지 말 것** (docs/MEASUREMENTS.md §14.3, §29).
#
# 처음 이 모듈의 데모는 `a_c` 를 3.0 -> 목표까지 올림 코사인으로 긋고 `obstacle` 과
# `fric_gain` 에도 같은 램프를 곱했다. 셋 다 틀렸다.
#
#  1. **시작값.** 모음에서 오는 혀는 완전 개방(3.0 cm²)에서 출발하지 않는다.
#     `phones.sibilant` 은 개시에 이미 0.35 로 좁혀 두고 거기서 목표로 간다.
#     3.0 에서 선형으로 그으면 가청 구간(≈0.35 아래)을 마지막 몇 ms 에 몰아서
#     지나므로 페이드 인이 사라진다 — 소스 세기가 `1/a_c^2.5` 로 가기 때문이다.
#  2. **이중 계상.** 다이폴의 페이드는 엔진이 기하항 `(A_ref/a_c)^2.5` 로 이미 만든다.
#     `obstacle` 에 램프를 또 곱하면 2.5 제곱이 두 번 걸린다.
#  3. **삼중 계상.** 진폭 포락도 레이놀즈/Stevens 가 `a_c` 와 `p_sub` 에서 만든다.
#     `fric_gain` 에 램프를 곱하는 것은 물리가 낸 포락 위에 손으로 하나 더 얹는 것이다.
#
# 그래서 **움직이는 것은 `a_c` 와 성문뿐**이고 나머지는 상수다. 포락은 물리가 낸다.
TONGUE_RISE_FRAC = 0.217        # v1 TONGUE_SUSTAIN_HOLD 0.62 / TONGUE_CLOSE_FRAC 0.57
TONGUE_FALL_FRAC = 0.163        # 그 비대칭이 실측 상승/하강비 1.28~1.35
TONGUE_RISE_MIN_S = 0.040       # 짧은 CV 의 바닥 (v2 가 절대값으로 쓰던 값)
TONGUE_FALL_MIN_S = 0.018
A_APPROACH = 0.35               # 개시 시점의 협착 — 이미 좁혀져 있다
PRESSURE_LEAD_S = 0.080         # 폐압이 혀보다 먼저 자리를 잡는다 (아래 주석)
GLOTTAL_LEAD_S = 0.040          # 성문은 폐압보다 **더** 먼저 열린다 (아래 주석)
# **성문 기식을 협착이 억누른다** — 지금 엔진에 없는 물리의 대리 손잡이.
#
# `glottis.physiology` 의 기식 포락은 `(1−adduction)²·√Ps` 다. 성문이 넓을수록
# **커지고**, 구강 협착을 **안 본다**. 그래서 /s/ 자세(성문 활짝 + 구강 좁힘)에서
# 기식이 최대로 나오고, 측정에서 개시 직전 구간이 치찰음 고원보다 3.7 dB 더 컸다 —
# 매 /s/ 앞에 "하—" 가 붙는다.
#
# 실제로는 좁은 구강 협착 뒤에 구강내압 Pm 이 쌓여 경성문 압력차(Ps − Pm)를 깎으므로
# 성문 기식이 억제된다 (v1 `aeroacoustic.py` 의 "구강내압과 발성 억제" 가 같은 물리를
# 파열음에 대해 적어 두었다). 엔진은 그 Pm 을 **버스트에만** 쓰고 기식에는 안 건다.
#
# 그 물리가 들어올 때까지(§19.5-1) `aspiration` 배율로 대신한다. **대리 손잡이임을
# 잊지 말 것** — 협착 면적에서 유도된 값이 아니라 자세에 붙인 상수다.
ASP_AT_APPROACH = 0.25          # 접근 자세에서 기식 배율
ASP_AT_PLATEAU = 0.10           # 고원(최협착)에서
CV_UNDERSHOOT = 2.2             # 130 ms 급 음절은 목표의 2.2 배 (v1 TONGUE_CV_A_MIN)


def undershoot(dur_s: float) -> float:
    """짧은 음절은 목표까지 못 간다. 130 ms 에서 2.2 배, 400 ms 이상이면 1.0."""
    x = min(max((0.40 - dur_s) / (0.40 - 0.13), 0.0), 1.0)
    return 1.0 + (CV_UNDERSHOOT - 1.0) * x


def gesture_keyframes(spec: SibilantSpec, prof, dur: float = 0.220,
                      t0: float = 0.060, p_sub: float = 8.0,
                      fric_gain: float = 24.0, tract_cm: float = 14.6,
                      vowel: str = "eu") -> list[dict]:
    """자세 하나를 내는 키프레임. `control.track_from_keyframes` 에 그대로 넣는다.

    보간은 엔진이 한다 (파라미터마다 독립 시간축, minjerk). 여기서는 **어느 시각에
    어떤 값인지**만 정하고 그 사이는 안 그린다.

    `obstacle` 소스는 혀가 제스처를 만들고, `wall` 소스(속삭임·/h/)는 **성문이**
    만든다 — 혀 협착이 없으므로 `a_c` 가 안 움직인다.
    """
    c = control_values(spec, p_sub=p_sub, a_g=0.20, tract_cm=tract_cm)
    f1, f2, f3 = (prof.vowels.get(vowel) or [464.0, 1608.0, 2906.0])
    # **폐압은 제스처가 아니다.** 발화 중 폐압은 이미 서 있고, 치찰음을 만드는 것은
    # 혀뿐이다. p_sub 를 개시 시각에 같이 올리면 그 계단이 성문 기식을 광대역으로
    # 켜서, 저·중역이 혀와 **무관하게** t0 에 선다 (측정: 0~2k 40 ms / 2~4k 60 ms 가
    # 지속시간을 130 → 600 ms 로 바꿔도 안 움직였다 — 고역만 76 → 112 ms 로 늘었다).
    # 그러면 페이드 인이 고역에만 생기고 개시가 "퍽" 하고 터진다.
    #
    # 그래서 폐압은 **개시보다 먼저** 자리를 잡는다. `PRESSURE_LEAD_S` 만큼 앞서
    # 목표값에 도달하고, 그 구간에는 성문이 벌어져 있어 소리가 안 난다.
    # **순서가 있다: 성문을 먼저 열고, 그 다음 폐압, 마지막에 혀.**
    # 폐압을 먼저 올리면 아직 내전된 성문이 울려 목소리가 난다 (측정: 치찰도가
    # 38.9 → 4.2 dB 로 무너지고 무게중심이 9461 → 6630 Hz 로 내려왔다).
    # 순서는 **성문 → 혀 → 폐압** 이다.
    #
    # 성문이 먼저다: 폐압을 먼저 올리면 아직 내전된 성문이 울려 목소리가 난다
    # (측정: 치찰도 38.9 → 4.2 dB, 무게중심 9461 → 6630 Hz).
    #
    # 그런데 성문만 열고 폐압을 올리면 이번에는 **/h/ 가 먼저 난다**. 이 엔진의
    # 기식 포락은 `(1−adduction)²·√Ps` 라 성문이 넓을수록 **커지고**, 구강이 열려
    # 있으면 그 소리가 성도 전체를 울린다. 측정: 선행 구간이 고원보다 3.7 dB 더
    # 컸다 — 매 /s/ 앞에 "하—" 가 붙는다.
    #
    # 그래서 **혀가 폐압보다 먼저** 자리를 잡는다. `A_APPROACH` 로 좁혀 두면 직렬
    # 오리피스에서 유량이 구강에 묶여 기식이 억제되고, 폐압이 오를 때 이미 마찰
    # 자세다. 실제 발화에서도 폐압 상승은 앞 음소 **동안** 일어나지 침묵에서
    # 시작하지 않는다.
    abd = 0.06 if not spec.voiced else 0.55
    t_gl = max(0.0, t0 - PRESSURE_LEAD_S - GLOTTAL_LEAD_S)
    kf: list[dict] = [
        {"t": 0.0, "p_sub": 0.0, "adduction": 0.6, "a_c": 3.0},
        {"t": t_gl, "adduction": abd},
        {"t": max(0.0, t0 - PRESSURE_LEAD_S), "a_c": A_APPROACH, "p_sub": p_sub,
         "aspiration": ASP_AT_APPROACH},
    ]

    if spec.source == "wall" and spec.place < 0.5:
        # 성문 협착 — 속삭임/기식. 혀는 모음 자세 그대로 두고 성문만 조인다.
        # 하류가 성도 전체이므로 포먼트가 출력을 정한다 (§19.6).
        #
        # **한계.** 지금 `FricationNoise` 는 협착을 구강 한 곳(`a_c`)으로 못박아
        # 두어 성문 협착의 난류를 레이놀즈로 못 만든다 (§19.5-1 미구현). 그래서
        # 여기서는 `adduction` 으로만 세기를 가른다 — 속삭임은 성문을 더 조이고
        # (`a_target` 작음) /h/ 는 덜 조인다. 물리가 아니라 **대리 손잡이**다.
        add_g = float(min(0.30, max(0.04, 0.02 + 0.9 * spec.a_target)))
        kf += [
            {"t": t0, "adduction": add_g, "a_c": 3.0,
             "c_place": spec.place, "front_len": c["front_len"], "obstacle": 0.0,
             "back_leak": spec.back_leak, "fric_gain": fric_gain,
             "f1": f1, "f2": f2, "f3": f3, "oral_open": 1.0},
            {"t": t0 + dur, "adduction": add_g},
            {"t": t0 + dur + 0.040, "p_sub": 0.0, "adduction": 0.6},
        ]
        return kf

    rise = max(TONGUE_RISE_MIN_S, TONGUE_RISE_FRAC * dur)
    fall = max(TONGUE_FALL_MIN_S, TONGUE_FALL_FRAC * dur)
    a_min = spec.a_target * undershoot(dur)
    adduct = 0.55 if spec.voiced else 0.06
    lead, lag = 0.035, 0.055                    # 성문이 먼저 열리고 늦게 닫힌다
    kf += [
        # **여기에 `adduction: 0.6` 을 두면 안 된다.** 원래 설계는 모음에서 오는
        # 경우라 개시 직전까지 성문이 닫혀 있었는데, 지금은 위에서 이미 열어 두었다.
        # 둘을 같이 두면 성문이 열렸다 **다시 닫혔다** 열린다 — 그 사이에 폐압이 이미
        # 서 있어서 목소리가 난다. 측정: 선행 구간의 `du` 가 −15.5 dB (고원에서는
        # −240 dB), adduction 중앙값이 0.516. 그것이 /s/ 앞에 붙던 소리의 정체다.
        # 기식이 아니라 **유성**이었고, `aspiration` 을 낮춰도 안 없어졌던 이유다.
        # 개시: 이미 0.35 로 좁혀져 있고, 그 밖의 손잡이는 **여기서 상수로 고정**된다.
        {"t": t0, "adduction": adduct, "a_c": A_APPROACH,
         "c_place": spec.place, "front_len": c["front_len"],
         "obstacle": spec.obstacle, "back_leak": spec.back_leak,
         "fric_gain": fric_gain, "oral_open": 1.0, "tract_gain": 0.9,
         "f1": f1, "f2": f2, "f3": f3},
        {"t": t0 + rise, "a_c": a_min, "aspiration": ASP_AT_PLATEAU},   # 고원 도달
        {"t": t0 + dur - fall, "a_c": a_min, "adduction": adduct},
        {"t": t0 + dur, "a_c": 3.0, "obstacle": 0.0, "back_leak": 0.3,
         "aspiration": 1.0},                                  # 해제 뒤 기식 꼬리
        {"t": t0 + dur + lag, "adduction": 0.6, "fric_gain": 1.0, "p_sub": 0.0},
    ]
    return kf


# ------------------------------------------------------------------ 합격 조건
#: §27 이 빠뜨렸던 칸. 크기 스펙트럼·포락 통계는 소스의 **종류**를 못 본다.
#: `obst_eff` 가 기준의 이 배수 아래로 내려가면 다이폴이 꺼진 것이다 — 실격.
OBST_EFF_MIN_RATIO = 0.8
CENTROID_TOL_HZ = 300.0
PEAK_TOL_HZ = 1000.0


def obst_eff_from_track(values, index, a_g: float = 0.2) -> float:
    """적합된 트랙에서 마찰 프레임의 `obst_eff` 중앙값.

    협착 프레임(`a_c` < 0.5)만 본다. 성문 면적은 트랙에 없으므로 무성 마찰음의
    전형값으로 고정한다 — 조건 간 **비교용**이라 상수 배는 비에서 지워진다.
    """
    import numpy as np
    ac = values[:, index["a_c"]]
    ob = values[:, index["obstacle"]]
    m = ac < 0.5
    if m.sum() < 5:
        m = np.ones(len(ac), bool)
    g = np.clip(OBSTACLE_A_REF / np.maximum(ac, 1e-4), 0.0, 1.0) ** OBSTACLE_JET_EXP
    return float(np.median((g * ob)[m]))


def accept(report: dict, baseline_obst_eff: float | None = None) -> tuple[bool, list[str]]:
    """치찰음의 합격 판정. **이것을 통과하지 못하면 다른 점수는 보지 않는다.**

    report: `turbulence.compare` 의 출력에 `obst_eff` 를 더한 dict.
    """
    fail = []
    if abs(report.get("centroid_err", 0.0)) > CENTROID_TOL_HZ:
        fail.append(f"무게중심 오차 {report['centroid_err']:+.0f} Hz "
                    f"(> {CENTROID_TOL_HZ:.0f})")
    if abs(report.get("peak_err", 0.0)) > PEAK_TOL_HZ:
        fail.append(f"봉우리 오차 {report['peak_err']:+.0f} Hz (> {PEAK_TOL_HZ:.0f})")
    if baseline_obst_eff is not None and baseline_obst_eff > 0:
        r = report.get("obst_eff", 0.0) / baseline_obst_eff
        if r < OBST_EFF_MIN_RATIO:
            fail.append(f"앞니 다이폴이 기준의 {r:.2f} 배로 꺼졌다 "
                        f"(< {OBST_EFF_MIN_RATIO})")
    return (not fail), fail
