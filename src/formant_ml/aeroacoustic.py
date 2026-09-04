"""공기음향(aeroacoustic) 난류 소스 — 마찰음/기식음의 '진짜' 물리 기반.

왜 다시 썼나
------------
이전 구현은 마찰음 진폭을 `유량^2.5` 같은 임의의 거듭제곱으로 만들었다. 그건
음향학이 아니라 곡선 맞추기였다. 실제 난류 소스는 세 가지 문헌 사실을 따른다.

1. **난류는 임계 레이놀즈수에서 켜진다** (Stevens 1971). 협착부 입자속도 v=U/A,
   유효 지름 d 로 Re = v·d/ν 를 만들면, Re 가 임계값 Re_c≈1800 을 넘을 때만
   난류(=마찰음)가 생긴다. 그 아래는 층류라 소리가 없다. 그래서 마찰음의
   시작/끝은 임의의 페이드가 아니라 **유량이 임계 유속을 넘고 못 넘는 사건**이다.
   모음처럼 협착이 열리면(A 큼) v 가 떨어져 Re<Re_c → 마찰음이 물리적으로 꺼진다.

2. **소스 세기는 협착부 압력강하에 비례한다** (Stevens 1971: "equivalent
   sound-pressure source whose magnitude is proportional to the pressure drop").
   ΔP = ½ρ(U/A)² (베르누이). 그래서 소스 진폭 ∝ (U/A)². 유량이 오르면 음량은
   그보다 훨씬 가파르게 오른다 — 부드러운 시작이 여기서 나온다(곡선을 손으로
   그리지 않아도).

3. **무게중심은 입자속도와 함께 오른다** (Stevens 1971: intensity·centre frequency
   vary monotonically with particle velocity). 협착이 열리며 v 가 떨어지면 무게중심도
   내려간다. 실측 /사/ 의 무게중심 하강(6700→3900 Hz)이 정확히 이것이다 —
   이제 곡선이 아니라 **협착 면적 궤적에서 유도**된다.

소스의 위치와 스펙트럼 모양(앞공동 극, 뒤공동 영점)은 Heinz & Stevens(1961),
Shadle(1985/1990), Narayanan & Alwan(2000)의 극-영점 모형이며 `dsp/sibilant.py`
가 담당한다. 이 모듈은 그 필터에 들어갈 **소스 진폭·색·시변 중심**을 만든다.

유성 마찰음의 성문동기 변조(Jackson & Shadle 2000)는 성문 유량으로 이 진폭을
곱해 만든다(`noise.TurbulenceSource` 의 glottal AM).

모두 CGS 단위 (g, cm, s). 미분가능(경사학습 경로에서 그대로 쓴다).
"""
from __future__ import annotations

import math

import torch

from .utils import ramp

# --- 공기 물성 (CGS) ---------------------------------------------------------
RHO = 1.14e-3        # 공기밀도 [g/cm^3] (체온·습윤 근사)
MU = 1.86e-4         # 동점성계수 [g/(cm·s)] = poise
NU = MU / RHO        # 운동점성 ≈ 0.163 cm^2/s
RE_C = 1800.0        # 임계 레이놀즈수 (Stevens 1971; 문헌 1700~2000)
RE_WIDTH = 600.0     # 켜짐이 계단이 아니라 부드럽게 (수치·미분 안정)


def hydraulic_diameter(area: torch.Tensor) -> torch.Tensor:
    """협착 단면적 -> 유효(수력) 지름 [cm]. 원형 근사 d = 2·sqrt(A/π)."""
    return 2.0 * (area.clamp_min(1e-6) / torch.pi).sqrt()


def particle_velocity(flow: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
    """부피유량 U [cm^3/s] 와 협착 면적 A [cm^2] -> 입자속도 v=U/A [cm/s]."""
    return flow / area.clamp_min(1e-6)


def reynolds(flow: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
    """협착부 레이놀즈수 Re = v·d/ν."""
    v = particle_velocity(flow, area)
    return v * hydraulic_diameter(area) / NU


def turbulence_gate(flow: torch.Tensor, area: torch.Tensor,
                    re_c: float = RE_C, width: float = RE_WIDTH) -> torch.Tensor:
    """난류 발생 여부 0~1. Re>Re_c 에서 부드럽게 켜진다(Stevens 1971)."""
    return torch.sigmoid((reynolds(flow, area) - re_c) / width)


def frication_source_amp(flow: torch.Tensor, area: torch.Tensor,
                         re_c: float = RE_C) -> torch.Tensor:
    """난류 소스 진폭 (임의단위, 프레임별 상대값).

    Stevens(1971): 소스 크기 ∝ 협착부 압력강하 ΔP = ½ρ(U/A)². 여기에 임계
    레이놀즈수 게이트를 곱한다. 결과는 (a) 협착이 좁고 유량이 셀 때 크고,
    (b) 협착이 열리면(모음) 저절로 0 으로 꺼진다.
    """
    v = particle_velocity(flow, area)
    dp = 0.5 * RHO * v * v                     # 동압 [dyn/cm^2]
    return dp * turbulence_gate(flow, area, re_c)


def velocity_centroid_scale(flow: torch.Tensor, area: torch.Tensor,
                            v_ref: float = 3000.0) -> torch.Tensor:
    """입자속도에 따른 무게중심 배율 (Stevens 1971).

    무게중심은 입자속도와 함께 오른다. v_ref(전형적 /s/ 협착 속도 ~3000 cm/s)에서
    1.0 이 되도록 정규화한 완만한 배율. sqrt 로 완만하게.

    .. warning::
       **이걸 성도 공진 주파수에 곱하면 안 된다.** Stevens 가 말하는 건 난류
       **소스** 스펙트럼의 무게중심이지 성도의 극/영점이 아니다. 공진은 기하가
       정한다 — 기류가 느리다고 앞니가 멀어지지 않는다. 실제로 곱해 봤더니
       /s/ 개시(압력이 낮아 속도가 느린 구간)에서 앞니 공명이 10.0 kHz 대신
       6.2 kHz 에 놓여 **/s/ 가 /ʃ/ 로 시작했다**. 소스 기울기는
       `source_tilt_shift`, 공진 기하는 `area_centroid_scale` 를 쓴다.

    남겨 둔 이유: 소스 무게중심 자체가 필요한 곳(학습/분석)에서 쓴다.
    """
    v = particle_velocity(flow, area)
    return (v / v_ref).clamp(0.2, 2.5).sqrt()


#: 앞공동 공진 배율의 지수. 협착이 열릴 때 공진이 얼마나 빨리 내려가는가.
#: 1/4 파장 f=c/(4·Lf) 에서 유효 앞공동 길이가 면적비의 거듭제곱으로 늘어난다고
#: 본 것이다(협착이 풀리면 최협착점이 혀 뒤쪽으로 물러나 앞공동이 길어진다).
#: 0.5 는 실측 /사/ 해제 구간의 무게중심 하강(6700 -> 3900 Hz, 비 0.58)에 맞춘
#: 값이다. 손으로 그린 곡선이 아니라 지수 하나이고, 협착 궤적이 바뀌면 하강도
#: 저절로 따라간다.
FRONT_CAVITY_EXP = 0.5

#: 입자속도가 2 배가 될 때 난류 **소스** 스펙트럼이 고역쪽으로 기우는 양 [dB/oct].
#: Stevens(1971): 소스 무게중심은 속도와 함께 오른다. 공진을 옮기는 대신 소스
#: 기울기로 준다 — 그래야 개시에서 봉우리 **위치**는 지문 그대로이고 **색깔**만
#: 어두워진다(그게 물리적으로 일어나는 일이다).
SOURCE_TILT_DB_PER_OCT = 1.5


def area_centroid_scale(area: torch.Tensor, a_ref: torch.Tensor | float | None = None,
                        exp: float = FRONT_CAVITY_EXP) -> torch.Tensor:
    """협착 면적 -> 앞공동 공진/반공진 주파수 배율 (**순수 기하**).

    치찰음 필터의 극(앞공동 1/4 파장)과 영점(뒤공동·설하공 반공진)은 성도
    **모양**이 정한다. 협착이 좁게 유지되는 동안에는 모양이 안 변하므로 배율은
    정확히 1.0 이고, 협착이 **열릴 때** 최협착점이 뒤로 물러나며 앞공동이 길어져
    공진이 내려간다(f = c/4Lf).

    `a_ref` 는 그 궤적에서 가장 좁은 면적 — 화자 지문(pole_f 등)이 적합된
    자세다. 그래서 배율은 항상 <= 1.0 이고, 협착이 가장 좁을 때 지문이 **그대로**
    재생된다. 정점 정규화 같은 보정이 필요 없다(예전엔 속도로 재느라 그게
    필요했고, 그 정규화가 개시를 /ʃ/ 로 만들었다).
    """
    if a_ref is None:
        a_ref = area.amin()
    return (a_ref / area.clamp_min(1e-6)).clamp(1e-3, 1.0) ** exp


#: 제트 속도가 기준 자세의 이 비율까지 떨어지면 앞니 다이폴이 사실상 사라진다.
#: Shadle(1985/1990) 의 장애물 소스는 제트가 앞니를 때려서 생기므로, 협착이 덜
#: 극단적이어서 제트가 느리면 다이폴도 약해진다. 지수 2 는 다이폴 소스 세기가
#: 속도의 거듭제곱으로 붙는다는 것(단극보다 가파르다)에서 온다.
OBSTACLE_JET_EXP = 1.0


def obstacle_strength(area: torch.Tensor, a_ref: float | None = None,
                      glottal_area: torch.Tensor | float = 0.12,
                      exp: float | None = None) -> torch.Tensor:
    """앞니(장애물) 공진의 세기 0~1. 기준 자세에서 1.0.

    치찰음 지문은 **길게 끈 /s/** 에서 적합됐다. 거기서는 혀가 목표 자세
    (`a_ref`)에 완전히 도달해 제트가 가장 빠르고, 앞니 다이폴이 스펙트럼을
    지배한다(실측 봉우리 9.7 kHz = 앞니 공명 10015 Hz).

    짧은 CV 의 '사' 는 다르다. 뒤따르는 모음을 예기해 협착이 목표까지 못 가고
    (undershoot) 제트가 느려서, 앞니를 때리는 힘이 약하다. 그러면 다이폴이
    물러나고 **앞공동 극이 드러난다** — 실측 봉우리 4.7~5.6 kHz(앞공동 극
    5274 Hz), 4~6 kHz 에 에너지의 43~48 %.

    **속도로 재면 안 된다.** 협착부 속도는 v = sqrt(2Ps/ρ)/sqrt(Ac²/Ag²+1) 인데,
    무성 마찰음은 성문이 크게 벌어져 있어(Ag 0.25 cm²) Ac ≪ Ag 인 동안 Ac²/Ag²
    가 거의 0 이라 **v 가 압력에만 묶인다**. 측정: 협착을 0.05 -> 0.10 으로 두 배
    넓혀도 속도비가 0.897 로 10 % 밖에 안 움직인다. 판별력이 없다.

    실제로 변하는 건 **제트의 기하**다. 협착이 넓어지면 제트가 굵고 퍼져서
    앞니라는 국소 장애물을 때리는 효율이 떨어진다(Shadle 의 장애물 소스는 좁은
    제트가 모서리에 부딪히는 구조다). 그래서 면적비로 잡는다:

        g = (a_ref / Ac)^exp,  단 1 을 넘지 않는다
    """
    if a_ref is None:
        a_ref = TONGUE_A_MIN
    if exp is None:
        exp = OBSTACLE_JET_EXP        # 호출 시점에 읽는다(기본인자 고정 방지)
    return (torch.as_tensor(float(a_ref)) / area.clamp_min(1e-6)).clamp(0.0, 1.0) ** exp


def glottal_drop_fraction(glottal_area: torch.Tensor,
                          constriction_area_: torch.Tensor) -> torch.Tensor:
    """직렬 저항 중 **성문**이 먹는 압력강하의 비율 0~1 (순수 기하).

    이 파일 머리의 직렬 모형 그대로다: Ps = ½ρU²(1/Ag² + 1/Ac²) 이므로 각
    협착이 먹는 몫은 1/A² 에 비례하고,

        f_glottis = (1/Ag²) / (1/Ag² + 1/Ac²) = Ac² / (Ac² + Ag²)

    난류 소스 진폭은 그 협착의 압력강하 ½ρv² 자체이므로, 이 비율이 곧 **성문
    난류(기식)와 구강 난류(마찰음)의 세기 배분**이다.

    왜 필요한가: `aspiration_source_amp` 의 결과를 제 최대값으로 정규화하면
    이 배분이 지워진다. 그러면 /s/ 처럼 구강 협착(0.10 cm²)이 성문(0.12 cm²)
    보다 좁아 **압력강하의 41 % 만** 성문에 걸리는 구간에서도 기식이 제 최대
    세기로 나온다. 기식은 성도 캐스케이드를 통과하므로 그 몫이 F1/F2 로 나와
    /s/ 정점 스펙트럼의 36 % 가 1~2 kHz 에 쌓였다(실제 /s/ 는 거의 0 이다).
    봉우리가 10 kHz 가 아니라 1.7 kHz 로 잡히는 원인이 이것이다.

    협착이 열리면(해제) 비율이 1 로 가므로 기식이 저절로 제 세기를 되찾는다 —
    마찰음과 목소리를 잇는 그 기식이다. 발성으로 끄는 게 아니라 **기하**로
    배분하는 것이라 전이가 비지 않는다.
    """
    ac2 = constriction_area_.clamp_min(1e-6).pow(2)
    ag2 = glottal_area.clamp_min(1e-6).pow(2)
    return ac2 / (ac2 + ag2)


#: 협착이 없는 중립 성도의 단면적 [cm^2]. 협착의 '좁음' 을 재는 기준이다.
NEUTRAL_TRACT_AREA = 3.0


#: 협착부(혀끝-치경 간극)의 길이 [cm]. /s/ 의 혀끝 협착은 1 cm 안팎이다.
CONSTRICTION_LEN = 1.0
#: 협착 앞쪽 공동(앞니까지)의 길이 [cm]. /s/ 는 1.5 cm 안팎 — `dsp/sibilant.py`
#: 의 앞공동 극(5~8 kHz)과 같은 기하다.
FRONT_CAVITY_LEN = 1.5


def constriction_transmission(constriction_area_: torch.Tensor,
                              a_open: float = NEUTRAL_TRACT_AREA,
                              l_c: float = CONSTRICTION_LEN,
                              l_f: float = FRONT_CAVITY_LEN) -> torch.Tensor:
    """**성문** 소스가 협착을 지나 입술까지 전달되는 비율 0~1 (순수 기하).

    `glottal_drop_fraction` 이 성문에서 난류가 얼마나 **생기는가** 라면, 이건
    그렇게 생긴 소리가 얼마나 **나오는가** 다. 둘은 다른 물리이고 둘 다 있다.

    성문에서 난 잡음이 입술까지 가려면 협착과 앞공동을 지나야 한다. 저역에서
    (파장 >> 길이) 둘 다 관성 임피던스 Z ∝ L/A 이므로, 협착이 없을 때
    (전부 A0) 대비 전달비는 임피던스 분배로

        T = (Lc + Lf)/A0  /  (Lc/Ac + Lf/A0)
          = (Lc + Lf)·Ac / (Lc·A0 + Lf·Ac)

    Ac -> A0 이면 정확히 1 (협착이 협착이 아니게 된다), Ac = 0.10 cm² 이면
    0.079 (약 -22 dB). 단순 면적비 Ac/A0 를 쓰면 협착이 풀리는 도중을 과소평가해
    (해제 중반에 0.23 vs 0.42) 마찰음이 꺼지는 창을 기식이 늦게 메운다.

    왜 필요한가: 이게 없으면 성문 잡음이 성도가 열려 있는 것처럼 방사된다.
    그러면 협착 **뒤에 갇혀 있어야 할** 뒤공동 공진(치경 F2 1750 Hz)이 /s/ 정점
    스펙트럼에 봉우리 대비 -1.8 dB 로 서고(에너지의 9.8 %), 실제 /s/ 에는 거의
    없는 1~2 kHz 성분이 마찰음을 어둡게 만든다.

    협착이 풀리면 1 로 가므로 전이에서 기식이 제 세기를 되찾는다. 발성이 아니라
    **혀**가 여는 것이라 전이가 비지 않는다(성문파열음이 안 된다).
    """
    ac = constriction_area_.clamp_min(0.0)
    return ((l_c + l_f) * ac / (l_c * max(a_open, 1e-6) + l_f * ac).clamp_min(1e-9)
            ).clamp(0.0, 1.0)


def source_tilt_shift(flow: torch.Tensor, area: torch.Tensor,
                      v_ref: torch.Tensor | float | None = None,
                      db_per_oct: float = SOURCE_TILT_DB_PER_OCT) -> torch.Tensor:
    """입자속도 -> 난류 **소스** 기울기 변화 [dB/oct] (Stevens 1971).

    속도가 빠를수록 난류 소스가 고역쪽으로 기운다. 공진 주파수는 건드리지 않고
    소스의 색깔만 바꾼다. `v_ref`(지문이 적합된 속도, 보통 그 구간의 최대 속도)
    에서 0 이고, 속도가 절반이면 `-db_per_oct` 만큼 어두워진다.
    """
    v = particle_velocity(flow, area)
    if v_ref is None:
        v_ref = v.amax()
    return db_per_oct * torch.log2((v / torch.as_tensor(v_ref).clamp_min(1e-6)
                                    ).clamp(0.05, 4.0))


def constriction_area(t: int, frame_rate: float, a_closed: float = 0.10,
                      a_open: float = 3.0, hold: float = 0.5,
                      release: float = 0.12, shape=None,
                      device=None, dtype=torch.float32) -> torch.Tensor:
    """협착 면적 A(t) [cm^2] 궤적 (1,T,1).

    마찰음은 좁게(a_closed≈0.1 cm²=10 mm²) 유지하다가 모음으로 열린다(a_open).
    `hold` 까지 협착을 유지하고 `release`(초) 동안 모음 면적으로 연다.
    `shape=[(위치,면적),...]` 로 직접 줄 수도 있다.

    이 면적 하나가 유량과 함께 (진폭·중심·난류 게이트)를 전부 결정한다 —
    치찰음 파라미터를 손으로 곡선으로 그릴 필요가 없어진다.
    """
    if shape is not None:
        return ramp(t, [(float(p), float(v)) for p, v in shape], device=device)
    dur = max((t - 1) / max(frame_rate, 1e-6), 1e-6)
    r = min(max(release, 1e-3) / dur, 0.9)
    h = min(max(hold, 0.0), 1.0 - r)
    return ramp(t, [(0.0, a_closed), (h, a_closed), (min(h + r, 1.0), a_open),
                    (1.0, a_open)], device=device)


# --- 통합 공기역학 구동: 성문 + 구강 협착이 직렬로 하나의 유량을 만든다 --------
# 폐압 Ps 가 성문(면적 Ag)과 구강 협착(면적 Ac)을 직렬로 통과한다. 각 협착의
# 운동에너지 손실(½ρ(U/A)²)이 더해지므로 (Stevens 1971; Flanagan 직렬저항 모형)
#
#     Ps = ½ρU²(1/Ag² + 1/Ac²)   →   U = sqrt(2·Ps/ρ) / sqrt(1/Ag² + 1/Ac²)
#
# 이 하나의 U 가 성문 기식(성문에서의 난류)과 구강 마찰(협착에서의 난류)을
# 동시에 만든다. 그래서 성대·기식·마찰음이 **같은 구동에서 결합**되어 나온다:
#  * /s/(무성): 성문 열림(Ag 큼)·구강 협착 좁음 → 구강에서 v 커서 마찰음.
#  * /z/(유성): 성문이 떨어 U 가 성문주기로 맥동 → 마찰음이 성문동기로 변조.
#  * /h/·기식: 구강 열림·성문 살짝 열림 → 성문에서 난류(기식).
def series_flow(pressure: torch.Tensor, glottal_area: torch.Tensor,
                oral_area: torch.Tensor) -> torch.Tensor:
    """직렬 협착(성문+구강)을 지나는 부피유량 U [cm^3/s].

    pressure: 성문하압 [dyn/cm^2] (1 cmH2O ≈ 981). glottal_area, oral_area [cm^2].
    """
    ps = pressure.clamp_min(0.0)
    inv = 1.0 / glottal_area.clamp_min(1e-4) ** 2 + 1.0 / oral_area.clamp_min(1e-4) ** 2
    return (2.0 * ps / RHO).sqrt() / inv.sqrt()


def aspiration_source_amp(flow: torch.Tensor, glottal_area: torch.Tensor,
                          re_c: float = RE_C) -> torch.Tensor:
    """성문에서의 기식 난류 소스 진폭. 구강 마찰과 같은 법칙, 위치만 성문.

    성대가 완전히 닫히면(Ag→0) 유량이 없어 기식도 없고, 살짝 열려 압력이 걸리면
    (기식·/h/) 난류가 난다. 성도 전체를 통과하므로 합성기에서 noise_entry=0.
    """
    return frication_source_amp(flow, glottal_area, re_c)


# --- 이빨(장애물) 다이폴 소스 ------------------------------------------------
# 치찰음(/s/ /ʃ/)이 비치찰음(/f/ /θ/)보다 20 dB 넘게 큰 이유는 소스의 종류가
# 다르기 때문이다(Shadle 1985, 1990). 혀끝 협착에서 나온 제트가 **앞니라는
# 장애물에 부딪히며** 다이폴 소스를 만든다. 채널 안에 퍼진 약한 소스가 아니라
# 장애물에 국한된 강한 다이폴이다.
#
# 다이폴은 단극(monopole)보다 방사 효율이 주파수에 비례해 커진다 -> 진폭 스펙트럼이
# **+6 dB/oct 로 상승**한다. 난류 소스 자체는 에디 크기 때문에 코너 위에서
# -6 dB/oct 로 떨어지는데(noise.TurbulenceSource 의 사전), 장애물 다이폴이 그것을
# 상쇄해 고역이 살아난다.
#
# 이게 빠지면 어떻게 되나 (실측 대조): 실측 /s/ 는 에너지의 79% 가 9~12 kHz 에
# 있고 4 kHz 아래는 1.4% 뿐인데, 다이폴 없이 합성하면 9~12 kHz 가 38% 로 줄고
# 5~6 kHz 에 없는 혹이 생기며 저역이 4.8 배 많아진다 -> '스' 로 안 들린다.
#: 실효 기울기 기본 10 dB/oct. 이론적 다이폴은 +6 dB/oct 인데, 합성 경로에는
#: 난류 소스 사전(TurbulenceSource.log_prior)의 -6 dB/oct 롤오프가 **또** 걸린다.
#: 치찰음 지문은 실제 출력 스펙트럼에 맞춰 적합했으므로 그 롤오프를 이미 품고
#: 있어서, 사전의 롤오프가 이중으로 세어진다. +6(다이폴) + 이중계상 상쇄 = 약 10.
#: 실측 대조 L1 오차: 6 dB/oct 0.355, 8 -> 0.219, 10 -> 0.119 (최적).
def obstacle_dipole_bands(n_bands: int, sample_rate: float = 24000.0,
                          db_per_oct: float = 10.0, f_ref: float = 2000.0,
                          f_max_boost: float = 11000.0) -> torch.Tensor:
    """앞니 다이폴의 대역게인 (n_bands,). f_ref 위로 +db_per_oct 로 상승.

    `f_max_boost` 위에서는 더 올리지 않는다(나이퀴스트 근처에서 발산 방지).

    """
    f = torch.linspace(0.0, sample_rate / 2, n_bands).clamp_min(50.0)
    oct_ = torch.log2((f / f_ref).clamp_min(1.0))
    oct_ = torch.minimum(oct_, torch.log2(torch.tensor(f_max_boost / f_ref)))
    if torch.is_tensor(db_per_oct):
        # 시변: (1,T,1) 기울기 -> (1,T,n_bands). 제트가 느려지면 다이폴이
        # 약해지므로 기울기 자체가 프레임마다 달라진다.
        return 10.0 ** (db_per_oct * oct_.reshape(1, 1, -1) / 20.0)
    return 10.0 ** (db_per_oct * oct_ / 20.0)


# --- 구강내압(intraoral pressure)과 발성 억제 ---------------------------------
# /s/ 는 협착이 좁아서(≈0.1 cm²) 협착 **뒤**에 압력이 쌓인다. 그 압력 Pm 은
# 성문을 가로지르는 압력차를 깎는다:
#
#     Ps = ΔP_glottis + ΔP_oral        (직렬 손실)
#     Pm  = ΔP_oral = ½ρ(U/Ac)²        (협착 뒤에 남는 압력 = 구강내압)
#     ΔP_glottis = Ps - Pm             (성대를 실제로 구동하는 압력)
#
# 발성 역치는 **Ps 가 아니라 (Ps-Pm)** 로 판정해야 한다. 협착이 좁으면 Pm 이 커져
# 성대를 구동할 압력이 남지 않아 목소리가 나오다 만다 — 유성 마찰음이 어려운
# 이유이고(Shadle 의 "puzzle of voiced fricatives"), /s/ 가 무성인 이유다.
#
# 그리고 협착을 풀면 Pm 이 **빠르고 부드럽게** 빠지면서 (Ps-Pm) 이 역치를 넘어
# 발성이 붙는다. 그 감쇄 시간이 곧 VOT 다 — 손으로 박는 값이 아니라 유도되는 값.
# 감쇄는 구강의 음향 컴플라이언스가 정한다: C = V/(ρc²), 시상수 τ = R·C.
# V≈60 cm³, c=35000 cm/s 면 τ 는 수 ms 수준이라 '빠르고 부드럽게' 가 맞다.
ORAL_VOLUME = 60.0        # 구강 체적 [cm^3] (협착 뒤 공동)
SOUND_SPEED = 35000.0     # [cm/s]


def intraoral_pressure(flow: torch.Tensor, oral_area: torch.Tensor) -> torch.Tensor:
    """협착 뒤에 걸리는 구강내압 Pm [dyn/cm^2] = 협착에서의 압력강하."""
    v = particle_velocity(flow, oral_area)
    return 0.5 * RHO * v * v


def oral_relax_tau(oral_area: torch.Tensor) -> torch.Tensor:
    """구강내압의 감쇄 시상수 τ [s]. 협착이 열릴수록 빨리 빠진다.

    음향 컴플라이언스 C = V/(ρc²), 협착의 (선형화) 저항 R ≈ ρc/Ac 로 두면
    τ = R·C = V/(c·Ac). Ac=0.1 cm² -> τ≈17 ms(잘 안 빠짐),
    Ac=3 cm²(모음) -> τ≈0.6 ms(즉시 빠짐). 해제가 빠르고 부드러운 이유다.
    """
    return ORAL_VOLUME / (SOUND_SPEED * oral_area.clamp_min(1e-3))


def relax_pressure(pm: torch.Tensor, oral_area: torch.Tensor,
                   frame_rate: float) -> torch.Tensor:
    """구강내압에 1차 지연을 준다 (구강 컴plaiance). pm: (1,T,1).

    정적 Pm 을 그대로 쓰면 협착이 열리는 순간 압력이 계단처럼 사라져 발성이
    '탁' 켜진다. 실제로는 공동에 갇힌 공기가 τ 로 빠져나가므로 부드럽다.
    """
    dt = 1.0 / max(frame_rate, 1e-6)
    tau = oral_relax_tau(oral_area)
    a = (dt / (tau + dt)).clamp(0.0, 1.0)          # 프레임별 계수
    out = torch.zeros_like(pm)
    prev = pm[:, :1, :]
    for i in range(pm.shape[1]):
        prev = prev + a[:, i:i + 1, :] * (pm[:, i:i + 1, :] - prev)
        out[:, i:i + 1, :] = prev
    return out


def transglottal_pressure(ps: torch.Tensor, pm: torch.Tensor) -> torch.Tensor:
    """성대를 실제로 구동하는 압력 (Ps - Pm), 음수는 0 으로."""
    return (ps - pm).clamp_min(0.0)


# --- 구강 공동을 '상태변수' 로: 되밀림(back-pressure)이 유량을 제한한다 -------
# 위의 `intraoral_pressure` 는 **정상상태** 값이다. 실제 구강은 압축성 공동이라
# 압력이 붙었다 빠지는 데 시간이 걸리고, 그 압력이 성문 쪽 유입을 **되밀어**
# 유량을 깎는다. 상태변수 Pm 하나로 쓰면
#
#     C·dPm/dt = U_in - U_out
#     U_in  = Ag·sqrt(2(Ps-Pm)/ρ)     (성문 통과 — Pm 이 클수록 덜 들어온다)
#     U_out = Ac·sqrt(2·Pm/ρ)         (협착 통과 — 마찰음을 만드는 그 유량)
#     C     = V/(ρc²)                 (공동의 음향 컴플라이언스)
#
# 정상상태(U_in=U_out)를 풀면 정확히 `series_flow` 가 나온다 — 같은 모형의
# 정적 극한이다. 달라지는 건 **과도구간**이고, 그게 /s/ 의 시작과 해제다.
#
# 시상수: 두 오리피스의 (선형화) 컨덕턴스 G = dU_in/d(-Pm) + dU_out/dPm 로
# τ = C/G. /s/ 유지 중에는 τ≈1~2 ms 라 Pm 이 사실상 정상상태를 따라가고,
# 협착을 풀면 Ac 가 커져 G 가 급증 -> τ 가 더 짧아져 압력이 **빠르고 부드럽게**
# 빠진다. 그 순간 (Ps-Pm) 이 발성 역치를 넘는다 = VOT.
#
# (예전 `oral_relax_tau` 는 저항을 음향 방사저항 ρc/Ac 로 잡아 τ 를 17~21 ms 로
#  봤는데, 준정적 기류에 맞는 저항은 난류 오리피스 저항이다. 여기서 쓰는 값이
#  그것이고, 같은 '느리게 쌓이고 빠르게 빠진다' 를 올바른 크기로 준다.)
ORAL_COMPLIANCE = ORAL_VOLUME / (RHO * SOUND_SPEED ** 2)   # [cm^5/dyn]


def oral_cavity(ps: torch.Tensor, glottal_area: torch.Tensor,
                oral_area: torch.Tensor, frame_rate: float,
                compliance: float = ORAL_COMPLIANCE) -> tuple:
    """구강 공동을 적분한다. 입력·출력 모두 (1,T,1).

    반환 (pm, u_out): 구강내압 [dyn/cm^2] 과 협착을 지나는 유량 [cm^3/s].

    프레임률(100 Hz)에 비해 τ 가 훨씬 짧으므로(dt/τ≈7) 전진 오일러는 발산한다.
    선형화 해가 정확한 **지수 적분**을 쓴다: 매 프레임 정상상태 Pm_ss 로
    시상수 τ 만큼 지수적으로 다가간다. 두 극한(τ≫dt, τ≪dt)에서 모두 정확하고
    무조건 안정하며 미분가능하다.
    """
    dt = 1.0 / max(frame_rate, 1e-6)
    ag = glottal_area.clamp_min(1e-4)
    ac_ = oral_area.clamp_min(1e-4)
    # 정상상태: series_flow 의 유량으로 만든 협착 압력강하가 그대로 Pm_ss.
    u_ss = series_flow(ps, ag, ac_)
    pm_ss = 0.5 * RHO * (u_ss / ac_) ** 2
    # 선형화 컨덕턴스 -> 시상수 (프레임별)
    dps = (ps - pm_ss).clamp_min(1.0)
    g = ag / (2.0 * RHO * dps).sqrt() + ac_ / (2.0 * RHO * pm_ss.clamp_min(1.0)).sqrt()
    tau = compliance / g.clamp_min(1e-12)
    a = 1.0 - torch.exp(-dt / tau.clamp_min(1e-9))       # 프레임별 접근계수
    pm = torch.zeros_like(pm_ss)
    prev = torch.zeros_like(pm_ss[:, :1, :])
    for i in range(pm_ss.shape[1]):
        prev = prev + a[:, i:i + 1, :] * (pm_ss[:, i:i + 1, :] - prev)
        pm[:, i:i + 1, :] = prev
    return pm, ac_ * (2.0 * pm.clamp_min(0.0) / RHO).sqrt()


#: 성문의 두께 [cm]. 성문 관성 M_g = ρ·Lg/Ag 에 쓴다.
GLOTTAL_LEN = 0.3

#: `oral_cavity_reactive` 의 프레임당 적분 부분단계 수. 헬름홀츠 주기(협착
#: 0.10 cm² 에서 4.4 ms)보다 훨씬 잘게 쪼개야 한다. 100 Hz 프레임에서 50 이면
#: 0.2 ms — 주기의 1/22 라 안정하다.
CAVITY_SUBSTEPS = 50


def oral_cavity_reactive(ps: torch.Tensor, glottal_area: torch.Tensor,
                         oral_area: torch.Tensor, frame_rate: float,
                         volume: float = ORAL_VOLUME,
                         l_c: float = CONSTRICTION_LEN,
                         l_g: float = GLOTTAL_LEN,
                         substeps: int = CAVITY_SUBSTEPS) -> tuple:
    """**관성(리액턴스)까지 넣은** 구강 공동. 반환 (pm, u_c, u_g).

    `oral_cavity` 는 공동을 순수 **컴플라이언스**로만 보고 협착을 저항성 오리피스로
    뒀다 — 1 차 RC 다. 거기엔 **음향 질량이 어디에도 없다.** 그래서 유량이 압력의
    순간값만으로 정해지고(기억이 없다), 진폭이 Ps 의 단조함수가 된다. 그 상태로는
    압력 아치를 완벽히 대칭으로 줘도 가청 포락선의 정점이 49.5 % 에 머문다 —
    "정점을 옮기려면 호흡 구동 자체가 비대칭이어야 한다" 는 결론이 거기서 나왔다.
    **그건 모형에 관성이 없어서 생긴 결론이다.**

    좁은 틈으로 공기를 밀어 넣으려면 그 안의 공기 기둥을 **가속**해야 한다:

        M_c = ρ·Lc/Ac   (협착의 음향 질량. Ac=0.10 cm², Lc=1 cm 면 0.0114 g/cm⁴)
        M_g = ρ·Lg/Ag   (성문도 같다)

    이 질량이 유량의 **변화율**에 저항하므로 유량이 압력을 따라가지 못하고 뒤진다.
    게다가 그 지연이 **유량에 의존한다** — 선형화 저항이 R = ρU/Ac² 라

        τ = M_c/R = Lc·Ac/U

    U=5 cm³/s 에서 20 ms, U=200 에서 0.5 ms. **개시에는 느리고 세지면 빨라진다.**
    이 비대칭은 호흡 제스처가 아니라 공기의 관성에서 나온다.

    그리고 사용자가 지적한 케이스가 여기서 저절로 나온다: 성문으로 들어온 기류가
    **협착을 통과해 마찰음이 되는 대신 공동을 부풀려 Pm 을 올린다**. 관성이 없는
    모형에서는 U_g 와 U_c 가 항상 같아서(직렬 정상류) 이 구간이 존재할 수 없었다.
    여기서는 셋을 따로 적분한다:

        dU_g/dt = (Ps − Pm − ½ρ·U_g|U_g|/Ag²) / M_g
        dU_c/dt = (Pm      − ½ρ·U_c|U_c|/Ac²) / M_c
        dPm/dt  = (U_g − U_c) / C ,        C = V/(ρc²)

    U_g > U_c 인 동안 그 차이가 그대로 공동을 충전한다. 정상상태에서는 U_g = U_c
    가 되어 `series_flow` 와 일치한다 — 즉 기존 모형은 이 모형의 준정상 극한이다.

    적분은 프레임 안에서 영차 유지(zero-order hold) 로 `substeps` 번 전진한다.
    헬름홀츠 공진(협착 0.10 cm² 에서 227 Hz)을 풀어야 하므로 프레임률로는 못 푼다.
    """
    ag = glottal_area.clamp_min(1e-4)
    ac_ = oral_area.clamp_min(1e-4)
    cap = volume / (RHO * SOUND_SPEED ** 2)          # 음향 컴플라이언스 C

    # 부분단계 수는 **협착이 가장 열렸을 때**로 정한다. 헬름홀츠 각진동수
    #   ω = 1/sqrt(M_c·C) = c·sqrt(Ac/(V·Lc))
    # 는 Ac 가 **클수록** 높다(협착 0.10 cm² 227 Hz, 모음 3 cm² 1246 Hz).
    # 좁은 협착 기준으로 잡으면 해제 순간에 발산한다(실제로 NaN 이 났다).
    # 심플렉틱(반음적) 오일러는 dt·ω < 2 에서 안정하므로 여유 있게 dt·ω <= 0.5.
    w_max = float(SOUND_SPEED * (ac_.amax() / (volume * l_c)).sqrt())
    need = int(math.ceil(w_max / (0.5 * max(frame_rate, 1e-6))))
    substeps = max(int(substeps), need)
    dt = 1.0 / (max(frame_rate, 1e-6) * substeps)

    t = ps.shape[1]
    pm_o = torch.zeros_like(ps)
    uc_o = torch.zeros_like(ps)
    ug_o = torch.zeros_like(ps)
    z = torch.zeros_like(ps[:, :1, :])
    pm, u_c, u_g = z, z, z
    for i in range(t):
        p_i, ag_i, ac_i = ps[:, i:i + 1], ag[:, i:i + 1], ac_[:, i:i + 1]
        m_g = RHO * l_g / ag_i
        m_c = RHO * l_c / ac_i
        for _ in range(substeps):
            # **심플렉틱(반음적) 오일러**: 유량을 현재 Pm 으로 먼저 전진시키고,
            # Pm 은 **갱신된** 유량으로 전진시킨다. 전진 오일러는 이 LC 계에서
            # 에너지를 키워 해제 구간(고주파 헬름홀츠)에서 발산한다.
            # 저항 항은 **부호를 살린다**(역류도 표현된다).
            r_g = 0.5 * RHO * u_g.abs() * u_g / ag_i ** 2
            r_c = 0.5 * RHO * u_c.abs() * u_c / ac_i ** 2
            u_g = u_g + dt * (p_i - pm - r_g) / m_g
            u_c = u_c + dt * (pm - r_c) / m_c
            pm = pm + dt * (u_g - u_c) / cap
        pm_o[:, i:i + 1], uc_o[:, i:i + 1], ug_o[:, i:i + 1] = pm, u_c, u_g
    return pm_o, uc_o, ug_o


#: 혀끝 협착의 최소 면적 [cm^2]. 실측 구강내압비에서 나온다.
#: 직렬 모형에서 Po/Ps = Ag²/(Ag²+Ac²) 이므로, Ag=0.12 에서 Ac=0.050 이면
#: Po/Ps = 0.85 — Signorello et al.(2018) 이 [asa]/[isi]/[usu] 의 /s/ 중간점에서
#: 잰 0.85~0.89 와 맞는다. 예전 값 0.10 은 0.59 를 내서 실측과 크게 어긋났다.
TONGUE_A_MIN = 0.050
#: 협착이 안 만들어진 상태(중립 혀 위치)의 면적 [cm^2].
TONGUE_A_REST = 0.60
#: 폐쇄가 해제보다 오래 걸리는 비율. 가청 포락선의 정점 위치와 **같은 수**다
#: (아래 참조). 실측 4 토큰의 상승/하강비 1.28~1.35 -> 정점 56~57 %.
TONGUE_CLOSE_FRAC = 0.57
#: CV(음절)에서 `cv_gesture_times` 에 넣는 정점 위치. 가청 목표(0.62)보다 높다 —
#: 해제와 **동시에 시작되는 내전**이 성문 유량을 깎아 하강을 한 번 더 줄이기
#: 때문이다(면적만 보는 모형에는 그 항이 없다). 0.72 를 넣으면 실제 가청 정점이
#: 59 % 로 나온다(실측 57~68 %).
TONGUE_CV_PEAK = 0.72
#: 음절(CV)에서 혀가 실제로 도달하는 협착 면적 [cm²]. 길게 끈 /s/ 의 목표
#: (`TONGUE_A_MIN` 0.050)보다 **넓다** — 짧은 제스처가 뒤따르는 모음을 예기해
#: 목표까지 못 가기 때문이다(undershoot). 그만큼 제트가 굵어 앞니 다이폴이
#: 약해지고 앞공동 극이 드러난다.
#:
#: 실측 대조로 잡았다: 0.11 에서 봉우리 5276 Hz, 4~6 kHz 50.2 %
#: (실측 4673~5556 Hz, 43~48 %). 0.050 이면 10218 Hz / 17 % 로 지속음과
#: 구분이 안 된다. 구강내압비도 Po/Ps = Ag²/(Ag²+Ac²) = 0.84 로 논문의
#: 0.85~0.89 안에 남는다.
TONGUE_CV_A_MIN = 0.11
#: 제스처 출발 자세를 가청 경계보다 이만큼 넓게 잡는다. 1.0 이면 시작하자마자
#: 소리가 나기 시작하므로 여유가 필요하다 — 페이드 인은 **소리가 안 나는 구간
#: 에서 출발해 경계를 넘어오는 과정**이고, 그 앞부분이 없으면 계단이 된다.
REST_AUDIBLE_MARGIN = 2.2


def min_jerk(x: torch.Tensor) -> torch.Tensor:
    """최소 저크(minimum-jerk) 변위 프로파일 0 -> 1. 양 끝에서 속도·가속도가 0.

    **왜 선형 램프를 쓰면 안 되는가.** 조음기는 질량을 가진 근육이 움직인다.
    변위가 시간에 대해 조각별 선형이면 꺾은점에서 가속도가 무한대이고, 그
    불연속이 유량 -> 난류 진폭으로 그대로 내려가 **계단**이 된다. 측정(합성
    '사', 프레임 10 ms): 마찰음 포락선이 -74.8 -> -58.2 dB 로 한 프레임에
    16.6 dB 뛰었고, 고원 끝에서는 40 ms 만에 -40 -> -128 dB 로 떨어졌다.
    스펙트로그램에서 /s/ 가 **수직 모서리를 가진 직사각형 블록**으로 보인다.
    실측 녹음은 같은 자리가 부드러운 혹이다. 사용자가 "치찰음 중반부에 파열음
    같은 소리" 라고 한 게 이 모서리다.

    조음 운동학의 표준 모형(Nelson 1983; Ostry & Munhall 1985; Perkell)은
    종 모양 속도 프로파일이다. 최소 저크 s(u) = u³(10 - 15u + 6u²) 가 그
    닫힌 해이고, 양 끝에서 속도와 가속도가 모두 0 이라 이어 붙여도 꺾이지 않는다.
    """
    u = x.clamp(0.0, 1.0)
    return u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)


def gesture_fractions(t: int, frame_rate: float, close_s: float,
                      hold_s: float = 0.0, release_s: float = 0.0) -> tuple:
    """제스처 구간 길이[s] -> 세그먼트 안에서의 비율 (닫기, 고원, 열기).

    `tongue_constriction_cv` 와 후두 제스처가 **같은 눈금**을 봐야 해서 따로 뺐다
    (후두 개대의 정점을 고원 한가운데에 놓으려면 고원의 시작을 알아야 한다).
    """
    n = max(int(t), 1)
    dur = max((n - 1) / max(frame_rate, 1e-6), 1e-6)
    cf = min(max(close_s / dur, 1e-3), 0.9)
    hf = min(max(hold_s / dur, 0.0), 0.95 - cf)
    rf = min(max(release_s / dur, 1e-3), 1.0 - cf - hf)
    return cf, hf, rf


#: 모달 발성의 성문 틈 [cm²]. 성대가 주기마다 완전히 닫히고 후두 틈만 남는다.
GLOTTIS_MODAL_AREA = 0.004
#: 개대 개시 -> 정점 [s]. Kim et al.(2022): 성문 개대 개시가 유량 정점을
#: 80~120 ms 앞선다. 되모으는 쪽은 모음을 위한 능동 내전이라 조금 빠르다.
GLOTTIS_ABDUCT_S = 0.10
GLOTTIS_ADDUCT_S = 0.11


def devoicing_gesture(t: int, frame_rate: float, peak_pos: float,
                      abd_area: float, modal_area: float = GLOTTIS_MODAL_AREA,
                      abduct_s: float = GLOTTIS_ABDUCT_S,
                      adduct_s: float = GLOTTIS_ADDUCT_S,
                      device=None) -> torch.Tensor:
    """무성 마찰음의 후두 제스처 — 성문 면적 Ag(t) [cm²] (1,T,1).

    **후두는 계단이 아니라 하나의 매끄러운 열고-닫기다.** Löfqvist & Yoshioka
    (1980, 1981, 1984) 의 투과광 성문도(Pgg)는 무성 마찰음에서 단일한 종 모양
    이고, 정점은 마찰음의 시간적 **중간** 근처다. 예전 구현은 개대를 0.25 로
    **고정**했다가 해제 시점에 꺾어 내렸는데, 그러면

      * 협착이 고원에 있는 동안 Ag 도 상수라 유량이 상수 -> 포락선이 60 ms
        동안 정확히 평평했다(측정: -40.45 dB 가 6 프레임 연속 동일).
      * 고원 끝에서 혀와 후두가 **동시에** 꺾여 40 ms 만에 88 dB 가 사라졌다.

    실제로는 Ac 가 고원이어도 Ag 가 종 모양이면 직렬 유량

        U = sqrt(2·Ps/ρ) / sqrt(1/Ag² + 1/Ac²)

    이 종 모양이 된다. Kim et al.(2022) 이 한국어 /s/ 에서 실제로 잰 구강
    유량이 그 혹이다(정점 0.08~0.12 L/s). 즉 마찰음 포락선의 몸통을 만드는 건
    혀가 아니라 **후두**이고, 혀는 그 위에 시작과 끝을 얹는다.

    면적은 **선형**으로 보간한다(로그가 아니다). 성문은 피열연골 변위에 대해
    삼각형으로 열려서 Ag ~ ½·L·d 로 변위에 거의 비례하고, Pgg 자체가 면적에
    비례하는 신호다 — 그 종 모양을 그대로 옮기려면 선형이 맞다. 혀끝 협착
    (`tongue_constriction_cv`)이 로그 면적인 것과 다른 이유가 이것이다.
    """
    n = max(int(t), 1)
    dur = max((n - 1) / max(frame_rate, 1e-6), 1e-6)
    i = torch.arange(n, device=device, dtype=torch.float32) / max(n - 1, 1)
    pk = min(max(float(peak_pos), 0.0), 1.0)
    ow = max(float(abduct_s) / dur, 1e-3)
    cw = max(float(adduct_s) / dur, 1e-3)
    bell = torch.where(i <= pk,
                       min_jerk((i - (pk - ow)) / ow),
                       1.0 - min_jerk((i - pk) / cw))
    a = float(modal_area) + (float(abd_area) - float(modal_area)) * bell
    return a.reshape(1, n, 1)


#: 지속 마찰음에서 혀가 **목표 자세를 유지하는** 구간의 비율.
#:
#: 0 이면 닫자마자 바로 여는 삼각형이라 혀가 토큰 내내 움직인다. 그러면 앞공동
#: 공진(`area_centroid_scale`)과 앞니 다이폴 세기가 계속 따라 움직여 **음색이
#: 요동친다**. 실측 긴 /s/ 는 반대다 — 스펙트럼 **모양이 거의 안 변하고**
#: 레벨만 변한다(6~11 kHz 기준 2.5~6 kHz 가 12/50/88 % 에서 -1.6/-2.8/-1.2 dB).
#: 사용자가 "사람 소리 같지 않다" 고 한 게 그 요동이다.
#: 지속 마찰음은 자세를 잡고 유지하며, 포락선은 호흡이 만든다.
TONGUE_SUSTAIN_HOLD = 0.62


def tongue_constriction(t: int, frame_rate: float, a_min: float = TONGUE_A_MIN,
                        a_rest: float = TONGUE_A_REST,
                        close_frac: float = TONGUE_CLOSE_FRAC,
                        hold_frac: float = TONGUE_SUSTAIN_HOLD,
                        device=None) -> torch.Tensor:
    """혀끝 제스처로서의 협착 면적 A(t) [cm²] (1,T,1).

    **이게 마찰음의 포락선을 만든다 — 호흡압이 아니다.**

    Signorello, Hassid & Demolin (2018, JASA 143(5) EL386) 이 기관 천자로 Ps 를
    직접 재 보니 마찰음 내내 **Ps 는 거의 일정**하고(8.0 -> 8.9 -> 8.4 hPa),
    변하는 건 **구강내압 Po** 다(2.5 -> 7.6 -> 5.6 hPa). 유량은 양끝이 높은
    **U 자**다(0.5 -> 0.3 -> 0.5 dm³/s). Kim et al.(2022) 은 한국어 /s/ 에서
    "높은 Pio 고원 = 음향 마찰음 길이" 라고 못박았다.

    즉 페이드 인은 **혀가 협착을 만들어 가는 과정**이다. 압력 스웰이 아니다.

    왜 정점 위치가 곧 폐쇄/해제 비율인가
    ------------------------------------
    Ps 가 일정하면 협착부 입자속도는

        v = U/Ac = sqrt(2Ps/ρ) / sqrt(Ac²/Ag² + 1)

    로 **Ac 의 단조감소 함수**다. 난류 진폭은 ½ρv² 이므로 포락선의 정점은
    **협착이 가장 좁은 순간**과 정확히 일치한다. 그리고 가청 경계(정점 대비
    10 %)도 Ac 하나로 정해지므로, 로그 면적에서 일정 속도로 닫고 여는 제스처면

        가청 포락선의 정점 위치 = 폐쇄시간 / (폐쇄시간 + 해제시간)

    가 **정확히** 성립한다. 실측 상승/하강비 1.28~1.35 는 곧 폐쇄가 해제보다
    1.33 배 느리다는 뜻이고, 그게 정점 56~57 % 다. 호흡 구동에 비대칭을 넣을
    이유가 없었다 — 비대칭은 **혀에** 있다.

    로그 면적에서 선형으로 보간한다(조음기가 일정 속도로 움직이면 면적은
    지수적으로 변한다 — 간극이 좁아질수록 같은 변위가 면적을 더 크게 바꾼다).
    """
    n = max(int(t), 1)
    lo, hi = math.log(max(a_min, 1e-6)), math.log(max(a_rest, a_min * 1.01))
    f = min(max(close_frac, 0.05), 0.95)
    h = min(max(hold_frac, 0.0), 0.9)
    # 고원을 정점 위치 f 를 중심으로 close:open 비율대로 나눠 넣는다.
    c_end = f - h * f                       # 폐쇄가 끝나는 지점
    o_beg = f + h * (1.0 - f)               # 해제가 시작하는 지점
    i = torch.arange(n, device=device, dtype=torch.float32) / max(n - 1, 1)
    down = hi + (lo - hi) * min_jerk(i / max(c_end, 1e-6))
    up = lo + (hi - lo) * min_jerk((i - o_beg) / max(1.0 - o_beg, 1e-6))
    logA = torch.where(i <= o_beg, down, up)
    return logA.exp().reshape(1, n, 1)


def audible_area(a_min: float, glottal_area: float, frac: float = 0.10) -> float:
    """가청 경계가 되는 협착 면적 [cm²] — 진폭이 정점의 `frac` 로 떨어지는 지점.

    Ps 가 일정하면 협착부 속도는 v = sqrt(2Ps/ρ)/sqrt(Ac²/Ag²+1) 이므로 진폭
    ½ρv² 의 **면적 의존은 1/(Ac²/Ag²+1) 하나**로 줄어든다(Ps 가 약분된다).
    정점은 Ac = a_min 이고, 거기서 frac 배가 되는 Ac 를 풀면

        Ac = Ag·sqrt( (1/frac)·(a_min²/Ag² + 1) − 1 )

    이 면적이 곧 "혀가 여기까지 좁혀야 소리가 나기 시작한다" 는 경계이고,
    제스처 시간을 가청 길이로 환산할 때 쓰인다(`cv_gesture_times`).

    .. note::
       위 닫힌 해는 레이놀즈 게이트를 뺀 값이다. 협착이 열리면 유속이 떨어져
       게이트가 **더 일찍** 난류를 끄므로 실제 경계는 이보다 좁다. 그래서 닫힌
       해를 상한으로 삼고 게이트까지 포함한 진짜 경계를 수치로 찾는다
       (게이트를 빼면 가청 길이가 110 ms 요청에 80 ms 로 나온다 — 측정).
    """
    ag = torch.tensor(float(max(glottal_area, 1e-9)))
    amin = torch.tensor(float(max(a_min, 1e-9)))
    ps = torch.tensor(8000.0)

    def amp(ac_):
        u = series_flow(ps, ag, ac_)
        return frication_source_amp(u, ac_)

    peak = float(amp(amin))
    r = (a_min / float(ag)) ** 2
    hi = float(ag) * math.sqrt(max((r + 1.0) / max(frac, 1e-6) - 1.0, 1e-9))
    lo = float(amin)
    for _ in range(60):                      # 이분법: amp(A) = frac·peak
        mid = 0.5 * (lo + hi)
        if float(amp(torch.tensor(mid))) > frac * peak:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cv_gesture_times(onset_s: float, peak_frac: float, a_min: float = TONGUE_A_MIN,
                     a_rest: float = TONGUE_A_REST,
                     a_open: float = NEUTRAL_TRACT_AREA,
                     glottal_area: float = 0.12) -> tuple:
    """가청 마찰음 길이와 정점 위치 -> (폐쇄시간, 해제시간) [s].

    독립 마찰음에서는 혀가 왔던 자리로 되돌아가므로 양쪽 로그 거리가 같고,
    그래서 "정점 위치 = 폐쇄/(폐쇄+해제)" 가 그대로 성립했다
    (`tongue_constriction` 주석). **CV 는 다르다** — 해제가 a_rest 가 아니라
    모음 면적까지 열리므로 로그 거리가 한쪽만 길다:

        L_close = ln(a_rest/a_min) = 2.48,   L_open = ln(a_open/a_min) = 4.09

    로그 면적에서 등속으로 움직이면 가청 경계 A_th 까지 걸리는 시간은
    (거리 비례)이므로

        상승 = ln(A_th/a_min)·close_s/L_close,  하강 = ln(A_th/a_min)·release_s/L_open

    이라 정점 위치는 close_s/release_s 가 아니라 그 둘을 각자의 로그 거리로
    나눈 비가 정한다. 같은 close_frac 0.57 을 그대로 CV 에 쓰면 정점이 75 %
    로 밀린다(측정) — 해제가 더 먼 거리를 같은 시간에 가느라 더 빨리 꺼지기
    때문이다. 그래서 CV 의 해제는 **시간으로는 폐쇄보다 길어야** 한다.

    반환값을 그대로 `tongue_constriction_cv` 에 넣으면 가청 길이와 정점 위치가
    요청한 값으로 나온다.
    """
    a_th = audible_area(a_min, glottal_area)
    k = math.log(max(a_th / max(a_min, 1e-9), 1.0 + 1e-9))
    l_close = math.log(max(a_rest / max(a_min, 1e-9), 1.0 + 1e-9))
    l_open = math.log(max(a_open / max(a_min, 1e-9), 1.0 + 1e-9))
    p = min(max(peak_frac, 0.05), 0.95)
    return (onset_s * p * l_close / k, onset_s * (1.0 - p) * l_open / k)



#: 해제 제스처가 폐쇄 제스처보다 몇 배 빠른가 (로그 면적에서의 조음기 속도비).
#: 해제는 근육을 놓는 탄도적 운동이라 능동적으로 좁혀 가는 폐쇄보다 빠르다.
#: 문헌의 조음 운동학은 1.3~1.5 배 범위다(Kent & Moll; Ostry & Munhall).
TONGUE_RELEASE_SPEEDUP = 1.4


def release_from_speed(close_s: float, a_min: float, a_rest: float,
                       a_open: float,
                       speedup: float = TONGUE_RELEASE_SPEEDUP) -> float:
    """폐쇄 시간과 **조음기 속도**로 해제 시간을 낸다 [s].

    `cv_gesture_times` 는 "정점 위치" 를 요구값으로 받아 해제 시간을 거꾸로
    풀었다. 그 풀이는 "포락선 정점 = 최소 면적 순간" 을 전제한다. 폐압 램프를
    되살린 뒤로는(UTTERANCE_PS_RISE_S) 그 전제가 깨졌다 — 정점은 압력이 정하고
    최소 면적은 고원 내내 유지되므로, 정점 위치로 해제 시간을 풀면 물리적으로
    말이 안 되는 속도가 나온다.

    측정: 정점 0.72 를 맞추려면 해제가 80.0 nepers/s 여야 했는데 폐쇄는
    31.1 nepers/s 였다 — 같은 혀끝이 2.6 배 빨리 움직이는 셈이다. 그 결과
    해제 구간에서 진폭이 **10 ms 에 26 dB** 떨어졌고(실측은 45 ms 에 15 dB),
    스펙트로그램에 수직 모서리가 남았다. 사용자가 들은 "치찰음 중반부의
    파열음" 이 이것이다.

    여기서는 정점을 요구하지 않는다. 혀끝은 한 가지 속도로 움직이고(해제가
    `speedup` 배 빠르다), 정점 위치는 압력·면적이 알아서 정하게 둔다.
    """
    l_close = math.log(max(a_rest / max(a_min, 1e-9), 1.0 + 1e-9))
    l_open = math.log(max(a_open / max(a_min, 1e-9), 1.0 + 1e-9))
    return close_s * (l_open / l_close) / max(speedup, 1e-3)

#: 발화 개시에서 폐압이 서는 데 걸리는 시간 [s]. 호흡근이 흉곽을 눌러 압력을
#: 만드는 과정이라 계단일 수 없다.
BREATH_ONSET_S = 0.045

#: **발화 개시**의 폐압 상승 시간 [s]. 위 45 ms 는 문장 안에서 다음 음절로
#: 넘어갈 때의 값이고, 무음에서 말을 시작할 때는 훨씬 길다 — 호기근이 이완압
#: 위로 압력을 세우는 데 100~200 ms 가 걸린다(Draper, Ladefoged & Whitteridge
#: 1959; Hixon). Signorello et al.(2018) 의 "마찰음 내내 Ps 일정" 은 이미 말이
#: 진행 중인 구간의 관찰이지 개시 구간이 아니다.
#:
#: 이게 왜 마찰음 포락선을 정하는가: Ps 가 일정할 때 진폭은
#: ½ρv² = Ps/(1+Ac²/Ag²) 라 **Ps 에 정확히 비례**한다. 그리고 Ac ≪ Ag 면
#: 면적 항이 포화하므로, 협착이 다 만들어진 뒤의 포락선은 **오직 Ps 곡선**이다.
#: 압력을 협착 완성 시점에 이미 다 세워 두면(예전 동작) 포락선에 상승이 없어
#: 마찰음이 수직 모서리를 가진 직사각형이 된다 — 측정: 10->90 % 상승 18 ms,
#: 실측 46~78 ms.
#: 0.15 -> 0.25. `breath_onset` 은 raised cosine 이라 **0~1 전체 길이**를 받고,
#: 그중 10->90 % 구간은 0.58 배다 — 0.25 s 는 상승 145 ms 로 위 문헌 범위(100~200)
#: 안이다. 0.15 로 두면 마찰음이 켜지기 **전에** 이미 압력이 45 % 라, 성문이
#: 벌어지는 동안 성문 난류가 크게 나서 /s/ 앞에 -33 dB 짜리 30 ms 고원이
#: 생긴다(실측 -44~-46). 그 고원 뒤에 마찰음이 따로 올라오니 개시가 2 단이 되고,
#: 그게 "치찰음이 펄스처럼 켜진다" 는 지적이다.
#: 측정(4~12 kHz, /s/ 정점 대비): 선행 기식 / 마찰음-모음 골 / 그 차이
#:   실측       -43.7, -45.8 / -15.6, -15.4 / 28.1, 30.4
#:   rise 0.15  -32.8        / -17.6        / 15.3
#:   rise 0.25  -44.6        / -16.9        / 27.6
UTTERANCE_PS_RISE_S = 0.25


def breath_onset(t: int, frame_rate: float, onset_s: float = BREATH_ONSET_S,
                 delay_s: float = 0.0, device=None) -> torch.Tensor:
    """발화 개시의 폐압 상승 0~1 (1,T,1). 부드러운 S 자(raised cosine).

    **왜 필요한가**: Signorello et al.(2018) 의 "Ps 는 마찰음 내내 거의 일정"
    은 마찰음 **구간 안**의 관찰이다. 발화가 시작될 때 압력이 계단으로 선다는
    뜻이 아니다. 그런데 혀 제스처 모드는 폐압을 상수로 두므로, 세그먼트 첫
    프레임부터 Ps 가 최대다.

    그러면 무슨 일이 나나: 발화 개시에는 혀가 아직 협착을 안 만들었고(성도가
    열려 있고) 성문도 마찰음용으로 크게 벌어져 있다. 직렬 저항이 양쪽 다 낮아
    **유량이 즉시 최대**가 되고, 성문 난류(기식)가 첫 프레임에 최대로 켜진다.
    그 잡음은 성도 **전체**를 통과하므로 포먼트로 울리고, 계단 입력이라
    감쇠 진동 = 파열음 버스트가 된다.

    측정(무음 + '사'): 경계에서 aspiration_bands 가 0 -> 0.1024 로 한 프레임에
    뛰고 파형이 ±0.95 까지 튄 뒤 30 ms 에 걸쳐 감쇠했다. 사용자가 "ksa" 의
    /k/ 로 듣던 것이 이것이다. 마찰음(noise_bands)은 그 시점에 0.0001 로
    정상이었다 — 범인은 마찰음이 아니라 기식이었다.

    `delay_s` 만큼 늦게 시작한다. **램프를 늘리는 것과 늦추는 것은 다르다** —
    압력이 서는 데 걸리는 시간은 생리적으로 40~60 ms 로 고정이고, 조음과
    맞추려면 그 램프를 **옮겨야** 한다. 늘려 버리면(폐쇄 시간 200 ms 에 맞춰
    램프도 200 ms 로) 마찰음 앞 130 ms 가 통째로 안 들리게 된다(측정).
    """
    n = max(int(t), 1)
    k = max(int(round(onset_s * frame_rate)), 1)
    d = max(int(round(delay_s * frame_rate)), 0)
    i = (torch.arange(n, device=device, dtype=torch.float32) - d).clamp_min(0.0)
    return (0.5 - 0.5 * torch.cos(torch.pi * (i / k).clamp(0.0, 1.0))).reshape(1, n, 1)


def tongue_constriction_cv(t: int, frame_rate: float, close_s: float,
                           release_s: float, hold_s: float = 0.0,
                           a_min: float = TONGUE_A_MIN,
                           a_rest: float = TONGUE_A_REST,
                           a_open: float = NEUTRAL_TRACT_AREA,
                           device=None) -> tuple:
    """자음+모음(CV)에서의 혀끝 제스처. 반환 (면적 (1,T,1), 해제 시작 위치 0~1).

    독립 마찰음(`tongue_constriction`)과 다른 점은 **되돌아가지 않는다**는 것이다.
    뒤에 모음이 오므로 혀는 협착을 만들었다가 모음 자세(a_open)로 **열고 만다**.

        a_rest ──(close_s)──> a_min ──(release_s)──> a_open ─────
                                 ↑
                          해제 시작 = 모든 타이밍의 기준점

    왜 이 지점이 기준인가: Ps 가 일정하면(Signorello et al. 2018) 협착부 속도가
    Ac 의 단조감소 함수라 **마찰음 정점이 최소 면적 순간과 정확히 일치**한다.
    같은 순간에 혀가 열리기 시작하고, 구강내압이 빠지기 시작하고, 포먼트 전이가
    출발한다. 그래서 하나의 조음 사건이 세 가지를 한꺼번에 촉발한다 — 타이밍을
    따로 맞출 필요가 없어진다(예전에는 발성 개시에 묶여 있어서 후두를 건드릴
    때마다 포먼트가 딸려 왔다. HANDOFF §9).

    로그 면적에서 선형 보간한다(조음기가 일정 속도로 움직이면 면적은 지수적으로
    변한다 — 간극이 좁을수록 같은 변위가 면적을 더 크게 바꾼다).

    **고원이 있어야 한다.** Kim et al.(2022) 은 한국어 /s/ 에서 "높은 Pio 고원
    이 음향 마찰음 길이의 공기역학적 대응물" 이라고 못박았다. 닫자마자 바로
    여는 삼각형이면 상승만 있고 몸통이 없다 — 게다가 Ps 가 일정할 때 진폭은
    1/(Ac²/Ag²+1) 이라 Ac ≪ Ag 에서 **포화**하므로, 가청 구간이 짧고 가파른
    상승 하나로 뭉친다(측정: 가청 80 ms, 정점 위치 26~32 %, 상승 20 ms —
    실측은 130 ms, 42~57 %, 48 ms).

    반환하는 위치는 **고원의 끝**(해제 시작)이다. 그게 곧 포먼트 전이·내전의
    기준점이다.
    """
    n = max(int(t), 1)
    cf, hf, rf = gesture_fractions(t, frame_rate, close_s, hold_s, release_s)
    lo = math.log(max(a_min, 1e-6))
    hi = math.log(max(a_rest, a_min * 1.01))
    op = math.log(max(a_open, a_min * 1.01))
    i = torch.arange(n, device=device, dtype=torch.float32) / max(n - 1, 1)
    closing = hi + (lo - hi) * min_jerk(i / cf)
    opening = lo + (op - lo) * min_jerk((i - cf - hf) / rf)
    logA = torch.where(i <= cf + hf, closing, opening)
    return logA.exp().reshape(1, n, 1), cf + hf


# --- 호흡 구동압: /s/ 의 페이드 인/아웃이 여기서 나온다 -----------------------
# 실측(업로드 녹음 4 토큰, 4 kHz 이상 대역 포락선):
#
#   토큰        길이     피크위치   상승     하강    상승/하강
#   사 #1      122 ms    57.1 %    70 ms    52 ms     1.35
#   사 #2      128 ms    68.2 %    87 ms    41 ms     2.12
#   긴 /s/ #1 1097 ms    56.1 %   615 ms   482 ms     1.28
#   긴 /s/ #2 1649 ms    56.7 %   935 ms   714 ms     1.31
#
# 두 가지가 읽힌다.
#
# 1. **페이드 인이 소리의 절반을 넘는다** (56~68 %). 이건 협착이나 게이트로는
#    안 나온다: 압력 아치를 완벽히 대칭으로 주고 레이놀즈 게이트 + 제곱법칙을
#    다 태워도 가청 포락선의 피크는 49.5 % 에 그대로 남는다(진폭이 Ps 의 단조
#    함수라 피크가 안 옮겨간다). 즉 **비대칭은 호흡 구동 자체에 있다** —
#    올릴 때가 내릴 때보다 느리다. 좁은 협착 뒤에 압력을 쌓는 건 호기근을
#    능동적으로 밀어 넣는 일이라 느리고, 끄는 건 그 힘을 놓는 일이라 빠르다.
#
# 2. **불변량은 시간이 아니라 비율이다.** 길이가 13 배(122 ms → 1.65 s) 차이나도
#    피크 위치는 56~57 % 로 같다. 그래서 시상수를 초로 박으면 안 되고 **구간
#    길이에 대한 비율**로 잡아야 한다 — 화자가 의도한 길이에 맞춰 힘 주는
#    속도를 조절하는, 계획된 운동 제스처다.
#
# 모형: 신경 구동(사각) -> 1차 지연 2단(근육 활성화 + 힘 발생) -> 압력.
# 오르내림의 시상수를 따로 둔다(능동 수축 vs 이완). 2단이라 꼭짓점이 뾰족하지
# 않고 둥글게 나온다 — 실측 포락선의 모양이 그렇다.
#: 실측 4 토큰에 맞춘 값. 이 셋으로 합성한 가청 포락선은 0.13~2.3 s 구간에서
#: 피크 56.4~57.8 %, 상승/하강 1.29~1.37 로 나온다(실측 56~57 %, 1.28~1.35).
#: 구강내압은 Ps 의 59 % 를 먹는다(문헌 60~70 %).
BREATH_HOLD = 0.40      # 신경 구동이 켜져 있는 비율
BREATH_RISE = 0.22      # 상승 시상수 / 구간길이 (능동 호기근 동원: 느리다)
BREATH_FALL = 0.07      # 하강 시상수 / 구간길이 (힘을 놓는다: 빠르다)
BREATH_PEAK = 0.57      # 가청 포락선의 정점 위치 (실측 56~57 %, 음절도 57~68 %)


def breath_drive(t: int, hold: float = BREATH_HOLD, rise: float = BREATH_RISE,
                 fall: float = BREATH_FALL, floor: float = 0.02,
                 peak: float = 1.0, sustain: float | None = None,
                 device=None) -> torch.Tensor:
    """호흡 구동압 Ps(t)/Ps0 (1,T,1). hold/rise/fall 은 **구간 길이 대비 비율**.

    `sustain` 을 주면 신경 구동이 0 으로 안 떨어지고 그 값에서 유지된다 —
    음절(/사/)에서 마찰음이 끝나도 모음을 낼 압력은 남아 있어야 하기 때문이다.
    그때 마찰음이 꺼지는 건 압력이 아니라 **협착이 열려서**다.
    """
    n = max(int(t), 1)
    drive = torch.full((1, n, 1), 0.0 if sustain is None else float(sustain),
                       device=device)
    drive[:, :max(int(round(hold * n)), 1), :] = 1.0
    ar = 1.0 - math.exp(-1.0 / max(rise * n, 1e-6))
    af = 1.0 - math.exp(-1.0 / max(fall * n, 1e-6))
    out = torch.zeros_like(drive)
    s1 = torch.zeros((1, 1, 1), device=device)
    s2 = torch.zeros((1, 1, 1), device=device)
    for i in range(n):
        d = drive[:, i:i + 1, :]
        s1 = s1 + torch.where(d > s1, ar, af) * (d - s1)
        s2 = s2 + torch.where(s1 > s2, ar, af) * (s1 - s2)
        out[:, i:i + 1, :] = s2
    return floor + (peak - floor) * out.clamp(0.0, 1.0)
