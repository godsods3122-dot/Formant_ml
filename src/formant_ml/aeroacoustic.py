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
