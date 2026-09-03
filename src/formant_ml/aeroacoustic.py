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
    1.0 이 되도록 정규화한 완만한 배율. 협착이 열려 v 가 떨어지면 <1 이 되어
    치찰음 봉우리가 내려간다 — 앞공동이 길어지는 기하 효과와 같은 방향이다.
    sqrt 로 완만하게(속도가 절반이면 중심은 0.7 배).
    """
    v = particle_velocity(flow, area)
    return (v / v_ref).clamp(0.2, 2.5).sqrt()


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
