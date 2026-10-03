"""시간 영역 성도 관 — 성문(비선형 유량)·관(임피던스)·벽·입술 방사·레이놀즈 난류를 한 풀이기로 (MEASUREMENTS §52.476).

사용자: *"포먼트의 주파수 + 위상 응답은 잘 고려하고 있나? … 리액턴스 + 임피던스도 작동되고 있나?"* / *"개편해봐. 그리고 모듈 순서 잘 고려해보고,
레이놀즈 조건을 다시 이용해야 이벤트와 성대가 거의 동시에 움직이는 이벤트를 처리할 수 있을 거 같아."* / *"조화근사 등등이 되어 있다면, 그런 부분도
최대한 비선형으로 맞춰."*

예전 성도는 포먼트 목록을 Klatt 공명기(극만, 최소위상) 직렬로 흉내 냈다 — 위상은 크기에서 따라 나올 뿐 관의 위상(전파·경계 리액턴스)이 아니고, 계수를
표본마다 바꾸는 직접형이라 포먼트가 움직이면 저장 상태가 새 계수와 어긋났다. 성문은 LF 파형을 배음 표로 더한 조화 근사였고 성도를 보지 않았다.
잡음은 따로 만들어 출력에 더했다.

여기서는 **물리 순서대로 한 번에** 푼다:

    폐압 P_s ─ 성문 (면적 A_g(t), 베르누이 + 관성 + 점성: 비선형) ─ 관 N 구간 (면적 A_i(t), 길이 L(t)) ─ 입술 방사 임피던스 ─ 방사 음압
                                       │                    │ 벽 (질량·저항·강성)
                                       └ 성문 난류          └ 협착 난류 (레이놀즈 수가 임계를 넘는 곳에서, 그 순간의 유량으로)

이산화: 엇갈린 격자(압력은 칸 가운데, 체적속도는 칸 경계)의 도약 차분(FDTD). 모든 경계 방정식은 **그 단계에서 암시적으로** 푼다(성문 이차식,
방사 R‖L) — 면적이 0 에 가까워져도(폐쇄) 발산하지 않는다. 벽·점성·열 손실과 방사가 대역폭을, 경계 리액턴스(성문 관성·방사 질량·벽 질량/강성)가
포먼트 자리를 물리대로 옮긴다.

단위 CGS (cm, g, s; 압력 dyn/cm² = 0.1 Pa, 1 cmH2O = 980.7 dyn/cm²).
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
except ModuleNotFoundError:            # pragma: no cover
    njit = None

#: **공기 물성 — 날숨의 조성과 온도에서** (§52.523, `physics/air.py`). 예전 상수는 말소리 문헌의 관례값(음속 350 m/s, 비열비 1.4, 열전도
#: 0.023 W/m·K, 정압 비열 1000 J/kg·K)이었다. 날숨(건조 기준 CO2 4 %, 포화 수증기)을 성도 평균 온도 `AIR_T_C` 에서 계산하면 음속 +0.5 %,
#: 열전도 +11 %, 비열비 −0.8 %, 점성 −2.6 %. 성문(≈36.5 °C) → 입술(≈34.5 °C) 기울기가 평균 대비 포먼트를 옮기는 몫은 F1 에서 ~0.1 % 라 성도는
#: 평균 하나로 둔다 (비강·기관의 영역별 값은 따로). 바꾸려면 첫 커널 호출 전에 `set_air` 를 부른다 (numba 에 굳는 상수).
AIR_T_C = 35.5
AIR_RH = 1.0
from ..physics import air as _air_mod
_AIR = _air_mod.props(AIR_T_C, AIR_RH).cgs()
C_SOUND = _AIR["C_SOUND"]        # cm/s
RHO = _AIR["RHO"]                # g/cm³
MU = _AIR["MU"]                  # dyn·s/cm²
NU = MU / RHO                    # cm²/s
GAMMA = _AIR["GAMMA"]
LAMBDA_TH = _AIR["LAMBDA_TH"]
CP = _AIR["CP"]
#: 영역별 공기 (§52.524). **비강**: 날숨에서 비갑개(아래·중간)가 공기를 식혀 열을 되찾는다 — 상피 표면 비인두 34.3 · 전정 33.5 °C (Pless et al. 2004
#: 날숨 수치 모의, Keck · Lindemann 계열), 비강 공기 평균 `AIR_T_NASAL_C`. **기관**: 심부 체온 37 °C. 부비동 곁관은 비강, 이상와·후두실은 성도 물성.
AIR_T_NASAL_C = 34.5
AIR_T_TRACHEA_C = 37.0
_AIRN = _air_mod.props(AIR_T_NASAL_C, AIR_RH).cgs()
_AIRT = _air_mod.props(AIR_T_TRACHEA_C, AIR_RH).cgs()
C_N, RHO_N, MU_N, GAMMA_N, LAMBDA_N, CP_N = (_AIRN[k] for k in ("C_SOUND", "RHO", "MU", "GAMMA", "LAMBDA_TH", "CP"))
C_T, RHO_T, MU_T, GAMMA_T, LAMBDA_T, CP_T = (_AIRT[k] for k in ("C_SOUND", "RHO", "MU", "GAMMA", "LAMBDA_TH", "CP"))


def set_air(T_c: float = 35.5, rh: float = 1.0, co2_dry: float | None = None) -> dict:
    """공기 물성을 바꾼다 (첫 커널 호출 전에만 — numba 상수). 반환: 새 CGS 값."""
    global C_SOUND, RHO, MU, NU, GAMMA, LAMBDA_TH, CP, AIR_T_C, AIR_RH
    kw = {} if co2_dry is None else {"co2_dry": co2_dry}
    v = _air_mod.props(T_c, rh, **kw).cgs()
    C_SOUND, RHO, MU, GAMMA, LAMBDA_TH, CP = (v[k] for k in ("C_SOUND", "RHO", "MU", "GAMMA", "LAMBDA_TH", "CP"))
    NU = MU / RHO
    AIR_T_C, AIR_RH = T_c, rh
    return v


def set_air_regions(T_nasal: float = 34.5, T_trachea: float = 37.0, rh: float = 1.0) -> None:
    """비강·기관의 공기 물성 (첫 커널 호출 전에만)."""
    global C_N, RHO_N, MU_N, GAMMA_N, LAMBDA_N, CP_N, C_T, RHO_T, MU_T, GAMMA_T, LAMBDA_T, CP_T, AIR_T_NASAL_C, AIR_T_TRACHEA_C
    vn, vt = _air_mod.props(T_nasal, rh).cgs(), _air_mod.props(T_trachea, rh).cgs()
    C_N, RHO_N, MU_N, GAMMA_N, LAMBDA_N, CP_N = (vn[k] for k in ("C_SOUND", "RHO", "MU", "GAMMA", "LAMBDA_TH", "CP"))
    C_T, RHO_T, MU_T, GAMMA_T, LAMBDA_T, CP_T = (vt[k] for k in ("C_SOUND", "RHO", "MU", "GAMMA", "LAMBDA_TH", "CP"))
    AIR_T_NASAL_C, AIR_T_TRACHEA_C = T_nasal, T_trachea
CMH2O = 980.665
# 벽 (단위 면적당). 질량 · 감쇠는 Flanagan 1972. **강성은 Hanna, Smith & Wolfe (2016, JASA 139:2924) 의 생체 측정** (§52.530): 닫힌 성문 · 입술로 잰
# 성도 벽 강성 2–4 kN/m (벽 전체), 실효 두께 1–2 cm, 벽 공진 R0 ≈ 20 Hz — 벽 면적 ~94 cm² 로 나누면 2–4 × 10⁴ dyn/cm³ (예전 Flanagan 값 4 × 10⁵ 은 10–20 배 굳어
# 벽 공진 82 Hz). 굳은 벽은 입 안 순응도가 2 × 10⁻⁴ cm⁵/dyn 뿐이라 장애음 폐쇄에서 입 안 압력이 10 ms 에 폐압까지 찼다 (Rothenberg 1968 의 폐쇄 중 수동 확장
# 0.6–1.6 × 10⁻³). 3 × 10⁴ 이면 공진 22 Hz (3D 모형 `fdtd3d.TISSUE_WALL` 의 19 Hz 와 같은 쪽), m13 재렌더의 유성→무성 20 → 14 %, 포락 같음.
WALL_M = 1.5             # g/cm²
WALL_R = 1.6e3           # dyn·s/cm³
WALL_K = 3.0e4           # dyn/cm³
# 성문 (Ishizaka & Flanagan 1972 의 기하)
GLOTTIS_D = 0.3          # 성대 두께 [cm] — 관성 ρ d / A_g
GLOTTIS_LEN = 1.2        # 성문 길이 [cm] — 점성 12 μ d ℓ² / A_g³
GLOTTIS_KE = 1.0         # 운동 손실 계수 (들어가며 1.37 − 나오며 회복 0.5 의 순 몫 근사)
RE_CRIT = 1800.0
#: 성대 모드의 성문 레이놀즈 수를 실제 성대 길이로 (§52.517). 거짓이면 예전의 고정 `GLOTTIS_LEN`.
RE_USE_LC = True
#: 성문 난류 세기를 협착 난류와 같은 Stevens 척도(동압 × v√A)로 (§52.517). 거짓이면 예전의 동압만.
GLOT_NOISE_STEVENS = True
#: 성문 난류를 **성문 위 관 안의 압력 음원**으로 (§52.518). 성문 유량 식의 압력 항으로 넣으면 그 잡음 압력은 성문이 열린 동안만 유량이 되어 — 음원 식과
#: 무관하게 — 성문 어드미턴스가 잡음을 주기마다 여닫았고, 주기 변조된 잡음은 f0 만큼 밀린 같은 잡음의 합이라 고역 가로줄이 됐다 (§52.517). 실제 기식
#: 잡음은 제트가 가성대·후두개에 부딪히는 자리에서 난다 — 이음매 `GLOT_NOISE_J` (후두 구조가 있으면 1 = 가성대 칸 출구) 에 직렬 압력 음원으로 넣는다.
#: 세기의 평활은 협착 난류와 같은 제트 발달 시간 `GLOT_NOISE_TAU_MS`.
GLOT_NOISE_SUPRA = True
GLOT_NOISE_J = 1
GLOT_NOISE_TAU_MS = 1.0
#: **성문 제트의 도달 몫** (§52.523). 기식 잡음은 성문 제트가 가성대(참성대 위 `GLOT_JET_D` cm)에 부딪혀 난다. 평면 제트는 퍼텐셜 코어(틈 폭의
#: `GLOT_JET_CORE` 배, Rajaratnam 1976)를 지나면 중심 동압이 (코어 길이 / 거리) 로 준다 — 세기 × (1 − exp(−코어/거리)), 틈 폭 = 면적 / 성대 길이.
#: 없으면 1 ms 로 고른 유량을 그 순간의 작은 면적으로 나눈 유속이 닫히는 순간 치솟아 **닫힘마다 잡음이 터졌다** — 주기 안의 짧은 사건은 주기마다
#: 무작위여도 주파수 축 f0 상관(가로줄)을 만든다 (038: 성문 잡음 0.00026 → 0.01 이면 줄 0.21 · 0.23 · 0.11 → 0.35 · 0.34 · 0.35).
GLOT_NOISE_JET = True
#: **성문 제트를 성대 길이 방향으로** (§52.524). 사용자: *"성대 중앙부에서 공기가 나가지만 가장자리는 기류가 상대적으로 느린 것"*. 예전 세기는 1 ms 로
#: 고른 유량을 그 순간의 면적으로 나눈 속도 하나였다 — 닫히는 순간 면적은 줄고 고른 유량은 뒤처져 속도가 치솟아 잡음이 닫힘마다 터졌다. 새 꼴:
#: 그 순간의 유량·면적에서 준정상 압력 강하 Δp = K ρu²/(2A²) + 12μd ℓ² u/A³ (흐름 식과 같은 꼴), 틈을 길이 방향으로 h(s) ∝ sin(πs) (막 첫 모드, ∫ = A/ℓ)
#: 로 `GN_PTS` 점에 두고 점마다 ½Kρv² + 12μd v/h² = Δp 를 풀어 그 자리 속도 — 좁은 가장자리는 점성으로 느리고 가운데가 빠르다. 점마다 Stevens 세기
#: ρv²(v/V_REF)√(a/A_REF) × 제트 도달 몫 × 레이놀즈 전이 (2vh/ν), 점들은 서로 무관한 음원 (세기 제곱합). 1 ms 난류 발달 평활은 유량이 아니라 이 세기에.
GLOT_NOISE_3D = True
GN_PTS = 8
#: **뒤틈(연골부 성문) 제트를 따로** (§52.530). 여성 모달 발성의 60–80 % 는 피열연골 사이 뒤틈이 늘 열려 있다 (Södersten & Lindestad 1990; 이 화자의
#: 적합 뒤틈 0.04 cm², 직류 유량 120–150 cm³/s). 예전에는 뒤틈 면적을 막부 사인 틈 h(s) ∝ sin(πs) 에 섞어 잡음이 막부가 열릴 때만 났다 — 주기마다 같은
#: 자리에 몰린 잡음은 무작위여도 f0 간격 줄을 세운다 (§52.523). 뒤틈은 제 폭 h = A_뒤 / `CHINK_LEN` (여성 연골부 ~0.35 cm, Hirano 1983) 의 점 하나:
#: 같은 성문 압력 강하 Δp 에서 ½Kρv² + 12μd v/h² = Δp 의 속도, Stevens 세기 ρv²(v/V_REF)√(A_뒤/A_REF) × 제트 도달 몫 × 레이놀즈 전이. 막부가 닫히면
#: Δp 가 폐압 가까이 올라 뒤틈 잡음은 닫힌 구간에 가장 세다 — 주기 안에 고르게 퍼진 잡음. 막부 점들은 막부 면적 (A − A_뒤) 으로. 뒤틈 면적에 대한
#: 이 세기의 기울기는 뺀다 (뒤틈 면적의 흐름 몫 기울기는 그대로).
GLOT_NOISE_CHINK = True
CHINK_LEN = 0.35
GLOT_JET_CORE = 5.5
GLOT_JET_D = 0.4
#: 협착 제트의 세기를 **넓어짐의 몫**으로 (§52.519). 예전에는 흐름 방향으로 20 % 넘게 넓어지는 이음매에서만 잡음을 켜고(켜면 온 세기) 그 밑에서는 0 이었다 —
#: 켜짐/꺼짐 스위치라 협착 모양이 조금만 바뀌어도 잡음이 통째로 켜지고 꺼졌고(물리적 도약), 문턱 밑에서는 기울기도 없었다. 무성화 '시' 의 협착이 세 칸에
#: 걸쳐 0.11 · 0.07 · 0.08 cm² 로 고르게 좁아 넓어짐 비가 1.14 라 20 ms 동안 마찰이 꺼졌다 (원본은 10 kHz 위가 가장 센 구간). 제트의 난류는 제트와 둘레의
#: 속도 차가 만든다 — 전단 동압 ρ(v₁ − v₂)² = ρv₁²(1 − A₁/A₂)², 분출 손실(Borda–Carnot, 유량 식의 `dk`)과 같은 몫. 넓어지는 모든 이음매에서 이 몫만큼.
JET_BC = True
#: 제트 발달 거리 [cm] (§52.519) — 협착 출구에서 떨어져 나온 제트는 몇 지름(3 mm 제트의 ~3 배) 하류까지 자유 제트로 퍼진 뒤 벽에 다시 붙는다. 넓어짐 몫은
#: 바로 옆 칸이 아니라 이 거리 안의 가장 넓은 칸과 견준다: ㅅ 출구 0.13 → 0.31 → 2.67 cm² 를 옆 칸으로만 재면 몫 0.34 (실제 제트는 2.67 로 나가 ~0.9)
#: 이고, 입술 칸으로 바로 크게 넓어지는 '시'·ㅊ 은 덜 깎여 — 잡음 세기 상수 하나로는 ㅅ(+6 ~ +9 dB 필요)·'시'(+3)·ㅊ(0)을 함께 못 맞췄다.
JET_LEN_CM = 1.0
#: 제트가 보는 넓어진 면적을 **매끈하게** (§52.523). 발달 거리 안 칸들의 최댓값(argmax)은 넓이가 비슷한 두 칸 사이에서 손실을 꺾고(수반 대 수치 미분
#: 0.2 % 어긋남 — 26·27 칸이 같은 3.0 cm² 인 시험 배치), 거리를 칸 수로 반올림해 길이에 대해 계단이었다. 물리로도 제트는 정해진 거리에서 끊기지
#: 않고 점점 퍼진다 — 거리 d 의 코사인 무게 w = ½(1 + cos(π d / JET_LEN_CM)) (첫 칸 1) 와 무게 준 로그–합–지수 (날카로움 `JET_SOFT_BETA` cm⁻²):
#: 넓이가 같으면 그 값, 발달 거리 0 이면 옆 칸 그대로.
JET_SOFT_BETA = 3.0
#: **난류 음원을 매끈하게** (§52.524). 레이놀즈 문턱 (1 − (Re_c/Re)²)₊, 동압 상한 min(ρv², 2P_s), 제트 판정의 참/거짓 스위치(a₂ > a₁), 좁은 쪽 min(a₁, a₂) 이
#: 면적에 대해 꺾여, 실제 판의 관 입력에서 면적을 흔든 차분이 보폭에 따라 흔들렸다 (잡음을 끄면 수렴). 난류 전이는 날카로운 문턱이 아니라 전이 구간이다:
#: 문턱은 폭 `TURB_GATE_W` 의 C¹ 양수부, 상한은 폭 `TURB_CAP_W` (상한 대비) 의 매끈한 최소, 제트 세기는 흐름 방향 위쪽 칸(목) 면적 a_위 와 아래쪽 넓어진
#: 면적 a_넓 의 (1 − a_위/a_넓)₊² — 스위치 없이 넓어짐이 사라지면 0 으로 간다. 성문 난류에도 문턱·상한을 같은 꼴로.
TURB_SMOOTH = True
TURB_GATE_W = 0.05
TURB_CAP_W = 0.05
         # 난류 임계 레이놀즈 수 (Stevens 1998: 1700~2000)
#: 협착 난류 음원 세기 (§52.481). 동압 ρv² (U²A⁻²) 만으로는 흐름 속도에 대한 의존이 약해, ㅅ 을 원본만큼 내면 모음(협착 Re ~6000)의 8–16 kHz
#: 도 12–16 dB 넘쳤다. Stevens (1971): 난류 음원 크기 ∝ U³ A^−2.5 = v³ A^0.5 — ρv² 에 (v/V_REF)(A/A_REF)^½ 를 곱한다. 제트가 앞니(장애물)에
#: 부딪히는 입술 쪽 `OBS_FRAC` 뒤의 이음매는 `1 + OBS_GAIN` 배 (Shadle 1985: 장애물 음원은 벽 음원보다 10–20 dB 세다).
V_REF = 3000.0           # cm/s
A_REF = 0.1              # cm²
OBS_FRAC = 0.9
OBS_GAIN = 5.0
FS_SIM = 96000.0         # 내부 표본률 — 출력(48 kHz)의 두 배
N_SECT = 28              # 관 구간 수 — L ≥ 11 cm 에서 쿠랑 수 c·dt/dx ≤ 0.93
A_FLOOR = 1e-4           # 폐쇄 면적 하한 [cm²] (0 이면 식이 특이하다)
#: 경계층(점성·열) 손실 (§52.482). 이음매의 직렬 임피던스 Z_v = (S dx/A²)·√(ρμ)·√(jω), 칸의 병렬 어드미턴스 Y_t = (S dx/ρc²)(γ−1)√(λ/ρc_p)·√(jω)
#: — 크기가 √ω 로 크고 저항과 리액턴스가 같다. 예전에는 1 kHz 한 점의 저항으로 고정해, 고역 공명의 대역폭이 모자랐다.
#: 시간 영역에서는 확산 표현으로 푼다: √(jω) ≈ Σ w_m · jω/(jω + ξ_m) — 가지마다 병렬 R–L (w_m ≥ 0 이면 수동), 상태 φ_m 은 흐름(압력)의 1 차
#: 저역통과. 극 넷(`DIFF_HZ`)과 무게는 이산 가지 응답에 맞춘다 (60 Hz–16 kHz 실수부 15 %·허수부 6 % 안). `REF_W` 는 계수의 정규화 기준이다.
REF_W = 2.0 * math.pi * 1000.0
DIFF_HZ = (30.0, 330.0, 2600.0, 5.0e5)
#: 젖고 무른 점막의 손실 배율. Hanna, Smith & Wolfe (2016, JASA 139:2924): 입술로 잰 실제 성도 공명(성문 닫힘) 대역폭 — 여성 0.3–4 kHz 70–90 Hz —
#: 은 매끈하고 마른 강체 관 이론의 감쇠 계수를 **5 배** 해야 맞는다 (표면적/부피 비, 젖은 면, 무른 벽). 비강의 넓은 표면은 `NAS_LOSS` 가 따로 곱한다.
LOSS_WET = 5.0


def _check():
    if njit is None:
        raise RuntimeError("tube_td 는 numba 가 필요하다")
    _nb_guard()


#: 비강 (연구개 포트 → 콧구멍). 기하는 화자 상수로 고정 — 적합하는 것은 포트 면적(연구개 열림)뿐이다.
N_NAS = 16
NAS_LEN = 11.5           # cm (연구개 → 콧구멍, Dang et al. 1994)
NAS_AREA = np.array([1.2, 1.4, 1.6, 2.0, 2.4, 2.8, 3.0, 3.0, 2.8, 2.4, 2.0, 1.6, 1.2, 0.9, 0.7, 0.6])
NAS_LOSS = 4.0           # 점막·비갑개의 넓은 표면 — 경계층 손실을 구강의 이만큼으로
#: **VTL 비강** (Birkholz VocalTractLab 2.4 `Tube.cpp` — Dang & Honda 1994 피험자 4 실측, 0.57 cm × 19 칸 = 10.8 cm, 연구개 쪽 → 콧구멍) 와
#: 부비동 넷 (Dang & Honda 1996 표 V: 부피 11.3·6.8·33.0·6.2 cm³, 목 0.3·0.3·0.45·1.0 cm × 0.185·0.185·0.145·0.11 cm², 붙는 칸 8·9·11·12)
#: — `set_nose("vtl")` (§52.490). 부비동은 헬름홀츠 곁관(공동 길이 0.8 cm, 면적 = 부피/0.8)으로.
#: **기관** (성문 아래 관, §52.491) — 면적 (NT,) · 길이. 비면 폐압이 성문에 바로 걸린다 (예전). `set_trachea`.
TRACHEA_AREA = np.zeros(0)
TRACHEA_LEN = 0.0


def set_trachea(on: bool = False) -> None:
    """기관 켜기/끄기. 여성 성문 아래 공명 Sg1 ≈ 700 Hz · Sg2 ≈ 1.6 kHz (Lulich 2010 계열) 에 맞춘 모양 (`out/_tmp/vf/calib_trachea.py`):
    전체 19 cm · 14 칸 — 성문 쪽 40 % 는 기관 1.8 cm², 폐 쪽 60 % 는 10 cm² 에서 좁아지는 기관지 (기하 등비) · 끝은 정합 저항 ρc/A 로 흡수 →
    성문 아래 임피던스 봉우리 680 · 1675 Hz. 균일한 관은 Sg2 ≈ 3·Sg1 (1/4 파장 홀수배)라 실제 비 2.3 을 못 낸다."""
    global TRACHEA_AREA, TRACHEA_LEN
    if not on:
        TRACHEA_AREA, TRACHEA_LEN = np.zeros(0), 0.0
        return
    n, nb = 14, int(round(0.6 * 14))
    TRACHEA_AREA = np.r_[np.geomspace(10.0, 1.8 * 1.3, nb), np.full(n - nb, 1.8)]
    TRACHEA_LEN = 19.0


NOSE_VTL_AREA = np.array([1.30, 1.25, 1.30, 1.20, 1.60, 2.50, 3.60, 2.25, 2.50, 1.75, 2.50, 2.10, 1.50, 1.50, 0.75, 1.25, 1.60, 1.65, 1.40])
NOSE_VTL_LEN = 0.57 * 19
SINUS_VTL = [("N8", 11.3 / 0.8, 0.8, 0.185, 0.3), ("N9", 6.8 / 0.8, 0.8, 0.185, 0.3),
             ("N11", 33.0 / 0.8, 0.8, 0.145, 0.45), ("N12", 6.2 / 0.8, 0.8, 0.11, 1.0)]
_NOSE_DEFAULT = (NAS_AREA.copy(), NAS_LEN)


def set_nose(kind: str = "default", length_scale: float = 1.0, area_scale: float = 1.0) -> None:
    """비강 기하를 바꾼다: "default" (예전 16 칸 11.5 cm) 또는 "vtl" (19 칸 + 부비동은 `SIDE_BRANCHES` 에 따로 더한다)."""
    global NAS_AREA, NAS_LEN, N_NAS
    a, l = (NOSE_VTL_AREA, NOSE_VTL_LEN) if kind == "vtl" else _NOSE_DEFAULT
    NAS_AREA = np.ascontiguousarray(np.asarray(a, float) * area_scale)
    NAS_LEN = float(l) * length_scale
    N_NAS = NAS_AREA.shape[0]
VEL_FRAC = 0.45          # 연구개 포트의 자리 (성문 0 → 입술 1)
VEL_D = 0.4              # 포트 두께 [cm] — 관성 ρ d / A_v
#: 연인두 포트의 **분출(베르누이) 손실 계수** (§52.530) — R_k = K ½ρ|U|/A_v² (한 걸음 앞 |U|, 소산이라 안정). 포트 저항이 푸아죄유 8πμd/A² 뿐이면
#: 거의 닫힌 포트(0.005 cm²)로 입 안 14 cmH2O 에서 ~180 cm³/s 가 샜다 (KWF_m13 ㄲ 폐쇄, 연인두 상한을 건 판) — 좁은 구멍의 흐름은 분출 손실이 정한다
#: (U = A √(2Δp/ρ) ≈ 25 cm³/s). 성문 · 구강 이음매에는 이미 있는 항.
VEL_KE = 1.0
A_VMAX = 1.2             # 포트 최대 면적 [cm²]
#: 이상와 (piriform fossae) — 후두개관 위에서 갈라지는 **닫힌 곁관** 한 쌍 (합친 면적). 1/4 파장 반공명이 고역에 골을 판다:
#: 남성 4–5 kHz (Dang & Honda 1997), 3D 인쇄 성도에서 3.9–4.7 kHz · 길이 17.6–21.2 mm · 성도 부피의 2–8 % (Delvaux & Howard 2014),
#: 여성은 5–6 kHz. 기하는 화자 상수(적합하지 않는다). 면적 0 이면 없다.
N_PIR = 2               # 칸 (칸 길이 ≥ c·dt 이도록 짧게 — 1/4 파장 골 자리는 PIR_LEN 으로 맞춘다)
PIR_LEN = 1.2            # cm — 038 모음의 6–6.5 kHz 골에 맞춘 값 (2 칸 이산화 + 입구 관성이라 유효 길이는 이보다 길다)
PIR_AREA = 0.8           # cm² (양쪽 합)
PIR_FRAC = 0.12          # 갈라지는 자리 (성문 0 → 입술 1) — 후두개관 윗끝 ≈ 2 cm
#: **닫힌 곁관 목록** (§52.489) [(붙는 자리 — 성문 0 → 입술 1 또는 "N<칸>" 비강 칸, 면적 cm², 길이 cm[, 목 면적 cm², 목 길이 cm]), …]. 목을 주면 첫 이음매의 관성·
#: 손실이 목의 것이 된다 — 헬름홀츠 곁공동 (곁관 칸은 안정 조건상 c·dt = 0.365 cm 보다 길어야 해서, 1/4 파장만으로는 ~9.7 kHz 위를 못 낸다). None 이면 이상와 하나 (PIR_FRAC, PIR_AREA, PIR_LEN).
#: 좌우 이상와를 따로(길이가 다르면 반공명이 둘), 성문 바로 위의 후두실(모르가니 굴)을 곁공동으로 둘 수 있다 — `voice_td.LARYNX`.
SIDE_BRANCHES = None


def side_branches(N: int, pir=None):
    """(SBA, SBL, SBI) 배열. pir (면적, 길이) 를 주면 예전 뜻의 이상와 하나 (시험·옛 판)."""
    if pir is not None:
        br = [(PIR_FRAC, pir[0], pir[1])]
    else:
        br = SIDE_BRANCHES if SIDE_BRANCHES is not None else [(PIR_FRAC, PIR_AREA, PIR_LEN)]
    SBA = np.array([float(b[1]) for b in br])
    SBL = np.array([max(float(b[2]), 1e-3) for b in br])
    SBI = np.array([N + int(b[0][1:]) if isinstance(b[0], str) else min(N - 1, max(0, int(b[0] * N))) for b in br], dtype=np.int64)
    SBNA = np.array([float(b[3]) if len(b) > 3 else 0.0 for b in br])            # 목 면적 (0 이면 목 없음 — 곁관 면적 그대로)
    SBNL = np.array([float(b[4]) if len(b) > 4 else 0.0 for b in br])            # 목 길이
    return SBA, SBL, SBI, SBNA, SBNL


#: **자기 진동 성대** — Birkholz 의 삼각 성문 두 질량 모형 (VocalTractLab 2.4 `TriangularGlottis`, MEASUREMENTS §52.488). 면적을 입력으로
#: 받지 않고, 아래·위 두 질량의 변위가 폐압·성문 위 압력에 밀려 스스로 떤다. 성대는 앞맞붙음(앞)에서 성대돌기(뒤)까지 곧은 두 날로,
#: 뒤쪽 쉼 변위 r (내전이 정한다)만큼 벌어진 삼각형이다 — 날이 닿으면 앞에서부터 닫힌다 (닿은 몫 α, 접촉 용수철 k_c·α).
#: 긴장 Q 가 질량 m/Q · 용수철 k·Q · 두께 d/√Q · 길이 L√Q 를 바꿔 f0 를 정한다 (f0 = 고유 f0 + dF0/dQ·(Q − 1)).
#: 정적 변수 `VF_STATIC` 의 차례: 길이 L, 두께 d1 d2, 질량 m1 m2, 감쇠비 ζ1 ζ2, 용수철 k1 k2, 접촉 kc1 kc2, 결합 kp, 입구·출구 길이.
#: 값은 W02 화자 파일의 것 (VTL 기본값과 같다). CGS.
VF_STATIC = np.array([1.3, 0.24, 0.06, 0.12, 0.03, 0.1, 0.6, 80000.0, 8000.0, 240000.0, 24000.0, 25000.0, 0.05, 0.01])
VF_NATURAL_F0 = 130.979962     # Q = 1 에서의 f0 [Hz] (W02)
VF_DF0_DQ = 132.065991         # [Hz]


@njit(cache=True)
def _vf_geom(x, r, Lc):
    """한 질량의 (막부 열린 면적, 열린 길이, 닿은 몫 α, 접촉 중심의 쉼 변위 r*) — 상대 변위 x, 뒤쪽 쉼 변위 r, 성대 길이 Lc.
    r ≥ 0: 삼각형 (앞 변위 x, 뒤 변위 r + x), r < 0: 눌려 평행 (앞·뒤 모두 r + x). 틈의 폭은 두 날 변위의 합이다."""
    if r >= 0.0:
        if x > 0.0:
            return Lc * (r + 2.0 * x), Lc, 0.0, r
        if r + x > 0.0 and r > 0.0:
            b = r + x                               # 뒤만 열림 — 꼭짓점이 날 안에 있다
            return Lc * b * b / r, Lc * b / r, -x / r, -0.5 * x
        return 0.0, 0.0, 1.0, 0.5 * r               # 닫힘 (변위가 NaN 이어도 여기로 — 0 나눗셈을 막는다)
    b = r + x
    if b > 0.0:
        return 2.0 * Lc * b, Lc, 0.0, r
    return 0.0, 0.0, 1.0, r


@njit(cache=True)
def _vf_geom_d(x, r, Lc):
    """`_vf_geom` 의 편미분 — (∂면적, ∂열린 길이, ∂α, ∂r*) 각각 (/∂x, /∂r, /∂Lc), 12 개."""
    if r >= 0.0:
        if x > 0.0:
            return 2.0 * Lc, Lc, r + 2.0 * x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
        if r + x > 0.0 and r > 0.0:
            b = r + x
            return (2.0 * Lc * b / r, Lc * b * (r - x) / (r * r), b * b / r,
                    Lc / r, -Lc * x / (r * r), b / r,
                    -1.0 / r, x / (r * r), 0.0,
                    -0.5, 0.0, 0.0)
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0
    b = r + x
    if b > 0.0:
        return 2.0 * Lc, 2.0 * Lc, 2.0 * b, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0



#: **성대 모드 3 — N 줄 덮개 + 몸체** (§52.504). 보–막 성대(Serry et al. 2026)를 성대 길이 방향의 첫 정재파 sin(πx/L) (막 밴드의 k = π/L 가지)
#: 로 투영하고, 상하(아래 = 입구)를 `VF_NSTRIP` 줄로 푼다. 줄 k 의 반틈새 h(s) = r_k·s + x_k·sin(πs) (s = x/L, r_k 뒤끝 쉼 변위, x_k 줄 변위 —
#: 바깥이 +). 줄 사이 덮개 전단·줄–몸체 층간 결합(입구 쪽이 `VS[15]` 배 굳다)·장력·몸체(인대+갑상피열근 보)의 축력·굽힘·기초, 갑상피열근의
#: 굽힘 모멘트(몸체를 안쪽으로 미는 정적 하중) 가 모두 틀마다의 물리 계수(VR 열 3–15)로 들어온다 — `beam_membrane.ns_coefs`.
VF_NSTRIP = 6
VF_NS_PTS = 16
#: **젖은 점막 — 점액막 두께의 흔들림** (§52.531). 사용자: *"성대 영상을 보면, 성대 내부가 굉장히 물기가 많고, 성대 영상에서 점액질이 늘어난 모습이 보일
#: 정도야"*. 맞닿는 겉면은 점액막 두께만큼 먼저 닿는다 — 그 두께가 성대 길이의 자리마다 · 주기마다 달라 (점액이 뭉치고 늘어나 끊어진다) 닫힘의 순간과
#: 모양이 주기마다 조금씩 다르다. 성대 길이 `VF_NS_PTS` 점마다 평균 0 · 표준편차 `VF_NS_FILM_UM` [μm] 의 흔들림 φ_i(t) (상관 시간 ~ 한 주기:
#: `VF_NS_FILM_HZ` 2 차 저역통과, 이웃 점과 [¼ ½ ¼] 로 이어짐), 유효 반틈새 h − φ_i — 접촉 · 열린 몫 · 면적이 모두 같은 h − φ 를 쓴다 (정방향 · 수반 같다).
#: 실현은 관 씨앗에 묶인다 (재현). 0 이면 끈다. 파이썬 쪽 상수라 실행 중 바꿀 수 있다.
VF_NS_FILM_UM = 0.0
VF_NS_FILM_HZ = 150.0
#: 점액막 흔들림이 미치는 거리 [cm] — 맞닿음 근처에서만 (§52.531). 유효 반틈새 h − φ·exp(−h⁺/δ): 겉면끼리 가까울 때(닿고 떨어질 때) 막 두께가 그대로 들고,
#: 크게 열린 성문(1 mm 넘는 틈)에서는 사라진다. 흔들림을 틈 전체에 두었더니 (δ = ∞) 열린 동안의 면적이 3–4 % 씩 흔들려 저역 주기 지터 2.5–5 %,
#: HNR 8.8–12.3 dB, 40–80 Hz 변조 +11 dB 로 거칠어졌다. numba 상수 (소스에서 바꾼다). 0 이면 틈 전체.
VF_NS_FILM_DECAY = 8.0e-3


def _film_field(T: int, seed: int) -> np.ndarray:
    """점액막 두께의 흔들림 (T + 2, VF_NS_PTS) [cm] — `VF_NS_FILM_UM`."""
    out = np.zeros((T + 2, VF_NS_PTS))
    if VF_NS_FILM_UM <= 0.0:
        return out
    from scipy.signal import butter, sosfilt
    rng = np.random.default_rng(9_000_011 + 31 * int(seed))
    pad = 4096
    x = rng.standard_normal((T + 2 + pad, VF_NS_PTS))
    x = sosfilt(butter(2, VF_NS_FILM_HZ, "low", fs=FS_SIM, output="sos"), x, axis=0)[pad:]
    xs = 0.5 * x
    xs[:, 1:] += 0.25 * x[:, :-1]
    xs[:, :-1] += 0.25 * x[:, 1:]
    xs /= max(float(xs.std()), 1e-12)
    return VF_NS_FILM_UM * 1e-4 * xs


#: 열림/닫힘 경계의 경사 폭 [cm] — 적분점의 반틈새 h 가 ±δ/2 안이면 열린 몫 w = h/δ + ½ (§52.504). 판마다의 값은 정적 배열 VS[18]
#: (`beam_membrane.CLOSURE_EDGE_CM`, §52.530) 로 커널에 넘긴다 — 이 상수는 VS[18] 이 0 일 때의 기본. 지시 함수(h > 0)로 두면 점 하나가
#: 경계를 건널 때마다 압력 힘·접촉 적분이 계단처럼 뛰어 수반이 그 뜀을 보지 못했고, 긴 적합에서 기울기가 어긋나 무너졌다 (BLB 65.9 → 0.15 %).
VF_NS_EDGE = 5.0e-3
#: 흐름 분리 자리의 연속 폭 — 줄 j 가 가장 좁은 면적 ag 에 얼마나 가까운지 w_j = clamp(1 − (G_j − ag)/(δ·ag), 0, 1), 줄 k 의 베르누이 몫
#: χ_k = 1 − max_{j≤k} w_j (입구에서 k 까지 좁은 목이 없으면 1). "가장 좁은 줄보다 입구 쪽이면 베르누이" 로 두면 가장 좁은 줄이 이웃 아닌 줄로
#: 건너뛸 때 사이 줄의 압력이 한 번에 뛰어, 실제 적합 손실이 제어에 대해 뜀이 있는 함수가 되었다 (방향 차분이 간격을 줄일수록 커졌다 — §52.504).
VF_NS_SEP = 0.3
#: 두 질량 성대(모드 1·2)의 흐름 면적 = 두 틈의 매끈한 최소 (a1^−p + a2^−p)^(−1/p) (§52.515) — 흐름 분리는 계단이 아니라 연속으로 옮겨 간다
#: (Pelorson et al. 1994). min(a1, a2) 의 모서리는 닫히기 시작하는 순간(수렴 → 발산) 유량 미분을 꺾어 주기마다 같은 모양의 고역 울림을 냈고,
#: 그것이 5–16 kHz 의 고른 배음 빗살(가로줄)이었다 — 지터 4 % 로도 안 지워졌다 (같은 모양 펄스열의 미세 구조는 주파수축으로 f0 마다 되풀이된다).
#: 모드 3 은 연속 분리(`VF_NS_SEP`)라 줄이 없었다. 0 이면 예전의 min. numba 에 굳는 상수라 소스에서 바꾼다.
VF_SEP_P = 6.0
#: **점막 닫힘을 매끈하게** (§52.523). 성대 길이 16 점의 양수부 max(h, 0) 은 점이 하나씩 닫힐 때마다 면적 기울기를 꺾었고, 닿음 경사
#: clamp(h/EDGE + ½) 는 양 끝에서 꺾였고, 흐름 분리는 가장 좁은 줄의 hard min 과 달리는 최댓값이라 그 줄이 바뀔 때 꺾였다 — 손실이 제어에 대해
#: 매끈하지 않았다 (몸체 구동판의 기울기 점검이 차분 보폭에 따라 뒤집혔다). 점막 표면의 점액막·무른 상피는 닿음을 `VF_NS_EDGE` (50 μm) 폭에 걸쳐
#: 점진적으로 만든다: 양수부는 그 폭의 C¹ 이차 이음 (h+e)²/(4e), 닿음 무게는 smoothstep, 분리 기준 면적은 줄들의 매끈한 최소, 분리 몫은
#: χ_k = Π_{j≤k} (1 − w_j) (w 가 0/1 이면 1 − max 와 같다), w 는 smoothstep. 거짓이면 예전 꼴.
VF_NS_SMOOTH = True


@njit(cache=True)
def _ns_geom(x, r, Lc, edge, ph):
    """줄 하나의 기하 — 길이 Lc 를 곱한 값 (두 성대의 면적, 나머지는 한쪽 성대의 모드 투영):
    (면적 a = 2Lc∫max(h,0), 열린 몫 O = Lc∫w·sin, 닿은 몫 S = Lc∫(1−w)·sin², 접촉 쉼 몫 C = Lc∫(1−w)·r·s·sin, ∂a/∂r, ∂a/∂x).
    edge — 닫힘 이음 폭 [cm] (VS[18], §52.530; 0 이하면 `VF_NS_EDGE`). ph — 점마다 점액막 두께 흔들림 [cm] (§52.531): 유효 h − ph."""
    if edge <= 0.0:
        edge = VF_NS_EDGE
    a = 0.0
    o = 0.0
    s2 = 0.0
    cf = 0.0
    dar = 0.0
    dax = 0.0
    ds = 1.0 / VF_NS_PTS
    e2 = 0.5 * edge
    for i in range(VF_NS_PTS):
        sv = (i + 0.5) * ds
        sn = math.sin(math.pi * sv)
        h0 = r * sv + x * sn
        if VF_NS_FILM_DECAY > 0.0 and h0 > 0.0:
            gf = math.exp(-h0 / VF_NS_FILM_DECAY)
            h = h0 - ph[i] * gf
            dh = 1.0 + ph[i] * gf / VF_NS_FILM_DECAY
        else:
            h = h0 - ph[i]
            dh = 1.0
        if VF_NS_SMOOTH:
            if h >= e2:
                a += h
                dar += sv * dh
                dax += sn * dh
            elif h > -e2:                               # 점액막 폭의 C¹ 이음 (§52.523)
                a += (h + e2) * (h + e2) / (4.0 * e2)
                sp1 = (h + e2) / (2.0 * e2)
                dar += sp1 * sv * dh
                dax += sp1 * sn * dh
            u = min(max(h / edge + 0.5, 0.0), 1.0)
            w = u * u * (3.0 - 2.0 * u)
        else:
            if h > 0.0:
                a += h
                dar += sv
                dax += sn
            w = min(max(h / edge + 0.5, 0.0), 1.0)
        o += w * sn
        s2 += (1.0 - w) * sn * sn
        cf += (1.0 - w) * (h - x * sn) * sn
    return 2.0 * Lc * a * ds, Lc * o * ds, Lc * s2 * ds, Lc * cf * ds, 2.0 * Lc * dar * ds, 2.0 * Lc * dax * ds


@njit(cache=True)
def _ns_geom2(x1, x2, r, Lc, edge, ph):
    """둘째 길이 모드까지 (§52.531): 반틈새 h(s) = r·s + x1·sin(πs) + x2·sin(2πs). (면적 a, 열린 몫 O1 · O2, 닿은 몫 S11 · S12 · S22,
    접촉 쉼 몫 C1 · C2, ∂a/∂r, ∂a/∂x1, ∂a/∂x2) — `_ns_geom` 과 같은 규약 (길이 Lc 를 곱한 값)."""
    if edge <= 0.0:
        edge = VF_NS_EDGE
    a = 0.0
    o1 = 0.0
    o2 = 0.0
    s11 = 0.0
    s12 = 0.0
    s22 = 0.0
    c1 = 0.0
    c2 = 0.0
    dar = 0.0
    da1 = 0.0
    da2 = 0.0
    ds = 1.0 / VF_NS_PTS
    e2 = 0.5 * edge
    for i in range(VF_NS_PTS):
        sv = (i + 0.5) * ds
        f1 = math.sin(math.pi * sv)
        f2 = math.sin(2.0 * math.pi * sv)
        h0 = r * sv + x1 * f1 + x2 * f2
        h = h0 - ph[i] * (math.exp(-h0 / VF_NS_FILM_DECAY) if (VF_NS_FILM_DECAY > 0.0 and h0 > 0.0) else 1.0)
        if h >= e2:
            a += h
            dar += sv
            da1 += f1
            da2 += f2
        elif h > -e2:
            a += (h + e2) * (h + e2) / (4.0 * e2)
            sp1 = (h + e2) / (2.0 * e2)
            dar += sp1 * sv
            da1 += sp1 * f1
            da2 += sp1 * f2
        u = min(max(h / edge + 0.5, 0.0), 1.0)
        w = u * u * (3.0 - 2.0 * u)
        o1 += w * f1
        o2 += w * f2
        s11 += (1.0 - w) * f1 * f1
        s12 += (1.0 - w) * f1 * f2
        s22 += (1.0 - w) * f2 * f2
        c1 += (1.0 - w) * (h - x1 * f1 - x2 * f2) * f1
        c2 += (1.0 - w) * (h - x1 * f1 - x2 * f2) * f2
    k = Lc * ds
    return (2.0 * k * a, k * o1, k * o2, k * s11, k * s12, k * s22, k * c1, k * c2, 2.0 * k * dar, 2.0 * k * da1, 2.0 * k * da2)


@njit(cache=True)
def _ns_mats2(n, VR, VS, VQ, S3, S3c, S3b, NS3, Km, Cm, Mv):
    """둘째 길이 모드까지의 강성 · 감쇠 · 질량 (차수 2N+1: [첫 모드 줄 N, 몸체, 둘째 모드 줄 N], §52.531). 둘째 모드: 장력 4 배 (k² — 2π/L),
    줄 사이 전단 · 점성은 같은 꼴, 몸체(첫 모드)와는 직교라 층간 결합은 바닥으로, 질량 같음, 조직 감쇠는 같은 감쇠비 (β/2 · 4cT), 접촉은 두 모드를
    S12 로 잇는다."""
    Qn = VQ[n]
    cT = VR[n, 3]
    cG = VR[n, 4]
    cK = VR[n, 5]
    cB = VR[n, 6]
    ms = VR[n, 7]
    mbd = VR[n, 8]
    bT = VR[n, 11]
    bet = VR[n, 12]
    cC = VR[n, 13]
    ceta = VR[n, 14]
    cF = VR[n, 15]
    Kcol = VS[14]
    Rr = VS[15]
    Ccol = VS[16]
    dys = bT / NS3
    nb = NS3
    D = 2 * NS3 + 1
    for i in range(D):
        Mv[i] = 0.0
        for j in range(D):
            Km[i, j] = 0.0
            Cm[i, j] = 0.0
    for k in range(NS3):
        q = NS3 + 1 + k
        prof = Rr + (1.0 - Rr) * (k + 0.5) / NS3
        # 첫 모드 (예전과 같다)
        Mv[k] = ms / NS3 / Qn
        Km[k, k] += cT / NS3 * Qn + Kcol * dys * S3[k] * Qn
        Cm[k, k] += bet * cT / NS3 + Ccol * dys * S3[k]
        kk = cK * prof / NS3 * Qn
        cc = cC / NS3
        Km[k, k] += kk
        Km[nb, nb] += kk
        Km[k, nb] -= kk
        Km[nb, k] -= kk
        Cm[k, k] += cc
        Cm[nb, nb] += cc
        Cm[k, nb] -= cc
        Cm[nb, k] -= cc
        # 둘째 모드
        Mv[q] = ms / NS3 / Qn
        Km[q, q] += 4.0 * cT / NS3 * Qn + Kcol * dys * S3b[k] * Qn + kk
        Cm[q, q] += 0.5 * bet * 4.0 * cT / NS3 + Ccol * dys * S3b[k] + cc
        # 접촉이 두 모드를 잇는다
        Km[k, q] += Kcol * dys * S3c[k] * Qn
        Km[q, k] += Kcol * dys * S3c[k] * Qn
        Cm[k, q] += Ccol * dys * S3c[k]
        Cm[q, k] += Ccol * dys * S3c[k]
        if k + 1 < NS3:
            g = cG * Qn
            for (u_, v_) in ((k, k + 1), (q, q + 1)):
                Km[u_, u_] += g
                Km[v_, v_] += g
                Km[u_, v_] -= g
                Km[v_, u_] -= g
                Cm[u_, u_] += ceta
                Cm[v_, v_] += ceta
                Cm[u_, v_] -= ceta
                Cm[v_, u_] -= ceta
    Mv[nb] = mbd / Qn
    Km[nb, nb] += cB * Qn
    Cm[nb, nb] += cF


@njit(cache=True)
def _ns_geom_d(x, r, Lc, edge, ph):
    """`_ns_geom` 의 경사 몫 도함수 — (∂O/∂x, ∂O/∂r, ∂S/∂x, ∂S/∂r, ∂C/∂x, ∂C/∂r). /∂Lc 는 값/Lc."""
    if edge <= 0.0:
        edge = VF_NS_EDGE
    dox = 0.0
    dor = 0.0
    dsx = 0.0
    dsr = 0.0
    dcx = 0.0
    dcr = 0.0
    ds = 1.0 / VF_NS_PTS
    for i in range(VF_NS_PTS):
        sv = (i + 0.5) * ds
        sn = math.sin(math.pi * sv)
        h0 = r * sv + x * sn
        if VF_NS_FILM_DECAY > 0.0 and h0 > 0.0:
            gf = math.exp(-h0 / VF_NS_FILM_DECAY)
            h = h0 - ph[i] * gf
            dh = 1.0 + ph[i] * gf / VF_NS_FILM_DECAY
        else:
            h = h0 - ph[i]
            dh = 1.0
        ct = h - x * sn                                 # 접촉 쉼 몫의 항 (r·s − φ·g)
        u = h / edge + 0.5
        if VF_NS_SMOOTH:
            uc = min(max(u, 0.0), 1.0)
            w = uc * uc * (3.0 - 2.0 * uc)
            wp = 6.0 * uc * (1.0 - uc) / edge
        else:
            w = min(max(u, 0.0), 1.0)
            wp = 1.0 / edge if (u > 0.0 and u < 1.0) else 0.0
        dox += wp * dh * sn * sn
        dor += wp * dh * sv * sn
        dsx += -wp * dh * sn * sn * sn
        dsr += -wp * dh * sv * sn * sn
        dcx += -wp * dh * sn * ct * sn + (1.0 - w) * (dh - 1.0) * sn * sn
        dcr += (1.0 - w) * dh * sv * sn - wp * dh * sv * ct * sn
    return (Lc * dox * ds, Lc * dor * ds, Lc * dsx * ds, Lc * dsr * ds, Lc * dcx * ds, Lc * dcr * ds)


@njit(cache=True)
def _spd_solve(Am, b, out):
    """대칭 양정치 Am x = b (촐레스키, 작다). Am·b 는 건드리지 않는다."""
    n = b.shape[0]
    Lm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = Am[i, j]
            for k in range(j):
                s -= Lm[i, k] * Lm[j, k]
            if i == j:
                Lm[i, i] = math.sqrt(max(s, 1e-300))
            else:
                Lm[i, j] = s / Lm[j, j]
    y = np.zeros(n)
    for i in range(n):
        s = b[i]
        for k in range(i):
            s -= Lm[i, k] * y[k]
        y[i] = s / Lm[i, i]
    for i in range(n - 1, -1, -1):
        s = y[i]
        for k in range(i + 1, n):
            s -= Lm[k, i] * out[k]
        out[i] = s / Lm[i, i]


@njit(cache=True)
def _ns_mats(n, VR, VS, VQ, S3, NS3, Km, Cm, Mv):
    """모드 3 의 강성·감쇠·질량 (접촉 몫 포함) — 표본 n."""
    Qn = VQ[n]
    cT = VR[n, 3]
    cG = VR[n, 4]
    cK = VR[n, 5]
    cB = VR[n, 6]
    ms = VR[n, 7]
    mbd = VR[n, 8]
    bT = VR[n, 11]
    bet = VR[n, 12]
    cC = VR[n, 13]
    ceta = VR[n, 14]
    cF = VR[n, 15]
    Kcol = VS[14]
    Rr = VS[15]
    Ccol = VS[16]
    dys = bT / NS3
    nb = NS3
    for i in range(NS3 + 1):
        Mv[i] = 0.0
        for j in range(NS3 + 1):
            Km[i, j] = 0.0
            Cm[i, j] = 0.0
    for k in range(NS3):
        Mv[k] = ms / NS3 / Qn
        prof = Rr + (1.0 - Rr) * (k + 0.5) / NS3
        kt = cT / NS3 * Qn + Kcol * dys * S3[k] * Qn
        Km[k, k] += kt
        Cm[k, k] += bet * cT / NS3 + Ccol * dys * S3[k]
        kk = cK * prof / NS3 * Qn
        cc = cC / NS3
        Km[k, k] += kk
        Km[nb, nb] += kk
        Km[k, nb] -= kk
        Km[nb, k] -= kk
        Cm[k, k] += cc
        Cm[nb, nb] += cc
        Cm[k, nb] -= cc
        Cm[nb, k] -= cc
        if k + 1 < NS3:
            g = cG * Qn
            Km[k, k] += g
            Km[k + 1, k + 1] += g
            Km[k, k + 1] -= g
            Km[k + 1, k] -= g
            Cm[k, k] += ceta
            Cm[k + 1, k + 1] += ceta
            Cm[k, k + 1] -= ceta
            Cm[k + 1, k] -= ceta
    Mv[nb] = mbd / Qn
    Km[nb, nb] += cB * Qn
    Cm[nb, nb] += cF


@njit(cache=True)
def _ns_blk(n, VR, VS, Qf, mf, NS3, Km, Cm, Mv, off):
    """좌우 판 (§52.533) 의 한 성대 블록 — 덮개 NS3 줄 + 몸체를 off 자리에 더한다 (접촉은 빼고, `_ns_mats_lr` 가 두 성대를 잇는다).
    긴장 Qf, 질량 배율 mf. `_ns_mats` 와 같은 계수 · 같은 식."""
    cT = VR[n, 3]
    cG = VR[n, 4]
    cK = VR[n, 5]
    cB = VR[n, 6]
    ms = VR[n, 7]
    mbd = VR[n, 8]
    bet = VR[n, 12]
    cC = VR[n, 13]
    ceta = VR[n, 14]
    cF = VR[n, 15]
    Rr = VS[15]
    nb = off + NS3
    for k in range(NS3):
        i = off + k
        Mv[i] = ms * mf / NS3 / Qf
        prof = Rr + (1.0 - Rr) * (k + 0.5) / NS3
        Km[i, i] += cT / NS3 * Qf
        Cm[i, i] += bet * cT / NS3
        kk = cK * prof / NS3 * Qf
        cc = cC / NS3
        Km[i, i] += kk
        Km[nb, nb] += kk
        Km[i, nb] -= kk
        Km[nb, i] -= kk
        Cm[i, i] += cc
        Cm[nb, nb] += cc
        Cm[i, nb] -= cc
        Cm[nb, i] -= cc
        if k + 1 < NS3:
            g = cG * Qf
            Km[i, i] += g
            Km[i + 1, i + 1] += g
            Km[i, i + 1] -= g
            Km[i + 1, i] -= g
            Cm[i, i] += ceta
            Cm[i + 1, i + 1] += ceta
            Cm[i, i + 1] -= ceta
            Cm[i + 1, i] -= ceta
    Mv[nb] = mbd * mf / Qf
    Km[nb, nb] += cB * Qf
    Cm[nb, nb] += cF


@njit(cache=True)
def _ns_mats_lr(n, VR, VS, VQ, S3, NS3, Km, Cm, Mv):
    """좌우 성대 따로 (§52.533, VS[20] = 1): 상태 [왼 덮개 NS3 · 왼 몸체 · 오른 덮개 NS3 · 오른 몸체]. 긴장 Q(1 ± VS[21]), 질량 (1 ± VS[22]).
    성문 기하는 두 반틈새의 평균 (x_L + x_R)/2 로 재고, 접촉압은 그 겹침이 정해 두 성대에 같게 (작용 · 반작용) — 접촉 강성 Kcol·dys·S·Q 를
    ½ (e_L + e_R)(e_L + e_R)ᵀ 꼴로 건다. 대칭 (x_L = x_R) 이면 단일 성대 `_ns_mats` 와 같은 운동."""
    Qn = VQ[n]
    dq = VS[21]
    dm = VS[22]
    D = 2 * (NS3 + 1)
    for i in range(D):
        Mv[i] = 0.0
        for j in range(D):
            Km[i, j] = 0.0
            Cm[i, j] = 0.0
    _ns_blk(n, VR, VS, Qn * (1.0 + dq), 1.0 + dm, NS3, Km, Cm, Mv, 0)
    _ns_blk(n, VR, VS, Qn * (1.0 - dq), 1.0 - dm, NS3, Km, Cm, Mv, NS3 + 1)
    dys = VR[n, 11] / NS3
    Kcol = VS[14]
    Ccol = VS[16]
    for k in range(NS3):
        a = k
        b = NS3 + 1 + k
        kc = 0.5 * Kcol * dys * S3[k] * Qn
        cc = 0.5 * Ccol * dys * S3[k]
        Km[a, a] += kc
        Km[a, b] += kc
        Km[b, a] += kc
        Km[b, b] += kc
        Cm[a, a] += cc
        Cm[a, b] += cc
        Cm[b, a] += cc
        Cm[b, b] += cc


@njit(cache=True)
def _ns_adj_lr(n, VR, VS, VQ, S3, NS3, MU, Y, X, XP, dt, gVR, gVS, lS3, lx, lxp):
    """`_ns_mats_lr` 운동의 수반 — `_ns_adj` 와 같은 규약 (μ = A⁻¹λ). 긴장 Q 의 기울기를 돌려주고 VS[21] · VS[22] 의 기울기를 더한다."""
    Qn = VQ[n]
    dq = VS[21]
    dm = VS[22]
    cT = VR[n, 3]
    cG = VR[n, 4]
    cK = VR[n, 5]
    cB = VR[n, 6]
    ms = VR[n, 7]
    mbd = VR[n, 8]
    bT = VR[n, 11]
    bet = VR[n, 12]
    Kcol = VS[14]
    Rr = VS[15]
    Ccol = VS[16]
    dys = bT / NS3
    T2 = dt * dt
    lQ = 0.0
    ldys = 0.0
    for f in range(2):
        sg = 1.0 if f == 0 else -1.0
        Qf = Qn * (1.0 + sg * dq)
        mf = 1.0 + sg * dm
        off = f * (NS3 + 1)
        nb = off + NS3
        lQf = 0.0
        for k in range(NS3):
            i = off + k
            kt = -T2 * MU[i] * Y[i]
            ct = dt * MU[i] * (X[i] - Y[i])
            gVR[n, 3] += kt * Qf / NS3 + ct * bet / NS3
            gVR[n, 12] += ct * cT / NS3
            lQf += kt * cT / NS3
            um = MU[i] - MU[nb]
            fr = (k + 0.5) / NS3
            prof = Rr + (1.0 - Rr) * fr
            kc_ = -T2 * um * (Y[i] - Y[nb])
            cc_ = dt * um * ((X[i] - X[nb]) - (Y[i] - Y[nb]))
            gVR[n, 5] += kc_ * prof * Qf / NS3
            gVS[15] += kc_ * cK * Qf / NS3 * (1.0 - fr)
            lQf += kc_ * cK * prof / NS3
            gVR[n, 13] += cc_ / NS3
            if k + 1 < NS3:
                us = MU[i] - MU[i + 1]
                ks_ = -T2 * us * (Y[i] - Y[i + 1])
                cs_ = dt * us * ((X[i] - X[i + 1]) - (Y[i] - Y[i + 1]))
                gVR[n, 4] += ks_ * Qf
                lQf += ks_ * cG
                gVR[n, 14] += cs_
            lM = MU[i] * (2.0 * X[i] - XP[i] - Y[i])
            gVR[n, 7] += lM * mf / (NS3 * Qf)
            lQf -= lM * ms * mf / (NS3 * Qf * Qf)
            gVS[22] += lM * sg * ms / (NS3 * Qf)
        kb_ = -T2 * MU[nb] * Y[nb]
        cb_ = dt * MU[nb] * (X[nb] - Y[nb])
        gVR[n, 6] += kb_ * Qf
        lQf += kb_ * cB
        gVR[n, 15] += cb_
        lMb = MU[nb] * (2.0 * X[nb] - XP[nb] - Y[nb])
        gVR[n, 8] += lMb * mf / Qf
        lQf -= lMb * mbd * mf / (Qf * Qf)
        gVS[22] += lMb * sg * mbd / Qf
        lQ += lQf * (1.0 + sg * dq)
        gVS[21] += lQf * sg * Qn
    for k in range(NS3):
        a = k
        b = NS3 + 1 + k
        um = MU[a] + MU[b]
        kt = -T2 * 0.5 * um * (Y[a] + Y[b])
        ct = dt * 0.5 * um * ((X[a] + X[b]) - (Y[a] + Y[b]))
        gVS[14] += kt * dys * S3[k] * Qn
        gVS[16] += ct * dys * S3[k]
        ldys += kt * Kcol * S3[k] * Qn + ct * Ccol * S3[k]
        lS3[k] += kt * Kcol * dys * Qn + ct * Ccol * dys
        lQ += kt * Kcol * dys * S3[k]
    gVR[n, 11] += ldys / NS3
    D = 2 * (NS3 + 1)
    Mv = np.zeros(D)
    Cm = np.zeros((D, D))
    Km = np.zeros((D, D))
    _ns_mats_lr(n, VR, VS, VQ, S3, NS3, Km, Cm, Mv)
    for k in range(D):
        c = 0.0
        for j in range(D):
            c += Cm[j, k] * MU[j]
        lx[k] += 2.0 * Mv[k] * MU[k] + dt * c
        lxp[k] += -Mv[k] * MU[k]
    return lQ


@njit(cache=True)
def _ns_adj(n, VR, VS, VQ, S3, NS3, MU, Y, X, XP, dt, gVR, gVS, lS3, lx, lxp):
    """모드 3 운동의 수반 — μ = A⁻¹λ 가 주어졌을 때 (A = M + C dt + K dt², 우변 F dt² + M(2x − x_) + C dt x):
    ∂A = −μ yᵀ, ∂우변 = μ. 패턴 u uᵀ (u = e_a − e_b) 를 가진 계수의 기울기는 강성 −dt²(uᵀμ)(uᵀy), 감쇠 dt(uᵀμ)(uᵀ(x − y)).
    계수(VR 열 3–15, VS 14–16) 기울기를 더하고, 접촉 적분 S3 의 기울기와 x·x_ 의 수반을 채우고, 긴장 Q 의 기울기를 돌려준다."""
    Qn = VQ[n]
    cT = VR[n, 3]
    cG = VR[n, 4]
    cK = VR[n, 5]
    cB = VR[n, 6]
    ms = VR[n, 7]
    mbd = VR[n, 8]
    bT = VR[n, 11]
    bet = VR[n, 12]
    Kcol = VS[14]
    Rr = VS[15]
    Ccol = VS[16]
    dys = bT / NS3
    nb = NS3
    T2 = dt * dt
    lQ = 0.0
    ldys = 0.0
    for k in range(NS3):
        kt = -T2 * MU[k] * Y[k]
        ct = dt * MU[k] * (X[k] - Y[k])
        # 장력 (K: cT Q / N, C: β cT / N)
        gVR[n, 3] += kt * Qn / NS3 + ct * bet / NS3
        gVR[n, 12] += ct * cT / NS3
        lQ += kt * cT / NS3
        # 접촉 (K: Kcol dys S3 Q, C: Ccol dys S3)
        gVS[14] += kt * dys * S3[k] * Qn
        gVS[16] += ct * dys * S3[k]
        ldys += kt * Kcol * S3[k] * Qn + ct * Ccol * S3[k]
        lS3[k] += kt * Kcol * dys * Qn + ct * Ccol * dys
        lQ += kt * Kcol * dys * S3[k]
        # 층간 결합 (u = e_k − e_b)
        um = MU[k] - MU[nb]
        f = (k + 0.5) / NS3
        prof = Rr + (1.0 - Rr) * f
        kc_ = -T2 * um * (Y[k] - Y[nb])
        cc_ = dt * um * ((X[k] - X[nb]) - (Y[k] - Y[nb]))
        gVR[n, 5] += kc_ * prof * Qn / NS3
        gVS[15] += kc_ * cK * Qn / NS3 * (1.0 - f)
        lQ += kc_ * cK * prof / NS3
        gVR[n, 13] += cc_ / NS3
        # 줄 사이 전단 (u = e_k − e_{k+1})
        if k + 1 < NS3:
            us = MU[k] - MU[k + 1]
            ks_ = -T2 * us * (Y[k] - Y[k + 1])
            cs_ = dt * us * ((X[k] - X[k + 1]) - (Y[k] - Y[k + 1]))
            gVR[n, 4] += ks_ * Qn
            lQ += ks_ * cG
            gVR[n, 14] += cs_
        # 질량 m_k = ms / N / Q
        lM = MU[k] * (2.0 * X[k] - XP[k] - Y[k])
        gVR[n, 7] += lM / (NS3 * Qn)
        lQ -= lM * ms / (NS3 * Qn * Qn)
    # 몸체
    kb_ = -T2 * MU[nb] * Y[nb]
    cb_ = dt * MU[nb] * (X[nb] - Y[nb])
    gVR[n, 6] += kb_ * Qn
    lQ += kb_ * cB
    gVR[n, 15] += cb_
    lMb = MU[nb] * (2.0 * X[nb] - XP[nb] - Y[nb])
    gVR[n, 8] += lMb / Qn
    lQ -= lMb * mbd / (Qn * Qn)
    gVR[n, 11] += ldys / NS3
    # x, x_ 의 수반: 우변 M(2x − x_) + C dt x
    Mv = np.zeros(NS3 + 1)
    Cm = np.zeros((NS3 + 1, NS3 + 1))
    Km = np.zeros((NS3 + 1, NS3 + 1))
    _ns_mats(n, VR, VS, VQ, S3, NS3, Km, Cm, Mv)
    for k in range(NS3 + 1):
        c = 0.0
        for j in range(NS3 + 1):
            c += Cm[j, k] * MU[j]
        lx[k] += 2.0 * Mv[k] * MU[k] + dt * c
        lxp[k] += -Mv[k] * MU[k]
    return lQ


@njit(cache=True)
def _sym3_solve(a, b, c, d, e, f, r1, r2, r3):
    """[[a d e] [d b f] [e f c]] y = r 의 해 (여인수) — 몸체–덮개 성대 한 표본 (§52.500). 대칭이라 수반(M⁻ᵀ)도 같은 함수."""
    C11 = b * c - f * f
    C12 = e * f - d * c
    C13 = d * f - b * e
    C22 = a * c - e * e
    C23 = d * e - a * f
    C33 = a * b - d * d
    det = a * C11 + d * C12 + e * C13
    return ((C11 * r1 + C12 * r2 + C13 * r3) / det, (C12 * r1 + C22 * r2 + C23 * r3) / det,
            (C13 * r1 + C23 * r2 + C33 * r3) / det)


@njit(cache=True)
def _jet_far(A, n, i, us, a1, a2, N, dx, G):
    """제트가 퍼져 나가는 넓어진 면적 (매끈, §52.523) — 흐름 방향 `JET_LEN_CM` 안의 코사인 무게 로그–합–지수.
    반환 (amx, ∂amx/∂dx, 첫 칸, 방향, 칸 수); G[s] = ∂amx/∂a_(첫 칸 + s·방향)."""
    if us > 0.0:
        j0 = i + 1
        dr = 1
        a0 = a2
    else:
        j0 = i
        dr = -1
        a0 = a1
    num = 0.0
    den = 0.0
    dnum = 0.0
    dden = 0.0
    ns = 0
    for s_ in range(N):
        j = j0 + s_ * dr
        if j < 0 or j >= N:
            break
        d = s_ * dx
        if s_ > 0 and d >= JET_LEN_CM:
            break
        if s_ == 0:
            w = 1.0
            dw = 0.0
            aj = a0
        else:
            w = 0.5 * (1.0 + math.cos(math.pi * d / JET_LEN_CM))
            dw = -0.5 * math.sin(math.pi * d / JET_LEN_CM) * math.pi * s_ / JET_LEN_CM      # ∂w/∂dx
            aj = max(A[n, j], A_FLOOR)
        e = w * math.exp(JET_SOFT_BETA * (aj - a0))
        G[s_] = e
        num += e
        den += w
        dnum += dw * (e / w if w > 0.0 else 0.0)
        dden += dw
        ns = s_ + 1
    amx = a0 + math.log(num / den) / JET_SOFT_BETA
    for s_ in range(ns):
        G[s_] = G[s_] / num
    damx_dx = (dnum / num - dden / den) / JET_SOFT_BETA
    return amx, damx_dx, j0, dr, ns


@njit(cache=True)
def _gq(u, ag, lre, ach=0.0):
    """성문 제트 난류 세기 (§52.524, `GLOT_NOISE_3D`) — 성대 길이 방향 점들의 Stevens 세기 제곱합의 제곱근 [동압 단위].
    ach > 0 이고 `GLOT_NOISE_CHINK` 면 뒤틈을 제 점으로 (§52.530)."""
    au = abs(u)
    if au <= 1e-12 or ag <= 1e-9:
        return 0.0
    k2 = 0.5 * GLOTTIS_KE * RHO
    dp = k2 * au * au / (ag * ag) + 12.0 * MU * GLOTTIS_D * lre * lre * au / (ag * ag * ag)
    tot = 0.0
    am = ag
    if GLOT_NOISE_CHINK and ach > 1e-9:
        am = max(ag - ach, 1e-9)
        h = ach / CHINK_LEN
        c1 = 12.0 * MU * GLOTTIS_D / (h * h)
        v = 2.0 * dp / (c1 + math.sqrt(c1 * c1 + 4.0 * k2 * dp))
        Re = 2.0 * v * h / NU
        if Re > 0.0:
            g, _dg = _spos(1.0 - (RE_CRIT / Re) ** 2, TURB_GATE_W)
            if g > 0.0:
                fj = (1.0 - math.exp(-GLOT_JET_CORE * h / GLOT_JET_D)) if GLOT_NOISE_JET else 1.0
                q = RHO * v * v * (v / V_REF) * math.sqrt(ach / A_REF) * fj * g
                tot += q * q
    hpk = 0.5 * math.pi * am / lre
    for j in range(GN_PTS):
        h = hpk * math.sin(math.pi * (j + 0.5) / GN_PTS)
        c1 = 12.0 * MU * GLOTTIS_D / (h * h)
        v = 2.0 * dp / (c1 + math.sqrt(c1 * c1 + 4.0 * k2 * dp))          # ½Kρv² + c1 v = Δp 의 양근
        Re = 2.0 * v * h / NU
        if Re <= 0.0:
            continue
        g, _dg = _spos(1.0 - (RE_CRIT / Re) ** 2, TURB_GATE_W)
        if g <= 0.0:
            continue
        fj = (1.0 - math.exp(-GLOT_JET_CORE * h / GLOT_JET_D)) if GLOT_NOISE_JET else 1.0
        q = RHO * v * v * (v / V_REF) * math.sqrt(h * lre / GN_PTS / A_REF) * fj * g
        tot += q * q
    return math.sqrt(tot)


@njit(cache=True)
def _gq_d(u, ag, lre, ach=0.0):
    """`_gq` 와 그 편미분 (∂/∂u, ∂/∂A, ∂/∂ℓ) — 매끈한 함수라 상대 보폭 1e-6 의 중앙 차분 (뒤틈 면적은 고정)."""
    q0 = _gq(u, ag, lre, ach)
    hu = 1e-6 * max(abs(u), 1e-3)
    ha = 1e-6 * ag
    hl = 1e-6 * lre
    du = (_gq(u + hu, ag, lre, ach) - _gq(u - hu, ag, lre, ach)) / (2.0 * hu)
    da = (_gq(u, ag + ha, lre, ach) - _gq(u, ag - ha, lre, ach)) / (2.0 * ha)
    dl = (_gq(u, ag, lre + hl, ach) - _gq(u, ag, lre - hl, ach)) / (2.0 * hl)
    return q0, du, da, dl


@njit(cache=True)
def _spos(x, e):
    """C¹ 양수부 — (값, 도함수). |x| < e 에서 (x + e)²/(4e) (§52.524)."""
    if x >= e:
        return x, 1.0
    if x <= -e:
        return 0.0, 0.0
    return (x + e) * (x + e) / (4.0 * e), (x + e) / (2.0 * e)


@njit(cache=True)
def _smin(a, b, w):
    """매끈한 최소 (a, b) — (값, ∂/∂a, ∂/∂b). 폭 e = w·|b| 안에서 |a − b| 를 이차로 (§52.524)."""
    e = w * abs(b) + 1e-30
    d = a - b
    if d >= e:
        return b, 0.0, 1.0
    if d <= -e:
        return a, 1.0, 0.0
    sa = d * d / (2.0 * e) + 0.5 * e
    dsd = d / e
    dse = -d * d / (2.0 * e * e) + 0.5
    return 0.5 * (a + b - sa), 0.5 * (1.0 - dsd), 0.5 * (1.0 + dsd - dse * w * (1.0 if b >= 0.0 else -1.0))


@njit(cache=True)
def _kvkt(rho, mu, gamma, lam, cp):
    """경계층 손실 계수 (점성 Kv, 열 Kt) — 영역 물성으로 (§52.524). `_consts` 와 같은 꼴."""
    return (LOSS_WET * math.sqrt(rho * mu / 2.0) * math.sqrt(REF_W),
            LOSS_WET * (gamma - 1.0) * math.sqrt(lam / (2.0 * rho * cp)) * math.sqrt(REF_W))


@njit(cache=True)
def _consts(fs):
    kc = 1.0 - math.exp(-1.0 / (0.001 * fs))     # 협착 난류 세기의 평활 1 ms
    kg = 1.0 - math.exp(-1.0 / ((GLOT_NOISE_TAU_MS if GLOT_NOISE_SUPRA else 0.2) * 1.0e-3 * fs))    # 성문 난류 세기의 평활
    Kv = LOSS_WET * math.sqrt(RHO * MU / 2.0) * math.sqrt(REF_W)
    Kt = LOSS_WET * (GAMMA - 1.0) * math.sqrt(LAMBDA_TH / (2.0 * RHO * CP)) * math.sqrt(REF_W)
    return kc, kg, Kv, Kt


_DIFF_CACHE: dict = {}


def diffusive(fs: float = FS_SIM):
    """확산 표현의 (a_m, w_m) — φ_m ← φ_m + a_m (x − φ_m), 손실 = K·Σ w_m (x − φ_m^이전) 이 K·√2·√(jω/REF_W) (= ω_ref 에서 실수부 K) 가
    되도록 이산 가지 응답 (z−1)/(z−1+a_m) 에 비음수 최소제곱으로 맞춘다."""
    key = float(fs)
    if key not in _DIFF_CACHE:
        from scipy.optimize import nnls
        dt = 1.0 / key
        f = np.geomspace(60.0, min(16000.0, 0.4 * key), 400)
        z = np.exp(2j * np.pi * f * dt)
        tgt = np.sqrt(2.0) * np.sqrt(2j * np.pi * f / REF_W)
        a = 1.0 - np.exp(-2.0 * np.pi * np.asarray(DIFF_HZ) * dt)
        H = np.stack([(z - 1.0) / (z - 1.0 + am) for am in a], 1) / np.abs(tgt)[:, None]
        t = tgt / np.abs(tgt)
        w, _ = nnls(np.vstack([H.real, H.imag]), np.concatenate([t.real, t.imag]))
        _DIFF_CACHE[key] = (np.ascontiguousarray(a), np.ascontiguousarray(w))
    return _DIFF_CACHE[key]


def _fwd(A, L, Ag, Ps, Av, Q, AN, NL, TRA, TRL, xi_g, xi_c, noise_g, noise_c, wall_on, loss_on, fs, SBA, SBL, SBI, SBNA, SBNL, XA, XW, VF, VQ, VR, VS, PT, PP, EV, out, rec,
         Tp, TU, TUg, TUm, TUL, Tpm, Tvw, Tyw, TUs, TUgs, TpN, TUN, TUv, TUmN, TULN, TpmN, TpP, TUP,
         TFv, TFt, TFvN, TFtN, TFvP, TFtP, TX, TpT, TUT, TFvT, TFtT, TPH):
    """한 발화를 푼다 (순방향) — 역방향이 쓸 상태를 기록한다. 모두 표본(FS_SIM)마다:

    A (T, N) 구강 구간 면적 [cm²], L (T,) 구강 길이 [cm], Ag (T,) 성문 면적 [cm²], Ps (T,) 폐압 [dyn/cm²], Av (T,) 연구개 포트 면적 [cm²],
    AN (NN,) 비강 면적(고정), xi_g (T,) · xi_c (T, N-1) 난류 여기, noise_g · noise_c 난류 세기 배율, wall_on·loss_on 0/1.
    out (T,) 방사 음압 ∝ d(U_입술 + U_콧구멍)/dt, rec (T, 7) 진단 [U_g, p_0, U_입술, Re_g, max Re_c, p_입술, U_콧구멍].
    기록 T* 는 (T+1, …) — 줄 n 이 단계 n **전**의 상태, 줄 n+1 이 뒤의 상태.
    순서 (물리 사슬): 성문 → 구강 이음매 → 연구개 포트 → 비강 이음매 → 이상와 이음매 → 입술 → 콧구멍 → 구강 칸 → 비강 칸 → 이상와 칸.
    SBA·SBL·SBI (K,) **닫힌 곁관** K 개의 면적·길이·붙는 구강 칸 (`SIDE_BRANCHES` — 이상와, 후두실; 면적 0 이면 없다), TpP (T+1, K·NPB)·TUP
    (T+1, K·NPB) 곁관 칸 압력·이음매 유량 (곁관 b 의 칸은 b·NPB … b·NPB+NPB−1, 첫 이음매가 구강 → 곁관 입구).
    XA·XW (M,) 경계층 손실의 확산 표현 (`diffusive`). 이음매 손실 = r0·U + rf·Σ w_m (U − φ_m) (r0 푸아죄유 정상류 저항), 칸 손실 = g·Σ w_m (p − ψ_m).
    TF* 는 단계 전 상태의 Σ w_m φ_m (역방향이 rf·g 의 기울기에 쓴다) — 구강 이음매·칸, 비강 이음매·칸, 이상와 이음매·칸.
    VF 1 이면 **자기 진동 성대** (`VF_STATIC`): Ag 는 떨지 않는 뒤쪽 틈(연골부) 면적, VQ (T,) 긴장 Q, VR (T, 2) 아래·위 질량의 뒤쪽 쉼 변위 [cm],
    VS 정적 변수. TX (T+2, 2) 두 질량의 상대 변위 — 줄 n+1 이 단계 n 의 변위(면적을 정한다), 줄 n+2 가 단계 n 이 낸 다음 변위.
    rec[:, 7:9] 은 아래·위 성문 면적 (틈 포함).
    PT (K,) 원본의 성문 닫힘 시각표 [표본] — 2 개 이상이면 **위상 고정 고리** (`PLL_GAINS`, §52.493): 막부 면적이 닫히는(또는 골에 닿는)
    때를 잡아 가장 가까운 PT 와의 어긋남으로 다음 주기의 긴장을 고친다 (비례 + 적분). **VQ 를 고친 값으로 덮어쓴다** — 역방향은 그것을 쓴다.
    PP 고리 상수 [kp, ki, umax, vmax, sens, gap, arm, pmin, pmax, ff, tau(표본), det, darm, kf], EV 잡은 닫힘 시각 (−1 이 남은 칸).
    """
    T, N = A.shape
    NN = AN.shape[0]
    NP = TpP.shape[1]
    M = XA.shape[0]
    W = 0.0
    for m in range(M):
        W += XW[m]
    dt = 1.0 / fs
    rc2 = RHO * C_SOUND * C_SOUND
    kc, kg, Kv, Kt = _consts(fs)
    KvN, KtN = _kvkt(RHO_N, MU_N, GAMMA_N, LAMBDA_N, CP_N)
    KvT, KtT = _kvkt(RHO_T, MU_T, GAMMA_T, LAMBDA_T, CP_T)
    rc2N = RHO_N * C_N * C_N
    rc2T = RHO_T * C_T * C_T
    iv = int(VEL_FRAC * N)
    NPB = NP // max(SBA.shape[0], 1)          # 곁관 하나의 칸 수
    dxn = NL / NN
    NT = TRA.shape[0]                          # 기관 칸 수 (0 이면 기관 없음 — 폐압이 성문에 바로)
    dxt = TRL / max(NT, 1)
    LgT = RHO_T * 0.5 * dxt / TRA[NT - 1] if NT > 0 else 0.0   # 성문 관성에 기관 윗칸의 반 (안정)
    phvT = np.zeros((NT, M))
    phtT = np.zeros((NT, M))
    phv = np.zeros((N - 1, M))
    pht = np.zeros((N, M))
    phvN = np.zeros((NN - 1, M))
    phtN = np.zeros((NN, M))
    phvP = np.zeros((NP, M))
    phtP = np.zeros((NP, M))
    # 위상 고정 고리 상태
    npt = PT.shape[0]
    pll = VF >= 1 and npt >= 2
    uc = 0.0
    ua = 0.0                                   # 실제로 건 배율 — uc 를 시정수 PP[10] 로 따라간다 (계단이면 질량·강성이 한 표본에 뛴다)
    alpha = 1.0 if PP[10] <= 0.0 else 1.0 - math.exp(-1.0 / PP[10])
    vint = 0.0
    jp = 0
    pk = 0.0
    plast = 0.0
    armed = False
    gprev = 0.0
    last_ev = -1.0e9
    nev = 0
    gmn = 0.0                                  # 이 표본의 막부 면적 (면적 방식)
    LR3 = VF == 3 and VS.shape[0] > 22 and VS[20] > 0.5                 # 좌우 성대 따로 (§52.533)
    LM3 = 2 if (VF == 3 and VS.shape[0] > 19 and VS[19] > 1.5 and not LR3) else 1   # 길이 모드 수 (§52.531)
    NS3 = (TX.shape[1] // 2 - 1) if LR3 else (TX.shape[1] - 1) // LM3   # 모드 3 의 줄 수 (몸체 하나를 뺀다)
    D3 = 2 * (NS3 + 1) if LR3 else LM3 * NS3 + 1
    G3 = np.zeros(NS3 + 1)
    O3 = np.zeros(NS3 + 1)
    S3 = np.zeros(NS3 + 1)
    C3 = np.zeros(NS3 + 1)
    O3b = np.zeros(NS3 + 1)
    S3b = np.zeros(NS3 + 1)
    S3c = np.zeros(NS3 + 1)
    C3b = np.zeros(NS3 + 1)
    PK3 = np.zeros(NS3 + 1)
    Km3 = np.zeros((D3, D3))
    Cm3 = np.zeros((D3, D3))
    Am3 = np.zeros((D3, D3))
    Mv3 = np.zeros(D3)
    R3 = np.zeros(D3)
    Y3 = np.zeros(D3)
    kmin3 = 0
    agm = 1.0
    Lc = GLOTTIS_LEN
    ksm = 1.0 - math.exp(-1.0 / (1.0e-4 * fs))  # dU/dt 평활 0.1 ms (MFDR 방식)
    dsm = 0.0
    dmin = 0.0
    tmin = 0.0
    dlast = 0.0
    pev = -1.0e9                               # 앞 닫힘 (주파수 고리)
    JG = np.zeros(N + 1)                       # 제트 넓어진 면적의 칸별 몫 (§52.523)
    for n in range(T):
        dx = L[n] / N
        # ---- 성문: L_g dU/dt + R_k U|U| + R_v U = P_s − p_0 (+ 성문 난류) — 이차식을 닫힌 꼴로
        ga1 = 0.0
        ga2 = 0.0
        if VF:
            # 두 질량이 정한 아래·위 면적 (막부 + 뒤쪽 틈). 흐름은 좁은 쪽이 정하고, 관성·점성은 두 칸 직렬 (두께 d_i, 성대 길이 Lc)
            if n - last_ev > PP[5]:                      # 떨림이 끊기면 고리를 푼다
                uc = 0.0
                vint = 0.0
                plast = 0.0
            if pll:
                ua += alpha * (uc - ua)
                VQ[n] = VQ[n] * math.exp(ua)
            Qn = VQ[n]
            if VF == 3:
                # N 줄: 흐름은 가장 좁은 줄이 정하고, 관성·점성은 줄들의 직렬 (두께 bT/N)
                Lc = VR[n, 10]
                dys = VR[n, 11] / NS3
                ch = max(Ag[n], 0.0)
                a0 = max(A[n, 0], A_FLOOR)
                ag = 1.0e30
                gmn = 1.0e30
                sLg = 0.0
                sRv = 0.0
                for k in range(NS3):
                    rk = VR[n, 0] + (VR[n, 1] - VR[n, 0]) * (k + 0.5) / NS3
                    if LM3 == 2:
                        ak, ok, o2k, s2k, s12k, s22k, cfk, c2k, dak, csk, cs2k = _ns_geom2(TX[n + 1, k], TX[n + 1, NS3 + 1 + k], rk, Lc, VS[18], TPH[n + 1])
                        O3b[k] = o2k
                        S3c[k] = s12k
                        S3b[k] = s22k
                        C3b[k] = c2k
                    else:
                        xk3 = 0.5 * (TX[n + 1, k] + TX[n + 1, NS3 + 1 + k]) if LR3 else TX[n + 1, k]   # 좌우 판: 평균 반틈새 (§52.533)
                        ak, ok, s2k, cfk, dak, csk = _ns_geom(xk3, rk, Lc, VS[18], TPH[n + 1])
                    gk = max(ak + ch, A_FLOOR)
                    G3[k] = gk
                    O3[k] = ok
                    S3[k] = s2k
                    C3[k] = cfk
                    if gk < ag:
                        ag = gk
                        kmin3 = k
                    if ak < gmn:
                        gmn = ak
                    sLg += dys / gk
                    sRv += dys / (gk * gk * gk)
                Lg = RHO * (sLg + 0.5 * dx / a0)
                Rv = 12.0 * MU * Lc * Lc * sRv
                agm = ag                                   # 가장 좁은 줄 — 흐름 분리(줄 압력 분포)의 기준
                if VF_SEP_P > 0.0:
                    # 흐름 면적은 줄들의 매끈한 최소 (§52.515) — 가장 좁은 줄이 바뀌는 순간 min 의 모서리가 유량 미분을 꺾어 고역 빗살을 냈다
                    # (BF C 5–16 kHz 줄 0.22–0.27). 지수는 줄 수에 맞춰 모두 같을 때의 치우침이 두 질량과 같게 (p·log N / log 2).
                    p3 = VF_SEP_P * math.log(NS3) / math.log(2.0)
                    s3 = 0.0
                    for k in range(NS3):
                        s3 += (agm / G3[k]) ** p3
                    ag = agm * s3 ** (-1.0 / p3)
                ga1 = ag
                ga2 = ag
            else:
                sq = math.sqrt(Qn)
                Lc = VS[0] * sq
                d1 = VS[1] / sq
                d2 = VS[2] / sq
                ch = max(Ag[n], 0.0)
                x1 = TX[n + 1, 0]
                x2 = TX[n + 1, 1]
                m1a, o1, al1, rs1 = _vf_geom(x1, VR[n, 0], Lc)
                m2a, o2, al2, rs2 = _vf_geom(x2, VR[n, 1], Lc)
                ga1 = max(m1a + ch, A_FLOOR)
                ga2 = max(m2a + ch, A_FLOOR)
                if VF_SEP_P > 0.0:
                    # 흐름 분리는 계단이 아니라 연속으로 옮겨 간다 (§52.515, Pelorson et al. 1994) — min(a1, a2) 의 모서리가 닫히기 시작하는
                    # 순간(수렴 → 발산) 유량 미분을 꺾어, 주기마다 같은 모양의 고역 울림을 냈다: 5–16 kHz 의 고른 배음 빗살(가로줄)
                    ag = (ga1 ** -VF_SEP_P + ga2 ** -VF_SEP_P) ** (-1.0 / VF_SEP_P)
                else:
                    ag = min(ga1, ga2)
                gmn = min(m1a, m2a)
                # 관성 = 두 질량 칸 + 성도 첫 칸의 반 (직렬). 첫 칸 몫이 없으면 성문이 벌어져 ρd/A 가 작아질 때 유량–첫 칸 압력의 엇갈린 풀이가
                # dt < 2√(L_g C) 를 잃고 발산했다 (쉼 변위 1.75 mm·면적 3 cm² 에서 p_0 → −10⁸, §52.488) — 구강 이음매처럼 양쪽 반 칸을 넣는다.
                a0 = max(A[n, 0], A_FLOOR)
                Lg = RHO * (d1 / ga1 + d2 / ga2 + 0.5 * dx / a0)
                Rv = 12.0 * MU * Lc * Lc * (d1 / (ga1 * ga1 * ga1) + d2 / (ga2 * ga2 * ga2))
        else:
            ag = max(Ag[n], A_FLOOR)
            Lg = RHO * GLOTTIS_D / ag
            Rv = 12.0 * MU * GLOTTIS_D * GLOTTIS_LEN * GLOTTIS_LEN / (ag * ag * ag)
        Lg += LgT
        psub = TpT[n, NT - 1] if NT > 0 else Ps[n]
        Rk = GLOTTIS_KE * RHO / (2.0 * ag * ag)
        ug0 = TUg[n]
        if GLOT_NOISE_3D:
            Lre = Lc if (VF and RE_USE_LC) else GLOTTIS_LEN
            ach = max(Ag[n], 0.0) if VF else 0.0                      # 성대 모드에서 Ag 는 뒤틈 면적
            ugs = TUgs[n] + kg * (_gq(ug0, ag, Lre, ach) - TUgs[n])    # 1 ms 로 고른 길이 방향 제트 세기 (§52.524)
            TUgs[n + 1] = ugs
            Re_g = 2.0 * abs(ug0) / (Lre * NU)
            pn_g = noise_g * ugs * xi_g[n]
        else:
            ugs = TUgs[n] + kg * (abs(ug0) - TUgs[n])
            TUgs[n + 1] = ugs
            # 틈의 수력 지름 2A/ℓ — ℓ 은 **실제 성대 길이** (§52.517). 고정 1.2 cm (옛 운동학 성문) 는 이 여성 화자의 막성 성대(적합 길이 0.7–1.0 cm)
            # 보다 길어 레이놀즈 수를 작게 내, 모음 중 성문 난류가 한 번도 임계(1800)를 넘지 못했다 — 원본 모음의 고역은 난류로 흩어져 있다.
            Lre = Lc if (VF and RE_USE_LC) else GLOTTIS_LEN
            Re_g = 2.0 * ugs / (Lre * NU)
            pn_g = 0.0
            hgg = 0.0
            if TURB_SMOOTH:
                if Re_g > 0.0:
                    hgg, _dhg = _spos(1.0 - (RE_CRIT / Re_g) ** 2, TURB_GATE_W)
            elif Re_g > RE_CRIT:
                hgg = 1.0 - (RE_CRIT / Re_g) ** 2
            if hgg > 0.0:
                vg = ugs / ag
                if TURB_SMOOTH:
                    q, _d1, _d2 = _smin(RHO * vg * vg, 2.0 * abs(Ps[n]), TURB_CAP_W)
                else:
                    q = min(RHO * vg * vg, 2.0 * abs(Ps[n]))
                if GLOT_NOISE_STEVENS:
                    # 협착 난류와 같은 Stevens 척도 — 음원 ∝ 동압 × v √A (§52.517). 동압만으로는 닫히는 순간 유량·면적이 함께 0 으로 가도 유속이
                    # 커서 닫힘마다 잡음이 칼날처럼 터졌고, 주기마다 같은 모양으로 변조된 잡음은 스펙트럼이 f0 만큼 밀린 같은 잡음의 합이라 고역 가로줄을
                    # 만들었다 (VTL C 성문 난류 −6: 8–16 kHz 줄 0.09 → 0.34). v√A 는 면적과 함께 0 으로 가 세기가 열린 동안 매끈하게 오르내린다.
                    q = q * vg * math.sqrt(ag) / (V_REF * math.sqrt(A_REF))
                if GLOT_NOISE_JET:
                    q = q * (1.0 - math.exp(-GLOT_JET_CORE * ag / (Lre * GLOT_JET_D)))
                pn_g = noise_g * q * hgg * xi_g[n]
        c_ = psub - Tp[n, 0] + (0.0 if GLOT_NOISE_SUPRA else pn_g) + (Lg / dt) * ug0
        b_ = Lg / dt + Rv
        if c_ >= 0.0:
            ug = 2.0 * c_ / (b_ + math.sqrt(b_ * b_ + 4.0 * Rk * c_))
        else:
            ug = -2.0 * (-c_) / (b_ + math.sqrt(b_ * b_ + 4.0 * Rk * (-c_)))
        TUg[n + 1] = ug
        if VF >= 1:
            # 닫힘 잡기 (고리가 없어도 기록한다). PP[11] 0: 막부 면적이 이 주기 마루의 반 아래에서 0 에 닿거나 골을 지날 때.
            # 1: **최대 유량 감소율** — 평활한 dU_g/dt 가 음의 봉우리를 지난 때 (성도를 들뜨게 하는 때; 열림에서 무장, 봉우리의 0.3 로 돌아오면 잡는다)
            evn = -1.0
            if PP[11] < 0.5:
                gm = gmn
                if gm > pk:
                    pk = gm
                if not armed:
                    if gm > max(0.6 * plast, PP[6]):
                        armed = True
                elif pk > PP[6]:
                    if gm <= 1e-9:
                        evn = float(n)
                    elif gm > gprev and gprev < 0.5 * pk:
                        evn = float(n - 1)
                gprev = gm
                if evn >= 0.0:
                    plast = pk
                    pk = 0.0
            else:
                dsm += ksm * ((ug - ug0) / dt - dsm)
                if not armed:
                    if dsm > max(0.2 * dlast, PP[12]):
                        armed = True
                        dmin = 0.0
                else:
                    if dsm < dmin:
                        dmin = dsm
                        tmin = float(n)
                    if dmin < -max(0.3 * dlast, PP[12]) and dsm > 0.3 * dmin:
                        evn = tmin
                        dlast = -dmin
            if evn >= 0.0:
                armed = False
                last_ev = evn
                if nev < EV.shape[0]:
                    EV[nev] = evn
                    nev += 1
            if evn >= 0.0 and pll:
                    while jp + 1 < npt and PT[jp + 1] <= evn:
                        jp += 1
                    jj = jp
                    if jp + 1 < npt and PT[jp + 1] - evn < evn - PT[jp]:
                        jj = jp + 1
                    per = PT[jj + 1] - PT[jj] if jj + 1 < npt else PT[jj] - PT[jj - 1]
                    e = evn - PT[jj]
                    # 주파수 고리: 잰 닫힘 간격 / 목표 주기 — 위상이 풀렸을 때만 (|e| ≥ 0.3 주기) 쓴다. 잠긴 동안 쓰면 합성 자신의 주기 떨림이
                    # 적분에 쌓여 긴장이 흔들렸다 (1–3 kHz 배음/잡음 비 15.4 → 11.3 dB, 포락 65.5 → 63.4 %).
                    ivl = evn - pev
                    if (PP[13] > 0.0 and abs(e) >= 0.3 * per and per > PP[7] and per < PP[8]
                            and ivl > 0.5 * per and ivl < 1.6 * per):
                        vint = min(max(vint + PP[13] * PP[4] * math.log(ivl / per), -PP[3]), PP[3])
                    if per > PP[7] and per < PP[8] and abs(e) < 0.45 * per:
                        ph = e / per                      # + 면 늦다 → 긴장을 올려 다음 주기를 줄인다
                        vint = min(max(vint + PP[1] * PP[4] * ph, -PP[3]), PP[3])
                        # 앞먹임: 다음 목표 주기가 이웃 평균(기본 긴장이 따르는 매끈한 f0)과 다른 만큼 (주기별 떨림)
                        uff = 0.0
                        if PP[9] > 0.0 and jj >= 1 and jj + 2 < npt:
                            pav = (PT[jj + 2] - PT[jj - 1]) / 3.0
                            if pav > PP[7] and pav < PP[8] and per > 0.7 * pav and per < 1.4 * pav:
                                uff = PP[9] * PP[4] * math.log(pav / per)
                        uc = min(max(vint + PP[0] * PP[4] * ph + uff, -PP[2]), PP[2])
                    else:
                        uc = vint
            if evn >= 0.0:
                pev = evn
        if VF == 3:
            # ---- 모드 3 운동: (M + C dt + K dt²) x' = F dt² + M(2x − x_) + C dt x — 접촉 강성은 암시적, 접촉 쉼 몫은 힘으로
            p0 = Tp[n, 0]
            ps = psub
            Lc = VR[n, 10]
            dys = VR[n, 11] / NS3
            Qn = VQ[n]
            if LR3:
                _ns_mats_lr(n, VR, VS, VQ, S3, NS3, Km3, Cm3, Mv3)
            elif LM3 == 2:
                _ns_mats2(n, VR, VS, VQ, S3, S3c, S3b, NS3, Km3, Cm3, Mv3)
            else:
                _ns_mats(n, VR, VS, VQ, S3, NS3, Km3, Cm3, Mv3)
            T2 = dt * dt
            mw = 0.0
            chi_ = 1.0
            agq = ag if (VF_NS_SMOOTH and VF_SEP_P > 0.0) else agm      # 분리 기준 = 매끈한 최소 (§52.523)
            for k in range(NS3 + 1):
                if k < NS3:
                    if VF_NS_SMOOTH:
                        zc = min(max(1.0 - (G3[k] - agq) / (VF_NS_SEP * agq), 0.0), 1.0)
                        chi_ *= 1.0 - zc * zc * (3.0 - 2.0 * zc)
                        rr = agq / G3[k]
                        pk_ = p0 + (ps - p0) * (1.0 - rr * rr) * chi_
                    else:
                        wj = min(max(1.0 - (G3[k] - agm) / (VF_NS_SEP * agm), 0.0), 1.0)
                        if wj > mw:
                            mw = wj
                        rr = agm / G3[k]
                        pk_ = p0 + (ps - p0) * (1.0 - rr * rr) * (1.0 - mw)
                    PK3[k] = pk_
                    F = pk_ * dys * O3[k] - VS[14] * dys * C3[k] * Qn
                    if k == 0:
                        F += 0.25 * (ps + pk_) * VS[12] * Lc * (2.0 / math.pi)
                    if k == NS3 - 1:
                        F += p0 * 0.5 * VS[13] * Lc * (2.0 / math.pi)
                else:
                    F = VR[n, 9]
                R3[k] = F
            if LM3 == 2:
                for k in range(NS3 + 1, D3):             # 둘째 길이 모드 — 균일 압력은 ∫sin(2πs) = 0, 열린 몫 · 접촉의 앞뒤 비대칭으로만 받는다
                    kk_ = k - NS3 - 1
                    R3[k] = PK3[kk_] * dys * O3b[kk_] - VS[14] * dys * C3b[kk_] * Qn
            if LR3:                                      # 좌우 판: 두 성대가 같은 압력 · 같은 몸체 하중을 받는다
                for k in range(NS3 + 1):
                    R3[NS3 + 1 + k] = R3[k]
            for k in range(D3):
                F = R3[k]
                cx = 0.0
                for j in range(D3):
                    cx += Cm3[k, j] * TX[n + 1, j]
                    Am3[k, j] = Cm3[k, j] * dt + Km3[k, j] * T2
                Am3[k, k] += Mv3[k]
                R3[k] = F * T2 + Mv3[k] * (2.0 * TX[n + 1, k] - TX[n, k]) + dt * cx
            _spd_solve(Am3, R3, Y3)
            for k in range(D3):
                TX[n + 2, k] = Y3[k]
        elif VF:
            # ---- 성대 운동 (VTL `TriangularGlottis::incTime` 과 같은 반암시적 차분). 날에 걸리는 압력은 준정상 베르누이 + 흐름 분리
            # (Story & Titze 1995): 분리는 좁은 자리에서 — 위 질량은 늘 성문 위 압력 p_0, 아래 질량은 수렴(a1 > a2)일 때
            # P_s − (P_s − p_0)(a2/a1)², 발산이면 p_0. 열릴 때(수렴)와 닫힐 때(발산)의 이 비대칭이 떨림에 에너지를 댄다.
            p0 = Tp[n, 0]
            ps = psub
            if ga1 > ga2:
                rr = ga2 / ga1
                p1 = ps - (ps - p0) * rr * rr
            else:
                p1 = p0
            F1 = p1 * o1 * d1 + 0.25 * (ps + p1) * VS[12] * Lc
            F2 = p0 * (o2 * d2 + 0.5 * VS[13] * Lc)
            m1 = VS[3] / Qn
            m2 = VS[4] / Qn
            k1 = VS[7] * Qn
            k2 = VS[8] * Qn
            kc1 = VS[9] * Qn
            kc2 = VS[10] * Qn
            kp = VS[11]
            R1 = 2.0 * (VS[5] + al1) * math.sqrt(VS[3] * VS[7])          # 닿은 몫만큼 임계 감쇠를 더한다 (VTL)
            R2 = 2.0 * (VS[6] + al2) * math.sqrt(VS[4] * VS[8])
            T2 = dt * dt
            A11 = m1 + R1 * dt + T2 * (k1 + kc1 * al1 + kp)
            A22 = m2 + R2 * dt + T2 * (k2 + kc2 * al2 + kp)
            Bc = -kp * T2
            E1 = F1 * T2 + 2.0 * m1 * x1 - m1 * TX[n, 0] + R1 * dt * x1 - T2 * kc1 * al1 * rs1
            E2 = F2 * T2 + 2.0 * m2 * x2 - m2 * TX[n, 1] + R2 * dt * x2 - T2 * kc2 * al2 * rs2
            if VF == 2:
                # 몸체–덮개 (§52.500): 덮개 스프링·감쇠는 몸체와의 상대 변위·속도에, 몸체는 벽에. 3 차 항은 앞 표본 값으로.
                xb = TX[n + 1, 2]
                mb = VS[14] / Qn
                kb = VS[15] * Qn * VR[n, 2]
                Rb = 2.0 * VS[16] * math.sqrt(VS[14] * VS[15])
                D1 = x1 - xb
                D2 = x2 - xb
                c1 = T2 * k1 * VS[17] * D1 * D1 * D1
                c2 = T2 * k2 * VS[17] * D2 * D2 * D2
                cb = T2 * kb * VS[18] * xb * xb * xb
                E1 = E1 - R1 * dt * xb - c1
                E2 = E2 - R2 * dt * xb - c2
                Eb = 2.0 * mb * xb - mb * TX[n, 2] + Rb * dt * xb - R1 * dt * D1 - R2 * dt * D2 + c1 + c2 - cb
                Ab = mb + Rb * dt + (R1 + R2) * dt + T2 * (kb + k1 + k2)
                y1, y2, yb = _sym3_solve(A11, A22, Ab, Bc, -(k1 * T2 + R1 * dt), -(k2 * T2 + R2 * dt), E1, E2, Eb)
                TX[n + 2, 0] = y1
                TX[n + 2, 1] = y2
                TX[n + 2, 2] = yb
            else:
                det = A11 * A22 - Bc * Bc
                TX[n + 2, 0] = (E1 * A22 - Bc * E2) / det
                TX[n + 2, 1] = (A11 * E2 - Bc * E1) / det
        # ---- 기관 이음매: 폐 → 칸 0 은 반 칸 관성 + 정합 저항 ρc/A (폐·기관지 쪽으로 흡수), 칸 사이는 양쪽 반 칸
        for j in range(NT):
            b2 = TRA[j]
            if j == 0:
                Jt = RHO_T * 0.5 * dxt / b2 / dt
                ah = b2
                rl = RHO_T * C_T / b2
                dpt = Ps[n] - TpT[n, 0]
            else:
                b1 = TRA[j - 1]
                Jt = 0.5 * RHO_T * dxt * (1.0 / b1 + 1.0 / b2) / dt
                ah = 2.0 * b1 * b2 / (b1 + b2)
                rl = 0.0
                dpt = TpT[n, j - 1] - TpT[n, j]
            rft = (2.0 * math.sqrt(math.pi * ah) * dxt / (ah * ah)) * KvT if loss_on else 0.0
            r0t = 8.0 * math.pi * MU_T * dxt / (ah * ah) if loss_on else 0.0
            F = 0.0
            for m in range(M):
                F += XW[m] * phvT[j, m]
            TFvT[n, j] = F
            TUT[n + 1, j] = (TUT[n, j] * Jt + dpt + rft * F) / (Jt + rl + r0t + rft * W)
            for m in range(M):
                phvT[j, m] += XA[m] * (TUT[n + 1, j] - phvT[j, m])
        # ---- 구강 이음매: 관성 ρ dx/2 (1/A_i + 1/A_{i+1}) (양쪽 반 칸 직렬 — 면적비와 무관하게 c·dt ≤ dx 로 안정)
        remax = 0.0
        for i in range(N - 1):
            a1 = max(A[n, i], A_FLOOR)
            a2 = max(A[n, i + 1], A_FLOOR)
            J = 0.5 * RHO * dx * (1.0 / a1 + 1.0 / a2) / dt
            ah = 2.0 * a1 * a2 / (a1 + a2)
            rf = (2.0 * math.sqrt(math.pi * ah) * dx / (ah * ah)) * Kv if loss_on else 0.0
            r0 = 8.0 * math.pi * MU * dx / (ah * ah) if loss_on else 0.0
            F = 0.0
            for m in range(M):
                F += XW[m] * phv[i, m]
            TFv[n, i] = F
            amin = min(a1, a2)
            us = TUs[n, i] + kc * (TU[n, i] - TUs[n, i])
            TUs[n + 1, i] = us
            aup = a1 if us >= 0.0 else a2                # 흐름 방향 위쪽 칸 — 제트가 나오는 목 (§52.524)
            Re = abs(us) * 2.0 / (math.sqrt(math.pi * (aup if TURB_SMOOTH else amin)) * NU)
            if Re > remax:
                remax = Re
            pn = 0.0
            # 협착 난류: 흐름 방향으로 넓어지는 이음매(제트)에서, 레이놀즈 수가 임계를 넘는 그 순간에만. 세기 = 동압 ρv² (베르누이 상한 2 P_s)
            # × Stevens 배율 (v/V_REF)(A/A_REF)^½ × 장애물. 상한은 **동압에만** 건다 — 곱 전체에 걸면 ㅅ 에서 배율이 통째로 잘려(§52.483)
            # 협착 모양이 세기를 못 정했다.
            if JET_BC:
                jet = (us > 0.0 and a2 > a1) or (us < 0.0 and a1 > a2)
            else:
                jet = (us > 0.0 and a2 > 1.2 * a1) or (us < 0.0 and a1 > 1.2 * a2)
            if TURB_SMOOTH and JET_BC:
                hg = 0.0
                if Re > 0.0:
                    hg, _dh = _spos(1.0 - (RE_CRIT / Re) ** 2, TURB_GATE_W)
                if hg > 0.0:
                    v = us / aup
                    gob = 1.0 + OBS_GAIN if (i + 1) >= OBS_FRAC * N else 1.0
                    av = abs(v)
                    dyn, _d1, _d2 = _smin(RHO * av * av, 2.0 * abs(Ps[n]), TURB_CAP_W)
                    q = dyn * av * math.sqrt(aup) * gob / (V_REF * math.sqrt(A_REF))
                    amx_, _dxj, _j0, _dr, _ns = _jet_far(A, n, i, us, a1, a2, N, dx, JG)
                    rj = aup / amx_
                    if rj < 1.0:
                        pn = noise_c * q * (1.0 - rj) * (1.0 - rj) * hg * xi_c[n, i]
            elif Re > RE_CRIT and jet:
                v = us / amin
                gob = 1.0 + OBS_GAIN if (i + 1) >= OBS_FRAC * N else 1.0
                av = abs(v)
                dyn = min(RHO * av * av, 2.0 * abs(Ps[n]))
                q = dyn * av * math.sqrt(amin) * gob / (V_REF * math.sqrt(A_REF))
                if JET_BC:
                    amx_, _dxj, _j0, _dr, _ns = _jet_far(A, n, i, us, a1, a2, N, dx, JG)
                    rj = amin / amx_
                    q *= (1.0 - rj) * (1.0 - rj)        # 제트의 전단 동압 ρ(v₁ − v₂)² = ρv₁²(1 − A₁/A₂)² — A₂ 는 제트 발달 거리 안의 가장 넓은 칸
                pn = noise_c * q * (1.0 - (RE_CRIT / Re) ** 2) * xi_c[n, i]
            # 분출 손실 (Borda–Carnot): 흐름 방향으로 넓어지는 이음매에서 ½ρU²(1/A_좁음 − 1/A_넓음)² — 제트의 동압이 회복되지 않는다.
            # 성문과 같은 이차식 닫힌 꼴 (b U + k U|U| = c). 없으면 성문이 벌어질 때 협착이 흐름을 못 막아 ㅅ 유량이 1600 cm³/s 를 넘었다 (§52.483).
            dk = 1.0 / a1 - 1.0 / a2
            if GLOT_NOISE_SUPRA and i == GLOT_NOISE_J:
                pn += pn_g                         # 성문 제트의 난류 — 가성대·후두개에 부딪히는 성문 위 자리의 압력 음원 (§52.518)
            c_ = TU[n, i] * J + (Tp[n, i] - Tp[n, i + 1]) + pn + rf * F
            b_ = J + r0 + rf * W
            if c_ >= 0.0:
                kk = 0.5 * RHO * dk * dk if dk > 0.0 else 0.0
                TU[n + 1, i] = 2.0 * c_ / (b_ + math.sqrt(b_ * b_ + 4.0 * kk * c_))
            else:
                kk = 0.5 * RHO * dk * dk if dk < 0.0 else 0.0
                TU[n + 1, i] = -2.0 * (-c_) / (b_ + math.sqrt(b_ * b_ + 4.0 * kk * (-c_)))
            for m in range(M):
                phv[i, m] += XA[m] * (TU[n + 1, i] - phv[i, m])
        # ---- 연구개 포트: (ρ d / A_v) dU/dt + R_v U = p_구강[iv] − p_비강[0], R_v = 8πμd/A_v² (푸아죄유 — 거의 닫힌 틈은 흐르지 못한다)
        # 포트 관성 = 포트 두께 + 구강 칸 반 (직렬). 포트 몫만 두었더니 연구개 자리가 0.1 cm² 로 좁고 포트가 0.47 cm² 일 때(ㄱ 앞 비음화)
        # 이음매 어드미턴스가 그 칸의 몫을 넘어 발산했다 (§52.478, S_selfdry15 의 비유한 손실).
        av = max(Av[n], A_FLOOR)
        aiv = max(A[n, iv], A_FLOOR)
        Jv = RHO * (VEL_D / av + 0.5 * dx / aiv) / dt
        Rpv = 8.0 * math.pi * MU * VEL_D / (av * av)
        Rkv = VEL_KE * 0.5 * RHO * abs(TUv[n]) / (av * av)           # 분출 손실 (§52.530)
        TUv[n + 1] = (TUv[n] * Jv + (Tp[n, iv] - TpN[n, 0])) / (Jv + Rpv + Rkv)
        # ---- 비강 이음매 (고정 기하, 큰 표면 손실)
        for j in range(NN - 1):
            b1 = AN[j]
            b2 = AN[j + 1]
            Jn = 0.5 * RHO_N * dxn * (1.0 / b1 + 1.0 / b2) / dt
            ah = 2.0 * b1 * b2 / (b1 + b2)
            rfn = NAS_LOSS * (2.0 * math.sqrt(math.pi * ah) * dxn / (ah * ah)) * KvN if loss_on else 0.0
            r0n = 8.0 * math.pi * MU_N * dxn / (ah * ah) if loss_on else 0.0
            F = 0.0
            for m in range(M):
                F += XW[m] * phvN[j, m]
            TFvN[n, j] = F
            TUN[n + 1, j] = (TUN[n, j] * Jn + (TpN[n, j] - TpN[n, j + 1]) + rfn * F) / (Jn + r0n + rfn * W)
            for m in range(M):
                phvN[j, m] += XA[m] * (TUN[n + 1, j] - phvN[j, m])
        # ---- 이상와 이음매: 입구 = 곁관 한 칸 + 구강 칸 반 (직렬 관성) + 칸 사이, 끝은 막힘 (유량 0)
        #      입구 관성에 구강 칸 반(ρ dx / 2A_ip)을 넣어야 구강 칸이 좁아져도 이음매 어드미턴스가 그 칸의 몫을 넘지 않는다 —
        #      곁관 몫만 두었더니 전체 발화 적합에서 A_ip 0.16 cm² (PA/A_ip = 5) 에서 발산했다.
        for bb in range(SBA.shape[0]):
            PA = SBA[bb]
            if PA <= 0.0:
                continue
            ip = SBI[bb]
            nas = ip >= N                                      # 비강 칸에 붙은 곁관 (부비동)
            ipn = ip - N if nas else 0
            dxp = SBL[bb] / NPB
            aip = AN[ipn] if nas else max(A[n, ip], A_FLOOR)
            dxa = dxn if nas else dx
            for jj in range(NPB):
                j = bb * NPB + jj
                h_ = 1.0
                an_ = PA
                ln_ = dxp
                if jj == 0 and SBNA[bb] > 0.0:                 # 목 (헬름홀츠 곁공동의 입구)
                    an_ = SBNA[bb]
                    ln_ = SBNL[bb]
                rho_b = RHO_N if nas else RHO                  # 부비동은 비강 공기, 후두 곁관은 성도 공기 (§52.524)
                Jp = rho_b * h_ * ln_ / an_ / dt + (rho_b * 0.5 * dxa / aip / dt if jj == 0 else 0.0)
                rfp = (2.0 * math.sqrt(math.pi * an_) * h_ * ln_ / (an_ * an_)) * (KvN if nas else Kv) if loss_on else 0.0
                r0p = 8.0 * math.pi * (MU_N if nas else MU) * h_ * ln_ / (an_ * an_) if loss_on else 0.0
                F = 0.0
                for m in range(M):
                    F += XW[m] * phvP[j, m]
                TFvP[n, j] = F
                pa_ = (TpN[n, ipn] if nas else Tp[n, ip]) if jj == 0 else TpP[n, j - 1]
                TUP[n + 1, j] = (TUP[n, j] * Jp + (pa_ - TpP[n, j]) + rfp * F) / (Jp + r0p + rfp * W)
                for m in range(M):
                    phvP[j, m] += XA[m] * (TUP[n + 1, j] - phvP[j, m])
        # ---- 입술: 반 칸 관성 + 방사 임피던스 R_r ‖ L_r (Flanagan 1972) — 암시적
        am = max(A[n, N - 1], A_FLOOR)
        Rr = 128.0 * RHO * C_SOUND / (9.0 * math.pi * math.pi * am)
        Lr = 8.0 * RHO / (3.0 * math.pi * math.sqrt(math.pi * am))
        I_ = RHO * 0.5 * dx / am / dt
        kap = Rr / (1.0 + Rr * dt / Lr)
        um = (TUm[n] * I_ + Tp[n, N - 1] + kap * TUL[n]) / (I_ + kap)
        pm = kap * (um - TUL[n])
        TUm[n + 1] = um
        Tpm[n + 1] = pm
        TUL[n + 1] = TUL[n] + dt * pm / Lr
        # ---- 콧구멍: 같은 방사 임피던스 (면적 고정)
        an = AN[NN - 1]
        RrN = 128.0 * RHO_N * C_N / (9.0 * math.pi * math.pi * an)
        LrN = 8.0 * RHO_N / (3.0 * math.pi * math.sqrt(math.pi * an))
        IN_ = RHO_N * 0.5 * dxn / an / dt
        kapN = RrN / (1.0 + RrN * dt / LrN)
        umN = (TUmN[n] * IN_ + TpN[n, NN - 1] + kapN * TULN[n]) / (IN_ + kapN)
        pmN = kapN * (umN - TULN[n])
        TUmN[n + 1] = umN
        TpmN[n + 1] = pmN
        TULN[n + 1] = TULN[n] + dt * pmN / LrN
        out[n] = (um - TUm[n] + umN - TUmN[n]) / dt
        # ---- 구강 칸: (A dx / ρc²) dp/dt = U_in − U_out − U_벽 − G p − dV/dt (− U_포트 at iv)
        for i in range(N):
            ai = max(A[n, i], A_FLOOR)
            uin = TUg[n + 1] if i == 0 else TU[n + 1, i - 1]
            uout = TUm[n + 1] if i == N - 1 else TU[n + 1, i]
            if i == iv:
                uout += TUv[n + 1]
            for bb in range(SBA.shape[0]):
                if SBA[bb] > 0.0 and SBI[bb] == i:
                    uout += TUP[n + 1, bb * NPB]
            S = 2.0 * math.sqrt(math.pi * ai)
            uw = 0.0
            if wall_on:
                vw = (Tvw[n, i] * WALL_M / dt + Tp[n, i] - WALL_K * Tyw[n, i]) / (WALL_M / dt + WALL_R)
                Tvw[n + 1, i] = vw
                Tyw[n + 1, i] = Tyw[n, i] + dt * vw
                uw = S * dx * vw
            else:
                Tvw[n + 1, i] = 0.0
                Tyw[n + 1, i] = 0.0
            g = (S * dx / rc2) * Kt if loss_on else 0.0
            dV = Q[n, i]           # 부피 변화 유량 [cm³/s] — 풀이기 밖에서 매끄럽게 계산한다 (프레임 보간의 계단이 고역을 냈다)
            C = ai * dx / rc2 / dt
            F = 0.0
            for m in range(M):
                F += XW[m] * pht[i, m]
            TFt[n, i] = F
            Tp[n + 1, i] = (Tp[n, i] * C + uin - uout - uw - dV + g * F) / (C + g * W)
            for m in range(M):
                pht[i, m] += XA[m] * (Tp[n + 1, i] - pht[i, m])
        # ---- 비강 칸
        for j in range(NN):
            bj = AN[j]
            S = 2.0 * math.sqrt(math.pi * bj)
            C = bj * dxn / rc2N / dt
            g = NAS_LOSS * (S * dxn / rc2N) * KtN if loss_on else 0.0
            uin = TUv[n + 1] if j == 0 else TUN[n + 1, j - 1]
            uout = TUmN[n + 1] if j == NN - 1 else TUN[n + 1, j]
            for bb in range(SBA.shape[0]):
                if SBA[bb] > 0.0 and SBI[bb] == N + j:
                    uout += TUP[n + 1, bb * NPB]
            F = 0.0
            for m in range(M):
                F += XW[m] * phtN[j, m]
            TFtN[n, j] = F
            TpN[n + 1, j] = (TpN[n, j] * C + uin - uout + g * F) / (C + g * W)
            for m in range(M):
                phtN[j, m] += XA[m] * (TpN[n + 1, j] - phtN[j, m])
        # ---- 기관 칸 (맨 윗칸에서 성문으로 나간다)
        for j in range(NT):
            bj = TRA[j]
            S = 2.0 * math.sqrt(math.pi * bj)
            C = bj * dxt / rc2T / dt
            g = (S * dxt / rc2T) * KtT if loss_on else 0.0
            uout = TUT[n + 1, j + 1] if j < NT - 1 else TUg[n + 1]
            F = 0.0
            for m in range(M):
                F += XW[m] * phtT[j, m]
            TFtT[n, j] = F
            TpT[n + 1, j] = (TpT[n, j] * C + TUT[n + 1, j] - uout + g * F) / (C + g * W)
            for m in range(M):
                phtT[j, m] += XA[m] * (TpT[n + 1, j] - phtT[j, m])
        # ---- 이상와 칸 (닫힌 끝)
        for bb in range(SBA.shape[0]):
            PA = SBA[bb]
            if PA <= 0.0:
                continue
            dxp = SBL[bb] / NPB
            nasb = SBI[bb] >= N
            rc2b = rc2N if nasb else rc2
            ktb = KtN if nasb else Kt
            for jj in range(NPB):
                j = bb * NPB + jj
                C = PA * dxp / rc2b / dt
                g = (2.0 * math.sqrt(math.pi * PA) * dxp / rc2b) * ktb if loss_on else 0.0
                uout = TUP[n + 1, j + 1] if jj < NPB - 1 else 0.0
                F = 0.0
                for m in range(M):
                    F += XW[m] * phtP[j, m]
                TFtP[n, j] = F
                TpP[n + 1, j] = (TpP[n, j] * C + TUP[n + 1, j] - uout + g * F) / (C + g * W)
                for m in range(M):
                    phtP[j, m] += XA[m] * (TpP[n + 1, j] - phtP[j, m])
        rec[n, 0] = ug
        rec[n, 1] = Tp[n + 1, 0]
        rec[n, 2] = um
        rec[n, 3] = Re_g
        rec[n, 4] = remax
        rec[n, 5] = pm
        rec[n, 6] = umN
        rec[n, 7] = ga1 if VF else ag
        rec[n, 8] = ga2 if VF else ag


def _bwd(A, L, Ag, Ps, Av, Q, AN, NL, TRA, TRL, xi_g, xi_c, noise_g, noise_c, wall_on, loss_on, fs, SBA, SBL, SBI, SBNA, SBNL, XA, XW, VF, VQ, VR, VS, gout, gGA,
         Tp, TU, TUg, TUm, TUL, Tpm, Tvw, Tyw, TUs, TUgs, TpN, TUN, TUv, TUmN, TULN, TpmN, TpP, TUP,
         TFv, TFt, TFvN, TFtN, TFvP, TFtP, TX, TpT, TUT, TFvT, TFtT, TPH,
         gA, gL, gAg, gPs, gAv, gQ, gng, gVQ, gVR, gVS):
    """`_fwd` 의 수반(역방향). gout (T,) = ∂손실/∂out → gA, gL, gAg, gPs, gAv, gng (2,) [noise_g, noise_c] 에 **더한다**.

    단계 안의 순방향 순서를 거꾸로: 비강 칸 → 구강 칸 → 출력 → 콧구멍 → 입술 → 비강 이음매 → 포트 → 구강 이음매 → 성문.
    l* 는 단계 n **뒤** 상태(기록 줄 n+1)의 수반, m* 는 **앞** 상태(줄 n)의 수반.
    경계층 손실의 확산 상태 φ_m 은 단계마다 맨 먼저 되감는다: φ' = (1−a)φ + a x' → x' 의 수반에 a·lφ, 앞 φ 의 수반에 (1−a)·lφ.
    """
    T, N = A.shape
    NN = AN.shape[0]
    NP = TpP.shape[1]
    M = XA.shape[0]
    W = 0.0
    for m in range(M):
        W += XW[m]
    lphv = np.zeros((N - 1, M))
    lpht = np.zeros((N, M))
    lphvN = np.zeros((NN - 1, M))
    lphtN = np.zeros((NN, M))
    lphvP = np.zeros((NP, M))
    lphtP = np.zeros((NP, M))
    dt = 1.0 / fs
    rc2 = RHO * C_SOUND * C_SOUND
    kc, kg, Kv, Kt = _consts(fs)
    KvN, KtN = _kvkt(RHO_N, MU_N, GAMMA_N, LAMBDA_N, CP_N)
    KvT, KtT = _kvkt(RHO_T, MU_T, GAMMA_T, LAMBDA_T, CP_T)
    rc2N = RHO_N * C_N * C_N
    rc2T = RHO_T * C_T * C_T
    iv = int(VEL_FRAC * N)
    NPB = NP // max(SBA.shape[0], 1)          # 곁관 하나의 칸 수
    dxn = NL / NN
    lpP = np.zeros(NP)
    lUP = np.zeros(NP)
    lp = np.zeros(N)
    lU = np.zeros(N - 1)
    lvw = np.zeros(N)
    lyw = np.zeros(N)
    lUs = np.zeros(N - 1)
    lpN = np.zeros(NN)
    lUN = np.zeros(NN - 1)
    lUg = 0.0
    lUm = 0.0
    lUL = 0.0
    lUgs = 0.0
    lUv = 0.0
    lUmN = 0.0
    lULN = 0.0
    NT = TRA.shape[0]
    dxt = TRL / max(NT, 1)
    LgT = RHO_T * 0.5 * dxt / TRA[NT - 1] if NT > 0 else 0.0
    lpT = np.zeros(NT)
    lUT = np.zeros(NT)
    lphvT = np.zeros((NT, M))
    lphtT = np.zeros((NT, M))
    lX1 = 0.0                  # 성대 변위의 수반 — lX: 단계 n 이 낸 변위 x_{n+1} (완결), pX: x_n 중 단계 n+1 이 '앞 변위'로 쓴 몫
    lX2 = 0.0
    pX1 = 0.0
    pX2 = 0.0
    lXb = 0.0                  # 몸체 질량 (모드 2)
    pXb = 0.0
    LR3 = VF == 3 and VS.shape[0] > 22 and VS[20] > 0.5                 # 좌우 성대 따로 (§52.533)
    if VF == 3 and VS.shape[0] > 19 and VS[19] > 1.5 and not LR3:
        raise ValueError("둘째 길이 모드(VS[19] = 2)의 수반은 아직 없다 (§52.531) — 정방향만")
    NS3 = (TX.shape[1] // 2 - 1) if LR3 else TX.shape[1] - 1      # 모드 3
    D3 = 2 * (NS3 + 1) if LR3 else NS3 + 1
    lX3 = np.zeros(D3)
    pX3 = np.zeros(D3)
    lx3 = np.zeros(D3)
    lxp3 = np.zeros(D3)
    G3 = np.zeros(NS3 + 1)
    O3 = np.zeros(NS3 + 1)
    S3 = np.zeros(NS3 + 1)
    C3 = np.zeros(NS3 + 1)
    AK3 = np.zeros(NS3 + 1)
    DA3 = np.zeros(NS3 + 1)
    CS3 = np.zeros(NS3 + 1)
    lG3 = np.zeros(NS3 + 1)
    lO3 = np.zeros(NS3 + 1)
    lS3 = np.zeros(NS3 + 1)
    lC3 = np.zeros(NS3 + 1)
    Km3 = np.zeros((D3, D3))
    Cm3 = np.zeros((D3, D3))
    Am3 = np.zeros((D3, D3))
    Mv3 = np.zeros(D3)
    MU3 = np.zeros(D3)
    W3 = np.zeros(NS3 + 1)
    MW3 = np.zeros(NS3 + 1)
    JM3 = np.zeros(NS3 + 1, dtype=np.int64)
    ZC3 = np.zeros(NS3 + 1)                       # 분리 무게의 경사 변수 (매끈한 꼴, §52.523)
    kmin3 = 0
    agm = 1.0
    p3 = 1.0
    Lc = GLOTTIS_LEN
    sLg = 0.0
    sRv = 0.0
    dys = 0.0
    JG = np.zeros(N + 1)                       # 제트 넓어진 면적의 칸별 몫 (§52.523)
    for n in range(T - 1, -1, -1):
        dx = L[n] / N
        ldx = 0.0
        mp = np.zeros(N)
        mU = np.zeros(N - 1)
        mvw = np.zeros(N)
        myw = np.zeros(N)
        mUs = np.zeros(N - 1)
        mpN = np.zeros(NN)
        mUN = np.zeros(NN - 1)
        mpP = np.zeros(NP)
        mUP = np.zeros(NP)
        mUg = 0.0
        mUm = 0.0
        mUL = 0.0
        mUgs = 0.0
        mUv = 0.0
        mUmN = 0.0
        mULN = 0.0
        mpT = np.zeros(NT)
        mUT = np.zeros(NT)
        mphvT = np.zeros((NT, M))
        mphtT = np.zeros((NT, M))
        # ---------------- 확산 상태 갱신 (역): φ' = (1−a)φ + a x'
        mphv = np.zeros((N - 1, M))
        mpht = np.zeros((N, M))
        mphvN = np.zeros((NN - 1, M))
        mphtN = np.zeros((NN, M))
        mphvP = np.zeros((NP, M))
        mphtP = np.zeros((NP, M))
        for m in range(M):
            a_ = XA[m]
            for i in range(N - 1):
                lU[i] += a_ * lphv[i, m]
                mphv[i, m] = (1.0 - a_) * lphv[i, m]
            for i in range(N):
                lp[i] += a_ * lpht[i, m]
                mpht[i, m] = (1.0 - a_) * lpht[i, m]
            for j in range(NN - 1):
                lUN[j] += a_ * lphvN[j, m]
                mphvN[j, m] = (1.0 - a_) * lphvN[j, m]
            for j in range(NN):
                lpN[j] += a_ * lphtN[j, m]
                mphtN[j, m] = (1.0 - a_) * lphtN[j, m]
            for j in range(NT):
                lUT[j] += a_ * lphvT[j, m]
                mphvT[j, m] = (1.0 - a_) * lphvT[j, m]
                lpT[j] += a_ * lphtT[j, m]
                mphtT[j, m] = (1.0 - a_) * lphtT[j, m]
            if NP > 0:
                for j in range(NP):
                    lUP[j] += a_ * lphvP[j, m]
                    mphvP[j, m] = (1.0 - a_) * lphvP[j, m]
                    lpP[j] += a_ * lphtP[j, m]
                    mphtP[j, m] = (1.0 - a_) * lphtP[j, m]
        # ---------------- 기관 칸 (역)
        for j in range(NT):
            bj = TRA[j]
            S = 2.0 * math.sqrt(math.pi * bj)
            C = bj * dxt / rc2T / dt
            g = (S * dxt / rc2T) * KtT if loss_on else 0.0
            D = C + g * W
            mpT[j] += lpT[j] * C / D
            lF = lpT[j] / D
            for m in range(M):
                mphtT[j, m] += lpT[j] * g * XW[m] / D
            lUT[j] += lF
            if j < NT - 1:
                lUT[j + 1] -= lF
            else:
                lUg -= lF
        # ---------------- 이상와 칸 (역)
        for bb in range(SBA.shape[0]):
            PA = SBA[bb]
            if PA <= 0.0:
                continue
            dxp = SBL[bb] / NPB
            nasb = SBI[bb] >= N
            rc2b = rc2N if nasb else rc2
            ktb = KtN if nasb else Kt
            for jj in range(NPB):
                j = bb * NPB + jj
                C = PA * dxp / rc2b / dt
                g = (2.0 * math.sqrt(math.pi * PA) * dxp / rc2b) * ktb if loss_on else 0.0
                D = C + g * W
                mpP[j] += lpP[j] * C / D
                lF = lpP[j] / D
                for m in range(M):
                    mphtP[j, m] += lpP[j] * g * XW[m] / D
                lUP[j] += lF
                if jj < NPB - 1:
                    lUP[j + 1] -= lF
        # ---------------- 비강 칸 (역)
        for j in range(NN):
            bj = AN[j]
            S = 2.0 * math.sqrt(math.pi * bj)
            C = bj * dxn / rc2N / dt
            g = NAS_LOSS * (S * dxn / rc2N) * KtN if loss_on else 0.0
            D = C + g * W
            mpN[j] += lpN[j] * C / D
            lF = lpN[j] / D
            for m in range(M):
                mphtN[j, m] += lpN[j] * g * XW[m] / D
            if j == 0:
                lUv += lF
            else:
                lUN[j - 1] += lF
            if j == NN - 1:
                lUmN -= lF
            else:
                lUN[j] -= lF
            for bb in range(SBA.shape[0]):
                if SBA[bb] > 0.0 and SBI[bb] == N + j:
                    lUP[bb * NPB] -= lF
        # ---------------- 구강 칸 (역)
        for i in range(N):
            ai = max(A[n, i], A_FLOOR)
            S = 2.0 * math.sqrt(math.pi * ai)
            C = ai * dx / rc2 / dt
            g = (S * dx / rc2) * Kt if loss_on else 0.0
            D = C + g * W
            pn1 = Tp[n + 1, i]
            pn0 = Tp[n, i]
            lpi = lp[i]
            mp[i] += lpi * C / D
            lF = lpi / D
            lC = lpi * (pn0 - pn1) / D
            lg = lpi * (TFt[n, i] - W * pn1) / D
            for m in range(M):
                mpht[i, m] += lpi * g * XW[m] / D
            if i == 0:
                lUg += lF
            else:
                lU[i - 1] += lF
            if i == N - 1:
                lUm -= lF
            else:
                lU[i] -= lF
            if i == iv:
                lUv -= lF
            for bb in range(SBA.shape[0]):
                if SBA[bb] > 0.0 and SBI[bb] == i:
                    lUP[bb * NPB] -= lF
            lai = 0.0
            gQ[n, i] += -lF
            lS = 0.0
            if wall_on:
                vw1 = Tvw[n + 1, i]
                luw = -lF
                lv = lvw[i] + luw * S * dx
                lS += luw * dx * vw1
                ldx += luw * S * vw1
                ly = lyw[i]
                myw[i] += ly
                lv += dt * ly
                den = WALL_M / dt + WALL_R
                mvw[i] += lv * (WALL_M / dt) / den
                mp[i] += lv / den
                myw[i] += -lv * WALL_K / den
            if loss_on:
                lS += lg * dx * Kt / rc2
                ldx += lg * S * Kt / rc2
            lai += lC * dx / (rc2 * dt)
            ldx += lC * ai / (rc2 * dt)
            lai += lS * math.pi / math.sqrt(math.pi * ai)
            if A[n, i] > A_FLOOR:
                gA[n, i] += lai
        # ---------------- 출력: out[n] = (Um_{n+1} − Um_n + UmN_{n+1} − UmN_n)/dt
        lUm += gout[n] / dt
        mUm -= gout[n] / dt
        lUmN += gout[n] / dt
        mUmN -= gout[n] / dt
        # ---------------- 콧구멍 (역) — 면적 고정
        an = AN[NN - 1]
        RrN = 128.0 * RHO_N * C_N / (9.0 * math.pi * math.pi * an)
        LrN = 8.0 * RHO_N / (3.0 * math.pi * math.sqrt(math.pi * an))
        IN_ = RHO_N * 0.5 * dxn / an / dt
        kapN = RrN / (1.0 + RrN * dt / LrN)
        mULN += lULN
        lpmN = lULN * dt / LrN
        lUmN += lpmN * kapN
        mULN += -lpmN * kapN
        EN = IN_ + kapN
        mUmN += lUmN * IN_ / EN
        mpN[NN - 1] += lUmN / EN
        mULN += lUmN * kapN / EN
        # ---------------- 입술 (역)
        am = max(A[n, N - 1], A_FLOOR)
        Rr = 128.0 * RHO * C_SOUND / (9.0 * math.pi * math.pi * am)
        Lr = 8.0 * RHO / (3.0 * math.pi * math.sqrt(math.pi * am))
        I_ = RHO * 0.5 * dx / am / dt
        kap = Rr / (1.0 + Rr * dt / Lr)
        um1 = TUm[n + 1]
        um0 = TUm[n]
        ul0 = TUL[n]
        pm1 = Tpm[n + 1]
        mUL += lUL
        lpm = lUL * dt / Lr
        lLr = -lUL * dt * pm1 / (Lr * Lr)
        lUm += lpm * kap
        mUL += -lpm * kap
        lkap = lpm * (um1 - ul0)
        E = I_ + kap
        mUm += lUm * I_ / E
        mp[N - 1] += lUm / E
        mUL += lUm * kap / E
        lI = lUm * (um0 - um1) / E
        lkap += lUm * (ul0 - um1) / E
        den = 1.0 + Rr * dt / Lr
        lRr = lkap / (den * den)
        lLr += lkap * kap * kap * dt / (Lr * Lr)
        lam = -lRr * Rr / am - lLr * 0.5 * Lr / am - lI * I_ / am
        ldx += lI * I_ / dx
        if A[n, N - 1] > A_FLOOR:
            gA[n, N - 1] += lam
        # ---------------- 이상와 이음매 (역)
        for bb in range(SBA.shape[0]):
            PA = SBA[bb]
            if PA <= 0.0:
                continue
            ip = SBI[bb]
            nas = ip >= N                                      # 비강 칸에 붙은 곁관 (부비동)
            ipn = ip - N if nas else 0
            dxp = SBL[bb] / NPB
            aip = AN[ipn] if nas else max(A[n, ip], A_FLOOR)
            dxa = dxn if nas else dx
            for jj in range(NPB):
                j = bb * NPB + jj
                h_ = 1.0
                an_ = PA
                ln_ = dxp
                if jj == 0 and SBNA[bb] > 0.0:                 # 목 (헬름홀츠 곁공동의 입구)
                    an_ = SBNA[bb]
                    ln_ = SBNL[bb]
                rho_b = RHO_N if nas else RHO                  # 부비동은 비강 공기, 후두 곁관은 성도 공기 (§52.524)
                Jp = rho_b * h_ * ln_ / an_ / dt + (rho_b * 0.5 * dxa / aip / dt if jj == 0 else 0.0)
                rfp = (2.0 * math.sqrt(math.pi * an_) * h_ * ln_ / (an_ * an_)) * (KvN if nas else Kv) if loss_on else 0.0
                r0p = 8.0 * math.pi * (MU_N if nas else MU) * h_ * ln_ / (an_ * an_) if loss_on else 0.0
                Hp = Jp + r0p + rfp * W
                lu = lUP[j]
                mUP[j] += lu * Jp / Hp
                for m in range(M):
                    mphvP[j, m] += lu * rfp * XW[m] / Hp
                if jj == 0 and not nas:
                    lJp = lu * (TUP[n, j] - TUP[n + 1, j]) / Hp
                    ldx += lJp * RHO * 0.5 / aip / dt
                    if A[n, ip] > A_FLOOR:
                        gA[n, ip] += lJp * (-RHO * 0.5 * dx / (aip * aip * dt))
                if jj == 0:
                    if nas:
                        mpN[ipn] += lu / Hp
                    else:
                        mp[ip] += lu / Hp
                else:
                    mpP[j - 1] += lu / Hp
                mpP[j] -= lu / Hp
        # ---------------- 비강 이음매 (역)
        for j in range(NN - 1):
            b1 = AN[j]
            b2 = AN[j + 1]
            Jn = 0.5 * RHO_N * dxn * (1.0 / b1 + 1.0 / b2) / dt
            ah = 2.0 * b1 * b2 / (b1 + b2)
            rfn = NAS_LOSS * (2.0 * math.sqrt(math.pi * ah) * dxn / (ah * ah)) * KvN if loss_on else 0.0
            r0n = 8.0 * math.pi * MU_N * dxn / (ah * ah) if loss_on else 0.0
            H = Jn + r0n + rfn * W
            lu = lUN[j]
            mUN[j] += lu * Jn / H
            for m in range(M):
                mphvN[j, m] += lu * rfn * XW[m] / H
            mpN[j] += lu / H
            mpN[j + 1] -= lu / H
        # ---------------- 연구개 포트 (역): U_v' = (U_v J_v + Δp) / (J_v + R_v)
        av = max(Av[n], A_FLOOR)
        aiv = max(A[n, iv], A_FLOOR)
        Jv = RHO * (VEL_D / av + 0.5 * dx / aiv) / dt
        Rpv = 8.0 * math.pi * MU * VEL_D / (av * av)
        uv0 = TUv[n]
        uv1 = TUv[n + 1]
        Rkv = VEL_KE * 0.5 * RHO * abs(uv0) / (av * av)
        Hv = Jv + Rpv + Rkv
        mUv += lUv * Jv / Hv
        mp[iv] += lUv / Hv
        mpN[0] -= lUv / Hv
        lJv = lUv * (uv0 - uv1) / Hv
        lRv_ = -lUv * uv1 / Hv
        mUv += lRv_ * VEL_KE * 0.5 * RHO * (1.0 if uv0 > 0.0 else (-1.0 if uv0 < 0.0 else 0.0)) / (av * av)
        if Av[n] > A_FLOOR:
            gAv[n] += lJv * (-RHO * VEL_D / (av * av * dt)) + lRv_ * (-2.0 * (Rpv + Rkv) / av)
        if A[n, iv] > A_FLOOR:
            gA[n, iv] += lJv * (-RHO * 0.5 * dx / (aiv * aiv * dt))
        ldx += lJv * RHO * 0.5 / (aiv * dt)
        # ---------------- 구강 이음매 (역)
        lcj = 0.0
        for i in range(N - 1):
            a1 = max(A[n, i], A_FLOOR)
            a2 = max(A[n, i + 1], A_FLOOR)
            J = 0.5 * RHO * dx * (1.0 / a1 + 1.0 / a2) / dt
            ah = 2.0 * a1 * a2 / (a1 + a2)
            rf = (2.0 * math.sqrt(math.pi * ah) * dx / (ah * ah)) * Kv if loss_on else 0.0
            r0 = 8.0 * math.pi * MU * dx / (ah * ah) if loss_on else 0.0
            u1 = TU[n + 1, i]
            u0 = TU[n, i]
            # b U + k U|U| = c 의 음함수 미분: (b + 2k|U|) dU = dc − U db − U|U| dk
            dk = 1.0 / a1 - 1.0 / a2
            kk = 0.5 * RHO * dk * dk if (u1 >= 0.0 and dk > 0.0) or (u1 < 0.0 and dk < 0.0) else 0.0
            lu = lU[i]
            lc = lu / (J + r0 + rf * W + 2.0 * kk * abs(u1))
            mU[i] += lc * J
            mp[i] += lc
            mp[i + 1] -= lc
            for m in range(M):
                mphv[i, m] += lc * rf * XW[m]
            lpn = lc
            if GLOT_NOISE_SUPRA and i == GLOT_NOISE_J:
                lcj = lc
            lb = -lc * u1
            lJ = lc * u0 + lb
            lrf = lc * TFv[n, i] + lb * W
            lr0 = lb
            la1 = lJ * (-0.5 * RHO * dx / (a1 * a1 * dt))
            la2 = lJ * (-0.5 * RHO * dx / (a2 * a2 * dt))
            if kk > 0.0:
                lkk = -lc * u1 * abs(u1)
                la1 += lkk * RHO * dk * (-1.0 / (a1 * a1))
                la2 += lkk * RHO * dk * (1.0 / (a2 * a2))
            ldx += lJ * J / dx
            if loss_on:
                lah = lrf * (-1.5 * rf / ah) + lr0 * (-2.0 * r0 / ah)
                ldx += lrf * rf / dx + lr0 * r0 / dx
                s_ = a1 + a2
                la1 += lah * 2.0 * a2 * a2 / (s_ * s_)
                la2 += lah * 2.0 * a1 * a1 / (s_ * s_)
            us = TUs[n + 1, i]
            amin = min(a1, a2)
            lus = lUs[i]
            Re = abs(us) * 2.0 / (math.sqrt(math.pi * amin) * NU)
            if JET_BC:
                jet = (us > 0.0 and a2 > a1) or (us < 0.0 and a1 > a2)
            else:
                jet = (us > 0.0 and a2 > 1.2 * a1) or (us < 0.0 and a1 > 1.2 * a2)
            if TURB_SMOOTH and JET_BC:
                aup = a1 if us >= 0.0 else a2
                Re = abs(us) * 2.0 / (math.sqrt(math.pi * aup) * NU)
                hg = 0.0
                dhg = 0.0
                if Re > 0.0:
                    hg, dhg = _spos(1.0 - (RE_CRIT / Re) ** 2, TURB_GATE_W)
                if hg > 0.0:
                    v = us / aup
                    gob = 1.0 + OBS_GAIN if (i + 1) >= OBS_FRAC * N else 1.0
                    av = abs(v)
                    kq = gob / (V_REF * math.sqrt(A_REF))
                    rv = RHO * av * av
                    ps2 = 2.0 * abs(Ps[n])
                    dyn, ddr, ddp = _smin(rv, ps2, TURB_CAP_W)
                    S = av * math.sqrt(aup) * kq
                    q = dyn * S
                    amx, jdx, j0, jdr, jns = _jet_far(A, n, i, us, a1, a2, N, dx, JG)
                    rj = aup / amx
                    if rj < 1.0:
                        ej = (1.0 - rj) * (1.0 - rj)
                        x = xi_c[n, i]
                        gng[1] += lpn * q * ej * hg * x
                        L0 = lpn * noise_c * x
                        lq = L0 * ej * hg
                        lej = L0 * q * hg
                        lhg = L0 * q * ej
                        lS = lq * dyn
                        ldyn = lq * S
                        lv = lS * (1.0 if v >= 0.0 else -1.0) * math.sqrt(aup) * kq
                        laup = lS * av * kq * 0.5 / math.sqrt(aup)
                        lv += ldyn * ddr * 2.0 * RHO * v
                        gPs[n] += ldyn * ddp * 2.0 * (1.0 if Ps[n] >= 0.0 else -1.0)
                        lRe = lhg * dhg * 2.0 * RE_CRIT * RE_CRIT / (Re * Re * Re)
                        lus += lRe * Re / us + lv / aup
                        laup += lRe * (-0.5 * Re / aup) + lv * (-v / aup)
                        lrj = lej * (-2.0 * (1.0 - rj))
                        laup += lrj / amx
                        lamx = lrj * (-rj / amx)
                        if us >= 0.0:
                            la1 += laup
                        else:
                            la2 += laup
                        for s_ in range(jns):
                            j = j0 + s_ * jdr
                            gj = lamx * JG[s_]
                            if j == i:
                                la1 += gj
                            elif j == i + 1:
                                la2 += gj
                            elif A[n, j] > A_FLOOR:
                                gA[n, j] += gj
                        ldx += lamx * jdx
            elif Re > RE_CRIT and jet:
                v = us / amin
                gob = 1.0 + OBS_GAIN if (i + 1) >= OBS_FRAC * N else 1.0
                av = abs(v)
                kq = gob / (V_REF * math.sqrt(A_REF))
                rv = RHO * av * av
                ps2 = 2.0 * abs(Ps[n])
                dyn = min(rv, ps2)
                S = av * math.sqrt(amin) * kq
                q = dyn * S
                ej = 1.0
                amx = max(a1, a2)
                jdx = 0.0
                j0 = i + 1 if a2 >= a1 else i
                jdr = 1
                jns = 0
                if JET_BC:
                    amx, jdx, j0, jdr, jns = _jet_far(A, n, i, us, a1, a2, N, dx, JG)
                rj = amin / amx
                if JET_BC:
                    ej = (1.0 - rj) * (1.0 - rj)          # 전단 동압 몫 (1 − A₁/A₂)², A₂ = 제트 발달 거리 안의 가장 넓은 칸
                h = 1.0 - (RE_CRIT / Re) ** 2
                x = xi_c[n, i]
                gng[1] += lpn * q * ej * h * x
                lqe = lpn * noise_c * h * x
                lh = lpn * noise_c * q * ej * x
                lq = lqe * ej
                lej = lqe * q
                # q = 동압 × S,  S = |v| √A k
                lS = lq * dyn
                lv = lS * (1.0 if v >= 0.0 else -1.0) * math.sqrt(amin) * kq
                lam_q = lS * av * kq * 0.5 / math.sqrt(amin)
                if rv < ps2:
                    lv += lq * S * 2.0 * RHO * v                                   # d(ρv²)/dv
                else:
                    gPs[n] += lq * S * 2.0 * (1.0 if Ps[n] >= 0.0 else -1.0)
                lRe = lh * 2.0 * RE_CRIT * RE_CRIT / (Re * Re * Re)
                lus += lRe * Re / us + lv / amin
                lamin = lRe * (-0.5 * Re / amin) + lv * (-v / amin) + lam_q
                lamx = 0.0
                if JET_BC:
                    # ej = (1 − amin/amx)²: ∂/∂amin = −2(1 − r)/amx, ∂/∂amx = 2(1 − r) amin/amx²
                    lamin += lej * (-2.0 * (1.0 - rj) / amx)
                    lamx = lej * (2.0 * (1.0 - rj) * amin / (amx * amx))
                if a1 <= a2:
                    la1 += lamin
                else:
                    la2 += lamin
                if JET_BC:                                   # 매끈한 넓어진 면적의 몫을 창 안 칸마다 (§52.523)
                    for s_ in range(jns):
                        j = j0 + s_ * jdr
                        gj = lamx * JG[s_]
                        if j == i:
                            la1 += gj
                        elif j == i + 1:
                            la2 += gj
                        elif A[n, j] > A_FLOOR:
                            gA[n, j] += gj
                    ldx += lamx * jdx
            mUs[i] += lus * (1.0 - kc)
            mU[i] += lus * kc
            if A[n, i] > A_FLOOR:
                gA[n, i] += la1
            if A[n, i + 1] > A_FLOOR:
                gA[n, i + 1] += la2
        # ---------------- 기관 이음매 (역)
        for j in range(NT):
            b2 = TRA[j]
            if j == 0:
                Jt = RHO_T * 0.5 * dxt / b2 / dt
                ah = b2
                rl = RHO_T * C_T / b2
            else:
                b1 = TRA[j - 1]
                Jt = 0.5 * RHO_T * dxt * (1.0 / b1 + 1.0 / b2) / dt
                ah = 2.0 * b1 * b2 / (b1 + b2)
                rl = 0.0
            rft = (2.0 * math.sqrt(math.pi * ah) * dxt / (ah * ah)) * KvT if loss_on else 0.0
            r0t = 8.0 * math.pi * MU_T * dxt / (ah * ah) if loss_on else 0.0
            Ht = Jt + rl + r0t + rft * W
            lu = lUT[j]
            mUT[j] += lu * Jt / Ht
            for m in range(M):
                mphvT[j, m] += lu * rft * XW[m] / Ht
            if j == 0:
                gPs[n] += lu / Ht
                mpT[0] -= lu / Ht
            else:
                mpT[j - 1] += lu / Ht
                mpT[j] -= lu / Ht
        # ---------------- 성문 (역)
        lga1 = 0.0
        lga2 = 0.0
        ld1 = 0.0
        ld2 = 0.0
        lLc = 0.0
        lQn = 0.0
        if VF == 3:
            Qn = VQ[n]
            Lc = VR[n, 10]
            dys = VR[n, 11] / NS3
            ch = max(Ag[n], 0.0)
            a0 = max(A[n, 0], A_FLOOR)
            ag = 1.0e30
            kmin3 = 0
            agm = 1.0
            p3 = 1.0
            sLg = 0.0
            sRv = 0.0
            for k in range(NS3):
                rk = VR[n, 0] + (VR[n, 1] - VR[n, 0]) * (k + 0.5) / NS3
                xk3 = 0.5 * (TX[n + 1, k] + TX[n + 1, NS3 + 1 + k]) if LR3 else TX[n + 1, k]
                ak, ok, s2k, cfk, dak, csk = _ns_geom(xk3, rk, Lc, VS[18], TPH[n + 1])
                gk = max(ak + ch, A_FLOOR)
                AK3[k] = ak
                G3[k] = gk
                O3[k] = ok
                S3[k] = s2k
                C3[k] = cfk
                DA3[k] = dak
                CS3[k] = csk
                if gk < ag:
                    ag = gk
                    kmin3 = k
                sLg += dys / gk
                sRv += dys / (gk * gk * gk)
            agm = ag
            p3 = 1.0
            if VF_SEP_P > 0.0:
                p3 = VF_SEP_P * math.log(NS3) / math.log(2.0)
                s3 = 0.0
                for k in range(NS3):
                    s3 += (agm / G3[k]) ** p3
                ag = agm * s3 ** (-1.0 / p3)
            for k in range(NS3 + 1):
                lG3[k] = 0.0
                lO3[k] = 0.0
                lS3[k] = 0.0
                lC3[k] = 0.0
            for k in range(D3):
                lx3[k] = 0.0
                lxp3[k] = 0.0
            Lg = RHO * (sLg + 0.5 * dx / a0)
            Rv = 12.0 * MU * Lc * Lc * sRv
            # ---- 모드 3 운동 (역): μ = A⁻¹ λ
            p0 = Tp[n, 0]
            ps = TpT[n, NT - 1] if NT > 0 else Ps[n]
            if LR3:
                _ns_mats_lr(n, VR, VS, VQ, S3, NS3, Km3, Cm3, Mv3)
            else:
                _ns_mats(n, VR, VS, VQ, S3, NS3, Km3, Cm3, Mv3)
            T2 = dt * dt
            for k in range(D3):
                for j in range(D3):
                    Am3[k, j] = Cm3[k, j] * dt + Km3[k, j] * T2
                Am3[k, k] += Mv3[k]
            _spd_solve(Am3, lX3, MU3)
            if LR3:
                lQn += _ns_adj_lr(n, VR, VS, VQ, S3, NS3, MU3, TX[n + 2], TX[n + 1], TX[n], dt, gVR, gVS, lS3, lx3, lxp3)
                # 두 성대가 같은 힘을 받는다 — 힘의 수반은 두 블록 μ 의 합
                for k in range(NS3 + 1):
                    MU3[k] += MU3[NS3 + 1 + k]
            else:
                lQn += _ns_adj(n, VR, VS, VQ, S3, NS3, MU3, TX[n + 2], TX[n + 1], TX[n], dt, gVR, gVS, lS3, lx3, lxp3)
            # 힘: F_k = p_k dys O_k − Kcol dys C_k Q (+ 입구·출구), F_b = fB
            gVR[n, 9] += T2 * MU3[NS3]
            lps_ = 0.0
            lp0_ = 0.0
            lag3 = 0.0
            ldys3 = 0.0
            c2p = 2.0 / math.pi
            # 분리 몫 χ_k 를 다시 세운다 (앞으로). 예전 꼴: 1 − max_{j≤k} w_j (달리는 최대와 그 자리), 매끈한 꼴: Π_{j≤k} (1 − w_j) (§52.523)
            smooth3 = VF_NS_SMOOTH and VF_SEP_P > 0.0
            if smooth3:
                agm = ag
            mw = 0.0
            jm = -1
            chi_ = 1.0
            for k in range(NS3):
                zc = min(max(1.0 - (G3[k] - agm) / (VF_NS_SEP * agm), 0.0), 1.0)
                if VF_NS_SMOOTH:
                    wj = zc * zc * (3.0 - 2.0 * zc)
                    ZC3[k] = zc
                else:
                    wj = zc
                W3[k] = wj
                if wj > mw:
                    mw = wj
                    jm = k
                chi_ *= 1.0 - wj
                MW3[k] = (1.0 - chi_) if VF_NS_SMOOTH else mw
                JM3[k] = jm
            for k in range(NS3):
                lF = T2 * MU3[k]
                rr = agm / G3[k]
                chi = 1.0 - MW3[k]
                pk_ = p0 + (ps - p0) * (1.0 - rr * rr) * chi
                lpk = lF * dys * O3[k]
                ldys3 += lF * (pk_ * O3[k] - VS[14] * C3[k] * Qn)
                lO3[k] += lF * pk_ * dys
                lC3[k] -= lF * VS[14] * dys * Qn
                gVS[14] -= lF * dys * C3[k] * Qn
                lQn -= lF * VS[14] * dys * C3[k]
                if k == 0:
                    lpk += lF * 0.25 * VS[12] * Lc * c2p
                    lps_ += lF * 0.25 * VS[12] * Lc * c2p
                    gVS[12] += lF * 0.25 * (ps + pk_) * Lc * c2p
                    lLc += lF * 0.25 * (ps + pk_) * VS[12] * c2p
                if k == NS3 - 1:
                    lp0_ += lF * 0.5 * VS[13] * Lc * c2p
                    gVS[13] += lF * p0 * 0.5 * Lc * c2p
                    lLc += lF * p0 * 0.5 * VS[13] * c2p
                bq = (1.0 - rr * rr) * chi
                lps_ += lpk * bq
                lp0_ += lpk * (1.0 - bq)
                lrr = -lpk * (ps - p0) * 2.0 * rr * chi
                lag3 += lrr / G3[k]
                lG3[k] += -lrr * rr / G3[k]
                if VF_NS_SMOOTH:
                    # χ_k = Π_{j≤k} (1 − w_j), w_j = smoothstep(z_j), z_j = 1 − (G_j − ag)/(δ ag)
                    lchi = lpk * (ps - p0) * (1.0 - rr * rr)
                    for j in range(k + 1):
                        zc = ZC3[j]
                        if zc <= 0.0 or zc >= 1.0:
                            continue
                        pr = 1.0
                        for i2 in range(k + 1):
                            if i2 != j:
                                pr *= 1.0 - W3[i2]
                        lw = -lchi * pr * 6.0 * zc * (1.0 - zc)
                        lG3[j] += lw * (-1.0 / (VF_NS_SEP * agm))
                        lag3 += lw * G3[j] / (VF_NS_SEP * agm * agm)
                else:
                    # χ_k = 1 − w_{jm(k)}: w_j = 1 − (G_j − ag)/(δ ag) 가 경사 안일 때만
                    j = JM3[k]
                    if j >= 0 and W3[j] > 0.0 and W3[j] < 1.0:
                        lw = -lpk * (ps - p0) * (1.0 - rr * rr)
                        lG3[j] += lw * (-1.0 / (VF_NS_SEP * agm))
                        lag3 += lw * G3[j] / (VF_NS_SEP * agm * agm)
            if smooth3:                                     # 분리 기준 = 매끈한 최소: ∂ag/∂G_k = (ag/G_k)^(p+1)
                for k in range(NS3):
                    lG3[k] += lag3 * (agm / G3[k]) ** (p3 + 1.0)
            else:
                lG3[kmin3] += lag3
            gVR[n, 11] += ldys3 / NS3
            if NT > 0:
                mpT[NT - 1] += lps_
            else:
                gPs[n] += lps_
            mp[0] += lp0_
        elif VF:
            Qn = VQ[n]
            sq = math.sqrt(Qn)
            Lc = VS[0] * sq
            d1 = VS[1] / sq
            d2 = VS[2] / sq
            ch = max(Ag[n], 0.0)
            x1 = TX[n + 1, 0]
            x2 = TX[n + 1, 1]
            m1a, o1, al1, rs1 = _vf_geom(x1, VR[n, 0], Lc)
            m2a, o2, al2, rs2 = _vf_geom(x2, VR[n, 1], Lc)
            ga1 = max(m1a + ch, A_FLOOR)
            ga2 = max(m2a + ch, A_FLOOR)
            if VF_SEP_P > 0.0:
                ag = (ga1 ** -VF_SEP_P + ga2 ** -VF_SEP_P) ** (-1.0 / VF_SEP_P)
            else:
                ag = min(ga1, ga2)
            a0 = max(A[n, 0], A_FLOOR)
            Lg = RHO * (d1 / ga1 + d2 / ga2 + 0.5 * dx / a0)
            Rv = 12.0 * MU * Lc * Lc * (d1 / (ga1 * ga1 * ga1) + d2 / (ga2 * ga2 * ga2))
            # ---- 성대 운동 (역): [A11 B; B A22] x' = E → μ = M⁻¹ λ, ∂/∂E = μ, ∂/∂M = −μ x'ᵀ
            p0 = Tp[n, 0]
            ps = TpT[n, NT - 1] if NT > 0 else Ps[n]
            rr = 0.0
            if ga1 > ga2:
                rr = ga2 / ga1
                p1 = ps - (ps - p0) * rr * rr
            else:
                p1 = p0
            m1 = VS[3] / Qn
            m2 = VS[4] / Qn
            k1 = VS[7] * Qn
            k2 = VS[8] * Qn
            kc1 = VS[9] * Qn
            kc2 = VS[10] * Qn
            kp = VS[11]
            s1 = 2.0 * math.sqrt(VS[3] * VS[7])
            s2 = 2.0 * math.sqrt(VS[4] * VS[8])
            R1 = (VS[5] + al1) * s1
            R2 = (VS[6] + al2) * s2
            T2 = dt * dt
            A11 = m1 + R1 * dt + T2 * (k1 + kc1 * al1 + kp)
            A22 = m2 + R2 * dt + T2 * (k2 + kc2 * al2 + kp)
            Bc = -kp * T2
            xn1 = TX[n + 2, 0]
            xn2 = TX[n + 2, 1]
            x1p = TX[n, 0]
            x2p = TX[n, 1]
            lxb = 0.0
            lxbp = 0.0
            if VF == 2:
                # 몸체–덮개 (역, §52.500): μ = M⁻¹ λ (M 대칭), ∂M = −μ x'ᵀ. E 는 x_n·x_{n−1}·F·접촉·3 차 항에, M 은 m·R·k 에.
                xb = TX[n + 1, 2]
                xbp = TX[n, 2]
                xnb = TX[n + 2, 2]
                mb = VS[14] / Qn
                kb = VS[15] * Qn * VR[n, 2]
                sb = 2.0 * math.sqrt(VS[14] * VS[15])
                Rb = VS[16] * sb
                eta = VS[17]
                etab = VS[18]
                D1 = x1 - xb
                D2 = x2 - xb
                Ab = mb + Rb * dt + (R1 + R2) * dt + T2 * (kb + k1 + k2)
                B1b = -(k1 * T2 + R1 * dt)
                B2b = -(k2 * T2 + R2 * dt)
                mu1, mu2, mub = _sym3_solve(A11, A22, Ab, Bc, B1b, B2b, lX1, lX2, lXb)
                lA11 = -mu1 * xn1
                lA22 = -mu2 * xn2
                lAb = -mub * xnb
                lBc = -(mu1 * xn2 + mu2 * xn1)
                lB1b = -(mu1 * xnb + mub * xn1)
                lB2b = -(mu2 * xnb + mub * xn2)
                lF1 = mu1 * T2
                lF2 = mu2 * T2
                dm1 = mu1 - mub
                dm2 = mu2 - mub
                lD1 = dm1 * (R1 * dt - 3.0 * T2 * k1 * eta * D1 * D1)
                lD2 = dm2 * (R2 * dt - 3.0 * T2 * k2 * eta * D2 * D2)
                lx1 = mu1 * 2.0 * m1 + lD1
                lx2 = mu2 * 2.0 * m2 + lD2
                lxb = mub * (2.0 * mb + Rb * dt - 3.0 * T2 * kb * etab * xb * xb) - lD1 - lD2
                lx1p = -mu1 * m1
                lx2p = -mu2 * m2
                lxbp = -mub * mb
                lm1 = mu1 * (2.0 * x1 - x1p) + lA11
                lm2 = mu2 * (2.0 * x2 - x2p) + lA22
                lmb = mub * (2.0 * xb - xbp) + lAb
                lR1 = dm1 * dt * D1 + (lA11 + lAb - lB1b) * dt
                lR2 = dm2 * dt * D2 + (lA22 + lAb - lB2b) * dt
                lRb = mub * dt * xb + lAb * dt
                lk1 = (lA11 + lAb - lB1b) * T2 - dm1 * T2 * eta * D1 * D1 * D1
                lk2 = (lA22 + lAb - lB2b) * T2 - dm2 * T2 * eta * D2 * D2 * D2
                lkb = lAb * T2 - mub * T2 * etab * xb * xb * xb
                lkc1 = -mu1 * T2 * al1 * rs1 + lA11 * T2 * al1
                lkc2 = -mu2 * T2 * al2 * rs2 + lA22 * T2 * al2
                lal1 = -mu1 * T2 * kc1 * rs1 + lA11 * T2 * kc1 + lR1 * s1
                lal2 = -mu2 * T2 * kc2 * rs2 + lA22 * T2 * kc2 + lR2 * s2
                lrs1 = -mu1 * T2 * kc1 * al1
                lrs2 = -mu2 * T2 * kc2 * al2
                lQn += ((-lm1 * m1 - lm2 * m2 - lmb * mb) / Qn + lk1 * VS[7] + lk2 * VS[8] + lkb * VS[15] * VR[n, 2]
                        + lkc1 * VS[9] + lkc2 * VS[10])
                gVS[3] += lm1 / Qn + lR1 * (VS[5] + al1) * s1 / (2.0 * VS[3])
                gVS[4] += lm2 / Qn + lR2 * (VS[6] + al2) * s2 / (2.0 * VS[4])
                gVS[5] += lR1 * s1
                gVS[6] += lR2 * s2
                gVS[7] += lk1 * Qn + lR1 * (VS[5] + al1) * s1 / (2.0 * VS[7])
                gVS[8] += lk2 * Qn + lR2 * (VS[6] + al2) * s2 / (2.0 * VS[8])
                gVS[9] += lkc1 * Qn
                gVS[10] += lkc2 * Qn
                gVS[11] += T2 * (lA11 + lA22) - T2 * lBc
                gVS[14] += lmb / Qn + lRb * VS[16] * sb / (2.0 * VS[14])
                gVS[15] += lkb * Qn * VR[n, 2] + lRb * VS[16] * sb / (2.0 * VS[15])
                gVS[16] += lRb * sb
                gVS[17] += -dm1 * T2 * k1 * D1 * D1 * D1 - dm2 * T2 * k2 * D2 * D2 * D2
                gVS[18] += -mub * T2 * kb * xb * xb * xb
                gVR[n, 2] += lkb * VS[15] * Qn
            else:
                det = A11 * A22 - Bc * Bc
                mu1 = (lX1 * A22 - Bc * lX2) / det
                mu2 = (A11 * lX2 - Bc * lX1) / det
                lA11 = -mu1 * xn1
                lA22 = -mu2 * xn2
                lF1 = mu1 * T2
                lF2 = mu2 * T2
                lx1 = mu1 * (2.0 * m1 + R1 * dt)
                lx2 = mu2 * (2.0 * m2 + R2 * dt)
                lx1p = -mu1 * m1
                lx2p = -mu2 * m2
                lm1 = mu1 * (2.0 * x1 - x1p) + lA11
                lm2 = mu2 * (2.0 * x2 - x2p) + lA22
                lR1 = mu1 * dt * x1 + lA11 * dt
                lR2 = mu2 * dt * x2 + lA22 * dt
                lkc1 = -mu1 * T2 * al1 * rs1 + lA11 * T2 * al1
                lkc2 = -mu2 * T2 * al2 * rs2 + lA22 * T2 * al2
                lal1 = -mu1 * T2 * kc1 * rs1 + lA11 * T2 * kc1 + lR1 * s1
                lal2 = -mu2 * T2 * kc2 * rs2 + lA22 * T2 * kc2 + lR2 * s2
                lrs1 = -mu1 * T2 * kc1 * al1
                lrs2 = -mu2 * T2 * kc2 * al2
                lQn += -lm1 * m1 / Qn - lm2 * m2 / Qn + lA11 * T2 * VS[7] + lA22 * T2 * VS[8] + lkc1 * VS[9] + lkc2 * VS[10]
                # 정적 변수 (화자 상수) — m = VS/Q, k = VS·Q, R = (ζ + α)·2√(m₀k₀), 결합 kp 는 A11·A22·B 에
                lBc = -(mu1 * xn2 + mu2 * xn1)
                gVS[3] += lm1 / Qn + lR1 * (VS[5] + al1) * s1 / (2.0 * VS[3])
                gVS[4] += lm2 / Qn + lR2 * (VS[6] + al2) * s2 / (2.0 * VS[4])
                gVS[5] += lR1 * s1
                gVS[6] += lR2 * s2
                gVS[7] += lA11 * T2 * Qn + lR1 * (VS[5] + al1) * s1 / (2.0 * VS[7])
                gVS[8] += lA22 * T2 * Qn + lR2 * (VS[6] + al2) * s2 / (2.0 * VS[8])
                gVS[9] += lkc1 * Qn
                gVS[10] += lkc2 * Qn
                gVS[11] += T2 * (lA11 + lA22) - T2 * lBc
            # 힘: F1 = p1 o1 d1 + ¼(P_s + p1) 입구 Lc,  F2 = p0 (o2 d2 + ½ 출구 Lc)
            lp1 = lF1 * (o1 * d1 + 0.25 * VS[12] * Lc)
            lo1 = lF1 * p1 * d1
            ld1 += lF1 * p1 * o1
            if NT > 0:
                mpT[NT - 1] += lF1 * 0.25 * VS[12] * Lc
            else:
                gPs[n] += lF1 * 0.25 * VS[12] * Lc
            lLc += lF1 * 0.25 * (ps + p1) * VS[12] + lF2 * p0 * 0.5 * VS[13]
            gVS[12] += lF1 * 0.25 * (ps + p1) * Lc
            gVS[13] += lF2 * p0 * 0.5 * Lc
            lp0 = lF2 * (o2 * d2 + 0.5 * VS[13] * Lc)
            lo2 = lF2 * p0 * d2
            ld2 += lF2 * p0 * o2
            if ga1 > ga2:
                if NT > 0:
                    mpT[NT - 1] += lp1 * (1.0 - rr * rr)
                else:
                    gPs[n] += lp1 * (1.0 - rr * rr)
                lp0 += lp1 * rr * rr
                lrr = -lp1 * (ps - p0) * 2.0 * rr
                lga2 += lrr / ga1
                lga1 += -lrr * rr / ga1
            else:
                lp0 += lp1
            mp[0] += lp0
        else:
            ag = max(Ag[n], A_FLOOR)
            Lg = RHO * GLOTTIS_D / ag
            Rv = 12.0 * MU * GLOTTIS_D * GLOTTIS_LEN * GLOTTIS_LEN / (ag * ag * ag)
        Lg += LgT
        Rk = GLOTTIS_KE * RHO / (2.0 * ag * ag)
        ug0 = TUg[n]
        ug1 = TUg[n + 1]
        b_ = Lg / dt + Rv
        lc = lUg / (2.0 * Rk * abs(ug1) + b_)
        lb = -lc * ug1
        lRk = -lc * ug1 * abs(ug1)
        if NT > 0:
            mpT[NT - 1] += lc
        else:
            gPs[n] += lc
        mp[0] -= lc
        lpng = lcj if GLOT_NOISE_SUPRA else lc
        mUg += lc * Lg / dt
        lLg = lc * ug0 / dt + lb / dt
        lRv = lb
        lag = -lRk * 2.0 * Rk / ag
        if VF == 3:
            s1 = 0.0
            s3 = 0.0
            for k in range(NS3):
                gk = G3[k]
                lG3[k] += -lLg * RHO * dys / (gk * gk) - lRv * 36.0 * MU * Lc * Lc * dys / (gk * gk * gk * gk)
                s1 += 1.0 / gk
                s3 += 1.0 / (gk * gk * gk)
            gVR[n, 11] += (lLg * RHO * s1 + lRv * 12.0 * MU * Lc * Lc * s3) / NS3
            lLc += lRv * 24.0 * MU * Lc * sRv
            ldx += lLg * RHO * 0.5 / a0
            if A[n, 0] > A_FLOOR:
                gA[n, 0] += -lLg * RHO * 0.5 * dx / (a0 * a0)
        elif VF:
            lga1 += -lLg * RHO * d1 / (ga1 * ga1) - lRv * 36.0 * MU * Lc * Lc * d1 / (ga1 * ga1 * ga1 * ga1)
            lga2 += -lLg * RHO * d2 / (ga2 * ga2) - lRv * 36.0 * MU * Lc * Lc * d2 / (ga2 * ga2 * ga2 * ga2)
            ld1 += lLg * RHO / ga1 + lRv * 12.0 * MU * Lc * Lc / (ga1 * ga1 * ga1)
            ld2 += lLg * RHO / ga2 + lRv * 12.0 * MU * Lc * Lc / (ga2 * ga2 * ga2)
            lLc += lRv * 24.0 * MU * Lc * (d1 / (ga1 * ga1 * ga1) + d2 / (ga2 * ga2 * ga2))
            ldx += lLg * RHO * 0.5 / a0
            if A[n, 0] > A_FLOOR:
                gA[n, 0] += -lLg * RHO * 0.5 * dx / (a0 * a0)
        else:
            lag += -lLg * (Lg - LgT) / ag - lRv * 3.0 * Rv / ag
        if GLOT_NOISE_3D:
            ugs = TUgs[n + 1]
            Lre = Lc if (VF and RE_USE_LC) else GLOTTIS_LEN
            x = xi_g[n]
            gng[0] += lpng * ugs * x
            lugs = lUgs + lpng * noise_g * x
            ach = max(Ag[n], 0.0) if VF else 0.0
            q0_, dqu, dqa, dql = _gq_d(ug0, ag, Lre, ach)
            lq = lugs * kg
            mUg += lq * dqu
            lag += lq * dqa
            if VF and RE_USE_LC:
                lLc += lq * dql
            mUgs += lugs * (1.0 - kg)
        else:
            ugs = TUgs[n + 1]
            lugs = lUgs
            Lre = Lc if (VF and RE_USE_LC) else GLOTTIS_LEN
            Re_g = 2.0 * ugs / (Lre * NU)
            hgg = 0.0
            dhgg = 0.0
            if TURB_SMOOTH:
                if Re_g > 0.0:
                    hgg, dhgg = _spos(1.0 - (RE_CRIT / Re_g) ** 2, TURB_GATE_W)
            elif Re_g > RE_CRIT:
                hgg = 1.0 - (RE_CRIT / Re_g) ** 2
                dhgg = 1.0
            if hgg > 0.0:
                vg = ugs / ag
                rv = RHO * vg * vg
                ps2 = 2.0 * abs(Ps[n])
                if TURB_SMOOTH:
                    q0, ddr, ddp = _smin(rv, ps2, TURB_CAP_W)
                else:
                    q0 = min(rv, ps2)
                    ddr = 1.0 if rv < ps2 else 0.0
                    ddp = 0.0 if rv < ps2 else 1.0
                sc = vg * math.sqrt(ag) / (V_REF * math.sqrt(A_REF)) if GLOT_NOISE_STEVENS else 1.0
                rj = GLOT_JET_CORE * ag / (Lre * GLOT_JET_D)
                ej = math.exp(-rj)
                fj = (1.0 - ej) if GLOT_NOISE_JET else 1.0
                q = q0 * sc * fj
                h = hgg
                x = xi_g[n]
                gng[0] += lpng * q * h * x
                lqf = lpng * noise_g * h * x                # ∂/∂q 전체
                lh = lpng * noise_g * q * x
                if GLOT_NOISE_JET:                          # ∂fj/∂A = e·r/A, ∂fj/∂ℓ = −e·r/ℓ (§52.523)
                    lfj = lqf * q0 * sc * ej * rj
                    lag += lfj / ag
                    if VF and RE_USE_LC:
                        lLc += -lfj / Lre
                lq = lqf * fj
                lvg = 0.0
                lags = 0.0
                if GLOT_NOISE_STEVENS:                      # ∂sc/∂v = sc/v, ∂sc/∂A = ½ sc/A
                    lvg += lq * q0 * sc / vg
                    lags = lq * q0 * 0.5 * sc / ag
                lvg += lq * sc * ddr * 2.0 * RHO * vg
                gPs[n] += lq * sc * ddp * 2.0 * (1.0 if Ps[n] >= 0.0 else -1.0)
                lRe = lh * dhgg * 2.0 * RE_CRIT * RE_CRIT / (Re_g * Re_g * Re_g)
                lugs += lRe * 2.0 / (Lre * NU) + lvg / ag
                if VF and RE_USE_LC:
                    lLc += -lRe * Re_g / Lre                # ∂Re/∂ℓ = −Re/ℓ
                lag += lvg * (-vg / ag) + lags
            mUgs += lugs * (1.0 - kg)
            mUg += lugs * kg * (1.0 if ug0 >= 0.0 else -1.0)
        if VF == 3:
            if VF_SEP_P > 0.0:                          # 매끈한 흐름 면적: ∂ag/∂G_k = (ag/G_k)^(p+1)
                for k in range(NS3):
                    lG3[k] += (lag + gGA[n]) * (ag / G3[k]) ** (p3 + 1.0)
            else:
                lG3[kmin3] += lag + gGA[n]
            lch = 0.0
            for k in range(NS3):
                f = (k + 0.5) / NS3
                lrk = 0.0
                lxg = 0.0
                if AK3[k] + ch > A_FLOOR:
                    la = lG3[k]
                    lch += la
                    lxg += la * CS3[k]                      # ∂a/∂x (열린 적분점의 sin 합 — 경사 몫 O 와 다르다)
                    lrk += la * DA3[k]
                    lLc += la * AK3[k] / Lc
                lLc += (lO3[k] * O3[k] + lS3[k] * S3[k] + lC3[k] * C3[k]) / Lc
                rk = VR[n, 0] + (VR[n, 1] - VR[n, 0]) * f
                xk3 = 0.5 * (TX[n + 1, k] + TX[n + 1, NS3 + 1 + k]) if LR3 else TX[n + 1, k]
                gd = _ns_geom_d(xk3, rk, Lc, VS[18], TPH[n + 1])
                lxg += lO3[k] * gd[0] + lS3[k] * gd[2] + lC3[k] * gd[4]
                if LR3:                                     # 평균 반틈새 (x_L + x_R)/2 — 두 성대에 반씩
                    lx3[k] += 0.5 * lxg
                    lx3[NS3 + 1 + k] += 0.5 * lxg
                else:
                    lx3[k] += lxg
                lrk += lO3[k] * gd[1] + lS3[k] * gd[3] + lC3[k] * gd[5]
                gVR[n, 0] += lrk * (1.0 - f)
                gVR[n, 1] += lrk * f
            if Ag[n] > 0.0:
                gAg[n] += lch
            gVR[n, 10] += lLc
            gVQ[n] += lQn
            # 잘린 역전파: 성대 상태의 수반을 시정수 VS[17] ms 로 잊는다. 떨림이 유성 시작마다 문턱 근처에서 새로 자라 그 시각이 매개변수에
            # 가파르게 반응하고, 정확한 수반은 그 긴 시간의 민감도를 모두 더해 실제 적합 손실의 방향 차분보다 수십 배 컸다 (§52.504).
            dec3 = math.exp(-1000.0 / (VS[17] * fs)) if VS[17] > 0.0 else 1.0
            for k in range(D3):
                lX3[k] = (pX3[k] + lx3[k]) * dec3
                pX3[k] = lxp3[k] * dec3
        elif VF:
            if VF_SEP_P > 0.0:                          # 매끈한 최소: ∂ag/∂a_i = (ag/a_i)^(p+1)
                lga1 += (lag + gGA[n]) * (ag / ga1) ** (VF_SEP_P + 1.0)
                lga2 += (lag + gGA[n]) * (ag / ga2) ** (VF_SEP_P + 1.0)
            elif ga1 <= ga2:                            # 흐름 면적 = min — 성문 면적 출력의 기울기 gGA 도 여기로 (§52.491)
                lga1 += lag + gGA[n]
            else:
                lga2 += lag + gGA[n]
            lch = 0.0
            lm1a = 0.0
            lm2a = 0.0
            if m1a + ch > A_FLOOR:
                lm1a = lga1
                lch += lga1
            if m2a + ch > A_FLOOR:
                lm2a = lga2
                lch += lga2
            if Ag[n] > 0.0:
                gAg[n] += lch
            q1 = _vf_geom_d(x1, VR[n, 0], Lc)
            q2 = _vf_geom_d(x2, VR[n, 1], Lc)
            lx1 += lm1a * q1[0] + lo1 * q1[3] + lal1 * q1[6] + lrs1 * q1[9]
            gVR[n, 0] += lm1a * q1[1] + lo1 * q1[4] + lal1 * q1[7] + lrs1 * q1[10]
            lLc += lm1a * q1[2] + lo1 * q1[5] + lal1 * q1[8] + lrs1 * q1[11]
            lx2 += lm2a * q2[0] + lo2 * q2[3] + lal2 * q2[6] + lrs2 * q2[9]
            gVR[n, 1] += lm2a * q2[1] + lo2 * q2[4] + lal2 * q2[7] + lrs2 * q2[10]
            lLc += lm2a * q2[2] + lo2 * q2[5] + lal2 * q2[8] + lrs2 * q2[11]
            lQn += lLc * 0.5 * Lc / Qn - 0.5 * (ld1 * d1 + ld2 * d2) / Qn
            gVQ[n] += lQn
            gVS[0] += lLc * sq
            gVS[1] += ld1 / sq
            gVS[2] += ld2 / sq
            lX1 = pX1 + lx1
            lX2 = pX2 + lx2
            pX1 = lx1p
            pX2 = lx2p
            if VF == 2:
                lXb = pXb + lxb
                pXb = lxbp
        elif Ag[n] > A_FLOOR:
            gAg[n] += lag + gGA[n]
        gL[n] += ldx / N
        lp = mp
        lU = mU
        lvw = mvw
        lyw = myw
        lUs = mUs
        lpN = mpN
        lUN = mUN
        lpP = mpP
        lUP = mUP
        lUg = mUg
        lUm = mUm
        lUL = mUL
        lUgs = mUgs
        lUv = mUv
        lUmN = mUmN
        lULN = mULN
        lphv = mphv
        lpht = mpht
        lphvN = mphvN
        lphtN = mphtN
        lpT = mpT
        lUT = mUT
        lphvT = mphvT
        lphtT = mphtT
        lphvP = mphvP
        lphtP = mphtP


_fwd_jit = njit(cache=True, fastmath=False)(_fwd) if njit is not None else None
_bwd_jit = njit(cache=True, fastmath=False)(_bwd) if njit is not None else None


def _tapes(T, N, NN=None, NP=None, NT=0):
    NN = N_NAS if NN is None else NN
    NP = N_PIR * len(SIDE_BRANCHES if SIDE_BRANCHES is not None else [0]) if NP is None else NP
    z = np.zeros
    return (z((T + 1, N)), z((T + 1, N - 1)), z(T + 1), z(T + 1), z(T + 1), z(T + 1), z((T + 1, N)), z((T + 1, N)),
            z((T + 1, N - 1)), z(T + 1), z((T + 1, NN)), z((T + 1, NN - 1)), z(T + 1), z(T + 1), z(T + 1), z(T + 1),
            z((T + 1, NP)), z((T + 1, NP)),
            z((T + 1, N - 1)), z((T + 1, N)), z((T + 1, NN - 1)), z((T + 1, NN)), z((T + 1, NP)), z((T + 1, NP)),
            z((T + 2, _ns_width())), z((T + 1, NT)), z((T + 1, NT)), z((T + 1, NT)), z((T + 1, NT)),
            z((T + 2, VF_NS_PTS)))


def _ns_width() -> int:
    """모드 3 상태 테이프 폭 — 좌우 판 (`beam_membrane.LR_FOLDS`, §52.533) 이면 2 (줄 + 몸체), 아니면 길이 모드 × 줄 + 몸체."""
    try:
        from ..physics import beam_membrane as _bm
        if getattr(_bm, "LR_FOLDS", False):
            return 2 * (VF_NSTRIP + 1)
    except Exception:
        pass
    return max(3, _ns_lmodes() * VF_NSTRIP + 1)


def _ns_lmodes() -> int:
    """모드 3 의 길이 모드 수 (`beam_membrane.LONG_MODES`, §52.531) — 테이프 폭과 정적 배열 VS[19] 가 같은 값을 쓴다."""
    try:
        from ..physics import beam_membrane as _bm
        return int(getattr(_bm, "LONG_MODES", 1))
    except Exception:
        return 1


#: 난류 음원의 스펙트럼 모양 (§52.481) — 저역통과 (차수, 모서리). 백색이면 방사(+6 dB/oct)를 거쳐 16–20 kHz 로 갈수록 커져 원본 ㅅ
#: (10–12 kHz 봉우리, 14 kHz 위 급락)과 반대 모양이 되고, 적합기가 난류를 눌러 ㅅ 을 유성으로 흉내 냈다. 난류 음원은 고역에서 떨어진다.
NOISE_LP_HZ = 12000.0
NOISE_LP_ORDER = 4
#: **성문 난류의 스트루할 봉우리** (§52.531). 제트의 난류 음원은 f ≈ St · v / h (St ≈ 0.2, v 제트 속도, h 제트 두께) 근처에 넓은 봉우리를 갖는다
#: (Stevens 1971; 소용돌이 떨어져 나감). 이 화자의 뒤틈 폭 0.043 cm² / 0.35 cm = 1.2 mm, 폐압 차 7 cmH2O 의 v = √(2Δp/ρ) ≈ 35 m/s → 5.6 kHz (막부가 열릴 때
#: 1–1.5 mm → 4.6–7 kHz). 예전에는 12 kHz 까지 평평한 잡음이었다 — 잡음을 올리면 10–16 kHz 가 넘치고(모음 +8, 비음 +16 dB), 올리지 않으면 모음 6–10 kHz 가
#: −7 dB (KWF_m20). 봉우리 5.6 kHz · Q 0.5 (−3 dB 2.3–13.7 kHz) 대역통과를 곱하면 (재렌더, 잡음 ×3–4) 모음 6–16 kHz 가 원본과 −1 ~ −6 dB 안, 5–8 kHz 줄
#: 0.07 (원본 0.05). 0 이면 끈다. 세기는 성문 잡음 이득(적합)이 정한다.
GLOT_NOISE_FP_HZ = 5600.0
GLOT_NOISE_Q = 0.5


#: **조음 자리별 난류 음원 스펙트럼** (§52.503, VTL `TdsModel::calcNoiseSources`, Birkholz). 이음매 i (성문에서 (i+1)/N × `NOISE_L_NOM` cm) 와
#: 앞니(입술 끝에서 `TEETH_FROM_LIPS_CM` 안쪽) 사이 거리 d 로: 0 ≤ d < 2.5 cm 이면 제트가 앞니에 부딪힌다 (ㅅ·ㅈ·ㅊ·ㄷ) — 차단 6 kHz·e^(−d/3),
#: 세기 e^(−d/3); d ≥ 2.5 cm 이면 벽 음원 (ㄱ·인두) — 1.5 kHz, 0.5; 앞니 앞(입술) 은 양순 파열 — 0.5 kHz, 0.2. 모두 2 차 저역통과, 낮은 주파수의
#: 밀도가 같게 (VTL 의 "1 kHz 대역폭 당 세기"). 하나의 12 kHz 4 차 잡음을 모든 이음매에 쓰면 ㄲ 파열이 17 kHz 까지 평평해 6 kHz 위가 원본보다
#: 20–30 dB 넘쳤다 (원본 파열은 1–2 kHz 에 모인다). 이음매별 세기에 앞쪽 장애물 배율(`OBS_GAIN`, numba 에 굳은 상수)은 나눠 없앤다.
NOISE_PLACE = False
NOISE_L_NOM = 15.0
TEETH_FROM_LIPS_CM = 1.0


def _place_noise_shape(N):
    """이음매별 (차단 [Hz], 세기) — `NOISE_PLACE` 규칙."""
    fc, g = np.zeros(N - 1), np.zeros(N - 1)
    xt = NOISE_L_NOM - TEETH_FROM_LIPS_CM
    for i in range(N - 1):
        d = xt - (i + 1) / N * NOISE_L_NOM
        if d < 0.0:
            fc[i], g[i] = 500.0, 0.2
        elif d < 2.5:
            fc[i], g[i] = 6000.0 * math.exp(-d / 3.0), math.exp(-d / 3.0)
        else:
            fc[i], g[i] = 1500.0, 0.5
        if (i + 1) >= OBS_FRAC * N:
            g[i] /= (1.0 + OBS_GAIN)
    return fc, g


#: 앞니형 → 벽형 전환 [cm] — VTL 의 2.5 cm 문턱을 1 cm 폭의 연속 전환으로 (앞니형의 차단은 그 사이 2.6 → 2.2 kHz, 벽형 1.5 kHz).
TEETH_WALL_D = (2.0, 3.0)


def _place_weights(d, dx):
    """이음매에서 앞니까지 거리 d [cm] (앞니가 하류면 +) → (앞니형 무게, 그 차단 [Hz], 그 세기, 벽형 무게, 입술형 무게) — 모두 d 에 연속 (§52.519).
    앞니형 (VTL: 혀 협착이 앞니 2.5 cm 안) 차단 6 kHz·e^(−d/3) · 세기 e^(−d/3), 벽형 1.5 kHz · 0.5, 입술형 (제트가 앞니보다 하류 — 입술 사이) 0.5 kHz · 0.2.
    입술형 무게는 제트 자리가 앞니를 지나 칸 하나(dx) 하류에 닿을 때 1 이 된다 — 칸 격자(0.54 cm)가 앞니를 칸 사이 어디에 두든 분류가 튀지 않게."""
    wl = np.clip(-d / dx, 0.0, 1.0)
    dp = np.maximum(d, 0.0)
    ww = np.clip((dp - TEETH_WALL_D[0]) / (TEETH_WALL_D[1] - TEETH_WALL_D[0]), 0.0, 1.0)
    return (1.0 - wl) * (1.0 - ww), 6000.0 * np.exp(-dp / 3.0), np.exp(-dp / 3.0), (1.0 - wl) * ww, wl


def _tv_lp2_py(x, fc, fs):
    """시변 2 차 버터워스 저역통과 (쌍선형, 직접형 I) — 차단이 표본마다 바뀐다 (앞니까지 거리가 조음과 함께 움직인다)."""
    y = np.empty_like(x)
    x1 = x2 = y1 = y2 = 0.0
    r2 = math.sqrt(2.0)
    for n in range(x.shape[0]):
        K = math.tan(math.pi * min(fc[n], 0.45 * fs) / fs)
        nm = 1.0 / (1.0 + r2 * K + K * K)
        b0 = K * K * nm
        a1 = 2.0 * (K * K - 1.0) * nm
        a2 = (1.0 - r2 * K + K * K) * nm
        yn = b0 * (x[n] + 2.0 * x1 + x2) - a1 * y1 - a2 * y2
        x2 = x1
        x1 = x[n]
        y2 = y1
        y1 = yn
        y[n] = yn
    return y


_tv_lp2 = njit(cache=True)(_tv_lp2_py) if njit is not None else _tv_lp2_py


def _noise(T, N, seed, teeth=None, L=None):
    """난류 여기 (성문 (T,), 협착 이음매 (T, N−1)). `NOISE_PLACE` 이고 teeth (입술–앞니 거리 [cm], 표본마다) · L (성도 길이 [cm]) 이 오면 이음매마다
    앞니까지의 실제 거리로 음원 모양을 정한다 (§52.519). 없으면 예전 규칙 (앞니 = 입술 끝에서 `TEETH_FROM_LIPS_CM`, 명목 길이 `NOISE_L_NOM`)."""
    rng = np.random.default_rng(seed)
    xg, xc = rng.standard_normal(T), rng.standard_normal((T, N - 1))
    if NOISE_PLACE:
        from scipy.signal import butter, sosfilt
        sos = butter(NOISE_LP_ORDER, NOISE_LP_HZ, "low", fs=FS_SIM, output="sos")
        ref = float(sosfilt(sos, xg).std())                     # 옛 12 kHz 잡음의 크기 — 전체 배율을 옛 판과 견줄 수 있게
        xg = sosfilt(sos, xg) / max(ref, 1e-12)
        if GLOT_NOISE_FP_HZ > 0.0:                              # 성문 난류의 스트루할 봉우리 (§52.531) — 크기는 그대로 (std 1)
            w0 = 2.0 * math.pi * GLOT_NOISE_FP_HZ / FS_SIM
            al = math.sin(w0) / (2.0 * GLOT_NOISE_Q)
            bp = np.array([[al / (1 + al), 0.0, -al / (1 + al), 1.0, -2.0 * math.cos(w0) / (1 + al), (1 - al) / (1 + al)]])
            yg = sosfilt(bp, xg)
            xg = yg / max(float(yg.std()), 1e-12)
        if teeth is not None and L is not None:
            Ls = np.ascontiguousarray(L, dtype=np.float64)[:T]
            D = np.ascontiguousarray(teeth, dtype=np.float64)[:T]
            dx = Ls / N
            sw = butter(2, 1500.0, "low", fs=FS_SIM, output="sos")
            sl = butter(2, 500.0, "low", fs=FS_SIM, output="sos")
            for i in range(N - 1):
                w = np.ascontiguousarray(xc[:, i])
                wt, fct, gt, ww, wl = _place_weights((Ls - D) - (i + 1) * dx, dx)
                col = np.zeros(T)
                if wt.max() > 0.0:
                    col += wt * gt * _tv_lp2(w, np.ascontiguousarray(fct), FS_SIM)
                if ww.max() > 0.0:
                    col += ww * 0.5 * sosfilt(sw, w)
                if wl.max() > 0.0:
                    col += wl * 0.2 * sosfilt(sl, w)
                if (i + 1) >= OBS_FRAC * N:
                    col /= (1.0 + OBS_GAIN)
                xc[:, i] = col / max(ref, 1e-12)
            return xg, xc
        fc, g = _place_noise_shape(N)
        for i in range(N - 1):
            s2 = butter(2, fc[i], "low", fs=FS_SIM, output="sos")
            xc[:, i] = sosfilt(s2, xc[:, i]) * g[i] / max(ref, 1e-12)
        return xg, xc
    if NOISE_LP_HZ > 0.0:
        from scipy.signal import butter, sosfilt
        sos = butter(NOISE_LP_ORDER, NOISE_LP_HZ, "low", fs=FS_SIM, output="sos")
        xg = sosfilt(sos, xg); xc = sosfilt(sos, xc, axis=0)
        xg /= max(float(xg.std()), 1e-12); xc /= max(float(xc.std()), 1e-12)
    return xg, xc


def simulate(A: np.ndarray, L: np.ndarray, Ag: np.ndarray, Ps: np.ndarray, Av: np.ndarray | None = None,
             noise_g: float = 0.0, noise_c: float = 0.0, seed: int = 0, walls: bool = True, losses: bool = True,
             fs: float = FS_SIM, Q: np.ndarray | None = None, pir: tuple[float, float] | None = None,
             vf: dict | None = None, tapes: bool = False):
    """numpy 순방향. 입력은 FS_SIM 표본률. (방사 음압 (T,), 진단 (T, 9)). Av 없으면 연구개 닫힘, Q (T,N) 없으면 부피 변화 0.
    pir (면적, 길이) — 없으면 모듈 상수 (PIR_AREA, PIR_LEN), (0, …) 이면 이상와 없음.
    vf — 자기 진동 성대 {"Q": (T,) 긴장, "R": (T, 2) 뒤쪽 쉼 변위 [cm], "static": (14,) (없으면 `VF_STATIC`)}. 그때 Ag 는 뒤쪽 틈 면적이다.
    tapes 이면 (방사 음압, 진단, 두 질량 변위 (T, 2))."""
    SBA, SBL, SBI, SBNA, SBNL = side_branches(np.asarray(A).shape[1], pir)
    _check()
    A = np.ascontiguousarray(A, dtype=np.float64)
    T, N = A.shape
    Av = np.zeros(T) if Av is None else np.ascontiguousarray(Av, dtype=np.float64)
    Q = np.zeros((T, N)) if Q is None else np.ascontiguousarray(Q, dtype=np.float64)
    xi_g, xi_c = _noise(T, N, seed)
    out = np.zeros(T)
    rec = np.zeros((T, 9))
    xa, xw = diffusive(fs)
    VF, VQ, VR, VS = _vf_args(vf, T)
    VQ = VQ.copy()                                  # 고리가 덮어쓴다
    PT = np.zeros(0) if vf is None or vf.get("pll") is None else np.ascontiguousarray(vf["pll"], dtype=np.float64)
    EV = np.full(T // 16 + 2, -1.0)
    tp = _tapes(T, N, NN=NAS_AREA.shape[0], NP=N_PIR * SBA.shape[0], NT=TRACHEA_AREA.shape[0])
    tp[-1][:] = _film_field(T, seed)                # 점액막 두께 흔들림 (§52.531)
    _fwd_jit(A, np.ascontiguousarray(L, dtype=np.float64), np.ascontiguousarray(Ag, dtype=np.float64),
             np.ascontiguousarray(Ps, dtype=np.float64), Av, Q, NAS_AREA, float(NAS_LEN), TRACHEA_AREA, float(TRACHEA_LEN), xi_g, xi_c,
             float(noise_g), float(noise_c),
             int(walls), int(losses), float(fs), SBA, SBL, SBI, SBNA, SBNL, xa, xw, VF, VQ, VR, VS, PT, pll_params(fs), EV, out, rec, *tp)
    simulate.last_ev = EV[EV >= 0]
    simulate.last_q = VQ
    if tapes:
        return out, rec, tp[24][1:T + 1]
    return out, rec


#: 위상 고정 고리 상수 (§52.493). kp·ki 는 주기당 비례·적분 몫 (닫힌 고리 극 |z| ≈ 0.7), umax·vmax 는 log 긴장 배율의 한계,
#: sens = d log Q / d log f0 (f0(Q) 표에서 0.73–1.2, 1 로 둔다), gap_ms 떨림이 끊겼다고 볼 틈, arm 닫힘을 잡을 막부 면적 [cm²].
#: ff 앞먹임 몫 — 다음 목표 주기와 이웃 세 주기 평균의 비 (주기별 떨림을 미리 준다).
#: 한계를 넓게 둔다 — 이 성대의 f0 는 폐압·내전·성도와 함께 크게 움직여서 (VKC 기본 긴장 표가 −178 cents 어긋남), ±0.2 에서는 적합이 폐압을
#: 옮길 때마다 고리가 한계에 붙어 풀렸고 (한계 9–11 %, 추종 90 % 2.1 ms) 적합이 무너졌다 (VLB·VLC 포락 → −5 %). ±0.6 · 적분 ±0.5 면 명령 ±25 %,
#: 폐압 4–14, 숨섞임, 320 Hz 에서 추종 중앙 0.04 · 90 % ≤ 0.11 ms, 한계 0 % (`out/_tmp/vf/pll_regimes.py`). 음높이를 맞추는 후두 긴장 제어다.
#: tau_ms 배율을 건 값이 고리 출력을 따라가는 시정수 — 0 이면 계단.
#: det 닫힘 잡기 — 0 막부 면적, 1 최대 유량 감소율 (dU_g/dt 음의 봉우리; darm 은 그 무장 문턱 [cm³/s²], 봉우리 −1.8e5 – −6.9e5).
#: **기본은 부드러운 고리** (§52.496): 주기마다 긴장을 크게 고치면(kp 0.7 · 앞먹임 · 계단) 원본의 주기별 떨림까지 쫓다가 파형의 주기성이 깨졌다
#: — 같은 제어에서 1–3 kHz 배음/잡음 비 6.0 dB (고리 없이 15.8, 원본 10.4), 포락 60.0 %. 느린 표류만 고치면 (kp 0.08 · ki 0.02 · 앞먹임 0 · τ 4 ms)
#: 15.4 dB · 갈라짐 0.69 · 포락 65.5 %. 후두 긴장은 한 주기 안에서 뛰지 못한다는 물리와도 맞다.
#: kf 주파수 고리 몫 — 잰 닫힘 간격과 목표 주기의 log 비를 (위상이 풀렸을 때) 적분에 더한다. 느린 위상 고리만으로는 기본 긴장이 10 % 넘게
#: 틀리면 끌어들이지 못하지만 (합성 시험 1.4 ms), 실제 적합 상태에서는 불규칙한 주기가 이것을 건드려 적분을 차서 해로웠다 (C 포락 57.0 → 48.7 %,
#: 추종 0.31 → 0.71 ms) — 기본 긴장은 음높이 맞춤(틀 배율)이 f0 표로 맞추므로 끈다.
PLL_GAINS = dict(kp=0.08, ki=0.02, umax=0.6, vmax=0.5, sens=1.0, gap_ms=25.0, arm=0.003, pmin_ms=1.2, pmax_ms=15.0, ff=0.0, tau_ms=4.0,
                 det=0, darm=2.0e4, kf=0.0)


def pll_params(fs: float = FS_SIM) -> np.ndarray:
    g = PLL_GAINS
    return np.array([g["kp"], g["ki"], g["umax"], g["vmax"], g["sens"], g["gap_ms"] * 1e-3 * fs, g["arm"],
                     g["pmin_ms"] * 1e-3 * fs, g["pmax_ms"] * 1e-3 * fs, g["ff"], g.get("tau_ms", 0.0) * 1e-3 * fs,
                     float(g.get("det", 0)), g.get("darm", 2.0e4), g.get("kf", 0.0)], dtype=np.float64)


def vf_static_bc() -> np.ndarray:
    """몸체–덮개 모드(2)의 정적 변수 (§52.500) — VF_STATIC 14 개 + [m_b g, k_b dyn/cm, ζ_b, η 덮개 cm⁻², η 몸체 cm⁻²]. Story & Titze 1995
    정상 발성(경우 C)의 비를 따른다: 몸체 질량 = 덮개 두 질량 × 2.5 (0.05 / 0.02 g), 몸체 강성 ≈ 덮개 스프링 합 × 12 (100 / (3.5 + 5) N/m),
    ζ_b 0.2, η 100 cm⁻²."""
    s = VF_STATIC
    return np.r_[s, [2.5 * (s[3] + s[4]), 12.0 * (s[7] + s[8]), 0.2, 100.0, 100.0]]


def _vf_args(vf, T):
    if vf is None:
        return 0, np.ones(T), np.zeros((T, 3)), vf_static_bc()
    mode = int(vf.get("mode", 1))
    st = (VF_STATIC if mode == 1 else vf_static_bc()) if vf.get("static") is None else vf["static"]
    R = np.ascontiguousarray(vf["R"], dtype=np.float64)
    if R.shape[1] == 2:
        R = np.ascontiguousarray(np.c_[R, np.ones(R.shape[0])])
    return (mode, np.ascontiguousarray(vf["Q"], dtype=np.float64), R, np.ascontiguousarray(st, dtype=np.float64))


try:
    import torch

    class TubeFn(torch.autograd.Function):
        """토치 자동미분 — 순방향·역방향 모두 numba. 입력 (A (T,N), L, Ag, Ps, Av (T,), Q (T,N), noise_g, noise_c, VQ (T,), VR (T,2)),
        출력 out (T,). vf 0 이면 VQ·VR 은 쓰지 않는다 (기울기 0)."""

        @staticmethod
        def forward(ctx, A, L, Ag, Ps, Av, Q, noise_g, noise_c, VQ, VR, VS, xi_g, xi_c, walls, losses, fs, vf, pll=None):
            _check()
            f = lambda x: np.ascontiguousarray(x.detach().double().cpu().numpy())
            a, l_, g_, p_, v_, q_, vq, vr, vs = f(A), f(L), f(Ag), f(Ps), f(Av), f(Q), f(VQ), f(VR), f(VS)
            vq = vq.copy()                          # 고리가 덮어쓴다 — 텐서와 메모리를 나누지 않게
            vq0 = vq.copy()
            PT = np.zeros(0) if pll is None or not vf else np.ascontiguousarray(pll, dtype=np.float64)
            EV = np.full(a.shape[0] // 16 + 2, -1.0)
            T, N = a.shape
            ng, nc = float(noise_g), float(noise_c)
            out = np.zeros(T)
            rec = np.zeros((T, 9))
            pa, pl, pi_, pna, pnl = side_branches(N)
            tapes = _tapes(T, N, NN=NAS_AREA.shape[0], NP=N_PIR * pa.shape[0], NT=TRACHEA_AREA.shape[0])
            tapes[-1][:] = _film_field(T, getattr(TubeFn, "_film_seed", 0))      # 점액막 두께 흔들림 (§52.531)
            xa, xw = diffusive(fs)
            _fwd_jit(a, l_, g_, p_, v_, q_, NAS_AREA, float(NAS_LEN), TRACHEA_AREA, float(TRACHEA_LEN), xi_g, xi_c, ng, nc, int(walls), int(losses), float(fs), pa, pl, pi_, pna, pnl, xa, xw,
                     int(vf), vq, vr, vs, PT, pll_params(fs), EV, out, rec, *tapes)
            ctx.saved = (a, l_, g_, p_, v_, q_, xi_g, xi_c, ng, nc, int(walls), int(losses), float(fs), pa, pl, pi_, pna, pnl, xa, xw,
                         int(vf), vq, vr, vs, tapes)
            ctx.qratio = vq / vq0 if PT.shape[0] >= 2 else None
            import os as _os_tx
            if _os_tx.environ.get("TD_TX_DUMP") and int(vf) == 3:
                # 진단 (§52.532): 성대 줄 변위 테이프 (T+2, 줄 · 길이 모드 + 몸체) · 표본별 성대 입력 — 카이모그램 · 위아래 위상차를 잰다
                np.savez(_os_tx.environ["TD_TX_DUMP"], TX=tapes[24], vr=vr, vs=vs, ga=rec[:, 7], fs=float(fs))
            TubeFn.last_ev = EV[EV >= 0]
            TubeFn.last_q = vq
            ctx.dev, ctx.dt = A.device, A.dtype
            ctx.nas = (NAS_AREA, float(NAS_LEN), TRACHEA_AREA, float(TRACHEA_LEN))
            ctx.rec = rec
            TubeFn.last_rec = rec
            if int(vf) in (1, 2) and VF_SEP_P > 0.0:     # 두 질량: 풀이기의 흐름 면적과 같은 매끈한 최소 (§52.515)
                ga = (np.maximum(rec[:, 7], A_FLOOR) ** -VF_SEP_P + np.maximum(rec[:, 8], A_FLOOR) ** -VF_SEP_P) ** (-1.0 / VF_SEP_P)
            else:
                ga = np.minimum(rec[:, 7], rec[:, 8])
            if not np.isfinite(out).all() or np.abs(out).max() > 1e6:
                import os
                if os.environ.get("TD_NAN_DUMP"):
                    np.savez(os.environ["TD_NAN_DUMP"], a=a, l=l_, g=g_, p=p_, v=v_, q=q_, vq=vq, vr=vr, vs=vs, xi_g=xi_g, xi_c=xi_c,
                             ng=ng, nc=nc, out=out, rec=rec, tx=tapes[24])
            return torch.as_tensor(out, dtype=A.dtype, device=A.device), torch.as_tensor(ga, dtype=A.dtype, device=A.device)

        @staticmethod
        def backward(ctx, gout, gga=None):
            a, l_, g_, p_, v_, q_, xi_g, xi_c, ng, nc, w, lo, fs, pa, pl, pi_, pna, pnl, xa, xw, vf, vq, vr, vs, tapes = ctx.saved
            T, N = a.shape
            gA, gL, gAg, gPs, gAv, gQ, gng = (np.zeros((T, N)), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T),
                                              np.zeros((T, N)), np.zeros(2))
            gVQ, gVR, gVS = np.zeros(T), np.zeros(vr.shape), np.zeros(vs.shape[0])
            _bwd_jit(a, l_, g_, p_, v_, q_, ctx.nas[0], ctx.nas[1], ctx.nas[2], ctx.nas[3], xi_g, xi_c, ng, nc, w, lo, fs, pa, pl, pi_, pna, pnl, xa, xw, vf, vq, vr, vs,
                     np.ascontiguousarray(gout.detach().double().cpu().numpy()),
                     (np.zeros(T) if gga is None else np.ascontiguousarray(gga.detach().double().cpu().numpy())),
                     *tapes, gA, gL, gAg, gPs, gAv, gQ, gng,
                     gVQ, gVR, gVS)
            if ctx.qratio is not None:
                gVQ *= ctx.qratio                   # 고리 배율은 상수로 본다
            t = lambda x: torch.as_tensor(x, dtype=ctx.dt, device=ctx.dev)
            return (t(gA), t(gL), t(gAg), t(gPs), t(gAv), t(gQ), t(gng[0]), t(gng[1]), t(gVQ), t(gVR), t(gVS),
                    None, None, None, None, None, None, None)
except ModuleNotFoundError:          # pragma: no cover
    TubeFn = None


def tube_torch(A, L, Ag, Ps, Av, noise_g, noise_c, seed: int = 0, walls: bool = True, losses: bool = True,
               fs: float = FS_SIM, Q=None, vf=None, return_area: bool = False, pll=None, vf_mode: int = 1, teeth=None):
    """미분 가능한 시간 영역 관 (토치). 입력은 FS_SIM 표본률 텐서, noise_* 는 0 차원 텐서(적합 가능).
    vf — 자기 진동 성대 (VQ (T,) 긴장, VR (T, 2) 뒤쪽 쉼 변위 [cm], 정적 변수 (14,) 텐서(적합 가능)·배열 또는 None) — 그때 Ag 는 뒤쪽 틈 면적.
    teeth — 입술–앞니 거리 [cm] (T,) 배열 (조음 모형이 주면, `NOISE_PLACE` 의 음원 자리, §52.519).
    마지막 풀이의 진단 (T, 9) 은 `TubeFn.last_rec` 에."""
    T, N = A.shape
    xi_g, xi_c = _noise(T, N, seed, teeth, None if teeth is None else L.detach().double().cpu().numpy())
    TubeFn._film_seed = int(seed)                    # 점액막 두께 흔들림의 씨앗 (§52.531)
    if Q is None:
        Q = torch.zeros_like(A)
    if vf is None:
        VQ, VR, vs, on = torch.ones_like(L), torch.zeros(T, 2, dtype=A.dtype, device=A.device), None, 0
    else:
        VQ, VR, vs = vf
        on = int(vf_mode)
    if vs is None:
        vs = torch.as_tensor(VF_STATIC if on <= 1 else vf_static_bc(), dtype=A.dtype, device=A.device)
    elif not torch.is_tensor(vs):
        vs = torch.as_tensor(np.asarray(vs, float), dtype=A.dtype, device=A.device)
    y, ga = TubeFn.apply(A, L, Ag, Ps, Av, Q, noise_g, noise_c, VQ, VR, vs, xi_g, xi_c, walls, losses, fs, on, pll)
    return (y, ga) if return_area else y


# ---------------------------------------------------------------------------------------------------------------------------------------
#: **numba 에 굳는 모듈 상수의 지킴** (§52.519). numba 는 커널을 처음 부를 때 컴파일하며 그 순간의 모듈 상수(JET_BC · NAS_LOSS · GLOT_NOISE_TAU_MS …)를
#: 기계어에 굳히고, 캐시에는 **지금 소스 파일의 도장**을 찍는다. 그래서 (1) 첫 호출 뒤의 `tube_td.X = …` 패치는 아무 효과가 없고, (2) 첫 호출 전의 패치는
#: 그 값으로 컴파일돼 소스 도장으로 캐시를 오염시키며, (3) 다른 판이 도는 중에 소스를 고치면 import 때 값으로 컴파일한 커널이 새 소스의 도장을 받는다 —
#: 지난 판 GT3 · GT6 (성문 난류 평활 3 · 6 ms) 의 출력이 소수점까지 같았고, `NAS_LOSS` 4 → 300 시험이 그대로였다. 첫 호출 때 상수가 import 때와 다르거나
#: 소스가 그사이 바뀌었으면 캐시 없이 새로 컴파일하고, 그 뒤에 상수가 또 바뀌면 멈춘다.
_NB_NAMES: tuple = ()
_NB_SNAP0 = None
_NB_SNAP = None
_NB_SRC_STAMP = None


def _nb_dispatchers() -> dict:
    from numba.core.registry import CPUDispatcher
    return {k: v for k, v in globals().items() if isinstance(v, CPUDispatcher)}


def _nb_values() -> dict:
    g = globals()
    return {n: (g[n].copy() if isinstance(g[n], np.ndarray) else g[n]) for n in _NB_NAMES}


def _nb_changed(a: dict, b: dict) -> list:
    return [n for n in a if not (np.array_equal(a[n], b[n]) if isinstance(a[n], np.ndarray) else a[n] == b[n])]


def _src_stamp():
    import os
    st = os.stat(__file__)
    return (st.st_mtime_ns, st.st_size)


def _nb_rebuild_fresh() -> None:
    """모든 numba 함수를 캐시 없이 다시 만든다 — 커널(`_fwd`·`_bwd`)은 다시 만든 도움 함수를 부르도록 맨 나중에."""
    global _fwd_jit, _bwd_jit
    g = globals()
    for k, d in list(_nb_dispatchers().items()):
        if k not in ("_fwd_jit", "_bwd_jit"):
            g[k] = njit(cache=False)(d.py_func)
    _fwd_jit = njit(cache=False, fastmath=False)(_fwd)
    _bwd_jit = njit(cache=False, fastmath=False)(_bwd)


def _nb_guard() -> None:
    global _NB_SNAP
    if not _NB_NAMES:
        return
    cur = _nb_values()
    if _NB_SNAP is None:
        diff = _nb_changed(cur, _NB_SNAP0)
        stale = _src_stamp() != _NB_SRC_STAMP
        if diff or stale:
            _nb_rebuild_fresh()
            why = ([f"실행 중 바뀐 상수 {', '.join(diff)}"] if diff else []) + (["소스가 import 뒤에 바뀜"] if stale else [])
            print(f"  tube_td: numba 커널을 캐시 없이 새로 컴파일한다 ({' · '.join(why)})", flush=True)
        _NB_SNAP = cur
        return
    diff = _nb_changed(cur, _NB_SNAP)
    if diff:
        raise RuntimeError(f"tube_td: numba 커널에 굳은 상수 {diff} 를 첫 호출 뒤에 바꿨다 — 효과가 없다. 첫 호출 전에 바꾸거나 소스 기본값을 고쳐라 (§52.519)")


if njit is not None:
    _nm = set()
    for _d in _nb_dispatchers().values():
        _nm.update(_d.py_func.__code__.co_names)
    _NB_NAMES = tuple(sorted(n for n in _nm if n in globals() and not n.startswith("__")
                             and isinstance(globals()[n], (bool, int, float, np.integer, np.floating, np.ndarray))))
    _NB_SNAP0 = _nb_values()
    _NB_SRC_STAMP = _src_stamp()
    del _nm, _d
