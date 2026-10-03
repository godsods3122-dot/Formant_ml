"""노이즈 생성기(여러 개) — 마찰(협착 공기역학), 기식(성문), 과도음(템플릿 + 변조).

세 생성기의 역할이 다르다는 점이 설계의 핵심이다.

* **마찰 노이즈는 물리로 직접 만든다.** 치찰음은 학습 단에서 쉽게 나오지 않는
  요소이므로 협착 면적·압력에서 레이놀즈 수를 계산해 세기·스펙트럼을 낸다.
  (Stevens 1971/1998, Shadle 1985: 소스 세기 ∝ (Re² − Re_c²)⁺, 스펙트럼 정점
  Strouhal ≈ 0.2, 장애물(앞니) 다이폴은 +6 dB/oct.)
* **기식 노이즈는 성문 난류**다. `glottis.py` 가 준 포락선(개방기 동기 AM 포함)
  에 색만 입힌다. 성도 **전체**를 통과한다.
* **과도음은 템플릿 + 변조**다. 혀가 입천장에 닿는 소리, 혀를 튕긴 뒤 입 바닥에
  닿는 소리, 침 소리, 입술 소리는 짧은 결정론적 사건이라 통계 학습보다
  **템플릿 뱅크**가 맞다. 템플릿은 학습 가능 텐서이고, 변조(속도/세기/기울기/
  미세 난수)로 매번 다르게 낸다.

모든 생성기는 "소스" 를 낸다. 성도 통과(앞공동 공명, 뒤로 새는 경로, 방사)는
`tract.py` 가 맡는다 — 소스와 필터를 섞지 않는다(v1 §5.2 의 실수).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .control import frames_to_samples
from .glottis import CMH2O, RHO
from .rng import NoiseBank
from .tviir import first_difference_coeffs, lowpass_coeffs, tv_biquad

NU = 0.15                # 공기 동점성 cm²/s
RE_CRIT = 1800.0         # 난류 개시 레이놀즈 수 (Stevens 1998, 1700~1800)
RE_REF = 8000.0          # /s/ 급 협착의 전형값 — 세기 정규화 기준
C_SOUND = 35000.0        # cm/s
#: **마찰 v2** — 구동의 문턱을 매끈하게(softplus, 폭 `FRIC_SOFT_W`*Re_c^2) 하고 샘플률 포락에 2 단 1 차
#: 평활(`FRIC_ENV_MS`)을 건다. v1 의 `(Re^2 - Re_c^2).clamp_min(0)^1.5 / a_c` 는 문턱 근처에서 1 ms 마디로
#: 이은 유량·협착의 작은 흔들림을 깊은 골로 증폭했다(§50.9, docs/NOISE_SOURCE_REVIEW.md §2).
FRIC_V2 = False
FRIC_SOFT_W = 0.2
FRIC_ENV_MS = 2.0

# **성문 주기에 동기한 마찰 AM 의 깊이.** 폐쇄기에 유량이 줄어 마찰도 준다.
# 깊이 0.35 는 예전 사각파 게이트(`1 − 0.5·(frac < 0.35)`)의 주기 평균을 보존하도록
# 고른 값이다. 기식 쪽 `glottis.ASP_AM_DEPTH` 와 함께 F0 동기 변조를 만든다 —
# 실측으로 정해야 하므로 상수로 빼 둔다.
FRIC_AM_DEPTH = 0.35

#: **잡음의 주파수 간 결맞음** (MEASUREMENTS §52.99). 기식·마찰 잡음의 포락은 두 몫으로 이루어진다 —
#: 성문 제트의 F0 동기 AM 은 **모든 주파수에 공통**이고, 국소 와류의 흔들림은 **대역마다 독립**이다.
#: 지금까지는 `asp_env` 라는 스칼라 하나만 곱해서 공통 몫이 사실상 1 이었다.
#:
#: 실측(코퍼스 12 파일, 512 창·2.7 ms 홉, 포락의 주파수 간 상관 대 간격; 사분범위 ±0.01~0.09):
#:
#:     대역      94Hz   188Hz   375Hz   750Hz  1500Hz
#:     2-6k     0.818   0.698   0.699   0.599   0.575
#:     6-10k    0.819   0.722   0.668   0.623   0.564
#:     10-16k   0.803   0.703   0.679   0.614   0.572
#:     16-22k   0.663   0.501   0.447   0.401   0.304
#:
#: 간격이 벌어져도 **평탄해지는 값이 공통 몫** `ρ∞` 다 — 2~16 kHz 에서 0.55, 16 kHz 위에서 0.29.
#: 94 Hz 지점은 분석창(512, 빈 간격 93.75 Hz)의 누설이 섞이므로 평탄값만 쓴다. 지수 적합의 시간상수가
#: **결맞음 대역폭** `B` 이고 185 Hz / 145 Hz 로 나온다.
#:
#: 그래서 대역 `i` 의 포락을 `slow·[1 + √ρ·m(t) + √(1−ρ)·u_i(t)]` 로 만든다 — `m` 은 공통 AM,
#: `u_i` 는 대역마다 독립인 `B` 대역 잡음(같은 rms). 상수가 아니라 **측정된 곡선**이므로 화자마다 다시 잰다
#: (`tmp/cohcurve.py`). `NOISE_BANDS = 0` 이면 예전 거동이다.
NOISE_BANDS = 6                        # 고역을 나눌 대역 수 (로그 등간격)
NOISE_SPLIT_HZ = 1500.0                # 이 위를 나눈다 — 아래는 성문 제트가 지배한다
NOISE_RHO = ((16000.0, 0.55), (1e9, 0.29))   # (상한 Hz, 공통 몫) — 실측
NOISE_COH_BW_HZ = 170.0                # 독립 성분의 결맞음 대역폭 [Hz]
#: **대역별 공통 AM 깊이** (사용자의 두 파동, MEASUREMENTS §52.276). `ρ` 는 독립 성분을 **더하기만** 하고 공통 성분 `m` 은
#: 못 줄인다 — 게다가 독립 성분은 주기 평균에서 상쇄되므로 '펄스에 묶인 정도'를 바꾸지 못한다(실측: ρ 0.25 대 0.85 가
#: 12 kHz 깊이 7.69 대 7.77 dB). 펄스 동기를 실제로 정하는 것은 **공통 AM 의 깊이**이고, 지금은 모든 대역이 똑같이 받는다.
#: 제트 난류가 센 대역은 펄스 동기가 풀리고(β 작게), 폐쇄 가장자리가 지배하는 대역은 묶인다(β 크게).
#: '(상한 Hz, β)' 목록. 기본은 전부 1.0 = 예전 거동.
NOISE_AM_BETA = ((1e9, 1.0),)
#: **대역별 기식 세기** [dB] (사용자의 두 파동, MEASUREMENTS §52.276). 펄스 동기 깊이를 실제로 정하는 것은 ρ 도 β 도 아니라
#: **기식이 펄스 사이를 얼마나 메우느냐**다. 실측(같은 제어열, 기식만 끔): 9~12 kHz 깊이가 4.42 → 15.42 dB 로 뒤집힌다.
#: 원본은 10.32 로 그 사이다 — 우리는 그 대역에 기식을 **너무 많이** 넣고, 5~9 kHz 에는 **적게** 넣는다.
#: 제트 난류 스펙트럼은 봉우리를 지나 굴러떨어지는데 우리 색(1 차 셸프)은 20 kHz 까지 평평하다. '(상한 Hz, dB)' 목록.
NOISE_BAND_GAIN_DB = ((1e9, 0.0),)
ASP_AM_SLOW_HZ = 60.0
#: 공통 AM `m = (open_phase − 0.5)` 의 **공칭 rms**. 청크에 의존하지 않도록 상수로 둔다 —
#: `open_phase = sin²(π·frac/0.65)` (frac < 0.65), 한 주기에 걸친 값이다 (소수 5 자리).
M_RMS_NOMINAL = 0.41079



# **앞니 다이폴이 협착 면적비에 붙는 거듭제곱** — 제트의 *기하* 효율.
#
# 이게 치찰음 고역의 페이드 인을 만드는 항이다. Shadle 의 장애물 소스는 **좁은
# 제트가 모서리에 부딪히는** 구조라, 협착이 넓으면 제트가 굵고 퍼져서 앞니라는
# 국소 장애물을 때리는 효율이 떨어진다. 협착이 목표 자세에 도달해야 비로소
# 다이폴이 스펙트럼을 지배한다.
#
#     g = (A_ref / A_c)^exp,  단 1 을 넘지 않는다
#
# 이게 없으면 소스의 **모양이 토큰 내내 고정**돼 대역들이 다 같이 올라온다 —
# 협착이 닫히는 동안 고역이 중역과 함께 서 버리고 페이드 인이 사라진다.
# 실측 (대역이 자기 최대의 −6 dB 에 닿는 시각, 긴 /s/):
#
#                  0-2k  2-4k  4-7k  7-11k 11-16k   고역−중역
#   사람 실측 #1    520   287    89    202    216      +127 ms
#   사람 실측 #2    343   467   258    521    554      +296 ms
#   기하항 없음     181    53    53     64     75       +21 ms   <- 묶여 있다
#
# 2.5 는 v1 이 마찰음 무게중심으로 적합한 값이다 (v1 aeroacoustic.py
# `OBSTACLE_JET_EXP`, 짧은 CV 의 언더슈트를 재현하는 지수). 1.0 이면 음절이
# 실측보다 1.3~1.9 kHz 높고, 3.0 이면 지나치다.
OBSTACLE_JET_EXP = 2.5
# 그 기준 면적 [cm²]. /s/ 급 협착의 목표 자세다 — 이 파일이 이미 두 군데에서
# 같은 값을 쓰고 있다 (Stevens 항의 `0.1 / a_c`, 아래 V_REF_DIPOLE 의 A).
# **화자별 a_min 이 아니라 고정 상수여야 한다.** 화자 자세로 정규화하면 g 가
# 협착의 절대 좁음이 아니라 "그 화자 기준 몇 %" 를 재게 되어, a_min 이 넓은
# 화자의 /s/ 도 앞니를 제 세기로 때리는 것이 된다. 프로파일 a_min 은 0.08~0.10
# 이라 고원에서는 g = 1 로 포화하고, 그래서 **고원의 거동은 예전 그대로**다 —
# 바뀌는 건 협착이 닫혀 가는 동안뿐이다.
OBSTACLE_A_REF = 0.1

# **앞니 다이폴이 제트 속도에 붙는 거듭제곱** (Curle 1955 의 U⁶ 법칙에서 U¹).
#
# 물리는 맞다: 고체 경계가 있는 흐름의 다이폴 방사 파워는 U⁶, 경계 없는 단극은
# U⁴ 이므로 다이폴의 **상대** 세기는 U^(3−2) = U¹ 이다. 다만 **이 항 혼자서는
# 페이드 인을 못 만든다.** 협착부 속도는
#
#     v = sqrt(2Ps/ρ) / sqrt(Ac²/Ag² + 1)
#
# 인데 무성 마찰음은 성문이 크게 벌어져 있어(Ag ≫ Ac) Ac²/Ag² 가 거의 0 이라
# **v 가 압력에만 묶인다**. 측정 (긴 /s/, 협착 구간 107 프레임): 제트 속도가
# 중앙 3536 / 최대 3536 cm/s 로 베르누이 상한 sqrt(2Ps/ρ) 에 붙어 있고,
# v/V_REF 가 1.05 여서 clamp 에 걸려 상수 1 이 된다. 그래서 vel_exp 0.0 과 1.0
# 이 rise 40/200/500 ms 어느 조건에서도 **바이트 단위로 같은 출력**을 냈다.
# v1 도 같은 결론을 독립으로 적었다 ("속도로 재면 안 된다 … 판별력이 없다",
# aeroacoustic.py `obstacle_strength`).
#
# 그래도 남겨 둔다 — 폐압이 실제로 오르내리는 구간(발화 개시·종결, 호흡 구동
# 지속 마찰음)에서는 v 가 상한 아래로 내려와 항이 산다. **페이드 인을 만드는
# 것은 위의 기하항이다.**
OBSTACLE_VEL_EXP = 1.0
# 그 기준 속도 [cm/s]. /s/ 급 협착(A ≈ 0.1 cm², Re ≈ RE_REF)에서의 입자 속도:
#   d = sqrt(4A/π) = 0.357 cm,  v = Re·ν/d = 8000 × 0.15 / 0.357 ≈ 3361 cm/s
# **구간 최대가 아니라 고정 상수여야 한다** — 최대로 정규화하면 청크마다 값이 달라져
# 스트리밍 = 오프라인 불변량이 깨진다 (v1 은 오프라인이라 `v.amax()` 를 썼다).
V_REF_DIPOLE = RE_REF * NU / math.sqrt(4.0 * 0.1 / math.pi)


def series_flow(p_sub_cmh2o, a_g, a_c):
    """성문(Ag)·구강 협착(Ac) 직렬 오리피스의 정상 유량 U [cm³/s] 와 협착 강하압.

    U = sqrt(2Ps/ρ) / sqrt(1/Ag² + 1/Ac²).  Po/Ps = Ag²/(Ag²+Ac²) (v1 §5.3 검증).
    """
    ps = p_sub_cmh2o.clamp_min(0.0) * CMH2O
    inv = 1.0 / a_g.clamp_min(1e-4) ** 2 + 1.0 / a_c.clamp_min(1e-4) ** 2
    u = torch.sqrt(2.0 * ps / RHO + 1e-12) / torch.sqrt(inv)   # +eps: 무음(ps=0)에서 기울기 NaN
    dp_c = 0.5 * RHO * (u / a_c.clamp_min(1e-4)) ** 2
    return u, dp_c


def reynolds(u, a_c):
    """협착 제트의 레이놀즈 수. d = 등가 직경 sqrt(4A/π)."""
    a = a_c.clamp_min(1e-4)
    v = u / a
    d = torch.sqrt(4.0 * a / math.pi + 1e-12)
    return v * d / NU, v, d


def butterworth_q(n_sections: int) -> list[float]:
    """2n 차 버터워스를 2 차 절편 n 개로 나눌 때의 Q 들.

    **같은 Q 를 반복하면 안 된다.** 같은 2 차 저역통과를 n 번 걸면 −3 dB 점이 아래로
    끌려 내려오고 무릎이 뭉개져 "완만한 내리막" 이 된다. 실측 /ㅅ/ 는 10 kHz 까지
    평평하다가 12 kHz 부터 절벽이다 — 그 모양은 통과대역이 평평하고 무릎이 선
    버터워스라야 나온다. Q_k = 1 / (2 cos((2k+1)π / 4n)).
    """
    return [1.0 / (2.0 * math.cos((2 * k + 1) * math.pi / (4 * n_sections)))
            for k in range(n_sections)]


def slow_modulation(white_frames: torch.Tensor, fs_frame: float, knee_hz, zi=None):
    """Fixed two-pole lowpass modulation (approximately unit variance).

    2 극 저역통과(RBJ, 꺾임 knee_hz)를 **상태를 이어가며** 적용한다 — FFT 로 만들면
    청크 경계에 따라 달라져 스트리밍 = 오프라인이 깨진다. 이득은 백색 입력의 출력
    분산이 1 이 되도록 근사 정규화한다.
    """
    # Preserve the old float64 coefficient calculation without detaching knee.
    k = torch.as_tensor(knee_hz, dtype=torch.float64, device=white_frames.device)
    if not bool(torch.isfinite(k).all() & (k > 0).all() & (k < fs_frame / 2).all()):
        raise ValueError("knee_hz must be finite and between zero and frame Nyquist")
    co = lowpass_coeffs(k, 0.707, fs_frame)
    y, zf = tv_biquad(white_frames, *co, zi=zi)
    gain = torch.sqrt(fs_frame / (2.0 * math.pi * k)).to(y.dtype)
    return y * gain, zf


#: 마찰 소스 고역 절벽 배율의 중심(ln 3.2)과 가둠 폭(±ln 2 → 1.6~6.4 배).
LOG_LP0 = math.log(3.2)
LP_RATIO_LIM = math.log(2.0)


class FricationNoise(nn.Module):
    """협착 공기역학 -> 마찰 소스 (샘플률). 색은 여기서 **소스 스펙트럼**까지만."""

    def __init__(self, fs: float, hop: int, mod_depth: float = 0.25,
                 amp_ref: float = 0.062, lp_ratio: float = 3.2,
                 lp_stages: int = 2, source_hf_shelf_db: float = 0.0):
        super().__init__()
        self.fs, self.hop = float(fs), int(hop)
        self.mod_depth = mod_depth
        # 제트 난류 스펙트럼의 고역 절벽: Strouhal 정점의 lp_ratio 배에서 4 차 버터워스.
        self.log_lp_ratio = nn.Parameter(torch.tensor(math.log(lp_ratio)))
        self.lp_stages = lp_stages
        # **채점은 소스가 아니라 방사된 출력에서 한다** (0.3.3). 소스만 보고 맞추면
        # 앞공동 극이 그 위에 −12 dB/oct 를 또 얹는 것을 놓친다. 실제로 소스 기준으로
        # 맞춘 lp_ratio 1.8 은 출력에서 남 /ㅅ/ 12~16 kHz 를 실측보다 17~27 dB 어둡게
        # 만들었다(정점 기준). 실측은 방사음이므로 채점도 방사음이어야 한다.
        #
        # 출력 기준 재적합 (남 ㅅ ×3 · 여 ㅆ, 정점 대역 기준 dB):
        #   lp_ratio 1.8 -> 3.2 :  rms 남 13.67 / 여 10.42  ->  남 2.44 / 여 2.04
        # 채점 대역: 남은 4~16 kHz (44.1 kHz 녹음이라 그 위는 안티에일리어싱),
        # 여는 4~22 kHz (48 kHz 녹음이라 16~22 kHz 도 실측이다). 4 kHz 아래는 방 잡음.
        # 통과대역(4~12 kHz)의 기울기는 이 절벽이 아니라 **앞공동 극의 대역폭**(프로파일)이
        # 정한다. 절벽은 그 위 12~22 kHz 의 상한이다.
        # 셸프는 −6~+6 dB 에서 rms 가 0.03 밖에 안 움직여 **사실상 작동하지 않는다.**
        # 값이 남아 있으면 뭔가 하는 것처럼 읽히므로 0 으로 둔다.
        self.source_hf_shelf_db = source_hf_shelf_db
        # Historical checkpoint key only: the fixed-shape lowpass never used
        # beta. No slope fitting is implemented; do not advertise a dead knob.
        self.register_buffer("log_beta", torch.tensor(math.log(2.0)))
        # Differentiable internals, NOT optimized by the default control fitter.
        self.log_knee = nn.Parameter(torch.tensor(math.log(8.0)))
        self.log_amp = nn.Parameter(torch.tensor(math.log(amp_ref)))

    def forward(self, c: dict, ag_dc: torch.Tensor, glottal_phase: torch.Tensor,
                voiced: torch.Tensor, noise=None, frame0: int = 0,
                state: dict | None = None, emit: int | None = None,
                noise_am: torch.Tensor | None = None,
                po_frames: torch.Tensor | None = None, uc_frames: torch.Tensor | None = None) -> dict:
        """c: 프레임률 (B,T) dict (p_sub, a_c, obstacle, fric_gain). ag_dc: **프레임률** (B,T) 성문 면적.

        noise: NoiseBank (위치 기반), frame0: 청크의 첫 프레임, state: 필터 상태(스트리밍).
        noise_am: optional shared unit-mean LF envelope from GlottalSource;
                  omitted retains legacy phase modulation and calling convention.
        반환: source (B,N) 마찰 소스 파형, env (B,N), f_peak (B,N), reynolds, flow, state
        """
        fs, hop = self.fs, self.hop
        b, t_all = c["p_sub"].shape
        t = t_all if emit is None else int(emit)
        n = t * hop
        noise = noise or NoiseBank()
        state = {} if state is None else state
        up = lambda v: frames_to_samples(v.unsqueeze(-1), hop)[..., 0][:, :n]
        a_c = up(c["a_c"]); ps = up(c["p_sub"])
        agf = ag_dc                                      # 프레임률 (B, T_all)
        ag_s = up(agf)
        # 연구개가 열리면 기류는 코로 빠진다. 구강 협착을 지나는 유량과 구강압 축적은 (1−velum) 배.
        # (비음의 구강 폐쇄 a_c≈0.02 에서 Re=3700 짜리 마찰 + 방전 버스트가 나던 것 — 측정)
        oral = (1.0 - c["velum"]).clamp(0.0, 1.0)
        if uc_frames is not None:
            # **구강압 상미분방정식의 협착 유량** (glottis.ORAL_ODE, §52.436) — 정상 유량식 대신. 폐쇄 중 쌓인 Po 가 개방에서
            # U_c = A_c √(2Po/ρ) 로 터져 나오므로 파열의 제트 속도·세기가 따로 된 식 없이 나온다.
            u = up(uc_frames)
        else:
            u, _ = series_flow(ps, ag_s, a_c)
            u = u * up(oral)
        # 구강압 저장 → 해제 시 방전 (파열음·파찰음의 버스트). 프레임률 1 차 계.
        #   닫힘(a_c < ~0.05): Po → Ps (τ 15 ms). 열림: Po → 정상값 (τ 6 ms).
        #   버스트 유량 = 저장된 초과압의 방전. ㅈ/ㅊ 개시가 10 ms 안에 −70→−30 dB 로 서는 것(계측).
        closed = torch.clamp((0.06 - c["a_c"]) / 0.04, 0.0, 1.0)
        closed = closed * closed * (3 - 2 * closed)
        po_ss = c["p_sub"] * agf ** 2 / (agf ** 2 + c["a_c"].clamp_min(1e-3) ** 2)   # 정상 구강압
        dt = hop / fs
        if po_frames is not None:
            # **성문이 이미 적분한 값을 그대로 쓴다** (§52.344). 같은 물리를 두 번 적분하면
            # 두 곳이 조용히 어긋난다 — 성문은 압력차로 떨고 마찰은 딴 압력으로 터진다.
            po_all = po_frames[:, :t_all]
        else:
            po = state.get("po", po_ss[:, 0] * 0.0)
            po_hist = []
            for i in range(t_all):
                tgt = (closed[:, i] * c["p_sub"][:, i]
                       + (1 - closed[:, i]) * po_ss[:, i]) * oral[:, i]
                tau = 0.015 * closed[:, i] + 0.006 * (1 - closed[:, i])
                po = po + dt * (tgt - po) / tau
                po_hist.append(po)
            po_all = torch.stack(po_hist, 1)
        # 틀마다 목록을 쌓지 않고 한 번에 (§52.468 — 같은 값)
        burst = (po_all - po_ss[:, :t_all]).clamp_min(0.0) * (1 - closed[:, :t_all])
        state["po"] = po_all[:, t - 1]                   # 상태는 내보내는 마지막 프레임 기준
        burst_env = up(torch.sqrt(burst.clamp_min(0.0) / CMH2O + 1e-12)) * 3.0     # 초과압 → 속도 배율
        re, v, d = reynolds(u, a_c)
        # Stevens: 소스 압력 ∝ ρ·v³·√A. Re = v·d/ν ∝ v·√A 이므로 같은 Re 에서 v³√A ∝ Re³/A —
        # 넓은 통로(모음 자세)는 같은 Re 라도 조용하다. (Re²−Re_c²)^1.5/(A/A_ref), A_ref=0.1 cm².
        if FRIC_V2:
            w = FRIC_SOFT_W * RE_CRIT ** 2
            over = w * torch.nn.functional.softplus((re ** 2 - RE_CRIT ** 2) / w)
            drive = (over / RE_REF ** 2) ** 1.5 * (0.1 / a_c.clamp_min(0.02))
        else:
            drive = (((re ** 2 - RE_CRIT ** 2).clamp_min(0.0) / RE_REF ** 2) ** 1.5
                     * (0.1 / a_c.clamp_min(0.02)))
        if uc_frames is None:
            drive = drive + burst_env ** 3 * (0.1 / a_c.clamp_min(0.02))   # 버스트: 방전 속도의 세제곱 (옛 경로)
        env = drive * torch.exp(self.log_amp) * up(c["fric_gain"])
        if FRIC_V2 and FRIC_ENV_MS > 0.0:
            a_s = 1.0 - math.exp(-1000.0 / (FRIC_ENV_MS * fs))
            for p_ in (0, 1):
                k_ = "env%d" % p_
                zi = state.get(k_)
                if zi is None:
                    e0 = env[:, :1].to(torch.float64)
                    zi = torch.cat([(1.0 - a_s) * e0, torch.zeros_like(e0)], -1)
                env, state[k_] = tv_biquad(env, a_s, 0.0, 0.0, -(1.0 - a_s), 0.0, zi=zi)
        # 장애물(앞니) 소스의 혹은 자유 제트의 Strouhal 정점보다 높고 넓다(Shadle). 같은 화자 A/B 에서
        # St=0.2 그대로 두면 1~3 kHz 가 10~12 dB 과했다. 정점 ×2.5, Q 0.5.
        f_peak = (0.5 * v / d.clamp_min(1e-3)).clamp(800.0, 0.45 * fs)
        # Fixed two-pole slow modulation + voiced glottal-cycle AM.
        mod, zmod = slow_modulation(
            noise.white("mod", frame0, t_all, b, ps.dtype, ps.device), fs / hop,
            torch.exp(self.log_knee), state.get("mod"))
        if t != t_all:                       # 상태는 내보내는 마지막 프레임 기준
            _, zmod = slow_modulation(noise.white("mod", frame0, t, b, ps.dtype, ps.device),
                                      fs / hop, torch.exp(self.log_knee), state.get("mod"))
        state["mod"] = zmod
        mod = 1.0 + self.mod_depth * up(mod)
        # 폐쇄기에 유량이 준다 -> 마찰 진폭도 준다. **매끄러운 유량 변조여야 한다.**
        #
        # 예전 형태 `1 − 0.5·voiced·(frac < 0.35)` 는 성문 주기의 앞 35 % 동안 진폭을
        # 절반으로 뚝 떨어뜨렸다 올리는 **샘플 해상도의 계단**이었다. 실측(대조군
        # 대비, docs/HANDOFF.md §7): 60~400 Hz 변조 대역이 18.4 % -> 47.2 % 로 뛴다 —
        # 사람이 "지글거린다" 고 듣는 바로 그 성분이다. 실제 성문-마찰 결합은 유량의
        # 연속 변조이지 계단이 아니다. `(frac < 0.35)` 는 꺾임이기도 해서 손실을 C² 가
        # 아니게 만든다 (relu 를 걷어낸 것과 같은 부류).
        #
        # 고친 형태는 **위상의 주기함수**인 올림 코사인이다. 폐쇄기(위상 0)에서 1,
        # 개방기(위상 π)에서 0 으로, 계단이 있던 곳과 같은 쪽으로 같은 뜻을 준다.
        # 깊이 0.35 는 **주기 평균을 보존**하도록 고른 값이다 (예전 1−0.5·0.35 = 0.825,
        # 지금 1−0.35·0.5 = 0.825). 그래서 마찰의 전체 수준은 바뀌지 않고, 변조가
        # F0 하나에만 실린다 — 계단이 만들던 F0 배음 계열이 사라진다.
        frac = glottal_phase[:, :n] / (2 * math.pi)
        gate = 0.5 * (1.0 + torch.cos(2 * math.pi * frac))
        am = 1.0 - FRIC_AM_DEPTH * voiced[:, :n] * gate
        if noise_am is not None:
            # Keep the legacy cycle mean (0.825 when voiced), but use exactly
            # the same LF flow shape as aspiration, with no extra fit parameter.
            am = (1.0 - 0.175 * voiced[:, :n]) * noise_am[:, :n]
        white = noise.white("fric", frame0 * hop, n, b, ps.dtype, ps.device)
        # 난류 소스의 스펙트럼.
        #
        # 초판은 Strouhal 정점에 2 차 대역통과(Q 0.5)를 걸었다. 그러면 정점 아래가 −6 dB/oct
        # 로 떨어져 1~2 kHz 가 실측보다 15 dB 낮았다(같은 화자 A/B). 실측 /s/ 는 정점 아래가
        # **완만한 어깨**다: 5~8 kHz 정점 대비 1~2 kHz 가 −17 dB, 2~3 kHz 가 −22 dB.
        #
        # 물리적으로도 그쪽이 맞다. 마찰음의 스펙트럼 정점을 만드는 것은 소스가 아니라
        # **앞공동 공진**이고(Stevens: 소스는 광대역), 소스 자신은 에디 크기가 정하는
        # 모서리 위에서만 떨어진다. 그래서 여기서는
        #   백색 → 기울기(tilt, 학습) → 정점의 lp_ratio 배에서 2 차 저역통과
        # 로 두고, 봉우리는 tract 의 앞공동 극이 만들게 한다.
        # 소스의 고역 셸프 (1 kHz 모서리). 실측 적합값 −9 dB: 광대역 백색보다 조금 어둡다.
        shelf = self.source_hf_shelf_db
        src = white
        # v1 의 `source_tilt_shift`(속도가 2 배면 소스가 1.5 dB/oct 밝아진다)를 여기
        # 셸프로 옮겨 봤다가 **뺐다.** 1 차 셸프는 모서리 위에서 평평해져서 2~4 kHz 와
        # 11~16 kHz 에 같은 이득을 준다 — 참 dB/oct 기울기가 아니라 대역을 못 가른다.
        # 실측으로도 tilt 0 → 3.0 에서 대역 개시 차가 +30 ms 로 꿈쩍 안 했다(§21.6).
        # 제대로 하려면 옥타브마다 셸프를 쌓아야 하는데, 그럴 근거가 아직 없다.
        if abs(float(shelf)) > 1e-6:
            a = math.exp(-2 * math.pi * 1000.0 / fs)
            g_hi = 10.0 ** (shelf / 20.0)
            hp, state["tl"] = tv_biquad(src, 0.5 * (1 + a), -0.5 * (1 + a), 0.0, -a, 0.0,
                                        zi=state.get("tl"))
            src = src + (g_hi - 1.0) * hp
        # 장애물(앞니) 다이폴: +6 dB/oct 성분을 obstacle 비율로 섞는다 (Curle)
        #
        # **다이폴 세기는 협착 자세에 묶여야 한다.** `c["obstacle"]` 은 "이 음소에
        # 앞니 장애물이 있는가" 라는 음소 상수(치찰음 0.15, /f/ /h/ 는 0)이지
        # 시간 궤적이 아니다. 그것만 쓰면 협착이 아직 넓은 개시 구간에서도 다이폴이
        # 제 세기로 실려, 고역이 중역과 **함께** 서고 페이드 인이 사라진다.
        #
        # 그래서 기하 효율 g = (A_ref/A_c)^2.5 를 곱한다 (OBSTACLE_JET_EXP 참조).
        # 좁은 제트라야 앞니를 때린다 — 넓으면 굵고 퍼져 국소 장애물을 못 때린다.
        # 고원(A_c ≤ A_ref)에서 g = 1 이므로 **적합된 고원 거동은 그대로**고,
        # 협착이 닫혀 가는 동안에만 다이폴이 물러나 앞공동 극이 드러난다.
        #
        # 속도항은 물리적으로 맞지만 폐압이 일정한 구간에서는 포화해 무력하다
        # (OBSTACLE_VEL_EXP 의 주석에 측정 있음). 기하항과 **곱**으로 함께 둔다.
        obst = up(c["obstacle"])
        if OBSTACLE_JET_EXP > 0.0:
            obst = obst * (OBSTACLE_A_REF / a_c.clamp_min(1e-4)).clamp(0.0, 1.0) ** OBSTACLE_JET_EXP
        if OBSTACLE_VEL_EXP > 0.0:
            obst = obst * (v / V_REF_DIPOLE).clamp(0.0, 1.0) ** OBSTACLE_VEL_EXP
        dip, state["dip"] = tv_biquad(src, *first_difference_coeffs(1.0, src), zi=state.get("dip"))
        src = (1.0 - obst) * src + obst * dip * 4.0
        # **고역 절벽.** 에디가 아무리 작아도 크기에 하한이 있어 소스는 어느 주파수 위에서
        # 떨어진다. 이 절벽이 `log_lp_ratio` 로 선언만 되어 있고 **걸리지 않고 있었다**:
        # 그래서 소스가 나이퀴스트까지 평평했고, 실측 남성 /ㅅ/ 이 정점 대비 12~16 kHz 에서
        # −30 dB 인데 합성은 −13 dB 에서 멈췄다(측정). 정점의 lp_ratio 배에서 2 차씩
        # `lp_stages` 단 — 한 단(−12 dB/oct)으로는 실측 기울기에 못 미친다.
        # 절벽 배율은 3.2 배 둘레 1.6~6.4 배로 가둔다(적합할 때 달아나지 않게). 기본값에서는 그대로 3.2 배다.
        llr = LOG_LP0 + LP_RATIO_LIM * torch.tanh((self.log_lp_ratio - LOG_LP0) / LP_RATIO_LIM)
        lpf = (f_peak * torch.exp(llr)).clamp(600.0, 0.45 * fs)
        for i, q in enumerate(butterworth_q(self.lp_stages)):
            src, state[f"lp{i}"] = tv_biquad(src, *lowpass_coeffs(lpf, q, fs),
                                             zi=state.get(f"lp{i}"))
        return dict(source=src * env * mod * am, env=env, f_peak=f_peak,
                    reynolds=re, flow=u, state=state)


#: **기식 v2.1 — 유성 정도로 가른다** (docs/NOISE_SOURCE_REVIEW.md §4).
#:
#: v2.0 은 성대가 떨든 말든 기식을 MVF(5.5 kHz) 위로만 보냈다. 문헌과 목표가 둘 다 반박한다:
#: MVF 는 시변이고 무성음에서 ~1 kHz 다(MVF 연구), Klatt 도 기식을 모음과 같은 포먼트에 통과시킨다.
#: 목표 실측 — 무성·비마찰 기식 프레임의 0.3~5 kHz 봉우리 깊이가 모음의 80 %, 모음 포락과 상관
#: +0.46 (s040) / +0.31 (s101): **무성 기식은 포먼트 구조를 가진 전대역 잡음**이다. 그래서
#:   * 유성분(떨림 진폭 비례 v): MVF 위 2 차 고역, 바닥 `ASP_V_FLOOR` — 성도의 저 Q 사본으로
#:   * 무성분(1 − v): 예전 셸프(3 kHz, 바닥 0.3) — **정상 종속 가지**(물리 포먼트)로
#: MVF 는 파일 전역 적합값(`log_mvf`, 2~9 kHz, 초기 5.5 kHz). 목표도 파일마다 중역 주기성이 다르다.
#: **기식의 고역 감쇠 모서리** [Hz] (사용자의 두 파동, MEASUREMENTS §52.276). 지금 기식의 색은 고역 셸프와 고역통과뿐이라
#: **20 kHz 까지 평평하다** — 굴러떨어질 수단이 아예 없다. 실제 제트 난류 스펙트럼은 봉우리를 지나 감쇠한다.
#: 그래서 우리는 9~16 kHz 에 난류를 과하게 깔고, 그것이 펄스 사이를 메워 세로줄을 지운다 —
#: 실측(같은 제어열, 기식만 끔): 9~12 kHz 펄스 동기 깊이 4.42 → 15.42 dB, 원본은 10.32 로 그 사이다.
#: 0 이면 끈다. `ASP_LP_ORDER` 극을 쓴다.
ASP_LP_HZ = 0.0
ASP_LP_ORDER = 1
ASP_SPLIT = False
#: **유성 기식에도 대역별 결맞음 구조를 건다** (사용자 지시, MEASUREMENTS §52.275). `ASP_SPLIT`(`--noise-v2`)이 켜지면
#: `forward` 가 먼저 반환해서 `_partial_coherence` 를 **아예 안 부른다** — 대역별 공통/독립 분리(ρ)가 통째로 빠진다.
#: 그래서 우리 고역 잡음은 대역마다 같은 AM 을 그대로 받아 펄스 동기가 어긋난다(§52.271: 5~9 kHz 과다, 9~16 kHz 부족).
#: 켜면 유성 성분에만 `_partial_coherence` 를 태운다 — 무성에는 성문 펄스가 없으므로 그대로 둔다.
ASP_SPLIT_BANDS = False
ASP_V_FLOOR = 0.05


def _asp_rolloff(a, b, fs, state):
    """기식에 고역 감쇠를 건다 (`ASP_LP_HZ`). 상태는 청크 사이로 이어진다."""
    if ASP_LP_HZ <= 0.0:
        return a, b, state
    r = math.exp(-2 * math.pi * float(ASP_LP_HZ) / float(fs))
    for k in range(max(1, int(ASP_LP_ORDER))):
        a, state[f"asplp{k}"] = tv_biquad(a, 1.0 - r, 0.0, 0.0, -r, 0.0, zi=state.get(f"asplp{k}"))
        if b is not None:
            b, state[f"asplpu{k}"] = tv_biquad(b, 1.0 - r, 0.0, 0.0, -r, 0.0,
                                              zi=state.get(f"asplpu{k}"))
    return a, b, state


#: **기식 음원의 셸프 꺾임** [Hz] (MEASUREMENTS §52.432). 3000 은 남성 /아/ 의 4~8 kHz HNR 에 맞춘 값이었다. 문헌: 성문 난류 음원은
#: 중심 주파수(틈의 면적·유량이 정함) 둘레 **2~3 옥타브에서 평평**하다 (Stevens, "Characteristics of the glottal turbulent noise
#: source"). 성문 제트 v ≈ 37 m/s, 폭 2~4 mm → 중심 ≈ 2~4 kHz → 평평 ≈ 0.7~10 kHz. 3 kHz 아래를 −10.5 dB 깎으면 파열 뒤 기식의
#: 0.5~2 kHz 가 원본보다 15~25 dB 약했다(C2 0.176~0.21 s). `copyfit --asp-corner` 로 바꾼다.
ASP_CORNER_HZ = 3000.0
#: **셸프 꺾임을 성문 제트의 스트로할 주파수로** (§52.432). 고정 3000 은 모음(좁은 틈, 수 kHz 중심)에 맞고, 파열 뒤 기식(넓은 틈,
#: 중심 ≈ 1 kHz)에는 중역이 15~25 dB 모자랐다. 둘은 같은 물리다 — 난류 음원의 중심 f ≈ St·v/d (St ≈ 0.2, 제트 소음), 틈이 넓으면 낮다.
#: 켜면 꺾임 = St·v/d_h 를 표본마다: A = ag_dc + 0.5·GLOT_A_PEAK·amp (평균 면적), d_h = 2A/성대 길이, v = √(2 ΔPg/ρ),
#: ΔPg = Ps·a_c²/(a_c²+A²). 모달(A≈0.1 cm², Ps 8) → ≈ 3.7 kHz (예전 3000 근처), 벌린 성문(A≈0.45, Ps 20) → ≈ 1.3 kHz.
ASP_STROUHAL = False
ASP_ST = 0.2


class AspirationNoise(nn.Module):
    """성문 기식: 포락선은 glottis 가 물리로 준다. 여기서는 색(완만한 고역 셸프)만."""

    order = 1           # 고역 셸프의 차수 (2 면 1 차 고역통과를 두 번 — 12 dB/oct)

    def __init__(self, fs: float, hop: int, corner_hz: float | None = None, floor: float = 0.3,
                 amp_ref: float = 0.3):      # 실측 남성 /아/ HNR 22 dB, 4~8 kHz 대역에 적합 (2026-09-06)
        super().__init__()
        self.fs, self.hop = float(fs), int(hop)
        self.corner, self.floor = (ASP_CORNER_HZ if corner_hz is None else corner_hz), floor
        self.log_amp = nn.Parameter(torch.tensor(math.log(amp_ref)))
        # v2.1: 유성 기식의 MVF [Hz] — 파일 전역 적합값. 0 에서 5.5 kHz.
        self.log_mvf = nn.Parameter(torch.tensor(0.0))

    MVF_LO, MVF_HI, MVF_INIT = 2000.0, 9000.0, 5500.0

    def mvf(self) -> torch.Tensor:
        lo, hi = math.log(self.MVF_LO), math.log(self.MVF_HI)
        x0 = (math.log(self.MVF_INIT) - lo) / (hi - lo)
        b0 = math.log(x0 / (1.0 - x0))
        return torch.exp(lo + (hi - lo) * torch.sigmoid(self.log_mvf + b0))

    def _band_gain(self, f_hz: float) -> float:
        """그 주파수의 **기식 세기 배율** (`NOISE_BAND_GAIN_DB`)."""
        for hi, g in NOISE_BAND_GAIN_DB:
            if f_hz < hi:
                return 10.0 ** (float(g) / 20.0)
        return 10.0 ** (float(NOISE_BAND_GAIN_DB[-1][1]) / 20.0)

    def _beta(self, f_hz: float) -> float:
        """그 주파수의 **공통 AM 깊이 배율** (`NOISE_AM_BETA`)."""
        for hi, b in NOISE_AM_BETA:
            if f_hz < hi:
                return float(b)
        return float(NOISE_AM_BETA[-1][1])

    def _rho(self, f_hz: float) -> float:
        """그 주파수의 **공통 몫** (`NOISE_RHO`, 실측 곡선)."""
        for hi, r in NOISE_RHO:
            if f_hz < hi:
                return float(r)
        return float(NOISE_RHO[-1][1])

    def _bands(self, x, state):
        """누적 저역통과의 차로 대역을 나눈다. **합이 원 신호와 같다** (시험이 고정한다).

        경계는 `NOISE_SPLIT_HZ` 부터 나이퀴스트까지 로그 등간격이고, 첫 대역은 그 아래 전부다.
        """
        edges = [NOISE_SPLIT_HZ * (self.fs / 2.0 / NOISE_SPLIT_HZ) ** (i / NOISE_BANDS)
                 for i in range(1, NOISE_BANDS)]
        out, prev, lo_f = [], None, NOISE_SPLIT_HZ
        for i, fc in enumerate([NOISE_SPLIT_HZ] + edges):
            r = math.exp(-2 * math.pi * fc / self.fs)
            lp, state[f"nb{i}"] = tv_biquad(x, 1.0 - r, 0.0, 0.0, -r, 0.0, zi=state.get(f"nb{i}"))
            out.append((lo_f if i else 0.0, fc, lp if prev is None else lp - prev))
            prev, lo_f = lp, fc
        out.append((lo_f, self.fs / 2.0, x - prev))
        return out

    def _partial_coherence(self, colored, asp_env, noise, sample0, state):
        """대역마다 **공통 AM + 독립 흔들림**을 건다 (`NOISE_BANDS` 참조).

        대역 i 의 포락 = `slow·[1 + √ρ·m + √(1−ρ)·u_i]`. `m` 은 공통(성문 제트) AM 의 정규화 성분이고
        `u_i` 는 `NOISE_COH_BW_HZ` 로 저역통과한 독립 잡음을 `m` 과 같은 rms 로 맞춘 것이다.
        두 대역 포락의 상관은 설계상 `ρ` 가 된다 (`u_i` 끼리 독립이므로).
        첫 대역(`NOISE_SPLIT_HZ` 아래)은 성문 제트가 지배하므로 `ρ = 1` — 공통 AM 그대로다.
        """
        slow, fast = self._am_split(asp_env, state)
        m = fast / slow.clamp_min(1e-9)
        # **청크 길이에 의존하는 정규화를 쓰면 안 된다.** 처음에는 `m` 과 `u` 를 각자의 rms 로 맞췄는데,
        # rms 는 그 청크 안에서 계산되므로 스트리밍과 오프라인이 갈렸다 (실측 0.088, 허용 2.5e-4 —
        # `test_streaming_equals_offline` 이 잡았다). 둘 다 **해석적 상수**로 정규화한다.
        rl = math.exp(-2 * math.pi * NOISE_COH_BW_HZ / self.fs)
        u_gain = math.sqrt((1.0 + rl) / (1.0 - rl))     # 1 극 저역통과의 출력 분산 (1−r)/(1+r) 의 역
        out = torch.zeros_like(colored)
        for i, (lo_f, hi_f, band) in enumerate(self._bands(colored, state)):
            if i == 0:
                out = out + band * asp_env.clamp_min(0.0)
                continue
            rho = self._rho(0.5 * (lo_f + hi_f))
            u = noise.white(f"aspu{i}", sample0, colored.shape[-1], colored.shape[0],
                            colored.dtype, colored.device)
            u, state[f"nu{i}"] = tv_biquad(u, 1.0 - rl, 0.0, 0.0, -rl, 0.0, zi=state.get(f"nu{i}"))
            # `û` 는 단위 분산(해석적), 깊이는 공통 AM 의 **공칭** rms 에서 온다 — 둘 다 청크와 무관하다.
            d_u = M_RMS_NOMINAL * math.sqrt(max(1.0 - rho, 0.0) / max(rho, 1e-6))
            fc = 0.5 * (lo_f + hi_f)
            env = slow * (1.0 + self._beta(fc) * m + d_u * (u * u_gain))
            out = out + band * env.clamp_min(0.0) * self._band_gain(fc)
        return out

    def _am_split(self, env, state):
        """`env` 를 느린 부분과 F0 속도 부분으로 가른다. 돌려주는 값 (느림, 빠름). `ASP_AM_HI` 참조."""
        r = math.exp(-2 * math.pi * ASP_AM_SLOW_HZ / self.fs)
        slow, state["amlp"] = tv_biquad(env, 1.0 - r, 0.0, 0.0, -r, 0.0, zi=state.get("amlp"))
        return slow, env - slow

    def forward(self, asp_env: torch.Tensor, noise=None, sample0: int = 0,
                state: dict | None = None, voiced: torch.Tensor | None = None,
                corner_t: torch.Tensor | None = None) -> dict:
        b, n = asp_env.shape
        noise = noise or NoiseBank()
        state = {} if state is None else state
        white = noise.white("asp", sample0, n, b, asp_env.dtype, asp_env.device)
        if ASP_SPLIT and voiced is not None:
            g = asp_env * torch.exp(self.log_amp)
            v = voiced[:, :n].to(white.dtype).clamp(0.0, 1.0)
            r_u = (torch.exp(-2 * math.pi * corner_t / self.fs) if corner_t is not None
                   else math.exp(-2 * math.pi * self.corner / self.fs))
            hu, state["hu"] = tv_biquad(white, 0.5 * (1 + r_u), -0.5 * (1 + r_u), 0.0, -r_u, 0.0,
                                        zi=state.get("hu"))
            col_u = self.floor * white + (1.0 - self.floor) * hu
            r_v = torch.exp(-2 * math.pi * self.mvf() / self.fs)
            hv, state["hv1"] = tv_biquad(white, 0.5 * (1 + r_v), -0.5 * (1 + r_v), 0.0, -r_v, 0.0,
                                         zi=state.get("hv1"))
            hv, state["hv2"] = tv_biquad(hv, 0.5 * (1 + r_v), -0.5 * (1 + r_v), 0.0, -r_v, 0.0,
                                         zi=state.get("hv2"))
            col_v = ASP_V_FLOOR * white + (1.0 - ASP_V_FLOOR) * hv
            col_v, col_u, state = _asp_rolloff(col_v, col_u, self.fs, state)
            if ASP_SPLIT_BANDS and NOISE_BANDS > 0:
                # 유성 기식만 대역별 공통/독립으로 가른다 (ρ). `g` 에 이미 `asp_env` 가 곱해져 있으므로
                # 포락으로는 `asp_env·v` 를 넘기고 이득만 따로 곱한다.
                amp = torch.exp(self.log_amp)
                src_v = self._partial_coherence(col_v, asp_env * v, noise, sample0, state) * amp
            else:
                src_v = col_v * g * v
            src_u = col_u * g * (1.0 - v)
            return dict(source=src_v + src_u, source_v=src_v, source_u=src_u, state=state)
        # 1 차 고역 셸프: floor + (1−floor)·HP(corner)
        r = (torch.exp(-2 * math.pi * corner_t / self.fs) if corner_t is not None
             else math.exp(-2 * math.pi * self.corner / self.fs))
        hp, state["hp"] = tv_biquad(white, 0.5 * (1 + r), -0.5 * (1 + r), 0.0, -r, 0.0,
                                    zi=state.get("hp"))
        if int(getattr(self, "order", 1)) >= 2:
            hp, state["hp2"] = tv_biquad(hp, 0.5 * (1 + r), -0.5 * (1 + r), 0.0, -r, 0.0,
                                         zi=state.get("hp2"))
        colored = self.floor * white + (1.0 - self.floor) * hp
        colored, _, state = _asp_rolloff(colored, None, self.fs, state)
        g = torch.exp(self.log_amp)
        if NOISE_BANDS <= 0:
            return dict(source=colored * asp_env * g, state=state)
        return dict(source=self._partial_coherence(colored, asp_env, noise, sample0, state) * g,
                    state=state)


# ------------------------------------------------------------ 과도음 템플릿
def _damped_burst(fs, f_hz, tau_ms, ms, noise=0.0, seed=0):
    n = int(fs * ms / 1000.0)
    t = np.arange(n) / fs
    g = np.random.default_rng(seed)
    x = np.exp(-t / (tau_ms / 1000.0)) * (np.sin(2 * np.pi * f_hz * t)
                                         + noise * g.standard_normal(n))
    return x / (np.abs(x).max() + 1e-12)


def default_templates(fs: float) -> dict[str, np.ndarray]:
    """물리적으로 그럴듯한 초기 템플릿. 학습/녹음으로 교체·추가한다."""
    g = np.random.default_rng(1)
    saliva = np.zeros(int(fs * 0.06))
    for k in g.integers(0, len(saliva) - 200, 7):
        burst = _damped_burst(fs, g.uniform(3500, 7000), 0.4, 3, 0.5, int(k))
        saliva[k:k + len(burst)] += burst * g.uniform(0.3, 1.0)
    return {
        "tongue_contact": _damped_burst(fs, 4200.0, 0.6, 4, 0.8, 2),   # 혀끝-구개 접촉 클릭
        "tongue_release": _damped_burst(fs, 2800.0, 1.2, 8, 1.5, 3),   # 접촉 해제(짧은 마찰 버스트)
        "tongue_floor":   _damped_burst(fs, 1600.0, 1.5, 8, 1.0, 4),   # 튕긴 혀가 입 바닥에 닿음
        "lip_smack":      _damped_burst(fs, 900.0, 2.5, 12, 0.6, 5),
        "saliva":         saliva / (np.abs(saliva).max() + 1e-12),
    }


class TransientTemplateBank(nn.Module):
    """템플릿 뱅크 + 변조. 이벤트 {t, id|name, amp, rate, tilt} -> 파형 (B,N)."""

    def __init__(self, fs: float, templates: dict[str, np.ndarray] | None = None,
                 max_len_ms: float = 80.0):
        super().__init__()
        self.fs = float(fs)
        tpl = templates or default_templates(fs)
        L = int(fs * max_len_ms / 1000.0)
        self.names = list(tpl.keys())
        bank = torch.zeros(len(tpl), L)
        for i, k in enumerate(self.names):
            v = torch.as_tensor(tpl[k], dtype=torch.float32)[:L]
            bank[i, :len(v)] = v
        self.bank = nn.Parameter(bank)                 # 학습 가능
        self.index = {k: i for i, k in enumerate(self.names)}

    def add(self, name: str, wave: np.ndarray) -> int:
        """새 템플릿(예: 녹음에서 오려낸 것). 학습 중 동적으로 늘어날 수 있다."""
        L = self.bank.shape[1]
        v = torch.zeros(L); w = torch.as_tensor(wave, dtype=torch.float32)[:L]
        v[:len(w)] = w
        with torch.no_grad():
            self.bank = nn.Parameter(torch.cat([self.bank.data, v.unsqueeze(0)], 0))
        self.names.append(name); self.index[name] = len(self.names) - 1
        return self.index[name]

    def render(self, events: list[dict], n: int, b: int = 1, noise=None,
               device=None, sample0: int = 0) -> torch.Tensor:
        """events: [{t: 초(청크 기준), name|id, amp(1.0), rate(1.0), tilt(0.0)}]. 반환 (B,N).

        변조 난수는 이벤트의 **절대 샘플 위치**로 시드되므로 청크 경계와 무관하다.
        """
        out = torch.zeros(b, n, device=device, dtype=self.bank.dtype)
        noise = noise or NoiseBank()
        for ev in events:
            i = self.index[ev["name"]] if "name" in ev else int(ev.get("id", 0))
            i = min(max(i, 0), self.bank.shape[0] - 1)
            tpl = self.bank[i]
            s_abs = sample0 + int(round(float(ev["t"]) * self.fs))
            # 변조: 속도(피치) 재표본화 + 미세 난수(같은 소리가 두 번 나지 않게)
            rate = float(ev.get("rate", 1.0)) * (1.0 + 0.05 * noise.scalar("transient", 2 * s_abs))
            amp = float(ev.get("amp", 1.0)) * 10 ** (0.1 * noise.scalar("transient", 2 * s_abs + 1) / 20)
            L = tpl.shape[0]
            m = max(8, int(L / rate))
            src = torch.arange(m, device=tpl.device, dtype=tpl.dtype) * rate
            i0 = src.floor().long().clamp(0, L - 2)
            w = src - i0
            y = tpl[i0] * (1 - w) + tpl[i0 + 1] * w
            tilt = float(ev.get("tilt", 0.0))
            if tilt != 0.0:                            # +: 밝게(미분), −: 어둡게(적분 근사)
                d = torch.cat([y[:1], y[1:] - y[:-1]])
                y = y + tilt * d * (4.0 if tilt > 0 else 0.5)
            s = int(round(float(ev["t"]) * self.fs))
            if s >= n or s + m <= 0:
                continue
            a, bnd = max(s, 0), min(n, s + m)
            out[:, a:bnd] = out[:, a:bnd] + amp * y[a - s: bnd - s]
        return out
