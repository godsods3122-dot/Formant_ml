"""압력 구동 성문 소스 — "성대와 주변부의 압력을 통하여 원음을 생성".

입력은 근육·호흡 좌표다: 성문하압 `p_sub`, 내전 `adduction`, 긴장 `tension`.
F0·진폭·Rd(음질)·기식은 **결과**다. 스크립트는 F0 를 직접 줄 수도 있다(`f0_target`).

물리 (Titze 1988 소진폭 이론의 요약):

    발성 역치압  Pth = (1.5 + 0.7·(f0/f0n)²) · (0.6 + 2.4·(1−add)²)      [cmH2O]
    진폭 목표    A*  = sqrt((Ps − Pth)⁺ / Pth) · gate(add)                 (Ps ≫ Pth 에서 포화)
    진폭 기동    dA/dt = σ·A·(1 − A/A*),  σ = k·f0·(Ps−Pth)/Pth            (로지스틱: 볼록 S 자)
    소멸         dA/dt = −(f0/3)·A                                          (역치 아래: 서너 주기 안에 죽음)
    F0           f0 = f0_base(tension)·(1 + 0.04·(Ps−Pth)⁺)·f0_scale·(1+J·z)
    Rd           Rd = 0.3 + 2.4·(1−add)^1.5 + rd_offset                    (압착 0.3 ~ 기식 2.7)
    기식 포락선  asp = (1−add)²·sqrt(Ps)·(1 + d·open_phase)                  (Klatt & Klatt 1990 AM)

파형은 LF 모델(Fant 1995, Rd 한 개)의 **하모닉 가산합성**이다. 위상은 샘플 단위로
누적되고, 나이퀴스트 근방은 소프트 마스크로 에일리어싱을 막는다. 하모닉 상대위상
(RPS, 화자 고유량)은 올패스 체인의 위상을 k·f0 에서 평가해 오프셋으로 넣는다.

v1 에서 검증된 사실을 그대로 가져왔다(docs/HANDOFF.md §6.0): 1 차 지연으로는
발성 개시 모양이 반대다(오목). 로지스틱이어야 한다(볼록 S). 그리고 씨앗이 남아
있지 않으면 되살아나지 않으므로 역치 아래에서는 씨앗까지 내려간다.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .control import frames_to_samples
from .rng import NoiseBank
from .tviir import tv_biquad

RHO = 1.14e-3            # 공기 밀도 g/cm³
CMH2O = 980.665          # 1 cmH2O = 980.665 dyn/cm²

#: **성문을 미는 것은 압력차다** (사용자 지적, MEASUREMENTS §52.344). 성문 진동의 구동은
#: 폐압 `p_sub` 가 아니라 **경성문 압력차** `p_sub − p_oral` 이다 — 마찰·파열 중에는 구강압이
#: 차올라 압력차가 줄고, 그 소리가 끝나면 구강압이 τ 6 ms 로 빠지며 압력차가 순간적으로 회복된다.
#:
#: 사용자: *"진짜 물리적인 내압 때문에 순간적으로 발생하는 거야."* 실측(L8 제어열에서 Po 를 재현,
#: 마찰이 끝난 뒤 40 ms 의 압력차 회복):
#:
#:   2.143 s  2.11 → 13.02 (**+10.91**)   2.366 s  1.63 → 8.80 (+7.17)   파열 1.767 s  +1.15
#:
#: 마찰이 파열보다 회복폭이 큰 것도 물리에 맞는다 — 마찰은 길어서 구강압이 충분히 차고 파열은 짧다.
#:
#: **이 물리는 이미 엔진에 있었다** — `noise.FricationNoise` 가 파열 버스트를 내려고 같은 Po 를
#: 적분하고 있었다. 그런데 성문은 그것을 몰라서 `over = p_sub − pth` 로 폐압만 봤다. 두 곳이 같은
#: Po 를 보게 잇는 것뿐이라 **새 파라미터가 늘지 않는다.**
#:
#: 0 이면 예전대로(폐압만), 1 이면 완전한 압력차. 실제 성문 유량은 Ag 를 지나며 압력을 나눠 가지므로
#: 1 이 상한이다.
ORAL_LOAD = 0.0

#: **구강압을 공기역학 상미분방정식으로** (MEASUREMENTS §52.436). 예전 식은 닫힘 τ 15 ms·열림 τ 6 ms 고정값이었다.
#:     C · dPo/dt = U_g − U_c − U_n,   C = V/(γ·P_atm) + C_w
#:     U_g = A_g √(2(Ps − Po)/ρ)   성문으로 들어옴 (베르누이)
#:     U_c = A_c·(1−velum) √(2 Po/ρ)   구강 협착으로 나감,   U_n = velum·A_vp √(2 Po/ρ)   코로 샘
#: 값 (Rothenberg 1968, *The breath-stream dynamics of simple-released-plosive production*): 벽 순응도 평균 ≈ 0.5 ml/cmH2O
#: (치조 0.38~0.53, 이완된 양순·볼 포함 4), 공동 부피 < 50 ml, 공기 압축 시상수 ≤ 4 ms, 벽 포함 40~400 ms. 여성 실측 /b/ (PMC2651765):
#: 폐쇄 75.5 ± 10.4 ms, Po 1.7 → 4.3 cmH2O, 폐쇄의 64 % 유성. 입이 열리면 시상수 → 0 이라 틀마다 **암시적**(뉴턴)으로 푼다.
#: 켜면 `ORAL_LOAD` = 1 로 성문이 Ps − Po 를 보고, 마찰 음원은 U_c 로 레이놀즈·세기를 잰다 — 파열 버스트가 따로 된 식 없이 나온다.
ORAL_ODE = False
#: 구강압 방정식의 빠른 경로 (numba + 수반 역전파, §52.468). 수치는 torch 판과 같다 — 끄면 torch 판.
ORAL_ODE_FAST = True
#: 성대 떨림 진폭 재귀의 빠른 경로 (numba + 수반 역전파, §52.468). 끄면 torch 판.
AMP_FAST = True
ORAL_V_CM3 = 40.0
ORAL_CW_ML = 0.5            # ml/cmH2O
ORAL_A_VP = 0.2             # 연구개 완전 개방의 인두-비강 틈 [cm²]


# ------------------------------------------------------------------ LF 모델
#: **추가 기울기(`tilt`)가 먹히는 상한 주파수** [Hz]. 0 이면 상한 없음(예전 거동).
#:
#: `tilt` 은 하모닉마다 10^(tilt·log2(f/1kHz)/20) 을 곱하는 **거듭제곱**이라, 상한이
#: 없으면 나이퀴스트까지 계속 오른다 — 적합값 +4.77 dB/oct 면 16 kHz 에서 +23 dB 다.
#: 그 결과 성문 유량미분이 **사각파처럼 날카로워진다.** 실측 (|Δdu|max / rms,
#: 유성 구간):
#:
#:     이상적 LF (tilt 0)      1.99
#:     적합값 tilt (+4.77)    **12.46**   <- 6.3 배
#:     tilt 상한 (+12)          37.03
#:
#: 물리적으로 성문 소스의 기울기는 귀환 위상(Ta)이 만드는 **1 차 저역통과**이지
#: 무한히 오르는 부스트가 아니다. 그래서 셸프로 만들어 어느 위에서 포화시킨다.
#:
#: A/B (같은 적합 트랙, 렌더만 다시):
#:
#:   | 상한 | du 날카로움 | 하모닉 스펙트럼 차 rms | 세로 얼룩 |
#:   |---|---|---|---|
#:   | 없음 | **12.47** | 1.11 dB | 1.520 |
#:   | 2 kHz | 2.56 | 3.07 | 1.525 |
#:   | 3 kHz | 2.54 | 1.81 | 1.521 |
#:   | **5 kHz** | **3.82** | **1.15** | 1.520 |
#:   | 8 kHz | 6.26 | 1.11 | 1.520 |
#:   | 12 kHz | 8.86 | 1.11 | 1.520 |
#:
#: 5 kHz 면 날카로움이 3.3 배 내려가는데 스펙트럼 대가가 0.04 dB 다 — 사실상 공짜다.
#: 3 kHz 는 더 부드럽지만 0.7 dB 를 문다.
#:
#: 왜 하필 그 언저리인가: 이 화자의 4~12 kHz 는 85 % 가 난류다(§11.1). 그 위에서
#: **하모닉 기울기를 더 올릴 대상 자체가 없다** — 올려 봐야 펄스만 날카로워진다.
TILT_MAX_HZ = 5000.0

#: **하모닉을 묶음으로 합성한다** (0 = 예전처럼 하나씩). 적합 한 회차의 **68.2 %**가
#: 아래 하모닉 루프였다 (`yang_00000040` 전체, 1374 ms 중 937 ms). 루프 상한이
#: `ceil(f_cut / f0_min)` 이라 발화 어딘가에 f0 95.6 Hz 가 한 번만 있어도 **하모닉 269 개를
#: 하나씩** 돌고, 회차마다 순방향·역전파로 수만 번의 작은 연산이 된다 — 시간이 계산이
#: 아니라 연산을 띄우는 비용으로 간다 (프로파일: `aten::mul` 72522 회, 평균 4.2 µs).
#:
#: 순차 누적을 둔 이유(아래 루프 주석)는 **실시간 경로의 스트리밍 = 오프라인 일치**다
#: — float32 축약 순서가 청크 길이에 따라 달라지면 고 Q 성도를 지나며 1e-3 로 커진다.
#: 오프라인 적합에는 그 제약이 없으므로 적합기만 이것을 켠다. 묶음 안의 합은 float64 로
#: 누적해 순서 민감도를 줄인다.
HARM_BLOCK = 0

#: **성문 분산 세기를 적합한다** (MEASUREMENTS §52.121). `voice.GLOTTAL_DISPERSION` 에 곱할
#: 전역 배율(e^{±DISP_LIM})을 손잡이로 푼다. 실측에서 최적 구간이 좁았다 — D=20·150 은 세로줄
#: 문지기 폐기, D=60 은 통과다. 이런 양을 상수로 박으면 발화마다 손으로 다시 맞춰야 한다.
#: `--dispersion D` 로 기준값을 준 뒤에만 뜻이 있다 (0 에 무엇을 곱해도 0 이다).
DISP_FIT = False
DISP_LIM = 0.9          # e^{0.9} = 2.46 배까지

#: **최대 유성 주파수(MVF)** — 조화+잡음 모형(Stylianou 1996; WORLD 의 비주기성과 같은
#: 생각)의 경계. 0 이면 끈다(배음을 나이퀴스트까지 만든다 — 예전 동작).
#:
#: 왜: 사용자가 스펙트로그램에서 "대충 6200 Hz 까지는 지문 같은 무늬가 있는 반면에 그 위로는
#: 필터 같은 경향만 있지 죄다 안개처럼 흩어져 있다" 고 보았다. 대역별 **파형** 주기성(F0 지연
#: 자기상관)을 재면 정확히 그렇다 (`yang_00000040` 유성 프레임):
#:
#:     kHz     1     2     3     4     5     6     7    8~15
#:     목표  0.67  0.42  0.28  0.16  0.11  0.02  0.01   ≈ 0
#:     L22   0.85  0.53  0.38  0.32  0.21  0.11 −0.02   ≈ 0
#:
#: 목표의 배음은 6 kHz 에서 끝나고 그 위는 **포락만** 성문 주기에 물린 잡음이다. 우리는
#: 배음을 끝까지 만들어 1~6 kHz 가 목표보다 20~100 % 날카롭다 ("하모닉을 너무 날카롭게
#: 깎았다"). 그래서 배음에 MVF 부드러운 저역통과를 걸고, 그 위는 성문 주기로 변조된
#: 기식 잡음이 **같은 성도**를 지나게 한다 — 여기원은 나누고 필터는 공유한다.
MVF_HZ = 0.0
MVF_WIDTH = 800.0

#: **배음별 위상 분산** [rad/kHz] — 고역 배음이 "지워지는" 게 아니라 **번져서** 안개가 되게
#: 한다. 0 이면 끈다.
#:
#: 사용자의 물리: *"고역에서는 임계 이상의 진동은 분산이 커져서 유사 노이즈로 붕괴되지만,
#: 포먼트는 살아있다."* 연화(점액·조직 감쇠)는 고역 배음의 **진폭**을 줄일 뿐 선은 남긴다.
#: 안개인데 포먼트가 사는 결은 **배음 위상이 차수에 따라 더 흔들려 선폭이 넓어지는** 쪽이다
#: — 선들이 겹쳐 연속체가 되고, 같은 필터를 지나니 포락은 그대로다.
#:
#: 목표에서 형태가 나온다. 대역별 파형 주기성의 1 kHz 대비 비가 2/3/4/5/6 kHz 에서
#: 0.63/0.42/0.24/0.16/0.03 이고, 가우시안 위상 흔들림이면 주기성이 exp(−σ²/2) 로 준다.
#: 역산하면 σ = 0.96/1.32/1.69/1.91/2.65 rad — **주파수에 거의 비례**한다 (≈0.4 rad/kHz).
#: 차수에 비례하는 분산, 즉 사용자가 말한 "임계 이상의 분산 증가" 다.
#:
#: 앞서 넣은 조화+잡음 분리(`MVF_HZ`)는 방향이 틀렸다 — MVF 위 배음을 **지우고** 따로 게이트한
#: 잡음을 얹으니 배음이 붕괴하는 게 아니라 갈아 끼워지고, 게이트가 세로 블록을 만든다.
#: 이것은 배음을 지우지 않는다. 위상만 흔드므로 **각 배음의 에너지가 보존**된다.
#:
#: `jitter`(공통 F0 흔들림)로는 안 된다 — 선폭이 k·f0·jitter 로만 넓어져 6 kHz(k≈20)에서 선을
#: 겹치게 하려면 지터가 5 %여야 하는데 상한이 1 %이고 적합값이 **이미 상한에 붙어 있다**
#: (0.0098) — 적합기가 더 흩뜨리고 싶어 하는데 수단이 없었다는 신호다. 여기서는 배음마다
#: **독립인** 흔들림을 쓴다.
#:
#: ψ_k 는 프레임률(1 kHz) 위치 기반 백색 난수를 선형 보간한 것 — 대역폭 ~500 Hz 라 선을
#: 이웃(f0 ≈ 300 Hz)과 겹칠 만큼 번지게 한다. **상태가 없으므로** 청크를 나눠도(실시간
#: 경로) 같은 값이 나온다. 세기 σ(f) = HARM_PHASE_JIT·max(0, f − HARM_PHASE_ONSET)/1000.
#: 화자의 성대 조직(점막파의 불규칙성)이 정하는 값이라 프로파일로 옮길 자리다.
HARM_PHASE_JIT = 0.0
HARM_PHASE_ONSET = 500.0
#: 난수 배치의 고정 폭 — 위치(프레임·배음)가 청크와 무관하게 정해진다. 배음 수가 이보다 많으면
#: 배음 번호를 이 폭으로 **감아** 쓴다(k % KMAX). 400 이던 때 적합이 F0 를 한 유성 프레임에서
#: 하한 가까이 내리자 배음 수가 400 을 넘어 위상 단계에서 IndexError 로 죽었다(out/L34/s101).
#: 512 면 F0 하한 50 Hz 에서도 24 kHz 까지 480 개라 감기지 않는다.
HARM_PHASE_KMAX = 512
#: 배음 위상 분산 세기의 **화자 전역 배율**을 적합한다 (MEASUREMENTS §51.12). 켜면 σ(f) 에
#: exp(0.9·tanh(x/0.9)) (0.4~2.5 배, 처음 1 배)를 곱하고, x 는 적합기의 전역 목록에 들어간다.
#: 목표의 대역별 주기성은 화자·녹음마다 다르다(s040 0.90/0.55/0.31/0.31/0.11, s101 0.84/0.41/0.21/0.17/0.13,
#: 1~5 kHz) — 고정 상수 0.25 rad/kHz 로는 s040 의 3.5~5.5 kHz 가 목표보다 덜 주기적이었다(0.11/0.04).
HJIT_FIT = False
#: **주기 동기 배음 위상 분산** (FOUNDATION §3.2, MEASUREMENTS §51.13). 켜면 ψ_k 를 1 ms 프레임 난수가 아니라
#: **성문 주기마다** 새로 뽑고(닫힌 구간 한가운데 — LF 위상 `HJIT_CYCLE_ANCHOR` — 를 주기 경계로), 주기 안에서는
#: 이웃 두 주기의 값을 선형으로 잇는다. 1 ms 난수는 한 주기 안을 네다섯 번 뒤섞어 고역 포락의 성문 주기 동기를
#: 지웠다: 적합기 로그의 "지글거림(유성)" F0 대역 변조가 목표의 0.53~0.59 배(`L52`) — 목표 고역 포락의 F0 동기는
#: 0.83~0.84 다. 주기마다 뽑으면 한 주기 안의 펄스 모양이 살아 있고 주기 사이만 흩어진다 — Hermes(1991)·
#: Mehta & Quatieri(2005)의 "펄스 동기 잡음이라야 목소리에 녹는다". 위상 난수의 주기 좌표는 기울기를 끊는다
#: (f0 로 잘못된 난수 기울기가 흐르지 않게). 오프라인 적합 전용이다 — 스트리밍에서는 주기 번호가 청크마다 새로 시작한다.
#: **화자 음원 EQ** (MEASUREMENTS §51.17). 켜면 성문 배음 소스에 고정 주파수의 봉우리 여섯을 곱한다 —
#: 이득만 화자 전역 적합값(±`SRC_EQ_MAX_DB`), 위치·폭은 고정. DC 이득은 1 이라 전체 수준은 안 건드린다.
#:
#: 왜: 지금 음원의 모양 자유도는 Rd(LF 한 모수)와 기울기 하나뿐이다. 문헌(A 층)은 음원을 기울기 **넷**
#: (H1–H2, H2–H4, H4–2 kHz, 2–5 kHz)으로 기술하고 그것이 화자 정체의 큰 몫이라고 한다(Kreiman·Garellek 2016).
#: 실측: 포락 오차의 85~93 %가 모음 0~1 kHz 인데(§51.2), 그 대역은 5.3 ms 창에서 배음 두셋만 들어가므로
#: **성문 펄스 모양 그 자체**다. 프레임별·대역별 최적 이득(89 %)과 프레임별 최적 정렬(+3.5 점)을 다 줘도
#: 남는 몫이 여기다. 시간에 따라 움직이지 않으므로 성도(시변)의 오차를 대신 흡수하지 못한다.
SRC_EQ = False
SRC_EQ_F = (150.0, 300.0, 600.0, 1200.0, 2400.0, 4800.0)
SRC_EQ_BW_REL = 0.7
SRC_EQ_MAX_DB = 9.0
#: **음원 고역 EQ** (MEASUREMENTS §52.410). 위 EQ 의 맨 위가 4.8 kHz(±9 dB)이고 기울기도 5 kHz 에서 멈춰(`TILT_MAX_HZ`),
#: 음원이 5 kHz 위 배음을 **구조적으로 못 올렸다.** 그 빈자리를 적합기가 주기마다 흔들리는 `hcorr` 로 메워 고역이
#: 흐릿해지고 16 kHz 위가 잡음이 됐다(C2 실측: `hcorr` 를 빼면 음원의 4~8 kHz 가 원본보다 14 dB 약하다). 화자 전역
#: 상수라 주기마다 안 흔들린다 — **주기적인** 고역을 낸다. 기존 EQ 와 따로 두어 옛 판의 `--init` 복원이 안 깨진다.
SRC_EQ_HF = False
SRC_EQ_HF_F = (7000.0, 12000.0, 17500.0)   # 17.5 kHz: 원본 유성 구간 17~20 kHz 배음이 우리보다 7 dB 세다 (§52.413)
SRC_EQ_HF_BW_REL = 0.8
SRC_EQ_HF_MAX_DB = 18.0

HJIT_CYCLE = False
HJIT_CYCLE_ANCHOR = 0.9

#: **배음 위상 흔들림 ψ 의 시간 상관** [ms] (MEASUREMENTS §52.265). 0 이면 지금까지처럼 프레임(1 ms)마다 독립이다.
#:
#: 왜: ψ 를 만드는 것은 점막파의 불규칙성인데, 성대 조직의 상태가 3 ms 주기마다 통째로 새로 뽑히지는 않는다.
#: 그리고 포락 값이 **같은 세기라도 빠르기에 따라 크게 다르다** — `scripts/diag/phasecost.py` 실측:
#: 20° 가 주기마다 무작위면 포락 84.00, 천천히 변하면 92.24 다(8 점). 우리 것은 1 kHz 난수라 한 주기(2.9 ms)를
#: 서너 번 뒤섞으므로 **가장 비싼 쪽**에 있다.
#:
#: 맞바꿈이 있다: 선을 번지게 하는 폭은 상관 시간에 반비례하므로 느리게 하면 선폭이 좁아진다(둘 다 로그에 찍힌다).
#: 세기 σ 는 보존한다 — 평활 커널로 줄어든 분산을 1/√Σh² 로 되돌린다.
HJIT_TAU_MS = 0.0


def _hjit_kernel(tau_ms: float, frame_ms: float, dtype, device):
    """시간 상관 커널 — 가우시안(σ = τ), **분산 보존**(1/√Σh²)해서 ψ 의 세기 σ 를 그대로 둔다."""
    import math as _m
    sig = max(float(tau_ms) / max(float(frame_ms), 1e-9), 1e-6)
    half = max(1, int(_m.ceil(3.0 * sig)))
    t = torch.arange(-half, half + 1, dtype=dtype, device=device)
    h = torch.exp(-0.5 * (t / sig) ** 2)
    h = h / h.sum()
    return h / torch.sqrt((h * h).sum())      # 평활로 줄어든 분산을 되돌린다

#: **성문 폐쇄 시각의 흩어짐** [s]. 성대는 부드러운 물질이라 성문 길이를 따라
#: **동시에 닫히지 않는다** — 앞뒤로 지퍼처럼 닫힌다. 길이 방향의 면적 요소들은
#: 병렬이므로 총 유량은 그 합이고, 따라서 유량미분은 이상적인 펄스를 **국소 폐쇄
#: 시각의 분포로 합성곱한 것**이 된다.
#:
#: 분포가 대칭이면 그 합성곱은 **영위상**이다 — 위상은 그대로 두고 크기만 깎는다.
#: 사용자가 지적한 "스무딩은 위상만 바꾸는 게 아니라 불필요한 고조파를 줄이는 것"이
#: 정확히 이 성질이다. 가우시안 분포(표준편차 σ)면 하모닉 진폭에 걸리는 배율이
#: exp(−(2πfσ)²/2) 다.
#:
#: 0 이면 항이 빠진다. **값은 A/B 로 정한다** — 실측에서 우리 유성 소스는 3~12 kHz
#: 하모닉이 오히려 2~5 dB **모자라고**(§15.2) 12~20 kHz 만 +1.4 dB 과하다. 그러니
#: 이 항은 "고역 전체를 깎는" 것이 아니라 **맨 위만 깎아** 적합기가 그 아래를 채울
#: 여지를 주는 쪽으로 써야 한다 (σ 가 작을수록 모서리가 높다).
GLOTTAL_CLOSURE_SPREAD_S = 0.0

#: 협착 전달비 T = (Lc+Lf)·Ac / (Lc·A0 + Lf·Ac) 의 기하 상수. **지금은 안 쓴다** —
#: 왜 안 쓰는지는 `physiology` 안의 주석과 docs/MEASUREMENTS.md §22 에 있다.
#: `ag_dc` 를 문헌 범위로 고치면 그때 같이 살릴 값이라 지우지 않고 둔다.
CONSTRICTION_LEN = 1.0      # 협착부(혀끝-치경 간극) 길이 [cm]
FRONT_CAVITY_LEN = 1.5      # 협착 앞쪽 공동(앞니까지) 길이 [cm]
NEUTRAL_TRACT_AREA = 3.0    # 협착 없는 중립 성도 단면적 [cm²]


#: **LF Rd 의 하한** (MEASUREMENTS §52.149). Fant(1995) 의 표준 범위는 0.3~2.7 이지만 그것은
#: 관례이지 물리가 아니다 — 매개화가 깨지는 곳은 `ra = (−1 + 4.8·Rd)/100 > 0`, 즉 **Rd > 0.2083**
#: 이다.
#:
#: 왜 내려야 하나: 적합된 제어열에서 `rd_offset` 이 **유성 프레임의 13.7 %에서 하한에 붙어 있고
#: 상한에는 0 %** 다 (분석 궤적도 같다). 즉 모형이 낼 수 있는 것보다 **더 날카로운 폐쇄**를
#: 원하는데 벽에 막혀 있다 — 사용자가 처음 지적한 *"펄스가 아주 짧고 강하게 나와야 하는데 죄다
#: 부드럽게 펴져 있다"* 가 이것이다.
#:
#: 하한을 내리면 (한 주기 정규화, 100 차 위 에너지 몫):
#:
#:   | Rd | 반치폭 | 고역 몫 |
#:   |---|---|---|
#:   | 0.30 | 0.0308 | −10.53 dB |
#:   | 0.25 | 0.0256 | **−8.08** |
#:   | 0.22 | 0.0225 | **−5.20** |
#:
#: 0.2084 아래는 `ta ≤ 0` 이라 적분이 성립하지 않으므로 코드가 막는다.
LF_RD_MIN = 0.3


def lf_pulse(rd: float, n: int = 4096) -> np.ndarray:
    """LF 유량미분 E(t) 한 주기 (Ee = 1 로 정규화). Fant(1995) 의 Rd 파라미터화."""
    rd = float(np.clip(rd, max(LF_RD_MIN, 0.212), 2.7))
    ra = (-1.0 + 4.8 * rd) / 100.0
    rk = (22.4 + 11.8 * rd) / 100.0
    rg = (rk / 4.0) * (0.5 + 1.2 * rk) / (0.11 * rd - ra * (0.5 + 1.2 * rk))
    tp = 1.0 / (2.0 * rg)
    te = tp * (1.0 + rk)
    ta = ra
    tc = 1.0
    # (i) ε·ta = 1 − exp(−ε(tc−te))  → 고정점
    eps = 1.0 / ta
    for _ in range(100):
        eps = (1.0 - math.exp(-eps * (tc - te))) / ta
    # (ii) ∫E = 0 → α 이분법
    wg = math.pi / tp
    t = (np.arange(n) + 0.5) / n

    def wave(alpha):
        e = np.zeros(n)
        m = t <= te
        e[m] = -np.exp(alpha * (t[m] - te)) * np.sin(wg * t[m]) / math.sin(wg * te)
        r = ~m
        e[r] = -(1.0 / (eps * ta)) * (np.exp(-eps * (t[r] - te)) - math.exp(-eps * (tc - te)))
        return e

    # α 가 클수록 개방기 후반(음의 부분)이 커져 ∫E 가 줄어든다 → ∫E > 0 이면 α 를 키운다.
    lo, hi = -50.0, 400.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if wave(mid).sum() > 0:
            lo = mid
        else:
            hi = mid
    e = wave(0.5 * (lo + hi))
    return e / (np.abs(e).max() + 1e-12)


#: **성구 축 — Rd 격자를 2.7 위로 잇는 머리 성구(가성) 가지** (MEASUREMENTS §52.419). 사용자: *"삑사리 같은 세세한 것들을
#: 구현하기 위해서 내가 성대 밴드 구조를 프로파일링한 건데, 잘 연결된 건지도 모르겠고"* — 연결돼 있지 않았다. C3 의 삑사리(1.111 s)는
#: **한 주기 안에** 주기가 141 → 128 표본으로 뛰고 H1–H2 가 6 → 22 dB, 4 kHz 위가 −5 dB 로 바뀐다(성구 전환). LF 는 Rd 2.7 에서
#: 음원 H1–H2 가 19.3 dB 로 끝나는데 목표는 성도를 빼면 26~37 dB 다 — 그 몫을 주기별 배음 보정이 흉내 냈고, 전환 주기에 고역을
#: +12 dB 터뜨렸다(보정을 끄면 H1–H2 가 7~9 dB 에 머문다 — 모형 자체는 성구를 모른다).
#:
#: 가성은 얇은 덮개만 떨고 성문이 다 닫히지 않아 유량이 **정현에 가깝다**(개방지수 → 1). 그래서 Rd > 2.7 에서는 LF(2.7) 펄스를
#: 폐쇄 시각 te 를 맞춘 정현 유량미분 쪽으로 섞는다: w = (Rd − 2.7)/(REGISTER_RD_MAX − 2.7). w = 1 이면 H1 만 남는다.
#: 2.7 이하에서는 LF 그대로라 이 값을 2.7 로 두면(기본) 아무것도 안 바뀐다.
REGISTER_RD_MAX = 2.7
LF_RD_TOP = 2.7


def rd_max() -> float:
    return max(float(REGISTER_RD_MAX), LF_RD_TOP)


def glottal_pulse(rd: float, n: int = 4096) -> np.ndarray:
    """성문 유량미분 한 주기 — Rd ≤ 2.7 은 LF, 그 위는 머리 성구 가지 (`REGISTER_RD_MAX`)."""
    rd = float(rd)
    if rd <= LF_RD_TOP or rd_max() <= LF_RD_TOP:
        return lf_pulse(rd, n)
    e = lf_pulse(LF_RD_TOP, n)
    w = float(np.clip((rd - LF_RD_TOP) / (rd_max() - LF_RD_TOP), 0.0, 1.0))
    te = (float(np.argmin(e)) + 0.5) / n
    t = (np.arange(n) + 0.5) / n
    s_ = np.sin(2.0 * np.pi * (t - te + 0.75))          # 음의 봉우리가 te 에 온다
    y = (1.0 - w) * e + w * s_
    return y / (np.abs(y).max() + 1e-12)


# **성문 개방기에 동기한 기식 AM 의 깊이** (Klatt & Klatt 1990).
#
# `asp_env = asp·(1 + d·voiced·(open_phase − 0.5))` 이므로 d=0.7 이면 기식 잡음이 매
# 성문 주기마다 ±35 % 변조된다. 그 변조는 F0 와 그 배음에 그대로 실린다.
#
# 실측 (yang_00000101, 고역 4~12 kHz 포락의 변조 스펙트럼 **절대** 에너지, 목표 대비):
# F0~2.5F0 대역이 2.37 배, 60~150 Hz 가 4.01 배 과다하다. 사람은 그 대역의 진폭 변조를
# 거칠기(roughness)로 듣는다 — 사용자가 "지지직거린다" 고 한 성분이다.
#
# 값은 실측으로 정해야 하므로 상수로 빼 둔다. 하드코딩된 채로는 A/B 를 못 돌린다.
ASP_AM_DEPTH = 0.7

#: **성문 제트의 레이놀즈 수로 기식을 낸다** (MEASUREMENTS §52.417). 사용자: *"성대 방해물이 레이놀즈에 영향을 주는 건
#: 고려했나? 동적으로 압력 등등이 변할 텐데?"* 지금까지 기식은 Klatt 의 경험식 `(1−add)²·√Ps·(1 + d·open_phase)` 였다 —
#: 성문 면적이 주기 안에서 열리고 닫히는 것도, 그 순간의 성문 양단 압력도 모른다. 마찰(협착)은 이미 `noise.reynolds`
#: 로 Re > Re_c 에서만 켜지는데 성문만 빠져 있었다.
#:
#: 켜면 표본마다
#:     A(t)   = ag_dc + GLOT_A_PEAK · amp · voiced · flow(φ, Rd)      성문 면적 [cm²] (flow 는 같은 LF 펄스의 유량, 봉우리 1)
#:     ΔPg(t) = Ps · a_c² / (a_c² + A(t)²)                             성문 몫의 압력 강하 (구강 협착과 나눔, 표본마다)
#:     v      = √(2 ΔPg / ρ),   d_h = 2A / GLOT_LEN_CM (틈의 수력 직경),   Re = v · d_h / ν
#:     원천   ∝ g(Re) · v³ · √A,   g = (softplus(Re² − Re_c²) / Re²)^1.5      (Stevens 1998 — 쌍극 원천 v³√A, 문턱 Re_c)
#: 성대가 닫히는 쪽으로 가면 틈이 좁아져 Re 가 문턱 밑으로 내려가 난류가 꺼지고, 열리면 켜진다 — 주기 안의 잡음
#: 모양이 경험 AM 이 아니라 면적·압력에서 나온다. 내전이 크면 닫힌 구간의 새는 틈이 좁아 거의 조용하고, 벌린
#: 성문(속삭임)은 늘 문턱 위다. `aspiration` 은 그대로 전체 배율로 남는다.
GLOTTAL_RE = False
GLOT_LEN_CM = 1.0         # 여성 막성 성대 길이 ~1.0 cm (Titze 1994)
GLOT_A_PEAK = 0.12        # 모달 발성의 최대 성문 면적 [cm²] (여성 0.1~0.15)
GLOT_RE_CRIT = 1800.0     # noise.RE_CRIT 와 같다
GLOT_RE_W = 0.2           # 문턱의 매끈함 (noise.FRIC_SOFT_W 와 같은 꼴)
GLOT_NU = 0.15            # 공기 동점성 cm²/s
GLOT_V_REF = (2.0 * 8.0 * 980.665 / 1.14e-3) ** 0.5    # 8 cmH2O 의 베르누이 속도 — 세기 정규화
GLOT_A_REF = 0.1
#: 옛 식과 같은 자리(add 0.6, Ps 8, 협착 없음)에서 주기 평균이 같아지는 배율 (§52.417 에서 잼).
GLOT_RE_GAIN = 0.67

#: **LF 잡음 포락의 성문 주기 변조 깊이** (`noise_modulation="lf"` 경로, MEASUREMENTS §52.147).
#:
#: `lf_noise_envelope` 가 `1 + d·voiced·flow` 로 기식·마찰 잡음을 성문 주기에 물린다. 예전에는
#: 이 `d` 가 **0.7 로 하드코딩**돼 있어 `--asp-am` 이 이 경로에 닿지 않았다(렌더 A/B 에서 0.3/0.7/1.0
#: 의 차이가 0.02 dB).
#:
#: 실측(음절, 6~20 kHz 포락의 변조 스펙트럼): 목표는 **F0 율(300~400 Hz)에 봉우리**가 있는데
#: (30.76 dB, 200~300 Hz 의 28.40 보다 높다) 우리는 그 율에서 **2.8~6.0 dB 모자라고** 대신
#: 20~300 Hz 에서 4~8 dB 과하다. 즉 고역이 성문 주기로 뛰지 않고 제어 흔들림으로 출렁인다.
LF_AM_DEPTH = 0.7

#: **유성 마스크를 떨림 진폭에 비례시킨다** [기준 진폭]. 0 이면 예전 이진 마스크
#: `amp > 1e-3` 그대로다.
#:
#: 이진 마스크는 성대가 조금이라도 떨면 성문 주기 AM(기식 `ASP_AM_DEPTH`, 마찰
#: `noise.FRIC_AM_DEPTH`)을 **전 깊이로** 건다. 실측(out/L25/s040, ㅊ 0.71~0.82 s): 떨림
#: 진폭 중앙 0.37 (모음 0.86), 최소 0.15 인데 마스크는 100 % 켜져 있었다 — 무성 치찰음의
#: 난류가 F0 로 전 깊이 썰렸다(MEASUREMENTS §50.4). 또 이진이라 "덜 떨면 덜 썰린다" 는
#: 기울기가 없다. >0 이면 `clamp(amp / VOICED_SOFT, 0, 1)` — 모음 떨림(≈0.8)에서 1,
#: 반쯤 떨면 절반 깊이다.
VOICED_SOFT = 0.0


def lf_table(n_rd: int = 24, rd_min: float | None = None, rd_max: float | None = None,
             n_harm: int = 400, n: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    """Rd 격자 -> 하모닉 복소계수 (n_rd, n_harm). 격자 사이는 선형보간(미분가능).

    `rd_max` 를 안 주면 성구 가지까지 (`REGISTER_RD_MAX`) — 그때 격자 눈금이 LF 구간과 같도록 칸 수를 늘린다."""
    lo = max(LF_RD_MIN, 0.212) if rd_min is None else rd_min
    if rd_max is None:
        rd_max = globals()["rd_max"]()
        if rd_max > LF_RD_TOP:
            n_rd = int(round(n_rd * (rd_max - lo) / (LF_RD_TOP - lo)))
    rds = torch.linspace(lo, rd_max, n_rd)
    coef = np.stack([np.fft.rfft(glottal_pulse(float(r), n))[1:n_harm + 1] / n
                     for r in rds])
    return rds, torch.tensor(coef, dtype=torch.complex64)


# ------------------------------------------------------------------ 소스
class GlottalSource(nn.Module):
    """근육·압력 좌표 -> (성문 유량미분, 순시위상, 기식 포락선, 성문 면적)."""

    def __init__(self, fs: float, hop: int, speaker: str = "female",
                 n_rd: int = 24, n_harm: int | None = None, k_growth: float = 0.25,
                 cycles_decay: float = 3.0, f0_min: float = 50.0,
                 f0_range: tuple[float, float, float] | None = None,
                 noise_modulation: str = "legacy", flow_reference: bool = False):
        super().__init__()
        if noise_modulation not in ("legacy", "lf"):
            raise ValueError("noise_modulation must be 'legacy' or 'lf'")
        self.noise_modulation = noise_modulation
        self.flow_reference = flow_reference
        self.fs, self.hop = float(fs), int(hop)
        self.k_growth, self.cycles_decay = k_growth, cycles_decay
        if f0_range is not None:
            self.f0_lo, self.f0_hi, self.f0_nom = map(float, f0_range)
        elif speaker == "female":
            self.f0_lo, self.f0_hi, self.f0_nom = 110.0, 440.0, 220.0
        else:
            self.f0_lo, self.f0_hi, self.f0_nom = 65.0, 260.0, 120.0
        n_harm = n_harm or int(fs / 2 / f0_min) + 1
        rds, coef = lf_table(n_rd, n_harm=n_harm)
        self.register_buffer("rd_grid", rds)
        self.register_buffer("lf_coef", coef)                      # (n_rd, K)
        self.register_buffer("k_idx", torch.arange(1, n_harm + 1, dtype=torch.float32))
        self.hjit_log = nn.Parameter(torch.tensor(0.0))       # 배음 위상 분산 배율 (HJIT_FIT)
        self.disp_log = nn.Parameter(torch.tensor(0.0))       # 성문 분산 배율 (DISP_FIT)
        self.src_eq_db = nn.Parameter(torch.zeros(len(SRC_EQ_F)))   # 화자 음원 EQ (SRC_EQ)
        self.src_eq_hf_db = nn.Parameter(torch.zeros(len(SRC_EQ_HF_F)))   # 음원 고역 EQ (SRC_EQ_HF)
        if noise_modulation == "lf" or flow_reference:
            # Integrate the SAME LF derivative used for the harmonic source, from
            # opening (phase zero) to closure. No independent open-quotient fit.
            flows = []
            peaks = []
            for r in rds:
                e = glottal_pulse(float(r))
                flow = np.concatenate(([0.0], np.cumsum(e)[:-1]))
                peaks.append(flow.max() / len(e))
                flow /= flow.max()
                flows.append(flow - flow.mean())
            # Derived table, not checkpoint state: legacy checkpoints still load.
            self.register_buffer("lf_flow", torch.tensor(np.stack(flows), dtype=torch.float32),
                                 persistent=False)
            self.register_buffer("lf_flow_peak", torch.tensor(peaks, dtype=torch.float32),
                                 persistent=False)

    def _src_eq(self, x, state):
        """성문 배음 소스에 거는 화자 전역 EQ (고정 주파수 봉우리 여섯, 이득만 적합). `SRC_EQ` 참조."""
        from .tviir import peak_coeffs
        for i, f in enumerate(SRC_EQ_F):
            gdb = SRC_EQ_MAX_DB * torch.tanh(self.src_eq_db[i] / SRC_EQ_MAX_DB)
            zr = torch.pow(10.0, gdb / 20.0).to(x.dtype)
            fq = torch.full_like(x, float(f))
            bw = torch.full_like(x, float(f) * SRC_EQ_BW_REL)
            key = f"seq{i}"
            x, state[key] = tv_biquad(x, *peak_coeffs(fq, bw, self.fs, zr), zi=state.get(key))
        return x

    def _src_eq_hf(self, x, state):
        """음원 고역 EQ (7·12 kHz 봉우리 둘, 이득만 적합). `SRC_EQ_HF` 참조."""
        from .tviir import peak_coeffs
        for i, f in enumerate(SRC_EQ_HF_F):
            gdb = SRC_EQ_HF_MAX_DB * torch.tanh(self.src_eq_hf_db[i] / SRC_EQ_HF_MAX_DB)
            zr = torch.pow(10.0, gdb / 20.0).to(x.dtype)
            fq = torch.full_like(x, float(f))
            bw = torch.full_like(x, float(f) * SRC_EQ_HF_BW_REL)
            key = f"seqh{i}"
            x, state[key] = tv_biquad(x, *peak_coeffs(fq, bw, self.fs, zr), zi=state.get(key))
        return x

    def _hjit(self):
        """배음 위상 분산 기울기 [rad/kHz]. HJIT_FIT 이면 전역 배율(0.4~2.5)을 곱한다."""
        if HJIT_FIT:
            return HARM_PHASE_JIT * torch.exp(0.9 * torch.tanh(self.hjit_log / 0.9))
        return HARM_PHASE_JIT

    # ---------------------------------------------------------- 생리 상태
    def threshold(self, f0, adduction):
        return (1.5 + 0.7 * (f0 / self.f0_nom) ** 2) * (0.6 + 2.4 * (1.0 - adduction) ** 2)

    def f0_base(self, tension):
        return self.f0_lo * (self.f0_hi / self.f0_lo) ** tension.clamp(0.0, 1.0)

    def oral_pressure(self, c: dict, ag_dc: torch.Tensor,
                      po0: torch.Tensor | None = None) -> torch.Tensor:
        """**구강압** Po 의 프레임률 동역학 [cmH2O]. `ORAL_LOAD` 참조.

        `noise.FricationNoise` 가 파열 버스트를 내려고 이미 쓰던 식·시상수를 그대로 쓴다 —
        닫힘(a_c < ~0.05)이면 Po → Ps (τ 15 ms), 열림이면 정상값 Ps·Ag²/(Ag²+Ac²) 로 (τ 6 ms).
        **여기서 한 번만 계산해 성문과 마찰이 같은 값을 보게 한다** (예전에는 마찰만 알고 있었다).
        """
        if ORAL_ODE:
            return self._oral_ode(c, ag_dc, po0)
        ps, a_c = c["p_sub"], c["a_c"].clamp_min(1e-3)
        oral = (1.0 - c["velum"]).clamp(0.0, 1.0)
        closed = torch.clamp((0.06 - c["a_c"]) / 0.04, 0.0, 1.0)
        closed = closed * closed * (3 - 2 * closed)
        po_ss = ps * ag_dc ** 2 / (ag_dc ** 2 + a_c ** 2)
        dt = self.hop / self.fs
        po = torch.zeros_like(ps[:, 0]) if po0 is None else po0.clone()
        out = []
        for i in range(ps.shape[1]):
            tgt = (closed[:, i] * ps[:, i] + (1 - closed[:, i]) * po_ss[:, i]) * oral[:, i]
            tau = 0.015 * closed[:, i] + 0.006 * (1 - closed[:, i])
            po = po + dt * (tgt - po) / tau
            out.append(po)
        return torch.stack(out, 1)

    def _oral_ode(self, c: dict, ag_dc: torch.Tensor, po0: torch.Tensor | None = None) -> torch.Tensor:
        """`ORAL_ODE` — 구강압 [cmH2O] (B,T). 협착 유량 U_c [cm³/s] 는 `self._last_uc` 에 남긴다.

        `ORAL_ODE_FAST` 면 numba 순전파 + 수반 역전파 (`engine/oral_ode_fast.py`, §52.468) — 같은 식·같은 뉴턴 8 회. 아니면 아래 torch 판."""
        from . import oral_ode_fast as _of
        if ORAL_ODE_FAST and _of.HAVE_FAST:
            ps = c["p_sub"].clamp_min(0.0) * CMH2O
            vel = c["velum"].clamp(0.0, 1.0)
            k_c = c["a_c"].clamp_min(0.0) * (1.0 - vel) * math.sqrt(2.0 / RHO)
            k_n = vel * ORAL_A_VP * math.sqrt(2.0 / RHO)
            k_g = ag_dc * math.sqrt(2.0 / RHO)
            C = ORAL_V_CM3 / (1.4 * 1.01325e6) + ORAL_CW_ML / CMH2O
            x0 = (torch.zeros_like(ps[:, 0]) if po0 is None else torch.sqrt((po0 * CMH2O).clamp_min(0.0))).detach()
            x = _of.oral_ode_x(ps, k_c + k_n, k_g, x0, C, self.hop / self.fs)
            self._last_uc = k_c * x
            return x * x / CMH2O
        return self._oral_ode_torch(c, ag_dc, po0)

    def _oral_ode_torch(self, c: dict, ag_dc: torch.Tensor, po0: torch.Tensor | None = None) -> torch.Tensor:
        """`_oral_ode` 의 torch 판 — 기준 구현 (빠른 경로의 시험이 대조한다)."""
        ps = c["p_sub"].clamp_min(0.0) * CMH2O
        vel = c["velum"].clamp(0.0, 1.0)
        k_c = c["a_c"].clamp_min(0.0) * (1.0 - vel) * math.sqrt(2.0 / RHO)      # U_c = k_c·√Po
        k_n = vel * ORAL_A_VP * math.sqrt(2.0 / RHO)
        k_g = ag_dc * math.sqrt(2.0 / RHO)
        C = ORAL_V_CM3 / (1.4 * 1.01325e6) + ORAL_CW_ML / CMH2O                    # cm⁵/dyn
        dt = self.hop / self.fs
        x = torch.zeros_like(ps[:, 0]) if po0 is None else torch.sqrt((po0 * CMH2O).clamp_min(0.0))
        out, uc = [], []
        for i in range(ps.shape[1]):
            P, ko, kg = ps[:, i], k_c[:, i] + k_n[:, i], k_g[:, i]
            x_prev2 = x * x
            # 풀 식 (x = √Po):  C(x² − x_prev²)/dt − kg·√(P − x²) + ko·x = 0 — x 에 단조 증가, 뉴턴 8 회
            x = torch.minimum(x, torch.sqrt(P + 1e-9))
            for _ in range(8):
                r = (P - x * x).clamp_min(1e-9)
                f = C * (x * x - x_prev2) / dt - kg * torch.sqrt(r) + ko * x
                df = 2 * C * x / dt + kg * x / torch.sqrt(r) + ko
                x = (x - f / df.clamp_min(1e-12)).clamp_min(0.0)
                x = torch.minimum(x, torch.sqrt(P + 1e-9))
            out.append(x * x / CMH2O)
            uc.append(k_c[:, i] * x)
        self._last_uc = torch.stack(uc, 1)
        return torch.stack(out, 1)

    def physiology(self, c: dict, amp0: torch.Tensor | None = None,
                   po0: torch.Tensor | None = None) -> dict:
        """프레임률 제어 (B,T) dict -> 프레임률 상태 dict. 로지스틱 기동은 프레임 루프.

        amp0: 이전 청크 끝의 (씨앗 포함) 진폭. 없으면 씨앗에서 시작."""
        ps, add, ten = c["p_sub"], c["adduction"].clamp(0, 1), c["tension"]
        direct = c["f0_target"] > 0
        f0 = torch.where(direct, c["f0_target"], self.f0_base(ten))
        pth = self.threshold(f0, add)
        ag_dc = 0.02 + 0.5 * (1.0 - add) ** 2.5                    # 정적 성문 면적 cm² (모달 ≈0.07 → U≈250 cm³/s)
        # **성문을 미는 것은 성문 양단의 압력차다** — 폐압이 아니라 `p_sub − p_oral` 이다 (§52.344).
        # 마찰·파열 중에는 구강압이 차올라 압력차가 줄고, 그 소리가 끝나면 τ 6 ms 로 빠지면서
        # 압력차가 **순간적으로 회복된다.** 실측(L8 제어열에서 Po 를 재현): 마찰이 끝난 뒤 40 ms 에
        # 압력차가 2.11 → 13.02 cmH2O 로 **10.91** 회복한다. 그 물리가 지금껏 마찰 소스에만 있었고
        # 성문은 몰랐다. 0 이면 예전처럼 폐압만 본다.
        po = self.oral_pressure(c, ag_dc, po0) if ORAL_LOAD > 0.0 else None
        over = (ps - (ORAL_LOAD * po if po is not None else 0.0) - pth).clamp_min(0.0)
        # 폐압-F0 결합은 **긴장으로 F0 를 정할 때만** 건다. `f0_target` 은 "이 주파수로
        # 울려라" 는 직접 지정이므로 그 위에 다시 곱하면 지정한 값이 안 나온다. 실제로
        # p_sub 6.7 · Pth 2.2 에서 ×1.18 이 걸려 238 Hz 지정이 281.6 Hz 로 났고(측정),
        # 복사합성에서 성문 펄스가 주기마다 0.18 주기씩 밀렸다.
        f0 = torch.where(direct, f0, f0 * (1.0 + 0.04 * over)) * c["f0_scale"]
        gate = torch.clamp((add - 0.10) / 0.15, 0.0, 1.0)
        gate = gate * gate * (3 - 2 * gate)                        # 벌린 성문(add≤0.10)은 안 떤다
        a_star = torch.sqrt(over / pth + 1e-12) * gate   # +eps: sqrt(0) 의 기울기가 무한대다
        a_star = a_star / (1.0 + 0.5 * a_star)                     # 포화
        dt = self.hop / self.fs
        seed = 0.02
        from . import amp_fast as _af
        if AMP_FAST and _af.HAVE_FAST:
            # 같은 닫힌 해를 numba 로 (§52.468) — 아래 torch 반복이 기준 구현이다.
            a0 = (torch.full_like(ps[:, 0], seed) if amp0 is None else amp0).detach()
            grow_all = self.k_growth * f0 * over / pth
            dfac_all = torch.exp(-dt * f0 / self.cycles_decay)
            amp_raw = _af.logistic_amp(a_star, grow_all, dfac_all, a0, seed, dt)
        else:
            amp_raw = self._amp_torch(a_star, f0, over, pth, ps, amp0, seed, dt)
        amp = (amp_raw - seed).clamp_min(0.0) / (1.0 - seed)
        rd = (0.3 + 2.4 * (1.0 - add) ** 1.5 + c["rd_offset"]).clamp(
            max(LF_RD_MIN, 0.212), rd_max())
        return self._physiology_rest(c, add, ps, ag_dc, f0, amp, amp_raw, rd, pth, po)

    def _amp_torch(self, a_star, f0, over, pth, ps, amp0, seed, dt):
        """성대 떨림 진폭 재귀의 torch 판 — 기준 구현 (`AMP_FAST` 의 시험이 대조한다)."""
        amp = []
        a = torch.full_like(ps[:, 0], seed) if amp0 is None else amp0.clone()
        for i in range(ps.shape[1]):
            tgt = a_star[:, i].clamp_min(seed)
            grow = self.k_growth * f0[:, i] * over[:, i] / pth[:, i]
            # **로지스틱은 닫힌 해로 적분한다** — 전방 오일러가 아니라.
            #
            #   dA/dt = σ·A·(1 − A/A*)  ⇒  A(t+dt) = A*·A / (A*·q + A·(1−q)),  q = e^{−σ·dt}
            #
            # 전방 오일러 `A + dt·σ·A·(1−A/A*)` 는 `dt·σ > 2` 에서 **불안정**하다.
            # 여기서 σ = k·f0·(Ps−Pth)/Pth 이고 k=0.25, dt=1 ms 이므로, 적합기가
            # p_sub 를 17 cmH2O(외치는 값)까지 밀고 f0 가 높으면 dt·σ 가 2 를 넘는다.
            # 그러면 1420 단계 재귀를 지나며 기울기가 폭주해 **비유한**이 된다.
            # 실측(FORMANT_ML_DEBUG_NONFINITE=1, yang_00000040 전체): 비유한 기울기가
            # 정확히 `p_sub` · `adduction` · `f0_target` 세 파라미터에서만, 격자점의
            # 절반(701/1420)에서 한꺼번에 났다 — 이 셋이 σ 와 A* 를 정하는 값이다.
            #
            # 닫힌 해는 어떤 dt·σ 에서도 안정이고 [0, A*] 를 벗어나지 않으며,
            # 프레임률이 바뀌어도 기동 모양이 같다. 분모는 A ≥ seed, A* ≥ seed 이므로
            # seed 아래로 내려가지 않는다.
            q = torch.exp(-(dt * grow).clamp_min(0.0))
            up = tgt * a / (tgt * q + a * (1.0 - q)).clamp_min(1e-6)
            down = a * torch.exp(-dt * f0[:, i] / self.cycles_decay)
            a = torch.where(a_star[:, i] > seed, up, down.clamp_min(seed))
            amp.append(a)
        return torch.stack(amp, 1)

    def _physiology_rest(self, c, add, ps, ag_dc, f0, amp, amp_raw, rd, pth, po):
        # 성문 난류는 **성문 양단의 압력 강하**로 난다. 구강 협착이 있으면 압력의 대부분이
        # 협착에서 떨어지고(Po/Ps = Ag²/(Ag²+Ac²), v1 §5.3) 성문 제트는 느려진다 — /s/ 동안
        # 성문 기식이 1~6 kHz 를 채우던 원인(측정: 앞공동 경로와 같은 크기).
        a_c = c["a_c"].clamp_min(1e-3)
        frac = a_c ** 2 / (a_c ** 2 + ag_dc ** 2)                  # ΔPg/Ps
        # **v1 의 `constriction_transmission` 은 여기 붙이지 않는다 — 옮겨 봤다가
        # 측정이 반증했다** (docs/MEASUREMENTS.md §22).
        #
        # v1 은 성문 난류가 (1) 얼마나 **생기고**(위 `frac` = `glottal_drop_fraction`)
        # (2) 얼마나 협착을 지나 **나오는가**(`constriction_transmission`) 를 **둘 다**
        # 곱한다. 물리는 맞다. 그런데 v2 에 그대로 얹으면 이중으로 깎인다 — 두 모형의
        # 성문 면적이 다르기 때문이다:
        #
        #     무성 마찰음의 Ag        v1 0.12~0.25 cm²      v2 `ag_dc` = 0.448 cm²
        #     그때 frac 이 주는 감쇠   −7.7 dB               **−26 dB**
        #
        # v2 는 성문을 3.7 배 넓게 열어 두므로 `frac` 하나가 이미 v1 의 두 항을 합친
        # 것(−30 dB)에 가깝다. 거기에 전달비(−22 dB)를 또 곱하면 −48 dB 다.
        #
        # 실측 (yang_00000101 전체, 전달비만 켜고 재적합):
        #     포락 94.67 → 93.19 %,  정밀 85.86 → 84.44 %,  손실 2.343 → 2.699
        #     유성 프레임 변조 1.50 → 2.46 / 4.19 → 5.63 / 1.32 → 2.37 (전부 악화,
        #       6 시드 범위가 안 겹친다 — 프레임의 77~82 % 가 여기다)
        #     마찰 60~150 Hz 만 4.99 → 2.74 로 좋아진다
        #     `aspiration` 0.031 → **0.011** — 붕괴가 오히려 깊어졌다
        #
        # 마지막 줄이 특히 중요하다. 이 항을 넣은 이유가 "적합기가 협착 구간만 기식을
        # 끌 수단이 없어 전역 배율로 끈다" 는 가설이었는데, 넣어도 전역 배율이 더
        # 내려갔다. **가설이 틀렸다** — 기식이 과한 곳은 협착 구간이 아니다.
        #
        # 다시 볼 때는 `ag_dc` 부터 봐라. 0.02 + 0.5·(1−add)^2.5 가 무성 마찰음
        # 자세(add 0.06)에서 0.448 cm² 를 주는데, 문헌의 개대 성문은 0.1~0.3 cm² 다
        # (Löfqvist; Cho·Jun·Ladefoged 2002). 거기가 맞으면 전달비도 같이 산다.
        # 세기는 ΔPg 에 선형 (√ 로 두면 /s/ 중 기식이 실측보다 10 dB 크다 — 같은 화자 A/B).
        asp = (1.0 - add) ** 2 * frac * torch.sqrt(ps.clamp_min(0.0) + 1e-12) * c["aspiration"]
        return dict(f0=f0, amp=amp, amp_raw=amp_raw, rd=rd, ag_dc=ag_dc, asp=asp, pth=pth,
                    po=po, uc=(self._last_uc if (ORAL_ODE and po is not None) else None))

    # ---------------------------------------------------------- 파형
    def _harmonics_blocked(self, phase, f0, tilt, log2f0, i0, wrd, rps, dispersion, fs,
                           f_nyq, width, f_cut, k_max, blk, psi_fr=None, n=None,
                           flow_reference=False):
        """하모닉 가산합성의 묶음판. 아래 순차 루프와 **같은 식**을 (B, N, blk) 로 푼다.

        `HARM_BLOCK` 주석이 근거다. 하모닉 차수는 커지기만 하므로, 한 묶음에 살아 있는
        하모닉이 하나도 없으면 그 뒤 묶음도 없다 — 거기서 멈춘다.
        """
        k = self.k_idx
        acc = torch.zeros(phase.shape, dtype=torch.float64, device=phase.device)
        flow = torch.zeros_like(acc) if flow_reference else None
        ph = phase.unsqueeze(-1)
        f0e = f0.unsqueeze(-1)
        tl = tilt.unsqueeze(-1)
        w0 = (1 - wrd).unsqueeze(-1)
        w1 = wrd.unsqueeze(-1)
        for j0 in range(0, k_max, blk):
            j1 = min(j0 + blk, k_max)
            kk = k[j0:j1].to(phase.dtype)                          # (Kb,)
            fk = f0e * kk                                          # (B,N,Kb)
            live = fk <= f_cut
            if not bool(live.any()):
                break
            tab = self.lf_coef[:, j0:j1]                           # (n_rd, Kb)
            cj = tab[i0] * w0 + tab[i0 + 1] * w1                   # (B,N,Kb)
            mask = torch.sigmoid((f_nyq - fk) / width) * live
            if MVF_HZ > 0.0:
                mask = mask * torch.sigmoid((MVF_HZ - fk) / MVF_WIDTH)
            if TILT_MAX_HZ > 0.0:
                oct_ = torch.log2(fk.clamp(20.0, TILT_MAX_HZ) / 1000.0)
                gain = 10.0 ** (tl * oct_ / 20.0)
            else:
                gain = 10.0 ** (tl * (log2f0.unsqueeze(-1) + torch.log2(kk)) / 20.0)
            th = ph * kk
            if rps is not None:
                th = th + rps[..., j0:j1]
            if dispersion != 0.0:
                xn = (fk / (0.5 * fs)).clamp(0.0, 1.0)
                th = th - dispersion * xn * xn
            if psi_fr is not None:
                jj = torch.arange(j0, j1, device=psi_fr.device) % psi_fr.shape[-1]
                psi = frames_to_samples(psi_fr.index_select(2, jj), self.hop)[:, :n]
                th = th + (self._hjit() * (fk - HARM_PHASE_ONSET).clamp_min(0.0)
                           / 1000.0) * psi
            w_k = mask * gain
            if GLOTTAL_CLOSURE_SPREAD_S > 0.0:
                w_k = w_k * torch.exp(
                    -0.5 * (2.0 * math.pi * fk * GLOTTAL_CLOSURE_SPREAD_S) ** 2)
            term = 2.0 * w_k * (cj.real * torch.cos(th) - cj.imag * torch.sin(th))
            acc = acc + term.double().sum(-1)
            if flow_reference:
                primitive = (2.0 * w_k * (cj.real * torch.sin(th) + cj.imag * torch.cos(th))
                             / (2.0 * math.pi * kk))
                flow = flow + primitive.double().sum(-1)
        return (acc, flow) if flow_reference else acc

    def _lf_index(self, rd: torch.Tensor):
        """(B,N) Rd -> (i0, w). 하모닉 계수는 **차수별로** 표에서 뽑는다.

        전에는 (B,N,K) 복소 텐서를 통째로 만들고 루프에서 `[..., j]` 로 잘랐다. 그러면
        역전파가 차수마다 (B,N,K) 짜리 0 텐서를 만들어 채운다 — K=234, N=19200 이면 한 번에
        37 MB, 5937 번이면 역전파의 68 % 였다(프로파일). 표를 차수별로 인덱싱하면 (B,N) 이다.
        """
        g = self.rd_grid
        pos = (rd.clamp(g[0], g[-1]) - g[0]) / (g[-1] - g[0]) * (len(g) - 1)
        i0 = pos.floor().long().clamp(0, len(g) - 2)
        return i0, pos - i0.to(pos.dtype)

    def _flow_pos(self, phase: torch.Tensor, rd: torch.Tensor) -> torch.Tensor:
        """같은 LF 펄스의 성문 유량 (봉우리 1, 열림 φ=0 에서 0 부터) — 표본마다 (B,N)."""
        tab = getattr(self, "_re_flow", None)
        if tab is None or tab.device != phase.device:
            rows = []
            for r in self.rd_grid.tolist():
                e = glottal_pulse(float(r))
                f = np.concatenate(([0.0], np.cumsum(e)[:-1]))
                rows.append(np.clip(f / (f.max() + 1e-12), 0.0, 1.0))
            tab = torch.tensor(np.stack(rows), dtype=phase.dtype, device=phase.device)
            self._re_flow = tab
        i, w = self._lf_index(rd)
        size = tab.shape[1]
        pos = torch.remainder(phase / (2 * math.pi), 1.0) * size
        j = pos.floor().long() % size
        u = pos - pos.floor()
        j1 = (j + 1) % size
        lo = tab[i, j] * (1 - u) + tab[i, j1] * u
        hi = tab[i + 1, j] * (1 - u) + tab[i + 1, j1] * u
        return lo * (1 - w) + hi * w

    def glottal_re_env(self, c: dict, st: dict, up, phase, rd, amp, voiced) -> torch.Tensor:
        """성문 제트 난류의 포락 (B,N) — `GLOTTAL_RE` 참조."""
        flow = self._flow_pos(phase, rd)
        area = up(st["ag_dc"]) + GLOT_A_PEAK * amp * voiced * flow
        a_c = up(c["a_c"].clamp_min(1e-3))
        frac = a_c ** 2 / (a_c ** 2 + area ** 2)
        dp = up(c["p_sub"].clamp_min(0.0)) * frac * CMH2O
        v = torch.sqrt(2.0 * dp / RHO + 1e-6)
        re = v * (2.0 * area / GLOT_LEN_CM) / GLOT_NU
        wq = GLOT_RE_W * GLOT_RE_CRIT ** 2
        over = wq * torch.nn.functional.softplus((re ** 2 - GLOT_RE_CRIT ** 2) / wq)
        g = (over / (re ** 2 + wq)) ** 1.5
        drive = g * (v / GLOT_V_REF) ** 3 * torch.sqrt(area / GLOT_A_REF)
        return GLOT_RE_GAIN * drive * up(c["aspiration"])

    def lf_noise_envelope(self, phase: torch.Tensor, rd: torch.Tensor,
                          voiced: torch.Tensor) -> torch.Tensor:
        """Unit-cycle-mean AM from normalized LF flow, with no unvoiced AM.

        Periodic Catmull–Rom interpolation is C1 across phase/table boundaries.
        Each Rd row is centered over a full cycle, not the current chunk; hence
        normalization needs neither future samples nor extra streaming state.
        The fixed 0.7 depth reuses aspiration's existing modulation strength.
        """
        i, wrd = self._lf_index(rd)
        size = self.lf_flow.shape[1]
        pos = torch.remainder(phase / (2 * math.pi), 1.0) * size
        j = pos.floor().long()
        w = pos - j.to(pos.dtype)
        points = []
        for offset in (-1, 0, 1, 2):
            col = (j + offset) % size
            points.append(self.lf_flow[i, col] * (1 - wrd)
                          + self.lf_flow[i + 1, col] * wrd)
        p0, p1, p2, p3 = points
        flow = p1 + 0.5 * w * (p2 - p0 + w * (
            2 * p0 - 5 * p1 + 4 * p2 - p3 + w * (3 * (p1 - p2) + p3 - p0)))
        # 깊이는 `LF_AM_DEPTH` 다. 1 을 넘으면 음수가 될 수 있으므로 바닥을 둔다.
        return (1.0 + LF_AM_DEPTH * voiced * flow).clamp_min(0.05)

    def forward(self, c: dict, phase0: torch.Tensor | None = None,
                rps: torch.Tensor | None = None, dispersion: float = 0.0,
                noise=None, frame0: int = 0,
                amp0: torch.Tensor | None = None, state: dict | None = None,
                emit: int | None = None,
                pulse_phase: torch.Tensor | None = None) -> dict:
        """c: 프레임률 (B,T) dict. 반환 샘플률 텐서들 (B,N).

        noise : NoiseBank (위치 기반 난수). frame0 : 이 청크의 첫 프레임 번호.
        amp0  : 이전 청크 끝의 기동 진폭(스트리밍). state : 지터/시머 저역통과 상태.
        emit  : 내보낼 프레임 수(기본 전부). 스트리밍에서는 T+1 프레임을 주고 T 만 내보내
                마지막 프레임의 샘플 보간이 다음 프레임을 보게 한다(선행 1 프레임).
        """
        if DISP_FIT and not torch.is_tensor(dispersion) and dispersion != 0.0:
            dispersion = dispersion * torch.exp(
                DISP_LIM * torch.tanh(self.disp_log / DISP_LIM))
        st = self.physiology(c, amp0=amp0,
                             po0=(state or {}).get("po"))
        hop, fs = self.hop, self.fs
        b, t_all = st["f0"].shape
        t = t_all if emit is None else int(emit)
        n = t * hop
        up = lambda v: frames_to_samples(v.unsqueeze(-1), hop)[..., 0][:, :n]
        f0, amp, rd = up(st["f0"]), up(st["amp"]), up(st["rd"])
        tilt, jit, shm = up(c["tilt"]), up(c["jitter"]), up(c["shimmer"])
        # 지터/시머: 프레임률 백색 난수(위치 기반) -> 1 극 저역통과(상태 유지) -> 샘플률
        state = {} if state is None else state
        noise = noise or NoiseBank()
        zj = noise.white("jitter", frame0, t_all, b, f0.dtype, f0.device)
        zs_ = noise.white("shimmer", frame0, t_all, b, f0.dtype, f0.device)
        a = 0.6                                                     # ~ 50 Hz 모서리 @1 kHz 프레임률
        g = math.sqrt((1 - a) / (1 + a)) * 2.0
        zj, zj_state = tv_biquad(zj, g * (1 - a), 0.0, 0.0, -a, 0.0, zi=state.get("j"))
        zs_, zs_state = tv_biquad(zs_, g * (1 - a), 0.0, 0.0, -a, 0.0, zi=state.get("s"))
        if t != t_all:                       # 상태는 내보내는 마지막 프레임(t) 기준이어야 한다
            _, zj_state = tv_biquad(noise.white("jitter", frame0, t, b, f0.dtype, f0.device),
                                    g * (1 - a), 0.0, 0.0, -a, 0.0, zi=state.get("j"))
            _, zs_state = tv_biquad(noise.white("shimmer", frame0, t, b, f0.dtype, f0.device),
                                    g * (1 - a), 0.0, 0.0, -a, 0.0, zi=state.get("s"))
        state["j"], state["s"] = zj_state, zs_state
        zs = frames_to_samples(torch.stack([zj, zs_], -1), hop)[:, :n]
        f0 = f0 * (1.0 + jit * zs[..., 0])
        amp = amp * (1.0 + shm * zs[..., 1]).clamp_min(0.0)
        # 위상 누적은 float64 로 (float32 cumsum 오차 × 하모닉 차수가 청크 경계에서 보인다)
        if pulse_phase is not None:
            # **펄스 잠금**: 위상을 f0 적분이 아니라 목표 녹음의 성문 폐쇄 시각에서 만든다
            # (MEASUREMENTS §51.16). 지터·시머의 실현이 아니라 **관측된 주기 경계**다.
            phase64 = pulse_phase.double()[:, :n]
        else:
            phase64 = torch.cumsum(2 * math.pi * f0.double() / fs, dim=-1)
            if phase0 is not None:
                phase64 = phase64 + phase0.double()
        ph_unw = phase64.detach()                 # 주기 동기 위상 분산의 주기 좌표 (HJIT_CYCLE)
        phase64 = torch.remainder(phase64, 2 * math.pi)
        phase = phase64.to(f0.dtype)                  # 상태(phase_last)는 float64 로 넘긴다:
        #                                               float32 위상 2e-7 rad × 하모닉 100 = 2e-5 rad 가 청크 경계에서 보였다
        # 하모닉 가산합성. **하모닉마다 순차 누적**한다 — (B,N,K).sum(-1) 은 텐서 크기에 따라
        # 축약 순서가 달라 float32 반올림이 청크 의존이 되고(3e-6), 그것이 고 Q 성도를 지나며
        # 스트리밍/오프라인 차이 1e-3 로 커졌다(측정). 순차 누적은 청크와 무관하고 메모리도 작다.
        i0, wrd = self._lf_index(rd)                               # (B,N), (B,N)
        k = self.k_idx
        f_nyq, width = 0.95 * fs / 2, 0.02 * fs / 2
        f_cut = f_nyq + 6.0 * width                    # 이 위는 마스크를 정확히 0 으로 (청크 무관 상한)
        log2f0 = torch.log2(f0.clamp_min(1.0) / 1000.0)
        du = torch.zeros_like(phase)
        flow = torch.zeros_like(phase) if self.flow_reference else None
        # NaN 안전. f0 에 NaN 이 들어오면 ceil(NaN) 이 그대로 통과해 int(NaN) 에서
        # **예외로 터진다** — 적합기의 "손실이 비유한이면 중단" 가드가 손실을 보기도 전이라
        # 원인을 못 찾는다. 하모닉 상한만 정하는 값이므로 NaN 은 상한으로 접고, 잘못된
        # 값은 아래 du 계산에서 NaN 으로 **전파**시켜 가드가 잡게 둔다.
        f0_all = torch.nan_to_num(f0.detach(), nan=1.0, posinf=1.0)
        # **하모닉 상한은 소리 나는 샘플의 f0 로 잡는다.** 무성 샘플은 진폭이 0 이라
        # (`du = du * amp`) 그 하모닉은 어차피 버려진다. 그런데 무성 구간의 f0_target 은
        # 적합기가 아무 데나 두는 값이라 낮게 떨어진다 — 실측(`yang_00000040`)에서 전체
        # 최소 95.6 Hz 는 무성 프레임이었고 유성 최소는 147.5 Hz 였다. 전체로 잡으면
        # 하모닉 269 개, 유성으로 잡으면 175 개다. 적합 한 회차의 68 %가 이 루프다.
        # 소리 나는 샘플이 하나도 없으면(무성 청크) 예전처럼 전체로 잡는다.
        on = amp.detach() > 1e-3
        f0_min = (f0_all[on] if bool(on.any()) else f0_all).min().clamp_min(1.0)
        k_max = int(torch.clamp(torch.ceil(f_cut / f0_min), 1, len(k)).item())
        psi_fr = None
        psi_cyc = None
        if HARM_PHASE_JIT > 0.0 and HJIT_CYCLE:
            # (B, n_c, KMAX) 주기 난수 + 샘플마다 (주기 번호, 주기 안 위치)
            kw = HARM_PHASE_KMAX
            u = ph_unw / (2.0 * math.pi) - HJIT_CYCLE_ANCHOR
            cyc = torch.floor(u)
            c0 = int(cyc.min().item())
            n_c = int(cyc.max().item()) - c0 + 2
            tab = noise.white("hphase_c", (c0 + 10) * kw, n_c * kw, b, f0.dtype,
                              f0.device).reshape(b, n_c, kw)
            ci = (cyc - c0).long()
            al = (u - cyc).to(f0.dtype)
            psi_cyc = (tab, ci, al)
        elif HARM_PHASE_JIT > 0.0:
            # (B, T, KMAX) 프레임률 난수. 위치 = 프레임·KMAX + 배음 — 청크와 무관하다.
            kw = HARM_PHASE_KMAX
            if HJIT_TAU_MS > 0.0:
                # **커널 폭만큼 앞뒤로 더 받아서 자른다** — 그래야 청크를 나눠도 값이 같다(위 주석의 성질).
                ker = _hjit_kernel(HJIT_TAU_MS, float(self.hop) * 1000.0 / fs, f0.dtype, f0.device)
                pad = (ker.numel() - 1) // 2
                wide = noise.white("hphase", (frame0 - pad) * kw, (t_all + 2 * pad) * kw, b,
                                   f0.dtype, f0.device).reshape(b, t_all + 2 * pad, kw)
                x = wide.permute(0, 2, 1).reshape(b * kw, 1, t_all + 2 * pad)
                psi_fr = torch.nn.functional.conv1d(x, ker.view(1, 1, -1))
                psi_fr = psi_fr.reshape(b, kw, t_all).permute(0, 2, 1)
            else:
                psi_fr = noise.white("hphase", frame0 * kw, t_all * kw, b, f0.dtype,
                                     f0.device).reshape(b, t_all, kw)
        if HARM_BLOCK > 0:
            harmonics = self._harmonics_blocked(
                phase, f0, tilt, log2f0, i0, wrd, rps, dispersion, fs,
                f_nyq, width, f_cut, k_max, int(HARM_BLOCK), psi_fr, n,
                flow_reference=self.flow_reference)
            if self.flow_reference:
                harmonic_du, harmonic_flow = harmonics
                du = du + harmonic_du.to(du.dtype)
                flow = flow + harmonic_flow.to(du.dtype)
            else:
                du = du + harmonics.to(du.dtype)
            k_max = 0                      # 아래 순차 루프를 건너뛴다
        for j in range(k_max):
            kk = k[j]
            fk = f0 * kk
            live = fk <= f_cut
            if not bool(live.any()):
                continue
            cj = self.lf_coef[i0, j] * (1 - wrd) + self.lf_coef[i0 + 1, j] * wrd   # (B,N)
            mask = torch.sigmoid((f_nyq - fk) / width) * live
            if MVF_HZ > 0.0:
                mask = mask * torch.sigmoid((MVF_HZ - fk) / MVF_WIDTH)
            if TILT_MAX_HZ > 0.0:
                # 셸프: 상한 위에서는 기울기가 더 안 오른다 (TILT_MAX_HZ 주석 참조).
                oct_ = torch.log2(fk.clamp(20.0, TILT_MAX_HZ) / 1000.0)
                gain = 10.0 ** (tilt * oct_ / 20.0)
            else:
                gain = 10.0 ** (tilt * (log2f0 + math.log2(float(kk))) / 20.0)
            # **`|cj|` 와 `∠cj` 를 따로 구하지 않는다.**
            #
            #     |cj|·cos(θ + ∠cj) = Re[cj·e^{iθ}] = cj.real·cosθ − cj.imag·sinθ
            #
            # 항등식이므로 결과는 완전히 같은데, `abs` 와 `angle` 이 사라진다. 그 둘은
            # cj = 0 에서 미분이 정의되지 않아 **NaN 기울기**를 낸다:
            #   ∂|z|/∂x = x/|z| → 0/0,   ∂∠z/∂x = −y/|z|² → 발산.
            # 그리고 `cj` 는 표의 두 행을 `rd` 로 **선형 보간한 값**이라 실제로 0 을
            # 지난다 — 이웃 행의 실수부 내적이 음수인 항목이 **47.2 %** 다 (11063 개 중
            # 5224 개). `rd` 는 `adduction` · `rd_offset` 에서 나오는 적합 파라미터이므로
            # 그 NaN 이 그대로 적합기로 흘러 들어간다 (MEASUREMENTS §31).
            #
            # 덤으로 삼각함수 호출이 하나 늘고 `abs`·`angle` 둘이 빠져 조금 싸다.
            th = phase * kk
            if rps is not None:
                th = th + rps[..., j]
            if dispersion != 0.0:
                # **하모닉 위상 분산.** 고역일수록 지연되는 이차 위상 −D·(f/f_nyq)²
                # (= 주파수에 선형인 그룹 지연 = 시간축으로 퍼지는 chirp). 실제 성대는
                # 딱딱하지 않아 점막파로 상하연이 시간차를 두고 닫히므로 이런 분산이
                # 생긴다. `rps` 와 달리 **샘플률 fk 를 그대로 써서** (B,N,K) 텐서를
                # 만들지 않는다 — K=240 이면 그것만 65 MB 다.
                xn = (fk / (0.5 * fs)).clamp(0.0, 1.0)
                th = th - dispersion * xn * xn
            if psi_fr is not None:
                th = th + (self._hjit() * (fk - HARM_PHASE_ONSET).clamp_min(0.0)
                           / 1000.0) * up(psi_fr[:, :, j % psi_fr.shape[-1]])
            elif psi_cyc is not None:
                tab, ci, al = psi_cyc
                col = tab[:, :, j % tab.shape[-1]]                       # (B, n_c)
                p0 = torch.gather(col, 1, ci)
                p1 = torch.gather(col, 1, (ci + 1).clamp(max=col.shape[1] - 1))
                th = th + (self._hjit() * (fk - HARM_PHASE_ONSET).clamp_min(0.0)
                           / 1000.0) * (p0 + al * (p1 - p0))
            w_k = mask * gain
            if GLOTTAL_CLOSURE_SPREAD_S > 0.0:
                # 폐쇄 시각의 흩어짐 -> **영위상** 크기 테이퍼 (위 상수 참조).
                w_k = w_k * torch.exp(
                    -0.5 * (2.0 * math.pi * fk * GLOTTAL_CLOSURE_SPREAD_S) ** 2)
            du = du + 2.0 * w_k * (cj.real * torch.cos(th) - cj.imag * torch.sin(th))
            if self.flow_reference:
                flow = flow + (2.0 * w_k * (cj.real * torch.sin(th) + cj.imag * torch.cos(th))
                               / (2.0 * math.pi * kk))
        if SRC_EQ:
            du = self._src_eq(du, state)
            if self.flow_reference:
                flow = self._src_eq(flow, state.setdefault("flow_eq", {}))
        if SRC_EQ_HF:
            du = self._src_eq_hf(du, state)
            if self.flow_reference:
                flow = self._src_eq_hf(flow, state.setdefault("flow_eq_hf", {}))
        du = du * amp
        if self.flow_reference:
            flow = flow * amp
        if "voice_gain" in c:
            # 성문 배음에만 거는 빠른 이득 (control.py 의 voice_gain). 기식 포락(asp_env)은 건드리지 않는다.
            du = du * up(torch.pow(10.0, c["voice_gain"] / 20.0))
            if self.flow_reference:
                flow = flow * up(torch.pow(10.0, c["voice_gain"] / 20.0))
        # 성문 개방기 (LF: 0 ~ te 가 열림) -> 기식 AM 마스크
        frac = phase / (2 * math.pi)
        open_phase = torch.sin(math.pi * frac.clamp(0, 1) / 0.65).clamp_min(0.0) ** 2
        open_phase = torch.where(frac < 0.65, open_phase, torch.zeros_like(open_phase))
        asp = up(st["asp"])
        if VOICED_SOFT > 0.0:
            voiced = (amp / VOICED_SOFT).clamp(0.0, 1.0)
        else:
            voiced = (amp > 1e-3).float()
        noise_am = None
        if self.noise_modulation == "lf":
            noise_am = self.lf_noise_envelope(phase, rd, voiced)
            # Legacy open mask mean = 0.65/2, so keep its mean source level
            # while replacing only the periodic shape (not an RMS guarantee).
            asp_env = asp * (1.0 + ASP_AM_DEPTH * voiced * (0.325 - 0.5)) * noise_am
        else:
            asp_env = asp * (1.0 + ASP_AM_DEPTH * voiced * (open_phase - 0.5))
        if GLOTTAL_RE:
            asp_env = self.glottal_re_env(c, st, up, phase, rd, amp, voiced)
        result = dict(du=du, phase=phase, asp_env=asp_env.clamp_min(0.0),
                    noise_am=noise_am, open_phase=open_phase * voiced,
                    amp=amp, f0=f0, ag_dc=up(st["ag_dc"]), ag_dc_frames=st["ag_dc"], voiced=voiced,
                    physiology=st, amp_last=st["amp_raw"][:, t - 1], state=state,
                    phase_last=phase64[:, -1:])
        if st.get("po") is not None:
            # 구강압은 **한 곳에서만** 적분한다 — 마찰 소스가 같은 값을 받아 쓴다 (§52.344).
            result["po_frames"] = st["po"]
            state["po"] = st["po"][:, t - 1].detach()
            if st.get("uc") is not None:
                result["uc_frames"] = st["uc"]
        if self.flow_reference:
            from .loaded_source import PULSE_FLOW_CM3_S
            peak = self.lf_flow_peak[i0] * (1 - wrd) + self.lf_flow_peak[i0 + 1] * wrd
            result["flow_scale"] = PULSE_FLOW_CM3_S / peak
            result["reference_flow"] = flow * result["flow_scale"]
        return result
