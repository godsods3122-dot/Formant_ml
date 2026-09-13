"""성도: 포먼트 + 위상차(올패스) 필터 -> 비강 필터 -> 방사.

전부 `tviir.tv_biquad` 위에 있다. 계수는 샘플마다 바뀌고 상태는 이어진다.
그래서 자음처럼 경계조건이 급변하는 곳에서 **과도응답과 위상은 방정식이 낸다**
— 우리가 프레임을 이어 붙이지 않는다.

신호 경로 (선형이므로 순서를 바꿔도 정상 상태는 같다; 과도 상태의 차이는 무시할 만하다)

    du (성문 유량미분) + 방사(기식) + back_leak·방사(마찰)  ──> 포먼트 캐스케이드 (K + 고차 보정) ─┐
    마찰·과도음 소스 ──> 방사 ──> 앞공동 극 + 뒤공동 영점 ─────────────────────────────────────┤
                                                                                              ▼
                                                          올패스 위상차 체인 -> 비강 극영점 -> 측지 영점 -> tract_gain

고차 극 보정(higher-pole correction)이 **필수**다. DC 정규화 공명기 K 개만 곱하면
K 번째 포먼트 위가 실제(무손실관 |H| ≥ 1)보다 수십 dB 어둡다 — v1 이 "모음의 9~13 kHz
가 34 dB 모자란다"(HANDOFF §6.8) 고 적은 것이 정확히 이것이다. 균일관 위치
F_n = (2n−1)·c/4L 의 고차 극을 나이퀴스트까지 넓은 대역폭으로 고정 배치한다.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .control import N_ALLPASS, N_FORMANTS, frames_to_samples
from .noise import C_SOUND
from .tviir import (allpass_coeffs, antiresonator_coeffs, first_difference_coeffs,
                    lowpass_coeffs, notch_coeffs, peak_coeffs, resonator_coeffs, tv_biquad)

#: 전역 대역폭 배율 두 개(`log_extra_bw`, `log_front_bw`)의 **물리 범위** [ln 배].
#: tanh 로 부드럽게 가둔다: 배율이 e^{±HF_BW_LIM} = 0.4~2.5 배 안에 머문다.
#:
#: 둘을 옵티마이저에 넣자(§49.1) 곧바로 상한 없이 달아났다. 실측(out/L31):
#: s040 `log_extra_bw` +3.09(고차 극 대역폭 **22 배**), s101 `log_front_bw` +4.36
#: (앞공동 대역폭 **80 배**). 고차 극을 22 배로 벌리면 캐스케이드가 F4 위 보정을
#: 잃어 **9.5 kHz 위가 통째로 꺼진다** — 사용자가 "고역이 다 죽었다" 고 한 그것이다.
#: 전역 스칼라 하나로 파일 전체의 고역을 끄는 길이 열려 있으면, 고역이 어긋난 대가를
#: 밴드 바닥이 막아 주는 손실에서는 끄는 쪽이 싸다. 관의 벽 손실·방사 손실이 대역폭을
#: 몇 배 넘게 바꾸지는 않는다(Fant: 벽 손실 30~70 Hz, 고차에서 방사 우세).
#:
#: **고차 극은 기본 법칙보다 좁힐 수 없다** (배율 ≥ ~1, `_extra_mult`). 좁히면 사다리
#: (F_K + n·c/2L, 약 1.2 kHz 간격)의 극마다 **F4 만 따라가는 수평 줄**이 선다 — 사용자가
#: "고역의 특징적인 평평한 줄" 이라 한 그것이다. 인과 실측(out/L25/s101 제어열 그대로,
#: 이 배율만 바꿔 재렌더, 10~16 kHz 미세구조의 50 ms 지연 상관 = 줄이 안 새로 뽑히는 정도):
#:
#:     ×0.50 0.132 | ×0.57(적합값) 0.116 | ×1.00 0.065 | ×1.65 0.052 | ×2.46 0.050 | 목표 0.051
#:
#: 앞공동 배율은 이 척도에 영향이 없었다(0.116 → 0.116). 1 차원 관의 사다리는 5~6 kHz
#: 위에서 이미 실제가 아니다(횡모드) — 좁은 사다리가 들릴 이유가 없다.
HF_BW_LIM = 0.916

#: **성도 궤적의 꺾임을 없앤다** — 포먼트·대역폭의 샘플률 궤적에 거는 2 단 1 차 평활의 시간상수 [ms].
#: 0 이면 끈다.
#:
#: 제어값은 1 ms 프레임이고 샘플로는 선형 보간한다. 그러면 궤적이 **1 ms 마다 꺾이고**, 강한 저역
#: 포먼트 에너지를 지나는 공명기가 꺾일 때마다 미세한 과도를 낸다. 1 ms 간격의 과도열은 **1 kHz
#: 간격의 빗살**이고, 저역보다 40~60 dB 작은 고역을 덮어 버린다 — 사용자가 "고역에 똑같이 생긴
#: 배음이 층층이 쌓여 있다" 고 한 그것이다. 실측(MEASUREMENTS §50.14): 고역 100 ms 국소 평균의
#: 1 kHz 지연 자기상관이 목표 0.05~0.1 인데 적합은 0.5~0.7. 마디 없는 F4 사인 궤적을 1 / 0.5 /
#: 0.25 ms 프레임으로 넣으면 요철이 1 kHz(+0.26) → 2 kHz(+0.09) → 없음 으로 **마디 간격을 따라간다.**
#: F1~F3·대역폭·폐압·rd·tilt 의 움직임은 이 요철을 안 만들고 F4 가 만든다.
#:
#: 1 차 평활은 연속인 입력을 C¹ 로 만든다(y' = (x − y)/τ 가 연속). 2 단이면 1 kHz 영상을 약 20 dB
#: 깎고, 궤적은 약 2τ = 1 ms 늦는다 — 포먼트 전이(수십 ms)에 비하면 없는 것과 같다. 상태를
#: 이어 가므로 청크를 나눠도(스트리밍) 같은 값이 나온다.
TRACK_SMOOTH_MS = 0.5

#: **잡음 가지 v2** — 잡음은 높은 Q 공진기를 지나지 않는다 (docs/NOISE_SOURCE_REVIEW.md §3).
#:
#: v1 은 기식과 새어 든 마찰(`back_leak`)을 성문 배음과 **함께** 모음 종속 가지(F1~F8 + 고차 사다리,
#: 10 kHz 에서 Q ≈ 18)에 넣었고, 마찰의 나머지는 앞공동 공진기 **하나**를 지났다 — 그 대역폭은 파일
#: 전역 스칼라 하나라 s040 은 하한(×0.4, 좁은 공진)에, s101 은 상한(×2.5)에 붙었다(`L32`~`L44`).
#: 잡음이 높은 Q 공진기를 지나면 그 주파수만 강화되어 울린다 — 사용자가 "극이 노이즈의 일부를 Q
#: 공진으로 강화한다" 고 한 것. 문헌(Klatt 1980 병렬 가지, DDSP 의 FIR 잡음 성형)은 잡음의 모양을
#: 넓은 포락으로 입힌다. v2:
#:   * 성문 배음만 모음 종속 가지를 지난다. 잡음은 **같은 포먼트 주파수의 저 Q 사본**(대역폭 >=
#:     `NOISE_BW_REL`*f, 고차 사다리 x`NOISE_EXTRA_BW`)을 지난다 — 포먼트는 따라가되 울리지 않는다.
#:   * 앞공동은 1 차 모드 f_p 와 3 차 모드 3 f_p 의 **병렬 저 Q 공진**(Q <= `FRONT_Q_MAX`). 극 주파수도
#:     성도 궤적과 같은 C1 평활을 받는다(1 ms 마디 -> 1 kHz 빗살, §50.14 와 같은 기제).
NOISE_V2 = False
#: **비강 여분 극의 대역폭 배율** (MEASUREMENTS §51.36).
#:
#: 비강 분기는 2.4 kHz 위에 875 Hz 간격의 극 16 개를 세워 "관" 의 고역을 살린다. 그런데 그 극들은
#: 시간에 고정되어 있고 캐스케이드라 봉우리가 dB 로 더해져, 연구개를 열면 **고역에 똑같이 생긴
#: 것들이 층층이 쌓인 빗살**이 된다 — 사용자가 되풀이해 지적한 바로 그 결함이다. 실측(`L91/s040`,
#: 감사의 `층층 10-16k`, 목표 0.049):
#:
#:     ×1 (지금)  0.124      ×2  **0.053**      ×3  0.055      ×5  0.055
#:     여분 극 제거 0.055      간격 반칸 이동 0.203 (빗살이 이 극들의 것임을 확인)
#:
#: 2 배면 대역폭이 간격의 세 배를 넘어 봉우리가 서로 녹아 매끈한 선반이 된다. 비용은 포락 **0.11 점**
#: (89.29 → 89.18) 이다.
#: **켰다 끄는 보조 공진** (MEASUREMENTS §51.40). 이웃 포먼트 사이에 극 하나씩(F1–F2, F2–F3, F3–F4).
#: 깊이는 `aux{k}_mix` 제어가 0~1 로 연속으로 켠다 — 0 이면 정확히 항등이다.
#: **고역 포락을 직접 적합한다** (MEASUREMENTS §51.44). 4 kHz 위에 로그 등간격 피킹 EQ `HF_EQ_N` 개를
#: 두고 그 이득(dB)을 화자 전역 파라미터로 적합한다. 0 dB 이면 정확히 항등이다.
#:
#: 왜 극으로 안 되는가: 고역은 고정 극들이 Q≈5 로 서로 겹쳐 **평평**하고, `HF_Q`·`NOISE_BW_REL`·
#: `NOISE_EXTRA_BW` 를 어떻게 돌려도 켑스트럼 포락의 봉우리−골이 6.6~6.7 dB 에서 안 움직인다
#: (목표 11.8). 적합기가 고역 극의 대역폭을 범위의 넓은 끝까지 미는 것도 손실이 고역의 **구조**를
#: 요구하지 않았기 때문이다. 극을 더 풀면 열이 8 개 늘어 움직임 단계가 무너진다(`M5` 실패,
#: 2.1 단계 −3.8 점). 그래서 **모양을 직접 잡는 손잡이**를 두고, 그것만 따로 적합한다.
HF_EQ = False
HF_EQ_N = 8
HF_EQ_LO, HF_EQ_HI = 4000.0, 20000.0
HF_EQ_Q = 1.1
HF_EQ_LIM = 12.0          # 이득 한계 [dB] — tanh 로 부드럽게 가둔다

#: **보조 영점 — 소리를 죽이는 극** (MEASUREMENTS §51.46). 틈마다 하나, 깊이는 `azr{k}_mix` 가
#: 0~1 로 연속으로 켠다. 0 이면 정확히 항등이다.
#:
#: 왜: 전극(all-pole) 캐스케이드에서 이웃 두 극 사이 골의 깊이는 그 극들의 주파수·대역폭이 정하고,
#: 그 이상은 **원리적으로 못 판다**. 목표의 골이 그보다 얼마나 깊은지 재면 (같은 화자 24 파일,
#: 틈마다 모음 17,000 프레임):
#:
#:     F1–F2  3 dB 넘게 깊은 프레임 **19 %**, 6 dB 넘게 8 %, 10 dB 넘게 2 %  (95 % 분위 7.1 dB)
#:     F2–F3  **29 %**, 13 %, 4 %  (8.9 dB)
#:     F3–F4  **30 %**, 13 %, 4 %  (9.0 dB)
#:
#: 즉 **프레임의 20~30 % 는 영점 없이는 못 만든다.** 중앙값은 0 근처이므로 상시로 켜면 나머지
#: 70 % 를 망친다 — 그래서 보조 공진과 같은 **켰다 끄는** 구조다. 95 % 분위가 7~9 dB 이니 극-영점
#: 쌍 하나면 충분하고, 틈마다 둘 이상 둘 근거는 없다.
def _live(t: "torch.Tensor", thr: float = 1e-4) -> bool:
    """이 손잡이를 **그래프에 올려야 하는가**.

    깊이가 0 인 가지를 통째로 건너뛰면 ``∂손실/∂mix`` 가 아예 **존재하지 않는다**. 그러면 기본값이
    0 인 손잡이는 영영 0 에 묶인다 — `out/M/M11`·`M12` 에서 `azr*_mix`·`aux*_mix` 의 전 구간
    표준편차가 5e-17 (부동소수점 잡음) 이었던 것이 이것이다. 적합 중(기울기가 흐르는 중)에는
    값이 0 이어도 올리고, 합성만 할 때는 예전처럼 건너뛴다.
    """
    return bool(t.requires_grad) or float(t.detach().abs().max()) > thr


AUX_ZEROS = False
N_AZR = 3
#: 영점의 폭 = 자리 × 이 값 × (1 − 깊이·0.5) — 깊이를 올리면 좁고 깊어진다.
#:
#: 0.35 는 **2.7 배 넓었다**. 실측(§51.51)으로 모음 0~4 kHz 에서 목표 골의 반높이 폭이 281~305 Hz,
#: 폭/자리가 중앙 **0.13** 이다. 0.35 면 2 kHz 에서 700 Hz 라 골이 아니라 분화구가 된다.
AZR_WFRAC = 0.15
#: 노치의 깊이 = 극/영점 대역폭 비 (`notch_coeffs` 의 `pole_ratio`). 4 면 **12 dB 가 천장**인데
#: 목표 골의 95 % 분위가 9~14 dB 다 — 젖음/마름 섞기까지 거치면 그 천장에 닿지도 못한다.
#: 사용자: *"영점이 더 강하게 작용되어야 할 거 같고"*. 8 = 18 dB.
AZR_RATIO = 8.0

#: **고역 영점** (MEASUREMENTS §51.50). 보조 영점(`AUX_ZEROS`)은 F1~F4 사이에만 있어서 **F4 위**를
#: 못 건드린다. 그런데 사용자가 되풀이해 지적한 "고역 아티팩트" 는 거기 있다.
#:
#: 실측 (`out/M/M1`, 마찰 프레임 161·248 개, 켑스트럼 포락 2048/리프터 96, 4~16 kHz 에서 넓은 추세
#: 대비 4 dB 넘게 파인 골):
#:
#:                     목표 골 깊이        우리 골 깊이      **못 판 깊이**
#:     s040   786 개   중앙 4.89 / 95 % 8.38   1.55 / 4.67   중앙 +3.80  95 % +7.54  최악 11.2
#:     s101  1812 개   중앙 5.46 / 95 % 9.26   0.83 / 4.27   중앙 +4.79  95 % +9.50  최악 14.5
#:
#: 즉 목표의 고역은 **골이 파인 물결**인데 우리 고역은 거의 **평평**하다. 극으로는 못 메운다 —
#: 골의 반깊이 폭이 260 Hz(켑스트럼 분해능의 바닥이라 실제로는 더 좁다) 인데 우리 고역 극은
#: 1 kHz 간격에 대역폭이 그보다 넓어서, `HF_Q`·`NOISE_BW_REL`·`NOISE_EXTRA_BW` 를 어떻게 돌려도
#: 봉우리−골이 6.6 dB 에서 안 움직였다 (§51.44). **좁고 깊은 골은 영점의 일이다.**
#:
#: 골의 간격은 중앙 820~1406 Hz 지만 변동계수가 0.83 이라 규칙적인 빗살이 아니다 — 그래서 뒤공동
#: 영점열(간격 하나)이 아니라 **자리가 자유로운 영점 셋**으로 둔다. 자리는 겹치는 세 띠 안이다.
HF_ZEROS = False
N_HZR = 3
HZR_BANDS = ((4000.0, 8000.0), (7000.0, 12000.0), (11000.0, 16000.0))
#: 노치 폭 = 자리 × 이 값 × (1 − 깊이·0.5). 실측으로 고역 골의 폭/자리는 중앙 **0.028**
#: (모음, 분해능 500 Hz) ~ **0.012** (마찰, 분해능 150 Hz) 다. 0.04 로 둔다 — 그보다 좁히면
#: 한 빈짜리 "얇은 줄" 이 되어 사용자가 지적한 고역의 가는 선이 된다.
HZR_WFRAC = 0.04
HZR_RATIO = 8.0

#: **고역의 뾰족한 마루** (§51.51). 사용자: *"고역에도 나름 첨예한 피크들이 있는데 얘네는 어떻게
#: 구현할 거야?"* 영점과 완전히 대칭인 구조다 — 같은 띠, 같은 폭, 같은 젖음/마름 섞기에
#: `notch_coeffs` 대신 `peak_coeffs` 를 쓴다 (봉우리 높이 ≈ `HZP_RATIO`, 3 이면 9.5 dB).
#:
#: 왜 사다리 극의 대역폭을 좁히는 걸로는 안 되는가: 그 극들은 **시간에 고정**이라 좁히면 곧바로
#: 빗살이 된다 (§51.36 의 "층층이 쌓인 것들"). 여기 마루는 프레임마다 깊이가 0 까지 내려갈 수
#: 있어서 조음과 함께 떴다 가라앉는다.
HF_POLES = False
N_HZP = 3
HZP_WFRAC = 0.04
HZP_RATIO = 3.0

AUX_POLES = False
#: **틈마다 몇 개를 둘 것인가** (F1–F2, F2–F3, F3–F4). 실측으로 정했다 — 같은 화자 40 개 파일,
#: 모음 69,000 프레임에서 LPC 22 차(16 kHz, 극 11 개)로 배정된 포먼트 **사이에** 들어가는 극의 수:
#:
#:     F1–F2  0 개 17 %  1 개 58 %  2 개 23 %  3 개 2 %   -> 95 % 분위 **2**
#:     F2–F3  0 개 22 %  1 개 43 %  2 개 27 %  3 개 8 %   -> 95 % 분위 **3**
#:     F3–F4  0 개 22 %  1 개 54 %  2 개 22 %  3 개 2 %   -> 95 % 분위 **2**
#:
#: 중앙값은 어느 틈이나 1 이고, 0 인 프레임이 17~22 % 다 — 그때는 `mix` 가 0 으로 내려가 두 포먼트가
#: 그냥 이웃이 된다(사용자: *"3개보다 작으면 두 포먼트를 합친 방향으로"*). 차수를 26 으로 올리면
#: 수가 부풀지만 그건 LPC 가 기울기를 흉내내며 만든 가짜 극이라 22 차를 읽는다.
AUX_PER_GAP = (2, 3, 2)
N_AUX = sum(AUX_PER_GAP)
#: 보조 공진의 대역폭 = 손실 법칙 × 이 값. 넓게 두어야 켤 때 뾰족한 줄이 아니라 어깨로 올라온다.
AUX_BW_REL = 1.6

NASAL_EXTRA_BW = 2.0
NOISE_BW_REL = 0.15
NOISE_EXTRA_BW = 2.5
FRONT_Q_MAX = 2.5
FRONT_MODE3_GAIN = 0.5
#: v2.1: 앞공동 Q 를 **장애물(`obstacle`) 정도에 묶는다** — Q = LO + (HI − LO)·obstacle. v2.0 의 일괄 상한
#: Q ≤ 2.5 는 지나쳤다: 목표 s040 ㅊ 의 봉우리 꼭대기가 Q 5.6 (−3 dB 폭 1.95 kHz @ 10.85 kHz), 치마는
#: 넓다(−6 dB 폭 4.1 kHz). 치찰음은 봉우리의 두드러짐이 정의적 특징이다(장애물 음원, Shadle). 진짜 결함은
#: Q 자체가 아니라 파일 전역 스칼라 하나였다 — 장애물 궤적을 따르면 음소마다 다르다.
FRONT_Q_OBSTACLE = False
FRONT_Q_LO, FRONT_Q_HI = 1.5, 6.0

#: 앞공동 Q 를 **제어값 `front_q`**(1~8, 20 ms 층)로 둔다 (§51.8). 장애물 연동(`FRONT_Q_OBSTACLE`)은 Q 를
#: 쌍극자 몫이라는 다른 일을 하는 값에 묶어, 적합기가 쌍극자를 낮게 두는 파일(s040, 장애물 0.096)에서
#: Q ≈ 1.9 로 눌렸다 — 치찰음 봉우리가 5~10 kHz 로 넓게 퍼지고 3~8 kHz 가 목표보다 6~13 dB 컸다. 목표 ㅊ 의
#: 봉우리 Q 는 5.6 이었다. 켜면 `FRONT_Q_OBSTACLE` 보다 먼저 쓴다.
FRONT_Q_CTRL = False

#: **화자 고정 고역 구조** (docs/FOUNDATION.md §3.3, 2 단계). 켜면 F5 위 극들이 F4 를 따라 움직이지 않는다.
#:
#: 사다리(F_K + n·c/2L)는 F4 에 매달려, 적합기가 F4 를 흔들면 5~16 kHz 극 열 개가 통째로 흔들려 고역을
#: 썰었고(§50.9: 치찰음 20~60 Hz 과잉의 주범이 F3·F4), 좁아지면 평평한 줄을 세웠다(§50.5). 실측 문헌은
#: 하인두(후두관 + 이상와)가 **모음에 무관하게 안정하고 화자 간 차이가 크다**고 한다 — 약 2.5 kHz 위
#: 스펙트럼의 화자성이다(Kitamura et al. 2005, Delvaux 2014). 그래서 고역 극은 시간에 따라 움직이지 않는
#: **화자 프로파일**로 둔다: 균일관 절대 위치 (2k−1)c/4L 에서 극마다 전역 로그 이동(±`HF_DF_LIM`),
#: 대역폭은 **기존 법칙 그대로**(F5~F_K 는 `default_bw`, 보정 극은 `default_bw` × 한쪽 가둠 배율) × 극마다
#: 전역 배율(1~2.5 배, 좁히기 금지), 그리고 **이상와 영점** 하나(3.3~6.1 kHz, 전역 적합).
#: 처음엔 Q≈5 로 넓혀 봤다가 되돌렸다: 이 종속 가지에서 고차 극은 잘린 극 때문에 모자라는 고역을
#: **되채우는** 보정이라, 넓히면 모음 성문 경로의 10~12 kHz 가 사다리보다 −25 dB, 14~16 kHz 가 −31 dB
#: 가 됐다(합성 모음 실측) — L31 의 고역 죽음과 같은 기제다. 위치만 F4 에서 떼어 고정한다.
HF_FIXED = False

#: **고차 극 꼬리 보정** (MEASUREMENTS §51.7). 켜면 종속 가지(성문·잡음 사본)의 입력에 정적 최소위상 FIR 을
#: 걸어, 극 14 개(마지막 ≈ 16.1 kHz)에서 끊긴 디지털 캐스케이드를 **무한 개의 극을 가진 손실 균일관**의
#: 포락에 맞춘다.
#:
#: 왜: 실제 관은 극이 끝없이 이어져 극 사이 골이 0 dB 근처에 머문다(무손실이면 1/cos). 극을 유한 개에서
#: 끊으면 그 위의 극들이 저역 쪽으로 드리우던 꼬리(Σ −ln(1 − f²/F_n²))가 통째로 빠진다. 균일관 설정에서
#: 잰 디지털 캐스케이드 − 아날로그 무한 관: 8 kHz −16, 10 kHz −27, 12 kHz −40, 14 kHz −60, 16 kHz −90 dB.
#: 보정 극 여섯(§35)이 10 kHz 아래는 메웠지만 그 위는 아니었다. 그래서 모음 고역 12~18 kHz 는 배음도
#: 기식도 **구조적으로** 낼 수 없었고(목표 대비 −26~−57 dB, `L48/s040` 모음 평균), 적합기는 고차 극
#: 대역폭 배율을 상한까지 올려 그 대역을 아예 끄는 쪽으로 갔다(`log_extra_bw` +2.66). 예전 판에서 그
#: 대역을 채운 것은 1 kHz 빗살(§50.14)이었다.
#:
#: 보정 곡선 = 평활(아날로그 무한 관 dB) − 평활(디지털 캐스케이드 dB), 균일관 (2n−1)c/4L·기본 대역폭 법칙.
#: F1~F4 가 움직여도 f ≫ F_n 에서 디지털/아날로그 비는 극 위치와 무관해 정적 보정으로 충분하다.
#: `HPC_FMAX` 위는 그 값에서 고정해 나이퀴스트 쪽 이득이 폭주하지 않게 한다(그 위는 캐스케이드가 내린다).
#: **출력 쪽**에 건다(`HPC_AT_INPUT = False`). 처음에는 입력 쪽에 걸었다 — 출력 쪽이면 공명기 계수 변화의 광대역
#: 과도가 같이 커질 것으로 봤다. 그러나 입력 쪽은 **+90~+112 dB 로 키운 고역이 시변 공명기를 지나며 저역으로
#: 새어 내린다**: 성문 개방기 감쇠(F1 을 F0 주기로 흔듦)를 켜자 `phones.ara` rms 가 0.61 → 40.6, 정점 2.5 → 1984
#: 로 폭주했고(상한 17 kHz; 14 kHz 에서도 정점 23.7), 적합 궤적(`L48/s040`)만으로도 1~4 kHz 프레임 수준이 99 %
#: 분위에서 +3.8~+6.2 dB 올라갔다. 출력 쪽이면 시변 단이 만든 과도는 뒤 단들과 꼬리 극을 신호와 똑같이 지난다 —
#: 물리적인 무한 관에서 꼬리 극이 하는 일과 같다.
HPC_AT_INPUT = False
HPC_TAIL = False

#: **성문 개방기 감쇠** (MEASUREMENTS §51.9). 켜면 성문 경로의 F1 대역폭·주파수를 성문 개방 곡선 o(t)(LF 개방기의
#: sin², 0~1, 성문 주기마다)에 따라 움직인다: B1(t) = B1·(1 + k_b·o), F1(t) = F1·(1 + k_f·o).
#:
#: 왜: 성문이 열린 동안에는 성도가 성문하 공간과 이어져 F1 이 크게 감쇠하고 약간 올라간다(Fant & Liljencrants,
#: Ananthapadmanabha & Fant 1982 — 성문 손실). 닫힌 주기 동안만 F1 이 제 대역폭으로 울린다. 고정 대역폭 공명기는
#: 개방기 내내 같은 속도로 울려, 5.3 ms 창(한 주기 남짓)으로 재는 포락 점수의 저역 오차 — 모음 0~1 kHz 가 SC 분자의
#: 85~93 %(§51.2) — 를 남긴다. 예전 판(`L25`)은 1~2 ms 층의 bw1 로 이것을 흉내 냈고, 조음 대역 제한이 그 길을 막았다.
#: k_b 는 0~4 배(시그모이드, 처음 1 배), k_f 는 −0.1~+0.2(처음 0), 둘 다 화자 전역 적합값. 잡음 사본에는 걸지 않는다.
OPEN_DAMP = False
OPEN_DAMP_MAX = 4.0
HPC_TAPS = 512
HPC_FMAX = 17000.0
HPC_SMOOTH_HZ = 700.0
HF_Q = 5.0          # (쓰지 않음 — 위 주석의 되돌린 시도)
HF_DF_LIM = 0.15
PIR_F0, PIR_BW, PIR_RATIO_MAX = 4500.0, 500.0, 6.0


def _extra_mult(x: torch.Tensor) -> torch.Tensor:
    """고차 극 대역폭 배율. x=0 에서 1, 위로는 e^{HF_BW_LIM}(2.5 배)까지, 아래로는
    거의 1(0.93 배)에서 멈춘다. 매끈한 한쪽 가둠 — softplus 로 음수 쪽을 접고 tanh 로
    위를 가둔다. 꺾임이 없어 손실이 C² 로 남는다."""
    w = 0.1
    sp = w * torch.nn.functional.softplus(x / w) - w * math.log(2.0)
    return torch.exp(HF_BW_LIM * torch.tanh(sp / HF_BW_LIM))

# 고차 극 보정을 최소위상 FIR 로 되돌리는 경로가 여기 있었다. 12 kHz 에서 +160 dB 가
# 필요해 수치적으로 성립하지 않아 기각했고(ADR 0009·0012), 대역폭 법칙을 하나로
# 합치면서 그 함수가 참조하던 `extra_bw_floor`·`extra_bw_slope` 도 사라졌다.
# 기록은 ADR 에 있다 — 죽은 코드로 남겨 두면 낡은 파라미터로 되살아난다.


class VocalTract(nn.Module):
    def __init__(self, fs: float, hop: int, length_cm: float = 14.6,
                 bw_floor: float = 40.0, bw_slope: float = 0.05,
                 n_formants: int = N_FORMANTS, n_extra: int | None = None,
                 front_bw_slope: float = 0.20):
        super().__init__()
        self.fs, self.hop = float(fs), int(hop)
        self.length_cm = length_cm
        self.bw_floor, self.bw_slope = bw_floor, bw_slope
        self.K = n_formants
        # 고차 극 보정. 세 가지가 **함께** 바뀌어야 한다 (하나만 바꾸면 전부 실패한다).
        #
        #   (1) 위치를 마지막 포먼트에 **상대**로 둔다.  관의 극 간격 c/(2L) 는
        #       조음과 무관한 절대 제약이다 (웹스터 방정식의 경계조건이 정한다).
        #       고정 위치 (2n−1)c/4L 에 두면 F_K 가 피팅으로 내려갈 때 F_K 와 첫
        #       고차 극 사이가 벌어져 **구멍**이 생긴다 — 실측 8990 Hz, §7.6 의
        #       미해결 항목. 재현했다: F8=7900 일 때 −5.22 dB @ 9110 Hz.
        #       상대 배치로 바꾸면 같은 조건에서 −3.37 dB @ 6025 Hz 로, 그것도
        #       인공물이 아니라 진짜 포먼트 사이 골이다.
        #   (2) 대역폭을 손실 법칙 (`default_bw`) 으로 좁힌다.  Q≈1 은 극을
        #       실질적으로 없앴다.  **위치를 안 바꾸고 대역폭만 좁히면 더 나쁘다**
        #       (−8.18 dB) — 좁은 극이 구멍의 가장자리를 더 또렷하게 만든다.
        #   (3) 상한을 0.70·fs/2 로 둔다.  0.98 까지 채웠던 예전 시도가 +151 dB 로
        #       폭주한 것은 Klatt 공명기의 분자가 상수라 z=−1 에서 이득이
        #       (1+r)²/(1−r)² (절당 +51 dB) 이기 때문이다.  아날로그 원형에는 없는
        #       인공물이라 상한으로 막아야 한다.  상한을 쓸어 측정한 결과 (면적
        #       함수 8 종, 0.2~12 kHz 모양 오차 RMS):
        #           0.55  11.4 dB | 0.60  7.2 | 0.65  2.5 | **0.70  1.5** | 0.80  5.8 | 0.90  11.5
        #       현행 (0.60 + Q≈1) 은 26.6 dB 였다.
        #
        # 합쳐서 8~10 kHz 레벨이 −68.3 dB → −9.3 dB 로 **59 dB** 올라온다.  곁가지의
        # +20/+35 dB 고역 셸프가 메우고 있던 것이 바로 이 구멍이다.
        self.extra_cap = 0.70
        self.n_extra = 6 if n_extra is None else int(n_extra)
        self.extra_spacing = C_SOUND / (2.0 * length_cm)
        f1 = C_SOUND / (4.0 * length_cm)
        self.register_buffer("uniform_formants",
                             torch.tensor([(2 * n - 1) * f1 for n in range(1, n_formants + 1)]))
        # 학습 파라미터: 고차 극 보정의 대역폭 배율(화자 고역 손실), 앞공동 대역폭 배율
        self.log_extra_bw = nn.Parameter(torch.tensor(0.0))
        self.log_front_bw = nn.Parameter(torch.tensor(0.0))
        # 화자 고정 고역 구조 (HF_FIXED). F5~F_K 와 보정 극마다 로그 위치 이동·대역폭 배율, 이상와 영점.
        n_hf = max(0, n_formants - 4) + (6 if n_extra is None else int(n_extra))
        self.hf_log_df = nn.Parameter(torch.zeros(n_hf))
        self.hf_log_bw = nn.Parameter(torch.zeros(n_hf))
        self.hf_eq_db = nn.Parameter(torch.zeros(HF_EQ_N))     # 고역 포락 EQ (HF_EQ)
        self.pir_log_f = nn.Parameter(torch.tensor(0.0))
        self.pir_depth = nn.Parameter(torch.tensor(-2.0))     # 시그모이드 — 처음엔 얕게
        # 성문 개방기 감쇠 (OPEN_DAMP): k_b = 4·sigmoid(x) (x = −ln 3 → 1 배), k_f = 0.15·tanh(y) + 0.05
        self.open_damp = nn.Parameter(torch.tensor(-math.log(3.0)))
        self.open_f1 = nn.Parameter(torch.tensor(-0.3466))    # 0.15·tanh(−0.3466)+0.05 ≈ 0
        # 앞공동 극의 대역폭 = 500 + 이 값 × f_p. **화자 프로파일이 준다** — 한 상수로
        # 두면 두 화자가 반대로 잡아당긴다. 실측 적합: 남 /ㅅ/ 0.08, 여 /ㅆ/ 0.36.
        self.front_bw_slope = float(front_bw_slope)
        self.nasal_zfrac = 0.80         # 측지 영점의 대역폭 = fz × 이 값 (깊이·폭)
        self.nasal_gain = 1.60          # 머머가 모음보다 −7 dB (실측 M −6.7) 이 되는 값
        # 비강도 관이다 — 극 3 개로 자르면 2 kHz 위가 −60 dB 로 무너진다(측정). 인두+비강
        # 20 cm 의 c/2L ≈ 875 Hz 간격으로 나이퀴스트까지 채운다 (ADR 0009 와 같은 이유).
        nsp = C_SOUND / (2.0 * 20.0)
        self.register_buffer("nasal_extra", torch.tensor(
            [2400.0 + k * nsp for k in range(1, 64) if 2400.0 + k * nsp < 0.70 * fs / 2],
            dtype=torch.float32))


    # ------------------------------------------------------------ 계수
    def default_bw(self, f):
        """극의 물리 기본 대역폭 (Klatt: B1 50~80, B4 175~280).

        고차 극은 이 법칙을 쓰지 않는다 — `_extra_cascade` 가 훨씬 센 감쇠를 쓴다.
        그 경계에서 Q 가 18 -> 0.9 로 급변하고, 거기에 −11.5 dB 골이 생긴다(실측:
        남성 8.5 kHz. 그 주파수는 (2K−1)·c/4L 이라 화자마다 다르다 — 코드 상수 K 가
        만든 인공물이다). **두 법칙을 하나의 매끈한 식으로 합치는 것을 시도했다가
        되돌렸다**: 고차 극이 날카로워지면서 종속 이득이 8~10 kHz 에서 +33.8 dB 가
        되고(남성), 복사합성이 그 리프트에 이득을 맞추느라 230 Hz 가 −27 dB 로
        무너졌다 (남 /사/ 포락 68.6 % -> 11.5 %). 무릎을 첫 고차 극에 묶어 Q≈1 로
        낮춰도 이번엔 9~13 kHz 를 메우는 힘이 6 dB -> 3.6 dB 로 약해졌다.
        골은 남은 문제로 MEASUREMENTS §7.6 에 적어 둔다.
        """
        return self.bw_floor + self.bw_slope * f
    # 사다리 혼합비. `w_k = 1/(1 + LADDER_BETA·(k − k_last))` — 가까운 극은 실측된
    # 마지막 포먼트를 따르고, 먼 극은 성도 길이가 정한 절대 위치로 수렴한다.
    LADDER_W = 0.6
    # 이웃 극의 최소 간격 / (c/2L). 실제 관 300 종에서 모드 4 이상의 간격은 최소
    # 0.256, 1 % 분위 0.629 였다 (F4→F5 만 1 % 분위 0.382). 0.30 은 그 아래라
    # 평소에는 걸리지 않는 **안전망**이다 — F4 가 절대 앵커보다 훨씬 위인 프레임에서
    # 혼합이 F5 를 F4 쪽으로 끌어내리는 것만 막는다.
    LADDER_MIN_GAP = 0.30

    def ladder_pole(self, prev, k: int, like):
        """실측되지 않은 k 번째 극의 위치. 두 앵커를 섞는다.

        * **상대 앵커** `F_{k−1} + c/(2L)` — 조음을 따라간다. 가까운 극에 정확하지만
          멀어질수록 오차가 쌓인다.
        * **절대 앵커** `(2k−1)·c/(4L)` — 성도 길이가 정한다. 조음을 못 따라가지만
          오차가 쌓이지 않는다.

        면적 함수 400 종으로 잰 모드 5~15 의 예측 오차 표준편차 [Hz] (§41):

            상대만 (w=1)   110 124 157 181 200 214 226 235 243 249 255   <- 발산한다
            절대만 (w=0)   251 211 177 152 132 118 106  97  89  82  77
            **혼합 w=0.6**  78  94 106 109 107 103  97  91  86  80  76

        혼합은 두 앵커 중 어느 쪽보다도 낫고, 무엇보다 **발산하지 않는다** — 상대
        앵커만 쓰면 오차가 모드마다 쌓여 255 Hz 까지 간다. 시드와 면적 변동 폭을
        바꿔도 같다. 부수 효과가 하나 더 있다: 마지막 자유 포먼트가 위쪽 극 전체를
        끌고 다니지 않게 된다. 무너진 창의 책임 귀속에서 `f4` 가 기울기의 36 % 를
        쥐고 있었는데 (§41.2), 한 단계마다 0.6 씩 줄어드니 F8 에 미치는 영향은
        0.6⁴ = 0.13 이다.

        **절대 앵커는 프로파일의 `tract_length_cm` 을 믿는다.** 그 값이 틀리면 먼 극이
        통째로 치우친다 — 화자 프로파일을 잡을 때 확인할 것.
        """
        ab = (2 * k - 1) * C_SOUND / (4.0 * self.length_cm)
        if prev is None:
            return torch.full_like(like, ab)
        w = self.LADDER_W
        f = w * (prev + self.extra_spacing) + (1.0 - w) * ab
        gap = self.LADDER_MIN_GAP * self.extra_spacing
        return prev + gap + self._soft_over(f - prev - gap, 0.2 * gap)

    def _up(self, v):
        return frames_to_samples(v.unsqueeze(-1), self.hop)[..., 0][:, :self._n_emit]

    def _formant_tracks(self, c: dict):
        """(f_k, bw_k) 샘플률 리스트. 0 이면 균일관 기본값 / 물리 기본 대역폭.

        **기본값 대치는 프레임률에서** 한 뒤 샘플률로 보간한다. 샘플률에서 하면
        0 ↔ 값 사이를 선형 보간하며 F1 이 10 Hz 를 지나간다(첫 렌더의 폭주 원인 2).
        """
        out = []
        prev = None
        for k in range(1, self.K + 1):
            f = c[f"f{k}"]
            # **원소별로** 고른다. 텐서 값으로 파이썬 분기를 걸면 청크 렌더와 통짜
            # 렌더가 갈라진다 (`test_streaming_equals_offline` 이 그것을 잡았다).
            if HF_FIXED and k > 4:
                # 화자 고정 고역 극 — 시간에 따라 움직이지 않는다.
                i = k - 5
                ab = (2 * k - 1) * C_SOUND / (4.0 * self.length_cm)
                fk = ab * torch.exp(HF_DF_LIM * torch.tanh(self.hf_log_df[i] / HF_DF_LIM))
                bk = self.default_bw(fk) * _extra_mult(self.hf_log_bw[i])
                f = torch.where(f > 0, f, fk.to(f.dtype).expand_as(f))
                prev = f
                bw = c[f"bw{k}"]
                bw = torch.where(bw >= 20.0, bw, bk.to(f.dtype).expand_as(f))
                out.append((self._up(f), self._up(bw)))
                continue
            f = torch.where(f > 0, f, self.ladder_pole(prev, k, f))
            prev = f
            bw = c[f"bw{k}"]
            bw = torch.where(bw >= 20.0, bw, self.default_bw(f))   # 20 Hz 미만 = 기본
            if k == 1:      # 연구개가 열리면 F1 이 넓어진다 (에너지가 비강으로 샌다)
                bw = bw + 120.0 * c["velum"]
            out.append((self._up(f), self._up(bw)))
        return out

    # ------------------------------------------------------------ 단계
    def _smooth_track(self, x, key, state):
        """샘플률 궤적 (B, N) 에 2 단 1 차 저역 (`TRACK_SMOOTH_MS`). 상태는 `state[key_p]`."""
        a = 1.0 - math.exp(-1000.0 / (TRACK_SMOOTH_MS * self.fs))
        for p in (0, 1):
            k = f"{key}_{p}"
            zi = state.get(k)
            if zi is None:           # 첫 값에서 정상상태로 시작한다 (0 → f 과도가 없게)
                x0 = x[:, :1].to(torch.float64)
                zi = torch.cat([(1.0 - a) * x0, torch.zeros_like(x0)], -1)
            x, state[k] = tv_biquad(x, a, 0.0, 0.0, -(1.0 - a), 0.0, zi=zi)
        return x

    def _uniform_digital_db(self, f: np.ndarray, noise: bool) -> np.ndarray:
        """균일관 설정의 종속 캐스케이드(K 극 + 보정 극) 크기응답 [dB] — `_cascade`·`_extra_cascade` 와 같은 식."""
        from .tviir import resonator_coeffs as _rc
        f1 = C_SOUND / (4.0 * self.length_cm)
        zi = np.exp(-1j * 2.0 * np.pi * f / self.fs)
        poles = []
        for n in range(1, self.K + 1):
            fn = (2 * n - 1) * f1
            bn = float(self.default_bw(torch.tensor(fn, dtype=torch.float64)))
            poles.append((fn, max(bn, NOISE_BW_REL * fn) if noise else bn))
        f_last, cap = poles[-1][0], self.extra_cap * self.fs / 2.0
        bx = NOISE_EXTRA_BW if noise else 1.0
        for _ in range(self.n_extra):
            raw = f_last + self.extra_spacing
            f_last = raw
            fk = float(self._soft_cap(torch.tensor(raw, dtype=torch.float64), cap))
            poles.append((fk, (float(self.default_bw(torch.tensor(fk, dtype=torch.float64))) + (raw - fk)) * bx))
        h = np.ones_like(zi)
        for fn, bn in poles:
            b0, _, _, a1, a2 = (float(np.asarray(v)) for v in _rc(torch.tensor(fn, dtype=torch.float64),
                                                                 torch.tensor(bn, dtype=torch.float64), self.fs))
            h = h * b0 / (1.0 + a1 * zi + a2 * zi * zi)
        return 20.0 * np.log10(np.abs(h) + 1e-300)

    def _analog_tube_db(self, f: np.ndarray, n_max: int = 4000) -> np.ndarray:
        """극이 끝없이 이어진 손실 균일관 (아날로그, DC 정규화 극의 곱) [dB]."""
        f1 = C_SOUND / (4.0 * self.length_cm)
        n = np.arange(1, n_max)[:, None]
        fn = (2 * n - 1) * f1
        bn = np.asarray(self.default_bw(torch.tensor(fn, dtype=torch.float64)), dtype=np.float64)
        return (-20.0 * np.log10(np.abs(1.0 - (f / fn) ** 2 + 1j * f * bn / fn ** 2))).sum(0)

    def _hpc_kernel(self, kind: str, device, dtype) -> torch.Tensor:
        """고차 극 꼬리 보정 최소위상 FIR (탭 `HPC_TAPS`). kind = 'g'(성문) / 'n'(잡음 사본). 한 번 만들고 둔다."""
        cache = self.__dict__.setdefault("_hpc_cache", {})
        key = (kind, HPC_TAPS, HPC_FMAX, HPC_SMOOTH_HZ, str(device), dtype)
        if key in cache:
            return cache[key]
        nfft = 8192
        f = np.arange(nfft // 2 + 1) * self.fs / nfft
        k = max(1, int(round(HPC_SMOOTH_HZ / (self.fs / nfft))))
        sm = lambda v: np.convolve(np.pad(v, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid")
        r = sm(self._analog_tube_db(f)) - sm(self._uniform_digital_db(f, kind == "n"))
        i0 = int(round(HPC_FMAX / (self.fs / nfft)))
        r[i0:] = r[i0]
        # 최소위상 (실 켑스트럼 접기)
        lg = np.log(10.0 ** (r / 20.0))
        c = np.fft.ifft(np.concatenate([lg, lg[-2:0:-1]])).real
        fold = np.zeros_like(c)
        fold[0], fold[nfft // 2] = c[0], c[nfft // 2]
        fold[1:nfft // 2] = 2.0 * c[1:nfft // 2]
        h = np.fft.ifft(np.exp(np.fft.fft(fold))).real[:HPC_TAPS]
        nt = HPC_TAPS // 4
        h[-nt:] *= 0.5 * (1.0 + np.cos(np.pi * np.arange(nt) / nt))
        ker = torch.as_tensor(h[::-1].copy(), dtype=dtype, device=device).view(1, 1, -1)
        cache[key] = ker
        return ker

    def _hpc(self, x, kind: str, state):
        """정적 FIR 을 상태(앞 탭−1 샘플)와 함께 건다 — 스트리밍과 오프라인이 같다."""
        ker = self._hpc_kernel(kind, x.device, x.dtype)
        m = ker.shape[-1] - 1
        key = f"hpc_{kind}"
        prev = state.get(key)
        if prev is None:
            prev = torch.zeros(x.shape[0], m, dtype=x.dtype, device=x.device)
        xx = torch.cat([prev.to(x.dtype), x], -1)
        state[key] = xx[:, -m:].detach() if not xx.requires_grad else xx[:, -m:]
        return torch.nn.functional.conv1d(xx.unsqueeze(1), ker)[:, 0]

    def _cascade(self, x, tracks, state, prefix):
        for i, (f, bw) in enumerate(tracks):
            key = f"{prefix}{i}"
            x, state[key] = tv_biquad(x, *resonator_coeffs(f, bw, self.fs), zi=state.get(key))
        return x

    @staticmethod
    def _soft_over(x, w: float):
        """0 이상만 남기는 부드러운 문턱. `w·softplus(x/w)`."""
        return w * torch.nn.functional.softplus(x / w)

    @staticmethod
    def _soft_cap(x, cap, w: float = 400.0):
        """상한에 부드럽게 붙는 최소값. 상한을 넘는 극들이 **겹치지 않게** 한다.

        `clamp` 를 쓰면 상한 위의 극이 전부 같은 자리에 쌓여 거기 날카로운 봉우리가
        선다 (F8 이 11.9 kHz 까지 가는 프레임이 있다). 이 식은 순증가라 순서를
        보존하면서 [cap − w·ln2, cap) 안으로 압축한다. 미분 가능하다.
        """
        return cap - w * torch.nn.functional.softplus((cap - x) / w)

    def _extra_cascade(self, x, tracks, state, prefix: str = "x", bw_extra: float = 1.0):
        """마지막 포먼트 위로 c/(2L) 간격의 극을 이어 붙인다.

        `tracks` 의 마지막 항목이 F_K 다 — 위치가 거기에 묶여 있으므로 F_K 가
        피팅으로 움직여도 그 위에 구멍이 남지 않는다.  상한에 닿으면 clamp 하지만,
        정상적인 F_K (~9 kHz) 에서는 마지막 극이 16.2 kHz 라 걸리지 않는다.
        """
        if self.n_extra <= 0:
            return x
        bw_scale = _extra_mult(self.log_extra_bw)
        f_last = tracks[-1][0]
        cap = self.extra_cap * self.fs / 2.0
        if HF_FIXED:
            # 화자 고정 보정 극: 절대 위치 (2k−1)c/4L, 전역 이동·대역폭. F_K 를 따라가지 않는다.
            j0 = max(0, self.K - 4)
            ref = x
            for i in range(self.n_extra):
                k = self.K + 1 + i
                ab = (2 * k - 1) * C_SOUND / (4.0 * self.length_cm)
                raw = ab * torch.exp(HF_DF_LIM * torch.tanh(self.hf_log_df[j0 + i] / HF_DF_LIM))
                fk = self._soft_cap(raw, cap)
                bw = ((self.default_bw(fk) + (raw - fk)) * bw_scale
                      * _extra_mult(self.hf_log_bw[j0 + i]) * bw_extra)
                fk_t, bw_t = fk.to(ref.dtype).expand_as(ref), bw.to(ref.dtype).expand_as(ref)
                x, state[f"{prefix}{i}"] = tv_biquad(x, *resonator_coeffs(fk_t, bw_t, self.fs),
                                                     zi=state.get(f"{prefix}{i}"))
            # 이상와 영점 (극-영점 쌍 노치, 깊이 = 대역폭 비)
            fz = PIR_F0 * torch.exp(0.3 * torch.tanh(self.pir_log_f / 0.3))
            ratio = 1.0 + (PIR_RATIO_MAX - 1.0) * torch.sigmoid(self.pir_depth)
            fz_t = fz.to(ref.dtype).expand_as(ref)
            x, state[f"{prefix}pir"] = tv_biquad(x, *notch_coeffs(fz_t, PIR_BW, self.fs, ratio),
                                                 zi=state.get(f"{prefix}pir"))
            return x
        for i in range(self.n_extra):
            key = f"{prefix}{i}"
            # **여기서는 순수 사다리다** (혼합하지 않는다). F5~F8 은 들리는 포먼트라
            # 위치 정확도가 중요해 절대 앵커를 섞지만, 보정 극에서 중요한 것은
            # 정확도가 아니라 **틈이 없는 것**이다 — 틈은 −5 dB 짜리 구멍을 만들고
            # (§35.3), 12 kHz 짜리 넓은 극이 150 Hz 어긋나는 것은 안 들린다.
            raw = f_last + self.extra_spacing
            f_last = raw
            fk = self._soft_cap(raw, cap)
            # 상한에 눌린 만큼 대역폭을 넓힌다. 안 그러면 눌린 극들이 상한 근처에
            # 25 Hz 간격으로 쌓여 거기 날카로운 봉우리가 선다 (F8 이 11.9 kHz 인
            # 프레임에서 실제로 그렇다). 상한은 1 차원 관 모형이 끝나는 자리라
            # 거기서 손실이 커지는 것은 물리적으로도 맞다 (횡모드·벽 손실).
            bw = (self.default_bw(fk) + (raw - fk)) * bw_scale * bw_extra
            x, state[key] = tv_biquad(x, *resonator_coeffs(fk, bw, self.fs), zi=state.get(key))
        return x

    def _front_cavity(self, x, c, state):
        """협착 하류 앞공동 극 + 뒤공동 영점. 치찰음 지문이 여기 있다."""
        L = self.length_cm
        fl = c["front_len"]
        l_front = self._up(torch.where(fl > 0, fl, (1.0 - c["c_place"]) * L).clamp(0.3, L))
        f_p = (C_SOUND / (4.0 * l_front)).clamp(500.0, 0.45 * self.fs)     # 1/4 파장
        bw_p = (500.0 + self.front_bw_slope * f_p) * torch.exp(
            HF_BW_LIM * torch.tanh(self.log_front_bw / HF_BW_LIM))
        l_back = (L - l_front).clamp_min(1.0)
        f_z = (C_SOUND / (2.0 * l_back)).clamp(300.0, 0.45 * self.fs)      # 뒤공동 반공진
        bw_z = 250.0 + 0.1 * f_z
        if NOISE_V2:
            # 병렬 저 Q 앞공동 (1 차·3 차 모드). 극 주파수는 C1 로 평활한다.
            if TRACK_SMOOTH_MS > 0.0:
                f_p = self._smooth_track(f_p, "sfp", state)
                f_z = self._smooth_track(f_z, "sfz", state)
                bw_p = self._smooth_track(bw_p, "sbp", state)
            f3 = (3.0 * f_p).clamp(max=0.45 * self.fs)
            if FRONT_Q_CTRL:
                q = self._up(c["front_q"]).clamp(1.0, 8.0)
                if TRACK_SMOOTH_MS > 0.0:
                    q = self._smooth_track(q, "sfq", state)
                bw_p, bw3 = f_p / q, f3 / q
            elif FRONT_Q_OBSTACLE:
                obs = self._up(c["obstacle"]).clamp(0.0, 1.0)
                if TRACK_SMOOTH_MS > 0.0:
                    obs = self._smooth_track(obs, "sob", state)
                q = FRONT_Q_LO + (FRONT_Q_HI - FRONT_Q_LO) * obs
                bw_p, bw3 = f_p / q, f3 / q
            else:
                bw_p = torch.maximum(bw_p, f_p / FRONT_Q_MAX)
                bw3 = torch.maximum(500.0 + self.front_bw_slope * f3, f3 / FRONT_Q_MAX)
            y1, state["fp"] = tv_biquad(x, *resonator_coeffs(f_p, bw_p, self.fs), zi=state.get("fp"))
            y3, state["fp3"] = tv_biquad(x, *resonator_coeffs(f3, bw3, self.fs), zi=state.get("fp3"))
            x = y1 + FRONT_MODE3_GAIN * y3
            x, state["fz"] = tv_biquad(x, *notch_coeffs(f_z, bw_z, self.fs, 3.0), zi=state.get("fz"))
            return x
        x, state["fp"] = tv_biquad(x, *resonator_coeffs(f_p, bw_p, self.fs), zi=state.get("fp"))
        # 영점은 반드시 극-영점 노치로: DC 정규화 영점쌍은 1.3 kHz 영점이 고역을 +40 dB
        # 들어올린다(측정: 앞공동 경로 13~20 kHz 가 지배적이었던 원인).
        x, state["fz"] = tv_biquad(x, *notch_coeffs(f_z, bw_z, self.fs, 3.0), zi=state.get("fz"))
        return x

    def _allpass(self, x, c, state):
        for k in range(1, N_ALLPASS + 1):
            f, r = c[f"ap{k}_f"], c[f"ap{k}_r"]
            if float(f.detach().abs().max()) == 0.0:
                continue
            r = torch.where(f > 0, r, torch.zeros_like(r)).clamp(0.0, 0.98)
            f = torch.where(f > 0, f, torch.full_like(f, 1000.0))
            f, r = self._up(f), self._up(r)
            x, state[f"ap{k}"] = tv_biquad(x, *allpass_coeffs(f, r, self.fs), zi=state.get(f"ap{k}"))
        return x

    def _nasal_branch(self, x, c, state):
        """비강 분기 — 성문 소스에서 **갈라져 나와 콧구멍으로 방사**된다 (병렬 경로).

        v1·v2 초판은 비강을 구강 캐스케이드 **뒤에 얹은 필터**로 두었다. 그러면 구강이
        닫힐 때(tract_gain·a_c) 저역까지 같이 죽어 모음↔비음 경계가 계단이 된다.
        실측(같은 화자, 5 ms): 80~300 Hz 는 모음과 머머 사이에서 **±2 dB 로 연속**이고
        0.8~2.5 kHz 만 8~11 dB 떨어진다. 병렬 분기라야 그 성질이 나온다.

        구조 (Fant 1960 / Fujimura 1962 의 표준):
            인두 + 비강의 극 3 개  ×  닫힌 구강이 만드는 **측지 영점** 1 개
        영점 주파수는 폐쇄 위치가 정한다 — 구강 측지가 길수록(ㅁ) 낮고 짧을수록(ㅇ) 높다.
        비강은 점막 손실이 커서 대역폭이 넓다(`nasal_damp`).

        모음의 **비음화**는 따로 만들지 않는다. 연구개가 열린 채 구강도 열려 있으면 두
        분기가 그냥 더해지고, 그 간섭이 F1 부근의 극-영점 쌍으로 나타난다.
        """
        if float(c["velum"].detach().max()) <= 1e-4:
            return torch.zeros_like(x)
        d = self._up(c["nasal_damp"])
        y = x
        for i, (key, bw) in enumerate((("nasal_f", 130.0), ("nasal_f2", 300.0),
                                       ("nasal_f3", 500.0))):
            f = self._up(c[key])
            y, state[f"nb{i}"] = tv_biquad(y, *resonator_coeffs(f, bw * d, self.fs),
                                           zi=state.get(f"nb{i}"))
        for j, f in enumerate(self.nasal_extra):
            fk = f.to(y.dtype).expand_as(y)
            y, state[f"nx{j}"] = tv_biquad(
                y, *resonator_coeffs(fk, (300.0 + 0.25 * fk) * d * NASAL_EXTRA_BW, self.fs),
                zi=state.get(f"nx{j}"))
        # **극쌍 보정이 있는 노치를 쓴다.** `antiresonator_coeffs` 는 영점쌍만 DC 에서
        # 정규화하므로 먼 대역이 통째로 들린다 — 실측(fz 1400 Hz, bw 200~1400):
        #   4 kHz +15~17 dB,  12 kHz +33~35 dB,  20 kHz +39~41 dB
        # `notch_coeffs` 는 같은 각도에 4 배 넓은 극쌍을 두어 노치 바깥을 1 로 되돌린다
        # (같은 값에서 12 kHz +0.6~+11.6 dB). 그 함수의 주석이 경고하던 바로 그 경우인데
        # 비강 분기만 옛 형태로 남아 있었다 (MEASUREMENTS §34).
        fz = self._up(c["nasal_z"])
        y, state["nbz"] = tv_biquad(y, *notch_coeffs(fz, fz * self.nasal_zfrac * d,
                                                     self.fs), zi=state.get("nbz"))
        return y * self.nasal_gain * self._up(c["nasal_gain"])

    def _aux_poles(self, x, c, tracks, state):
        """**켰다 끄는 보조 공진** — 이웃 포먼트 사이에 극 하나를 연속으로 띄운다 (§51.40).

        사용자: *"포먼트를 8개 정도 만들어서 그 중 4개 정도만 2개 간격으로 살려놓고, 분기가 생기면
        그 사이에 있던 죽은 포먼트를 켜서 점진적으로 합류할 수 있도록 하는 거야."*

        왜: 지금 극의 개수가 고정이라, 목표에서 두 포먼트가 **합쳐졌다 갈라질 때** 적합기는 있는 극을
        끌고 건너가는 수밖에 없다. 그것이 사용자가 들은 과도한 무브먼트다 — 실측으로 목표의 이웃
        간격이 가장 좁은 20 % 구간에서 우리 제어의 속도가 가장 먼 20 % 보다 **1.7~5.8 배** 빠르다.
        사이에 잠자는 극이 있으면 새 공진이 제자리에서 서서히 떠오르면 되고 이웃은 가만히 있으면 된다.

        구현은 `_lateral` 과 같은 **젖음/마름 섞기** 다 — `y = x + mix·(reson(x) − x)`. mix=0 이면
        **정확히 항등**이라 출발점(mix 의 기본값 0)이 전혀 바뀌지 않는다. 주파수는 자유도로 두지 않고
        이웃 두 포먼트의 **기하평균**에 매단다 — §40 에서 F5~F8 을 자유로 풀었다가 열이 32 → 40 이
        되어 무너진 전례가 있다. 대역폭은 그 자리의 손실 법칙(`default_bw`)이다.
        """
        keys = [f"aux{k}_mix" for k in range(1, N_AUX + 1)]
        if not any(_live(c[k]) for k in keys if k in c):
            return x
        # 틈마다 `AUX_PER_GAP[j]` 개를 **로그 등간격**으로 놓는다 — 포먼트는 로그 축에서 고르다.
        slots = [(j, i + 1, m + 1) for j, m in enumerate(AUX_PER_GAP) for i in range(m)]
        y = x
        for key, (j, i, m1) in zip(keys, slots):
            if key not in c:
                continue
            if not _live(c[key]):
                continue
            mix = self._up(c[key])                       # 시그모이드가 이미 (-0.3, 1.0) 로 묶는다
            fa, _ = tracks[j]
            fb, _ = tracks[j + 1]
            fa, fb = fa.clamp_min(1.0), fb.clamp_min(1.0)
            fk = fa * (fb / fa) ** (i / m1)
            bw = self.default_bw(fk) * AUX_BW_REL
            w, state[key] = tv_biquad(y, *resonator_coeffs(fk, bw, self.fs),
                                       zi=state.get(key))
            y = y + mix * (w - y)
        return y

    def _hf_eq(self, x, state):
        """고역 포락 EQ — 로그 등간격 피킹 필터 `HF_EQ_N` 개. `HF_EQ` 참조."""
        if not HF_EQ or not hasattr(self, "hf_eq_db"):
            return x
        g = HF_EQ_LIM * torch.tanh(self.hf_eq_db / HF_EQ_LIM)
        if float(g.detach().abs().max()) < 1e-4:
            return x
        fc = torch.logspace(math.log10(HF_EQ_LO), math.log10(min(HF_EQ_HI, 0.95 * self.fs / 2)),
                            HF_EQ_N, dtype=torch.float64)
        y = x
        for i in range(HF_EQ_N):
            A = torch.pow(torch.tensor(10.0, dtype=torch.float64), g[i].double() / 40.0)
            w0 = 2.0 * math.pi * float(fc[i]) / self.fs
            al = math.sin(w0) / (2.0 * HF_EQ_Q)
            cw = math.cos(w0)
            a0 = 1.0 + al / A
            b0, b1, b2 = (1.0 + al * A) / a0, (-2.0 * cw) / a0, (1.0 - al * A) / a0
            a1, a2 = (-2.0 * cw) / a0, (1.0 - al / A) / a0
            e = torch.ones_like(y)
            y, state[f"hfeq{i}"] = tv_biquad(y, b0 * e, b1 * e, b2 * e, a1 * e, a2 * e,
                                             zi=state.get(f"hfeq{i}"))
        return y

    def _aux_zeros(self, x, c, tracks, state):
        """**켰다 끄는 보조 영점** — 이웃 포먼트 사이에서 소리를 죽인다 (§51.46).

        `_lateral`·`_aux_poles` 와 같은 젖음/마름 섞기: `y = x + mix·(notch(x) − x)`.
        mix=0 이면 **정확히 항등**이다. 자리는 `azr{k}_f` 가 0 이면 이웃 두 포먼트의 기하평균.
        """
        keys = [f"azr{k}_mix" for k in range(1, N_AZR + 1)]
        if not any(_live(c[k]) for k in keys if k in c):
            return x
        y = x
        for j, key in enumerate(keys):
            if key not in c:
                continue
            if not _live(c[key]):
                continue
            mix = self._up(c[key])                       # 위와 같다 — 여기서 clamp 하면 0 에서 기울기가 죽는다
            fa, _ = tracks[j]
            fb, _ = tracks[j + 1]
            fa, fb = fa.clamp_min(1.0), fb.clamp_min(1.0)
            fz = torch.sqrt(fa * fb)
            fkey = f"azr{j + 1}_f"
            if fkey in c:                                # 배율 1 이 기하평균 — 늘 그래프에 있다
                fz = torch.clamp(fz * self._up(c[fkey]), fa * 1.05, fb * 0.95)
            bw = fz * AZR_WFRAC * (1.0 - 0.5 * mix).clamp_min(0.2)
            w, state[key] = tv_biquad(y, *notch_coeffs(fz, bw, self.fs, AZR_RATIO),
                                      zi=state.get(key))
            y = y + mix * (w - y)
        return y

    def _hf_zeros(self, x, c, state):
        """**고역 영점** — F4 위의 좁고 깊은 골을 판다 (§51.50). `HF_ZEROS` 참조.

        `_aux_zeros` 와 같은 젖음/마름 섞기 `y = x + mix·(notch(x) − x)` 라 깊이 0 이면 항등이다.
        포먼트에 매달지 않고 제 띠(`HZR_BANDS`) 안에서 자리를 스스로 찾는다 — 목표의 골은
        규칙적인 빗살이 아니다(간격 변동계수 0.83).
        """
        keys = [f"hzr{k}_mix" for k in range(1, N_HZR + 1)]
        if not any(_live(c[k]) for k in keys if k in c):
            return x
        y = x
        for j, key in enumerate(keys):
            if key not in c or not _live(c[key]):
                continue
            mix = self._up(c[key])
            lo, hi = HZR_BANDS[j]
            fkey = f"hzr{j + 1}_f"
            fz = self._up(c[fkey]).clamp(lo, hi) if fkey in c else                 torch.full_like(mix, math.sqrt(lo * hi))
            bw = fz * HZR_WFRAC * (1.0 - 0.5 * mix).clamp_min(0.2)
            w, state[key] = tv_biquad(y, *notch_coeffs(fz, bw, self.fs, HZR_RATIO),
                                      zi=state.get(key))
            y = y + mix * (w - y)
        return y

    def _hf_peaks(self, x, c, state):
        """**고역의 뾰족한 마루** — `_hf_zeros` 의 대칭짝 (§51.51). `HF_POLES` 참조."""
        keys = [f"hzp{k}_mix" for k in range(1, N_HZP + 1)]
        if not any(_live(c[k]) for k in keys if k in c):
            return x
        y = x
        for j, key in enumerate(keys):
            if key not in c or not _live(c[key]):
                continue
            mix = self._up(c[key])
            lo, hi = HZR_BANDS[j]
            fkey = f"hzp{j + 1}_f"
            fp = self._up(c[fkey]).clamp(lo, hi) if fkey in c else                 torch.full_like(mix, math.sqrt(lo * hi))
            bw = fp * HZP_WFRAC * (1.0 - 0.5 * mix).clamp_min(0.2)
            w, state[key] = tv_biquad(y, *peak_coeffs(fp, bw, self.fs, HZP_RATIO),
                                      zi=state.get(key))
            y = y + mix * (w - y)
        return y

    def _lateral(self, x, c, state):
        """설측음의 측지 영점 2 개. 깊이는 `lat_mix` 로 **연속으로** 켜고 끈다.

        **깊이는 젖음/마름 섞기로 준다** — `y = x + mix·(notch(x) − x)`.
        mix=0 이면 항등이고 mix=1 이면 노치 그대로다. 주파수를 0 으로 껐다 켜는 방식은
        로그 파라미터의 영차 유지 때문에 한 프레임에 도약하고, 그 계수 도약이 필터
        상태와 만나 클릭이 되므로 쓰지 않는다. 섞기는 mix 에 대해 연속이라 그 문제가 없다.

        **예전에는 대역폭을 `lat_bw / mix` 로 넓혀서 껐다. 그건 틀렸다.**
        주석에 "대역폭이 발산하면 응답이 정확히 1" 이라고 적어 두었는데 아니다.
        `notch_coeffs` 는 영점쌍(반지름 rz)과 4 배 넓은 극쌍(rp)을 두고 **DC 에서**
        정규화한다. 대역폭을 키우면 둘 다 원점으로 오그라드는데 극이 4 배 빨리
        오그라들므로, DC 는 1 이어도 **고역이 들린다.** 실측(적합값 lat_z 3300 Hz,
        lat_mix 0.065 → 실효 대역폭 4554 Hz):

            250 Hz −0.02 | 1 k −0.26 | 4 k −0.07 | 8 k **+7.4** | 12 k **+11.0** | 16 k **+12.6** dB

        영점이 둘이라 12~16 kHz 에서 +20 dB 를 넘는다. 적합기는 이것을 **공짜 고역
        셸프**로 썼다 — 파찰음 /ㅊ/ 구간에서 설측을 끄면 에너지가 9.5 dB 줄고 파형
        첨도가 14.15 → 9.45 로 내려간다(목표 3.49). 사용자가 "누가 봐도 튀는 파형"
        이라고 지적한 것이 이것이다 (MEASUREMENTS §34).
        """
        if float(c["lat_mix"].detach().max()) <= 1e-3:
            return x
        mix = self._up(c["lat_mix"]).clamp(0.0, 1.0)
        y = x
        for k in (1, 2):
            fz = c[f"lat_z{k}"]
            if float(fz.detach().max()) <= 0.0:
                continue
            fz = torch.where(fz > 0, fz, torch.full_like(fz, 3000.0))
            y, state[f"lz{k}"] = tv_biquad(y, *notch_coeffs(self._up(fz),
                                                            self._up(c["lat_bw"]), self.fs),
                                           zi=state.get(f"lz{k}"))
        return x + mix * (y - x)

    # ------------------------------------------------------------ 전체
    def forward(self, du: torch.Tensor, fric: torch.Tensor, asp: torch.Tensor,
                transient: torch.Tensor, c: dict, state: dict | None = None,
                asp_u: torch.Tensor | None = None, g_open: torch.Tensor | None = None) -> dict:
        """소스 (B,N) 들과 프레임률 제어 c (B,T_all). N ≤ T_all·hop 이면 선행 프레임이 있는 것이다.

        g_open: 성문 개방 곡선 (B,N) 0~1 — `OPEN_DAMP` 일 때 성문 경로 F1 을 주기마다 감쇠한다."""
        state = {} if state is None else state
        self._n_emit = du.shape[1]
        up = self._up
        # 성도 전체를 float64 로. 캐스케이드가 10 kHz 에서 −85 dB 까지 내려갔다가 고차 극 보정이
        # 그만큼 되돌리므로, 단 사이의 float32 반올림(6e-8)이 −50 dB 잡음으로 올라온다.
        dt_in = du.dtype
        du, fric, asp, transient = du.double(), fric.double(), asp.double(), transient.double()
        c = {k: v.double() for k, v in c.items()}
        # 난류·과도음은 압력원이므로 입술 방사(미분)를 받는다. du 는 이미 미분이다.
        asp_r, state["ra"] = tv_biquad(asp, *first_difference_coeffs(1.0, asp), zi=state.get("ra"))
        fr = fric + transient
        fr_r, state["rf"] = tv_biquad(fr, *first_difference_coeffs(1.0, fr), zi=state.get("rf"))
        leak = up(c["back_leak"])
        x_n = asp_r + leak * fr_r                  # 성도를 지나는 잡음 (기식 + 새어 든 마찰)
        x_g = du if NOISE_V2 else du + x_n
        if asp_u is not None:
            # 무성 기식은 물리 포먼트(정상 종속 가지)를 지난다 (v2.1).
            asp_u_r, state["rau"] = tv_biquad(asp_u.double(), *first_difference_coeffs(1.0, asp_u),
                                              zi=state.get("rau"))
            x_g = x_g + asp_u_r
        # 구강 게이트는 **입력에** 건다. 출력에 걸면 폐쇄 동안에도 캐스케이드가 소스를 그대로
        # 받아 울리고, 해제 순간 그 저장된 에너지가 50 배로 풀려 12 배짜리 클릭이 났다(측정).
        # 입력을 막으면 공진기는 제 대역폭으로 1~2 ms 만에 잦아들고, 해제에서는 다시 울려
        # 오르는 데 시간이 걸린다 — 실측의 "중역이 45 ms 에 걸쳐 올라온다" 가 그것이다.
        # 두 분기는 **유량을 나눠 갖는다**. 연구개 포트와 구강이 동시에 열려 있으면(비음화
        # 모음) 각자 절반씩이고, 구강이 닫히면 전부 비강으로 간다(머머). 나누지 않으면
        # 비음화 구간에서 두 경로가 그대로 더해져 12 배짜리 봉우리가 났다(측정).
        vel = up(c["velum"]).clamp(0.0, 1.0)
        opn = up(c["oral_open"]).clamp(0.0, 1.0)
        tot = (vel + opn).clamp_min(1e-3)
        oral, nas = opn / tot, vel / tot
        tracks = self._formant_tracks(c)
        if TRACK_SMOOTH_MS > 0.0:
            tracks = [(self._smooth_track(f, f"sf{i}", state), self._smooth_track(bw, f"sb{i}", state))
                      for i, (f, bw) in enumerate(tracks)]
        tr_g = tracks
        if OPEN_DAMP and g_open is not None:
            o = g_open.to(tracks[0][0].dtype)[:, :tracks[0][0].shape[1]]
            kb = OPEN_DAMP_MAX * torch.sigmoid(self.open_damp)
            kf = 0.15 * torch.tanh(self.open_f1) + 0.05
            f1_, b1_ = tracks[0]
            tr_g = [(f1_ * (1.0 + kf * o), b1_ * (1.0 + kb * o))] + list(tracks[1:])
        xg_in = x_g * oral
        if HPC_TAIL and HPC_AT_INPUT:
            xg_in = self._hpc(xg_in, "g", state)
        y_g = self._cascade(xg_in, tr_g, state, "f")
        if AUX_POLES:
            y_g = self._aux_poles(y_g, c, tr_g, state)
        if AUX_ZEROS:
            y_g = self._aux_zeros(y_g, c, tr_g, state)
        y_g = self._extra_cascade(y_g, tr_g, state)
        if HPC_TAIL and not HPC_AT_INPUT:
            y_g = self._hpc(y_g, "g", state)
        if NOISE_V2:
            # 잡음은 저 Q 사본을 지난다 — 같은 주파수, 대역폭 >= NOISE_BW_REL*f.
            tr_n = [(f, torch.maximum(bw, NOISE_BW_REL * f)) for f, bw in tracks]
            xn_in = x_n * oral
            if HPC_TAIL and HPC_AT_INPUT:
                xn_in = self._hpc(xn_in, "n", state)
            y_nc = self._extra_cascade(self._cascade(xn_in, tr_n, state, "n"), tr_n,
                                       state, prefix="nx", bw_extra=NOISE_EXTRA_BW)
            if HPC_TAIL and not HPC_AT_INPUT:
                y_nc = self._hpc(y_nc, "n", state)
            y_g = y_g + y_nc
        y_f = self._front_cavity((1.0 - leak) * fr_r * oral, c, state)
        # 비강도 같은 이유로 **입력에** 건다. 출력에 걸면 연구개가 닫힌 동안에도 분기가
        # 소스를 받아 울리고, 열리는 순간 그 에너지가 통째로 튀어나온다(측정: 12 배 클릭).
        y_n = self._nasal_branch((x_g + x_n if NOISE_V2 else x_g) * nas, c, state)
        y = y_g + y_f
        y = self._allpass(y, c, state)
        y = y + y_n
        # 고역 영점은 **합쳐진 뒤**에 건다. 고치려는 골은 마찰 구간에 있고, 그 에너지는 배음
        # 가지(`y_g`)가 아니라 잡음 가지(`y_nc`)와 앞공동(`y_f`)에서 온다 — 배음 가지에만
        # 걸면 정작 마찰의 고역은 그대로다.
        if HF_POLES:
            y = self._hf_peaks(y, c, state)
        if HF_ZEROS:
            y = self._hf_zeros(y, c, state)
        y = self._lateral(y, c, state)
        y = self._hf_eq(y, state)
        y = y * up(c["tract_gain"])
        return dict(audio=y.to(dt_in), glottal_path=y_g.to(dt_in), front_path=y_f.to(dt_in),
                    nasal_path=y_n.to(dt_in), state=state)
