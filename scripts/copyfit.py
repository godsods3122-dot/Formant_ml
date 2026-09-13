"""복사합성 — 녹음의 노이즈를 지우고, 스펙트럼이 맞을 때까지 물리 파라미터를 추적한다.

    OMP_NUM_THREADS=2 python scripts/copyfit.py data/voices/yang_00000034.wav \
        --profile profiles/yang_female.json --from 0.510 --to 0.665 --out out/fit

무엇을 하는가
    1. 잡음 프로파일 추정 -> 위너 스펙트럼 차감 (engine/denoise.py)
    2. 1 ms 프레임마다 F0·유성도·포먼트·세기·마찰 추정 -> 제어열 초기값 (engine/analyze.py)
    3. 미분가능 엔진으로 제어열을 역추정 (engine/fit.py) — 전역 스칼라 -> 성김/촘촘함 프레임별
    4. 원본·복원·차이를 wav 로, 제어열을 npz 로, 일치율 표를 표준출력으로

일치율을 어떻게 읽는가
    **포락** (멜 dB) — 조음이 맞는가. 사람이 듣는 음색에 대응한다.
    **정밀** (선형 다해상도 STFT) — F0 궤적과 성문 펄스 위치까지 맞는가.
    **보정 정밀** — 실현 분산을 추정해 뺀 진단값. 대역·변조·잡음비와 함께 읽는다.

34.5 % 는 동일 PSD 의 독립 가우시안 잡음을 비평활 STFT 크기로 비교할 때의
이론적 기준이지 모든 치찰음의 상한이 아니다. `trust` 는 분산 차감 뒤 남은 거리
비율이며 신뢰확률이 아니다. 분해 불가인 보정값은 신뢰구간의 하한도 아니다.
어떤 일치율도 인간과 구별 불가능함을 보장하지 않는다.
자세한 것은 engine/turbulence.py 머리말과 docs/MEASUREMENTS.md §9.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import soundfile as sf
import torch

from formant_ml.engine.analyze import analyze
from formant_ml.engine.calibration import (apply_calibration, legacy_fields,
                                           load_calibration)
from formant_ml.engine.control import INDEX, PARAM_NAMES
from formant_ml.engine.denoise import denoise, noise_profile, snr_report
from formant_ml.engine.fit import CopySynthFitter
from formant_ml.engine.profile import DEFAULT_PROFILE, SpeakerProfile
from formant_ml.engine.voice import EngineConfig, VoiceEngine

BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 5000),
         (5000, 8000), (8000, 12000), (12000, 24000)]


def band_db(x: np.ndarray, sr: int) -> list[float]:
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    tot = p.sum() + 1e-20
    return [10 * np.log10(p[(f >= lo) & (f < hi)].sum() / tot + 1e-20) for lo, hi in BANDS]


def fidelity_summary(fid: dict) -> str:
    score = f"{fid['fine_corr']:.2f} %" if fid["resolved"] else "분해 불가"
    return (f"보정 정밀: {score} (진단값 {fid['fine_corr']:.2f}, "
            f"잔여거리 비율 {fid['trust']:.2f}, 잡음비 {fid['noise_ratio']:.2f}, "
            f"실현 기준 추정 {fid['floor']:.1f} %, "
            f"시간평균 스펙트럼 {fid['spectrum_match']:.1f} %)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--profile", default=None, help="화자 프로파일 json")
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=None)
    ap.add_argument("--out", default="out/fit")
    ap.add_argument("--frame-ms", type=float, default=1.0)
    ap.add_argument("--global-iters", type=int, default=200)
    ap.add_argument("--stage-iters", type=int, default=100)
    ap.add_argument("--phase-iters", type=int, default=200,
                    help="마지막 위상 단계. 0 이면 끔 (크기만 맞춘다)")
    ap.add_argument("--lr-phase", type=float, default=None,
                    help="위상 단계의 lr 을 고정한다. 생략하면 짧은 탐침으로 고르는데, "
                         "그 탐침은 **짧은 시야로 작은 lr 에 편향**돼 있다 (MEASUREMENTS §39.1)")
    ap.add_argument("--lr-global", type=float, default=None,
                    help="생략하면 짧은 탐침으로 자동 선택 (구간마다 맞는 값이 다르다)")
    ap.add_argument("--lr-frame", type=float, default=0.04)
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--phase", type=float, default=0.0, help="복소 STFT 항 가중(2 단계용)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu",
                    help="cpu / cuda / mps. 적합 한 건이 RSS 3.8 GB 를 쓴다 — 병렬로 "
                         "돌릴 개수는 GPU 메모리가 아니라 그것으로 정하라")
    ap.add_argument("--patience", type=int, default=0,
                    help="이 회차 동안 손실이 안 줄면 그 단계를 끝낸다 (0 = 끔). "
                         "긴 음원에서는 켜는 편이 낫다 — 수렴한 단계에 예산을 다 쓴다")
    ap.add_argument("--room-from", default=None, metavar="STEM",
                    help="이미 있는 **마른** 적합 결과(copyfit --out 값)에서 녹음 경로 IR 을 "
                         "추정해 순방향 모형에 넣는다. 그러면 적합기가 방·마이크·코덱의 "
                         "응답을 성도·소스 파라미터로 흡수하지 않아 추정되는 물리량이 "
                         "**마른 목소리**의 것이 된다 (engine/room.py, MEASUREMENTS §23)")
    ap.add_argument("--room-taps", type=int, default=None, help="IR 탭 수 (기본 4096 = 85 ms)")
    ap.add_argument("--room-force", action="store_true",
                    help="홀드아웃에서 좋아지지 않아도 방을 넣는다")
    ap.add_argument("--room-lambda", type=float, default=None,
                    help="IR 추정의 정규화 세기 (기본 0.1). 크면 IR 이 δ 에 가까워져 "
                         "방의 시간 구조가 사라진다")
    ap.add_argument("--prior", action="append", default=None, metavar="이름=값",
                    help="사전 가중(fit.PRIOR_W) 덮어쓰기. 여러 번 줄 수 있다. "
                         "예: --prior aspiration=3.0")
    ap.add_argument("--bw-law", type=float, default=None,
                    help="포먼트 대역폭을 손실 법칙(40+0.05F) 쪽으로 당기는 세기 "
                         "(fit.BW_LAW_W). 대역폭은 기하가 아니라 손실이 정하는 "
                         "양이라 자유 파라미터로 두면 적합기가 극을 뭉개 스펙트럼 "
                         "기울기를 흉내 낸다 (MEASUREMENTS §35)")
    ap.add_argument("--flux", type=float, default=None,
                    help="세로 얼룩(스펙트럼 플럭스) **일치** 항의 세기 (fit.FLUX_W). "
                         "지금 손실은 _expect 때문에 얼룩을 원리적으로 못 본다 — "
                         "이 항이 멜 대역 dB 의 프레임 간 |Δ| 통계를 목표와 맞춘다. "
                         "최소화가 아니라 일치다 (난류는 원래 흔들린다)")
    ap.add_argument("--vel-w", type=float, default=None,
                    help="포먼트 가속도 벌점 세기 (fit.VEL_W, 기본 0). 무릎은 실측 "
                         "중앙 가속도 (F1 0.7 / F2 1.9 / F3 3.4 / F4 3.5 Hz/ms²)")
    ap.add_argument("--tilt-cap", type=float, default=None,
                    help="소스 기울기(tilt)가 먹히는 상한 주파수 [Hz] "
                         "(glottis.TILT_MAX_HZ, 기본 5000). 0 이면 상한 없음 — "
                         "그러면 성문 펄스가 사각파처럼 날카로워진다 (MEASUREMENTS §25)")
    ap.add_argument("--artic-vel", type=float, default=None,
                    help="조음 속도 한계 벌점의 세기 (fit.ARTIC_VEL_W, 기본 1.0). "
                         "0 이면 끈다 — 마찰음이 유성 구간에서 1 ms 만에 켜지는 것을 "
                         "막는 항이다 (MEASUREMENTS §24)")
    ap.add_argument("--motion", action="store_true",
                    help="제어열을 값이 아니라 **움직임**으로 매개화한다 (fit.MOTION). "
                         "자유 파라미터를 가속도로 두고 두 번 적분하므로 위치·속도의 "
                         "연속성이 벌점이 아니라 구조로 보장되고, 남는 불연속인 "
                         "가속도는 lam_smooth 가 저크 벌점으로 문다")
    ap.add_argument("--delta", type=float, default=None, metavar="W",
                    help="스펙트럼 전이 항 (fit.DELTA_W, MEASUREMENTS §51.29): 로그 멜의 시간 미분(Furui 회귀 델타, ±10 ms)을 "
                         "목표와 맞춘다. 지금 손실은 포먼트 전이가 뭉개져도 자음이 밀려도 반응하지 않는다")
    ap.add_argument("--formant", type=float, default=None, metavar="W",
                    help="포먼트 자리 항 (fit.FORMANT_W, MEASUREMENTS §51.29): F1·F2·F3 탐색 대역의 스펙트럼 무게중심(Bark)을 "
                         "목표와 맞춘다. 전대역 스펙트럼 거리는 봉우리 하나가 옮겨도 거의 안 움직인다")
    ap.add_argument("--band-balance", type=float, default=None, metavar="0-1",
                    help="대역 균형 가중 (fit.BAND_BALANCE): 포락 dB 항에서 로그 간격 8 묶음이 **같은 총량**을 갖게 한다. "
                         "0 = 멜 빈 수 비례(예전), 1 = 완전 균형. F2·F3 가 F1 보다 3~4 배 덜 보이던 편향을 없앤다")
    ap.add_argument("--seg-balance", type=float, default=None, metavar="0-1",
                    help="구간 균형 가중 (fit.SEG_BALANCE): 포락 dB 항에서 모음·마찰·비음·전이·조용함이 **같은 총량**을 "
                         "갖게 한다. 0 = 길이 비례(예전), 1 = 완전 균형")
    ap.add_argument("--gesture", type=float, default=None, metavar="W",
                    help="제스처 벌점 (fit.GESTURE_W, §51.25): 이웃 목표의 차에 L1 을 건다 — 변화의 개수를 벌해 "
                         "몇 번의 큰 전이와 긴 멈춤으로 몬다. L2 평활(--smooth-w)은 반대로 움직임을 고르게 편다")
    ap.add_argument("--action", choices=("bspline", "target"), default=None,
                    help="액션의 기저 (fit.MOTION_MODE, MEASUREMENTS §51.25). bspline = 격자점의 값(예전). "
                         "target = 조음 문헌의 목표 근사 — 격자값은 **목표**이고 궤적은 임계감쇠 3 차 응답이라 "
                         "목표가 안 바뀌는 동안 단조 접근뿐이다(흔들림이 표현 불가)")
    ap.add_argument("--motion-single", action="store_true",
                    help="--motion 의 다중 해상도를 끄고 격자 한 층만 쓴다 "
                         "(fit.MOTION_MULTI=False). 층 쌓기의 몫을 가르는 대조군")
    ap.add_argument("--motion-l2", type=float, default=None,
                    help="움직임 모드에서 **층별 L2** 세기 (fit.MOTION_L2). 고운 층일수록 "
                         "무겁게 문다 — 거친 층이 담을 수 있는 것은 거친 층이 담게 하고 "
                         "고운 층은 꼭 필요한 곳에만 쓰이게 한다")
    ap.add_argument("--motion-grid", type=float, default=None,
                    help="움직임 모드의 가장 고운 격자 [ms] (fit.MOTION_MIN_GRID_MS, 기본 2). "
                         "작을수록 미세구조를 담지만 난동도 표현 가능해지므로 --motion-l2 와 같이 쓴다")
    ap.add_argument("--bound", type=float, default=None,
                    help="경계의 매끄러운 벽 세기 (fit.BOUND_W). 시그모이드 재매개화는 "
                         "값이 경계에 붙으면 기울기가 죽어 고착된다 — 경계 안쪽에 "
                         "반발 벽을 두어 그 전에 되민다 (내부점법의 로그 장벽과 같은 취지)")
    ap.add_argument("--phase-ramp", type=float, default=None,
                    help="위상 가중을 계단이 아니라 **연속변형**으로 올린다. 위상 단계 "
                         "예산의 이 비율에 걸쳐 0 -> phase_w 로 올린다 (0.25 정도). "
                         "계단으로 바꾸면 해가 골짜기로 떨어졌다 회복하지 못한다 (§27, §46.1)")
    ap.add_argument("--balance", action="store_true",
                    help="손실 항의 기울기 균형을 켠다 (fit.BALANCE). 단계마다 항별 "
                         "기울기 규범을 재서 가중이 '포락 대비 영향력' 을 뜻하게 한다. "
                         "끄면 파형 상관이 포락보다 ~40 배 세게 적합을 끈다")
    ap.add_argument("--slow-noise", action="store_true",
                    help="잡음 이득(fric_gain·tract_gain·aspiration)을 거친 층(20 ms)에서만 "
                         "움직이게 한다 (fit.MOTION_SLOW). out/L7 만 치찰음 잡음을 피한 성질 — "
                         "잡음은 고요하게 방출하고 필터만 움직인다 — 을 그것만 가져온다")
    ap.add_argument("--noise-global", action="store_true",
                    help="fric_gain·aspiration 을 시간 제어가 아니라 전역 눈금으로 만든다 "
                         "(fit.NOISE_GLOBAL). 마찰의 시간 구조는 협착·레이놀즈 수가 정하고 "
                         "적합기는 a_c 로 마찰을 켜고 끈다. 모음에 샌 마찰 잡음 융단(10 kHz 위 "
                         "평탄)을 막는다")
    ap.add_argument("--harm-jit", type=float, default=None, metavar="rad/kHz",
                    help="배음별 위상 분산 (glottis.HARM_PHASE_JIT). 고역 배음을 지우지 않고 **번지게** "
                         "해서 안개로 무너뜨린다 — 위상만 흔드니 포먼트 포락은 보존된다. "
                         "목표에서 역산한 값이 ~0.35 (3 kHz 위 주기성이 목표와 겹친다)")
    ap.add_argument("--harm-jit-onset", type=float, default=None, metavar="Hz",
                    help="배음 위상 분산이 시작하는 주파수 (glottis.HARM_PHASE_ONSET, 기본 500). "
                         "500 이면 2 kHz 에서 이미 위상이 흔들려 에너지가 몰린 F1·F2 배음까지 "
                         "결맞음을 잃고, 파형 상관과 싸워 끊김이 39 → 60 으로 늘었다 (out/L28)")
    ap.add_argument("--noise-expect", type=float, default=None, metavar="MS",
                    help="난류 우세 빈을 시간 평활하는 폭 (fit.NOISE_EXPECT_MS, 기본 25). 목표와 합성에 "
                         "같이 걸려 편향은 없지만 **그보다 짧은 끊김을 손실이 못 본다** — 25 ms 면 "
                         "치찰음의 20 ms 세로 토막이 포락 오차 0.8 dB 로만 보인다(탐침)")
    ap.add_argument("--reseed", action="store_true",
                    help="반복마다 잡음 씨앗을 새로 뽑는다 (fit.RESEED). 고정 씨앗이면 적합기가 "
                         "제어열을 흔들어 난수 한 벌을 목표의 무작위 요동에 끼워 맞춘다")
    ap.add_argument("--sched", choices=("cos", "sgdr"), default=None,
                    help="학습률 스케줄 (fit.SCHED). sgdr = 단계마다 감쇠형 웜 리스타트 코사인 — "
                         "국소 최소를 벗어날 기회를 준다")
    ap.add_argument("--sgdr-cycles", type=int, default=None, help="단계당 재시동 주기 수 (기본 3)")
    ap.add_argument("--sgdr-decay", type=float, default=None, help="주기마다 정점 lr 배율 (기본 0.7)")
    ap.add_argument("--smooth-w", type=float, default=None,
                    help="lam_smooth (격자 1 차 차분 L2, 기본 3e-3). 0 이면 끈다")
    ap.add_argument("--prior-w", type=float, default=None,
                    help="lam_prior (사전값 L2, 기본 1e-4). 0 이면 끈다")
    ap.add_argument("--unvoiced-edge", type=int, default=None, metavar="MS",
                    help="무성 벌점이 비켜 가는 유성 경계 여유 (fit.UNVOICED_EDGE, 기본 10 ms)")
    ap.add_argument("--base-smooth", type=float, default=None, metavar="K",
                    help="분석 출발점도 조음 대역으로 거른다 (fit.BASE_SMOOTH): --slow 로 층을 묶은 파라미터의 "
                         "분석 궤적을 σ = K × 층 ms 가우시안으로 편다. 권장 0.45. 포먼트 추적기의 20~60 Hz 흔들림이 "
                         "모음 고역을 거칠게 만든다 (MEASUREMENTS §51.5)")
    ap.add_argument("--voice-gain", action="store_true",
                    help="성문 배음에만 거는 빠른 이득 제어(voice_gain)를 적합한다. 없으면 0 dB 로 얼린다 — "
                         "주기별 진폭(시머)을 잡음을 썰지 않고 따라간다 (docs/FOUNDATION.md, MEASUREMENTS §51)")
    ap.add_argument("--front-q", action="store_true",
                    help="앞공동 Q 를 제어값 front_q (1~8) 로 적합한다 (tract.FRONT_Q_CTRL, v2 앞공동). 없으면 얼린다")
    ap.add_argument("--fric-lp", action="store_true",
                    help="마찰 소스 고역 절벽 배율(1.6~6.4)을 파일 전역으로 적합한다 (fit.FRIC_LP_FIT)")
    ap.add_argument("--init", default=None, metavar="STEM",
                    help="앞선 적합(STEM_track.npz)의 제어열·전역 엔진값에서 출발한다 (두 번째 패스 — 다듬기). "
                         "펄스·유성·마찰 표시와 사건은 이번 분석의 것을 쓴다")
    ap.add_argument("--grids", default=None, metavar="MS,MS,...",
                    help="2.x 단계들의 격자 [ms] (fit.GRID_MS, 기본 20,10,5,1). 창은 단계 순서대로 늘어난다. "
                         "같은 값을 되풀이하면 층을 더하지 않고 그 격자에서 창만 늘려 이어 적합한다")
    ap.add_argument("--fast", action="append", default=None, metavar="이름",
                    help="이 파라미터만 1 ms 격자까지 푼다 (fit.MOTION_FAST, MEASUREMENTS §51.19). 나머지는 전부 2 ms 층에서 "
                         "잠근다. voice_gain 에 쓰면 목표의 주기별 진폭(시머 실현)을 따라갈 수 있다")
    ap.add_argument("--src-eq", action="store_true",
                    help="화자 음원 EQ (glottis.SRC_EQ, MEASUREMENTS §51.17): 성문 배음에 고정 주파수 봉우리 여섯의 "
                         "이득(±9 dB)을 화자 전역으로 적합한다 — 음원 기울기 넷(Kreiman·Garellek)의 자리")
    ap.add_argument("--pulse-lock", action="store_true",
                    help="성문 위상을 목표 녹음의 폐쇄 시각에 잠근다 (fit.PULSE_LOCK, MEASUREMENTS §51.16). "
                         "f0_target 은 적합에서 빠진다 (위상을 만들지 않으므로)")
    ap.add_argument("--mvf", type=float, default=None, metavar="Hz",
                    help="기식 MVF 를 이 값에 박고 적합하지 않는다 (fit.MVF_FROZEN, v2.1 기식 가르기)")
    ap.add_argument("--phase-lf", action="store_true",
                    help="위상 항을 배음 위상 분산 시작 주파수 아래로 한정한다 (fit.PHASE_FMAX = --harm-jit-onset, §51.15)")
    ap.add_argument("--hjit-cycle", action="store_true",
                    help="배음 위상 분산을 1 ms 난수가 아니라 성문 주기마다 뽑는다 (glottis.HJIT_CYCLE, §51.13) — 펄스 동기 고역")
    ap.add_argument("--hjit-fit", action="store_true",
                    help="배음 위상 분산 세기의 화자 전역 배율(0.4~2.5)을 적합한다 (glottis.HJIT_FIT, MEASUREMENTS §51.12)")
    ap.add_argument("--open-damp", action="store_true",
                    help="성문 개방기 F1 감쇠 (tract.OPEN_DAMP, MEASUREMENTS §51.9): 성문이 열린 동안 F1 대역폭·주파수를 "
                         "주기마다 올린다. 세기 둘은 화자 전역 적합값")
    ap.add_argument("--hpc-fmax", type=float, default=None, metavar="Hz",
                    help="HPC 꼬리 보정을 이 주파수 위에서 고정한다 (tract.HPC_FMAX, 기본 17000). 16~18 kHz 가 목표보다 크면 낮춘다")
    ap.add_argument("--hpc", action="store_true",
                    help="고차 극 꼬리 보정 (tract.HPC_TAIL, MEASUREMENTS §51.7): 극 14 개에서 끊긴 캐스케이드를 "
                         "무한 균일관의 포락에 맞추는 정적 최소위상 FIR. 없으면 12~16 kHz 가 40~90 dB 어둡다")
    ap.add_argument("--hf-fixed", action="store_true",
                    help="화자 고정 고역 구조 (tract.HF_FIXED, docs/FOUNDATION.md §3.3): F5 위 극은 F4 를 따라가지 않고 "
                         "절대 위치 + 전역 이동·넓은 대역폭(Q≈5), 이상와 영점 하나")
    ap.add_argument("--noise-v2", action="store_true",
                    help="잡음 가지 v2 (docs/NOISE_SOURCE_REVIEW.md §3): 잡음은 저 Q 성도 사본·병렬 저 Q 앞공동을 "
                         "지나고, 마찰 구동은 매끈한 문턱 + 2 ms 포락 평활, 기식은 MVF(5.5 kHz) 위 2 차 고역")
    ap.add_argument("--asp-corner", type=float, default=None, metavar="Hz",
                    help="기식 잡음 셸프의 모서리 (aspiration.corner, 기본 3000). 목표 모음은 약 6 kHz 까지 "
                         "배음 무늬가 있고 그 위만 안개다 — 모서리가 3 kHz 면 기식을 올릴 때 맑아야 할 중역에 "
                         "숨소리가 섞인다 (MEASUREMENTS §50.17)")
    ap.add_argument("--asp-floor", type=float, default=None,
                    help="기식 셸프의 저역 바닥 (aspiration.floor, 기본 0.3 = 모서리 아래 −10 dB)")
    ap.add_argument("--floor", action="store_true",
                    help="목표의 녹음 바닥 잡음을 합성에 고정으로 더한다 (fit.FLOOR_NOISE). 없으면 적합기가 "
                         "무음의 바닥을 숨소리 + 성도 공명으로 흉내 내 무음에 공명 줄이 선다")
    ap.add_argument("--re-lead", type=float, default=None, metavar="W",
                    help="미래를 보는 레이놀즈 전이 (fit.RE_LEAD_W, MEASUREMENTS §51.24): 목표에 앞으로 25 ms 안에 "
                         "난류가 오면 Re 를 문턱 위로, 앞뒤로 없으면 아래로 민다. 피팅이므로 미래 정보를 쓴다")
    ap.add_argument("--re-lead-ms", type=float, default=None, metavar="MS",
                    help="그 미리보기 창 [ms] (fit.RE_LEAD_MS, 기본 25)")
    ap.add_argument("--move-budget", type=float, default=None, metavar="W",
                    help="움직임 예산 (fit.MOVE_W, MEASUREMENTS §51.20): 포먼트·대역폭은 분석 궤적 속도의 1.3 배까지, "
                         "음원 손잡이(voice_gain·tilt·rd_offset·p_sub·adduction)는 생리 상한까지만 빠르게 움직이게 문다")
    ap.add_argument("--fast-only", action="append", default=None, metavar="이름",
                    help="이 제어는 **빠른 요동만** 낸다 (fit.FAST_ONLY, §51.22): 값에서 자기 추세(8 ms)를 빼고 유성 "
                         "구간에만 건다. voice_gain 에 쓰면 소리 크기의 느린 윤곽은 폐압·성도 이득이 지게 된다")
    ap.add_argument("--events", type=float, default=None, metavar="W",
                    help="펄스 이벤트 층 (fit.EVENT_W, §51.45): 관측이 급히 움직이는 자리마다 **계단 하나**를 "
                         "얹는다. 계단은 2 ms 안에 다 올라 격자·τ·출발점 평활의 제약을 안 받는다. "
                         "--move-fine 이 찾은 자리를 쓴다")
    ap.add_argument("--hf-stage", type=int, default=None, metavar="N",
                    help="움직임 단계 뒤에 **고역만 따로** N 반복 적합한다 (fit.HF_STAGE_ITERS, §51.44). "
                         "제어 궤적은 얼리고 고역 전역 손잡이만, 4 kHz 위 멜 띠만 보는 손실로")
    ap.add_argument("--hf-eq", action="store_true",
                    help="4 kHz 위 고역 포락을 피킹 EQ 8 개로 **직접 적합**한다 (tract.HF_EQ, §51.44). "
                         "극으로는 고역 구조가 안 생긴다(봉우리−골 6.6 대 목표 11.8 dB) — 모양을 직접 잡는다")
    ap.add_argument("--hf-free", action="store_true",
                    help="고역 극 F5~F8 을 적합 대상으로 푼다 (analyze.HF_FREE, §51.42). 분석이 사다리로 출발점을 "
                         "채우고, --formant-band 의 구역(섭동 이론 ×0.80~1.25) 안에서만 움직인다. --hf-fixed 와 같이 쓰지 않는다")
    ap.add_argument("--aux-zeros", action="store_true",
                    help="이웃 포먼트 사이에 **켰다 끄는 보조 영점**(소리를 죽이는 극) 셋을 둔다 "
                         "(tract.AUX_ZEROS, §51.46). 전극 모형이 못 파는 골이 프레임의 20~30 %% 다")
    ap.add_argument("--hf-zeros", action="store_true",
                    help="**F4 위**에 자리가 자유로운 영점 셋을 둔다 (tract.HF_ZEROS, §51.50). 마찰 구간 "
                         "4~16 kHz 에서 목표의 골이 중앙 4.9~5.5 dB 인데 우리는 0.8~1.6 dB 다 — "
                         "좁고 깊은 골은 극으로 못 만든다")
    ap.add_argument("--hf-poles", action="store_true",
                    help="F4 위에 **뾰족한 마루** 셋을 둔다 (tract.HF_POLES, §51.51). 영점과 대칭이다 — "
                         "모음 4~16 kHz 에서 목표의 마루가 중앙 5.2 dB 인데 우리는 0.7 dB 다")
    ap.add_argument("--aux-poles", action="store_true",
                    help="이웃 포먼트 사이에 **켰다 끄는 보조 공진** 셋을 둔다 (tract.AUX_POLES, §51.40). "
                         "목표에서 포먼트가 합쳐졌다 갈라질 때 있는 극을 끌고 건너가는 대신 사이의 극을 "
                         "서서히 켠다. 깊이 0 은 정확히 항등이라 출발점이 안 바뀐다")
    ap.add_argument("--formant-band", default=None, metavar="JSON",
                    help="포먼트마다 **제 구역**을 준다 (fit.FORMANT_BAND, §51.38). 파일의 formant_band 만 읽고 "
                         "화자 상수는 얼리지 않는다. 지금 f1~f8 의 제어 범위가 전부 12 kHz 까지 열려 있어 "
                         "적합된 f4 가 7321 Hz 까지 간다(관측 3876~5426)")
    ap.add_argument("--no-velum", action="store_true",
                    help="비강 분기를 켜지 않는다 (analyze.NASAL_VELUM = False). 머머 검출과 구강 포먼트 보간은 그대로 — "
                         "연구개를 여는 것만 끈다. A/B 용이다")
    ap.add_argument("--speaker-lock", default=None, metavar="JSON",
                    help="화자 공통 보정값만 읽고 잠근다. MVF·위상 분산·음원 EQ·녹음 경로는 "
                         "잠그지 않는다. scripts/speaker_profile.py 의 JSON 또는 *_track.npz")
    ap.add_argument("--recording-lock", default=None, metavar="JSON",
                    help="같은 녹음 환경의 출력 EQ·방 IR을 읽고 고정한다. 발화별 레벨 정규화는 "
                         "유지한다. *_constants.json 또는 *_track.npz; --room-from 과 함께 쓰지 않는다")
    ap.add_argument("--spec-ripple", type=float, default=None, metavar="W",
                    help="**고역 물결**이 목표보다 얕은 만큼을 문다 (fit.HFRIP_W, §51.53). 포락 손실은 "
                         "멜 띠(12 kHz 에서 516 Hz)가 목표 구조(281 Hz)보다 넓어서 고역의 마루·골을 "
                         "아예 못 본다 — 그래서 영점·마루 손잡이를 켜 줘도 적합기가 쓰질 않았다")
    ap.add_argument("--prominence", type=float, default=None, metavar="W",
                    help="공진 돌출 항 (fit.PROM_W, §51.33): 대역마다 **봉우리−골 깊이**를 목표와 맞춘다. "
                         "지금 F2·F3 이 목표보다 2~3 dB 눌려 있고(3 dB 넘게 얕은 창이 30~43 %%) F1 은 2~3 dB 과하다")
    ap.add_argument("--attack", type=float, default=None, metavar="W",
                    help="어택 항 (fit.ATTACK_W, §51.31): 목표의 2~8 kHz 가 3 ms 안에 서는 자리에서 **서는 크기**를 "
                         "맞춘다. 지금 파열은 통째로 빠져 있다 — 목표 13~31 dB 대 합성 −7.5~+9.2 dB")
    ap.add_argument("--burst-fine", action="store_true",
                    help="파열 자리 ±20 ms 에서만 제어 해상도를 푼다 (fit.BURST_FINE, §51.31). --slow 로 묶인 "
                         "조음 손잡이가 그 창 안에서만 고운 층을 쓴다 — 다른 곳은 그대로 묶여 있다")
    ap.add_argument("--move-fine", type=float, default=None, metavar="PCT",
                    help="관측 포먼트가 ms 당 이 %% 넘게 움직이는 자리에서도 제어 해상도를 푼다 "
                         "(fit.MOVE_FINE_PCT, §51.37). --burst-fine 과 같은 창을 쓴다")
    ap.add_argument("--burst-win", type=float, default=None, metavar="MS",
                    help="파열 창 반폭 [ms] (fit.BURST_WIN_MS, 기본 20). 파열 앞의 폐쇄를 만들 자리가 된다")
    ap.add_argument("--hold-fric", action="store_true",
                    help="마찰 **안**에서 포먼트·대역폭을 그 구간의 고원 값으로 붙잡는다 (fit.HOLD_FRIC, §51.30). "
                         "사용자가 치찰음이 좋다고 한 out/L7 은 마찰 안에서 f3·f4 가 0.003 %%/ms 로 얼어 있었다. "
                         "심어서 잰 비용은 포락 −0.06 %% 이하다")
    ap.add_argument("--hold-ramp", type=float, default=None, metavar="MS",
                    help="마찰 가장자리에서 잡아 두기가 풀리는 거리 [ms] (fit.HOLD_RAMP_MS, 기본 15). 전이는 막지 않는다")
    ap.add_argument("--move-cap", action="append", default=None, metavar="이름=값",
                    help="움직임 예산의 생리 상한 덮어쓰기 (fit.MOVE_CAP). 예: --move-cap voice_gain=0.5")
    ap.add_argument("--quiet", type=float, default=None, metavar="W",
                    help="조용한 구간 넘침 벌점 (fit.QUIET_W, MEASUREMENTS §51.11): 목표가 정점 −45 dB 아래인 10 ms 창에서 "
                         "합성이 목표보다 2 dB 넘게 크면 문다 — 발화 끝·시작 앞 숨소리 잡음띠")
    ap.add_argument("--unvoiced", type=float, default=None, metavar="W",
                    help="목표가 무성 마찰인 프레임에서 성문 떨림을 문다 (fit.UNVOICED_W). 무성 ㅊ 에서 "
                         "성대가 떨어 치찰음을 F0 로 써는 것을 막는다 (MEASUREMENTS §50.9)")
    ap.add_argument("--voiced-soft", type=float, default=None, metavar="AMP",
                    help="유성 마스크를 떨림 진폭에 비례시킨다 (glottis.VOICED_SOFT, 기준 진폭; "
                         "0.8 이면 모음 떨림에서 1). 기본 0 = 이진 amp>1e-3")
    ap.add_argument("--fric-am", type=float, default=None,
                    help="마찰의 성문 주기 AM 깊이 (noise.FRIC_AM_DEPTH, 기본 0.35). 성문이 울면 "
                         "마찰 진폭이 F0 마다 ±35 %% 토막난다. 적합기가 무성 치찰음 한복판에서도 "
                         "성대를 끄지 않으므로(ㅊ: 폐압 6.5, 내전 0.31) 치찰음이 300 Hz 로 세로 "
                         "토막난다. 무성 치찰음의 협착 난류는 성문 주기로 게이트되지 않는다")
    ap.add_argument("--asp-am", type=float, default=None,
                    help="기식의 성문 주기 AM 깊이 (glottis.ASP_AM_DEPTH, 기본 0.7). 이 게이트가 "
                         "고역의 세로 블록을 만든다 — 기식을 물리값에 고정하니 고역 포락 동기가 "
                         "0.999 (목표 0.84) 로 지나치게 주기적이 됐다 (out/L24)")
    ap.add_argument("--hnm", type=float, default=None, metavar="MVF_Hz",
                    help="조화+잡음 모형으로 여기원을 나눈다 (glottis.MVF_HZ). 배음은 이 주파수 위로 "
                         "부드럽게 꺼지고, 기식 잡음은 상보적으로 그 위에 몰린다 (셸프 코너 = MVF, "
                         "바닥 0.05). 필터(성도)는 공유한다. 이 화자의 목표 배음은 ~6 kHz 에서 끝난다")
    ap.add_argument("--freeze", action="append", default=None, metavar="이름",
                    help="이 파라미터를 적합하지 않고 분석 초기값에 둔다. 여러 번 줄 수 있다. "
                         "예: --freeze aspiration — 기식을 물리값(1.0)에 고정한다. 적합기는 "
                         "기식을 물리값의 1~18 %%로 짓누르는데(RUN_LOCAL §4.5 의 "
                         "'aspiration 붕괴'), 그러면 유성 고역에 난류가 없어 지나치게 주기적이 된다")
    ap.add_argument("--param-tau", action="store_true",
                    help="파라미터마다 생리적 시상수로 적합 보정분을 저역통과한다 "
                         "(fit.PARAM_TAU_PHYS: 호흡 30 ms > 후두 15 ms > 포먼트 5 ms > 협착 2 ms). "
                         "궤적은 원래 파라미터마다 따로지만 빠르기 상한이 공유돼 폐압이 30 Hz 로 "
                         "떨었다. 분석 초기값은 그대로 두고 보정분만 거른다")
    ap.add_argument("--tau", action="append", default=None, metavar="이름=ms",
                    help="파라미터별 시상수를 직접 준다 (--param-tau 위에 덮어쓴다)")
    ap.add_argument("--slow", action="append", default=None, metavar="이름=ms",
                    help="파라미터별 가장 고운 층 [ms] 을 직접 준다. 여러 번 줄 수 있다")
    ap.add_argument("--bal", action="append", default=None, metavar="항=몫",
                    help="--balance 의 항별 몫 덮어쓰기 (fit.BAL_W, 포락 = 1 대비). "
                         "여러 번 줄 수 있다. 예: --bal corr=3 --bal phase=3")
    ap.add_argument("--l1", type=float, default=None,
                    help="제어열 증분에 L1 (기본 0 = 끔). 지금 정칙화가 전부 L2 라 "
                         "적합기가 격자 전체에 자잘한 보정을 흩뿌린다 — 희소하게 만들면 "
                         "어떤 파라미터가 실제로 움직여야 하는지도 드러난다")
    ap.add_argument("--corr", type=float, default=None,
                    help="유성 구간에서 20 ms 파형 상관이 무너진 만큼을 문다 (기본 0 = 끔). "
                         "**끊겨 들리는 결함을 잡는 항이다** — 그 자리의 멜 오차는 오히려 "
                         "낮을 수 있다(위상만 뒤집힌 것). MEASUREMENTS §44")
    ap.add_argument("--hnr", type=float, default=None,
                    help="주기성(조화 대 비조화)을 목표에 일치시킨다 (기본 0 = 끔). "
                         "없으면 적합기가 하모닉을 잡음으로 바꿔 같은 스펙트럼을 만든다 "
                         "— 실측 fric_gain +1022 %%. MEASUREMENTS §44")
    ap.add_argument("--harmonic", type=float, default=None, metavar="W",
                    help="유성 0~3 kHz 배음별 크기 오차 (기본 1). 목표 위상에 동기화한 최소제곱으로 "
                         "비교한다. 새 제어열은 없다. 0이면 이전 손실과 BW 관측 사전으로 돌아간다")
    ap.add_argument("--cont", type=float, default=None,
                    help="창별 손실이 직전 창보다 나빠진 만큼을 문다 (기본 0 = 끔). "
                         "평균 손실은 국소 붕괴를 못 본다 — 50 ms 구간 상관이 1.00 인데 "
                         "한 구간만 −0.06 으로 뒤집히는 일이 실제로 있다")
    ap.add_argument("--sharp", type=float, default=None,
                    help="유성 구간이 목표보다 뭉툭한 만큼을 문다 (기본 0 = 끔). "
                         "포락 일치율은 이것을 볼 수 없다 — 멜 밴드가 포먼트 골보다 넓다")
    ap.add_argument("--subf0", type=float, default=None,
                    help="F0 아래가 **목표보다** 시끄러운 만큼을 문다 (기본 0 = 끔). "
                         "목표의 프라이·서브하모닉은 벌하지 않는다")
    ap.add_argument("--hf-ripple", type=float, default=None, dest="ripple",
                    help="`--ripple` 의 다른 이름. 제어열 잔물결 벌점 세기 (fit.RIPPLE_W). 조음 대역(0~20 Hz) "
                         "위에서 트랙이 흔들리는 것만 문다 — 지지직과 저역 초과가 "
                         "둘 다 여기서 온다 (docs/MEASUREMENTS.md §13, §16)")
    a = ap.parse_args()
    if a.recording_lock and a.room_from:
        ap.error("--recording-lock and --room-from are mutually exclusive")
    if a.harmonic is not None and (not np.isfinite(a.harmonic) or a.harmonic < 0):
        ap.error("--harmonic must be finite and nonnegative")
    torch.set_num_threads(a.threads)
    # 손실 가중은 **모듈 전역**이고 CopySynthFitter 가 생성 시점에 읽는다. 그러므로
    # 적합기를 만들기 전에 여기서 덮어써야 한다.
    #
    # **예전 판은 이 덮어쓰기가 조건문 안에 있었고, 그 조건이 ripple·prior·artic_vel·
    # vel_w·flux·bw_law 만 보았다.** 그래서 `--corr/--hnr/--cont/--sharp/--subf0` 중
    # 하나만 주면 블록 자체가 안 돌아 가중이 0 인 채로 적합이 돌았다 — 플래그가 조용히
    # 무시된 것이다. 실측: 같은 짧은 구간에서 `--subf0 1.0` 을 준 손실이 안 준 것과
    # **바이트 단위로 같았다** (둘 다 2.163376). 표로 바꿔, 손잡이를 새로 달 때
    # 조건문을 같이 고쳐야 하는 일 자체를 없앤다.
    from formant_ml.engine import fit as _fit
    if a.motion:
        _fit.MOTION = True
    if a.reseed:
        _fit.RESEED = True
    if a.unvoiced is not None:
        _fit.UNVOICED_W = float(a.unvoiced)
    if a.quiet is not None:
        _fit.QUIET_W = float(a.quiet)
    if a.re_lead is not None:
        _fit.RE_LEAD_W = float(a.re_lead)
    if a.re_lead_ms is not None:
        _fit.RE_LEAD_MS = float(a.re_lead_ms)
    if a.move_budget is not None:
        _fit.MOVE_W = float(a.move_budget)
    if a.fast_only:
        _fit.FAST_ONLY = set(a.fast_only)
    if a.spec_ripple is not None:
        _fit.HFRIP_W = float(a.spec_ripple)
    if a.prominence is not None:
        _fit.PROM_W = float(a.prominence)
    if a.attack is not None:
        _fit.ATTACK_W = float(a.attack)
    if a.burst_fine:
        _fit.BURST_FINE = True
    if a.move_fine is not None:
        _fit.MOVE_FINE_PCT = float(a.move_fine)
    if a.burst_win is not None:
        _fit.BURST_WIN_MS = float(a.burst_win)
    if a.hold_fric:
        _fit.HOLD_FRIC = True
    if a.hold_ramp is not None:
        _fit.HOLD_RAMP_MS = float(a.hold_ramp)
    for item in (a.move_cap or ()):
        k, _, v = item.partition("=")
        if k not in _fit.MOVE_CAP:
            raise SystemExit(f"--move-cap: 모르는 파라미터 {k!r}")
        _fit.MOVE_CAP[k] = float(v)
    if a.unvoiced_edge is not None:
        _fit.UNVOICED_EDGE = int(a.unvoiced_edge)
    if a.floor:
        _fit.FLOOR_NOISE = True
    if a.sched:
        _fit.SCHED = a.sched
    if a.sgdr_cycles is not None:
        _fit.SGDR_CYCLES = int(a.sgdr_cycles)
    if a.sgdr_decay is not None:
        _fit.SGDR_DECAY = float(a.sgdr_decay)
    if a.noise_expect is not None:
        _fit.NOISE_EXPECT_MS = float(a.noise_expect)
    if a.balance:
        _fit.BALANCE = True
    if a.param_tau:
        _fit.PARAM_TAU_MS.update(_fit.PARAM_TAU_PHYS)
    if a.base_smooth is not None:
        _fit.BASE_SMOOTH = float(a.base_smooth)
    for item in (a.tau or ()):
        k, _, v = item.partition("=")
        _fit.PARAM_TAU_MS[k] = float(v)
    if a.slow_noise:
        _fit.MOTION_SLOW.update(_fit.SLOW_NOISE)
    if a.noise_global:
        _fit.MOTION_SLOW.update(_fit.NOISE_GLOBAL)
    for item in (a.slow or ()):
        k, _, v = item.partition("=")
        _fit.MOTION_SLOW[k] = float(v)
    if a.fast:
        # **한 파라미터만 1 ms 격자까지**: 하한을 내리고 나머지는 전부 2 ms 층에서 잠근다.
        _fit.MOTION_FAST = set(a.fast)
        _fit.MOTION_MIN_GRID_MS = 1.0
        for _n in _fit.DEFAULT_PARAMS:
            if _n not in _fit.MOTION_FAST:
                _fit.MOTION_SLOW[_n] = max(_fit.MOTION_SLOW.get(_n, 0.0), 2.0)
    for item in (a.bal or ()):
        k, _, v = item.partition("=")
        if k not in _fit.BAL_W:
            raise SystemExit(f"--bal: 모르는 항 {k!r} (가능: {', '.join(_fit.BAL_W)})")
        _fit.BAL_W[k] = float(v)
    if a.action:
        _fit.MOTION_MODE = a.action
    if a.gesture is not None:
        _fit.GESTURE_W = float(a.gesture)
    if a.delta is not None:
        _fit.DELTA_W = float(a.delta)
    if a.seg_balance is not None:
        _fit.SEG_BALANCE = float(a.seg_balance)
    if a.band_balance is not None:
        _fit.BAND_BALANCE = float(a.band_balance)
    if a.formant is not None:
        _fit.FORMANT_W = float(a.formant)
    if a.motion_single:
        _fit.MOTION_MULTI = False
    if a.motion_grid is not None:
        _fit.MOTION_MIN_GRID_MS = float(a.motion_grid)
    for flag, name in (("ripple", "RIPPLE_W"), ("artic_vel", "ARTIC_VEL_W"),
                       ("vel_w", "VEL_W"), ("flux", "FLUX_W"), ("bw_law", "BW_LAW_W"),
                       ("corr", "CORR_W"), ("hnr", "HNR_W"), ("motion_l2", "MOTION_L2"), ("bound", "BOUND_W"), ("phase_ramp", "PHASE_RAMP"), ("cont", "CONT_W"),
                       ("sharp", "SHARP_W"), ("subf0", "SUBF0_W")):
        v = getattr(a, flag)
        if v is not None:
            setattr(_fit, name, float(v))
    if a.tilt_cap is not None:
        from formant_ml.engine import glottis as _g
        _g.TILT_MAX_HZ = float(a.tilt_cap)
    for item in (a.prior or ()):       # --tilt-cap 블록 안에 갇혀 있던 것을 꺼냈다
        k, _, v = item.partition("=")
        if k not in _fit.PRIOR_W and k not in _fit.DEFAULT_PARAMS:
            raise SystemExit(f"--prior: 모르는 파라미터 {k!r}")
        _fit.PRIOR_W[k] = float(v)

    y, sr = sf.read(a.wav)
    if y.ndim > 1:
        y = y.mean(1)
    y = np.asarray(y, dtype=np.float64)
    if not a.no_denoise:
        noise = noise_profile(y, sr)
        yd = denoise(y, sr, noise)
        r = snr_report(y, yd, sr, noise)
        print(f"잡음 제거: 바닥 {r['noise_db_in']:.1f} -> {r['noise_db_out']:.1f} dB "
              f"(−{r['removed_db']:.1f} dB), 정점 {r['peak_db_in']:.1f} -> "
              f"{r['peak_db_out']:.1f} dB")
        y = yd

    t1 = a.t1 if a.t1 is not None else len(y) / sr
    seg = y[int(a.t0 * sr):int(t1 * sr)]
    prof = SpeakerProfile.load(a.profile) if a.profile else DEFAULT_PROFILE
    hop = max(1, int(round(a.frame_ms * sr / 1000.0)))
    if a.no_velum:
        from formant_ml.engine import analyze as _an
        _an.NASAL_VELUM = False
    track = analyze(seg, sr, prof, hop, t0=a.t0, full=y)
    print(f"구간 {a.t0:.3f}~{t1:.3f} s, {track.n_frames} 프레임 × {track.frame_ms} ms",
          flush=True)

    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=a.frame_ms,
                                   speaker="female" if prof.f0_nominal > 165 else "male",
                                   residual=False), prof)
    initial_constants = None
    recording_constants = load_calibration(a.recording_lock) if a.recording_lock else None
    if a.init:
        # 두 번째 패스: 앞선 적합의 제어열을 출발점으로, 전역 엔진값도 되살린다.
        with np.load(a.init + "_track.npz", allow_pickle=False) as _z:
            _v = np.asarray(_z["values"], dtype=np.float64)
            if float(_z["frame_ms"]) != track.frame_ms:
                raise ValueError("--init control frame spacing must match --frame-ms")
            if _v.ndim != 2 or not np.isfinite(_v).all():
                raise ValueError("--init controls must be a finite two-dimensional array")
            if "names" in _z:
                old_names = tuple(_z["names"].tolist())
                if old_names != tuple(PARAM_NAMES[:len(old_names)]):
                    raise ValueError("--init control names do not match the engine schema")
        initial_constants = load_calibration(a.init + "_track.npz")
        if _v.shape[1] < track.values.shape[1]:          # 나중에 더한 제어값은 기본값으로
            _v = np.concatenate([_v, np.tile(track.values[:1, _v.shape[1]:], (len(_v), 1))], 1)
        n_ = min(len(_v), track.n_frames)
        track.values[:n_] = _v[:n_, :track.values.shape[1]]
        print(f"출발점: {a.init} (제어열 {n_} 프레임, 공통·발화 상수 읽음)", flush=True)
    room_ir = (initial_constants["recording"].get("room_ir")
               if initial_constants is not None else None)
    if recording_constants is not None:
        if not recording_constants["recording"]:
            raise ValueError("--recording-lock requires recording-scope constants")
        room_ir = recording_constants["recording"].get("room_ir")
    if a.room_from:
        from formant_ml.engine import room as _room
        # **마른 출력이 있으면 그쪽을 쓴다.** 이미 방을 넣고 적합한 결과라면
        # `_fit.wav` 는 방을 통과한 소리다 — 그걸로 다시 추정하면 방이 두 번 들어간다.
        src = (a.room_from + "_dry.wav" if os.path.exists(a.room_from + "_dry.wav")
               else a.room_from + "_fit.wav")
        dry, _ = sf.read(src)
        ref, _ = sf.read(a.room_from + "_target.wav")
        dry = dry.mean(1) if dry.ndim > 1 else dry
        ref = ref.mean(1) if ref.ndim > 1 else ref
        m = min(len(dry), len(ref))
        kw = dict(taps=a.room_taps or _room.DEFAULT_TAPS,
                  lam=a.room_lambda if a.room_lambda is not None else _room.DEFAULT_LAMBDA)
        # **못 본 절반에서 실제로 좋아지는지 먼저 본다.** IR 이 항상 도움이 되지는
        # 않는다 — 마른 모형이 이미 잘 맞는 파일에서는 역합성곱이 잡을 계통 성분이
        # 없고, 그때 IR 은 오히려 나빠지게 한다 (engine/room.holdout_gain).
        hg = _room.holdout_gain(dry[:m], ref[:m], **kw)
        room_ir = _room.estimate_ir(dry[:m], ref[:m], **kw)
        print(f"녹음 경로 IR: {src} 에서 {len(room_ir)} 탭 "
              f"({len(room_ir)/48000*1000:.0f} ms) 추정, 직접음 {_room.direct_gain(room_ir):.3f}",
              flush=True)
        print(f"  홀드아웃(앞 절반 추정 → 뒤 절반 시험): {hg['before']:.3f} -> {hg['after']:.3f}"
              f"  {'쓴다' if hg['improves'] else '**안 좋아진다**'}", flush=True)
        if not hg["improves"] and not a.room_force:
            print("  -> 방을 넣지 않는다 (--room-force 로 강제 가능)", flush=True)
            room_ir = None
    if a.fric_am is not None:
        from formant_ml.engine import noise as _nz
        _nz.FRIC_AM_DEPTH = float(a.fric_am)
    if a.voiced_soft is not None:
        from formant_ml.engine import glottis as _gl
        _gl.VOICED_SOFT = float(a.voiced_soft)
    if a.harm_jit is not None or a.asp_am is not None or a.harm_jit_onset is not None:
        from formant_ml.engine import glottis as _gl
        if a.harm_jit is not None:
            _gl.HARM_PHASE_JIT = float(a.harm_jit)
        if a.harm_jit_onset is not None:
            _gl.HARM_PHASE_ONSET = float(a.harm_jit_onset)
        if a.asp_am is not None:
            _gl.ASP_AM_DEPTH = float(a.asp_am)
    if a.hf_fixed:
        from formant_ml.engine import tract as _tr3
        _tr3.HF_FIXED = True
    if a.hpc:
        from formant_ml.engine import tract as _tr4
        _tr4.HPC_TAIL = True
        if a.hpc_fmax is not None:
            _tr4.HPC_FMAX = float(a.hpc_fmax)
    if a.front_q:
        from formant_ml.engine import tract as _tr5
        _tr5.FRONT_Q_CTRL = True
    if a.grids:
        _fit.GRID_MS = tuple(float(x) for x in a.grids.split(","))
        assert 1 <= len(_fit.GRID_MS) <= len(_fit.STAGES), "--grids 는 1~4 개"
    if a.hjit_fit:
        from formant_ml.engine import glottis as _gl7
        _gl7.HJIT_FIT = True
    if a.src_eq:
        from formant_ml.engine import glottis as _gl10
        _gl10.SRC_EQ = True
    if a.pulse_lock:
        _fit.PULSE_LOCK = True
    if a.mvf is not None and hasattr(eng.aspiration, "log_mvf"):
        import math as _m
        _as = eng.aspiration
        _lo, _hi = _m.log(_as.MVF_LO), _m.log(_as.MVF_HI)
        _x = min(max((_m.log(a.mvf) - _lo) / (_hi - _lo), 1e-4), 1 - 1e-4)
        _x0 = (_m.log(_as.MVF_INIT) - _lo) / (_hi - _lo)
        with torch.no_grad():
            _as.log_mvf.fill_(_m.log(_x / (1 - _x)) - _m.log(_x0 / (1 - _x0)))
        _fit.MVF_FROZEN = True
        print(f"기식 MVF 고정 {float(_as.mvf()):.0f} Hz", flush=True)
    if a.phase_lf:
        from formant_ml.engine import glottis as _gl9
        _fit.PHASE_FMAX = float(a.harm_jit_onset if a.harm_jit_onset is not None else _gl9.HARM_PHASE_ONSET)
    if a.hjit_cycle:
        from formant_ml.engine import glottis as _gl8
        _gl8.HJIT_CYCLE = True
    if a.open_damp:
        from formant_ml.engine import tract as _tr6
        _tr6.OPEN_DAMP = True
    if a.fric_lp:
        _fit.FRIC_LP_FIT = True
    if a.noise_v2:
        # v2.1 (docs/NOISE_SOURCE_REVIEW.md §4): 기식은 유성 정도로 갈라 유성분만 MVF(적합) 위로,
        # 무성분은 전대역으로 정상 종속 가지에. 앞공동 Q 는 장애물에 묶는다. (out/L46 은 v2.0 이다.)
        from formant_ml.engine import noise as _nz2, tract as _tr2
        _tr2.NOISE_V2 = True
        _tr2.FRONT_Q_OBSTACLE = True
        _nz2.FRIC_V2 = True
        _nz2.ASP_SPLIT = True
    if a.asp_corner is not None:
        eng.aspiration.corner = float(a.asp_corner)
    if a.asp_floor is not None:
        eng.aspiration.floor = float(a.asp_floor)
    if a.hnm:
        from formant_ml.engine import glottis as _gl
        _gl.MVF_HZ = float(a.hnm)
        eng.aspiration.corner = float(a.hnm)      # 기식은 MVF 위에 몰린다 (배음과 상보)
        eng.aspiration.floor = 0.05
    if a.device != "cpu":
        eng = eng.to(a.device)
    from formant_ml.engine.fit import DEFAULT_PARAMS as _DP
    frozen = set(a.freeze or ())
    if not a.voice_gain:
        frozen.add("voice_gain")          # 켜지 않으면 0 dB 로 얼린다 (예전 판과 같은 모형)
    if not a.front_q:
        frozen.add("front_q")             # 켜지 않으면 쓰이지 않는 값이라 얼린다
    # 엔진이 **안 쓰는** 손잡이를 적합 대상에 남기면 열만 늘어난다 — §40 의 열 희석이 그것이다.
    # `out/M/M11` 이 33 → 43 열로 시작해 2.1 단계에서 85.4 → 77.6 으로 떨어졌다.
    if not a.aux_poles:
        frozen.update(f"aux{k}_mix" for k in range(1, 8))
    if not a.aux_zeros:
        frozen.update(f"azr{k}_mix" for k in range(1, 4))
        frozen.update(f"azr{k}_f" for k in range(1, 4))
    if not a.hf_zeros:
        frozen.update(f"hzr{k}_mix" for k in range(1, 4))
        frozen.update(f"hzr{k}_f" for k in range(1, 4))
    if not a.hf_poles:
        frozen.update(f"hzp{k}_mix" for k in range(1, 4))
        frozen.update(f"hzp{k}_f" for k in range(1, 4))
    if a.pulse_lock:
        frozen.add("f0_target")           # 위상은 목표 펄스가 만든다 — f0 는 관측되지 않는 자유도가 된다
    bad = frozen - set(_DP)
    if bad:
        raise SystemExit(f"--freeze: 적합 대상이 아닌 이름 {sorted(bad)}")
    if a.events is not None:
        _fit.EVENT_W = float(a.events)
    if a.hf_stage is not None:
        _fit.HF_STAGE_ITERS = int(a.hf_stage)
    if a.hf_eq:
        from formant_ml.engine import tract as _tr10
        _tr10.HF_EQ = True
    if a.hf_free:
        from formant_ml.engine import analyze as _an2
        _an2.HF_FREE = True
        _fit.DEFAULT_PARAMS = tuple(_fit.DEFAULT_PARAMS) + ("f5", "f6", "f7", "f8",
                                                           "bw5", "bw6", "bw7", "bw8")
    if a.aux_zeros:
        from formant_ml.engine import tract as _tr11
        _tr11.AUX_ZEROS = True
    if a.hf_zeros:
        from formant_ml.engine import tract as _tr12
        _tr12.HF_ZEROS = True
    if a.hf_poles:
        from formant_ml.engine import tract as _tr13
        _tr13.HF_POLES = True
    if a.aux_poles:
        from formant_ml.engine import tract as _tr9
        _tr9.AUX_POLES = True
    if a.formant_band:
        import json as _json2
        _fb = _json2.load(open(a.formant_band, encoding="utf-8")).get("formant_band", {})
        _fit.FORMANT_BAND = {k: (float(v[0]), float(v[1])) for k, v in _fb.items()}
        _qb = _json2.load(open(a.formant_band, encoding="utf-8")).get("q_band", {})
        _fit.Q_BAND = {k: (float(v[0]), float(v[1])) for k, v in _qb.items()}
        print("포먼트 구역: " + ", ".join(f"{k} {v[0]:.0f}~{v[1]:.0f}"
                                        for k, v in sorted(_fit.FORMANT_BAND.items())), flush=True)
    if initial_constants is not None:
        apply_calibration(eng, initial_constants)
    if a.hf_eq:
        eng.tract.recording_eq_enabled = True
    locked_constants = ()
    if a.speaker_lock:
        _sp = load_calibration(a.speaker_lock)
        if not _sp["speaker"]:
            raise ValueError("--speaker-lock requires speaker-scope constants")
        locked_constants += apply_calibration(eng, _sp, ("speaker",))
        if "formant_band" in _sp:
            _fit.FORMANT_BAND = {k: (float(v[0]), float(v[1])) for k, v in _sp["formant_band"].items()}
            print("  포먼트 구역: " + ", ".join(f"{k} {v[0]:.0f}~{v[1]:.0f}"
                                              for k, v in sorted(_fit.FORMANT_BAND.items())), flush=True)
        excluded = sorted(set(_sp["recording"]) | set(_sp["utterance"]))
        if excluded:
            print("  화자 잠금에서 제외: " + ", ".join(excluded), flush=True)
        print(f"화자 상수 잠금: {a.speaker_lock} 에서 {len(locked_constants)} 항목", flush=True)
    if recording_constants is not None:
        locked_constants += apply_calibration(eng, recording_constants, ("recording",))
        print(f"녹음 경로 잠금: {a.recording_lock} (발화별 레벨 정규화는 유지)", flush=True)
    initial_utterance = initial_constants["utterance"] if initial_constants is not None else {}
    fit = CopySynthFitter(eng, seg, sr, track, phase_weight=a.phase, room_ir=room_ir,
                          device=a.device, lam_l1=float(a.l1 or 0.0),
                          harmonic_weight=a.harmonic,
                          locked_constants=locked_constants,
                          initial_gain_db=initial_utterance.get("gain_db"),
                          initial_pulse_phi0=initial_utterance.get("pulse_phi0", 0.0),
                          params=tuple(p for p in _DP if p not in frozen))
    inactive = sorted(set(locked_constants) - fit.constant_parameters.keys())
    if inactive:
        print("  현재 렌더 옵션에서 비활성인 저장 상수: " + ", ".join(inactive), flush=True)
    if fit.harmonic is not None:
        print(f"배음 크기: 가중 {fit.harmonic_weight:g}, "
              f"목표 관측 {fit.harmonic.observations}개 (0~3 kHz)", flush=True)
    if a.smooth_w is not None:
        fit.lam_smooth = float(a.smooth_w)
    if a.prior_w is not None:
        fit.lam_prior = float(a.prior_w)
    kw = {} if a.lr_global is None else {"lr_global": a.lr_global}
    rep = fit.fit_staged(global_iters=a.global_iters, stage_iters=a.stage_iters,
                         lr_frame=a.lr_frame, phase_iters=a.phase_iters,
                         **({"lr_phase": a.lr_phase} if a.lr_phase else {}),
                         patience=a.patience, **kw)
    print(rep)

    # 보정값은 진단용이며 청취상 동등성이나 신뢰구간을 뜻하지 않는다.
    fid = fit.fidelity()
    print(fidelity_summary(fid))
    if not fid["resolved"]:
        print("  * 분산 보정으로 편향을 판별하지 못했다. 하한·합격 판정으로 쓰지 말고 "
              "잡음비와 아래 치찰음 지문을 함께 읽을 것.")

    out = fit.render()
    harmonic_report = None
    if fit.harmonic is not None:
        harmonic_report = fit.harmonic.report(torch.as_tensor(out, device=a.device))
        if harmonic_report["observations"]:
            print(f"배음 오차: {harmonic_report['mae_db']:.2f} dB, "
                  f"95% {harmonic_report['p95_db']:.2f} dB, "
                  f"최악 {harmonic_report['max_db']:.2f} dB", flush=True)
        else:
            print("배음 오차: 관측 가능한 유성 창 없음", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sf.write(a.out + "_target.wav", seg, sr)
    sf.write(a.out + "_fit.wav", out, 48000)
    if room_ir is not None:
        # 마른 출력도 같이 낸다 — 학습 데이터로 나가는 것은 이쪽의 파라미터다.
        with torch.no_grad():
            sf.write(a.out + "_dry.wav", fit.synth_dry()[0].cpu().numpy(), 48000)
        np.save(a.out + "_room.npy", room_ir)
    res = fit.result_track()
    # **엔진 쪽 적합 파라미터도 같이 남긴다.** `log_extra_bw`·`log_front_bw` 는 제어열이
    # 아니라 성도 모듈의 파라미터라 `values` 에 없다. 빠뜨리면 재렌더가 그 둘을 0 으로 되돌려
    # 고역이 저장된 `_fit.wav` 와 달라진다 — 실측 `out/L22/s040` 의 10~15 kHz 포락 동기가
    # 저장본 0.994 대 재렌더 0.525 였다. 재렌더할 때는 `engine_params` 를 도로 넣어라.
    constants = fit.acoustic_constants()
    old_fields = legacy_fields(constants)
    eng_p, hf_p, asp_p = (old_fields[n] for n in ("engine_params", "hf_params", "asp_params"))
    if hasattr(eng.glottis, "src_eq_db"):
        from formant_ml.engine import glottis as _glp
        if _glp.SRC_EQ:
            _q = (_glp.SRC_EQ_MAX_DB * torch.tanh(eng.glottis.src_eq_db / _glp.SRC_EQ_MAX_DB)).tolist()
            print("  음원 EQ dB " + " ".join(f"{f/1000:.1f}k:{g:+.1f}" for f, g in zip(_glp.SRC_EQ_F, _q)), flush=True)
    if getattr(eng.tract, "open_damp", None) is not None:
        from formant_ml.engine import tract as _trp
        if _trp.OPEN_DAMP:
            print(f"  개방기 감쇠 k_b {float(_trp.OPEN_DAMP_MAX * torch.sigmoid(eng.tract.open_damp)):.2f}, "
                  f"k_f {float(0.15 * torch.tanh(eng.tract.open_f1) + 0.05):+.3f}", flush=True)
    if _fit.FRIC_LP_FIT:
        print(f"  마찰 절벽 배율 {float(torch.exp(eng.frication.log_lp_ratio)):.2f} (가둠 전)", flush=True)
    asp_p["mvf_hz"] = float(eng.aspiration.mvf().detach().cpu())
    if asp_p:
        print(f"  기식 MVF {asp_p['mvf_hz']:.0f} Hz", flush=True)
    np.savez(a.out + "_track.npz", values=res.values, frame_ms=res.frame_ms,
             names=np.array(PARAM_NAMES),
             engine_params=np.array(json.dumps(eng_p)),
             asp_params=np.array(json.dumps(asp_p)),
             hf_params=np.array(json.dumps(hf_p)),
             acoustic_constants=np.array(json.dumps(constants, allow_nan=False)),
             gain_db=np.array(fit.gain_db()))
    with open(a.out + "_constants.json", "w", encoding="utf-8") as f:
        json.dump(constants, f, ensure_ascii=False, indent=1, allow_nan=False)
    with open(a.out + "_report.json", "w", encoding="utf-8") as f:
        json.dump({"env": rep.env, "fine": rep.fine, "per_size": rep.per_size,
                   "fine_corr": fid["fine_corr"], "floor": fid["floor"],
                   "resolved": fid["resolved"], "trust": fid["trust"],
                   "noise_ratio": fid["noise_ratio"],
                   "score_kind": "diagnostic_not_perceptual_equivalence",
                   "harmonic": harmonic_report, "harmonic_weight": fit.harmonic_weight,
                   "constant_budget": fit.constant_budget(), "frame_controls": len(fit.names),
                   "event_coefficients": 0 if fit.w_ev is None else fit.w_ev.numel(),
                   "spectrum_match": fid["spectrum_match"],
                   "loss": rep.loss, "gain_db": fit.gain_db(),
                   "moved": {n: [x, z] for n, x, z in fit.moved()}},
                  f, ensure_ascii=False, indent=1)

    print("\n대역 에너지 (총합 대비 dB)")
    print("           " + " ".join(f"{lo // 1000}-{hi // 1000}k".rjust(6) for lo, hi in BANDS))
    print("목표      " + " ".join(f"{v:6.1f}" for v in band_db(seg, sr)))
    print("합성      " + " ".join(f"{v:6.1f}" for v in band_db(out, 48000)))
    # 치찰음 지문 — 조음 위치가 맞는가 (Jongman et al. 2000). 대역 에너지표가 맞아도
    # 무게중심이 1 kHz 어긋나면 /s/ 가 /ʃ/ 로 들린다.
    from formant_ml.engine import turbulence as tb
    c = tb.compare(fit.target[0].numpy(), out, 48000.0)
    print("\n치찰음 지문 (목표 -> 합성)")
    print(f"  무게중심 {c['target']['centroid']:7.0f} -> {c['synth']['centroid']:7.0f} Hz "
          f"({c['centroid_err']:+.0f})   봉우리 {c['target']['peak']:6.0f} -> "
          f"{c['synth']['peak']:6.0f} Hz ({c['peak_err']:+.0f})")
    print(f"  치찰도   {c['target']['sibilance']:7.1f} -> {c['synth']['sibilance']:7.1f} dB "
          f"({c['sibilance_err']:+.1f})   대역 MAE {c['band_mae_db']:.2f} dB   "
          f"변조 MAE {c['mod_mae_pct']:.1f} %p")

    # 지글거림 — **마찰 프레임만** 골라 고역 포락의 변조 지수를 목표와 나눈다.
    # 위의 "변조 MAE" 는 파일 전체를 대역별 백분율로 재므로 마찰 구간의 맥동이
    # 유성 구간에 묻힌다 (docs/MEASUREMENTS.md §13).
    ac = res.values[:, INDEX["a_c"]]
    ps = res.values[:, INDEX["p_sub"]]
    f0v = res.values[:, INDEX["f0_target"]]
    hop48 = int(round(res.frame_ms * 48000.0 / 1000.0))
    for label, sel in (("마찰", (ps > 2.0) & (ac < 0.5)),
                       ("유성", (ps > 2.0) & (ac > 1.0))):
        if sel.sum() < 20:
            continue
        f0m = float(np.median(f0v[sel]))
        mb = [(0.8 * f0m, 2.5 * f0m), (60.0, 150.0), (150.0, 400.0)]
        msk = np.repeat(sel.astype(float), hop48)
        tgt48 = fit.target[0].numpy()
        vt, lt = tb.env_modulation_index(tgt48, 48000.0, msk, mb)
        vs, ls = tb.env_modulation_index(out, 48000.0, msk, mb)
        r = vs / np.maximum(vt, 1e-30)
        print(f"  지글거림({label} {sel.mean()*100:2.0f}%, F0 {f0m:.0f} Hz)  "
              f"F0대역 {r[0]:5.2f}x  60-150 {r[1]:5.2f}x  150-400 {r[2]:5.2f}x   "
              f"대역레벨 {lt:.1f} -> {ls:.1f} dB")

    print("\n움직인 파라미터 (초기 중앙값 -> 적합 중앙값)")
    for n, x, z in fit.moved():
        if abs(z - x) > 0.05 * max(abs(x), 1e-6):
            print(f"  {n:12s} {x:10.3f} -> {z:10.3f}")
    print(f"\n결과: {a.out}_target.wav / {a.out}_fit.wav / {a.out}_track.npz")


if __name__ == "__main__":
    main()
