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

import math
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import soundfile as sf
import torch

from formant_ml.engine.analyze import analyze
from formant_ml.engine.calibration import (apply_calibration, legacy_fields,
                                           load_calibration, source_config)
from formant_ml.engine.control import INDEX, PARAM_NAMES, PARAMS
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


#: 제어열·엔진 상수 밖에 있는 **적합기의 전역 스칼라** (로그 영역). 판마다 저장하고 `--init` 에서 되살린다 (§52.439).
from formant_ml.engine.pipeline import FIT_SCALARS  # noqa: E402  (저장할 적합기 전역 스칼라 — 복원은 engine.pipeline.restore_state)


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
    ap.add_argument("--lr-global", type=float, default=None,
                    help="생략하면 짧은 탐침으로 자동 선택 (구간마다 맞는 값이 다르다)")
    ap.add_argument("--lr-frame", type=float, default=0.04,
                    help="2.x 단계의 학습률 (단계마다 0.75 배로 준다). 2.1 이 90 %% 아래에서 **수렴**하면 "
                         "지역 최소를 의심해 이 값을 올려 본다 (사용자 지시, MEASUREMENTS §52.93)")
    ap.add_argument("--lr-wide", action="store_true",
                    help="1 단계 lr 자동 탐색 범위를 위로 넓힌다 (0.015~0.09 -> 0.015~0.25, §52.93). "
                         "지금까지 모든 판이 범위의 **맨 위 0.09** 를 골랐다 — 범위가 낮다는 뜻이다")
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu",
                    help="cpu / cuda / mps. **cpu 를 쓰라** — 실측으로 cuda 가 2.2~3.1 배 느리다 "
                         "(MEASUREMENTS §52.90: 걸음이 작은 연산 수만 개라 커널 실행 부하가 지배하고, 시변 IIR 의 "
                         "결합 스캔이 float64 인데 이 장치의 fp64 는 fp32 의 1/46 이며, numba 빠른 경로는 GPU "
                         "텐서를 호스트로 복사해 돈다). 적합 한 건이 RSS 3.8 GB 를 쓴다 — 병렬로 돌릴 개수는 "
                         "GPU 메모리가 아니라 그것으로 정하라")
    ap.add_argument("--patience", type=int, default=0,
                    help="이 회차 동안 손실이 안 줄면 그 단계를 끝낸다 (0 = 끔). "
                         "긴 음원에서는 켜는 편이 낫다 — 수렴한 단계에 예산을 다 쓴다")
    ap.add_argument("--prior", action="append", default=None, metavar="이름=값",
                    help="사전 가중(fit.PRIOR_W) 덮어쓰기. 여러 번 줄 수 있다. "
                         "예: --prior aspiration=3.0")
    ap.add_argument("--tilt-cap", type=float, default=None,
                    help="소스 기울기(tilt)가 먹히는 상한 주파수 [Hz] "
                         "(glottis.TILT_MAX_HZ, 기본 5000). 0 이면 상한 없음 — "
                         "그러면 성문 펄스가 사각파처럼 날카로워진다 (MEASUREMENTS §25)")
    ap.add_argument("--motion", action="store_true",
                    help="제어열을 값이 아니라 **움직임**으로 매개화한다 (fit.MOTION). "
                         "자유 파라미터를 가속도로 두고 두 번 적분하므로 위치·속도의 "
                         "연속성이 벌점이 아니라 구조로 보장되고, 남는 불연속인 "
                         "가속도는 lam_smooth 가 저크 벌점으로 문다")
    ap.add_argument("--motion-grid", type=float, default=None,
                    help="움직임 모드의 가장 고운 격자 [ms] (fit.MOTION_MIN_GRID_MS, 기본 2). "
                         "작을수록 미세구조를 담지만 난동도 표현 가능해지므로 --motion-l2 와 같이 쓴다")
    ap.add_argument("--balance", action="store_true",
                    help="손실 항의 기울기 균형을 켠다 (fit.BALANCE). 단계마다 항별 "
                         "기울기 규범을 재서 가중이 '포락 대비 영향력' 을 뜻하게 한다. "
                         "끄면 파형 상관이 포락보다 ~40 배 세게 적합을 끈다")
    ap.add_argument("--noise-global", action="store_true",
                    help="fric_gain·aspiration 을 시간 제어가 아니라 전역 눈금으로 만든다 "
                         "(fit.NOISE_GLOBAL). 마찰의 시간 구조는 협착·레이놀즈 수가 정하고 "
                         "적합기는 a_c 로 마찰을 켜고 끈다. 모음에 샌 마찰 잡음 융단(10 kHz 위 "
                         "평탄)을 막는다")
    ap.add_argument("--seed", type=int, default=None,
                    help="렌더 난수 씨앗 (EngineConfig.seed, 기본 0). 같은 궤적을 씨앗만 바꿔 "
                         "두 번 렌더하면 **실현의 차이만** 남는다 — 도달 가능한 상한을 재는 데 쓴다")
    ap.add_argument("--prior-w", type=float, default=None,
                    help="lam_prior (사전값 L2, 기본 1e-4). 0 이면 끈다")
    ap.add_argument("--voice-gain", action="store_true",
                    help="성문 배음에만 거는 빠른 이득 제어(voice_gain)를 적합한다. 없으면 0 dB 로 얼린다 — "
                         "주기별 진폭(시머)을 잡음을 썰지 않고 따라간다 (docs/FOUNDATION.md, MEASUREMENTS §51)")
    ap.add_argument("--init", default=None, metavar="STEM",
                    help="앞선 적합(STEM_track.npz)의 제어열·전역 엔진값에서 출발한다 (두 번째 패스 — 다듬기). "
                         "펄스·유성·마찰 표시와 사건은 이번 분석의 것을 쓴다")
    ap.add_argument("--grids", default=None, metavar="MS,MS,...",
                    help="2.x 단계들의 격자 [ms] (fit.GRID_MS, 기본 20,10,5,1). 창은 단계 순서대로 늘어난다. "
                         "같은 값을 되풀이하면 층을 더하지 않고 그 격자에서 창만 늘려 이어 적합한다")
    ap.add_argument("--src-eq", action="store_true",
                    help="화자 음원 EQ (glottis.SRC_EQ, MEASUREMENTS §51.17): 성문 배음에 고정 주파수 봉우리 여섯의 "
                         "이득(±9 dB)을 화자 전역으로 적합한다 — 음원 기울기 넷(Kreiman·Garellek)의 자리")
    ap.add_argument("--pulse-lock", action="store_true",
                    help="성문 위상을 목표 녹음의 폐쇄 시각에 잠근다 (fit.PULSE_LOCK, MEASUREMENTS §51.16). "
                         "f0_target 은 적합에서 빠진다 (위상을 만들지 않으므로)")
    ap.add_argument("--no-fit", action="store_true",
                    help="적합하지 않고 --init 제어열 그대로 렌더·채점만 한다 (MEASUREMENTS §52.82). "
                         "저장된 판을 그 명령줄로 되살려 진단할 때 쓴다 — 모듈 전역을 바꾸는 깃발이 많아 "
                         "제어열만으로는 같은 소리가 안 난다")
    ap.add_argument("--pulse-refine", type=int, default=0, metavar="N",
                    help="--pulse-lock 과 함께: 앞 N 개 2.x 단계가 끝날 때마다 렌더를 목표와 견줘 창마다 최적 지연을 "
                         "재고 그만큼 잠금 위상을 앞당긴다 (fit.PULSE_REFINE, MEASUREMENTS §52.80). 표시는 복사 파형에 "
                         "찍히고 위상은 음원에 걸리므로, 그 사이 성도 군지연(모음마다 0.2~0.4 ms)을 되먹임으로 흡수한다")
    ap.add_argument("--open-damp", action="store_true",
                    help="성문 개방기 F1 감쇠 (tract.OPEN_DAMP, MEASUREMENTS §51.9): 성문이 열린 동안 F1 대역폭·주파수를 "
                         "주기마다 올린다. 세기 둘은 화자 전역 적합값")
    ap.add_argument("--hf-fixed", action="store_true",
                    help="화자 고정 고역 구조 (tract.HF_FIXED, docs/FOUNDATION.md §3.3): F5 위 극은 F4 를 따라가지 않고 "
                         "절대 위치 + 전역 이동·넓은 대역폭(Q≈5), 이상와 영점 하나")
    ap.add_argument("--noise-v2", action="store_true",
                    help="잡음 가지 v2 (docs/NOISE_SOURCE_REVIEW.md §3): 잡음은 저 Q 성도 사본·병렬 저 Q 앞공동을 "
                         "지나고, 마찰 구동은 매끈한 문턱 + 2 ms 포락 평활, 기식은 MVF(5.5 kHz) 위 2 차 고역")
    ap.add_argument("--floor", action="store_true",
                    help="목표의 녹음 바닥 잡음을 합성에 고정으로 더한다 (fit.FLOOR_NOISE). 없으면 적합기가 "
                         "무음의 바닥을 숨소리 + 성도 공명으로 흉내 내 무음에 공명 줄이 선다")
    ap.add_argument("--hf-ladder", choices=("legacy", "spread"), default="legacy",
                    help="고역 보정 사다리 (§51.58). legacy = 예전(대역폭 하한 0.933, 자리 ±16 %%), "
                         "spread = 하한 1.65·자리 ±65 %% 로 풀어 **균일한 빗살이 안 서게** 한다. "
                         "13~19 kHz 에 켜켜이 쌓인 가로줄이 legacy 의 증상이다")
    ap.add_argument("--add-prior", type=float, default=None, metavar="W",
                    help="유성 틀 내전 **수준**의 코퍼스 분포 사전 세기 (fit.ADD_PRIOR_W, MEASUREMENTS §52.442). "
                         "로짓 스튜던트 t, 중앙 0.44 (코퍼스 역모형), 무성 틀에는 안 건다")
    ap.add_argument("--sib-prior", type=float, default=None, metavar="W",
                    help="치찰 틀 협착 면적의 분포 사전 (fit.SIB_PRIOR_W) + 치찰 협착 관측(analyze.SIB_OBS, 초기 a_c 0.1 cm²) — "
                         "적합기가 /ㅅ/ 을 성문 기식으로 흉내 내던 지름길을 막는다 (§52.446)")
    ap.add_argument("--sib-prior-edge", type=float, default=None, metavar="MS",
                    help="치찰 사전의 무게를 구간 양 끝 MS 안에서 0 → 1 로 올린다 (fit.SIB_PRIOR_EDGE_MS, §52.472). 표지는 마찰이 사라질 때 "
                         "끝나는데 그때는 협착이 이미 열리는 중이다 — 038 [ɕ] 끝에서 사전이 협착을 붙잡아 전 대역 10~20 dB 가 모자랐다")
    ap.add_argument("--nasal-i-guard", type=float, nargs="?", const=8.0, default=None, metavar="DB",
                    help="비음 머머 검출에서 /이/ 를 가린다 (analyze.NASAL_I_PROM_DB, §52.445): 2.5–4 kHz 봉우리가 1.2–2 kHz 보다 DB(기본 8) "
                         "넘게 솟는 틀은 비음이 아니다")
    ap.add_argument("--creak", action="store_true",
                    help="불규칙 발성(creak) 펄스를 Praat 펄스 틈에서 찾아 더한다 (analyze.CREAK_PULSES, MEASUREMENTS §52.441). "
                         "그 틀은 유성·F0 는 펄스에서·내전 ≥ 0.6, 펄스 차단 게이트는 국소 간격으로 끊김을 판정")
    ap.add_argument("--burst-back-pct", type=float, default=None, metavar="PCT",
                    help="파열 검출에서 **앞 구간의 기준 수준을 잡는 분위** [%%] (analyze.BURST_BACK_PCT, "
                         "기본 100 = 최댓값). 폐쇄가 조용하지 않으면 최댓값이 폐쇄를 가려 파열을 놓친다 — "
                         "사용자가 짚은 스윕 자리의 +21 dB 파열이 문턱을 1.1 dB 차로 못 넘었다 (§52.395)")
    ap.add_argument("--stop-close", action="store_true",
                    help="파열 앞의 **무음을 폐쇄로 관측**해 a_c 를 닫는다 (analyze.STOP_CLOSE, §52.353). "
                         "a_c 초기값은 고역/저역 비에서만 나와 **소리가 없는 폐쇄를 원리적으로 못 본다** — "
                         "실측 L11 에서 파열 다섯 중 셋이 a_c 2.2~2.9 cm²(활짝 열림)였고 그 자리에서 "
                         "포먼트가 수백 Hz 를 휩쓸어 sweep 이 됐다. --burst-smooth 뒤에 걸린다")
    ap.add_argument("--front-obs", action="store_true",
                    help="**front_len 을 프레임마다 관측한다** (analyze.FRONT_LEN_OBS, §52.353·354). "
                         "지금은 전 프레임이 prof.sib_front_len_cm(0.92 cm) 상수라 적합 궤적이 2.4 cm 를 "
                         "못 넘고, 그래서 **연구개 협착(앞공동 6~8 cm)이 원리적으로 불가능**하다. "
                         "잡음 우세 프레임의 스펙트럼 정점에서 c/(4·f_p) 로 잰다")
    ap.add_argument("--front-free", type=float, default=None, metavar="W",
                    help="PRIOR_W['front_len'] 를 W 로 낮춘다 (기본 10). 연구개 협착은 앞공동이 6~8 cm "
                         "여야 하는데 **관측이 안 된다** — 앞공동 공진이 스펙트럼을 지배하는 것은 "
                         "치찰음뿐이고 기식은 정점이 모음 포먼트에 선다(§52.360). 관측하는 척하는 대신 "
                         "손실이 요구할 때 적합기가 뒤로 보낼 수 있게 자유만 준다")
    ap.add_argument("--burst-smooth", type=float, default=None, metavar="MS",
                    help="파열 자리 ±MS 의 **조음 궤적을 이어 붙인다** (analyze.smooth_over_bursts, §52.329). "
                         "포먼트 추적기가 광대역 파열에 공진을 맞춰 f2 속도가 110 Hz/ms(바깥 3.1, 생리 상한 93 "
                         "초과), f0 가 15.0(바깥 0.44) 으로 튄다. 적합기가 그것을 따라가며 활공으로 번져 "
                         "sweep 으로 들린다. 파열의 소리는 --click-events / eventfit 이 진다. 12 정도")
    ap.add_argument("--hold-fric", action="store_true",
                    help="마찰 **안**에서 포먼트·대역폭을 그 구간의 고원 값으로 붙잡는다 (fit.HOLD_FRIC, §51.30). "
                         "사용자가 치찰음이 좋다고 한 out/L7 은 마찰 안에서 f3·f4 가 0.003 %%/ms 로 얼어 있었다. "
                         "심어서 잰 비용은 포락 −0.06 %% 이하다")
    ap.add_argument("--hold-gap", action="store_true",
                    help="**유성도 마찰도 아닌** 사각지대에서 공명의 모양을 분석 궤적으로 갈아 끼우고 "
                         "수준만 남긴다 (fit.HOLD_GAP, §52.339). 사용자가 짚은 스윕 넷이 전부 마찰 "
                         "시작 9~44 ms 앞의 그 사각지대였다 — 손실도 hold 도 움직임 예산도 안 보는 자리다")
    ap.add_argument("--mprior", type=float, default=None, metavar="W",
                    help="기관 운동 사전 (fit.MPRIOR_W, §52.425): 내전·협착 면적의 5 ms 차분에 문헌 최고 속도를 99 분위로 둔 "
                         "스튜던트 t 음의 로그 가능도 — 빠른 동작을 막지 않고 비싸게 한다")
    ap.add_argument("--late-reverb", type=float, nargs="?", const=-15.0, default=None, metavar="G_DB",
                    help="방 잔향 꼬리를 목소리 뒤·녹음 사슬 앞에 더한다 (fit.LATE_REVERB, §52.428): 지수 감쇠 잡음, RT60 과 "
                         "잔향/직접 비 두 스칼라를 적합. 값은 초기 비 [dB] (기본 -15)")
    ap.add_argument("--late-reverb-rt", type=float, default=None, metavar="S", help="잔향 꼬리 초기 RT60 [s] (기본 0.30)")
    ap.add_argument("--stop-release-sharp", action="store_true",
                    help="폐쇄 관측의 개방을 가장 가파른 1–16 kHz 상승에 맞추고 개방 경사를 1 ms 로 (analyze.STOP_RELEASE_SHARP, §52.451)")
    ap.add_argument("--vowel-onset", action="store_true",
                    help="전사(metadata_*.csv)의 첫 음절이 모음(초성 ㅇ)이면 발화 첫머리 파열을 성문 시작으로 보고 구강 파열에서 뺀다 (§52.467)")
    ap.add_argument("--declick-auto", action="store_true",
                    help="녹음 속 고립 클릭을 **자동으로** 찾아(scripts/diag/corpus_transients.py 의 규칙: 되풀이 안 함·앞뒤 수준 같음·짧음·동반 없음) "
                         "engine/declick.py 검출기도 동의하는 곳만 메운다 (§52.464). --declick-at 과 함께 쓰면 둘을 합친다")
    ap.add_argument("--declick-at", default=None, metavar="T1,T2,...",
                    help="녹음 속 발화가 아닌 클릭(마우스 등)을 이 파일 시각 [s] 둘레 ±5 ms 에서 찾아 1.5 kHz 위만 이웃 성문 주기로 메운다 — "
                         "적합 전, 잡음 제거 전에 (engine/declick.py, MEASUREMENTS §52.461)")
    ap.add_argument("--release-rate", type=float, default=None, metavar="W",
                    help="파열 개방의 면적 증가 속도를 파열별로 적합하고 문헌 분포(양순 100·치조 50·연구개 25 cm²/s, Stevens 1998) 사전을 무게 W 로 건다 "
                         "(voice.RELEASE_RATE, fit.RELEASE_PRIOR_W, MEASUREMENTS §52.456)")
    ap.add_argument("--aero-gate", action="store_true",
                    help="공기역학(구강압·마찰·기식)이 구강 게이트 oral_open 도 보게 한다 — 직렬 오리피스 유효 면적 (voice.ORAL_AERO_GATE, §52.449)")
    ap.add_argument("--voice-bar-fixed", type=float, default=None, metavar="DB",
                    help="보이스 바 세기를 DB 에 고정한다 (화자 상수, fit.VB_FIXED, §52.447). --voice-bar 와 함께")
    ap.add_argument("--late-reverb-fixed", action="store_true",
                    help="잔향 RT·세기를 적합하지 않고 주어진 값에 고정한다 (녹음 상수, fit.LR_FIXED, §52.444)")
    ap.add_argument("--asp-strouhal", action="store_true",
                    help="기식 음원 셸프의 꺾임을 성문 제트의 스트로할 주파수 St·v/d 로 표본마다 (noise.ASP_STROUHAL, §52.432) — "
                         "모음(좁은 틈)은 수 kHz, 파열 뒤 기식(넓은 틈)은 ≈1 kHz")
    ap.add_argument("--voice-bar", action="store_true",
                    help="닫힌 성도의 벽 방사(보이스 바)를 더한다 (fit.VOICE_BAR, §52.433): 성문 음원 × 200 Hz 벽 공진 × 닫힌 정도, 세기는 적합")
    ap.add_argument("--oral-ode", type=float, nargs="?", const=0.5, default=None, metavar="CW",
                    help="구강압을 공기역학 상미분방정식으로 (glottis.ORAL_ODE, §52.436): C·dPo/dt = U_g − U_c − U_n, 벽 순응도 CW "
                         "[ml/cmH2O] (기본 0.5, Rothenberg 1968). 성문이 Ps−Po 를 보고 마찰은 협착 유량으로 레이놀즈·세기를 잰다")
    ap.add_argument("--transient-ramp", type=int, default=None, metavar="N",
                    help="단계마다 첫 N 회 동안 --transient 무게를 0 → W 로 올린다 (fit.TRANS_RAMP, §52.463) — 튐 많은 출발점에서 첫 붕괴를 막는다")
    ap.add_argument("--transient", type=float, default=None, metavar="W",
                    help="원본에 없는 짧은 튐 벌점 (fit.TRANS_W, §52.463): 0.5 ms 포락이 목표의 ±1.5 ms 최대보다 수준·솟음 둘 다 넘는 몫. "
                         "포락 손실(5.3 ms 창)이 못 보는 1~3 ms 클릭을 막는다")
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
    ap.add_argument("--slow", action="append", default=None, metavar="이름=ms",
                    help="파라미터별 가장 고운 층 [ms] 을 직접 준다. 여러 번 줄 수 있다")
    ap.add_argument("--hnr", type=float, default=None,
                    help="주기성(조화 대 비조화)을 목표에 일치시킨다 (fit.HNR_W, **기본 2.0 = 켜짐**). "
                         "없으면 적합기가 하모닉을 잡음으로 바꿔 같은 스펙트럼을 만든다 "
                         "— 실측 fric_gain +1022 %%. MEASUREMENTS §44")
    ap.add_argument("--harmonic", type=float, default=None, metavar="W",
                    help="유성 0~3 kHz 배음별 크기 오차 (기본 1). 목표 위상에 동기화한 최소제곱으로 "
                         "비교한다. 새 제어열은 없다. 0이면 이전 손실과 BW 관측 사전으로 돌아간다")
    ap.add_argument("--band-limit", type=float, default=None, metavar="HZ",
                    help="녹음 사슬의 **고정 대역 상한** [Hz] (fit.BAND_LIMIT_HZ, 기본 끔). 이 코퍼스의 벽은 "
                         "20050 Hz 다 — 12 파일 전부에서 21 kHz 가 20 kHz 보다 25~30 dB 아래다. 이걸 안 씌우면 "
                         "18.8·20.1 kHz 멜 대역의 오차가 11.5·6.2 dB 로 나머지의 열 배가 된다 (§52.291)")
    ap.add_argument("--floor-ref", action="store_true",
                    help="--floor 의 바닥을 **파일 전체**(잡음 제거 후, 목표와 같은 단계)의 가장 조용한 5 %% 프레임에서 잰다 "
                         "(fit.FLOOR_REF). 구간 안 무음에는 클릭·숨소리가 섞여 바닥이 8~10 dB 과했다 (§52.401)")
    ap.add_argument("--band-limit-lo", type=float, default=None, metavar="HZ",
                    help="녹음 사슬의 **저역 차단** [Hz] (fit.BAND_LIMIT_LO_HZ, 기본 끔). 원본은 60 Hz 아래가 비어 있는데 "
                         "우리는 +10~18 dB 컸다 (§52.401). 영위상 버터워스 크기, 대역 상한과 같은 자리")
    ap.add_argument("--extra-bw-floor", type=float, default=None, metavar="X",
                    help="고차 극 폭 배율의 하한 (tract.EXTRA_BW_FLOOR, 기본 1.65)")
    ap.add_argument("--hf-formant-bw-floor", type=float, default=None, metavar="X",
                    help="F5~F8 만의 폭 배율 하한 (tract.HF_FORMANT_BW_FLOOR, 기본 = EXTRA_BW_FLOOR). 목표 10.4 kHz 봉우리 "
                         "폭 약 520 Hz 를 하한 1.65 로는 못 낸다 (§52.245)")
    ap.add_argument("--harm-prior-fmax", type=float, default=None, metavar="HZ",
                    help="대역폭 관측 사전을 끄는 위쪽 끝 [Hz] (fit.HARM_PRIOR_FMAX, 기본 3000). --harm-fmax 와 따로 둔다 — "
                         "올리면 배음 항이 넓어지는 것과 bw 사전이 꺼지는 것이 한꺼번에 바뀌어 원인을 못 가른다 (§52.221)")
    ap.add_argument("--hnr-band", action="append", default=None, metavar="LO-HI",
                    help="HNR 항이 볼 통과 대역을 더한다 [Hz] (fit.HNR_BANDS). 여러 번 줄 수 있다. "
                         "예: --hnr-band 3000-4500. 대역 하나를 상수로 박아 두면 결함이 옮겨갔을 때 "
                         "손실이 못 본다 — 실측으로 1.5~3 kHz 는 고쳐졌고 3~4.5 kHz 가 남았다")
    ap.add_argument("--harm-fmax", type=float, default=None,
                    help="배음 크기 항의 위쪽 끝 [Hz] (fit.HARM_FMAX, 기본 3000). 포먼트 대역폭이 "
                         "드러나는 자는 하모닉 크기의 모양뿐이라 (빗살 대비는 비주기성만 본다) "
                         "이 위의 극은 폭을 잡아 주는 것이 없다 — 실측으로 F3 가 100 Hz 넓었다. "
                         "관측 불가 하모닉은 항이 스스로 버리므로 올려도 안전하다")
    ap.add_argument("--tube", action="store_true",
                    help="관 모드 (tract.TUBE, MEASUREMENTS §52.475): 포먼트 F1~F8·대역폭 B1~B8 을 조음 손잡이(로그 면적의 코사인 6 성분)와 "
                         "성도 길이에서 웹스터 고유모드·손실 계산으로 얻는다. 자유 포먼트·대역폭은 얼리고, 초기값은 분석 포먼트에서 역산한다")
    ap.add_argument("--td", action="store_true",
                    help="시간 영역 관 경로 (voice.TD_TUBE, MEASUREMENTS §52.476): 성문 면적 → 비선형 유량 → 임피던스 관(벽·입술 방사·비강) → "
                         "레이놀즈 난류를 한 풀이기로. 조음 손잡이(art1..6·tract_len)와 협착(a_c·c_place)·입술·연구개를 적합한다")
    ap.add_argument("--artic", choices=("cos", "maeda", "w02"), default="cos",
                    help="관 모양을 내는 방식 (voice_td.ARTIC, MEASUREMENTS §52.485, --td 와 함께). cos: 틀마다 자유 코사인 곡선 + 협착 (옛 곡선 맞춤). "
                         "maeda: 조음기 일곱(턱·혀 몸통 위치·모양·혀끝·입술 높이·돌출·후두 높이) → Maeda 조음 모형 → 면적·성도 길이. "
                         "초기값은 분석 포먼트의 조음 공간 코드북 역산 (비터비로 매끄러운 경로). "
                         "w02: VocalTractLab 여성 화자 W02 (MRI 3 차원 기하) 의 해부학 변수 17 개 → 대리 신경망 → 면적·성도 길이 (§52.486)")
    ap.add_argument("--psub-prior", type=float, default=None, metavar="W",
                    help="폐압의 생리 사전 무게 (fit.PSUB_PRIOR_W, §52.487): 말소리 틀의 log 폐압에 스튜던트 t — 중앙 7 cmH2O, 90 %% 약 4–13. "
                         "이어 적합(--prior-w 0)이 분석값 묶음을 풀면 폐압이 이득과 겹치는 방향으로 17–20 cmH2O 까지 부풀었다")
    ap.add_argument("--larynx", action="store_true",
                    help="후두 구조 (voice_td.LARYNX, §52.489, --td 와 함께): 좌우 이상와 + 후두실 곁관, 가성대 좁힘(쉼 틈 화자 상수 + 내전 제어 ff_add)")
    ap.add_argument("--nose", choices=("default", "vtl"), default="default",
                    help="비강 기하 (voice_td.NOSE, §52.490): default (예전 16 칸) 또는 vtl (Dang & Honda 19 칸 + 부비동 넷, 여성 배율 길이 ×0.9 · 단면 ×0.7)")
    ap.add_argument("--trachea", action="store_true",
                    help="기관 (성문 아래 관, tube_td.set_trachea, §52.491): 폐압을 폐 쪽 끝에, 성문은 기관 윗칸 압력으로 — 여성 Sg1 680 · Sg2 1675 Hz")
    ap.add_argument("--throat", action="store_true",
                    help="목 조임 (voice_td.THROAT, §52.490): 후두개관 좁힘 면적을 화자 상수로 적합 (가성대 내전은 --larynx 의 ff_add)")
    ap.add_argument("--ga-ref", default=None, metavar="판",
                    help="성문 역산 항의 기준 (판_ga.npy — 운동학판의 성문 면적 48 kHz, --save-ga 로 만든다; fit.GA_REF, §52.492)")
    ap.add_argument("--ga-w", type=float, default=None, metavar="W", help="성문 역산 항의 무게 (fit.GA_W)")
    ap.add_argument("--save-ga", action="store_true", help="끝에 성문 면적(48 kHz)을 out_ga.npy 로 저장")
    ap.add_argument("--speaker-from", default=None, metavar="판",
                    help="다른 판(판_track.npz)의 화자 상수(성대 정적 변수·가성대·후두개관·틈 등, speaker 범위)를 읽어 **고정**한다 (§52.491) — "
                         "구간마다 화자 상수가 제각각 보상 손잡이가 되지 않게, 발화 전체에서 한 번 적합한 값을 쓴다")
    ap.add_argument("--phoneme-init", default=None, metavar="정렬.json",
                    help="곡선 조음(--artic cos)의 협착 초기값을 MFA 정렬로 (korean.phoneme_init, §52.491): 폐쇄·마찰·파찰·비음을 제자리에, 치찰 표지도 넓힌다")
    ap.add_argument("--art-vowel", type=float, default=None, metavar="A",
                    help="모음 열림 조건 (fit.ART_VOWEL_A, §52.491, --art-cons 와 함께): 모음 구간 가운데에서 관 최소 면적 ≥ A cm² (0.25)")
    ap.add_argument("--art-velum", action="store_true",
                    help="연구개도 음소 목표로 (fit.ART_VELUM, §52.489): 비음 열림, 다른 자음 닫힘, 목표 근사 τ 20 ms")
    ap.add_argument("--art-cons", type=float, default=None, metavar="W",
                    help="자음 목표의 협착 명세 무게 (fit.ART_CONS_W, §52.487, --phoneme-targets 와 함께): 폐쇄음·비음은 폐쇄 몫에서 구강 최소 면적 "
                         "≤ 0.005 cm², 마찰음은 0.03–0.25 cm²")
    ap.add_argument("--hold-silence", action="store_true",
                    help="--init 출발점에서 말 앞뒤 쉼의 성문 제어(갑상피열근·Rd·긴장)를 발성 경계 값으로 채운다 (§52.510)")
    ap.add_argument("--tube-refine", type=int, default=1, metavar="R",
                    help="관 해상도 배율 (voice_td.TUBE_REFINE, §52.508): 칸 28·R · 내부 표본률 96·R kHz — 28 칸은 12–16 kHz 를 깎는다 (56 칸에서 수렴)")
    ap.add_argument("--art-place", action="store_true",
                    help="조음 자리 (fit.ART_PLACE, §52.507, --art-cons 와 함께): 폐쇄·마찰을 MRI 자세에서 잰 자리 창 안에서 재고, 창 밖은 열려 있게")
    ap.add_argument("--art-larynx", action="store_true",
                    help="성문 내전도 음소 목표로 (fit.ART_LARYNX, §52.507, --phoneme-targets 와 함께) — 틀별 적합에서 뺀다")
    ap.add_argument("--art-init", choices=("track", "mri"), default=None,
                    help="음소 목표 초기값 (--phoneme-targets): track = 모두 제어열의 구간 중앙값, mri = 자음은 W02 MRI 자세. "
                         "기본은 --init 이 있으면 track")
    ap.add_argument("--tube-ac", action="store_true",
                    help="조음 모형(--artic w02·maeda)에서 구강압·치찰 사전이 관의 실제 구강 최소 면적을 보게 한다 (voice.TUBE_AC, §52.487)")
    ap.add_argument("--vf-anat", type=float, default=None, metavar="W",
                    help="성대 정적 배율의 해부 범위 사전 무게 (fit.VF_ANAT_W, §52.494) — --vf-static-fit 과 함께")
    ap.add_argument("--vf-body", action="store_true",
                    help="몸체–덮개 성대 (voice_td.VF_BODY, §52.500): 덮개 두 질량이 몸체 질량에 붙는 세 질량 모형, 몸체 강성은 제어 vf_ta (갑상피열근 대리)")
    ap.add_argument("--hf-env", type=float, default=None, metavar="W",
                    help="고역 포락 손실 (fit.HFENV_W, §52.283·52.505): --hf-env-lo 위 멜 대역만 골라 그 대역들 안에서 dB 오차를 문다")
    ap.add_argument("--hf-env-lo", type=float, default=None, metavar="HZ", help="--hf-env 가 보는 아래 끝 [Hz] (기본 8000)")
    ap.add_argument("--voicing", type=float, default=None, metavar="W",
                    help="떨림 맞춤 항 (fit.VOICING_W, §52.505): 5 ms 틀 저역 주기성을 원본과 맞춰 유성/무성 분리를 문다")
    ap.add_argument("--vf-level-lock", type=int, default=0, metavar="N",
                    help="성대 모드의 음량 맞춤 되풀이 수 (fit.VF_LEVEL_LOCK, §52.504) — 틀 음량 모양을 비적합 폐압 배율로")
    ap.add_argument("--vf-ns", action="store_true",
                    help="N 줄 보–막 성대 (voice_td.VF_NS, 성대 모드 3, §52.504): Serry et al. 2026 보–막 모형을 첫 정재파로 투영, 상하 N 줄 덮개 + 몸체. "
                         "목표 f0 → 늘어남 ε, 제어 vf_ta = 갑상피열근 활성, 조직 법칙에서 층 응력·기하·질량·감쇠")
    ap.add_argument("--w02-anat", action="store_true",
                    help="W02 조음의 해부 적응 (voice_td.W02_ANAT, §52.501): VTL AnatomyParams 13 개(후두 길이·폭, 인두 길이 …)를 화자 상수로 적합")
    ap.add_argument("--w02-age", action="store_true",
                    help="나이 축 해부 (voice_td.W02_AGE, §52.523, --w02-anat 을 함께 켠다): W02 해부를 VTL 여성 성장식(Goldstein 1980)의 나이 비율로 옮기고 "
                         "개인차는 길이 ±15 %%·각 ±6° 로 묶는다 — 나이 12–25 세 (성인 여성 성도 길이 범위의 아래 끝이 12–13 세)")
    ap.add_argument("--codec-aac", default="", metavar="BITRATE",
                    help="실제 AAC 코덱을 적합 고리에 (fit.CODEC_AAC, §52.523) — 원본이 거친 방송 코덱. 손실은 코덱을 거친 소리로, 기울기는 직통. "
                         "결과 _fit.wav 는 코덱 없이, _fit_codec.wav 는 코덱을 거쳐 쓴다. 예: 64k")
    ap.add_argument("--tie-noise", action="store_true",
                    help="성문·협착 난류의 세기 계수를 하나로 (voice_td.TIE_NOISE, §52.524) — 같은 Stevens 꼴의 제트 난류")
    ap.add_argument("--vf-neural", action="store_true",
                    help="신경 작용 (voice_td.VF_NEURAL, §52.523): 후두근(CT·TA) 긴장의 운동 단위 요동·생리적 떨림 (Titze 1991) 을 보–막 성대에")
    ap.add_argument("--vf-body-drive", action="store_true",
                    help="몸체 구동 점막 성대 (voice_td.VF_BODY_DRIVE, §52.523, --vf-ns 위에서): 몸체(갑상피열근+인대)를 원본 닫힘에 묶인 펄스 위상의 "
                         "구동점에 매고, 덮개(점막 막 줄)는 제 동역학으로 — 닫힘의 모양은 점막이, 시각은 몸체가 정한다")
    ap.add_argument("--vf-pll", action="store_true",
                    help="성대 위상 고정 고리 (fit.VF_PLL, §52.493): 풀이기 안에서 성문 닫힘을 원본 들뜸 시각표에 맞춘다 (주기 시각 배율 대신)")
    ap.add_argument("--vf-cycle-lock", type=int, default=0, metavar="N",
                    help="성대 모드의 주기 시각 맞춤 반복 수 (fit.VF_CYCLE_LOCK, §52.491): 원본 주기 시각에 성대 닫힘을 긴장 제어로 맞춘다")
    ap.add_argument("--vf-no-psub-rule", action="store_true",
                    help="성대 모드에서 폐압 초기값 규칙(voice_td.vf_psub_rule)을 끄고 분석값을 쓴다 (원인 가르기)")
    ap.add_argument("--vf-scale-init", default=None, metavar="len,thick,mk,damp",
                    help="성대 정적 변수 배율의 초기값 (voice_td.VF_SCALE_INIT; 기본 0.65,1,0.65,1 — 1,1,1,1 이면 VTL 원래 값)")
    ap.add_argument("--vf-static-fit", action="store_true",
                    help="성대 정적 변수의 화자 배율(길이·두께·질량강성·감쇠)을 화자 상수로 적합 (fit.VF_STATIC_FIT, §52.489)")
    ap.add_argument("--vf-f0-map", choices=("vtl", "table"), default="vtl",
                    help="성대 모드의 f0 → 긴장 대응 (voice_td.VF_F0_MAP, §52.488): vtl 직선 또는 이 모형이 실제로 내는 f0(Q) 표")
    ap.add_argument("--glottis", choices=("kin", "vf"), default="kin",
                    help="성문 모형 (voice_td.GLOTTIS, MEASUREMENTS §52.488, --td 와 함께). kin: 펄스 위상으로 정한 운동학 변위 + 접촉. "
                         "vf: 자기 진동 성대 (Birkholz 삼각 성문 두 질량) — 폐압·내전·긴장으로 스스로 떤다. 펄스 잠금·위상 오프셋을 쓰지 않고, "
                         "합성 f0 를 재어 긴장 배율을 고친다 (fit.vf_pitch_lock)")
    ap.add_argument("--vf-ns-lock", action="store_true",
                    help="모드 3 성대 배율·뒤쪽 틈·좌우 비대칭을 고정한다. 보정표의 값과 같은지는 별도로 검사하며 값을 자동 복사하지 않는다")
    ap.add_argument("--velum-bounds", action="store_true",
                    help="입 장애음 구간에서 연인두 통로를 0.005 cm² 아래로 단단히 (voice_td.VELUM_BOUNDS, §52.530, --phoneme-targets 와 함께)")
    ap.add_argument("--vf-shear-max", type=float, default=None, metavar="X",
                    help="모드 3 덮개 상하 전단 배율 (td_log_ns_shear, 기준 MU_COVER 500 Pa)의 모형 상한. "
                         "m23에서 2.06 배는 17–24°/mm, 0.5 배는 30°/mm; 개 후두 위상차를 인간 강성의 정상범위로 직접 환산할 수 없다")
    ap.add_argument("--vf-body-free", type=float, default=None, metavar="A",
                    help="몸체 구동의 매는 강성을 구동 폭 a 에 따라 a²/(a² + A²) 로 (voice_td.VF_BODY_DRIVE_FREE, §52.533) — 구동이 약할 때 점막 성대가 제 물리로 떤다")
    ap.add_argument("--vf-lr", action="store_true",
                    help="좌우 성대 따로 (beam_membrane.LR_FOLDS, §52.533) — 긴장 · 질량 비대칭을 화자 상수로 (상한 5 %%, 긴장 출발 2 %%)")
    ap.add_argument("--vf-rest-modal", type=float, default=None, metavar="UM",
                    help="모달 내전 (voice_td.VF_ADD_MODAL) 에서의 성대돌기 위 날 쉼 반틈새 [μm] (voice_td.VF_R_MODAL, 기본 0; Zhang 2009의 90 μm는 모형 기하값이지 여성 모집단 평균이 아니다)")
    ap.add_argument("--ff-bounds", action="store_true",
                    help="공명음 구간의 가성대 틈 모형 하한 2.0 mm (voice_td.FF_BOUNDS, --phoneme-targets 와 함께). "
                         "간격에서 면적으로의 변환도 근사이며 인체 정상범위의 보편적 하한은 아니다")
    ap.add_argument("--vf-closure-edge", type=float, default=None, metavar="UM",
                    help="모드 3 성대의 닫힘 이음 폭 [μm] (beam_membrane.CLOSURE_EDGE_CM, §52.530; 기본 50)")
    ap.add_argument("--no-vf-lock", action="store_true",
                    help="바깥 음높이 고리를 끄고 출발 판의 긴장 배율 vf_qcorr 을 버린다 (fit.VF_QCORR_RESET, §52.530)")
    ap.add_argument("--pulse-regular", action="store_true",
                    help="구동 박자 정규화 (fit.PULSE_REGULAR, §52.530): 주기 길이는 원본의 Praat 주기점, 정렬은 지금 펄스 위상과의 느린 차이")
    ap.add_argument("--physio-dyn", action="store_true",
                    help="조음 · 후두 자세의 생리 동역학 (§52.530): 가성대 내전 σ 10 ms + 20 /s, 연구개 σ 10 ms, 조음 평활 σ = 1.6 τ (목표 근사 모형의 대역)")
    ap.add_argument("--vf-ns-hard", action="store_true",
                    help="모형의 화자 상수 범위를 단단히: 강성 [0.8, 6], 감쇠비 [0.05, 0.3], 길이 [0.85, 1.15], 두께 [0.7, 1.3]. 독립적인 인체 정상범위 측정값은 아니다")
    ap.add_argument("--pth-w", type=float, default=None, metavar="W",
                    help="발성 문턱 항 무게 (fit.PTH_W): 원본 유성 틀의 폐압 부족을 벌점으로 (PTH_MARGIN × 모형 문턱). 무성 틀은 이 항에서 제외")
    ap.add_argument("--pth-map", default=None, metavar="npz", help="발성 문턱 지도 (fit.PTH_MAP, out/_tmp/vf/pth_sweep.py 결과)")
    ap.add_argument("--vf-ns-calib", default=None, metavar="npz",
                    help="모드 3 성대의 화자 보정 (scripts/vf_ns_calib.py 결과, §52.530): f0(ε) 표를 바꾸고, --pth-map 이 없으면 같은 파일의 문턱 지도를 쓴다")
    ap.add_argument("--vf-ns-calib-approx", action="store_true",
                    help="보정표의 조건 불일치·상수 적합을 경고 후 허용한다 (옛 판 재현용 근사). 생리 검증으로 해석하지 않는다")
    ap.add_argument("--psub-max", type=float, default=None, metavar="P",
                    help="말소리 폐압 상한 [cmH2O] (fit.PSUB_MAX, §52.530, 단단히): 제어 범위 상한을 바꾸고 출발 판 폐압을 눌러 넣는다")
    ap.add_argument("--pitch-w", type=float, default=None, metavar="W",
                    help="음높이 항 무게 (fit.PITCH_W, §52.529): 원본·합성의 저역 자기상관 주기 (목표 0.8–1.25 배 창의 부드러운 최대) 의 log 차")
    ap.add_argument("--head-fir", default=None, metavar="npz",
                    help="머리·몸통·마이크 거리 FIR (fit.HEAD_FIR, §52.527): 실제 머리 스캔의 상반성 전달 G/(jω) 를 출력에 (profiles/head/P0156_20cm.npz)")
    ap.add_argument("--lip-bounds", action="store_true",
                    help="음소별 입술 범위를 관에 단단히 (voice_td.W02_LIP_BOUNDS, §52.527, --phoneme-targets 와 함께): 평순·원순 모음 내밂, 모음·비양순 자음의 입술 열림 하한")
    ap.add_argument("--art-free", action="store_true",
                    help="음소 구간은 협착·자리·성문 명세에만 쓰고 조음기는 틀별로 푼다 (fit.ART_TARGETS 거짓, §52.522)")
    ap.add_argument("--phoneme-targets", default=None, metavar="정렬.json",
                    help="음소 목표 조음 (fit.ART_SEGMENTS, §52.487, --artic w02 와 함께): 정렬 JSON {segments: [{label, t0, t1, shape}]} (발화 기준 초). "
                         "조음기는 틀마다 적합하지 않고 음소마다 목표 하나 + 경계 시각 (± 30 ms), 궤적은 목표 근사 동역학")
    ap.add_argument("--trans-leak", action="store_true",
                    help="튐 벌점을 옛 softplus 경첩으로 (fit.TRANS_LEAK, §52.463~471 판 되살리기·비교용). 기본은 문턱 아래 기울기 0 (§52.472)")
    a = ap.parse_args()
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
    if a.hnr_band:
        bands = []
        for it in a.hnr_band:
            lo, _, hi = it.partition("-")
            bands.append((float(lo), float(hi)))
        _fit.HNR_BANDS = tuple(bands)
        # 찍히지 않으면 안 든 것이다 — 조용한 무효가 네 번 있었다 (§52.178).
        print("  HNR 덧대역: " + ", ".join(f"{lo:.0f}-{hi:.0f} Hz" for lo, hi in bands)
              + f" (기본 {_fit.HNR_BP_HZ[0]:.0f}-{_fit.HNR_BP_HZ[1]:.0f} Hz 에 더함)", flush=True)
    if getattr(a, "tie_noise", False):
        from formant_ml.engine import voice_td as _vtdn
        _vtdn.TIE_NOISE = True
        print("  난류 세기: 성문 = 협착 (같은 Stevens 꼴, §52.524)", flush=True)
    if a.codec_aac:
        _fit.CODEC_AAC = a.codec_aac
        print(f"  녹음 코덱: AAC {a.codec_aac} 를 적합 고리에 (직통 기울기, 결과 파일은 코덱 없이, §52.523)", flush=True)
    if a.harm_fmax is not None:
        _fit.HARM_FMAX = float(a.harm_fmax)
    if a.tube:
        from formant_ml.engine import tract as _trt
        _trt.TUBE = True
    if a.td:
        from formant_ml.engine import voice as _vtd
        _vtd.TD_TUBE = True
        # 시간 영역 경로의 적합 설정 (§52.479–481) — 펄스 오프셋은 시간, 위상 원점은 닫힘, 넓은 다듬기, 치찰 틀 내전 낮춤,
        # 접촉 부드러움 적합, 조음(art1–6) 20 ms. 옛 Klatt 경로는 건드리지 않는다.
        from formant_ml.engine import voice_td as _vtdp
        if a.tube_refine > 1:
            _vtdp.set_tube_refine(a.tube_refine)          # 관 해상도 배율 (§52.508) — 엔진을 만들기 전에
            print(f"  관 해상도: {_vtdp.td.N_SECT} 칸 · {_vtdp.td.FS_SIM / 1000:g} kHz (배율 {a.tube_refine})", flush=True)
        _fit.PHI0_AS_TIME = True
        _vtdp.PHASE_AT_CLOSURE = True
        _fit.REFINE_MAX_MS = 2.0
        _fit.SIB_ABDUCT = 0.15
        _vtdp.EPS_FIT = True
        _fit.PARAM_TAU_PHYS.update({f"art{k}": 20.0 for k in range(1, 7)})
        from formant_ml.engine import analyze as _anl
        _anl.BURST_BAND2 = (5000.0, 14000.0)        # 치조·파찰 파열 검출 (§52.481)
        if a.artic == "maeda":
            # 조음기 시상수 [ms] — 목표 근사 모형(Birkholz, Kröger & Neuschaefer-Rube 2011: 성문 위 조음기 ~12 ms)을 따르되, 무거운 턱·후두·
            # 입술 돌출은 더 느리게. 틀마다 자유로운 곡선이 아니라 질량 있는 기관의 움직임이다.
            _vtdp.ARTIC = "maeda"
            _fit.PARAM_TAU_PHYS.update({"jaw": 25.0, "tongue_pos": 20.0, "tongue_shape": 20.0, "tongue_tip": 12.0,
                                        "lip_ht": 12.0, "lip_pr": 25.0, "larynx": 40.0})
            print("  조음: Maeda 조음 모형 (조음기 일곱, 시상수 턱 25·혀 20·혀끝 12·입술 12/25·후두 40 ms)", flush=True)
        if a.artic == "w02":
            # 시상수: 목표 근사 모형 (Birkholz et al. 2011, 성문 위 조음기 ~12 ms) — 턱·설골(후두)은 더 무겁다.
            _vtdp.ARTIC = "w02"
            if getattr(a, "w02_anat", False):
                _vtdp.W02_ANAT = True
                print("  조음 해부 적응: VTL AnatomyParams 13 개를 화자 상수로 (W02 ±20 % 안, §52.501)", flush=True)
            if getattr(a, "w02_age", False):
                _vtdp.W02_ANAT = True
                _vtdp.W02_AGE = True
                print(f"  나이 축 해부: W02 × 여성 성장식 비율, 나이 {_vtdp.W02_AGE_LO:g}–{_vtdp.W02_AGE_HI:g} 세 (기준 {_vtdp.W02_AGE_REF:g}), "
                      f"개인차 ±{100 * _vtdp.W02_DEV:g} %·±{_vtdp.W02_DEV_DEG:g}° (§52.523)", flush=True)
            _tau = {"HX": 40.0, "HY": 40.0, "JX": 25.0, "JA": 25.0, "LP": 20.0, "LD": 12.0, "TCX": 15.0, "TCY": 15.0, "TTX": 10.0,
                    "TTY": 10.0, "TBX": 12.0, "TBY": 12.0, "TRX": 20.0, "TRY": 20.0, "TS1": 15.0, "TS2": 15.0, "TS3": 15.0}
            _fit.PARAM_TAU_PHYS.update({f"vtl_{k}": v for k, v in _tau.items()})
            print("  조음: W02 (VocalTractLab 여성, MRI 3 차원 기하 → 대리 신경망), 해부학 변수 17 개, 시상수 혀끝 10·혀 12–20·턱 25·설골 40 ms",
                  flush=True)
            # 성문 뒤쪽 틈 (§52.486): 옛 엔진 분석에서 온 내전 사전 중심(0.44 → 틈 0.137 cm², 직류 ~500 cm³/s)은 정상 발성의 평균 흐름
            # (200–250 mL/s 이하 — 넘으면 불완전 폐쇄)을 크게 넘고, MRI 여성 성도에서 모든 포먼트를 무너뜨려 적합기가 단면을 부풀려 메웠다.
            # 중심을 틈 0.035 cm² (내전 0.754, 8 cmH2O 에서 직류 ~100–150 cm³/s), 눈금 하나가 틈 0.025–0.056 cm² 에 들게 옮긴다.
            _fit.ADD_PRIOR_MU = 1.12
            _fit.ADD_PRIOR_S = 0.55
    if a.larynx:
        from formant_ml.engine import voice_td as _vtdl
        _vtdl.set_larynx(True)
        print("  후두 구조 (W2a 에서 추론): 좌우 이상와 1.024 cm (6.4 kHz) + 후두실 헬름홀츠 (11.5 kHz) 곁관, 가성대 좁힘 (쉼 틈 적합, 내전 ff_add)",
              flush=True)
    if a.nose != "default":
        from formant_ml.engine import voice_td as _vtdn
        _vtdn.set_nose(a.nose)
        print(f"  비강: {a.nose} (Dang & Honda 19 칸 + 부비동, 길이 ×{_vtdn.NOSE_SCALE[0]} · 단면 ×{_vtdn.NOSE_SCALE[1]})", flush=True)
    if a.trachea:
        from formant_ml.engine import tube_td as _tdt
        _tdt.set_trachea(True)
        print("  기관: 19 cm (기관 1.8 cm² + 넓어지는 기관지), 성문 아래 공명 680 · 1675 Hz", flush=True)
    if a.throat:
        from formant_ml.engine import voice_td as _vtdt
        _vtdt.THROAT = True
        _vtdt.EPI_ON = True
        _vtdt.EPI_INIT_AREA = _vtdt.THROAT_EPI_INIT
        print(f"  목 조임: 후두개관 {_vtdt.EPI_LEN_CM:g} cm 를 화자 상수 면적으로 좁힌다 (초기 {_vtdt.THROAT_EPI_INIT} cm², 적합)", flush=True)
    if a.ga_ref:
        _gref = np.load(a.ga_ref + "_ga.npy")
        _fit.GA_REF = np.asarray(_gref, np.float64).reshape(-1)
        _fit.GA_W = float(a.ga_w) if a.ga_w is not None else 1.0
        print(f"  성문 역산 항: {a.ga_ref}_ga.npy ({_fit.GA_REF.size} 표본), 무게 {_fit.GA_W:g}", flush=True)
    if a.art_cons is not None:
        _fit.ART_CONS_W = float(a.art_cons)
    if a.pitch_w is not None:
        _fit.PITCH_W = float(a.pitch_w)
    if a.pth_w is not None:
        _fit.PTH_W = float(a.pth_w)
    if a.pth_map:
        _fit.PTH_MAP = a.pth_map
    if a.vf_ns_calib:
        from formant_ml.engine import voice_td as _vtdc
        from formant_ml.engine.vf_calibration import validate_pitch_table
        with np.load(a.vf_ns_calib, allow_pickle=False) as _cal:
            validate_pitch_table(_cal["eps_tab"], _cal["f0_tab"])
            _vtdc.VF_NS_EPS_TAB = tuple(float(v) for v in _cal["eps_tab"])
            _vtdc.VF_NS_F0_TAB = tuple(float(v) for v in _cal["f0_tab"])
            if not a.pth_map and "pth" in _cal.files:
                _fit.PTH_MAP = a.vf_ns_calib
        print(f"  모드 3 화자 보정 {a.vf_ns_calib}: f0(ε) 표 " + " ".join(f"{e:+.2f}:{f:.0f}" for e, f in zip(_vtdc.VF_NS_EPS_TAB, _vtdc.VF_NS_F0_TAB)),
              flush=True)
    if a.psub_max is not None:
        _fit.PSUB_MAX = float(a.psub_max)
    if a.pulse_regular:
        _fit.PULSE_REGULAR = True
    if a.velum_bounds:
        from formant_ml.engine import voice_td as _vtdv
        _vtdv.VELUM_BOUNDS = True
        print(f"  입 장애음 연인두 상한 {_vtdv.VELUM_OBS_MAX:g} cm² (앞뒤 {_vtdv.VELUM_RAMP_MS:g} ms 비스듬히, §52.530)", flush=True)
    if a.vf_shear_max is not None:
        from formant_ml.engine import voice_td as _vtdsh
        _vtdsh.VF_NS_HARD = dict(_vtdsh.VF_NS_HARD or {}, shear=(0.1, float(a.vf_shear_max)))
        print(f"  덮개 상하 전단 상한 ×{a.vf_shear_max:g} (z 위상차, §52.533): {_vtdsh.VF_NS_HARD}", flush=True)
    if a.vf_body_free is not None:
        from formant_ml.engine import voice_td as _vtdbf
        _vtdbf.VF_BODY_DRIVE_FREE = float(a.vf_body_free)
        print(f"  몸체 구동: 구동 폭 {a.vf_body_free:g} 아래에서는 풀어 둔다 (§52.533)", flush=True)
    if a.vf_lr:
        import formant_ml.physics.beam_membrane as _bmlr
        from formant_ml.engine import voice_td as _vtdlr
        _bmlr.LR_FOLDS = True
        print(f"  좌우 성대 따로: 긴장 비대칭 (0, {_vtdlr.VF_LR_DQ_MAX:g}) 출발 {_vtdlr.VF_LR_DQ_INIT:g}, 질량 ±{_vtdlr.VF_LR_DM_MAX:g} (§52.533)", flush=True)
    if a.vf_rest_modal is not None:
        from formant_ml.engine import voice_td as _vtdr
        _vtdr.VF_R_MODAL = float(a.vf_rest_modal) * 1e-4
        print(f"  모달 쉼 반틈새 +{a.vf_rest_modal:g} μm (내전 {_vtdr.VF_ADD_MODAL:g} 에서 ×{_vtdr._vf_r_modal_w(_vtdr.VF_ADD_MODAL):.2f}, 압착 끝 0, §52.532)", flush=True)
    if a.ff_bounds:
        from formant_ml.engine import voice_td as _vtdf
        _vtdf.FF_BOUNDS = True
        print(f"  공명음 가성대 틈 하한 {_vtdf.FF_GAP_MIN_MM:g} mm (면적 π/4·틈·성대 길이, 구간 안쪽 {_vtdf.FF_RAMP_MS:g} ms 비스듬히, §52.532)", flush=True)
    if a.vf_closure_edge is not None:
        import formant_ml.physics.beam_membrane as _bme
        _bme.CLOSURE_EDGE_CM = float(a.vf_closure_edge) * 1e-4
        print(f"  성대 닫힘 이음 폭 {a.vf_closure_edge:g} μm (§52.530)", flush=True)
    if a.no_vf_lock:
        _fit.VF_QCORR_RESET = True
        _fit.VF_LOCK_ITERS = 0
    if a.physio_dyn:
        from formant_ml.engine import voice_td as _vtdd
        _vtdd.FF_SMOOTH_MS = 10.0
        _vtdd.VELUM_SMOOTH_MS = 10.0
        _vtdd.W02_SMOOTH = _vtdd.W02_SMOOTH_TAM
        print(f"  생리 동역학: 가성대 σ {_vtdd.FF_SMOOTH_MS:g} ms · {_vtdd.FF_VMAX:g} /s, 연구개 σ {_vtdd.VELUM_SMOOTH_MS:g} ms, "
              f"조음 σ = {_vtdd.W02_SMOOTH:g} τ (§52.530)", flush=True)
    if a.vf_ns_hard:
        from formant_ml.engine import voice_td as _vtdh
        _vtdh.VF_NS_HARD = dict(_vtdh.VF_NS_HARD_DEFAULT, **(_vtdh.VF_NS_HARD or {}))   # 앞에서 건 전단 상한 (--vf-shear-max) 은 남긴다
        print(f"  성대 화자 상수 생리 범위 (단단히): {_vtdh.VF_NS_HARD}", flush=True)
    if a.head_fir:
        _fit.HEAD_FIR = a.head_fir
        print(f"  머리·몸통·거리 FIR: {a.head_fir} (실제 머리 스캔의 상반성 전달, 20 cm)", flush=True)
    if a.art_vowel is not None:
        _fit.ART_VOWEL_A = float(a.art_vowel)
    if a.tube_ac:
        from formant_ml.engine import voice as _vce
        _vce.TUBE_AC = True
    if a.psub_prior is not None:
        _fit.PSUB_PRIOR_W = float(a.psub_prior)
    if a.td and a.glottis == "vf":
        from formant_ml.engine import voice_td as _vtdg
        _vtdg.GLOTTIS = "vf"
        _vtdg.VF_F0_MAP = a.vf_f0_map
        _fit.VF_STATIC_FIT = bool(a.vf_static_fit)
        _fit.VF_CYCLE_LOCK = int(a.vf_cycle_lock)
        if getattr(a, "vf_body", False):
            _vtdg.VF_BODY = True
        if getattr(a, "vf_ns", False):
            _vtdg.VF_NS = True
            if getattr(a, "vf_neural", False):
                _vtdg.VF_NEURAL = True
                print(f"  신경 작용: 후두근 운동 단위 {_vtdg.VF_NEURAL_NMU} 개 · 떨림 {_vtdg.VF_NEURAL_TREMOR[0]:g} Hz (§52.523)", flush=True)
            if getattr(a, "vf_body_drive", False):
                _vtdg.VF_BODY_DRIVE = True
                import formant_ml.physics.beam_membrane as _bmd
                _bmd.ADJ_TAU_MS = 0.0                    # 구동 덮개는 감쇠하는 강제 응답 — 수반을 자르지 않는다 (정확한 기울기, §52.523)
                print(f"  몸체 구동 점막 성대: 몸체를 펄스 위상 구동점에 (매는 강성 ×{_vtdg.VF_BODY_DRIVE_K:g}, 진폭 상한 {10 * _vtdg.VF_BODY_DRIVE_XMAX:g} mm), "
                      "덮개는 동역학 (§52.523)", flush=True)
        if getattr(a, "vf_level_lock", 0):
            _fit.VF_LEVEL_LOCK = int(a.vf_level_lock)
        if getattr(a, "vf_anat", None) is not None:
            _fit.VF_ANAT_W = float(a.vf_anat)
        if getattr(a, "vf_pll", False):
            _fit.VF_PLL = True
            _fit.VF_CYCLE_LOCK = max(int(_fit.VF_CYCLE_LOCK), 3)
        if a.vf_scale_init:
            _v4 = [float(v) for v in a.vf_scale_init.split(",")]
            _vtdg.VF_SCALE_INIT = dict(zip(("len", "thick", "mk", "damp"), _v4))
        _fit.PHI0_FIXED = 0.0                       # 펄스 위상은 쓰이지 않는다 — 오프셋 훑기가 헛돈다
        # 0.44는 옛 엔진 합성음으로 학습한 역모형의 추정 좌표다 (§52.426), 인체 쉼 틈의 직접 측정값은 아니다.
        _fit.ADD_PRIOR_MU = math.log(_vtdg.VF_ADD_MODAL / (1.0 - _vtdg.VF_ADD_MODAL))
        _fit.ADD_PRIOR_S = 0.37
        # 튐 벌점의 비교 창 — 원본 펄스 봉우리를 늘 품게 폭(2w)을 이 화자의 가장 긴 주기(f0 ≈ 180 Hz, 5.5 ms) 넘게. ±1.5 ms 는 펄스 잠금
        # (합성 펄스 = 원본 펄스 시각)을 전제한 자라, 위상이 묶이지 않은 성대의 닫힘 펄스가 원본 주기 사이 골에 떨어져 주기마다 벌점을 받았고
        # 그 항의 기울기가 지형을 망쳐 적합이 한 걸음도 못 나갔다 (§52.488).
        _fit.TRANS_MATCH_MS = 3.0
        print("  성문: 자기 진동 성대 (삼각 성문 두 질량, VTL W02 정적 변수) — 긴장·쉼 변위·수렴·폐압, 뒤쪽 틈은 화자 상수", flush=True)
    if getattr(a, "hf_env", None) is not None:
        _fit.HFENV_W = float(a.hf_env)
        if a.hf_env_lo is not None:
            _fit.HFENV_LO = float(a.hf_env_lo)
        print(f"  고역 포락 손실: {_fit.HFENV_LO:g} Hz 위, 세기 {_fit.HFENV_W:g}", flush=True)
    if getattr(a, "voicing", None) is not None:
        _fit.VOICING_W = float(a.voicing)
        print(f"  떨림 맞춤 항 (유성/무성): 세기 {_fit.VOICING_W:g}", flush=True)
    if a.trans_leak:
        _fit.TRANS_LEAK = True
        print("  튐 벌점 경첩: 옛 softplus (문턱 아래로 샌다 — 비교용)", flush=True)
    if a.hold_fric:
        _fit.HOLD_FRIC = True
    if a.mprior is not None:
        _fit.MPRIOR_W = float(a.mprior)
        print(f"  기관 운동 사전: 무게 {_fit.MPRIOR_W:g}, {_fit.MPRIOR_LAG_MS:g} ms 차분, t(ν={_fit.MPRIOR_NU:g}), 눈금 "
              + ", ".join(f"{k} {v:g}" for k, v in _fit.MPRIOR_SCALE.items()), flush=True)
    if a.add_prior is not None:
        _fit.ADD_PRIOR_W = float(a.add_prior)
        print(f"  내전 수준 사전: 무게 {_fit.ADD_PRIOR_W:g} (유성 틀, logit t)", flush=True)
    if a.late_reverb is not None:
        _fit.LATE_REVERB = True
        _fit.LR_G0_DB = float(a.late_reverb)
        if a.late_reverb_rt is not None:
            _fit.LR_RT0 = float(a.late_reverb_rt)
        _fit.LR_FIXED = bool(a.late_reverb_fixed)
        print(f"  방 잔향 꼬리: {'고정' if _fit.LR_FIXED else '초기'} RT60 {_fit.LR_RT0 * 1000:.0f} ms, 잔향/직접 {_fit.LR_G0_DB:+g} dB "
              f"({'녹음 상수 — 적합 안 함' if _fit.LR_FIXED else '둘 다 적합'}), {_fit.LR_ONSET_MS:g} ms 부터", flush=True)
    if a.asp_strouhal:
        from formant_ml.engine import noise as _nz
        _nz.ASP_STROUHAL = True
        print(f"  기식 셸프 꺾임 = 스트로할 St {_nz.ASP_ST:g}·v/d (표본마다)", flush=True)
    if a.stop_release_sharp:
        from formant_ml.engine import analyze as _an_rs
        _an_rs.STOP_RELEASE_SHARP = True
        print(f"  폐쇄 개방: 가장 가파른 1–16 kHz 상승 앞까지 닫고 {_an_rs.STOP_RELEASE_RAMP_MS:g} ms 경사로 연다", flush=True)
    if a.release_rate is not None:
        from formant_ml.engine import voice as _vc_rr
        _vc_rr.RELEASE_RATE = True
        _fit.RELEASE_PRIOR_W = float(a.release_rate)
        print(f"  파열 개방 속도: 파열별 적합, 사전 무게 {_fit.RELEASE_PRIOR_W:g} — log R 혼합 t(ν={_fit.RELEASE_NU:g}, 눈금 {_fit.RELEASE_S:g}) "
              f"중앙 100/50/25 cm²/s", flush=True)
    if a.aero_gate:
        from formant_ml.engine import voice as _vc_ag
        _vc_ag.ORAL_AERO_GATE = True
        print(f"  공기역학 유효 협착 = 직렬(a_c, 입술 {_vc_ag.AERO_OPEN_MAX:g} cm² × ((oral_open−0.02)/0.98)²) — 게이트가 닫히면 공기도 막힌다", flush=True)
    if a.voice_bar:
        _fit.VOICE_BAR = True
        if a.voice_bar_fixed is not None:
            _fit.VB_FIXED = True
            _fit.VB_LOG_G0 = float(a.voice_bar_fixed) * 0.11512925464970229   # ln(10)/20
            print(f"  보이스 바 세기 고정: {float(a.voice_bar_fixed):+g} dB (화자 상수 — 적합 안 함)", flush=True)
        print("  보이스 바: 벽 공진 200 Hz (폭 100), 닫힌 정도 (1−oral_open)(1−velum), 세기 적합", flush=True)
    if a.oral_ode is not None:
        from formant_ml.engine import glottis as _go
        _go.ORAL_ODE, _go.ORAL_CW_ML, _go.ORAL_LOAD = True, float(a.oral_ode), 1.0
        print(f"  구강압 상미분방정식: 벽 순응도 {_go.ORAL_CW_ML:g} ml/cmH2O, 공동 {_go.ORAL_V_CM3:g} ml, 연구개 틈 {_go.ORAL_A_VP:g} cm², "
              f"성문 부하 {_go.ORAL_LOAD:g}", flush=True)
    if a.transient_ramp is not None:
        _fit.TRANS_RAMP = int(a.transient_ramp)
        print(f"  튐 벌점 경사: 단계마다 첫 {_fit.TRANS_RAMP} 회 0 → 1", flush=True)
    if a.transient is not None:
        _fit.TRANS_W = float(a.transient)
        print(f"  튐 벌점 켬: 무게 {_fit.TRANS_W:g} (진행 줄의 '튐ms' = 자의 문턱을 넘은 시간)", flush=True)
    if a.hold_gap:
        _fit.HOLD_GAP = True
        print(f"  사각지대 붙잡기: 유성도 마찰도 아닌 구간의 공명을 분석 궤적 모양으로 "
              f"(램프 {_fit.HOLD_GAP_RAMP_MS:g} ms, 기준 평활 σ "
              f"max({_fit.HOLD_GAP_SMOOTH_MS:g} ms, {_fit.HOLD_GAP_SMOOTH_FRAC:g}×구간))",
              flush=True)
    if a.floor:
        _fit.FLOOR_NOISE = True
    if a.band_limit_lo is not None:
        _fit.BAND_LIMIT_LO_HZ = float(a.band_limit_lo)
    if _fit.BAND_LIMIT_LO_HZ > 0.0:
        print(f"  녹음 저역 차단: {_fit.BAND_LIMIT_LO_HZ:g} Hz ({_fit.BAND_LIMIT_LO_ORDER} 차, 영위상)", flush=True)
    if a.balance:
        _fit.BALANCE = True
    if a.param_tau:
        _fit.PARAM_TAU_MS.update(_fit.PARAM_TAU_PHYS)
        # 표에 없는 파라미터는 **무제한**이 된다 — `hz*_f` 가 사실상 백색으로 떠는 까닭이다.
        # `--tau-default` 를 주면 못 찾은 것들을 그 값으로 채운다 (fit.PARAM_TAU_PHYS_DEFAULT).
        if _fit.PARAM_TAU_PHYS_DEFAULT > 0.0:
            miss = [n for n in _fit.DEFAULT_PARAMS
                    if n not in _fit.PARAM_TAU_MS and n not in _fit.PARAM_TAU_PHYS_FAST]
            for n in miss:
                _fit.PARAM_TAU_MS[n] = _fit.PARAM_TAU_PHYS_DEFAULT
            print(f"  시상수 기본 {_fit.PARAM_TAU_PHYS_DEFAULT:.0f} ms 로 채운 파라미터 "
                  f"{len(miss)} 개: {', '.join(miss[:8])}"
                  + (" ..." if len(miss) > 8 else ""), flush=True)
    if a.noise_global:
        _fit.MOTION_SLOW.update(_fit.NOISE_GLOBAL)
    if a.front_free is not None:
        _fit.PRIOR_W["front_len"] = float(a.front_free)
        print(f"  앞공동 자리 사전 가중: {float(a.front_free):g} (기본 10)", flush=True)
    for item in (a.slow or ()):
        k, _, v = item.partition("=")
        _fit.MOTION_SLOW[k] = float(v)
    _org = {k: v for k, v in _fit.MOTION_SLOW.items() if k in ("adduction", "a_c", "voice_gain", "p_sub", "velum", "oral_open")}
    if _org:
        # **찍는다** — 기관 속도 제약이 켜졌는지 로그에서 보이게 (§52.422).
        pass
    if a.motion_grid is not None:
        _fit.MOTION_MIN_GRID_MS = float(a.motion_grid)
    if a.hnr is not None:
        _fit.HNR_W = float(a.hnr)
    # **`--bal` 은 맨 마지막이다.** 앞의 `--comb`·`--fastfit` 도 같은 칸을 쓰는데, 그 블록이
    # 뒤에 있어서 `--bal comb=3` 을 조용히 덮어쓰고 있었다 (실측: `--bal comb=3` 과 `comb=6` 이
    # 둘 다 균형 가중 0.821 로 같았다 — 판 둘을 헛돌렸다). 명시적인 덮어쓰기가 이겨야 한다.
    if a.tilt_cap is not None:
        from formant_ml.engine import glottis as _g
        _g.TILT_MAX_HZ = float(a.tilt_cap)
        print(f"  음원 기울기 상한: {_g.TILT_MAX_HZ:g} Hz" + (" (없음)" if _g.TILT_MAX_HZ <= 0 else ""), flush=True)
    for item in (a.prior or ()):       # --tilt-cap 블록 안에 갇혀 있던 것을 꺼냈다
        k, _, v = item.partition("=")
        if k not in _fit.PRIOR_W and k not in _fit.DEFAULT_PARAMS:
            raise SystemExit(f"--prior: 모르는 파라미터 {k!r}")
        _fit.PRIOR_W[k] = float(v)
        print(f"  사전 가중 {k} = {_fit.PRIOR_W[k]:g}", flush=True)   # 켜진 것을 로그로 확인한다

    # **읽기·정리** — `engine.pipeline.load_clean` (§52.470): 클릭 메우기(손·자동) → 잡음 제거. 표본 자리·길이 불변.
    from formant_ml.engine import pipeline as _pl
    _clean = _pl.load_clean(a.wav, profile=((SpeakerProfile.load(a.profile) if a.profile else DEFAULT_PROFILE)
                                            if (a.declick_at or a.declick_auto) else None),
                            declick_at=a.declick_at, declick_auto=a.declick_auto, no_denoise=a.no_denoise,
                            log=lambda m: print(m, flush=True))
    y, y_raw, sr = _clean.y, _clean.y_raw, _clean.sr
    if a.floor_ref:
        _fit.FLOOR_REF = np.asarray(y, dtype=np.float64)   # 목표와 같은 단계 — 잡음 제거 뒤, 자르기 전

    t1 = a.t1 if a.t1 is not None else len(y) / sr
    seg = y[int(a.t0 * sr):int(t1 * sr)]
    seg_raw = y_raw[int(a.t0 * sr):int(t1 * sr)]      # 잡음 제거 전 — 코덱 차단 분석용 (§52.206)
    prof = SpeakerProfile.load(a.profile) if a.profile else DEFAULT_PROFILE
    # **보정 제약을 찍는다** — 찍히지 않으면 켜졌는지 알 수 없다 (§52.188, §52.409)
    print(f"  주기별 배음 보정 제약: L2 {_fit.HCORR_L2:g} (배음 {_fit.HCORR_L2_KNEE} 위로 하나마다 +{_fit.HCORR_L2_SLOPE:g}), "
          f"주기 간 도약 {_fit.HCORR_TV:g}", flush=True)
    hop = max(1, int(round(a.frame_ms * sr / 1000.0)))
    if a.burst_back_pct is not None:
        # **`analyze` 앞에서** 걸어야 한다 — 파열 검출이 그 안에서 돈다. 뒤에 두었다가
        # 깃발이 통째로 먹지 않아 판 하나를 버렸다 (§52.255 와 같은 부류).
        from formant_ml.engine import analyze as _an_bp
        _an_bp.BURST_BACK_PCT = float(a.burst_back_pct)
        print(f"  파열 검출 기준 분위: {float(a.burst_back_pct):g} % (기본 100 = 최댓값)", flush=True)
    from formant_ml.engine import prepare as _prep
    _segment = _prep.Segment.cut(y, sr, a.t0, t1)
    if a.sib_prior is not None:
        # **분석 앞에서** — 관측은 분석 안에서 돈다
        from formant_ml.engine import analyze as _an_sb
        _an_sb.SIB_OBS = True
        _fit.SIB_PRIOR_W = float(a.sib_prior)
        print(f"  치찰 협착 사전: 무게 {_fit.SIB_PRIOR_W:g}", flush=True)
    if a.sib_prior_edge is not None:
        _fit.SIB_PRIOR_EDGE_MS = float(a.sib_prior_edge)
    if a.nasal_i_guard is not None:
        from formant_ml.engine import analyze as _an_ni
        _an_ni.NASAL_I_PROM_DB = float(a.nasal_i_guard)
    if a.creak:
        # **분석 앞에서** 켠다 — 분석 뒤에 켜면 먹지 않는다 (§52.397 과 같은 함정)
        from formant_ml.engine import analyze as _an_cr
        _an_cr.CREAK_PULSES = True
    # ---- 엔진·적합기 설정 (§52.469: 예전에는 분석·엔진 생성 뒤에 있었다 — 계산 전에 모두 건다) ----
    from formant_ml.engine import analyze as _an_cfg
    _an_cfg.FRONT_LEN_OBS = bool(a.front_obs)       # 표지 — 예전엔 `prepare` 가 관측 도중에 켰다
    _an_cfg.STOP_CLOSE = bool(a.stop_close)
    if a.fric_am is not None:
        from formant_ml.engine import noise as _nz
        _nz.FRIC_AM_DEPTH = float(a.fric_am)
    if a.voiced_soft is not None:
        from formant_ml.engine import glottis as _gl
        _gl.VOICED_SOFT = float(a.voiced_soft)
    if a.asp_am is not None:
        from formant_ml.engine import glottis as _gl
        _gl.ASP_AM_DEPTH = float(a.asp_am)
    if a.harm_prior_fmax is not None:
        _fit.HARM_PRIOR_FMAX = float(a.harm_prior_fmax)
    if a.band_limit is not None:
        _fit.BAND_LIMIT_HZ = float(a.band_limit)
        print(f"  녹음 대역 상한: {_fit.BAND_LIMIT_HZ:g} Hz (선형 위상 {_fit.BAND_LIMIT_TAPS} 탭)", flush=True)
    if a.extra_bw_floor is not None or a.hf_formant_bw_floor is not None:
        from formant_ml.engine import tract as _trz
        if a.extra_bw_floor is not None:
            _trz.EXTRA_BW_FLOOR = float(a.extra_bw_floor)
        if a.hf_formant_bw_floor is not None:
            _trz.HF_FORMANT_BW_FLOOR = float(a.hf_formant_bw_floor)
        print(f"  폭 하한 사다리 {_trz.EXTRA_BW_FLOOR:g} / "
              f"F5~F8 {_trz.HF_FORMANT_BW_FLOOR if _trz.HF_FORMANT_BW_FLOOR is not None else _trz.EXTRA_BW_FLOOR:g}", flush=True)
    if a.hf_fixed:
        from formant_ml.engine import tract as _tr3
        _tr3.HF_FIXED = True
    if a.grids:
        _fit.GRID_MS = tuple(float(x) for x in a.grids.split(","))
        assert 1 <= len(_fit.GRID_MS) <= len(_fit.STAGES), "--grids 는 1~4 개"
    if a.src_eq:
        from formant_ml.engine import glottis as _gl10
        _gl10.SRC_EQ = True
    if a.pulse_lock:
        _fit.PULSE_LOCK = True
    if a.pulse_refine:
        _fit.PULSE_REFINE = int(a.pulse_refine)
    if a.open_damp:
        from formant_ml.engine import tract as _tr6
        _tr6.OPEN_DAMP = True
    if a.noise_v2:
        # v2.1 (docs/NOISE_SOURCE_REVIEW.md §4): 기식은 유성 정도로 갈라 유성분만 MVF(적합) 위로,
        # 무성분은 전대역으로 정상 종속 가지에. 앞공동 Q 는 장애물에 묶는다. (out/L46 은 v2.0 이다.)
        from formant_ml.engine import noise as _nz2, tract as _tr2
        _tr2.NOISE_V2 = True
        _tr2.FRONT_Q_OBSTACLE = True
        _nz2.FRIC_V2 = True
        _nz2.ASP_SPLIT = True
    if a.hf_ladder == "legacy":
        from formant_ml.engine import tract as _tr14
        _tr14.EXTRA_BW_FLOOR, _tr14.EXTRA_BW_CEIL = 0.933, 2.499
        _tr14.HF_DF_LIM = 0.15
        # **명시한 깃발이 사다리 모드보다 이긴다** (MEASUREMENTS §52.255). 이 블록이 `--extra-bw-floor` 뒤에 있어 그 값을
        # 조용히 0.933 으로 되돌렸고, 그래서 하한을 바꾼 재렌더 둘이 비트 단위로 같았다.
        if a.extra_bw_floor is not None:
            _tr14.EXTRA_BW_FLOOR = float(a.extra_bw_floor)
    from formant_ml.engine import tract as _trl
    print(f"  고역 사다리 {a.hf_ladder}: 폭 하한 {_trl.EXTRA_BW_FLOOR:g} · 상한 {_trl.EXTRA_BW_CEIL:g} · 자리 ±{_trl.HF_DF_LIM:g} 로그",
          flush=True)
    # **조리법 고정** (§52.469) — 여기서부터 계산이다. 이 뒤에 바뀌는 설정 전역은 판 끝에 모두 드러낸다.
    from formant_ml.engine import recipe as _recipe
    if a.td and a.artic == "w02" and a.phoneme_targets:
        # 음소 목표 구간 (§52.487) — 모듈 전역이므로 조리법 고정 전에 쓴다 (§52.469)
        import json as _json
        _al = _json.load(open(a.phoneme_targets, encoding="utf-8"))["segments"]
        _t0 = float(a.t0 or 0.0); _t1 = float(a.t1) if a.t1 is not None else 1e9
        _fit.ART_SEGMENTS = [(max(sg["t0"], _t0) - _t0, min(sg["t1"], _t1) - _t0, sg.get("shape")) for sg in _al
                             if sg["t1"] > _t0 and sg["t0"] < _t1]
        _fit.ART_INIT_FROM_TRACK = bool(a.init) if a.art_init is None else a.art_init == "track"
        _fit.ART_PLACE = bool(a.art_place)
        _fit.ART_LARYNX = bool(a.art_larynx)
        _fit.ART_LABELS = [sg.get("label", "") for sg in _al if sg["t1"] > _t0 and sg["t0"] < _t1]
        _fit.ART_VELUM = bool(a.art_velum)
        _fit.ART_TARGETS = not a.art_free
        if a.lip_bounds:
            from formant_ml.engine import voice_td as _vtdl
            _vtdl.W02_LIP_BOUNDS = True
            print(f"  음소별 입술 범위: 관에 단단히 (평순 내밂 [−0.3, 0.3] · LD ≥ 0.2, 원순 [0.4, 1.0] · LD ≥ 0.1, 비양순 자음 LD ≥ 0.15; "
                  f"과제 {_vtdl.W02_TASK}: LD ≤ {_vtdl.SPEECH_LD_MAX}, 턱 ≥ {_vtdl.SPEECH_JA_MIN})", flush=True)
        if a.art_velum:
            _fit.PARAM_TAU_PHYS.setdefault("velum", 20.0)
    _recipe_snap = _recipe.snapshot()
    _recipe.save(_recipe_snap, a.out + "_recipe.json")
    track = _prep.observe(_segment, prof, hop, front_obs=bool(a.front_obs), log=lambda m: print(m, flush=True))
    if a.vowel_onset:
        # 모음으로 시작하는 발화의 첫머리 순간음은 구강 파열이 아니다 — `engine.pipeline.vowel_onset_filter` (§52.467)
        _pl.vowel_onset_filter(track, _segment, y, sr, _pl.transcript_for(a.wav), a.t0, log=lambda m: print(m, flush=True))
    print(f"구간 {a.t0:.3f}~{t1:.3f} s, {track.n_frames} 프레임 × {track.frame_ms} ms",
          flush=True)

    initial_constants = None
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
        if a.hold_silence:
            # **말 앞뒤 쉼의 성문 제어는 발성 경계의 값으로** (§52.510). 쉼에는 근거가 없어 앞 판에서 갑상피열근 활성이 0.95–0.99 로 떠 있었고,
            # 속도 한계(10/s) 아래 발성 값(~0.3)까지 60–80 ms 가 걸려 보–막 성대가 0.12 s 에야 떨기 시작했다 (원본 0.05 s). 발성 전 후두의
            # 준비 조정(prephonatory adjustment)이 이 자세를 미리 잡는다.
            _vo = np.asarray(track.voiced, bool)[:n_] if np.asarray(track.voiced).size else np.zeros(0, bool)
            if _vo.any():
                i0_, i1_ = int(np.argmax(_vo)), int(n_ - 1 - np.argmax(_vo[::-1]))
                _held = [nm_ for nm_ in ("vf_ta", "rd_offset", "tension", "f0_target") if nm_ in INDEX]
                for nm_ in _held:
                    j_ = INDEX[nm_]
                    track.values[:i0_, j_] = track.values[i0_, j_]
                    track.values[i1_ + 1:n_, j_] = track.values[i1_, j_]
                print(f"  쉼의 성문 제어를 발성 경계 값으로: 앞 {i0_} 틀 · 뒤 {n_ - 1 - i1_} 틀 ({', '.join(_held)})", flush=True)
                if "vf_ta" in INDEX:
                    # **떨림 시작은 분기점이라 적합이 못 넘는다** (§52.512): 성대가 떨지 않는 동안 갑상피열근에 대한 기울기가 0 이다. BGA 는 발성 첫
                    # 틀(48 ms)에서도 활성 0.9 라 폐압 5.5–5.8 cmH2O 에서 떨지 않았다 (틈 면적 고정, 유량 떨림 0) — 첫 유성 틀의 값으로 채워서는 소용이
                    # 없다. 유성 구간 중앙도 모자랐다 (BGA 중앙 0.51 — 0.53 에서도 안 떨었고 0.06 에서야 떨었다). 유성 첫 100 틀 안에서 가장 낮은
                    # 활성(앞 판에서 실제로 떨 수 있었던 값)의 틀까지를 그 값으로 채운다.
                    j_ = INDEX["vf_ta"]
                    w_ = track.values[i0_:min(i0_ + 100, i1_ + 1), j_]
                    k_ = i0_ + int(np.argmin(w_))
                    if track.values[i0_, j_] > track.values[k_, j_] + 0.1:
                        track.values[:k_, j_] = track.values[k_, j_]
                        print(f"  발성 첫머리 갑상피열근: {k_} 틀까지 {track.values[k_, j_]:.2f} 로 (첫 유성 틀 {w_[0]:.2f})", flush=True)
    # **파열 기반 관측은 `--init` 복원 뒤에 건다** (§52.397). 앞에 두면 복원이 그 값을
    # 덮어써서, 다듬기 판에서는 `--burst-smooth`·`--stop-close` 가 통째로 무효가 된다 —
    # 파열 검출을 고쳐 7 곳을 찾아 놓고도 결과가 직전 판과 비트 단위로 같았다.
    if a.no_fit and a.init:
        # **재렌더는 파열 관측을 다시 걸지 않는다** (MEASUREMENTS §52.449). 관측은 `--init` 복원 뒤에 걸리게 되어 있어(§52.397, 다듬기에서 관측을
        # 지키려는 것) 재렌더에서 적합된 폐쇄값을 관측값으로 덮었다 — K1a 재렌더의 ㅍ 개방 중역이 원판 −12.8 → +11.2 dB 로 달랐다.
        print("  --no-fit + --init: 파열 관측(궤적 잇기·폐쇄)을 다시 걸지 않는다 — 적합된 제어열 그대로", flush=True)
    else:
        _prep.burst_observations(track, _segment, hop, burst_smooth_ms=a.burst_smooth, stop_close=bool(a.stop_close),
                                 log=lambda m: print(m, flush=True))
    if a.td and not (a.no_fit and a.init):
        # **앞뒤 무음에는 날숨이 없다** (§52.481). 분석은 폐압을 유성 틀 사이로 이어 붙여 무음 끝까지 3–4 cmH2O 를 두고, 무성 끝 틀은 성문을
        # 벌려(내전 ~0.03) 둔다 — 시간 영역 관에서는 그것이 1.2 L/s 의 날숨이 되어 레이놀즈 난류가 무음을 쉿 소리로 채웠다(038 끝 0.55–0.67 s,
        # 원본은 거의 무음). 원본 수준이 정점 −45 dB 아래인 앞·뒤 무음 구간의 폐압을 20 ms 경사로 0 에 가깝게 내린다.
        _lvl = 20.0 * np.log10(np.sqrt(np.add.reduceat(seg[:len(seg) // hop * hop] ** 2,
                                                        np.arange(0, len(seg) // hop * hop, hop)) / hop) + 1e-12)
        _loud = np.nonzero(_lvl > _lvl.max() - 45.0)[0]
        if _loud.size:
            _ip = INDEX["p_sub"]; _n = track.values.shape[0]; _r = max(1, int(round(20.0 / track.frame_ms)))
            _g = np.ones(_n)
            _f0, _f1 = int(_loud[0]), int(_loud[-1])
            if _f0 > _r:
                _g[:_f0] = np.clip((np.arange(_f0) - (_f0 - _r)) / _r, 0.05, 1.0)
            if _f1 < _n - 1 - _r:
                _g[_f1 + 1:] = np.clip(((_f1 + _r) - np.arange(_f1 + 1, _n)) / _r, 0.05, 1.0)
            track.values[:, _ip] = track.values[:, _ip] * _g
            print(f"  앞뒤 무음 폐압 내림: 앞 {_f0} 틀, 뒤 {_n - 1 - _f1} 틀 (정점 −45 dB 아래)", flush=True)
    recording_constants = None
    speaker_constants = load_calibration(a.speaker_from + "_track.npz") if a.speaker_from else None
    source_options = source_config((initial_constants, speaker_constants, recording_constants))
    source_override = False
    config = EngineConfig(sample_rate=48000, frame_ms=a.frame_ms,
                          speaker="female" if prof.f0_nominal > 165 else "male",
                          residual=False, **source_options)
    if config.loaded_source_enabled and (a.open_damp or a.device != "cpu"):
        ap.error("Loaded source requires CPU and cannot be combined with --open-damp")
    if a.seed is not None:
        config.seed = int(a.seed)
    eng = VoiceEngine(config, prof)
    print(f"Source: {config.glottal_source}, load coupling {config.load_coupling:g}"
          f"{' (explicit A/B override)' if source_override else ''}; "
          "new frame controls 0, fitted scalars 0", flush=True)
    room_ir = (initial_constants["recording"].get("room_ir")
               if initial_constants is not None else None)
    if recording_constants is not None:
        if not recording_constants["recording"]:
            raise ValueError("--recording-lock requires recording-scope constants")
        room_ir = recording_constants["recording"].get("room_ir")
    if a.device != "cpu":
        eng = eng.to(a.device)
    from formant_ml.engine.fit import DEFAULT_PARAMS as _DP
    frozen = set(a.freeze or ())
    if not a.voice_gain:
        frozen.add("voice_gain")          # 켜지 않으면 0 dB 로 얼린다 (예전 판과 같은 모형)
    # 엔진이 **안 쓰는** 손잡이를 적합 대상에 남기면 열만 늘어난다 — §40 의 열 희석이 그것이다.
    # `out/M/M11` 이 33 → 43 열로 시작해 2.1 단계에서 85.4 → 77.6 으로 떨어졌다.
    if a.pulse_lock:
        frozen.add("f0_target")           # 위상은 목표 펄스가 만든다 — f0 는 관측되지 않는 자유도가 된다
    if a.tube or a.td:
        from formant_ml.engine.fit import TUBE_PARAMS as _TP
        TUBE_PARAMS_NAMES = set(_TP) | {"c_place"}
        _DP = tuple(_DP) + tuple(_TP) + (("c_place", "pulse_shift", "fold_skew") if a.td else ()) + (("ff_add",) if a.td and a.larynx else ()) + (("vf_ta",) if a.td and (getattr(a, "vf_body", False) or getattr(a, "vf_ns", False)) else ())
        frozen.update(("f1", "f2", "f3", "f4", "bw1", "bw2", "bw3", "bw4"))
        if a.td:
            # 시간 영역 경로가 안 쓰는 손잡이 — 난류는 레이놀즈 조건이, 비강·측지는 관이, 음원 모양은 성문 면적·유량이 낸다.
            frozen.update(("tilt", "aspiration", "fric_gain", "obstacle", "back_leak", "front_len", "nasal_f", "nasal_f2",
                           "nasal_f3", "nasal_z", "nasal_damp", "nasal_gain", "lat_mix", "lat_bw", "tract_gain", "jitter",
                           "shimmer"))
        from formant_ml.engine.tube import N_ART as _NA
        _art = [INDEX[f"art{k}"] for k in range(1, _NA + 1)]
        if a.td and a.artic == "w02":
            from formant_ml.engine.articulation import W02_FIT as _WF, invert_formants_w02 as _inv_w
            _DP = tuple(p for p in _DP if p not in TUBE_PARAMS_NAMES) + tuple(f"vtl_{n}" for n in _WF)
            if a.phoneme_targets and not a.art_free:
                _segs = _fit.ART_SEGMENTS
                _DP = tuple(p for p in _DP if not p.startswith("vtl_") and not (a.art_velum and p == "velum")
                            and not (a.larynx and p == "ff_add") and not (a.art_larynx and p == "adduction"))                  # 가성대도 음소 목표가 낸다          # 조음기는 틀별 적합에서 뺀다 — 음소 목표가 낸다
                print(f"  음소 목표 조음: {a.phoneme_targets} — 구간 {len(_segs)} 개: "
                      + " ".join(f"{sh or 'sil'}@{x0:.3f}" for x0, x1, sh in _segs)[:400], flush=True)
            frozen.update(("a_c", "oral_open"))
            _wn = [INDEX[f"vtl_{n}"] for n in _WF]
            _all = None
            if not np.any(track.values[:, _wn] != np.array([PARAMS[f"vtl_{n}"].default for n in _WF])):
                _fobs = np.stack([np.asarray(track[f"f{k}"], float) for k in range(1, 4)], 1)
                _vo = np.asarray(track.voiced, bool) if np.asarray(track.voiced).size == track.n_frames else np.ones(track.n_frames, bool)
                _pw, _err, _names = _inv_w(_fobs, _vo.astype(float))
                for k, n in enumerate(_names):
                    track.values[:, INDEX[f"vtl_{n}"]] = _pw[:, k]
                print(f"  조음: W02 코드북 역산 — 유성 틀 {int(_vo.sum())}, 로그 F1–F3 오차 rms 중앙 {100 * np.median(_err[_vo]):.1f} %, "
                      f"90 % {100 * np.percentile(_err[_vo], 90):.1f} %", flush=True)
            else:
                print("  조음: 제어열의 W02 변수를 그대로 쓴다", flush=True)
        elif a.td and a.artic == "maeda":
            from formant_ml.engine.articulation import PAR_NAMES as _AN, invert_formants as _inv_m
            _DP = tuple(p for p in _DP if p not in TUBE_PARAMS_NAMES) + tuple(_AN)
            frozen.update(("a_c", "oral_open"))
            _am = [INDEX[n] for n in _AN]
            if not np.any(track.values[:, _am]):
                _fobs = np.stack([np.asarray(track[f"f{k}"], float) for k in range(1, 4)], 1)
                _vo = np.asarray(track.voiced, bool) if np.asarray(track.voiced).size == track.n_frames else np.ones(track.n_frames, bool)
                _sc = tuple(float(torch.exp(q.detach())) for q in (eng.td.log_sc_ph, eng.td.log_sc_or, eng.td.log_sc_area))
                _pm, _err = _inv_m(_fobs, _vo.astype(float), _sc)
                track.values[:, _am] = _pm
                print(f"  조음: 코드북 역산 — 유성 틀 {int(_vo.sum())}, 로그 F1–F3 오차 rms 중앙 {100 * np.median(_err[_vo]):.1f} %, "
                      f"90 % {100 * np.percentile(_err[_vo], 90):.1f} %", flush=True)
            else:
                print("  조음: 제어열의 조음기 값을 그대로 쓴다", flush=True)
        elif not np.any(track.values[:, _art]):
            # **초기값은 분석 포먼트에서 역산한다.** 유성 틀은 가중 1, 나머지 0.2 (포먼트 추정이 흔들린다).
            from formant_ml.engine.tube import invert_formants
            _L = float(eng.tract.length_cm)
            _fobs = np.stack([np.asarray(track[f"f{k}"], float) for k in range(1, 5)], 1)
            _vo = np.asarray(track.voiced, bool) if np.asarray(track.voiced).size == track.n_frames else np.ones(track.n_frames, bool)
            _w = np.where(_vo, 1.0, 0.2)
            _c = invert_formants(torch.as_tensor(_fobs), _L, weight=torch.as_tensor(_w)).numpy()
            track.values[:, _art] = _c
            track.values[:, INDEX["tract_len"]] = _L
            from formant_ml.engine.tube import tube_formants as _tf
            _F, _ = _tf(torch.as_tensor(_c), torch.full((track.n_frames,), _L, dtype=torch.float64), 4, bandwidths=False)
            _e = np.abs(np.log(_F.numpy() / np.maximum(_fobs, 50.0)))[_vo & (_fobs[:, 0] > 50)]
            print(f"  관 모드: 조음 손잡이 역산 — 성도 {_L:.1f} cm, 유성 틀 F1~F4 오차 중앙 "
                  + " / ".join(f"{100 * np.median(_e[:, k]):.1f}" for k in range(4)) + " %", flush=True)
        else:
            print("  관 모드: 제어열의 조음 손잡이를 그대로 쓴다", flush=True)
    bad = frozen - set(_DP)
    if bad:
        raise SystemExit(f"--freeze: 적합 대상이 아닌 이름 {sorted(bad)}")

    if initial_constants is not None:
        apply_calibration(eng, initial_constants, allow_source_override=source_override)
    locked_constants = ()
    if speaker_constants is not None:
        locked_constants += apply_calibration(eng, speaker_constants, ("speaker",), allow_source_override=source_override)
        print(f"  화자 상수 고정: {a.speaker_from} 에서 {len(locked_constants)} 개 — " + ", ".join(sorted(locked_constants)), flush=True)
    if recording_constants is not None:
        locked_constants += apply_calibration(eng, recording_constants, ("recording",),
                                              allow_source_override=source_override)
    if getattr(a, "vf_ns_lock", False):
        from formant_ml.engine.vf_calibration import locked_names
        _lk = locked_names()
        locked_constants += _lk
        print(f"  모드 3 성대 상수 고정 (§52.530 — f0(ε) 표 · 문턱 지도를 잰 상수): " + ", ".join(_lk), flush=True)
    if a.vf_ns_calib or (a.pth_map and _fit.PTH_W > 0):
        from formant_ml.engine.vf_calibration import check_calibration
        _tables = [a.vf_ns_calib] if a.vf_ns_calib else []
        if _fit.PTH_W > 0 and _fit.PTH_MAP and _fit.PTH_MAP not in _tables:
            _tables.append(_fit.PTH_MAP)
        for _table in _tables:
            check_calibration(_table, eng.td, locked_constants, approximate=a.vf_ns_calib_approx,
                              log=lambda m: print(m, flush=True))
    if a.td and a.glottis == "vf" and a.vf_ns:
        from formant_ml.engine.vf_calibration import pitch_coverage
        _cov = pitch_coverage(track["f0_target"], track.voiced if track.voiced.size == track.n_frames else None)
        print(f"  NS 음높이 보정 범위: {_cov['measured_hz']} Hz, 유성 {_cov['frames']} 틀 중 "
              f"표 밖 {_cov['outside']} · 늘어남 한계에 포화 {_cov['saturated']}", flush=True)
        if _cov["outside"]:
            print("  WARNING: 표 밖 음높이는 외삽이다. vf_ns_calib.py로 허용 늘어남 범위를 다시 측정해야 하며, "
                  "포화된 목표들은 서로 다른 성대 입력을 만들지 못한다.", flush=True)
    initial_utterance = initial_constants["utterance"] if initial_constants is not None else {}
    if a.phoneme_init and not a.init and (not a.td or a.artic == "cos"):
        import json as _json2
        from formant_ml.engine.korean import phoneme_init as _pinit
        _al2 = _json2.load(open(a.phoneme_init, encoding="utf-8"))["segments"]
        _t0_ = float(a.t0 or 0.0); _t1_ = float(a.t1) if a.t1 is not None else 1e9
        _segs2 = [dict(sg, t0=max(sg["t0"], _t0_) - _t0_, t1=min(sg["t1"], _t1_) - _t0_) for sg in _al2 if sg["t1"] > _t0_ and sg["t0"] < _t1_]
        _fr2 = _pinit(track.values, INDEX, _segs2, track.frame_ms / 1000.0)
        if np.asarray(track.sibilant).size == track.n_frames:
            track.sibilant = np.asarray(track.sibilant, bool) | _fr2
        else:
            track.sibilant = _fr2
        print(f"  음소 초기값 (곡선 조음): {a.phoneme_init} — 구간 {len(_segs2)} 개, 마찰 틀 {int(_fr2.sum())} 개를 치찰 표지에 더함", flush=True)
    if a.td and a.glottis == "vf" and not a.init and not a.vf_no_psub_rule:
        # 성대 모드의 폐압 초기값 — 목표 음압에서 (voice_td.vf_psub_rule, §52.489)
        _hs = int(round(track.frame_ms * sr / 1000.0))
        _yy = np.asarray(seg, float)
        # 음압은 ±10 ms 창으로 (주기보다 길게 — 2 ms 창은 펄스가 창 어디에 드는지로 ±수 dB 흔들려 폐압에 음높이 결 잔물결을 남겼다, §52.497)
        _hw = max(_hs, int(0.010 * sr))
        _lv = np.array([10 * np.log10(np.mean(_yy[max(0, i * _hs - _hw):i * _hs + _hw] ** 2) + 1e-20) for i in range(track.n_frames)])
        _ps, _ns = _vtdg.vf_psub_rule(_lv, track.voiced, track.fricative, track.frame_ms)
        track.values[:, INDEX["p_sub"]] = _ps
        print(f"  성대 폐압 초기값: 음압 규칙 (2 배당 {_vtdg.VF_PS_DB2:g} dB, 중앙 {_vtdg.VF_PS_REF:g} cmH2O), 무음 틀 {_ns} 개 → "
              f"{_vtdg.VF_PS_SIL:g} cmH2O, 범위 {_ps.min():.1f}–{_ps.max():.1f}", flush=True)
    fit = CopySynthFitter(eng, seg, sr, track, room_ir=room_ir,
                          device=a.device,
                          harmonic_weight=a.harmonic,
                          locked_constants=locked_constants,
                          initial_gain_db=initial_utterance.get("gain_db"),
                          initial_pulse_phi0=initial_utterance.get("pulse_phi0", 0.0),
                          params=tuple(p for p in _DP if p not in frozen))
    # **적합 대상 상수를 판마다 찍는다.** 이 저장소의 상습 실패는 "손잡이를 켰는데 어느
    # 옵티마이저 목록에도 없어서 영원히 초기값" 이다 (§37.3, §52.137). 목록을 눈에 보이게 둔다.
    _fitted = sorted(n for n in fit.constant_parameters if n not in fit.locked_constants)
    print(f"  적합 상수 {len(_fitted)} 개: " + (", ".join(_fitted) or "없음"), flush=True)
    inactive = sorted(set(locked_constants) - fit.constant_parameters.keys())
    if inactive:
        print("  현재 렌더 옵션에서 비활성인 저장 상수: " + ", ".join(inactive), flush=True)
    if fit.harmonic is not None:
        print(f"배음 크기: 가중 {fit.harmonic_weight:g}, "
              f"목표 관측 {fit.harmonic.observations}개 (0~{fit.harmonic.fmax / 1000:.1f} kHz)",
              flush=True)
    if a.prior_w is not None:
        fit.lam_prior = float(a.prior_w)
    kw = {} if a.lr_global is None else {"lr_global": a.lr_global}
    if a.lr_wide and a.lr_global is None:
        kw["lr_global"] = (0.015, 0.03, 0.05, 0.09, 0.15, 0.25)
    if a.init:
        # 펄스 잠금 → 주기별 보정 → 전역 스칼라 → 파열 개방 속도 — `engine.pipeline.restore_state` (§52.470)
        _pl.restore_state(fit, a.init, log=lambda m: print(m, flush=True))
    if a.no_fit:
        print("  --no-fit: 적합을 건너뛰고 제어열 그대로 렌더한다 (재현용)", flush=True)
        rep = fit.fit(0, sizes=_fit.STAGES[-1])
    else:
        rep = fit.fit_staged(global_iters=a.global_iters, stage_iters=a.stage_iters,
                             lr_frame=a.lr_frame, phase_iters=0,
                             patience=a.patience, **kw)
    print(rep)
    if os.environ.get("LOSS_TERMS"):
        _tb = fit.term_breakdown()
        print("  손실 항 (값 · 무게 · 몫): " + ", ".join(f"{k} {v:.3g}·{w:.3g}·{100 * s:.0f}%" for k, (v, w, s) in _tb.items()), flush=True)

    # 보정값은 진단용이며 청취상 동등성이나 신뢰구간을 뜻하지 않는다.
    fid = fit.fidelity()
    print(fidelity_summary(fid))
    if not fid["resolved"]:
        print("  * 분산 보정으로 편향을 판별하지 못했다. 하한·합격 판정으로 쓰지 말고 "
              "잡음비와 아래 치찰음 지문을 함께 읽을 것.")

    out = fit.render()
    if os.environ.get("GA_DUMP") and getattr(fit, "_last_ga", None) is not None:      # 성문 면적 (48 kHz) — 음원 기준용 (§52.524)
        np.save(os.environ["GA_DUMP"], fit._last_ga[0].detach().double().cpu().numpy())
    # **위상 정합률** (fit.PHASE_BANDS, MEASUREMENTS §52.79·§52.80): 대역별 (지연 0 파형 상관) / (목표 주기 상한).
    # 포락 점수는 위상을 직접 재지 않아서 1.5 kHz 위가 0.05 인 판과 0.5 인 판을 구별하지 못한다.
    try:
        phase_match = fit.phase_scores(out)
        print("  위상 정합률  " + "  ".join(f"{k} {v:.1f} %" for k, v in phase_match.items()), flush=True)
    except Exception as _e:                     # 진단이므로 실패해도 판을 버리지 않는다
        phase_match = None
        print(f"  위상 정합률: 못 쟀다 ({_e})", flush=True)
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
    if _fit.CODEC_AAC:                                  # 코덱을 거친 것 (원본과 같은 조건, §52.523)
        sf.write(a.out + "_fit_codec.wav", fit.render_codec(), 48000)
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
    # **판을 재현할 수 있게 명령줄을 통째로 남긴다** (MEASUREMENTS §52.82). 이 스크립트의 깃발 상당수는
    # 엔진·손실의 **모듈 전역**을 바꾸는데(`--aux-zeros`, `--hf-zeros`, `--hf-poles`, `--noise-v2`, `--src-eq`, …)
    # 그 값이 어디에도 안 남아서, 저장된 제어열로 재렌더하면 다른 소리가 났다 (실측 −14.6 / −22.8 점).
    # 진단이 조용히 틀린 답을 내는 자리였다. `pulse_phi0` 도 재현에 필요하므로 함께 남긴다.
    # `argv` 를 **그대로** 남긴다 — 이름(dest)을 깃발로 되돌리려면 파서가 필요한데(`--from` 의 dest 는 `t0` 다)
    # 원문을 두면 그럴 필요가 없다. 사람이 읽을 용도로 풀어 쓴 값도 함께 남긴다.
    cli_args = np.array(json.dumps({"argv": list(sys.argv[1:]),
                                    "resolved": {k: v for k, v in sorted(vars(a).items())
                                                 if isinstance(v, (bool, int, float, str, type(None)))}}))
    np.savez(a.out + "_track.npz", values=res.values, frame_ms=res.frame_ms,
             release_log_r=(fit.rel_log_r.detach().cpu().numpy().astype(np.float64)
                            if getattr(fit, "rel_log_r", None) is not None else np.zeros(0)),
             # 음소 목표·경계 (§52.487) — 없으면 이어 적합이 궤적의 구간 중앙값에서 목표를 다시 잡는데, 궤적은 목표 근사 동역학으로 이미
             # 무뎌진 값이라 패스마다 조음 폭이 줄었다 (WPB ㄲ 폐쇄 0.000 → 0.27 → 0.5 → 1.0 cm²).
             art_z=(fit.art_z.detach().cpu().numpy().astype(np.float64) if getattr(fit, "art_z", None) is not None else np.zeros(0)),
             art_d=(fit.art_d.detach().cpu().numpy().astype(np.float64) if getattr(fit, "art_d", None) is not None else np.zeros(0)),
             # 자기 진동 성대의 음높이 맞춤 (프레임률 긴장 배율, §52.488)
             vf_qcorr=(eng.td.vf_qcorr[0].detach().cpu().numpy().astype(np.float64)
                       if getattr(getattr(eng, "td", None), "vf_qcorr", None) is not None else np.zeros(0)),
             vf_pscorr=(eng.td.vf_pscorr[0].detach().cpu().numpy().astype(np.float64)
                        if getattr(getattr(eng, "td", None), "vf_pscorr", None) is not None else np.zeros(0)),
             vf_qcyc=(np.asarray(eng.td.vf_qcyc, np.float64) if getattr(getattr(eng, "td", None), "vf_qcyc", None) is not None
                      else np.zeros(0)),
             vf_pll_off=(np.asarray(fit._pll_off, np.float64) if getattr(fit, "_pll_off", None) is not None else np.zeros(0)),
             vf_pll_diag=(np.asarray(fit._pll_diag, np.float64) if getattr(fit, "_pll_diag", None) is not None else np.zeros(0)),
             vf_pll_t=(np.asarray(eng.td.vf_pll_t, np.float64) if getattr(getattr(eng, "td", None), "vf_pll_t", None) is not None else np.zeros(0)),
             vf_pll_os=np.array(float(__import__("formant_ml.engine.voice_td", fromlist=["OS"]).OS)),   # 시각표의 표본 단위 (관 해상도 배율, §52.508)
             names=np.array(PARAM_NAMES), cli_args=cli_args,
             pulse_phi0=np.array(float(fit.pulse_phi0.detach().item())),
             # 보정된 샘플률 펄스 위상 — 없으면 재렌더가 적합 당시 소리를 재현하지 못한다 (§52.249)
             pulse_phase=(fit._pulse_phase[0].detach().cpu().numpy().astype(np.float64)
                          if getattr(fit, "_pulse_phase", None) is not None else np.zeros(0)),
             # 주기별 배음 보정 (fit.HCORR_K, §52.268) — 없으면 빈 배열. 재렌더·구조 분석이 읽는다.
             hcorr=(fit.hcorr.detach().cpu().numpy().astype(np.float64)
                    if getattr(fit, "hcorr", None) is not None else np.zeros(0)),
             hcorr_scale=np.array(float(getattr(fit, "_hc_scale", 1.0))),
             # 적합기 전역 스칼라 — 보이스 바·방 잔향 (§52.439). 없으면 `--init` 이 초기값으로 되돌린다.
             fit_scalars=np.array(json.dumps({k: float(getattr(fit, k).detach().reshape(-1)[0])
                                              for k in FIT_SCALARS if getattr(fit, k, None) is not None})),
             hcorr_c0=np.array(int(getattr(fit, "_hc_c0", 0))),
             # 결정적 방 FIR (fit.ROOM_FIR_MS, §52.479) — 탭 값 (직접음 뒤 1 표본부터)
             room_fir=((_fit.ROOM_FIR_SCALE * fit.room_theta).detach().cpu().numpy().astype(np.float64)
                       if getattr(fit, "room_theta", None) is not None else np.zeros(0)),
             engine_params=np.array(json.dumps(eng_p)),
             asp_params=np.array(json.dumps(asp_p)),
             hf_params=np.array(json.dumps(hf_p)),
             acoustic_constants=np.array(json.dumps(constants, allow_nan=False)),
             gain_db=np.array(fit.gain_db()))
    with open(a.out + "_constants.json", "w", encoding="utf-8") as f:
        json.dump(constants, f, ensure_ascii=False, indent=1, allow_nan=False)
    # **세로줄 폐기 문지기** (사용자 지시, MEASUREMENTS §52.98). 점수와 무관하게, 세로줄이 주기적으로
    # 배열되는 정도의 **목표 대비 초과**가 가로 초과보다 크면 그 판은 폐기다. 결함 판정을 눈이 아니라
    # 절차에 맡겨야 재발을 막는다. 아울러 **최악 빈**의 고정 줄 자리도 함께 찍는다 (대역 중앙값은 그것을 가린다).
    stripe_verdict = None
    try:
        sys.path.insert(0, "scripts/diag")
        import stripes as _st
        _tt = seg if isinstance(seg, np.ndarray) else np.asarray(seg)
        _drop, _bad, _ex = _st.verdict(np.asarray(out, dtype=float)[:len(_tt)], np.asarray(_tt, dtype=float), sr)
        stripe_verdict = {"discard": bool(_drop),
                          "bands": {k: {"dh": v[0], "dv": v[1]} for k, v in _ex.items()}}
        _pa, _f = _st.stripe_profile(np.asarray(_tt, dtype=float), sr)
        _pb, _ = _st.stripe_profile(np.asarray(out, dtype=float)[:len(_tt)], sr)
        if _pa is not None and _pb is not None:
            _d = _pb - _pa
            _j = int(np.argmax(_d))
            stripe_verdict["worst_line_hz"] = float(_f[_j])
            stripe_verdict["worst_line_excess"] = float(_d[_j])
            print(f"  세로줄 문지기: {'**폐기**' if _drop else '통과'}"
                  f"   최악 고정 줄 {_f[_j]:.0f} Hz (초과 {_d[_j]:+.3f})", flush=True)
        if _drop:
            # 판정 규칙은 `stripes.verdict`: **대역 빗살 초과 Δ빗살 > DISCARD_FLOOR** 다. 예전 문구("Δ세로 > Δ가로")는 규칙에 없는
            # 비교를 적어 판정 사유를 오해하게 했다 (§52.257).
            print("  !! 폐기: " + ", ".join(f"{k} Δ빗살 {dv:+.3f} > 문턱 {_st.DISCARD_FLOOR:.3f} (Δ가로 {dh:+.3f})"
                                           for k, dv, dh in _bad), flush=True)
    except Exception as _e:
        print(f"  세로줄 문지기: 못 쟀다 ({_e})", flush=True)

    # **지각 자** (MEASUREMENTS §52.83). DNSMOS 는 onnxruntime·librosa 를 끌고 오므로 프로젝트 환경에 섞지 않는다 —
    # `FORMANT_EVAL_PYTHON` 에 전용 인터프리터를 주면 그때만 부른다. 없으면 조용히 건너뛴다.
    percept = None
    _evalpy = os.environ.get("FORMANT_EVAL_PYTHON")
    if _evalpy and os.path.exists(_evalpy):
        try:
            # Windows 기본 코덱은 cp949 라 한국어 출력이 깨진다 — 인코딩을 명시한다 (실측: UnicodeDecodeError).
            _env = dict(os.environ, PYTHONIOENCODING="utf-8")
            _r = subprocess.run([_evalpy, "scripts/diag/percept.py", a.out],
                                capture_output=True, text=True, timeout=600,
                                encoding="utf-8", errors="replace", env=_env)
            percept = (_r.stdout or "").strip().splitlines()
            if not percept:
                print(f"  지각 자: 출력이 없다 (종료 {_r.returncode}) {(_r.stderr or '')[:200]}", flush=True)
            for _ln in percept:
                print("  " + _ln, flush=True)
        except Exception as _e:
            print(f"  지각 자: 못 쟀다 ({_e})", flush=True)
    def _coherence_report(fit, out):
        """배음 사이 결맞음 — 목표의 값과 합성의 값. 항이 꺼져 있으면 `None`."""
        coh = getattr(fit, "coherence", None)
        if coh is None or not coh.enabled:
            return None
        try:
            y = np.asarray(out, dtype=np.float32)
            if len(y) < coh.n_samples:
                y = np.pad(y, (0, coh.n_samples - len(y)))
            return coh.report(torch.as_tensor(y[:coh.n_samples]))
        except Exception as e:                    # 판을 이것 때문에 잃지 않는다
            return {"error": str(e)}

    with open(a.out + "_report.json", "w", encoding="utf-8") as f:
        json.dump({"env": rep.env, "fine": rep.fine, "per_size": rep.per_size,
                   "fine_corr": fid["fine_corr"], "floor": fid["floor"],
                   "resolved": fid["resolved"], "trust": fid["trust"],
                   "noise_ratio": fid["noise_ratio"],
                   "score_kind": "diagnostic_not_perceptual_equivalence",
                   "harmonic": harmonic_report, "harmonic_weight": fit.harmonic_weight,
                   "constant_budget": fit.constant_budget(), "frame_controls": len(fit.names),
                   "event_coefficients": 0 if fit.w_ev is None else fit.w_ev.numel(),
                   "phase_match": phase_match,
                   "cli_args": json.loads(str(cli_args)),
                   "percept_dnsmos": percept,
                   "stripe_gate": stripe_verdict,
                   "bands": rep.bands,
                   "spectrum_match": fid["spectrum_match"],
                   "coherence": _coherence_report(fit, out),
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
    _late = _recipe.diff(_recipe_snap)
    print(f"조리법 고정 확인: 계산 시작 뒤에 바뀐 설정 전역 {len(_late)} 개"
          + ("" if not _late else " — " + "; ".join(f"{m.split('.')[-1]}.{k}: {o!r} → {v!r}"[:120] for m, k, o, v in _late)), flush=True)
    try:
        # **도약 자** (scripts/diag/jumps.py A 층, §52.463) — 모든 판의 끝에 원본에 없는 튐을 센다.
        import importlib.util as _ilu
        _sp = _ilu.spec_from_file_location("_jumps", os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag", "jumps.py"))
        _jm = _ilu.module_from_spec(_sp)
        _sp.loader.exec_module(_jm)
        _ev = _jm.events(np.asarray(sf.read(a.out + "_target.wav")[0], float), np.asarray(sf.read(a.out + "_fit.wav")[0], float), 48000)
        _sc = _jm.score(_ev)
        print(f"도약 자: 원본에 없는 튐 {_sc['n']} 개, Σ초과솟음 {_sc['sum_dp']:.0f} dB, 최대 {_sc['max_dp']:.1f} dB"
              + ("  — " + ", ".join(f"{d['t']:.3f}s" for d in sorted(_ev, key=lambda d: -d['dp'])[:6]) if _ev else ""), flush=True)
    except Exception as _e:
        print(f"도약 자: 못 잼 ({_e})", flush=True)
    if a.save_ga:
        with torch.no_grad():
            fit.render()
        _ga_ = getattr(fit, "_last_ga", None)
        if _ga_ is not None:
            np.save(a.out + "_ga.npy", _ga_[0].detach().double().cpu().numpy())
            print(f"  성문 면적 저장: {a.out}_ga.npy", flush=True)
    print(f"\n결과: {a.out}_target.wav / {a.out}_fit.wav / {a.out}_track.npz")


if __name__ == "__main__":
    main()
