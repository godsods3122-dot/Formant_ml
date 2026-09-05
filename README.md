# Formant_ml — 물리 기반 음성 생성 모델

> **신경망은 파형을 만들지 않는다. 물리모델의 손잡이만 예측한다.**
> 파형은 성대 방정식, 성도 공명, 난류 노이즈가 만든다.

기존 신경 보코더는 파형(또는 스펙트로그램)을 직접 생성하기 때문에, 실패했을 때
사람 소리에 없는 종류의 소리 — 위상 흔들림에서 오는 버즈, 업샘플링 에일리어스,
프레임 경계 클릭 — 를 낸다. 이 레포는 생성 과정을 **음성 생성의 물리**로 바꿔서
그런 실패 모드가 *구조적으로 발생할 수 없게* 만드는 실험이다.

```
 [ 성문 소스 ]        [ 난류 노이즈 ]           [ 성도 공명 ]        [ 위상 정형 ]
 LF 파형 가산합성  +  학습된 난류 × 성문동기AM →  포먼트 캐스케이드  →  올패스 필터  →  음성
 (또는 2질량 ODE)     (협착 하류 + 치찰음필터)   또는 KL 도파관        (군지연만 변경)
        ↑                    ↑                       ↑                   ↑
        └────────── 신경망은 이 파라미터들만 예측한다 ──────────────────┘
```

- 설계와 로드맵: [`docs/PLAN.md`](docs/PLAN.md)
- 구현된 방정식: [`docs/THEORY.md`](docs/THEORY.md)
- 선행연구 정리: [`docs/LITERATURE.md`](docs/LITERATURE.md)
- **목소리를 만들고 조종하는 법: [`docs/VOICE.md`](docs/VOICE.md)**
- 진행 중인 /s/ 치찰음 작업 인수인계: [`docs/HANDOFF.md`](docs/HANDOFF.md)
- 진행 중인 /ㄹ/ 문헌 조사: [`docs/RIEUL.md`](docs/RIEUL.md)
- **유음·복사합성 인수인계: [`docs/HANDOFF_LIQUID.md`](docs/HANDOFF_LIQUID.md)**
  (고역이 '토막' 으로 끊기는 원인 진단은 그 문서 §2)

## 빠른 시작

```bash
# 0) 환경 구축 (venv + 의존성 + editable 설치). PyPI 만 쓴다.
bash scripts/setup_env.sh && source .venv/bin/activate
# 스크립트를 쓰지 않는다면:
#   pip install torch numpy scipy soundfile pyyaml pytest && export PYTHONPATH=src

# 1) 학습 없이, 방정식만으로 소리를 만들어 본다 (out/*.wav)
python -m formant_ml.demo --out out

# 2) 검증 92종 (물리 엔진 11 + 목소리 제어 55 + 공기역학 18 + 학습 8)
python -m pytest tests -q
# 또는 개별로: python tests/test_dsp.py / test_voice.py / test_aeroacoustic.py

# 3) 스크립트로 원하는 소리를 만든다 (웃음·숨·프라이·속삭임 포함)
python -m formant_ml.render examples/laugh_and_speech.yaml -o out/line.wav
python -m formant_ml.render --list-params      # 조종 가능한 모든 파라미터

# 4) 실제 녹음에서 '그 사람의 목소리'를 뽑아낸다
python -m formant_ml.analysis.extract \
    --wav data/me/*.wav --vowel-wav data/me/vowel_a.wav \
    --sibilant-wav data/me/s.wav --glissando-wav data/me/gliss.wav \
    --out profiles/me.json --name me

# 5) 복사합성 학습 (24 kHz 모노 wav 폴더)
python -m formant_ml.train --data data/wavs --steps 20000 --out runs/exp1
```

## 목소리를 조종한다는 것

모든 소리 — 모음, 치찰음, 웃음, 한숨, 성대 프라이, 속삭임 — 가 **같은 물리
손잡이의 다른 조합**이다. 특별 취급하는 코드 경로가 없다. 그래서 전부 한 장의
스크립트에서 연속적으로 섞을 수 있다.

```yaml
timeline:
  - {type: syllable, onset: s, vowel: a, dur: 0.5, f0: [150, 128],
     sib_pole_f: 6800, sib_zero_f: 2900}          # 치찰음 지문을 직접 지정
  - {type: laugh, dur: 1.2, rate_hz: 5.5, voiced: 0.85, tilt: 3.0}
  - {type: breath, dur: 0.35, inhale: true}
  - {type: whisper, vowel: eo, dur: 0.45}
  - {type: creak, dur: 0.35, rate_hz: 42}
```

값은 상수 `1.2`, 시작-끝 `[0.5, 2.2]`, 브레이크포인트 `[[0, 0.5], [1, 2.2]]`
중 아무 형식이나 쓸 수 있고, **모든 파라미터에 같은 규칙**이 적용된다.
전체 목록은 `--list-params` 또는 [`docs/VOICE.md`](docs/VOICE.md).

## 저장소 구조

```
src/formant_ml/
  config.py          설정 (샘플레이트, 파라미터 물리 범위)
  presets.py         모음 포먼트 / 마찰음 / 면적함수 프리셋
  voice.py           VoiceProfile — 한 화자를 이루는 숫자들 (JSON)
  gestures.py        웃음·숨·한숨·프라이·속삭임·흐느낌·헛기침
  aerodynamics.py    성문하압·내전 -> 세기/F0/성문파/기식 (발성 시작의 물리)
  prosody.py         운율 계획 (속도/피치 엔벨로프/억양 폭/호흡) — LLM 제어면
  streaming.py       청크 단위 실시간 합성 (지연 ~21 ms, CPU 5x 실시간)
  score.py           스크립트(YAML/JSON) -> 제어 파라미터
  render.py          스크립트 -> wav (CLI)
  demo.py            학습 없는 물리 합성 데모
  train.py           복사합성 학습 루프
  dsp/
    core.py          LTV 필터, FFT 컨볼루션, 보간 (재귀 없음 = 길이에 병렬)
    glottal.py       LF 파형 사전 + 대역제한 가산합성 (+ tilt / 위상차 / 지터)
    vocalfold.py     비대칭 2질량 자가진동 모델 (분기/성구/이중음)
    filters.py       공명·반공명·올패스·기울기·부분 캐스케이드 (설계상 항상 안정)
    tract.py         Kelly-Lochbaum 도파관 전달함수 (래티스 재귀)
    noise.py         학습되는 난류 소스 (스펙트럼 사전 + 변조 스펙트럼)
    sibilant.py      치찰음 극-영점 필터 (화자 지문 6개 숫자)
    phase.py         위상차 파라미터: 하모닉 상대위상(RPS), 임의 주파수 위상 평가
  models/
    residual.py      잔차 보정망 (물리모델이 설명 못 하는 부분만, 구조적 상한)
    synth.py         전체 합성기 (하모닉 경로 / 노이즈 경로 분리)
    encoder.py       mel + F0 -> 물리 파라미터 (포먼트 순서 구조적 보장)
    losses.py        멀티해상도 STFT + 대역 + 위상(IF/GD) + 상대위상 + 주기성
  analysis/
    registers.py     성대 진동 모드: H1-H2 -> Rd, 서브하모닉, 성구, **파사지오**
    sibilant.py      치찰음 지문 경사하강 추정
    phase.py         상대위상 측정 -> 위상차 올패스 적합
    extract.py       위 전부를 묶어 VoiceProfile 을 만드는 CLI
  data/
    features.py      STFT, 멜, 로그대역 에너지, YIN F0
    dataset.py       wav 폴더 로더
tests/test_dsp.py    물리 엔진 검증 11종
tests/test_voice.py  목소리 제어/분석 검증 26종
examples/            스크립트 예제
scripts/analyze.py   포먼트/무게중심/HNR 분석
```

## 현재 상태

**물리 엔진** (`tests/test_dsp.py`)

| 검증 항목 | 결과 |
|---|---|
| 균일관(17.5 cm) 공진 | 498 / 1500 / 2502 Hz (이론 500/1500/2500) |
| 포먼트 캐스케이드 정확도 | 목표 대비 **±3% 이내**, DC 이득 1.000 |
| 성문 소스 F0 | 오차 **< 0.05 Hz** (90/200/400 Hz) |
| 에일리어싱 | 비하모닉 성분 **-50 dB 이하** |
| 도파관 안정성 | 임의 면적함수에 대해 모든 극점 \|z\| < 1 |
| 성대 자가진동 | 긴장도 q 로 폐쇄 주기율 제어 (q=1 → 170 Hz, q=2 → 247 Hz) |
| end-to-end 미분 | 인코더 → 물리모델 → 손실 전 구간 그래디언트 흐름 |
| LTV 필터 항등성 | 오차 7e-7 |

**목소리 제어와 역추정** (`tests/test_voice.py`)

| 손잡이 | 소리에서 되찾는가 |
|---|---|
| 소스 기울기 `tilt` | ±8 dB/oct 를 돌리면 5 kHz 위 에너지가 16 dB 단조 변화 |
| 위상차 `disp_*` | 크기 스펙트럼 변화 0.5% 이내, 파형은 175% 변화. 1800 Hz/r=0.85 → **1805 Hz/0.849 로 복원** (잔차 0.006 rad) |
| 치찰음 극 | 설정 3200/5000/7000/9000 Hz → 재추출 3165/4893/6913/8913 Hz |
| 개방지수 → `Rd` | 합성 Rd → H1-H2 → Rd 왕복 오차 **0.05 이내** (열린 모음) |
| **파사지오** | 224 Hz 에 넣은 성구 전환을 **221 Hz 에서 검출** (매끄러운 글리산도에서는 오검출 없음) |
| 프로파일 전체 | 합성 목소리를 추출하면 F0 140→140, Rd 1.0→1.1, 치찰음 극 7400→7387 Hz |
| 실측 정합 | 실제 녹음("스")의 /s/ 스펙트럼과 평균 절대오차 **5.0 dB** (수정 전 15.6) |
| 소리 크기 | 모음-마찰음 비가 프로파일값과 ±1 dB 이내로 일치 (실측 +5.4 dB) |
| 다중 질량 성대 | 5겹에서 점막파 +0.46 ms(주기의 7%), 하연이 상연을 앞선다 |
| 치찰음 봉우리 모양 | 저역 스커트 +7.7 → **+44 dB/oct** (둥근 돔 → 삼각형), 첨도는 2.98 로 유지 |
| 발성 시작 | 압력이 오르며 세기 0→0.53, F0 110→120 Hz, Rd 2.40→1.20 이 함께 움직인다 |
| 마찰음 구강 결합 | `noise_back_leak` 로 1 kHz 가 −27.8 → −16.6 dB, /s/+/아/ 와 /s/+/이/ 가 10 dB 차이(동시조음) |
| 난류 소스 롤오프 | 5 kHz 위 −6 dB/oct. 8k→11k 감쇠 −2.1 → **−4.9 dB** (사람 4~8) |
| 치찰음 음조감 | 스펙트럼 평탄도 0.059 → **0.15** (사람 0.2~0.4). 위상 무작위화로 스펙트럼이 원인임을 확인 |
| 난류 매끄러움 | /s/·/ʃ/·속삭임의 진폭 첨도 **2.97~3.03** (핑크 노이즈 2.96). 손잡이를 끝까지 올려도 4.3 이하 |
| 난류 | `roughness` 로 비정상성 제어, 스펙트럼 사전/변조 지수가 학습 파라미터 |
| 경계 클릭 | 평활 정도와 무관하게 순간 피크가 본체의 2배 이내 |
| 스트리밍 | 청크 20~250 ms 로 잘라 만들어도 오프라인 대비 차이 **−60 dB 이하** |
| 운율 | 속도 배율이 길이에 반영, 피치가 화자 음역 안에 유지, 긴 발화에 호흡 자동 삽입 |

## 최근에 고친 것 (그리고 왜 중요한가)

1. **고역이 비던 문제.** 포먼트를 6개만 두면 최상단 극 위에서 캐스케이드가
   극당 -12 dB/oct 씩 겹쳐 떨어진다. F0=60 Hz 화자의 8~11 kHz 에너지가
   상대적으로 8e-7 이었다. 나이퀴스트까지 1 kHz 당 1 개(12 개)로 늘리자
   5e-3 이 되었다 — **76 dB 개선**. 여기에 소스 기울기 `tilt`(옥타브당 dB)를
   따로 두어 6~12 kHz 를 직접 조절한다. 하모닉 수도 180 → 240 (F0=50 Hz 에서도
   나이퀴스트까지 채운다).
2. **무음 → 마찰음 전이의 클릭.** 두 가지 원인이 겹쳐 있었다.
   (a) 모음 프리셋이 5개뿐인데 12단까지 마지막 값을 반복해 **극이 겹쳤다**(Q⁴).
   (b) 노이즈 경로의 부분 캐스케이드를 `Π(1−w+wH)` 로 보간해 감쇠가 사라졌다.
   로그 영역 보간 + peak 정규화 + 협착 위치는 보간하지 않기로 바꿔 해결.
3. **치찰음·속삭임의 지글거림.** 난류 변조를 광대역(에너지의 65%가 50 Hz 위)으로
   둔 것이 원인이었다. 백색 소스가 이미 빠른 요동을 담고 있어서, 그 위에 광대역
   곱셈 변조를 얹으면 두 잡음의 곱이 되어 진폭 분포의 꼬리가 두꺼워진다
   (첨도 2.97 → 4.35). 변조를 느린 대역(꺾임 8 Hz, −12 dB/oct)으로 제한하고
   포락선 클리핑을 없애자 **2.97** 로 돌아왔다 — 핑크 노이즈와 같은 수준.
4. **`data/` 패키지가 커밋되어 있지 않았다.** `.gitignore` 의 `data/` 가
   소스 디렉터리까지 먹어서 `train.py` 와 `losses.py` 가 임포트조차 되지 않았다.

## 이 접근의 한계 (미리 밝힘)

물리모델은 잡음을 없애는 대신 **표현력 상한**을 만든다. 전형적인 실패는
히스/지직이 아니라 *부저 같은 과도한 주기성*과 *모델이 표현 못 하는 소리에서의
뭉개짐*이다. 그래서 로드맵의 Phase 4에 잔차(residual) 보정 단계를 두되,
잔차 에너지에 페널티를 걸어 "신경망이 결국 다 해버리는" 붕괴를 막는 설계로 간다.

역추정 쪽 한계는 [`docs/VOICE.md`](docs/VOICE.md) §5 에 측정치와 함께 적어 두었다
(요약: Rd 는 열린 모음에서만 믿을 만하고, 소스 tilt 추정에 아직 ~2 dB/oct 오차가
남는다). 자세한 내용은 [`docs/PLAN.md`](docs/PLAN.md) §0.
