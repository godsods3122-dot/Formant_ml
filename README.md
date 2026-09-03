# Formant_ml — 물리 기반 음성 생성 모델

> **신경망은 파형을 만들지 않는다. 물리모델의 손잡이만 예측한다.**
> 파형은 성대 방정식, 성도 공명, 난류 노이즈가 만든다.

기존 신경 보코더는 파형(또는 스펙트로그램)을 직접 생성하기 때문에, 실패했을 때
사람 소리에 없는 종류의 소리 — 위상 흔들림에서 오는 버즈, 업샘플링 에일리어스,
프레임 경계 클릭 — 를 낸다. 이 레포는 생성 과정을 **음성 생성의 물리**로 바꿔서
그런 실패 모드가 *구조적으로 발생할 수 없게* 만드는 실험이다.

```
 [ 성문 소스 ]        [ 난류 노이즈 ]           [ 성도 공명 ]        [ 위상 정형 ]
 LF 파형 가산합성  +  백색잡음 × 성문동기AM  →  포먼트 캐스케이드  →  올패스 필터  →  음성
 (또는 2질량 ODE)     (협착 하류만 통과)        또는 KL 도파관        (군지연만 변경)
        ↑                    ↑                       ↑                   ↑
        └────────── 신경망은 이 파라미터들만 예측한다 ──────────────────┘
```

- 설계와 로드맵: [`docs/PLAN.md`](docs/PLAN.md)
- 구현된 방정식: [`docs/THEORY.md`](docs/THEORY.md)
- 선행연구 정리: [`docs/LITERATURE.md`](docs/LITERATURE.md)

## 빠른 시작

```bash
pip install torch numpy scipy soundfile
export PYTHONPATH=src

# 1) 학습 없이, 방정식만으로 소리를 만들어 본다 (out/*.wav)
python -m formant_ml.demo --out out

# 2) 물리 엔진 검증 (균일관 공진, 포먼트 정확도, 에일리어싱, 미분가능성 …)
python tests/test_dsp.py

# 3) 합성 결과 분석 (포먼트/무게중심/HNR)
python scripts/analyze.py out/*.wav

# 4) 복사합성 학습 (24 kHz 모노 wav 폴더)
python -m formant_ml.train --data data/wavs --steps 20000 --out runs/exp1
```

`demo` 가 만드는 것 — 전부 신경망 없이 방정식만으로:

| 파일 | 내용 |
|---|---|
| `01_vowel_a_pressed_to_breathy.wav` | /아/ 지속음. 비브라토 + Rd 를 긴장→기식으로 스윕 |
| `02_diphthong_a_i_u.wav` | 아→이→우 포먼트 활음 |
| `03_fricative_s.wav` | /ㅅ/ — 성대 진동 없이 협착부 난류만 |
| `04_syllable_sa.wav` | /사/ — 무성 마찰에서 유성 모음으로 전이 |
| `05_waveguide_area_function.wav` | 포먼트가 아니라 **성도 단면적**으로 제어 |
| `06_vocalfold_{modal,tense,diplophonic}.wav` | 2질량 성대 ODE의 진동 모드 3종 |

## 저장소 구조

```
src/formant_ml/
  config.py          설정 (샘플레이트, 파라미터 물리 범위)
  presets.py         모음 포먼트 / 마찰음 / 면적함수 프리셋
  demo.py            학습 없는 물리 합성 데모
  train.py           복사합성 학습 루프
  dsp/
    core.py          LTV 필터, FFT 컨볼루션, 보간 (재귀 없음 = 길이에 병렬)
    glottal.py       LF 파형 사전 + 대역제한 가산합성
    vocalfold.py     비대칭 2질량 자가진동 모델 (분기/성구/이중음)
    filters.py       공명·반공명·올패스 (설계상 항상 안정)
    tract.py         Kelly-Lochbaum 도파관 전달함수 (래티스 재귀)
    noise.py         난류 소스 + 성문동기 진폭변조
  models/
    synth.py         전체 합성기 (하모닉 경로 / 노이즈 경로 분리)
    encoder.py       mel + F0 → 물리 파라미터 (포먼트 순서 구조적 보장)
    losses.py        멀티해상도 STFT + **위상 미분(IF/GD)** + 물리 정규화
  data/
    features.py      멜, STFT, YIN F0
    dataset.py       wav 폴더 로더
tests/test_dsp.py    물리 엔진 검증 11종
scripts/analyze.py   포먼트/무게중심/HNR 분석
```

## 현재 상태 (Phase 0 완료)

물리 엔진이 검증된 수치를 낸다:

| 검증 항목 | 결과 |
|---|---|
| 균일관(17.5 cm) 공진 | 498 / 1500 / 2502 Hz (이론 500/1500/2500) |
| 포먼트 캐스케이드 정확도 | 목표 대비 **±3% 이내**, DC 이득 1.000 |
| 성문 소스 F0 | 오차 **< 0.05 Hz** (90/200/400 Hz) |
| 에일리어싱 | 비하모닉 성분 **-50 dB 이하** |
| 도파관 안정성 | 임의 면적함수에 대해 모든 극점 \|z\| < 1 |
| 성대 자가진동 | q=1 → 170 Hz, q=2 → 247 Hz (긴장도로 F0 제어) |
| end-to-end 미분 | 인코더 → 물리모델 → 손실 전 구간 그래디언트 흐름 |
| LTV 필터 항등성 | 오차 7e-7 |

다음 단계는 실제 음성으로 복사합성(Phase 1). 로드맵은 [`docs/PLAN.md`](docs/PLAN.md) §4.

## 이 접근의 한계 (미리 밝힘)

물리모델은 잡음을 없애는 대신 **표현력 상한**을 만든다. 전형적인 실패는
히스/지직이 아니라 *부저 같은 과도한 주기성*과 *모델이 표현 못 하는 소리에서의
뭉개짐*이다. 그래서 로드맵의 Phase 4에 잔차(residual) 보정 단계를 두되,
잔차 에너지에 페널티를 걸어 "신경망이 결국 다 해버리는" 붕괴를 막는 설계로 간다.
자세한 내용은 [`docs/PLAN.md`](docs/PLAN.md) §0.
