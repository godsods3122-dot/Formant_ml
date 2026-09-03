# 선행연구 정리

> 이 문서는 "물리 기반 + 학습" 음성합성이 실제로 어디까지 와 있는지 정리한 것이다.
> 결론부터: **아이디어의 방향은 이미 검증된 연구 흐름**이고(2019~2026), 우리가 새로
> 할 수 있는 부분은 (a) 성대 자가진동 모델까지 미분가능 체인에 넣기,
> (b) 노이즈를 협착 위치 기반으로 분리 학습하기, (c) 위상 미분량을 손실로 쓰기 —
> 이 셋의 조합이다.

## 1. 미분가능 소스-필터 / DDSP 계열 (우리 설계의 직계 조상)

| 논문 | 요지 | 우리에게 주는 것 |
|---|---|---|
| [Neural source-filter models (Wang et al., 2019)](https://arxiv.org/abs/1904.12088) | 신경망이 파형을 직접 만들되 소스-필터 구조를 강제. 유성/무성을 따로 만들어 섞어야 마찰음이 산다고 명시 | "하모닉 + 노이즈를 따로 만들어 합친다"는 우리 가설의 최초 실증 |
| [DDSP 리뷰 (Hayes et al., 2023)](https://arxiv.org/abs/2308.15422) | 미분가능 DSP 전반 정리 | 구현 관용구(LTV-FIR, 필터 안정성 파라미터화) 표준 |
| [GOLF (Yu & Fazekas, ISMIR 2023)](https://arxiv.org/abs/2306.17252) · [코드](https://github.com/yoyololicon/golf) | 성문파형 웨이브테이블 + 미분가능 LPC. SOTA 대비 **파라미터 15~21%**, 추론 10배 빠름 | Rd 웨이브테이블 보간으로 미분가능성을 얻는 트릭을 그대로 채택 |
| [HiFi-Glot (2024)](https://arxiv.org/abs/2409.14823) | 신경 보코더가 성문 여기신호를 만들고 **미분가능 공명필터**가 포먼트를 만듦. 포먼트/틸트/피치를 직접 제어 | "포먼트를 명시적 손잡이로 두면서 음질도 낸다"는 존재 증명 |
| [Speaker-independent neural formant synthesis (2023)](https://arxiv.org/abs/2306.01957) | 고전 포먼트 합성기의 제어성 + 신경 보코더의 음질 | 제어 파라미터 집합 설계의 참고 |
| [Differentiable mel-cepstral synthesis filter (2022)](https://arxiv.org/abs/2211.11222) | 합성필터를 미분가능 모듈로 만들어 음향모델과 동시 최적화 | end-to-end 학습 구조 |
| [Ultra-lightweight neural DDSP vocoder](https://arxiv.org/abs/2508.14709) | F0 + periodicity + 성도필터만으로 경량 고품질 | 실시간/온디바이스 목표치의 현실적 기준선 |

## 2. 조음(articulatory) / 도파관 물리모델

| 논문 | 요지 |
|---|---|
| [Vocal Tract Area Estimation by Gradient Descent (2023)](https://arxiv.org/abs/2307.04702) | Kelly-Lochbaum 전달함수를 미분해서 **면적함수를 경사하강으로 역추정** |
| [Differentiable Articulatory Copy-Synthesis of Biphonic Singing (2026)](https://arxiv.org/abs/2606.04943) | 미분가능 KL 도파관 + B-스플라인 성도 파라미터화 + 학습가능 감쇠, 오디오에서 end-to-end 최적화 |
| [Fast articulatory synthesis using DDSP (2024)](https://arxiv.org/abs/2409.02451) | 조음 파라미터 -> 음성, 파라미터 효율 |
| [Four Decades of Digital Waveguides (2026)](https://arxiv.org/abs/2604.12878) | 도파관 수치안정성/손실모델 총정리 |

## 3. 성대(성문) 물리모델 — "진동 모드 조절"의 근거

| 논문 | 요지 |
|---|---|
| Titze (1988), *Physics of small-amplitude oscillation* [(PubMed)](https://pubmed.ncbi.nlm.nih.gov/3372869/) | body-cover 가설, 점막파(mucosal wave)로 자가진동 조건 유도 |
| Story & Titze (1995), *Voice simulation with a body-cover model* | 저차원 자가진동 모델의 표준 구현 |
| Steinecke & Herzel (1995) | **비대칭 2질량 모델의 분기(bifurcation)**: 1:1 → 서브하모닉 → 비주기. 이중음/성구 전환이 같은 방정식에서 나옴 |
| [Triangular body-cover model with 5 intrinsic muscles (2021)](https://arxiv.org/abs/2108.01115) | 후두 근육 활성도 → 진동 모드. "모드 조절"을 근육 좌표계로 |
| [Beam–membrane vocal fold model (2026)](https://arxiv.org/abs/2606.13480) | 자세(posturing)와 성문 형상을 함께 다루는 최신 모델 |
| [PINN for speech production (2025)](https://arxiv.org/abs/2511.00428) | 성대 진동/성도 음향의 지배방정식을 신경망으로 직접 풂. **충돌의 비미분성**을 미분가능 근사로 우회 |
| [ARMAX-LF (2024)](https://arxiv.org/abs/2410.04704) | DNN으로 성문/성도 파라미터 동시 추정 |

## 4. 노이즈(치찰음·기식음)

- Stylianou, *Harmonic plus Noise Model* (IEEE TSAP 2001) — 하모닉/노이즈 분해의 고전. [PDF](https://www.ee.columbia.edu/~dpwe/e6820/papers/Styl01-hnm.pdf)
- Klatt & Klatt (1990) — 기식성(breathiness)은 **성문 개방기에 동기화된 진폭변조 노이즈**로 모델링해야 자연스럽다. 이걸 안 하면 "하모닉 + 별개의 쉬익 소리" 두 층으로 들린다.
- 마찰음의 스펙트럼은 협착부 **앞공동(front cavity) 길이**가 결정한다 → /s/(6~8 kHz)와 /ʃ/(2.5~4 kHz)의 차이. 우리 구조에서는 노이즈가 협착 하류 단만 통과하도록 강제해 이걸 학습 대상에서 제외했다.

## 5. 위상 — "AI 특유의 잡음"의 진짜 정체

- [Revisiting Vocos: phasiness (2026)](https://arxiv.org/html/2607.24323v1) — 크기 스펙트럼 모델링은 잘 되지만 위상이 약하면 phasiness가 남는다.
- [FA-GAN (Interspeech 2024)](https://www.isca-archive.org/interspeech_2024/shen24b_interspeech.pdf) — 아티팩트 없는 위상 인지 보코더.
- [RPU 위상복원 (2020)](https://arxiv.org/abs/2002.05832), [von Mises DNN (2018)](https://arxiv.org/abs/1807.03474) — 위상은 **순시주파수(IF)와 군지연(GD)** 로 다뤄야 한다(랩 문제 회피).
- 요점: 위상 불안정 → "전기적 버즈". 시간영역에서 위상 연속적으로 합성하면 이 실패 모드 자체가 사라진다.

## 6. 이 레포의 포지션

```
        고전 포먼트 합성기(Klatt)          신경 보코더(HiFi-GAN 등)
        + 완전한 제어/해석성                + 높은 자연성
        - 로봇 같은 음질                    - 블랙박스, 위상 아티팩트, 제어 불가
                        \                 /
                         물리 파라미터만 학습
                       (파형은 방정식이 생성)
```
