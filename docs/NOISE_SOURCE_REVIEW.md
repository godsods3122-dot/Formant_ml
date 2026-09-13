# 잡음 음원 구조 — 문헌 검토와 재설계안 (0.3.19, 2026-09-11)

사용자: *"그냥 잡음이지. 이 경향은 극이 지나치게 노이즈의 일부를 Q 공진을 강화했거나, 주기적
잡음이나 엔벨로프가 남았을 때 나는 건데, 세로 토막을 없애는 게 우선이고, 더이상 이건 손실함수의
문제가 아니라 구조적 문제가 심각하다. 그 부분을 뒤엎는 게 베스트다. 논문을 싹다 찾아."*

이 문서는 (1) 문헌이 잡음 음원을 어떻게 만드는지, (2) 지금 엔진이 그와 어디서 어긋나는지(실측과
함께), (3) 무엇으로 갈아엎는지를 적는다. 실측 근거는 `docs/MEASUREMENTS.md` §50 이다.

## 1. 문헌 — 네 갈래

### 1.1 물리 음원: 협착 난류는 "압력 강하에 비례하는 등가 음압원" 이고, 모양은 앞공동이 낸다

* **Stevens (1971)**, *Airflow and turbulence noise for fricative and stop consonants: static
  considerations*, JASA 50(4B):1180–1192. 협착의 난류 음원을 **압력 강하에 비례하는 등가 음압원**으로
  두고, 방사음의 특성은 음원 위치·방사·음원 자체의 성질이 정한다.
* **Shadle** (장애물 음원 대 벽 음원). /s, ʃ/ 는 협착 하류의 **급한 장애물(앞니)** 에서, /ç, x/ 는
  벽을 비스듬히 때리는 흐름에서 소리가 난다. 장애물 음원은 **쌍극자**(= 압력원).
* **Narayanan & Alwan (2000)**, *Noise source models for fricative consonants*, IEEE TSAP. 장애물
  근처에 놓은 **3 모수 쌍극자 음원**으로 마찰음 스펙트럼의 세부 대부분이 설명된다.
* **Birkholz et al. (2006)**, *Noise sources and area functions for the synthesis of fricative
  consonants*. 협착에 **단극자 하나**, 장애물(앞니·입술·벽)에 **쌍극자 하나** — 관 모형 안에서.
* **Sinder, Krane & Flanagan (1998)**, *Synthesis of fricative sounds using an aeroacoustic noise
  generation model*, JASA 103. Howe 의 음향 유추에 기반한 시간 영역 음원 — 세기·임피던스·시간 특성을
  폐압·협착 지름·하류 형상에서 정한다.
* 스펙트럼 모양: 앞공동이 짧을수록 공진이 높다 — /s/ 의 주 봉우리는 **앞공동 공진**이고, 봉우리와
  저역 골의 차가 치찰음을 가르는 척도로 쓰인다(Jongman; Li et al. 2007).

### 1.2 포먼트 합성: 마찰은 **병렬 가지**로, 넓은 공진기와 우회로로

* **Klatt (1980)**, *Software for a cascade/parallel formant synthesizer*, JASA 67(3). 모음은 종속
  가지, **마찰은 병렬 가지**의 공진기들(각자 진폭 제어)로 만든다. 앞공동이 너무 짧아 공명이
  무의미한 소리(/f, v, θ, ð, p, b/)는 **우회로**(평평한 스펙트럼)를 쓴다. 잡음 음원 하나를 마찰과
  기식이 같이 쓴다. KlattGrid(Weenink 2009)가 같은 구조의 현대 구현이다.

### 1.3 기식·유성 마찰: 잡음은 **성문 주기에 맞춰 터져야** 목소리에 붙는다

* **Hermes (1991)**, *Synthesis of breathy vowels: some research methods*, Speech Communication.
  **정상(stationary) 잡음은 따로 떨어진 음원으로 들린다.** 펄스열과 **같은 주기의 시간 포락**을
  가진 고역 잡음(펄스와 같은 에너지)이라야 지각적으로 모음에 녹아 자연스러운 숨소리가 된다.
* **Mehta & Quatieri (2005)**, *Synthesis, analysis, and pitch modification of the breathy vowel*,
  WASPAA. 기식 잡음을 **주기 음원 파형으로 시간 변조** — 변조와 그 동기가 자연스러움에 결정적.
* **Stylianou (HNM)**. 최대 유성 주파수(MVF) 아래는 배음, **위는 잡음**이고, 유성 구간의 잡음은
  **피치에 동기한 삼각형 포락**으로 변조한다 — 유성 마찰음의 자연스러움에 필요.
* **Drugman (DSM, 2009)**. 잔차의 결정 성분(MVF 아래)과 **고역통과 잡음 + 에너지 포락**(MVF 위).
  종래 보코더의 "윙윙거림" 을 줄였다.
* **Jackson & Shadle (2000)**, *Frication noise modulated by voicing, as revealed by pitch-scaled
  decomposition*, JASA 108(4). 유성 마찰에서 **잡음이 맥동**한다. **Pincas & Jackson (2005)**:
  유성에 의한 마찰 AM 은 **포화한다**(깊이에 상한이 있다).

### 1.4 혼합 여기 보코더·미분 가능 합성: 잡음은 **스펙트럼 포락으로 모양을 입힌다**

* **Morise, WORLD / D4C (2016)**. 대역 비주기성(5 대역)으로 혼합 여기.
* **Wang & Yamagishi, h-NSF (2019)**. 배음 가지와 잡음 가지를 따로 두고, 둘을 가르는 저역·고역 필터의
  **차단 주파수(MVF)를 학습**한다. **Cyclic noise (2020)**: 펄스열 ⊛ 정적 잡음 = 준주기 잡음.
* **Engel et al., DDSP (2020)**. 배음 + **시변 FIR 로 모양을 입힌 잡음** — 선형 위상 FIR(창 설계)로
  위상 왜곡과 주파수 응답 물결을 막는다. **공진기가 없으니 울림(Q)이 없다.**
* **Yu et al. (DAFx 2024)**, *Differentiable all-pole filters for time-varying audio systems*.
* 시변 필터의 계수는 **샘플률로 보간**해야 지퍼 잡음이 없다(음향 DSP 의 상식; 로그 간격 보간이
  포먼트 전이에 자연스럽다).

## 2. 지금 엔진이 어긋나는 곳 (실측)

| 문헌의 원칙 | 지금 엔진 | 실측 |
|---|---|---|
| 마찰 스펙트럼은 **앞공동**(짧은 관)이 넓게 만든다; 병렬 가지·우회로 | 백색 → 앞공동 **공진기 하나** + `back_leak` 로 **모음 종속 가지(F1~F8 + 고차 사다리, Q ≈ 18)** 를 통과 | 앞공동 대역폭은 **파일 전역 스칼라 하나**: s040 은 하한(×0.4, 좁은 공진)에, s101 은 상한(×2.5)에 붙는다 (`L32`~`L44`). 좁히면 그 파일의 모든 마찰 잡음에 높은 Q 가 걸린다 — **"극이 잡음의 일부를 Q 공진으로 강화"** |
| 음원 세기 ∝ 압력 강하 (매끈한 함수) | 구동 = (Re² − Re_c²)^1.5 / a_c — 문턱 근처에서 **작은 유량 변화를 깊은 골로 증폭** | ㅊ 에서 적합기가 폐압 2.3·내전 0.8 의 문턱 근처 해를 고르면 흐름이 조금만 흔들려도 토막 (`L34`) |
| 기식은 **MVF 위 고역통과 잡음 + 성문 동기 포락**, 깊이는 유성 정도에 비례·포화 | 3 kHz 모서리 1 차 셸프(바닥 −10 dB), **파일 전역 상수**(`--noise-global`), 유성 마스크 **이진**, 성도 종속 가지(높은 Q)를 통과 | 전역 기식을 올리면 맑아야 할 3~5 kHz 에 숨소리(`L43/s101` 주기성 0.01, 목표 0.14~0.22), 내리면 고역이 죽음(`L43/s040` 고역생존 27 %) |
| 잡음 포락·계수는 샘플률로 매끈하게 | 성도 궤적 1 ms 마디 → 1 kHz 빗살 (**고침**, §50.14); 마찰 진폭은 프레임률 선형 보간 | 치찰음 고역 포락의 20~60 Hz 과잉 +3 dB 가 남는다 (§50.9) |

## 3. 재설계 — 잡음 가지를 따로 세운다 (`--noise-v2`, 스트리밍 안전한 시간 영역)

원칙: **잡음은 높은 Q 공진기를 지나지 않는다. 잡음의 포락은 샘플률에서 매끈하다. 기식은 MVF 위에서만,
성문에 동기해, 유성 정도에 비례해 터진다.** 모음 종속 가지(높은 Q)는 성문 배음만 지난다.

### 3.1 마찰 (병렬 저 Q 가지)

* 음원: 백색 → 기존 음원 셸프·쌍극자 기울기(Curle) 유지.
* 모양: **병렬 저 Q 공진기 둘** — 앞공동 1 차 모드 f_p = c/4l, 3 차 모드 3 f_p (−6 dB). 대역폭은
  법칙값과 **`B ≥ f / Q_MAX`** (Q_MAX = 2.5) 중 큰 쪽. 전역 대역폭 배율은 이 가지에서 뺀다.
* `back_leak` 는 모음 종속 가지 대신 **저 Q 사본**(같은 포먼트 주파수, 대역폭 ≥ 0.15 f)으로 보낸다.
* 세기: 구동 = 매끈한 문턱(softplus, 폭 = Re_c 의 20 %)의 1.5 제곱 — C∞ 이고 문턱 아래도 기울기가
  산다. 샘플률 포락에 2 ms 2 단 1 차 평활(성도 궤적과 같은 방식)을 건다.

### 3.2 기식 (MVF 위, 성문 동기)

* 음원: 백색 → **2 차 고역통과, 모서리 = MVF**(기본 5.5 kHz; 배음 위상 분산 개시와 같은 값으로 묶는다
  — 배음이 안개로 무너지는 곳에서 잡음이 이어받는다), 바닥 0.05.
* 포락: 성문 개방기에 동기한 매끈한 펄스(올림 코사인), 깊이 = `clamp(amp/0.8, 0, 1)` × `ASP_AM_MAX`
  (포화, Pincas & Jackson).
* 모양: 성도의 **저 Q 사본**(대역폭 ≥ 0.15 f, 고차 사다리 ×2.5)을 지난다 — 포먼트는 따라가되 울리지
  않는다.
* 제어: 기식은 느린 궤적(20 ms 층, §50.16 `L44`).

### 3.3 판정

`census.py` 의 층층·치찰토막·고역생존·줄 지속, `probe_sib_mod.py`(치찰음 변조 스펙트럼), 치찰음 음색
(`out/_tmp/sibshape.py`: 대역 수준·무게중심), 모음 대역 주기성(`out/_tmp/mvf.py`)과 **사용자가 보는 해상도의
스펙트로그램**(1024 점, 전대역)으로 `L44` 와 나란히 본다. 척도가 좋아졌다는 것만으로 "해결" 이라 하지 않는다.

## 참고 문헌 (링크)

* Stevens 1971 — https://pubs.aip.org/asa/jasa/article/50/4B/1180/719649/
* Narayanan & Alwan 2000 — https://sail.usc.edu/span/pdfs/narayanan2000noise.pdf
* Birkholz 2006 — https://www.vocaltractlab.de/publications/birkholz-2006-rib.pdf
* Sinder, Krane & Flanagan 1998 — https://pubs.aip.org/asa/jasa/article/103/5_Supplement/2775/560459/
* Klatt 1980 — https://www.fon.hum.uva.nl/david/ma_ssp/doc/Klatt-1980-JAS000971.pdf
* Weenink 2009 (KlattGrid) — https://www.isca-archive.org/interspeech_2009/weenink09_interspeech.pdf
* Hermes 1991 — https://www.sciencedirect.com/science/article/abs/pii/016763939190053V
* Mehta & Quatieri 2005 — https://ieeexplore.ieee.org/document/1540204/
* Stylianou (HNM) — https://www.ee.columbia.edu/~dpwe/e6820/papers/Styl01-hnm.pdf
* Drugman (DSM) — https://arxiv.org/abs/2001.00842
* Jackson & Shadle 2000 — https://pubmed.ncbi.nlm.nih.gov/11051468/
* Pincas & Jackson 2005 — https://www.isca-archive.org/interspeech_2005/pincas05_interspeech.pdf
* Morise 2016 (D4C) — https://www.sciencedirect.com/science/article/pii/S0167639316300413
* Wang & Yamagishi 2019 (h-NSF, 학습 MVF) — https://arxiv.org/abs/1908.10256
* Wang & Yamagishi 2020 (cyclic noise) — https://arxiv.org/abs/2004.02191
* Engel et al. 2020 (DDSP) — https://openreview.net/pdf?id=B1x1ma4tDr
* Yu et al. 2024 (미분 가능 시변 전극 필터) — https://arxiv.org/abs/2404.07970
