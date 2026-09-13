# 참고 문헌 모음 (papers/)

PDF 원문은 저작권 때문에 저장소에 올리지 않는다 (`.gitignore` 의 `papers/*.pdf`). 이 목록만 추적한다.
다른 기계에서는 아래 링크로 다시 받는다. 2026-09-11 에 모았다.

## 신뢰층 — 무엇을 엄격히 따르고 무엇을 참고만 하는가

사용자: *"옛날에 측정한 자료들은 AI 와 연관성이 감소되기 때문에 괜찮고, 최근 AI 는 그 입장이 그리 정확하진 않을
것이다."* 그 판단을 이렇게 적용한다.

| 층 | 무엇 | 쓰는 법 |
|---|---|---|
| **0** | **목표 녹음의 실측** (이 프로젝트가 맞추려는 대상) | 최상위. 문헌과 부딪히면 이것이 이긴다 |
| **A** | 사람·공기·귀를 잰 오래된 측정·물리·지각 실험 (재현됨) | 엄격히 따른다. 단 목표 실측과 어긋나면 목표를 따른다 |
| **B** | 공학 관례 (합성기 배선, 고정 MVF, 대역 비주기성 개수) | 그 시대 계산·제어 제약의 **선택** — 참고만 |
| **C** | 최근 AI·신경 보코더 논문 | 신경망이 나머지를 메운다는 전제의 벤치마크 주장 — 가장 약하게. 쓸 수 있는 **도구**(미분 가능 필터 등)만 가져온다. 그 결과물이 아직 사람과 구분된다는 것 자체가 그 주장들이 전부가 아니라는 반례다 |

## A 층 — 측정·물리·지각

| 파일 | 요지 (우리에게 걸리는 것) |
|---|---|
| `2000_Narayanan_Alwan_noise_source_models_fricatives` | 장애물 근처 3 모수 쌍극자 음원으로 마찰음 스펙트럼 세부 대부분이 설명된다 |
| `2006_Birkholz_noise_sources_area_functions_fricatives` | 협착에 단극자, 장애물(앞니·입술·벽)에 쌍극자 — 관 모형 안에서 |
| `2000_Jackson_Shadle_frication_modulated_by_voicing` | 유성 마찰에서 잡음이 성문 주기로 맥동한다 (피치 척도 분해로 실측) |
| `2005_Pincas_Jackson_AM_of_frication_by_voicing_saturates` | 유성에 의한 마찰 AM 은 **포화**한다 — 깊이에 상한 |
| `Shadle_aerodynamics_puzzle_voiced_fricatives` | 유성 마찰의 공기역학적 긴장 (성문 저항 대 협착 유량) |
| `2005_Kitamura_hypopharyngeal_cavities_individual_variation` | 하인두(후두관 + 이상와) 모양은 **모음에 무관하게 안정**하고 **화자 간 차이가 크다** → 약 2.5 kHz 위 스펙트럼의 화자성 |
| `2014_Delvaux_piriform_fossae_spectral_impact_PLOS` | 이상와가 고역 스펙트럼에 영점(골)을 만든다 — 3D 인쇄 성도로 실측 |
| `1990_Glasberg_Moore_auditory_filter_shapes_ERB` | 청각 여파기 폭 ERB = 24.7(4.37 f/kHz + 1) — 지각 분해능의 자 |
| `1999_Karlsson_DL_formant_frequency_high_F0` | 높은 F0 (여성)에서 포먼트 주파수 식별 한계 — 배음이 성기면 포먼트를 덜 정밀하게 듣는다 |
| `2014_Monson_perceptual_significance_HFE_voice` | 5 kHz 위 에너지가 음질·방향·명료도·**자연스러움**(7~10.9 kHz)에 기여 |
| `2016_Garellek_voice_source_spectral_slopes` | 음원을 기울기 넷(H1–H2, H2–H4, H4–2 kHz, 2–5 kHz)으로 — Kreiman 심리음향 음질 모형의 음원 쪽 |
| `2007_Li_spectral_measures_sibilants` | 치찰음 척도: 봉우리와 저역 골의 차(AmpD), 앞공동 길이를 따르는 중역 봉우리(FM) |
| `2024_Jongman_phonetics_of_fricatives` | 마찰음 음성학 총설 — 봉우리 위치·스펙트럼 모멘트·상대 진폭이 조음 위치를 가른다 |
| `2011_Flynn_comparing_vowel_formant_normalisation` | 모음 정규화 비교 — 화자 내재·모음 외재·포먼트 내재 방식(Lobanov, Nearey 로그 평균)이 해부학적 차이를 가장 잘 지운다 |

## B 층 — 공학 관례

| 파일 | 요지 |
|---|---|
| `1980_Klatt_cascade_parallel_formant_synthesizer` (암호 걸린 PDF), `Klatt_chapter3_cascade_parallel_description` | 모음·기식은 종속 가지, 마찰은 병렬 가지(공진기마다 진폭), 짧은 앞공동은 우회로 |
| `2009_Weenink_KlattGrid` | Klatt 구조의 현대 구현 (Praat) |
| `2001_Stylianou_harmonic_plus_noise_model` | MVF 아래 배음, 위는 잡음 + 피치 동기 삼각 포락 |
| `2009_Drugman_deterministic_plus_stochastic_residual` | 결정 성분(MVF 아래) + 고역통과 잡음 + 에너지 포락 |
| `2020_maximum_voiced_frequency_estimation` | MVF 는 음질에 따라 변한다 (숨소리 → 낮고, 강한 발성 → 높고, 무성 ~1 kHz) |
| `2021_high_quality_vocoding_design_signal_processing` | 신호처리 보코더 설계 총설 |
| `2011_Birkholz_model_based_articulatory_trajectories`, `2020_Birkholz_VocalTractLab_2.3_manual` | 조음 궤적의 표적 근사 모형, VTL 의 잡음 음원 배치 |
| `2022_review_articulatory_models_speech_production` | 조음 합성 모형 총설 |
| `2011_KARMA_formant_antiformant_tracking` | 포먼트·반포먼트 칼만 추적 |

## C 층 — 최근 AI·미분 가능 합성 (도구만)

| 파일 | 가져올 것 / 따르지 않을 것 |
|---|---|
| `2019_Wang_Yamagishi_hNSF_trainable_MVF` | MVF 를 적합값으로 둔다(도구). 벤치마크 우열 주장은 따르지 않는다 |
| `2020_Wang_Yamagishi_cyclic_noise_NSF` | 준주기 잡음 = 펄스열 ⊛ 정적 잡음 (주기마다 새로 뽑는 고역 무늬의 참고) |
| `2020_Engel_DDSP`, `2023_review_differentiable_DSP_music_speech` | 잡음에 공진기가 아니라 FIR 포락으로 모양을 입힌다 |
| `2024_Yu_differentiable_allpole_timevarying` | 시변 전극 필터의 역전파 (도구) |
| `2021_UnivNet_multiresolution_spectrogram_discriminators`, `2022_NeuralDPS_deterministic_stochastic_multiband` | "STFT 손실만으로는 고역이 과평활된다" — 신경 생성기 전제. 우리 쪽에는 **질감 통계 손실**로 옮긴다 |
| `2021_human_perception_of_audio_deepfakes` | 사람이 합성음을 알아채는 단서 |

## 받지 못한 것 (브라우저로 연다)

* Stevens 1971, *Airflow and turbulence noise for fricative and stop consonants* — https://pubs.aip.org/asa/jasa/article/50/4B/1180/719649/
* Flanagan 1955, *A difference limen for vowel formant frequency* — https://pubs.aip.org/asa/jasa/article/27/3/613/751873/ (포먼트 주파수 DL 3~5 %, 대역폭 DL 20~40 %)
* Hawks 1994, *Difference limens for formant patterns of vowel sounds* — (MIT 사본 404)
* Chistovich & Lublinskaya 1979, *The 'center of gravity' effect in vowel spectra* — https://www.sciencedirect.com/science/article/abs/pii/0378595579900121 (포먼트 간격 3~3.5 Bark 안이면 합쳐 들린다)
* Shackleton & Carlyon 1994, *The role of resolved and unresolved harmonics in pitch perception* — http://audition.ens.fr/P2web/eval2007/DP_shackleton-1994-harmonic_resolution_pitch.pdf (약 10 번째 배음 위는 분해되지 않는다)
* Kreiman et al. 2021, *Validating a psychoacoustic model of voice quality* — https://pmc.ncbi.nlm.nih.gov/articles/PMC7822631/
* McDermott & Simoncelli 2011, *Sound texture perception via statistics of the auditory periphery* — https://www.sciencedirect.com/science/article/pii/S0896627311005629
* 2023, *A two-stage spectral model for sound texture perception* — https://pmc.ncbi.nlm.nih.gov/articles/PMC9950610/
* Li et al. 2010, *Evaluating the spectral distinction between sibilant fricatives through a speaker-centered approach* — https://pmc.ncbi.nlm.nih.gov/articles/PMC3027155/
* Zañartu/Mehta et al. 2010, *A computational model to predict changes in breathiness resulting from variations in aspiration noise level* — https://pmc.ncbi.nlm.nih.gov/articles/PMC2891879/
* Fitch 1997, *Vocal tract length and formant frequency dispersion correlate with body size* — https://pubs.aip.org/asa/jasa/article/101/5_Supplement/3136/561751/
* Hermes 1991, *Synthesis of breathy vowels* — https://www.sciencedirect.com/science/article/abs/pii/016763939190053V
* Mehta & Quatieri 2005, *Synthesis, analysis, and pitch modification of the breathy vowel* — https://ieeexplore.ieee.org/document/1540204/
* Morise 2016, *D4C* — https://www.sciencedirect.com/science/article/pii/S0167639316300413
* Perception of jitter and shimmer in synthetic vowels — https://www.sciencedirect.com/science/article/pii/S0095447019310691 (자연스러우려면 약간의 거칠기가 필요하다)
* Listeners' weighting of acoustic cues to synthetic speech naturalness — https://www.sciencedirect.com/science/article/abs/pii/S0167639310001627 (청자는 **인공물과 불연속**에 가장 큰 무게를 준다)
