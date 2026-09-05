# 기준 자료 (recordings / papers)

계정이나 세션을 옮겨도 매번 다시 올리지 않도록 레포에 둔다.
`.gitignore` 의 `*.wav` 규칙에 `!reference/**` 예외를 넣어 두었다.

`src/` 가 아니라 여기 두는 이유: `src/` 는 `pyproject.toml` 의
`[tool.setuptools.packages.find] where = ["src"]` 가 훑는 파이썬 패키지 트리다.
16 MB 짜리 PDF·WAV 를 그 안에 두면 배포 패키지에 딸려 들어간다.

---

## recordings/ — 화자 본인 녹음 (남성, 44.1 kHz)

| 파일 | 내용 |
|---|---|
| `ko_liquid_ra-eulla-ara_male_44k.wav` | "라 / 을라 / 아라 / 아라" 네 토큰 |
| `ko_tongue_raise-curl_male_44k.wav` | 혀가 위로 가서 닿는 구간, 혀가 말려서 뒤로 가는 구간 |

**주의: 이 레포는 public 이다.** 본인 음성이 공개 저장소에 올라간다는 뜻이므로,
원치 않으면 레포를 private 으로 돌리거나 이 폴더를 빼면 된다.

측정에 쓰는 구간(초). `scripts/copysynth.py --from/--to` 에 그대로 넣는다:

| 토큰 | 시작 | 끝 |
|---|---|---|
| 라 | 0.50 | 1.02 |
| 을라 | 1.58 | 2.55 |
| 아라 #1 | 3.20 | 3.86 |
| 아라 #2 | 5.36 | 6.08 |

화자 정규화(남성 -> 목표 여성)는 본인 /아/ 761/1222/2492 와 목표 여성 /아/
850/1220/2810 의 포먼트별 비 **1.117 / 0.998 / 1.128** 을 쓴다.
자세한 것은 [`../docs/HANDOFF_LIQUID.md`](../docs/HANDOFF_LIQUID.md) §3.2.

## papers/ — 전부 오픈액세스 (재배포 가능)

| 파일 | 출처 | 라이선스 |
|---|---|---|
| `Lee2015_Korean-liquid-across-prosodic-positions_rtMRI.pdf` | Lee, Goldstein & Narayanan, ICPhS 2015 (USC SPAN) | 학회 공개본 |
| `Hwang2019_Korean-laterals-3D-ultrasound.pdf` | Hwang, Charles & Lulich, *Phonetics and Speech Sciences* 11(1) 19–27, 2019 | **CC BY-NC 4.0** |
| `Cathcart2012_alveolar-tap-articulatory-variation.pdf` | Cathcart, *UC Berkeley PhonLab Annual Report* 8(8), 2012. DOI 10.5070/P79n00618c | eScholarship 공개 |
| `Ying2026_lateral-channel-F3-modulation_EMA.pdf` | Ying, *Speech Communication* 176:103345 | **CC BY 4.0** |

각 논문에서 뽑아 쓴 수치와 그 한계는
[`../docs/RIEUL.md`](../docs/RIEUL.md) §1, 요약은
[`../docs/HANDOFF_LIQUID.md`](../docs/HANDOFF_LIQUID.md) §3.1 에 있다.

**이 세션의 작업 환경은 웹 전문 접근이 전부 막혀 있었다**(WebFetch/직접 HTTP 가
전 도메인 차단). 그래서 논문을 파일로 받아야 했다. 다음 사람도 같은 제약이면
여기 있는 PDF 가 그대로 쓰인다.
