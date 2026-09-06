# Changelog

형식: [Keep a Changelog](https://keepachangelog.com/ko/1.1.0/). 버전 규칙: `docs/VERSIONING.md`.

## [0.2.3] — 2026-09-06

사용자 지적("비강↔구강이 뚝뚝 끊긴다 / 치찰음이 성대 진동과 따로 논다")을 새 여성 녹음 2 개와
기존 녹음의 **ms 단위 전수 분해**로 확인하고 구조를 바꿨다. 계측은 `docs/MEASUREMENTS.md` §6.

### Added
- `scripts/segment_analyze.py` (5 ms 프레임 전수 계측), `scripts/transitions.py` (전사 정렬 없이
  비음·마찰 전이를 찾아 궤적 출력).
- 병렬 비강 분기 (ADR 0010): `oral_open`·`nasal_f2`·`nasal_f3`·`nasal_damp`·`nasal_gain` 파라미터,
  비강 꼬리 극(나이퀴스트까지), 유량 분배.
- `lat_mix` — 측지 영점의 깊이를 연속으로 여닫는다.

### Changed
- 마찰 소스가 대역통과 → **광대역**(셸프 −9 dB + 정점 1.4 배 저역통과). 정점은 앞공동 극이 만든다.
- 성문 제스처가 협착보다 35 ms 먼저 열리고 55~110 ms 늦게 닫힌다.
- 비음 제스처: 연구개(선행 60 / 후행 90 ms)와 구강 폐쇄(35 / 18 ms)가 다른 시간축.
- `scripts/ab_male.py` 가 비교 창을 신호에서 자동 검출한다.
- `fric_gain` 상한 4 → 64. 마찰 세기 기준점을 `phones.SIB_REF_DB` 한 곳으로.

### Fixed
- 비음 머머의 2 kHz 위가 −60 dB 로 무너지던 것 (극 3 개로 잘라서). 꼬리 극으로 rms 11 → 2.3 dB.
- 구강·비강 게이트를 출력에 걸어 해제 순간 저장 에너지가 12 배로 튀던 클릭.
- 측지 영점을 주파수 0 으로 껐다 켜며 생기던 8 배 스파이크 (로그 파라미터의 영차 유지).
- 마찰음 1~3 kHz 가 15 dB 낮던 것. 파찰음이 `fric_gain` 상한에 10 dB 잘리던 것.

## [0.2.2] — 2026-09-06

### Fixed (같은 화자 A/B 로 찾은 장치 결함, ADR 0009)
- **모음 4~8 kHz 가 35~60 dB 어두웠다** — DC 정규화 공명기 11 개의 곱이 6 kHz −27 / 10 kHz −85 dB.
  나이퀴스트까지 극을 채워 완결(표본화된 관). 실측 남성 /아/ 1/3 oct 오차 4.1 dB.
- 비음 머머 중 마찰·버스트 (연구개 열림을 무시한 구강 유량), 3 kHz 저역통과가 머머 고역을 −80 dB 로 죽인 것.
- /s/ 중 성문 기식이 1~3 kHz 를 10 dB 채운 것 (기식 ∝ ΔPg 선형).
- 마찰 소스 혹(Strouhal 0.2 → 장애물 0.5·v/d, Q 0.5), 앞공동 대역폭.

### Added
- `profiles/user_male.json`, `scripts/ab_male.py` — 같은 화자 A/B (녹음 조각 / 합성 wav 짝 + 계측표).
- 성도 float64 경로.

## [0.2.1] — 2026-09-06

### Added
- `engine/profile.py` `SpeakerProfile` + `profiles/yang_female.json` — 여성 화자 문장 녹음 계측값 (ADR 0008).
- `docs/MEASUREMENTS.md`, `scripts/measure_ref.py` — 여성 문장 / 사용자 비음·치찰음 녹음 계측표.
- 구강압 저장·방전 버스트 (파열/파찰 개시 물리), `phones.affricate` (ㅈ/ㅊ/ㅉ), `phones.nasal(coda=)`,
  구 "이리닐씨리래" 데모, 청취 세트 9 종 (`scripts/v2_listen.py --profile`).
- `tviir.peak_coeffs` (극-영점 봉우리), 비강 3 kHz 저역통과.

### Changed
- 음절 길이 300 → 120~140 ms (계측). 탄음 골 깊이·길이는 프로파일. 탄음 과도음 템플릿 기본 0.
- 성문 기식은 성문 양단 압력 강하 √ΔPg 에 비례 (협착 중 기식이 1~6 kHz 를 채우던 것).
- `front_len` 은 0=끔 규약(영차 유지). 마찰 세기 보정: `fric_gain=1` ⇒ 모음 대비 −11 dB.

### Fixed
- 스트리밍에서 구강압 상태가 프레임 성문 면적 대신 첫 샘플을 쓰던 것.
- 경음/파찰음의 내전이 발성 게이트를 넘어 유성음이 되던 것.

## [0.2.0] — 2026-09-06

### Added
- `formant_ml.engine` — v2 엔진. 시간영역 시변 IIR(결합 스캔·numba), 압력 구동 성문 소스,
  마찰/기식/과도음 템플릿 세 생성기, 포먼트+올패스 → 비강 → 측지 성도, 잔차 보정망
  (shifted softplus TCN), 감정 토큰 레지스트리, `VoiceEngine` (오프라인 = 스트리밍),
  한국어 음소 제스처 `phones.py` (여성 기본값: 아라·라·사·나).
- `tests/engine/` 성질 테스트 40 여 종.
- 문서: `docs/AUDIT_v1.md`, `docs/ARCHITECTURE.md`(Mermaid UML), `docs/VERSIONING.md`, `docs/adr/0001~0006`.
- `scripts/v2_listen.py` 청취 세트 + 측정표.

### Changed
- v1 (`dsp/`, `models/`, `score.py`, `liquid.py`, `aeroacoustic.py`) 은 **동결**. 테스트는 유지.

### Fixed (v2 개발 중 측정으로 잡은 것)
- 대역폭이 0 을 지나 극이 단위원에 붙던 규약 문제 (ADR 0006).
- DC 정규화 영점쌍의 원거리 이득 (+25~40 dB) → 극-영점 노치 (ADR 0005).
- 나이퀴스트 근방 이산 공명기 이득 폭주 → 고차 극 상한 0.7·fs/2.
- 큰 대역폭 공명기를 저역통과로 쓰던 것 → RBJ 2차 저역통과.

## [0.1.0] — 2026-09-05
- v1: LF 가산합성, 포먼트 캐스케이드/KL 도파관(주파수영역), 치찰음 극영점, 유음 다질량 혀끝,
  복사합성 학습 루프. 인수인계 `docs/HANDOFF.md`, `docs/RIEUL.md`.
