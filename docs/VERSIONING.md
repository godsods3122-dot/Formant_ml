# 버전 관리 규칙

## SemVer

`MAJOR.MINOR.PATCH` (`pyproject.toml` 의 `version`, git 태그 `vX.Y.Z`).

* **MAJOR**: 제어 파라미터 표(`engine/control.py::PARAMS`)의 이름·단위·의미가 바뀌거나
  스크립트 형식이 호환되지 않을 때. 저장된 토큰 레지스트리/스크립트가 깨지는 변화.
* **MINOR**: 파라미터 추가(기본값으로 옛 스크립트가 그대로 동작), 새 생성기/템플릿,
  새 음소 제스처, 학습 기능.
* **PATCH**: 버그 수정, 상수 재보정, 문서.

`0.x` 동안은 MINOR 가 호환 깨짐을 포함할 수 있다. 0.2.0 이 v2 엔진의 첫 버전이다.

## 브랜치

* `main` — 항상 테스트 통과. 태그는 여기서만.
* `claude/<topic>-<id>` / `feat/<topic>` — 작업 브랜치. PR 로 `main` 에 병합.
* 실험(청취 A/B, 상수 스윕)은 브랜치에서 하고 **결과 표를 커밋 메시지나 docs 에 남긴 뒤** 코드는 되돌려도 된다.

## 커밋 메시지

첫 줄: 무엇을 왜 (한국어 가능). 본문: 측정값(전/후), 되돌린 시도, 남은 문제.
"상수 X 를 3 → 5" 만 적지 말고 **어떤 측정이 그 값을 골랐는지** 적는다.

## 변경 기록

`CHANGELOG.md` (Keep a Changelog 형식). 릴리스마다 Added / Changed / Fixed / Removed.

## 결정 기록 (ADR)

`docs/adr/NNNN-제목.md`. 구조를 바꾸는 결정은 ADR 을 먼저 쓴다(맥락 → 결정 → 결과 → 대안).
되돌릴 때는 새 ADR 로 이전 ADR 을 supersede 한다. 지우지 않는다.

## 호환성 표

| 구성요소 | 안정성 | 비고 |
|---|---|---|
| `engine.control.PARAMS` | 0.2 부터 준안정 | 이름 변경은 MAJOR |
| 키프레임 JSON | 준안정 | `{"t":초, 이름:값, "event":{...}}` |
| `TokenRegistry` JSON | 준안정 | `delta` 길이 = P |
| v1 (`dsp/`, `models/`, `score.py`) | 동결 | 0.3 에서 제거 예정 |

## 재현성

* 렌더는 `EngineConfig.seed` 로 결정적이다(난수 생성기 하나).
* 청취 세트는 `scripts/v2_listen.py` 가 `out/v2/` 에 만들고, 같은 스크립트가 측정표를 출력한다.
  피드백을 받을 때는 **커밋 해시 + 파일 이름 + 측정표**를 같이 남긴다.
