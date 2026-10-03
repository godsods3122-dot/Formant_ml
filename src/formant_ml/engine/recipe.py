"""조리법 고정 — 설정(모듈 전역)과 계산을 가른다 (MEASUREMENTS §52.469).

사용자: *"문제가 생기면 테스트 몇 번으로 바로 버그를 추적할 수 있어야 하는데, 버그가 자꾸 종속적으로 여기 저기 붙는 느낌"*.
`copyfit` 의 깃발 수백 개는 엔진 모듈의 전역을 덮어쓰고, **덮어쓰는 순서가 곧 동작**이었다 — 분석 뒤에 설정한 깃발이 통째로 안 먹은 판(§52.x
`--burst-back-pct`), 되살린 값을 뒤 단계가 덮은 판(§52.466 펄스 위상) 이 같은 부류다.

`snapshot()` 은 `formant_ml.engine.*` 의 대문자 전역(설정값) 중 값으로 비교할 수 있는 것(수·문자열·None·그것들의 튜플/목록/사전)을 찍는다.
**계산을 시작하기 직전에 한 번 찍고**, 판이 끝날 때 `diff()` 로 그 뒤에 바뀐 전역을 전부 드러낸다. 찍은 것은 `<판>_recipe.json` 으로 남겨
어떤 설정으로 돈 판인지 파일 하나로 되살필 수 있게 한다. 배열처럼 데이터를 담은 전역(`FLOOR_REF`, `DIRECT_EVENTS`)은 설정이 아니므로 뺀다.
"""
from __future__ import annotations

import json
import sys

_PRIM = (bool, int, float, str, type(None))


def _plain(v, depth=0):
    if isinstance(v, _PRIM):
        return True
    if depth < 3 and isinstance(v, (tuple, list)):
        return all(_plain(x, depth + 1) for x in v)
    if depth < 3 and isinstance(v, dict):
        return all(isinstance(k, _PRIM) and _plain(x, depth + 1) for k, x in v.items())
    if depth < 3 and isinstance(v, (set, frozenset)):
        return all(isinstance(x, _PRIM) for x in v)
    return False


def _norm(v):
    if isinstance(v, (set, frozenset)):
        return sorted(v, key=repr)
    if isinstance(v, tuple):
        return [_norm(x) for x in v]
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _norm(x) for k, x in sorted(v.items(), key=lambda kv: repr(kv[0]))}
    return v


def snapshot(prefix: str = "formant_ml.engine") -> dict:
    """{모듈: {이름: 값}} — 지금 불러와져 있는 엔진 모듈의 대문자 설정 전역."""
    out = {}
    for name, mod in sorted(sys.modules.items()):
        if mod is None or not name.startswith(prefix):
            continue
        vals = {}
        for k, v in vars(mod).items():
            if k.isupper() and not k.startswith("_") and _plain(v):
                vals[k] = _norm(v)
        if vals:
            out[name] = vals
    return out


def diff(before: dict, after: dict | None = None) -> list[tuple[str, str, object, object]]:
    """(모듈, 이름, 전, 후) — `before` 뒤에 바뀌거나 새로 생긴 설정 전역."""
    after = snapshot() if after is None else after
    out = []
    for mod, vals in after.items():
        old = before.get(mod, {})
        for k, v in vals.items():
            if k not in old:
                if mod in before:           # 모듈은 있었는데 이름이 새로 생김
                    out.append((mod, k, None, v))
            elif old[k] != v:
                out.append((mod, k, old[k], v))
    return out


def save(snap: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=1, sort_keys=True)
