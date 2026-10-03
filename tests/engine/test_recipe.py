"""조리법 고정 (`engine/recipe.py`, MEASUREMENTS §52.469) — 설정과 계산을 가르는 장치의 시험."""
import re
from pathlib import Path

from formant_ml.engine import fit as F
from formant_ml.engine import recipe


def test_snapshot_sees_engine_settings_and_diff_catches_a_late_change():
    snap = recipe.snapshot()
    assert "formant_ml.engine.fit" in snap and "TRANS_W" in snap["formant_ml.engine.fit"]
    old = F.TRANS_W
    try:
        F.TRANS_W = old + 1.0                      # 계산 도중 누가 설정을 바꿨다
        d = recipe.diff(snap)
        assert ("formant_ml.engine.fit", "TRANS_W", old, old + 1.0) in d
    finally:
        F.TRANS_W = old
    assert recipe.diff(snap) == []


def test_data_globals_are_not_settings():
    import numpy as np
    old = F.DIRECT_EVENTS
    try:
        F.DIRECT_EVENTS = np.zeros(4, np.float32)   # 배열은 데이터 — 설정 목록에 없다
        assert "DIRECT_EVENTS" not in recipe.snapshot()["formant_ml.engine.fit"]
    finally:
        F.DIRECT_EVENTS = old


def test_copyfit_writes_no_module_global_after_the_freeze_point():
    """`copyfit` 은 계산 시작(조리법 고정) 뒤에 엔진 모듈 전역을 쓰지 않는다 — 쓰면 순서가 동작이 된다 (§52.469)."""
    src = Path(__file__).resolve().parents[2].joinpath("scripts", "copyfit.py").read_text(encoding="utf-8").split("\n")
    marks = [i for i, l in enumerate(src) if "_recipe_snap = _recipe.snapshot()" in l]
    assert len(marks) == 1
    pat = re.compile(r"^[^#]*\b[A-Za-z_0-9]+\.[A-Z][A-Z0-9_]+\s*=\s*[^=]")
    late = [(i + 1, l.strip()) for i, l in enumerate(src) if i > marks[0] and pat.match(l)]
    assert late == [], late
