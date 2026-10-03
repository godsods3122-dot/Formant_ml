"""`--hf-ladder legacy` 가 명시한 폭 하한을 덮어쓰지 않는다 (MEASUREMENTS §52.255)."""
import re


def test_explicit_floor_wins_over_the_ladder_mode():
    src = open("scripts/copyfit.py", encoding="utf-8").read()
    i = src.index('if a.hf_ladder == "legacy":')
    block = src[i:i + 900]
    assert "if a.extra_bw_floor is not None:" in block
    assert "_tr14.EXTRA_BW_FLOOR = float(a.extra_bw_floor)" in block


def test_the_ladder_mode_is_logged():
    src = open("scripts/copyfit.py", encoding="utf-8").read()
    assert "고역 사다리 {a.hf_ladder}" in src


def test_default_mode_is_still_legacy():
    """기본값은 바꾸지 않는다 — A/B(`spread`) 뒤에 옮긴다."""
    src = open("scripts/copyfit.py", encoding="utf-8").read()
    m = re.search(r'"--hf-ladder", choices=\("legacy", "spread"\), default="(\w+)"', src)
    assert m and m.group(1) == "legacy"
