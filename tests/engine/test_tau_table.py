"""`PARAM_TAU_PHYS` 의 빈칸 — 보정 극·영점이 상한 없이 떨던 구조 (MEASUREMENTS §52.237·240).

실측: 그 열들이 25 ms 보다 빠르게 떨어 고역 배음마다 측파대를 만든다(13~16 kHz HIR −3 dB, 세로 결맞음
0.52 대 목표 0.11). 43 열을 전부 얼리면 둘 다 사라진다(0.041 · HIR 1.72).
"""
from formant_ml.engine.fit import (DEFAULT_PARAMS, PARAM_TAU_PHYS, PARAM_TAU_PHYS_DEFAULT,
                                   PARAM_TAU_PHYS_FAST)


def test_every_fitted_column_has_a_time_constant():
    """**표에 없으면 무제한**이 되는 구조가 결함이었다 — 적합 열은 모두 표에 있거나 명시적으로 빠른 쪽이어야 한다."""
    missing = [p for p in DEFAULT_PARAMS
               if p not in PARAM_TAU_PHYS and p not in PARAM_TAU_PHYS_FAST]
    assert missing == [], missing


def test_correction_poles_are_slower_than_articulators():
    """보조 극·영점은 조음이 아니다 — 포먼트(5~6 ms)보다 느려야 한다."""
    art = max(PARAM_TAU_PHYS[k] for k in ("f1", "f2", "f3", "f4"))
    for k in list(PARAM_TAU_PHYS):
        if k.startswith(("aux", "azr", "hzr", "hzp")):
            assert PARAM_TAU_PHYS[k] > art, (k, PARAM_TAU_PHYS[k], art)


def test_voice_gain_stays_fast():
    """`voice_gain` 은 주기별 시머라 묶으면 항이 존재할 이유가 없어진다 — 예외로 남긴다."""
    assert "voice_gain" in PARAM_TAU_PHYS_FAST
    assert "voice_gain" not in PARAM_TAU_PHYS


def test_default_is_finite():
    """표에 없는 새 열도 무제한으로 떨지 않는다 — 기본 15 ms (§52.475)."""
    assert PARAM_TAU_PHYS_DEFAULT > 0.0


def test_spectral_columns_are_a_subset_of_real_table_entries():
    """`--tau-spectral-off` 가 빼는 열은 표에 실제로 있는 파라미터여야 한다 — 없는 이름이면 절제가 아무것도 안 뺀다 (§52.256)."""
    from formant_ml.engine import fit as _F
    from formant_ml.engine.control import PARAM_NAMES as _PN
    assert _F.PARAM_TAU_SPECTRAL <= set(_F.PARAM_TAU_PHYS)
    assert _F.PARAM_TAU_SPECTRAL <= set(_PN)
    # 조음기 열은 건드리지 않는다
    assert not ({"f1", "f2", "f3", "f4", "bw1", "adduction", "p_sub"} & _F.PARAM_TAU_SPECTRAL)
