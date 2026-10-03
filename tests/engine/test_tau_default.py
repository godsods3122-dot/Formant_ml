"""표에 없는 파라미터의 시상수 기본값 (`fit.PARAM_TAU_PHYS_DEFAULT`).

`PARAM_TAU_PHYS` 는 나중에 더한 파라미터를 못 따라왔다 — 59 개 중 **33 개**가 표에 없어서
`--param-tau` 를 켜도 그것들만 빠르기 상한이 없다. 실측(§52.176): `hz*_f` 의 150~400 Hz 흔들림이
그 다음 파라미터(`f4`)보다 **40 dB** 크고, 0~20 Hz 대비 19 dB 밖에 차이가 안 난다(= 사실상 백색).
"""
from formant_ml.engine import fit as F


def test_the_default_is_a_safe_finite_time_constant():
    """표에 없는 손잡이는 **느린 쪽(후두 15 ms)** 으로 떨어진다 (MEASUREMENTS §52.475). 0(제한 없음)이던 때 새로 더한 관 모드
    조음 손잡이가 1 ms 마다 떨어 포먼트가 1 ms 당 50~185 Hz 흔들렸다(갈라짐 σ 0.49)."""
    assert F.PARAM_TAU_PHYS_DEFAULT == 15.0


def test_the_gap_is_now_closed():
    """예전에는 표가 파라미터 목록을 못 따라와 33 개가 비어 있었다(이 시험이 그 사실을 고정했다).

    §52.244 에서 표를 채웠다 — 보정 극·영점이 상한 없이 떨어 고역 배음 사이를 채우고 세로선을 만든다는 것을
    절제로 확인했기 때문이다(§52.237·240). 이제는 **다시 벌어지지 않는 것**을 지킨다.
    """
    miss = [n for n in F.DEFAULT_PARAMS if n not in F.PARAM_TAU_PHYS and n not in F.PARAM_TAU_PHYS_FAST]
    assert miss == [], miss




def test_voice_gain_is_exempt():
    """성문의 주기별 이득이다 — F0 율(347 Hz) 변조를 잡으라고 둔 것이라 묶으면 안 된다."""
    assert "voice_gain" in F.PARAM_TAU_PHYS_FAST
    assert "voice_gain" not in F.PARAM_TAU_PHYS


def test_the_articulators_keep_their_own_values():
    """기본값은 **채우기**일 뿐 덮어쓰기가 아니다."""
    assert F.PARAM_TAU_PHYS["a_c"] == 2.0
    assert F.PARAM_TAU_PHYS["p_sub"] == 30.0
    assert F.PARAM_TAU_PHYS["f2"] == 5.0
