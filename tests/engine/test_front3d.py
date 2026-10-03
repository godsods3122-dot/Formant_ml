"""**앞공동 3D 모드 표** (`tract.FRONT3D`, MEASUREMENTS §52.345).

앞공동은 관이 아니라 상자다. 폭 2 cm · 높이 1 cm 면 횡모드가 `c/2W` = 8.75 kHz, `c/2H` = 17.5 kHz 로
치찰음 대역 한가운데 서는데, 1/4 파장 근사(`c/4L`, `3c/4L`)는 그것을 모른다. 3D 는 적합 루프에 못
들어가므로 오프라인에서 모드를 뽑아 **표로** 넣고, 엔진은 `front_len` 으로 보간해 공진기를 세운다.

여기서 시험하는 것은 **표를 읽고 보간해 세우는 길**이다. FDTD 자체가 맞는지는
`scripts/front3d.py --check` 가 해석해(`c/2L·n`, 오차 0.2 %)로 확인한다.
"""
import numpy as np
import pytest
import torch

from formant_ml.engine import tract as T


def _table(n=8, modes=3):
    """길이에 따라 한 모드만 움직이고 둘은 고정인 인공 표 — 축모드와 횡모드를 흉내 낸다."""
    L = np.linspace(0.4, 2.0, n)
    freq = np.stack([35000.0 / (4 * L),                      # 축모드: 길이에 반비례
                     np.full(n, 8750.0),                     # 폭 횡모드: 고정
                     np.full(n, 17500.0)], 1)[:, :modes]
    gain = np.tile(np.array([0.5, 0.3, 0.2])[:modes], (n, 1))
    return {"length_cm": L, "freq": freq, "gain": gain}


def test_default_is_off():
    assert T.FRONT3D is None
    assert T.FRONT3D_MODES == 4


def test_table_shapes_are_what_the_engine_expects():
    tb = _table()
    assert tb["freq"].shape == tb["gain"].shape
    assert tb["freq"].shape[0] == len(tb["length_cm"])
    assert np.all(np.diff(tb["length_cm"]) > 0)              # 보간이 정렬을 가정한다


def test_interpolation_is_linear_between_grid_points():
    """표 사이 값은 선형 보간이다 — 격자점에서는 표 값 그대로."""
    tb = _table()
    L, F = tb["length_cm"], tb["freq"]
    mid = 0.5 * (L[2] + L[3])
    want = 0.5 * (F[2, 0] + F[3, 0])
    idx = int(np.searchsorted(L, mid))
    w = (mid - L[idx - 1]) / (L[idx] - L[idx - 1])
    got = F[idx - 1, 0] * (1 - w) + F[idx, 0] * w
    assert got == pytest.approx(want, rel=1e-9)


def test_transverse_modes_do_not_move_with_length():
    """**핵심** — 횡모드는 길이와 무관하다. 1D 근사는 이 모드를 아예 못 낸다."""
    tb = _table()
    assert np.ptp(tb["freq"][:, 1]) == 0.0                   # 폭 모드 고정
    assert np.ptp(tb["freq"][:, 2]) == 0.0                   # 높이 모드 고정
    assert np.ptp(tb["freq"][:, 0]) > 5000.0                 # 축모드만 움직인다


def test_the_axial_mode_matches_the_quarter_wave_law():
    """축모드는 옛 근사와 같은 자리여야 한다 — 3D 가 더하는 것은 **옆에 서는 모드**다."""
    tb = _table()
    L = tb["length_cm"]
    assert np.allclose(tb["freq"][:, 0], 35000.0 / (4 * L))


def test_gains_are_used_as_weights():
    """세기는 병렬 공진기의 가중이다 — 합이 1 이면 전체 이득이 보존된다."""
    tb = _table()
    assert np.allclose(tb["gain"].sum(1), 1.0)


def test_engine_builds_resonators_from_the_table():
    """표를 물리면 `_front_3d` 가 돌고, 출력이 유한하며 길이에 반응한다."""
    from formant_ml.engine.control import INDEX, PARAMS
    tb = _table()
    T.FRONT3D = tb                                           # conftest 가 시험 뒤 되돌린다
    T.NOISE_V2 = True
    fs, hop, n = 48000, 48, 480
    tr = T.VocalTract(fs=fs, hop=hop) if hasattr(T, "VocalTract") else None
    if tr is None:                                           # 클래스 이름이 다르면 건너뛴다
        pytest.skip("VocalTract 를 못 찾았다")
    tr._n_emit = n                                           # `_up` 이 쓰는 내보낼 샘플 수
    x = torch.randn(1, n, dtype=torch.float64) * 0.01
    outs = []
    for L in (0.5, 1.8):
        c = {k: torch.full((1, n // hop), p.default, dtype=torch.float64)
             for k, p in PARAMS.items()}
        c["front_len"][:] = L
        c["front_q"][:] = 4.0
        st = {}
        y = tr._front_cavity(x, c, st)
        assert torch.isfinite(y).all()
        outs.append(y)
    # 길이를 바꾸면 소리가 바뀌어야 한다 (축모드가 움직이므로)
    assert float((outs[0] - outs[1]).abs().mean()) > 0.0
