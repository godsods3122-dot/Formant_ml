"""배음 크기 항의 위쪽 끝 (`fit.HARM_FMAX`) — 3 kHz 하드코딩을 손잡이로 꺼냈다.

통제 실험(§52.171): 포먼트 대역폭이 드러나는 자는 **하모닉 크기의 모양**뿐이다. 빗살 대비는
대역폭 50 -> 800 Hz 에 무반응이고 비주기성만 본다. 그러므로 이 상한 위의 극은 폭을 잡아 주는
것이 아무것도 없다 — 실측으로 목표 F3 209 Hz 대 판 308~341 Hz 였다.
"""
import numpy as np
import pytest

from formant_ml.engine import fit as F
from formant_ml.engine.harmonic import HarmonicMagnitudeLoss

FS = 48000
HOP = 48          # 1 ms


def _voiced(f0=347.0, n=int(FS * 0.3), seed=0):
    """F0 의 배음이 6 kHz 까지 살아 있는 신호."""
    t = np.arange(n) / FS
    y = sum(np.cos(2 * np.pi * k * f0 * t) / k for k in range(1, 18))
    return (y / np.abs(y).max() * 0.5).astype(np.float64)


def _loss(fmax):
    x = _voiced()
    nf = len(x) // HOP
    f0 = np.full(nf, 347.0)
    return HarmonicMagnitudeLoss(x, FS, HOP, f0, np.ones(nf, bool), np.zeros(nf, bool),
                                 fmax=fmax, device="cpu")


def test_the_default_is_unchanged():
    """기본값을 바꾸지 않는다 — 먼저 A/B 로 확인한 뒤에 옮긴다."""
    assert F.HARM_FMAX == 3000.0


def test_raising_it_observes_more_harmonics():
    lo, hi = _loss(3000.0), _loss(6000.0)
    assert hi.observations > lo.observations
    assert hi.fmax == 6000.0


def test_unobservable_harmonics_are_dropped_by_itself():
    """상한을 올려도 잡음에 묻힌 하모닉은 `amp2 > 10·variance` 가 버린다.

    배음이 6 kHz 에서 끊긴 신호에 상한 12 kHz 를 주어도 관측 수가 6 kHz 때와 크게 다르지 않아야
    한다 — 다르면 잡음을 하모닉으로 세고 있다는 뜻이다.
    """
    a, b = _loss(6000.0), _loss(12000.0)
    assert b.observations <= a.observations * 1.35


def test_it_is_wired_into_the_fitter():
    import inspect
    src = inspect.getsource(F.CopySynthFitter.__init__)
    assert "min(HARM_FMAX, self.f_max)" in src


@pytest.mark.parametrize("bad", [0.0, -100.0, float(FS)])
def test_a_nonsense_limit_is_refused(bad):
    """0 < fmax < fs/2 를 벗어나면 조용히 비는 대신 **거절한다** — 오타가 묻히지 않게."""
    with pytest.raises(ValueError, match="invalid target, pitch or sampling grid"):
        _loss(bad)


def test_prior_fmax_is_independent_of_harm_fmax(monkeypatch):
    """`--harm-fmax` 를 올려도 대역폭 관측 사전의 끝은 `HARM_PRIOR_FMAX` 가 정한다 (§52.221).

    둘이 묶여 있으면 배음 항을 넓히는 실험이 곧 사전을 끄는 실험이 되어 원인을 못 가른다.
    """
    import inspect

    from formant_ml.engine import fit as F

    src = inspect.getsource(F.CopySynthFitter.__init__)
    assert "min(self.harmonic.fmax, HARM_PRIOR_FMAX)" in src
    assert F.HARM_PRIOR_FMAX == 3000.0
