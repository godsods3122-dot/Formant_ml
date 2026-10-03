"""저장된 주기별 배음 보정을 되살린다 (`fit.restore_hcorr`, MEASUREMENTS §52.295)."""
import numpy as np
import torch

from formant_ml.engine import fit as F


class _Z:
    def __init__(self, **kw):
        self._d = kw
        self.files = tuple(kw)

    def __getitem__(self, k):
        return self._d[k]


class _Fit:
    """`_hcorr_prepare` 가 이미 자리를 잡아 둔 상태를 흉내 낸다."""
    def __init__(self, k, c):
        self.hcorr = torch.zeros((2, k, c), dtype=torch.float64)
        self._hc_scale = 1.0
        self._hc_c0 = 0

    def _hcorr_prepare(self):
        return None


def test_no_saved_value_says_so():
    assert "없음" in F.restore_hcorr(_Fit(4, 3), _Z())


def test_empty_array_says_so():
    z = _Z(hcorr=np.zeros(0))
    assert "없음" in F.restore_hcorr(_Fit(4, 3), z)


def test_restores_values_scale_and_origin(monkeypatch):
    monkeypatch.setattr(F, "HCORR_K", 4)
    a = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)
    z = _Z(hcorr=a, hcorr_scale=np.array(0.25), hcorr_c0=np.array(7))
    f = _Fit(4, 3)
    msg = F.restore_hcorr(f, z)
    assert np.allclose(f.hcorr.numpy(), a)
    assert f._hc_scale == 0.25
    assert f._hc_c0 == 7
    assert "복원" in msg and "넓혔다" not in msg


def test_smaller_saved_array_lands_in_the_corner(monkeypatch):
    monkeypatch.setattr(F, "HCORR_K", 6)
    f = _Fit(6, 5)
    msg = F.restore_hcorr(f, _Z(hcorr=np.ones((2, 4, 3))))
    assert np.allclose(f.hcorr[:, :4, :3].numpy(), 1.0)
    assert np.allclose(f.hcorr[:, 4:, :].numpy(), 0.0)
    assert np.allclose(f.hcorr[:, :, 3:].numpy(), 0.0)
    assert "넓혔다" not in msg


def test_larger_saved_array_grows_the_slot(monkeypatch):
    """자리가 모자라면 넓힌다 — 자르면 판 끝의 주기들이 보정 없이 울린다."""
    monkeypatch.setattr(F, "HCORR_K", 4)
    a = np.ones((2, 4, 9))
    f = _Fit(4, 3)
    msg = F.restore_hcorr(f, _Z(hcorr=a))
    assert f.hcorr.shape == (2, 4, 9)
    assert f.hcorr.requires_grad
    assert np.allclose(f.hcorr.detach().numpy(), 1.0)
    assert "넓혔다" in msg


def test_off_when_k_is_zero(monkeypatch):
    monkeypatch.setattr(F, "HCORR_K", 0)
    msg = F.restore_hcorr(_Fit(4, 3), _Z(hcorr=np.ones((2, 4, 3))))
    assert "안 쓴다" in msg


def test_reports_when_the_run_could_not_enable_it(monkeypatch):
    monkeypatch.setattr(F, "HCORR_K", 4)

    class _Off(_Fit):
        def __init__(self):
            super().__init__(4, 3)
            self.hcorr = None

    assert "못 켰다" in F.restore_hcorr(_Off(), _Z(hcorr=np.ones((2, 4, 3))))
