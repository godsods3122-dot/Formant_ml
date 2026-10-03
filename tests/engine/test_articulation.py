"""조음 모형 (`engine/articulation.py`) — 원래 Maeda C 구현과의 일치, 미분 가능성, 균일 관 변환 (MEASUREMENTS §52.485)."""
from pathlib import Path

import numpy as np
import torch

from formant_ml.engine.articulation import MaedaLAM, SpeakerScale, uniform_tube

REF = Path(__file__).parent / "data" / "maeda_lam_ref.npz"


def test_matches_the_original_c_implementation():
    """sensein/VocalTractModels lam_lib.c 를 컴파일해 낸 기준 출력 55 자세 (중립, 변수별 ±2, 무작위 40) — 벽·입술 막기를 부드럽게 한 몫만 다르다."""
    z = np.load(REF)
    with torch.no_grad():
        A, x = MaedaLAM()(torch.as_tensor(z["params"]))
    eA = np.abs(A.numpy() - z["area"]) / np.maximum(z["area"], 0.05)
    ex = np.abs(x.numpy() - z["length"]) / np.maximum(z["length"], 0.05)
    assert np.median(eA) < 1e-5 and np.percentile(eA, 99) < 0.02, (np.median(eA), np.percentile(eA, 99))
    assert np.median(ex) < 1e-5 and ex.max() < 0.01
    assert abs(z["length"][0].sum() - 16.274) < 0.01            # 원 화자(남성) 중립 자세 성도 길이


def test_gradients_are_finite_through_a_closure():
    """혀가 벽에 닿는 자세(혀 몸통 위치 +3, 혀끝 +3)에서도 기울기가 유한하다 — 막기를 부드럽게 한 까닭."""
    m = MaedaLAM()
    p = torch.tensor([[0.0, 3.0, 0.0, 3.0, -1.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    A, x = m(p)
    Au, L = uniform_tube(A, x, 28, m.n_ph, SpeakerScale())
    (Au.log().sum() + L.sum()).backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert A.min() < 0.05                                         # 막힌 칸이 있다


def test_uniform_tube_keeps_volume_length_and_lip_area():
    m = MaedaLAM()
    p = torch.as_tensor(np.random.default_rng(3).uniform(-2, 2, (20, 7)))
    with torch.no_grad():
        A, x = m(p)
        Au, L = uniform_tube(A, x, 28)
    assert torch.allclose(L, x.sum(-1))
    V, Vu = (A * x).sum(-1), (Au * (L / 28).unsqueeze(-1)).sum(-1)
    rel = ((Vu - V) / V).abs()
    assert (rel < 0.05).all(), rel.max()                          # 입술 끝 칸만 원래 면적으로 되돌려 조금 다르다
    assert torch.allclose(Au[:, -1], A[:, -1])


def _td_vowel(p, grad=False):
    """TDPath (ARTIC="maeda") 로 0.15 s 유성 모음 — (48 kHz 출력, 경로)."""
    from formant_ml.engine import voice_td as vt
    old = vt.ARTIC
    vt.ARTIC = "maeda"
    try:
        path = vt.TDPath(48000.0, 48).double()
        T = 150
        f = lambda v: torch.full((1, T), float(v), dtype=torch.float64)
        c = {"p_sub": f(8.0), "velum": f(0.0), "voice_gain": f(0.0), "rd_offset": f(0.0), "fold_skew": f(0.6)}
        pt = torch.tensor(p, dtype=torch.float64, requires_grad=grad)
        for k, n in enumerate(vt.ART_NAMES):
            c[n] = pt[k].expand(1, T)
        st = {"amp": f(0.8), "ag_dc": f(0.02)}
        n = T * 48
        ph = (2 * np.pi * 220.0 * torch.arange(n, dtype=torch.float64) / 48000.0).unsqueeze(0)
        y = path(c, st, ph, n)["audio"][0]
        return y, path, pt
    finally:
        vt.ARTIC = old


def _f1f2(y):
    """LPC (16 kHz, 차수 14) 극에서 F1·F2."""
    from scipy.signal import resample_poly
    from scipy.linalg import solve_toeplitz
    x = resample_poly(np.asarray(y)[2400:], 1, 3) * np.hamming(len(np.asarray(y)[2400:]) // 3 + (len(np.asarray(y)[2400:]) % 3 > 0))
    r = np.correlate(x, x, "full")[len(x) - 1:len(x) + 15]
    a = np.concatenate([[1.0], solve_toeplitz(r[:14], -r[1:15])])
    z = np.roots(a); z = z[(np.imag(z) > 0) & (np.abs(z) > 0.8)]
    f = np.sort(np.angle(z) * 16000 / (2 * np.pi)); f = f[f > 200]
    return f[0], f[1]


def test_maeda_path_makes_distinct_vowels_and_passes_gradients():
    """조음기가 모음을 가른다: 턱 벌림·혀 뒤(/a/)는 F1 이 높고 F2 가 낮다, 턱 닫힘·혀 앞(/i/)은 그 반대. 기울기는 조음기와 화자 배율까지 흐른다."""
    ya, _, _ = _td_vowel([-2.0, 1.5, 0.0, 0.0, 1.0, 0.0, 0.0])
    yi, _, _ = _td_vowel([1.0, -1.5, 0.0, 0.0, 0.5, -1.0, 0.0])
    (a1, a2), (i1, i2) = _f1f2(ya.detach().numpy()), _f1f2(yi.detach().numpy())
    assert a1 > i1 + 200 and i2 > a2 + 600, ((a1, a2), (i1, i2))
    y, path, pt = _td_vowel([0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0], grad=True)
    (y[3000:] ** 2).sum().backward()
    assert torch.isfinite(pt.grad).all() and (pt.grad.abs() > 0).sum() >= 5
    for q in (path.log_sc_ph, path.log_sc_or, path.log_sc_area):
        assert q.grad is not None and torch.isfinite(q.grad) and q.grad.abs() > 0
