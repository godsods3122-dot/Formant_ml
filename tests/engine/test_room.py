"""추정한 방 IR — **못 본 구간에서도** 좋아져야 한다."""
import numpy as np
import torch

from formant_ml.engine.room import apply_ir, estimate_ir


def _room(x, fs=48000.0, rt60=0.25, seed=3):
    rng = np.random.default_rng(seed)
    n = int(1.2 * rt60 * fs)
    t = np.arange(n) / fs
    ir = rng.standard_normal(n) * np.exp(-6.907755 * t / rt60)
    # 직접음이 너무 세면 wet ≈ dry 라 시험할 여지가 없다 (기준 상관 0.995).
    # 1.2 이면 기준이 0.7 대라 IR 이 실제로 뭔가 할 수 있는지 보인다.
    ir[0] += 1.2 * np.sqrt((ir ** 2).sum())
    ir /= np.sqrt((ir ** 2).sum())
    return np.convolve(x, ir)[:len(x)]


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


def test_estimated_ir_generalises_to_held_out_audio():
    """앞 절반으로 추정한 IR 이 **뒤 절반**에서도 상관을 올려야 한다.

    이 성질이 깨지면 방을 잡은 게 아니라 그 구간의 잡음을 외운 것이고, 그런 IR 을
    순방향에 넣으면 적합기가 소스 오차를 방으로 떠넘긴다.
    """
    fs = 48000.0
    rng = np.random.default_rng(0)
    n = int(1.0 * fs)
    # 성문 펄스 비슷한 주기 신호 + 잡음 — 마른 '합성' 역할
    t = np.arange(n) / fs
    dry = np.zeros(n)
    dry[::int(fs / 220)] = 1.0
    dry = np.convolve(dry, np.exp(-np.arange(200) / 40.0), "same")
    dry += 0.05 * rng.standard_normal(n)
    wet = _room(dry, fs)
    half = n // 2
    h = estimate_ir(dry[:half], wet[:half], taps=1024, lam=0.01)
    out = apply_ir(torch.as_tensor(dry[None, half:]), torch.as_tensor(h))[0].numpy()
    before = _corr(wet[half:], dry[half:])
    after = _corr(wet[half:], out)
    assert after > before + 0.05


def test_apply_ir_is_differentiable_and_length_preserving():
    x = torch.randn(1, 2048, dtype=torch.float64, requires_grad=True)
    h = torch.randn(64, dtype=torch.float64)
    y = apply_ir(x, h)
    assert y.shape == x.shape
    y.pow(2).sum().backward()
    assert torch.isfinite(x.grad).all() and float(x.grad.abs().max()) > 0


def test_a_delta_impulse_response_is_identity():
    x = torch.randn(1, 512, dtype=torch.float64)
    h = torch.zeros(16, dtype=torch.float64); h[0] = 1.0
    assert torch.allclose(apply_ir(x, h), x, atol=1e-9)
