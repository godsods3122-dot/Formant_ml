"""코덱 대역폭 트랙 (`fit.CODEC_TRACK`) — 방송 코덱의 시변 차단을 목표에서 분석해 렌더에 씌운다 (MEASUREMENTS §52.206·209)."""
import numpy as np
import torch

from formant_ml.engine import fit as F

FS = 48000.0


class _Stub:
    _codec_prepare = F.CopySynthFitter._codec_prepare
    _codec = F.CopySynthFitter._codec

    def __init__(self, target):
        self.fs = FS
        self.target = torch.as_tensor(target, dtype=torch.float32).reshape(1, -1)


def _half_cut(n=int(FS * 1.2), cut_hz=17500.0, seed=0):
    """앞 절반은 전대역, 뒤 절반은 `cut_hz` 위를 벽돌처럼 자른 잡음 — 코덱의 대역폭 전환."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n) * 0.05
    h = n // 2
    X = np.fft.rfft(x[h:])
    fr = np.fft.rfftfreq(n - h, 1 / FS)
    X[fr > cut_hz] = 0.0
    x[h:] = np.fft.irfft(X, n - h)
    return x.astype(np.float32)


def _band_db(x, lo, hi, a, b):
    seg = x[int(a * FS):int(b * FS)]
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    fr = np.fft.rfftfreq(len(seg), 1 / FS)
    return 10 * np.log10(S[(fr >= lo) & (fr < hi)].mean() + 1e-20)


def test_detects_the_switch():
    s = _Stub(_half_cut())
    s._codec_prepare()
    cut = s._codec_cut
    T = len(cut)
    first, second = cut[int(T * 0.1):int(T * 0.4)], cut[int(T * 0.6):int(T * 0.9)]
    assert np.median(first) >= 0.49 * FS                   # 앞은 차단 없음
    assert abs(np.median(second) - 17500.0) < 500.0         # 뒤는 17.5 kHz


def test_mask_removes_the_band_only_where_the_target_was_cut():
    tgt = _half_cut()
    s = _Stub(tgt)
    y = np.random.default_rng(7).standard_normal(len(tgt)).astype(np.float32) * 0.05     # 전대역 렌더
    out = s._codec(torch.as_tensor(y).reshape(1, -1))[0].numpy()
    # 뒤 절반: 18.5~20 kHz 가 30 dB 넘게 빠진다
    assert _band_db(y, 18500, 20000, 0.75, 1.1) - _band_db(out, 18500, 20000, 0.75, 1.1) > 30.0
    # 앞 절반: 18.5~20 kHz 가 그대로 (0.5 dB 안)
    assert abs(_band_db(y, 18500, 20000, 0.1, 0.45) - _band_db(out, 18500, 20000, 0.1, 0.45)) < 0.5
    # 차단 아래 대역은 어디서나 그대로
    assert abs(_band_db(y, 2000, 15000, 0.75, 1.1) - _band_db(out, 2000, 15000, 0.75, 1.1)) < 0.3


def test_no_cut_target_is_identity():
    rng = np.random.default_rng(3)
    tgt = (rng.standard_normal(int(FS * 0.6)) * 0.05).astype(np.float32)
    s = _Stub(tgt)
    y = torch.as_tensor(rng.standard_normal(len(tgt)).astype(np.float32) * 0.05).reshape(1, -1)
    out = s._codec(y)
    assert float((out - y).abs().max()) < 1e-4


def test_gradient_reaches_the_render():
    s = _Stub(_half_cut())
    y = torch.as_tensor(np.random.default_rng(5).standard_normal(int(FS * 1.2)).astype(np.float32) * 0.05)
    y = y.reshape(1, -1).requires_grad_(True)
    s._codec(y).pow(2).sum().backward()
    assert y.grad is not None and float(y.grad.abs().sum()) > 0.0


def test_off_by_default():
    assert F.CODEC_TRACK is False


def test_codec_source_overrides_the_target():
    """차단은 **잡음 제거 전** 신호에서 재야 한다 — 제거기가 고역 바닥을 깎아 없던 차단을 만든다 (§52.206)."""
    tgt = _half_cut(cut_hz=16500.0)                    # 잡음 제거로 더 내려간 것처럼 보이는 목표
    s = _Stub(tgt)
    s.codec_source = torch.as_tensor(_half_cut(cut_hz=17500.0)).reshape(1, -1)   # 원본
    s._codec_prepare()
    second = s._codec_cut[int(len(s._codec_cut) * 0.6):int(len(s._codec_cut) * 0.9)]
    assert abs(np.median(second) - 17500.0) < 500.0     # 목표(16.5 k)가 아니라 원본(17.5 k)을 따른다


def test_codec_source_length_is_matched_to_the_target():
    """목표는 프레임 경계까지 0 으로 채워져 원본보다 길다 — 표본별 무게는 **목표 길이**여야 한다."""
    tgt = np.concatenate([_half_cut(), np.zeros(3000, np.float32)])
    s = _Stub(tgt)
    s.codec_source = torch.as_tensor(_half_cut()).reshape(1, -1)
    w = s._codec_prepare()
    assert w.shape == (len(F.CODEC_ANCHORS), s.target.shape[-1])
    assert torch.allclose(w.sum(0), torch.ones(w.shape[-1]), atol=1e-5)   # 무게 합은 늘 1


def _tone_train(n=int(FS * 1.2), f0=200.0):
    """완전히 주기적인 펄스열 — 배음 구조가 유지되는지 보는 자."""
    x = np.zeros(n, np.float32)
    x[::int(round(FS / f0))] = 1.0
    return x


def test_harmonic_structure_below_the_cut_is_untouched():
    """**§52.212 의 실패를 막는 자**: STFT 마스크는 배음 사이를 채우고 선폭을 넓혔다. FIR 교차감쇠는 그러면 안 된다."""
    s = _Stub(_half_cut())
    x = _tone_train()
    out = s._codec(torch.as_tensor(x).reshape(1, -1))[0].numpy()
    seg = slice(int(0.7 * FS), int(1.1 * FS))            # 차단이 걸린 뒤쪽
    w = np.hanning(seg.stop - seg.start)
    A = np.abs(np.fft.rfft(x[seg] * w)); B = np.abs(np.fft.rfft(out[seg] * w))
    fr = np.fft.rfftfreq(seg.stop - seg.start, 1 / FS)
    m = (fr > 200.0) & (fr < 15000.0)
    d = 20 * np.log10((B[m] + 1e-12) / (A[m] + 1e-12))
    assert abs(d).max() < 0.1, float(abs(d).max())       # 차단 아래는 배음도 배음 사이도 그대로


def test_switch_is_crossfaded_not_blocked():
    """전환 자리에서 블록 경계가 생기면 안 된다 — 표본별 교차감쇠라 국소 에너지가 매끄럽다."""
    s = _Stub(_half_cut())
    x = _tone_train()
    out = s._codec(torch.as_tensor(x).reshape(1, -1))[0].numpy()
    k = 240                                               # 5 ms 창의 국소 에너지
    e = np.convolve(out ** 2, np.ones(k) / k, "same")
    mid = e[int(0.55 * FS):int(0.65 * FS)]
    assert mid.max() / max(mid.min(), 1e-20) < 4.0        # 전환 구간에 계단·구멍이 없다


def test_lowpass_helper_is_identity_at_nyquist():
    h = F._codec_lowpass(FS / 2, FS)
    assert h[len(h) // 2] == 1.0 and abs(h).sum() == 1.0
