"""파형만 보고 재는 음성 측정 — 녹음과 합성음에 똑같이 쓴다.

합성 내부(접촉 횟수, 간극, 면적함수)에 접근하지 않는다. 그래야 실제 녹음과
같은 잣대로 비교되고, "기계는 설계대로 도는데 소리는 사람 소리가 아니다" 를
잡을 수 있다. 기존 테스트 114 개가 전부 내부만 봐서 놓친 것이 이것이다.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.signal import lfilter, resample_poly

ANALYSIS_SR = 16000          # 포먼트 추적 표준 (F1~F4 가 8 kHz 안에 든다)


def load(path: str, target_sr: int = ANALYSIS_SR):
    y, sr = sf.read(path, always_2d=True)
    y = y.mean(axis=1).astype(np.float64)
    if sr != target_sr:
        g = np.gcd(int(sr), int(target_sr))
        y = resample_poly(y, target_sr // g, sr // g)
    return y, target_sr


def frames(y, sr, win_ms=25.0, hop_ms=10.0):
    w, h = int(win_ms * sr / 1000), int(hop_ms * sr / 1000)
    n = max(0, 1 + (len(y) - w) // h)
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    return y[idx] * np.hanning(w)[None, :], h


def lpc(x, order):
    """자기상관 + Levinson-Durbin."""
    r = np.correlate(x, x, "full")[len(x) - 1:][: order + 1]
    if r[0] <= 0:
        return None
    a = np.zeros(order + 1)
    a[0], e = 1.0, r[0]
    for i in range(1, order + 1):
        acc = r[i] + np.dot(a[1:i], r[i - 1:0:-1]) if i > 1 else r[i]
        k = -acc / e
        a[1:i + 1] = a[1:i + 1] + k * a[i - 1::-1][:i]
        e *= (1.0 - k * k)
        if e <= 0:
            return None
    return a


def formants(y, sr, order=None, fmin=200.0, fmax=5000.0, bwmax=500.0, n=4):
    """프레임별 포먼트 (F, BW). 반환 (T, n) Hz, NaN = 미검출."""
    order = order or (2 + sr // 1000)
    pre = lfilter([1.0, -0.97], [1.0], y)
    fr, _ = frames(pre, sr)
    out = np.full((len(fr), n), np.nan)
    for t, f in enumerate(fr):
        a = lpc(f, order)
        if a is None:
            continue
        rts = np.roots(a)
        rts = rts[np.imag(rts) > 0]
        if not len(rts):
            continue
        hz = np.angle(rts) * sr / (2 * np.pi)
        bw = -0.5 * (sr / (2 * np.pi)) * np.log(np.abs(rts) + 1e-12)
        keep = (hz > fmin) & (hz < fmax) & (bw < bwmax)
        hz = np.sort(hz[keep])
        out[t, : min(n, len(hz))] = hz[:n]
    return out


def energy_db(y, sr, hop_ms=10.0, win_ms=25.0):
    fr, _ = frames(y, sr, win_ms, hop_ms)
    return 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)


def band_db(y, sr, lo, hi, hop_ms=10.0, win_ms=25.0):
    """대역 에너지 [dB] 프레임별."""
    fr, _ = frames(y, sr, win_ms, hop_ms)
    S = np.abs(np.fft.rfft(fr, 1024, axis=1))
    f = np.linspace(0, sr / 2, S.shape[1])
    m = (f >= lo) & (f < hi)
    return 20 * np.log10(np.sqrt((S[:, m] ** 2).mean(1)) + 1e-9)


def voicing(y, sr, hop_ms=10.0, win_ms=25.0, fmin=60.0, fmax=400.0):
    """프레임별 주기성(0~1)과 F0. 자기상관 최대치."""
    fr, _ = frames(y, sr, win_ms, hop_ms)
    lo, hi = int(sr / fmax), int(sr / fmin)
    per, f0 = np.zeros(len(fr)), np.zeros(len(fr))
    for t, f in enumerate(fr):
        f = f - f.mean()
        if f.std() < 1e-6:
            continue
        ac = np.correlate(f, f, "full")[len(f) - 1:]
        ac = ac / (ac[0] + 1e-12)
        seg = ac[lo:min(hi, len(ac))]
        if not len(seg):
            continue
        k = int(seg.argmax())
        per[t], f0[t] = max(seg[k], 0.0), sr / (lo + k)
    return per, f0


