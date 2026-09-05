"""프레임별 포먼트 추적 — 복사합성용.

`extract.lpc_formants` 는 화자 **평균** 포먼트를 낸다(프로파일 추출용).
복사합성은 프레임마다 다른 값이 필요하고, 무엇보다 **연속성**이 필요하다.
매 프레임 독립으로 근을 뽑아 정렬만 하면 포먼트가 서로 자리를 바꿔 가며
튀고, 그 궤적으로 합성하면 사람 소리가 아니라 기계음이 된다.

여기서는 (1) LPC 근을 뽑고 (2) 직전 프레임의 궤적에 **가까운 것부터 배정**하고
(3) 빈 곳을 보간한 뒤 (4) 가볍게 중앙값 평활한다.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def _lpc(x: np.ndarray, order: int):
    r = np.correlate(x, x, "full")[len(x) - 1:][: order + 1]
    if r[0] <= 0:
        return None
    r = r / r[0]
    r[0] += 1e-6
    a = np.zeros(order + 1)
    a[0], e = 1.0, r[0]
    for i in range(1, order + 1):
        acc = r[i] + (np.dot(a[1:i], r[i - 1:0:-1]) if i > 1 else 0.0)
        k = -acc / e
        a[1:i + 1] = a[1:i + 1] + k * a[i - 1::-1][:i]
        e *= (1.0 - k * k)
        if e <= 0:
            return None
    return a


def _roots(frame: np.ndarray, sr: int, order: int, fmin: float, fmax: float,
           max_bw: float):
    a = _lpc(frame, order)
    if a is None:
        return np.empty(0), np.empty(0)
    z = np.roots(a)
    z = z[np.imag(z) > 0]
    if not len(z):
        return np.empty(0), np.empty(0)
    f = np.angle(z) * sr / (2 * np.pi)
    bw = -(sr / np.pi) * np.log(np.abs(z) + 1e-12)
    k = (f > fmin) & (f < fmax) & (bw < max_bw) & (bw > 0)
    f, bw = f[k], bw[k]
    o = np.argsort(f)
    return f[o], bw[o]


def track_formants(y: np.ndarray, sr: int = 24000, hop: int = 240,
                   win: int = 1024, n: int = 6, order: int | None = None,
                   fmin: float = 120.0, fmax: float = 9000.0,
                   max_bw: float = 900.0, preemph: float = 0.97,
                   jump_hz: float = 350.0):
    """(T, n) 포먼트 [Hz] 와 (T, n) 대역폭 [Hz]. 연속 궤적으로 이어 붙인다.

    `jump_hz` 보다 멀리 뛰는 근은 그 궤적의 후보로 보지 않는다 — 이것이
    포먼트끼리 자리를 바꾸는 것을 막는다.
    """
    order = order or int(2 + sr / 1000)
    x = lfilter([1.0, -preemph], [1.0], y)
    t = max(0, 1 + (len(x) - win) // hop)
    w = np.hanning(win)
    F = np.full((t, n), np.nan)
    B = np.full((t, n), np.nan)
    prev = None
    for i in range(t):
        f, bw = _roots(x[i * hop: i * hop + win] * w, sr, order, fmin, fmax,
                       max_bw)
        if not len(f):
            continue
        if prev is None:
            m = min(n, len(f))
            F[i, :m], B[i, :m] = f[:m], bw[:m]
        else:
            used = np.zeros(len(f), bool)
            for s in range(n):
                if np.isnan(prev[s]):
                    continue
                d = np.abs(f - prev[s])
                d[used] = np.inf
                j = int(np.argmin(d))
                if d[j] < jump_hz:
                    F[i, s], B[i, s] = f[j], bw[j]
                    used[j] = True
            # 남은 근을 빈 슬롯에 주파수 순서를 지켜 채운다
            free = [s for s in range(n) if np.isnan(F[i, s])]
            for s, j in zip(free, np.where(~used)[0]):
                F[i, s], B[i, s] = f[j], bw[j]
        row = F[i]
        prev = np.where(np.isnan(row), prev if prev is not None else row, row)
    F, B = _fill(F, B, sr)
    # 포먼트는 물리적으로 서로 교차하지 않는다. 추적 후 정렬로 남은 튐을 없앤다
    # (에너지가 죽는 끝단에서 근이 고역으로 튀는 것을 실제로 겪었다).
    o = np.argsort(F, axis=1)
    return np.take_along_axis(F, o, 1), np.take_along_axis(B, o, 1)


def _fill(F: np.ndarray, B: np.ndarray, sr: int):
    """결측을 메우고 가볍게 평활한다.

    **빈 슬롯을 0 으로 두면 안 된다.** 합성기가 f_min 으로 클램프해서 저역에
    가짜 공명기를 만들고, 그 하나하나가 -12 dB/oct 씩 감쇠를 더한다. 실제로
    빈 슬롯 4 개가 150 Hz 짜리 유령 극 4 개가 되어 고역을 40 dB 죽였다.
    못 찾은 슬롯은 **나이퀴스트 근처에 넓은 대역폭**으로 둬서 무해하게 만든다.
    """
    F, B = F.copy(), B.copy()
    t, n = F.shape
    idx = np.arange(t)
    dead = sr * 0.45
    for s in range(n):
        for A, spare in ((F, dead), (B, 1200.0)):
            col = A[:, s]
            ok = ~np.isnan(col)
            if ok.sum() == 0:
                col[:] = spare          # 무해한 극: 아주 높고 아주 넓다
            elif ok.sum() < t:
                col[:] = np.interp(idx, idx[ok], col[ok])
            if t >= 3:
                A[1:-1, s] = np.median(
                    np.stack([col[:-2], col[1:-1], col[2:]]), axis=0)
    return F, B
