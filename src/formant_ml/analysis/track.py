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

    **가장 신뢰할 만한 프레임에서 씨를 뿌리고 양방향으로 추적한다.**
    첫 프레임부터 앞으로만 가면 무음 구간에서 잘못 잡힌 궤적이 끝까지 이어진다
    — 실제로 슬롯 5 가 무음의 8146 Hz 를 물고 있어서, 모음 구간의 실재 극
    5118 Hz 가 들어갈 자리를 잃고 통째로 버려졌다(고역이 20 dB 넘게 죽었다).

    한 프레임 안의 배정은 (1) 직전 궤적에 `jump_hz` 안으로 가장 가까운 근을
    순서를 지켜 이어 붙이고 (2) 남은 근은 **앵커 사이의 빈 자리에만** 넣는다.
    아무렇게나 채우면 한 궤적이 빠졌을 때 나머지가 한 칸씩 밀려 들어온다
    (F1 이 한 프레임 빠지자 F2 가 슬롯 0 으로 와서 출력이 11 dB 튀었다).
    """
    order = order or int(2 + sr / 1000)
    x = lfilter([1.0, -preemph], [1.0], y)
    t = max(0, 1 + (len(x) - hop * 0 - win) // hop)
    if t == 0:
        return np.zeros((0, n)), np.zeros((0, n))
    w = np.hanning(win)
    cand = [_roots(x[i * hop: i * hop + win] * w, sr, order, fmin, fmax, max_bw)
            for i in range(t)]
    energy = np.array([np.sqrt((y[i * hop: i * hop + win] ** 2).mean())
                       for i in range(t)])

    F = np.full((t, n), np.nan)
    B = np.full((t, n), np.nan)
    # 씨 프레임은 **근이 가장 많이 잡힌 곳**으로 고른다(동수면 에너지가 큰 쪽).
    # 에너지만 보고 고르면 그 프레임이 놓친 극의 슬롯이 통째로 밀려서, 이후
    # 모든 프레임이 그 밀림을 물려받는다(3318 Hz 가 사라지고 고역이 어두워졌다).
    nroots = np.array([len(c[0]) for c in cand])
    score = nroots * 1000.0 + energy / max(energy.max(), 1e-12)
    seed = int(np.argmax(score))
    f0, b0 = cand[seed]
    m = min(n, len(f0))
    F[seed, :m], B[seed, :m] = f0[:m], b0[:m]

    for direction in (1, -1):
        prev = F[seed].copy()
        i = seed + direction
        while 0 <= i < t:
            f, bw = cand[i]
            if len(f):
                _assign(F, B, i, f, bw, prev, n, jump_hz)
                row = F[i]
                prev = np.where(np.isnan(row), prev, row)
            i += direction
    return _fill(F, B, sr)


def _assign(F, B, i, f, bw, prev, n, jump_hz):
    """한 프레임의 근을 궤적에 배정한다 (순서 보존)."""
    used = np.zeros(len(f), bool)
    ptr = 0
    for s in range(n):
        p = prev[s]
        if np.isnan(p):
            continue
        while ptr < len(f) and f[ptr] < p - jump_hz:
            ptr += 1
        if ptr < len(f) and abs(f[ptr] - p) <= jump_hz:
            F[i, s], B[i, s] = f[ptr], bw[ptr]
            used[ptr] = True
            ptr += 1
    for j in np.where(~used)[0]:
        c = f[j]
        for s in range(n):
            if not np.isnan(F[i, s]):
                continue
            below = [F[i, k] for k in range(s) if not np.isnan(F[i, k])]
            above = [F[i, k] for k in range(s + 1, n) if not np.isnan(F[i, k])]
            if (not below or below[-1] < c) and (not above or above[0] > c):
                F[i, s], B[i, s] = c, bw[j]
                break


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
