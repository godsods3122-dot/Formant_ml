"""녹음 속 **발화가 아닌 클릭**(마우스 등)을 찾아 메운다 (MEASUREMENTS §52.461).

사용자: *"'보' 할 때 마우스 클릭 소리가 나면서 그걸 노이즈로 인지하게 되고, 그게 뒷 발화에서 오염된 걸로 나타나네"* — `yang_00000079` 1.2085 s 의
광대역 과도음을 적합기가 목소리 기구(주기별 보정·기식)로 흉내 냈고, 1.129 s 의 또 하나는 분석이 **파열**로 읽어 폐쇄·개방 관측을 엉뚱한 자리에 걸었다.

**검출** — 6~20 kHz 대역의 0.25 ms 포락 e(t) [dB] 에서, 국소 바닥 b(t) = ±`BASE_MS` 중앙값 대비
  * 솟음 e − b ≥ `RISE_DB` (성문 폐쇄의 고역 터짐은 이 화자 모음에서 +8~11 dB, 클릭 +13~19 dB 로 쟀다),
  * **짧다**: 정점 뒤 `DECAY_MS` 안에 b + `BACK_DB` 아래로 돌아온다 (파열은 마찰·기식이 이어져 안 돌아온다),
  * **날카롭게 선다**: 정점 앞 `ONSET_MS` 에는 b + `BACK_DB` 아래였다,
  * **말소리 개시·파열이 아니다**: 사건 직후 3~8 ms 의 0.3~6 kHz 수준이 직전 −6~−1 ms 보다 `ONSET_DB` 넘게 오르지 않는다 (파열은 폐쇄 → 기식·모음으로
    오르고, 클릭은 앞뒤가 같다 — 모음 한가운데든 무음이든),
  * **성문 고역 터짐이 아니다**: 유성 주기 안이면 그 주기의 4~16 kHz 파형이 이웃 ±2 주기 중앙값에서 `CYCLE_RES_MIN` 넘게 벗어난다 (성문 터짐은
    주기마다 되풀이된다). 처음 판의 "±12 ms 안 다른 솟음보다 6 dB 크다" 는 모음 개시 자리의 클릭(079 1.2085 s)을 성문 터짐과 못 갈랐다.
**메우기** — 솟음 구간(± `PAD_MS`)의 `SPLIT_HZ` 위 성분만(저역은 그대로) 앞뒤 `CTX_MS` 문맥으로 맞춘 AR(`AR_ORDER`) 모형의 최소 제곱 보간(LSAR; Janssen et al. 1986, Vaseghi)으로
바꾼다. 표본의 자리는 그대로라 **시간 원점이 바뀌지 않는다**(§52.431). 길이도 같다.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

RISE_DB = 12.0
BACK_DB = 6.0
BASE_MS = 10.0
DECAY_MS = 4.0
ONSET_MS = 2.0
ONSET_DB = 8.0          # 사건 직후 중역이 직전보다 이만큼 오르면 말소리 개시·파열로 본다
CYCLE_RES_MIN = 1.0     # 유성 주기 고역 잔차 / 이웃 힘 — 이보다 작으면 되풀이되는 성문 터짐
PAD_MS = 0.5
CTX_MS = 20.0
AR_ORDER = 40
MAX_SPAN_MS = 6.0
SPLIT_HZ = 1500.0        # 이 위만 메운다
XFADE_MS = 0.25


def _env_db(x: np.ndarray, sr: int, hop: int) -> np.ndarray:
    hi = min(20000.0, 0.45 * sr)
    y = sosfiltfilt(butter(4, (6000.0, hi), "bp", fs=sr, output="sos"), x)
    w = max(1, hop)
    e = np.convolve(y * y, np.ones(w) / w, mode="same")[::hop]
    return 10.0 * np.log10(e + 1e-20)


def _band_db(x, sr, lo, hi, a, b):
    a, b = max(0, a), min(len(x), b)
    if b - a < 8:
        return -200.0
    return 10.0 * np.log10(np.mean(x[a:b] ** 2) + 1e-20)


def _cycle_residual(h: np.ndarray, pulses: np.ndarray, t: float, sr: int, m: int = 128) -> float | None:
    """t 가 든 성문 주기의 고역 파형이 이웃 주기(±2)의 중앙값에서 얼마나 벗어나나 — 잔차 힘 / 이웃 힘. 유성 무리 밖이면 None."""
    j = int(np.searchsorted(pulses, t) - 1)
    if j < 2 or j + 3 >= len(pulses):
        return None
    d = np.diff(pulses[j - 2:j + 4])
    if d.max() > 1.8 * np.median(d) or d.max() > 0.02:
        return None

    def cyc(q):
        i0, i1 = int(pulses[q] * sr), int(pulses[q + 1] * sr)
        c = h[i0:i1]
        return np.interp(np.linspace(0, len(c), m, endpoint=False), np.arange(len(c)), c)
    me = cyc(j)
    nb = np.array([cyc(q) for q in (j - 2, j - 1, j + 1, j + 2)])
    pred = np.median(nb, 0)
    e_nb = np.mean(nb ** 2)
    return float(np.mean((me - pred) ** 2) / max(e_nb, 1e-20))


def detect(x: np.ndarray, sr: int, pulses: np.ndarray | None = None) -> list[tuple[int, int, float]]:
    """클릭 [(시작 표본, 끝 표본, 솟음 dB)]. 끝은 포함하지 않는다. pulses: 성문 펄스 [s] (있으면 유성 주기 잔차로 성문 고역 터짐을 뺀다)."""
    x = np.asarray(x, dtype=np.float64)
    hop = max(1, int(round(0.00025 * sr)))
    e = _env_db(x, sr, hop)
    ms = 1000.0 * hop / sr
    b = median_filter(e, size=int(2 * BASE_MS / ms) + 1, mode="nearest")
    d = e - b
    mid = sosfiltfilt(butter(4, (300.0, 6000.0), "bp", fs=sr, output="sos"), x)
    hiband = sosfiltfilt(butter(4, (4000.0, min(16000.0, 0.45 * sr)), "bp", fs=sr, output="sos"), x)
    out = []
    k = 1
    n = len(e)
    on, dec = int(ONSET_MS / ms), int(DECAY_MS / ms)
    while k < n - 1:
        if not (d[k] >= RISE_DB and d[k] >= d[k - 1] and d[k] >= d[k + 1]):
            k += 1
            continue
        pre_ok = k - on >= 0 and d[k - on] < BACK_DB
        j = k
        while j < min(n, k + dec) and d[j] >= BACK_DB:
            j += 1
        post_ok = j < min(n, k + dec)
        a = k
        while a > 0 and d[a - 1] >= BACK_DB:
            a -= 1
        tk = k * hop
        # 말소리 개시·파열 개방이 아니다 — 사건 직후 3~8 ms 의 중역이 직전 −6~−1 ms 보다 ONSET_DB 넘게 오르지 않는다
        rise = (_band_db(mid, sr, 0, 0, tk + int(0.003 * sr), tk + int(0.008 * sr))
                - _band_db(mid, sr, 0, 0, tk - int(0.006 * sr), tk - int(0.001 * sr)))
        onset_ok = rise < ONSET_DB
        # 성문 고역 터짐이 아니다 — 유성 주기면 이웃 주기와 모양이 달라야 한다
        per_ok = True
        if pulses is not None and len(pulses) > 5:
            r = _cycle_residual(hiband, np.asarray(pulses, float), tk / sr, sr)
            if r is not None:
                per_ok = r >= CYCLE_RES_MIN
        if pre_ok and post_ok and onset_ok and per_ok:
            s0 = int((a * hop) - PAD_MS * sr / 1000.0)
            s1 = int((j * hop) + PAD_MS * sr / 1000.0)
            if (s1 - s0) <= MAX_SPAN_MS * sr / 1000.0:
                out.append((max(0, s0), min(len(x), s1), float(d[k])))
            k = j + 1
            continue
        k += 1
    return out


def _lsar(seg: np.ndarray, miss: slice, order: int) -> np.ndarray:
    """seg 안의 miss 구간을 AR 최소 제곱 보간으로 채운 값."""
    known = np.ones(len(seg), dtype=bool)
    known[miss] = False
    ctx = seg.copy()
    ctx[~known] = 0.0
    # 자기상관은 알려진 부분에서만 (앞·뒤 문맥 각각)
    r = np.zeros(order + 1)
    for part in (seg[:miss.start], seg[miss.stop:]):
        if len(part) > order + 1:
            p = part * np.hanning(len(part))
            full = np.correlate(p, p, mode="full")[len(p) - 1:len(p) + order]
            r[:len(full)] += full
    if r[0] <= 0:
        return seg[miss].copy()
    r[0] *= 1.0 + 1e-6
    a = solve_toeplitz(r[:order], -r[1:order + 1])
    coef = np.concatenate(([1.0], a))                     # e[n] = Σ coef_i x[n−i]
    n = len(seg)
    rows = n - order
    A = np.zeros((rows, n))
    for i, c in enumerate(coef):
        A[np.arange(rows), np.arange(rows) + order - i] = c
    Au, Ak = A[:, ~known], A[:, known]
    xu, *_ = np.linalg.lstsq(Au, -Ak @ seg[known], rcond=None)
    return xu


def _periodic_pred(h: np.ndarray, pulses: np.ndarray, n0: int, n1: int, sr: int) -> np.ndarray | None:
    """표본 n0..n1 의 고역을 이웃 ±2 성문 주기의 같은 주기 안 자리 값의 중앙값으로 예측한다. 유성 무리 밖이면 None."""
    t = np.arange(n0, n1) / sr
    js = np.searchsorted(pulses, t) - 1
    if js.min() < 2 or js.max() + 3 >= len(pulses):
        return None
    d = np.diff(pulses[js.min() - 2:js.max() + 4])
    if d.max() > 1.8 * np.median(d) or d.max() > 0.02:
        return None
    out = np.zeros(len(t))
    for i, (tt, j) in enumerate(zip(t, js)):
        phi = (tt - pulses[j]) / (pulses[j + 1] - pulses[j])
        vals = []
        for q in (j - 2, j - 1, j + 1, j + 2):
            tq = pulses[q] + phi * (pulses[q + 1] - pulses[q])
            vals.append(np.interp(tq * sr, np.arange(len(h)), h))
        out[i] = np.median(vals)
    return out


def repair(x: np.ndarray, sr: int, clicks, pulses: np.ndarray | None = None) -> np.ndarray:
    """클릭 구간의 **`SPLIT_HZ` 위 성분만** 바꾼다 — 저역은 그대로 둔다.

    유성 주기 안이면 **이웃 ±2 주기의 같은 자리 값의 중앙값**(목소리는 주기적, 클릭은 아니다)으로, 아니면 AR 보간(LSAR)으로. 경계는 `XFADE_MS`
    올림 코사인으로 잇는다. 눈금(§52.461): 전 대역 AR 은 모음 안에서 클릭 오차 −47 dB 를 −33 dB 로 **키웠고**, 고역만 AR 도 모음 한가운데
    (1.305 s)에서는 −47 → −42 dB 로 나빴다 — 3~4 ms 의 배음을 AR 이 못 잇는다.
    """
    x = np.asarray(x, dtype=np.float64)
    hp = sosfiltfilt(butter(4, SPLIT_HZ, "hp", fs=sr, output="sos"), x)
    lp = x - hp
    h = hp.copy()
    ctx = int(CTX_MS * sr / 1000.0)
    nx = max(1, int(XFADE_MS * sr / 1000.0))
    for s0, s1, _ in clicks:
        a0, b0 = max(0, s0 - nx), min(len(h), s1 + nx)
        pred = _periodic_pred(hp, np.asarray(pulses, float), a0, b0, sr) if pulses is not None and len(pulses) > 5 else None
        if pred is None:
            a, b = max(0, a0 - ctx), min(len(h), b0 + ctx)
            pred = _lsar(hp[a:b], slice(a0 - a, b0 - a), AR_ORDER)
        w = np.ones(b0 - a0)
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, nx))
        w[:nx] = ramp
        w[-nx:] = ramp[::-1]
        h[a0:b0] = (1.0 - w) * hp[a0:b0] + w * pred
    return lp + h


def declick(x: np.ndarray, sr: int, pulses: np.ndarray | None = None, log=None, at=None, win_ms: float = 5.0):
    """(메운 신호, 클릭 목록). log 가 있으면 한 줄 찍는다.

    at: 확인된 클릭 시각 [s] 목록 — 주면 그 둘레 ±win_ms 안의 검출만 쓴다. 자동 검출은 아직 creak(불규칙 펄스)과 일부 파열을 클릭으로
    잡는다(§52.461) — 코퍼스 전체에는 `at` 로만 쓴다.
    """
    c = detect(x, sr, pulses)
    if at is not None:
        at = np.asarray(at, float)
        c = [q for q in c if at.size and np.min(np.abs(0.5 * (q[0] + q[1]) / sr - at)) <= win_ms / 1000.0]
    y = repair(x, sr, c, pulses) if c else np.asarray(x, dtype=np.float64)
    if log is not None:
        log(f"  클릭 제거: {len(c)} 곳" + ("" if not c else " — " + ", ".join(
            f"{s0 / sr:.4f} s ({(s1 - s0) * 1000.0 / sr:.1f} ms, +{dd:.0f} dB)" for s0, s1, dd in c)))
    return y, c
