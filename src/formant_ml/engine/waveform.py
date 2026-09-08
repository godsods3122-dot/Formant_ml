"""파형 일치도 — 스펙트럼이 아니라 **파형 자체**가 맞는지.

왜 따로 재는가
--------------
크기 스펙트럼이 맞아도 위상이 틀리면 파형은 다르다. 이 프로젝트의 전제가 "기존 TTS 는
스펙트럼만 맞추고 위상을 망가뜨린다" 이므로, 위상을 직접 재는 자를 갖고 있어야 한다.

무엇을 재고 무엇을 안 재는가
----------------------------
**난류는 파형으로 비교할 수 없다.** 마찰 잡음의 표본 하나하나는 미시적 소용돌이가
정하고, 완벽한 물리 모형이라도 그 실현을 재현하지 못한다. 재현할 수 있는 것은 그
통계(포락·스펙트럼)뿐이다. 이건 치찰음만의 이야기가 아니다 — 모음 속 기식 잡음도
같다. 그래서 신호를 **조화 성분과 잔차**로 가르고,

* 조화 성분: 파형으로 비교한다 (분절 SNR, 하모닉별 위상 오차)
* 잔차 성분: 통계로 비교한다 (대역 포락)

조화 모형
---------
프레임마다 국소 F0 를 알고 있으므로(성문 펄스에서 나온다) 3 주기 창에서
`x(n) ≈ Σ_k a_k cos(2πk f0 n/fs) + b_k sin(...)` 를 최소제곱으로 푼다. 이것이 조화
모형(harmonic model)이고, 잔차는 그 나머지다.

위상 오차에서 **하모닉 차수에 비례하는 성분은 빼고** 본다. 그건 단순한 시간 이동이라
파형의 모양과 무관하다. 남는 것이 곧 **필터의 위상 응답 오차** — v2 가 시변 IIR 로
잡으려던 바로 그 양이다.
"""
from __future__ import annotations

import numpy as np


def harmonic_basis(n: int, f0: float, fs: float, k_max: int) -> np.ndarray:
    """(n, 2K) 최소제곱 기저 [cos k, sin k]."""
    t = (np.arange(n) - n / 2.0) / fs
    k = np.arange(1, k_max + 1)[:, None]
    w = 2 * np.pi * f0 * k * t[None, :]
    return np.concatenate([np.cos(w), np.sin(w)], axis=0).T


def harmonic_fit(x: np.ndarray, f0: float, fs: float, k_max: int | None = None,
                 ridge: float = 1e-6):
    """한 프레임의 조화 모형. -> (복소 진폭 (K,), 조화 파형 (n,))."""
    n = len(x)
    k_max = k_max or max(1, int(0.45 * fs / max(f0, 1.0)))
    k_max = min(k_max, max(1, n // 4))          # 조건수: 미지수 2K < n/2
    B = harmonic_basis(n, f0, fs, k_max)
    G = B.T @ B
    G[np.diag_indices_from(G)] += ridge * np.trace(G) / len(G)
    c = np.linalg.solve(G, B.T @ x)
    a, b = c[:k_max], c[k_max:]
    return a - 1j * b, B @ c


def decompose(y: np.ndarray, fs: float, f0: np.ndarray, hop: int,
              periods: float = 3.0):
    """(조화 진폭 (T,K), 조화 파형 (N,), 잔차 (N,)).

    창은 **주기의 정수배**로 잡는다 — 그래야 하모닉이 서로 직교에 가까워 최소제곱이
    안정하다. 겹침가산은 Hann 으로 하고 창 합으로 나눈다.
    """
    n = len(y)
    t = len(f0)
    k_max = max(1, int(0.45 * fs / max(float(np.min(f0[f0 > 0])) if (f0 > 0).any() else 100.0, 1.0)))
    amps = np.zeros((t, k_max), dtype=complex)
    har = np.zeros(n)
    wsum = np.zeros(n)
    for i in range(t):
        f = float(f0[i])
        if not np.isfinite(f) or f <= 0:
            continue
        w = int(round(periods * fs / f)) // 2 * 2
        c = i * hop + hop // 2
        a, b = c - w // 2, c + w // 2
        if a < 0 or b > n or w < 16:
            continue
        seg = y[a:b]
        amp, fit = harmonic_fit(seg, f, fs, k_max)
        amps[i, :len(amp)] = amp
        win = np.hanning(w)
        har[a:b] += fit * win
        wsum[a:b] += win
    har = np.where(wsum > 1e-9, har / np.maximum(wsum, 1e-9), 0.0)
    return amps, har, y - har


def seg_snr(target: np.ndarray, synth: np.ndarray, fs: float, win_ms: float = 20.0,
            floor_db: float = -20.0, mask: np.ndarray | None = None) -> float:
    """분절 SNR (dB). 프레임마다 10log10(Σt²/Σ(t−s)²) 를 재고 평균낸다.

    프레임별로 재는 이유: 전체 한 번이면 큰 구간이 지배해서 조용한 구간의 어긋남이
    안 보인다. 아래 바닥(floor_db)을 두는 것은 무음 구간의 −∞ 가 평균을 삼키기 때문.
    """
    w = int(fs * win_ms / 1000.0)
    n = min(len(target), len(synth)) // w * w
    if n == 0:
        return float("nan")
    a = target[:n].reshape(-1, w)
    b = synth[:n].reshape(-1, w)
    num = (a ** 2).sum(1)
    den = ((a - b) ** 2).sum(1) + 1e-20
    keep = num > num.max() * 1e-6
    if mask is not None:
        m = np.asarray(mask)[:len(num)] if len(mask) >= len(num) else \
            np.pad(np.asarray(mask), (0, len(num) - len(mask)))
        keep = keep & m.astype(bool)
    if not keep.any():
        return float("nan")
    return float(np.maximum(10 * np.log10(num[keep] / den[keep]), floor_db).mean())


def phase_error(at: np.ndarray, as_: np.ndarray, min_rel_db: float = -30.0):
    """하모닉별 위상 오차 (도). -> (모양 오차, 시간 이동 성분 ms 상당의 기울기, 쓴 하모닉 수)

    차수에 비례하는 성분(= 순수한 시간 이동)은 가중 최소제곱으로 빼고 본다. 남는 것이
    파형의 **모양**을 바꾸는 오차이고, 그게 필터 위상 응답의 오차다.
    """
    keep = np.abs(at) > np.abs(at).max() * 10 ** (min_rel_db / 20.0)
    keep &= np.abs(as_) > 0
    k = np.arange(1, len(at) + 1)[keep]
    if len(k) < 3:
        return float("nan"), float("nan"), 0
    d = np.angle(at[keep] * np.conj(as_[keep]))
    d = np.unwrap(d)
    wgt = np.abs(at[keep])
    # d ≈ slope·k + offset  (slope 가 시간 이동)
    A = np.stack([k, np.ones_like(k)], 1) * wgt[:, None]
    sol, *_ = np.linalg.lstsq(A, d * wgt, rcond=None)
    resid = d - (sol[0] * k + sol[1])
    rms = float(np.sqrt((wgt * resid ** 2).sum() / wgt.sum()))
    return float(np.degrees(rms)), float(np.degrees(sol[0])), int(len(k))


def report(target: np.ndarray, synth: np.ndarray, fs: float, f0: np.ndarray,
           hop: int, voiced: np.ndarray | None = None) -> dict:
    """파형 일치도 한 묶음. 조화 성분은 파형으로, 잔차는 통계로."""
    n = min(len(target), len(synth))
    target, synth = target[:n], synth[:n]
    at, ht, rt = decompose(target, fs, f0, hop)
    as_, hs, rs = decompose(synth, fs, f0, hop)
    ph, sl, nk = [], [], []
    t = min(at.shape[0], as_.shape[0])
    for i in range(t):
        if voiced is not None and i < len(voiced) and not voiced[i]:
            continue
        if not np.any(at[i]):
            continue
        p, s, k = phase_error(at[i], as_[i])
        if np.isfinite(p):
            ph.append(p); sl.append(s); nk.append(k)
    return dict(
        snr_full=seg_snr(target, synth, fs),
        snr_harmonic=seg_snr(ht, hs, fs),
        harmonic_frac_t=float((ht ** 2).sum() / max((target ** 2).sum(), 1e-20)),
        harmonic_frac_s=float((hs ** 2).sum() / max((synth ** 2).sum(), 1e-20)),
        residual_level_db=float(10 * np.log10(((rs ** 2).mean() + 1e-20)
                                              / ((rt ** 2).mean() + 1e-20))),
        phase_shape_deg=float(np.mean(ph)) if ph else float("nan"),
        phase_shift_deg_per_harmonic=float(np.mean(sl)) if sl else float("nan"),
        n_frames=len(ph),
        n_harmonics=float(np.mean(nk)) if nk else 0.0,
    )
