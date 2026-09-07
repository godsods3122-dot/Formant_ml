"""녹음의 정지성 잡음 제거 — 복사합성의 전처리.

적합의 목표가 **녹음의 스펙트럼**이므로, 방 잡음이 목표에 섞이면 엔진이 그 잡음까지
만들려고 파라미터를 비튼다. 실제로 겪었다: 실측 /s/ 의 80~600 Hz 는 잡음 바닥 아래인데
켑스트럼 포락선만 보면 "저역 바닥이 필요하다" 로 읽힌다 (docs/MEASUREMENTS.md §6.4).

방법: 정지성 잡음의 파워 스펙트럼을 조용한 프레임에서 추정하고, Wiener 이득
`G = max(floor, (P − α·N)/P)` 를 프레임마다 걸어 겹침가산으로 되돌린다. 이득을 주파수축·
시간축으로 살짝 평활해 '음악적 잡음'(musical noise) 을 줄인다.
"""
from __future__ import annotations

import numpy as np


def noise_profile(y: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512,
                  quantile: float = 0.10) -> np.ndarray:
    """조용한 프레임의 파워 스펙트럼 중앙값 -> 잡음 프로파일 (n_fft//2+1,)."""
    w = np.hanning(n_fft)
    frames = [y[i:i + n_fft] * w for i in range(0, len(y) - n_fft, hop)]
    P = np.abs(np.fft.rfft(np.stack(frames), axis=-1)) ** 2
    e = P.sum(-1)
    idx = np.argsort(e)[:max(3, int(len(e) * quantile))]
    return np.median(P[idx], axis=0)


def denoise(y: np.ndarray, sr: int, noise: np.ndarray | None = None,
            n_fft: int = 2048, hop: int | None = None, over: float = 1.5,
            floor_db: float = -18.0, smooth: int = 3) -> np.ndarray:
    """Wiener 감쇠. `over` 는 과감쇠 계수, `floor_db` 는 이득 하한(음악적 잡음 억제).

    OLA 규약: hop = n_fft/4, 해닝을 분석·합성 양쪽에 걸고 `Σw²` 로 나눈다. **가장자리를
    빼먹으면 안 된다** — 신호를 n_fft 만큼 덧대고 되돌린다. 안 하면 경계에서 Σw²→0 이라
    출력이 폭주하고(측정: 피크 6.2), Praat 이 그 파일에서 F0 를 아예 못 잡는다.
    """
    hop = hop or n_fft // 4
    if noise is None:
        noise = noise_profile(y, sr, n_fft, hop)
    w = np.hanning(n_fft + 1)[:n_fft]
    floor = 10.0 ** (floor_db / 20.0)
    n = len(y)
    pad = n_fft
    yp = np.concatenate([np.zeros(pad), y, np.zeros(pad + n_fft)])
    out = np.zeros_like(yp)
    wsum = np.zeros_like(yp)
    prev = None
    for i in range(0, len(yp) - n_fft, hop):
        X = np.fft.rfft(yp[i:i + n_fft] * w)
        P = np.abs(X) ** 2
        g = np.clip((P - over * noise) / (P + 1e-20), floor ** 2, 1.0) ** 0.5
        if smooth > 1:
            g = np.convolve(g, np.ones(smooth) / smooth, mode="same")
        g = g if prev is None else 0.5 * g + 0.5 * prev
        prev = g
        out[i:i + n_fft] += np.fft.irfft(X * g, n_fft) * w
        wsum[i:i + n_fft] += w ** 2
    out = out / np.maximum(wsum, 1e-6)
    return out[pad:pad + n]


def snr_report(y: np.ndarray, yd: np.ndarray, sr: int, noise: np.ndarray) -> dict:
    """제거 전후의 잡음 바닥 (조용한 구간의 대역 레벨)."""
    def floor_db(x):
        p = noise_profile(x, sr)
        return float(10 * np.log10(p.sum() + 1e-20))
    def peak_db(x):
        return float(20 * np.log10(np.abs(x).max() + 1e-20))
    a, b = floor_db(y), floor_db(yd)
    return dict(noise_db_in=a, noise_db_out=b, removed_db=a - b,
                peak_db_in=peak_db(y), peak_db_out=peak_db(yd),
                # 옛 이름 (호출부 호환)
                before=a, after=b)
