"""목표 녹음의 **늦은 잔향**을 뺀다 — 마른 모형이 마른 목표를 보게.

왜 필요한가 (docs/MEASUREMENTS.md §19)
--------------------------------------
목표는 방에서 잡은 소리이고 엔진은 마르다(무향). 방은 마찰음의 포락선을 **매끄럽게**
만드는데, 그 매끄러움은 **정적 필터로 흉내 낼 수 없다** — 시간 구조라서다. 그래서
적합기는 방의 *크기 스펙트럼*만 성도·소스 파라미터로 흡수하고(대역 MAE 0.25 dB),
포락선의 매끄러움은 못 맞춘 채로 남긴다. 그 남은 양이 사람이 "지지직" 으로 듣는
것이고, 동시에 **흡수한 만큼 물리 파라미터가 왜곡돼 있다**는 뜻이다.

증거: 4~12 kHz 포락의 60~400 Hz 변조 비율이 목표 13.6 %, 순수 가우시안 대조군
17.8 %, 마른 합성 27.4 %. **목표가 대조군보다 낮다** — 마른 난류는 원리적으로
가우시안보다 매끄러울 수 없으므로 그 매끄러움은 경로가 만든 것이다.

그러면 방을 순방향 모형에 넣으면 되지 않는가
---------------------------------------------
**안 된다 — 진짜 IR 이 없으면.** 합성 IR(측정한 RT60 을 갖는 지수감쇠 잡음)을
순방향에 넣으면 크기는 비슷해져도 **초기 반사의 위상이 목표와 다르다.** 이 프로젝트는
파형·위상을 맞추는 것이 전제라(`docs/` 여러 곳) 그 손해가 이득보다 크다. 그래서
반대로 **목표에서 빼는** 쪽을 택한다. 늦은 잔향의 빼기는 크기 스펙트럼에서만
일어나므로 직접음의 위상은 건드리지 않는다.

방법
----
Lebart·Boucher·Denbigh (2001) / Habets (2007) 의 통계 모형. 방의 에너지 감쇠가
지수라면, 시각 n 의 **늦은** 잔향 전력은 Δ 프레임 전의 총 전력에

    α = exp(−2·δ·Δ·hop/fs),   δ = 3·ln 10 / RT60

를 곱한 것과 같다. 그것을 크기 스펙트럼에서 빼고 바닥을 둔다. Δ 는 직접음과
초기 반사를 남기는 창(기본 50 ms)이고, 그 안쪽은 **건드리지 않는다** — 초기 반사는
음색의 일부이고 위상도 거기 들어 있다.

RT60 은 대역마다 다르므로 대역별로 α 를 만든다. 이 화자의 방 실측
(오프셋 뒤 대역 에너지의 하강, n=16~32):

    500-1k 0.299 s   1-2k 0.381 s   2-4k 0.294 s   4-8k 0.299 s   8-12k 0.246 s

**과빼기를 주의할 것.** 늦은 잔향은 추정량이라 프레임마다 틀린다. `over` 를 1 보다
크게 하면 잡음 바닥이 파이고, 그게 다시 마찰음의 포락을 **거칠게** 만든다 — 고치려는
것과 정확히 반대다. 기본 1.0 과 바닥 0.2 (−14 dB) 는 그 방향으로 안전하게 잡았다.
"""
from __future__ import annotations

import numpy as np

#: 대역별 RT60 [s] — (하한 Hz, 상한 Hz, RT60). `scripts/add_room.py` 와 같은 실측값.
RT60_BANDS = ((0.0, 700.0, 0.30), (700.0, 1500.0, 0.38), (1500.0, 3000.0, 0.29),
              (3000.0, 6000.0, 0.30), (6000.0, 24000.0, 0.25))
#: 직접음 + 초기 반사로 남길 창 [s]. 이 안쪽은 빼지 않는다.
EARLY_S = 0.050
#: 빼기의 바닥 (크기 비). 0.2 = −14 dB. 0 으로 두면 뮤지컬 노이즈가 생긴다.
FLOOR = 0.2


def _alpha(freqs: np.ndarray, delta_frames: int, hop: int, fs: float) -> np.ndarray:
    """빈마다의 감쇠 계수 α. RT60 이 긴 대역일수록 1 에 가깝다(잔향이 오래 남는다)."""
    a = np.empty_like(freqs)
    dt = delta_frames * hop / fs
    for lo, hi, rt in RT60_BANDS:
        m = (freqs >= lo) & (freqs < hi)
        if not m.any():
            continue
        a[m] = np.exp(-2.0 * (3.0 * np.log(10.0) / rt) * dt)
    a[freqs >= RT60_BANDS[-1][1]] = np.exp(
        -2.0 * (3.0 * np.log(10.0) / RT60_BANDS[-1][2]) * dt)
    return a


def dereverb(x: np.ndarray, fs: float, n_fft: int = 1024, over: float = 1.0,
             early_s: float = EARLY_S, floor: float = FLOOR) -> np.ndarray:
    """늦은 잔향을 크기 스펙트럼에서 뺀다. 위상은 그대로 둔다 (직접음의 위상이다)."""
    x = np.asarray(x, dtype=np.float64)
    hop = n_fft // 4
    win = np.hanning(n_fft + 1)[:n_fft]
    n = len(x)
    pad = n_fft
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad + n_fft)])
    t = 1 + (len(xp) - n_fft) // hop
    fr = np.stack([xp[i * hop:i * hop + n_fft] * win for i in range(t)])
    X = np.fft.rfft(fr, axis=-1)
    mag, ph = np.abs(X), np.angle(X)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    d = max(1, int(round(early_s * fs / hop)))
    a = _alpha(freqs, d, hop, fs)
    late = np.zeros_like(mag)
    if t > d:
        late[d:] = np.sqrt(a) * mag[:-d]          # 전력에 α -> 크기에 sqrt(α)
    out = np.maximum(mag - over * late, floor * mag)
    Y = out * np.exp(1j * ph)
    fr2 = np.fft.irfft(Y, n=n_fft, axis=-1) * win
    y = np.zeros(len(xp))
    wsum = np.zeros(len(xp))
    w2 = win * win
    for i in range(t):
        y[i * hop:i * hop + n_fft] += fr2[i]
        wsum[i * hop:i * hop + n_fft] += w2
    y = y / np.maximum(wsum, 1e-9)
    return y[pad:pad + n]


def reverb_ratio(x: np.ndarray, y: np.ndarray) -> float:
    """뺀 에너지의 비율 [%] — 얼마나 지웠는지 한 줄로 보고할 때."""
    ex, ey = float((x ** 2).sum()), float((y ** 2).sum())
    return 100.0 * max(ex - ey, 0.0) / max(ex, 1e-30)
