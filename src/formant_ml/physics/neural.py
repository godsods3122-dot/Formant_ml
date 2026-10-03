"""**신경 작용 기제 — 후두근 긴장의 운동 단위 요동과 생리적 떨림** (MEASUREMENTS §52.523).

사용자: *"이전에 말했듯, 신경 작용 기제라든지 이런 것도 다 봐야 해"*. 근육의 힘은 매끈한 제어값이 아니라 운동 단위들의 연축(twitch) 합이다.
Titze (1991, "A model for neurologic sources of aperiodicity in vocal fold vibration", JSHR 34:460–472) 의 꼴을 따른다:

* 근육 하나 = 운동 단위 `n_mu` 개. 단위 k 의 크기 a_k 는 크기 원리(Henneman)대로 지수 분포 (작은 단위가 많다).
* 단위마다 평균 발화율 r_k ~ U(rate) (후두근 발성 중 20–40 Hz), 발화 간격은 감마 분포 (변동계수 `isi_cv`).
* 연축 응답 h(t) = (t/τ)·exp(1 − t/τ) (임계 감쇠, 수축 시간 τ — 갑상피열근 ~15 ms, 윤상갑상근 ~30 ms; Titze 1994 표).
* 공통 구동 — 생리적 떨림: 모든 단위의 발화율을 (1 + m·sin(2π f_tr t + φ)) 로 함께 흔든다 (4–12 Hz, 깊이 수 %).
* 출력은 평균 1 로 정규화한 상대 긴장 T(t)/⟨T⟩. 상대 요동 σ ≈ 1/√(유효 단위 수).

실현은 씨앗으로 고정한다 (재현). 기울기는 내지 않는다 (요동은 적합 대상이 아니다; 세기는 생리 값).
"""
from __future__ import annotations

import numpy as np

#: 근육별 문헌 값 (Titze 1994 *Principles of Voice Production* 표 2.x · Titze 1991): 수축 시간 [ms], 발화율 범위 [Hz]
MUSCLES = {
    "ta": dict(twitch_ms=15.0, rate=(20.0, 40.0)),
    "ct": dict(twitch_ms=30.0, rate=(15.0, 30.0)),
}


def tension(n: int, fs: float, muscle: str = "ct", n_mu: int = 200, isi_cv: float = 0.15, tremor_hz: float = 6.0,
            tremor_depth: float = 0.03, seed: int = 0, mean_size: float = 1.0) -> np.ndarray:
    """상대 근육 긴장 (n,) — 평균 1. `fs` 표본률. 처음 0.3 s 는 미리 돌려 버린다 (발화가 정상 상태가 되게)."""
    p = MUSCLES[muscle]
    rng = np.random.default_rng(10_007 + 97 * seed + (0 if muscle == "ct" else 1))
    pre = int(0.3 * fs)
    N = n + pre
    t = np.arange(N) / fs
    tau = p["twitch_ms"] * 1e-3
    hl = int(8 * tau * fs)
    th = np.arange(hl) / fs
    h = (th / tau) * np.exp(1.0 - th / tau)
    drive = 1.0 + tremor_depth * np.sin(2 * np.pi * tremor_hz * t + rng.uniform(0, 2 * np.pi))
    spikes = np.zeros(N)
    sizes = rng.exponential(mean_size, n_mu)
    rates = rng.uniform(*p["rate"], n_mu)
    k_shape = 1.0 / isi_cv ** 2                                  # 감마 간격의 모양 (변동계수 = 1/√k)
    cum = np.concatenate([[0.0], np.cumsum(drive) / fs])          # 공통 구동을 시간 왜곡으로: 발화율 × drive
    for a, r in zip(sizes, rates):
        # 왜곡된 시간 s = ∫ drive dt 위에서 감마 간격 발화 → 실제 시각으로 되돌린다
        s = rng.uniform(0, 1.0 / r)
        ev = []
        while s < cum[-1]:
            ev.append(s)
            s += rng.gamma(k_shape, 1.0 / (r * k_shape))
        idx = np.searchsorted(cum, np.asarray(ev)) - 1
        idx = idx[(idx >= 0) & (idx < N)]
        np.add.at(spikes, idx, a)
    T = np.convolve(spikes, h)[:N]
    T = T[pre:]
    return T / max(T.mean(), 1e-12)


def stats(x: np.ndarray, fs: float) -> dict:
    """상대 요동 σ 와 4–12 Hz 떨림 몫, 주기 규모(> 50 Hz) 몫."""
    d = x - x.mean()
    X = np.abs(np.fft.rfft(d)) ** 2
    f = np.fft.rfftfreq(d.size, 1 / fs)
    tot = X[1:].sum()
    return dict(sigma=float(d.std()), tremor=float(X[(f >= 4) & (f <= 12)].sum() / tot),
                fast=float(X[f > 50].sum() / tot))
