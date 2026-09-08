"""빠른 조음 경계에 비선형 요동이 있는가 — 엔진과 무관하게 **녹음만** 본다.

왜 필요한가
-----------
"경계 조건이 빠르게 바뀌면 그 사이에 비선형 요동이 있을 수 있다" 는 가설은 그럴듯하다.
그런데 그게 사실이라면 그 순간의 녹음은 **조화 모형**(= 어떤 선형 시스템을 주기 소스로
구동한 결과)으로 설명되지 않는 에너지를 가져야 한다. 그러니 우리 엔진을 끌어들이지 말고
녹음만 놓고 재면 된다. 엔진의 결함과 물리의 요동이 섞이지 않는다는 게 이 측정의 값어치다.

자의 검증 (`tests/engine/test_waveform.py` 와 같은 방식)
-------------------------------------------------------
백색잡음 72~73 %, 순수 조화 0.0 %, 조화 99 % + 잡음 1 % 혼합이 1.6 % 로 읽힌다.
즉 이 자는 "얼마나 비주기적인가" 를 제대로 센다.

실측 결과 (여성 `ilin-ilsil`, 유효 4,480 프레임)
------------------------------------------------
| 조음자 속도 |dF/dt| | 비조화 비율 |
|---|---|
| 정지 (하위 20 %)  | 0.4 % |
| 빠름 (상위 20 %)  | 1.3 % |
| 매우 빠름 (상위 5 %) | 1.0 % |

속도 ~ 비조화 상관 **+0.05**. 요동은 있지만 −20 dB 짜리다. 우리 설측·탄음의 조화 SNR 은
0 dB 근처이므로 **오차가 요동보다 100 배 크다** — 지금 무너지는 이유가 아니다.
"""
from __future__ import annotations

import argparse

import numpy as np
import parselmouth as pm
import soundfile as sf

from formant_ml.engine.denoise import denoise, noise_profile
from formant_ml.engine.waveform import decompose


def probe(path: str, f0_lo: float = 120.0, f0_hi: float = 580.0,
          gate_db: float = -25.0) -> dict:
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(1)
    y = denoise(y, sr, noise_profile(y, sr))
    snd = pm.Sound(y, sampling_frequency=sr)
    pt = snd.to_pitch(time_step=0.001, pitch_floor=f0_lo, pitch_ceiling=f0_hi)
    fm = snd.to_formant_burg(time_step=0.001, max_number_of_formants=5.0,
                             maximum_formant=5500.0, window_length=0.025)
    hop = int(round(0.001 * sr))
    nf = len(y) // hop
    f0 = np.zeros(nf); F1 = np.zeros(nf); F2 = np.zeros(nf); voi = np.zeros(nf, bool)
    last = 200.0
    for i in range(nf):
        t = (i + 0.5) * 0.001
        v = pt.get_value_at_time(t)
        if v is not None and v == v:
            f0[i] = last = v; voi[i] = True
        else:
            f0[i] = last
        for j, arr in ((1, F1), (2, F2)):
            w = fm.get_value_at_time(j, t)
            arr[i] = w if w is not None and w == w else np.nan
    F1 = np.nan_to_num(F1); F2 = np.nan_to_num(F2)

    _, har, res = decompose(y, float(sr), f0, hop)
    nonh = np.zeros(nf); rms = np.zeros(nf)
    for i in range(nf):
        a, b = i * hop, min(i * hop + hop, len(y))
        eh, er = (har[a:b] ** 2).sum(), (res[a:b] ** 2).sum()
        nonh[i] = er / (eh + er + 1e-20)
        rms[i] = np.sqrt((y[a:b] ** 2).mean() + 1e-20)
    db = 20 * np.log10(rms / rms.max() + 1e-12)
    vel = np.zeros(nf)
    vel[1:-1] = np.hypot(F1[2:] - F1[:-2], F2[2:] - F2[:-2]) / 2.0
    loud = (db > gate_db) & voi & (F1 > 0)
    # 포먼트 추정이 통째로 튄 프레임(상위 2 %)은 조음이 아니라 추정 실패다.
    sel = loud & (vel <= np.percentile(vel[loud], 98))
    return dict(sel=sel, vel=vel, nonh=nonh, db=db, n=nf)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wav")
    a = ap.parse_args()
    r = probe(a.wav)
    sel, vel, nonh, db = r["sel"], r["vel"], r["nonh"], r["db"]
    q = np.percentile(vel[sel], [20, 50, 80, 95])
    print(f"유효 프레임 {int(sel.sum())} 개 (전체 {r['n']})")
    print(f"조음자 속도 분위 20/50/80/95 : "
          f"{q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f} / {q[3]:.0f} Hz/ms\n")
    print(f"{'조음자 속도 구간':>22s} {'프레임':>6s} {'비조화 비율':>12s} {'세기 dB':>9s}")
    edges = [0.0, q[0], q[1], q[2], q[3], 1e9]
    names = ["정지 (하위 20 %)", "느림", "보통", "빠름 (상위 20 %)", "매우 빠름 (상위 5 %)"]
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        m = sel & (vel >= lo) & (vel < hi)
        if m.sum() < 5:
            continue
        print(f"{nm:>22s} {int(m.sum()):6d} {100 * np.median(nonh[m]):11.1f}% "
              f"{np.median(db[m]):8.1f}")
    print(f"\n속도 ~ 비조화 상관: {np.corrcoef(vel[sel], nonh[sel])[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
