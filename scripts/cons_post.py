"""**협화 후처리** — 성문 주기마다 고역(5.6~17 kHz)의 반주기 협화를 원본 쪽으로 맞춘다 (MEASUREMENTS §52.415).

사용자: *"이전에 협화항에 대해 말했었는데, 그걸 후처리로라도 넣는 게 더 정합성에 좋아보이는데."* 협화 적분(§52.183)은 손실로
넣으면 2 단계 초반에 거의 꺼져 전체 손실의 0.25 % 였다(§52.188). 척도가 원본과 판을 가장 잘 가른 대역은 5.6~17 kHz 이고
(§52.196) 거기서 우리 판이 **덜 협화적**이다(+0.7~1.7 dB, 주기의 54~65 %).

협화도 `Q = mean|x(t+T/2) − x(t)| / rms(x)` (주기 안, 대역 통과 신호; 작을수록 반주기 간격으로 닮았다 = 협화적).
원본보다 덜 협화적인 주기에만 그 대역을 앞뒤 반주기 평균 쪽으로 섞는다:
    y_b = x_b + β · ((x_b(t+T/2) + x_b(t−T/2))/2 − x_b)
β 는 주기마다 원본 Q 에 맞도록 격자로 풀고(0~0.8), 이웃 3 주기 중앙값으로 매끄럽게 한다. 한쪽으로만 문다(원본보다 더 협화적으로
밀지 않는다 — 배음 사이에 대한 사용자 경고와 같은 원칙). 5.6 kHz 아래와 파형 중심은 건드리지 않는다.

    python scripts/cons_post.py out/C/C2Q2 [--out out/C/C2Q2_cons_fit.wav]
"""
import argparse

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

FS = 48000
BAND = (5600.0, 17000.0)
BETA_GRID = np.linspace(0.0, 0.8, 17)


def cycles(pp: np.ndarray):
    c = np.floor(pp / (2 * np.pi))
    b = np.flatnonzero(np.diff(c) != 0) + 1
    return [(int(a), int(e)) for a, e in zip(b[:-1], b[1:]) if 60 <= e - a <= 400]


def q_of(xb, a, e):
    T = e - a
    h = T // 2
    if a - h < 0 or e + h >= len(xb):
        return None
    seg = xb[a:e]
    r = np.sqrt(np.mean(seg ** 2)) + 1e-20
    return float(np.mean(np.abs(xb[a + h:e + h] - seg)) / r)


def blend(xb, a, e, beta):
    T = e - a
    h = T // 2
    seg = xb[a:e]
    avg = 0.5 * (xb[a + h:e + h] + xb[a - h:e - h])
    return seg + beta * (avg - seg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--out", default=None)
    ap.add_argument("--level-db", type=float, default=35.0, help="이 dB 안(파일 최대 대비)의 주기만 본다")
    ap.add_argument("--min-period", type=float, default=0.6, help="원본 저역(80~4000 Hz)의 이웃 주기 상관이 이 값을 넘는 주기만 섞는다")
    a = ap.parse_args()
    t = np.asarray(sf.read(a.stem + "_target.wav")[0], float)
    x = np.asarray(sf.read(a.stem + "_fit.wav")[0], float)[:len(t)]
    pp = np.asarray(np.load(a.stem + "_track.npz")["pulse_phase"], float)[:len(t)]
    sos = butter(6, BAND, "bp", fs=FS, output="sos")
    tb, xb = sosfiltfilt(sos, t), sosfiltfilt(sos, x)
    cyc = cycles(pp)
    lv = np.array([10 * np.log10(np.mean(t[s:e] ** 2) + 1e-20) for s, e in cyc])
    live = lv > lv.max() - a.level_db
    # **원본이 그 주기에 실제로 주기적일 때만** 섞는다. 치찰음·무성 틈에도 펄스 위상은 F0 로 이어져 있어 "주기" 가 잡히는데,
    # 거기서 반주기 섞기는 주기마다 움직이는 빗살 필터가 되어 치찰에 sweep 을 냈다 (C3 0.63~0.68 s, 사용자 청취).
    lo = sosfiltfilt(butter(4, (80.0, 4000.0), "bp", fs=FS, output="sos"), t)
    per = np.zeros(len(cyc))
    for j, (s0, e0) in enumerate(cyc):
        T0 = e0 - s0
        if e0 + T0 >= len(lo):
            continue
        u, v = lo[s0:e0], lo[e0:e0 + T0]
        per[j] = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-20))
    live &= per > a.min_period
    print(f"  원본 주기성 > {a.min_period:g} 인 주기만: {int(live.sum())} 개", flush=True)
    betas = np.zeros(len(cyc))
    dq = []
    for j, (s, e) in enumerate(cyc):
        if not live[j]:
            continue
        qt, qx = q_of(tb, s, e), q_of(xb, s, e)
        if qt is None or qx is None:
            continue
        dq.append(20 * np.log10(qx / qt))
        if qx <= qt:
            continue
        best, bb = abs(qx - qt), 0.0
        for bt in BETA_GRID[1:]:
            y = xb.copy()
            y[s:e] = blend(xb, s, e, bt)
            q = q_of(y, s, e)
            if q is not None and abs(q - qt) < best:
                best, bb = abs(q - qt), bt
        betas[j] = bb
    from scipy.ndimage import median_filter
    betas = median_filter(betas, 3, mode="nearest")
    yb = xb.copy()
    for j, (s, e) in enumerate(cyc):
        if betas[j] > 0:
            yb[s:e] = blend(xb, s, e, betas[j])
    y = x + (yb - xb)
    dq2 = [20 * np.log10(q_of(yb, s, e) / q_of(tb, s, e)) for j, (s, e) in enumerate(cyc)
           if live[j] and q_of(tb, s, e) and q_of(yb, s, e)]
    dq, dq2 = np.array(dq), np.array(dq2)
    print(f"  주기 {len(cyc)} 개 (소리 있는 {int(live.sum())}), 섞은 주기 {int((betas > 0).sum())} 개, β 중앙 "
          f"{np.median(betas[betas > 0]) if (betas > 0).any() else 0:.2f}")
    print(f"  고역 협화 차 (우리 − 원본) [dB]: 전 중앙 {np.median(dq):+.2f} · 덜 협화적 주기 {np.mean(dq > 0.5) * 100:.0f} %  →  "
          f"후 중앙 {np.median(dq2):+.2f} · {np.mean(dq2 > 0.5) * 100:.0f} %")
    out = a.out or a.stem + "_cons_fit.wav"
    sf.write(out, y.astype(np.float32), FS)
    print("  썼다:", out)


if __name__ == "__main__":
    main()
