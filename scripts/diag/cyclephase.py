"""**주기마다 배음 위상이 얼마나 흔들리나** — 대역별, 국소 창에서 (MEASUREMENTS §52.459).

사용자: *"스펙트로그램 자체는 맞는데 왤케 퍼지듯이 들리지? 원본보다 더 리버브가 낀 듯한 소리가 나는데. … 위상차 문제는 아니지?
위상차가 흔들려도 리버브 같은 효과가 충분히 있을 거 같은데. 잔향도 그렇고."*

목표 펄스로 한 주기씩 잘라 M 표본으로 다시 떠(배음 k → 빈 k), 주기 j 의 배음 복소 진폭 X_k(j) 를 얻는다. F0·포먼트가 천천히 움직여 생기는
위상 흐름은 **2 차 차분**으로 뺀다: d_k(j) = arg[X_k(j+1)·X_k(j−1)·conj(X_k(j))²]. 주기마다 독립인 흔들림 σ 이면 var(d) = 6σ² 이므로
σ = 원형 표준편차(d)/√6 [rad]. 대역마다 배음 크기(dB)로 가중한 중앙값. 합성도 **같은 펄스**로 자른다.

    python scripts/diag/cyclephase.py out/C/W2b out/C/W2a --win 0.52 0.84
    python scripts/diag/cyclephase.py --selftest
"""
import argparse
import sys

import numpy as np
import soundfile as sf

FS, M = 48000, 256
BANDS = ((500., 1500.), (1500., 3000.), (3000., 4500.), (4500., 6000.), (6000., 9000.))


def harmonics(x, pul):
    """주기마다 (f0, X[1..M/2-1])."""
    out = []
    for a, b in zip(pul[:-1], pul[1:]):
        i0, i1 = int(round(a * FS)), int(round(b * FS))
        if i1 - i0 < 40 or i1 - i0 > FS // 60 or i1 > len(x) or i0 < 0:
            out.append(None)
            continue
        s = x[i0:i1]
        r = np.interp(np.linspace(0, len(s), M, endpoint=False), np.arange(len(s)), s)
        out.append((1.0 / (b - a), np.fft.rfft(r)[1:M // 2]))
    return out


def jitter(x, pul):
    H = harmonics(x, pul)
    res = {b: [] for b in BANDS}
    for j in range(1, len(H) - 1):
        if H[j - 1] is None or H[j] is None or H[j + 1] is None:
            continue
        f0, X = H[j]
        d = np.angle(H[j + 1][1] * H[j - 1][1] * np.conj(X) ** 2)
        a = np.abs(X)
        fk = np.arange(1, len(X) + 1) * f0
        for lo, hi in BANDS:
            m = (fk >= lo) & (fk < hi) & (a > 1e-9)
            if m.sum():
                res[(lo, hi)].append((d[m], 20 * np.log10(a[m])))
    out = {}
    for b, v in res.items():
        if not v:
            out[b] = np.nan
            continue
        d = np.concatenate([q[0] for q in v])
        w = np.concatenate([q[1] for q in v])
        keep = w > np.percentile(w, 30)                      # 약한 배음(잡음 바닥)은 뺀다 — 목표와 합성에 같은 규칙
        d = d[keep]
        R = np.abs(np.mean(np.exp(1j * d)))
        circ = np.sqrt(-2 * np.log(max(R, 1e-12)))
        out[b] = circ / np.sqrt(6.0)
    return out


def show(name, r):
    print(f"  {name:>12s} " + " ".join(f"{r[b]:7.3f}" for b in BANDS))


def run(stems, win):
    sys.path.insert(0, "src")
    from formant_ml.engine.analyze import glottal_pulses
    from formant_ml.engine.profile import SpeakerProfile
    prof = SpeakerProfile.load("profiles/yang_female.json")
    t = np.asarray(sf.read(stems[0] + "_target.wav")[0], float)
    pul = np.asarray(glottal_pulses(t, FS, prof), float)
    pul = pul[(pul >= win[0]) & (pul <= win[1])]
    print(f"창 {win[0]:.3f}–{win[1]:.3f} s, 목표 펄스 {len(pul)} 개 — 주기별 배음 위상 흔들림 σ [rad] (작을수록 주기마다 같은 위상)")
    print(f"  {'판':>12s} " + " ".join(f"{lo/1000:g}-{hi/1000:g}k".rjust(7) for lo, hi in BANDS))
    show("목표", jitter(t, pul))
    for s in stems:
        y = np.asarray(sf.read(s + "_fit.wav")[0], float)[:len(t)]
        show(s.split("/")[-1], jitter(y, pul))


def selftest():
    """덧셈 합성: F0 250→220 Hz, 배음 크기는 고정 포락, 위상 = k·(주기 시작) + 흔들림(1.5 kHz 위만, σ). 잔향만 더한 판도."""
    rng = np.random.default_rng(0)
    n = int(0.6 * FS)
    f0 = np.linspace(250, 220, n)
    ph = 2 * np.pi * np.cumsum(f0 / FS)
    pul = (np.flatnonzero(np.diff(np.floor(ph / (2 * np.pi))) > 0) + 1) / FS
    cyc = np.floor(ph / (2 * np.pi)).astype(int)
    K = 40

    def make(sig):
        y = np.zeros(n)
        jit = rng.standard_normal((cyc.max() + 2, K + 1)) * sig
        for k in range(1, K + 1):
            amp = 1.0 / k * (1 + 3 * np.exp(-((k * 235 - 3500) / 600) ** 2))
            j = jit[cyc, k] if k * 235 > 1500 else 0.0
            y += amp * np.cos(k * ph + j)
        return y

    ok = True
    print("  참 σ (1.5 kHz 위)  → 잰 σ " + " ".join(f"{lo/1000:g}-{hi/1000:g}k".rjust(7) for lo, hi in BANDS))
    base = None
    for s in (0.0, 0.3, 0.6):
        r = jitter(make(s), pul)
        print(f"  {s:4.1f}              " + " ".join(f"{r[b]:7.3f}" for b in BANDS))
        hb = [r[b] for b in BANDS[2:]]
        good = all(abs(v - s) < 0.12 + 0.25 * s for v in hb) and r[BANDS[0]] < 0.08   # 흔들림 0 의 바닥 ≤ 0.11 (F0 가 12 % 미끄러짐)
        ok &= good
        if s == 0.0:
            base = make(0.0)
    # 잔향: 백색 잡음 × exp(−6.91 t / 0.35 s), 잔향/직접 −11.5 dB, 5 ms 부터
    L = int(0.8 * FS)
    h = rng.standard_normal(L) * np.exp(-6.91 * np.arange(L) / FS / 0.35)
    h[:int(0.005 * FS)] = 0
    h *= np.sqrt(10 ** (-11.5 / 10) / np.sum(h ** 2))
    yr = base + np.convolve(base, h)[:n]
    r = jitter(yr, pul)
    print(f"  σ 0 + 잔향 −11.5 dB  " + " ".join(f"{r[b]:7.3f}" for b in BANDS))
    good = all(r[b] > 0.04 for b in BANDS[1:])
    print(f"  {'통과' if ok else '실패'}  위상 흔들림을 되읽는다;  {'통과' if good else '실패'}  잔향도 흔들림으로 읽힌다(> 0.04 rad)")
    return 0 if (ok and good) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="*")
    ap.add_argument("--win", nargs=2, type=float, default=(0.0, 99.0))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    run(a.stems, a.win)
