"""성문 음원 실측 — `wavs/` 에서 20 개를 뽑아 성대 조음 모델의 부족한 곳을 찾는다.

사용자: *"wavs 폴더에서 20개 정도 꺼내서 성대 조음 모델을 좀 더 정교하게 만들어봐 …
무브먼트를 조금 더 생리적인 움직임으로 개선해야 할 듯. 떨림이라든지, 그런 것들."*

재는 것 (모두 유성·비마찰 프레임만):
  1. F0 미세구조  — 지터(주기 대 주기 %), **떨림**(2~12 Hz 대역의 F0 변조: 속도와 깊이)
  2. 진폭 미세구조 — 시머(%), 진폭 떨림(2~12 Hz AM 깊이)
  3. 음원 스펙트럼 — H1-H2, H2-H4, H4-H8 과 그것이 F0·세기와 **같이 움직이는가**
  4. Rd 궤적       — 분포와 **변화 속도**(Rd/s). 제어 손잡이의 생리적 상한이 여기서 나온다
  5. 시작·끝       — 발성 시작에서 정상 진폭까지 몇 주기인가
  6. 주기 배증     — 이중음(diplophonia) 이 얼마나 나오는가

    python scripts/glottal_study.py [파일 수]
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from formant_ml.engine.profile import SpeakerProfile      # noqa: E402
from formant_ml.engine.segment import fricative_mask      # noqa: E402
from formant_ml.engine.analyze import analyze             # noqa: E402
from scipy.signal import stft                             # noqa: E402


def cycle_marks(x, fs, f0, voi):
    """주기 표시 — **직전 주기와의 정규화 상관**이 최대인 자리를 잡는다.

    "창 안의 |x| 최대" 로 잡으면 한 주기 안에서 봉우리가 바뀔 때 표시가 통째로 건너뛰어
    지터가 실제보다 10 배 부풀어 나온다 (첫 판에서 11 % 가 나왔는데 모달 발성의 실측은
    0.2~1 % 다). 상관으로 잡으면 파형 모양이 같은 자리끼리 대응된다.
    """
    marks = []
    n = len(x)
    i = 0
    while i < n - 1:
        j = min(int(i / (fs / 1000.0)), len(f0) - 1)
        if not voi[j] or f0[j] <= 0:
            i += int(fs / 200.0)
            continue
        T = int(round(fs / f0[j]))
        if T < 24 or i + 3 * T >= n:
            i += max(T, 8)
            continue
        if not marks:
            marks.append(i + int(np.argmax(np.abs(x[i:i + T]))))
            i = marks[-1]
            continue
        p0 = marks[-1]
        ref = x[p0:p0 + T]
        if len(ref) < T or p0 + 2 * T >= n:
            break
        best, bk = -2.0, p0 + T
        for k in range(p0 + int(0.75 * T), p0 + int(1.30 * T)):
            seg = x[k:k + T]
            if len(seg) < T:
                break
            d = np.linalg.norm(ref) * np.linalg.norm(seg)
            if d <= 0:
                continue
            r = float(np.dot(ref, seg) / d)
            if r > best:
                best, bk = r, k
        if best < 0.5:                      # 주기성이 무너진 자리 — 끊고 다시 시작
            marks.append(-1)
            i = bk + T
            continue
        marks.append(bk)
        i = bk
    return np.array(marks, int)


def band_mod(y, fs_env, lo, hi):
    """포락 y 의 lo~hi Hz 성분의 rms 와 봉우리 주파수."""
    y = y - np.mean(y)
    n = len(y)
    if n < 32:
        return 0.0, 0.0
    Y = np.fft.rfft(y * np.hanning(n))
    f = np.fft.rfftfreq(n, 1.0 / fs_env)
    m = (f >= lo) & (f <= hi)
    if not m.any():
        return 0.0, 0.0
    p = np.abs(Y[m]) ** 2
    rms = float(np.sqrt(2.0 * p.sum()) / n * 2.0)
    return rms, float(f[m][int(np.argmax(p))])


def harm_levels(x, fs, f0):
    """H1, H2, H4, H8 의 수준 [dB] — 한 프레임의 스펙트럼에서 배음 자리를 읽는다."""
    n = len(x)
    X = np.abs(np.fft.rfft(x * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    out = []
    for k in (1, 2, 4, 8):
        fc = k * f0
        if fc > f[-1] * 0.9:
            out.append(np.nan); continue
        m = (f > fc * 0.85) & (f < fc * 1.15)
        out.append(20 * np.log10(X[m].max() + 1e-12) if m.any() else np.nan)
    return out


def main() -> int:
    nfile = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    prof = SpeakerProfile.load("profiles/yang_female.json")
    files = sorted(glob.glob("wavs/*.wav"))
    step = max(1, len(files) // nfile)
    files = files[::step][:nfile]
    print(f"파일 {len(files)} 개 — {os.path.basename(files[0])} ~ {os.path.basename(files[-1])}\n", flush=True)

    J, S = [], []
    TR_F, TR_A = [[], [], []], [[], [], []]
    TR_FR, TR_AR = [[], [], []], [[], [], []]
    H12, H24, H48, RD, DRD, F0A, ONS, DBL = [], [], [], [], [], [], [], []
    for path in files:
        y, fs = sf.read(path)
        y = np.asarray(y, float)
        if y.ndim > 1:
            y = y.mean(1)
        try:
            tr = analyze(y, fs, prof, 48, t0=0.0, full=y)
        except Exception as e:                       # 분석이 실패하는 파일은 건너뛴다
            print(f"  ! {os.path.basename(path)} 분석 실패 {e}", flush=True)
            continue
        f0 = np.asarray(tr["f0_target"], float)
        voi = np.asarray(tr.voiced, bool)
        fm = np.asarray(fricative_mask(y, fs, prof, 48), bool)
        m = min(len(f0), len(voi), len(fm))
        f0, voi, fm = f0[:m], voi[:m], fm[:m]
        good = voi & ~fm & (f0 > 0)
        if good.sum() < 200:
            continue
        F0A.append(np.median(f0[good]))
        rd = np.asarray(tr["rd_offset"], float)[:m]
        RD.extend(rd[good].tolist())
        DRD.extend((np.abs(np.diff(rd)) * 1000.0)[good[:-1]].tolist())   # Rd/s

        # --- 주기 대 주기: 지터·시머·주기 배증 -----------------------------
        mk = cycle_marks(y, fs, f0, good)
        # 끊긴 자리(-1)로 조각을 나눠 **조각 안에서만** 주기 대 주기를 잰다.
        pieces, cur = [], []
        for v in mk:
            if v < 0:
                if len(cur) > 12:
                    pieces.append(np.array(cur))
                cur = []
            else:
                cur.append(v)
        if len(cur) > 12:
            pieces.append(np.array(cur))
        jj, ss, dd = [], [], []
        for pc in pieces:
            T = np.diff(pc) / fs
            ok = (T > 1.0 / 500) & (T < 1.0 / 60)
            if ok.sum() < 10:
                continue
            T = T[ok]
            jj.append(100.0 * np.mean(np.abs(np.diff(T))) / np.mean(T))
            amp = np.array([np.abs(y[a:b]).max() for a, b in zip(pc[:-1], pc[1:])])[ok]
            ss.append(100.0 * np.mean(np.abs(np.diff(amp))) / np.mean(amp))
            d1 = np.abs(np.diff(T)); d2 = np.abs(T[2:] - T[:-2])
            k = min(len(d1) - 1, len(d2))
            if k > 4:
                dd.append(100.0 * float(np.mean(d2[:k] < d1[:k])))
        if jj:
            J.append(float(np.median(jj))); S.append(float(np.median(ss)))
            if dd:
                DBL.append(float(np.median(dd)))

        # --- 떨림: F0 와 진폭의 2~12 Hz 변조 -------------------------------
        runs, i = [], 0
        while i < m:
            if good[i]:
                j = i
                while j + 1 < m and good[j + 1]:
                    j += 1
                if j - i >= 300:
                    runs.append((i, j + 1))
                i = j + 1
            else:
                i += 1
        env = np.sqrt(np.convolve(y * y, np.ones(48) / 48, "same")[::48][:m] + 1e-20)
        for a, b in runs:
            seg = f0[a:b]
            seg = seg / max(np.median(seg), 1e-6)
            ea = 20 * np.log10(env[a:b] + 1e-12)
            ea = ea - np.convolve(ea, np.ones(101) / 101, "same")
            for lo_, hi_, box in ((1.0, 3.0, 0), (4.0, 8.0, 1), (8.0, 15.0, 2)):
                r, fr = band_mod(seg, 1000.0, lo_, hi_)
                TR_F[box].append(100.0 * r); TR_FR[box].append(fr)
                r2, fr2 = band_mod(ea, 1000.0, lo_, hi_)
                TR_A[box].append(r2); TR_AR[box].append(fr2)
            # 발성 시작: 앞 50 ms 에서 10 -> 90 %
            if a > 5:
                w = env[a:min(b, a + 120)]
                if len(w) > 20 and w.max() > 0:
                    t10 = np.flatnonzero(w > 0.1 * w.max())
                    t90 = np.flatnonzero(w > 0.9 * w.max())
                    if len(t10) and len(t90):
                        ONS.append(float(t90[0] - t10[0]))

        # --- 음원 스펙트럼 기울기 -----------------------------------------
        idx = np.flatnonzero(good)[::40]
        for j in idx:
            a = int(j * 48)
            if a + 1024 > len(y):
                break
            h = harm_levels(y[a:a + 1024], fs, f0[j])
            if not np.any(np.isnan(h)):
                H12.append(h[0] - h[1]); H24.append(h[1] - h[2]); H48.append(h[2] - h[3])

    def q(v, name, unit, *, pct=(5, 25, 50, 75, 95)):
        v = np.asarray([x for x in v if np.isfinite(x)], float)
        if not len(v):
            print(f"  {name}: 없음"); return
        s = "  ".join(f"{p}% {np.percentile(v, p):7.3f}" for p in pct)
        print(f"  {name:22s} [{unit}]  {s}   (n={len(v)})", flush=True)

    print("== 주기 대 주기")
    q(J, "지터 (주기 대 주기)", "%")
    q(S, "시머 (주기 대 주기)", "%")
    q(DBL, "주기 배증 비율", "%")
    print(chr(10) + "== 변조 — 억양(1~3 Hz) / **떨림**(4~8 Hz) / 빠른 것(8~15 Hz)")
    for b, lab in enumerate(("1-3 Hz", "4-8 Hz", "8-15 Hz")):
        q(TR_F[b], f"F0 변조 깊이 {lab}", "%")
        q(TR_A[b], f"진폭 변조 깊이 {lab}", "dB")
    print("\n== 음원 스펙트럼")
    q(H12, "H1-H2", "dB"); q(H24, "H2-H4", "dB"); q(H48, "H4-H8", "dB")
    print("\n== Rd 와 그 속도")
    q(RD, "rd_offset", "Rd"); q(DRD, "|dRd/dt|", "Rd/s")
    print("\n== 그 밖")
    q(F0A, "파일별 중앙 F0", "Hz"); q(ONS, "발성 시작 10-90 %", "ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
