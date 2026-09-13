"""적합 결과를 **한 자로 전수** 잰다 — 비조화·스펙트로그램·파형·제어열을 한 표에.

    python scripts/census.py out/L13/s040 out/L16/s040 ...
    python scripts/census.py --tags L0 L6 L13 L16 --files s040 s101

왜 이 스크립트를 만들었나
    이 프로젝트는 자 하나만 보면 반드시 속았다 (RUN_LOCAL §6). 포락 점수는 멜
    밴드가 포먼트 골보다 넓어 공진을 못 보고(§37), 시간평균 스펙트럼은 조용한
    프레임의 결손을 묻고(§38.2b), 스펙트럼 크기만 재는 자는 하모닉이 잡음으로 바뀐
    사건에 침묵했다(§44). 그래서 서로 다른 것을 보는 자를 **한 번에** 댄다.

무엇을 재는가 (모두 목표 대비)
    포락·정밀        copyfit 의 `_report.json` (참고용)
    스펙 250-15k     시간평균 스펙트럼 일치. 15 kHz 위는 뺀다 — s040 목표는 20 kHz 코덱
                     절벽까지 에너지가 있어 그 빈이 전대역 점수를 지배한다 (§43.2 교훈)
    주기성 오차      유성 프레임, 대역별 F0 지연 자기상관의 |합성 − 목표|. 비조화의 자
    고역 동기        유성 프레임 8~12 / 12~16 kHz **포락**의 F0 지연 자기상관. 실제
                     고역은 성문 펄스마다 터지므로 F0 에 물린다 (목표 ~0.83)
    끊김            20 ms 파형 상관이 0 아래인 창 수 (목표 정점 −45 dB 위 창만)
    STFT 오차        프레임별 dB 오차의 평균 |·|, 대역별. 장기평균이 못 보는 것
    제어열 난동      제어열 에너지 중 60 Hz 위 비율의 평균·최대 (조음은 20 Hz 아래)
    줄 6-10/10-16    고역 미세구조(dB − 주파수 1 kHz 이동중앙값)의 **50 ms 지연 상관**.
                     안개(잡음)면 창마다 새로 뽑혀 0 에 가깝고, 수평 줄이 서 있으면 높다.
                     사용자가 본 "고역의 평평한 줄" 의 자다 (MEASUREMENTS §50.5, 목표 ~0.05)
    층층6-10/10-16  6~10 / 10~16 kHz 의 100 ms 국소 평균 스펙트럼 요철을 주파수 방향으로 자기상관한
                     값의 약 1 kHz(950~1060 Hz) 지연 최대. 사용자가 본 "고역에 똑같이 생긴 배음이
                     층층이" 의 자다 — 성도 궤적이 1 ms 프레임마다 꺾여 생긴 1 kHz 빗살(§50.14).
                     목표 0.12 / 0.05, 고치기 전 적합 0.3~0.4 / 0.4~0.6. **합성의 그 대역이 목표보다
                     20 dB 넘게 낮은 창은 뺀다** — 소리가 없으면 결함도 안 보인다(고역이 10 kHz
                     아래에서 잘린 `L7` 이 10~16 kHz 만 재면 깨끗하게 나왔다, 사용자 지적).
                     위의 "줄" 척도(시간 지속)는 이것을 원리적으로 못 본다
    고역생존        10~16 kHz 대역이 목표의 같은 창보다 20 dB 넘게 낮지 않은 창의 비율 [%].
                     "고역이 다 죽었다" 의 자
    치찰토막        마찰 프레임에서 4~16 kHz 포락(2.5 ms 창)의 40 ms 추세 대비 요동 [dB].
                     세로로 토막나면 커진다 (§50.4). 목표값을 함께 찍는다
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

BANDS = ((300, 1000), (1000, 3000), (3000, 5000), (5000, 8000), (8000, 12000),
         (12000, 16000))


def _bp(x, lo, hi, fs):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return np.fft.irfft(X * ((f >= lo) & (f < hi)), len(x))


def _env(x, fs, cut=1500.0):
    a = np.abs(np.fft.ifft(np.fft.fft(x) * (2 * (np.fft.fftfreq(len(x)) > 0))))
    A = np.fft.rfft(a)
    f = np.fft.rfftfreq(len(a), 1.0 / fs)
    return np.fft.irfft(A * (f < cut), len(a))


def _per(x, fs, f0, centres, wp=3.0):
    lag = int(round(fs / f0))
    w = int(wp * lag)
    o = []
    for c in centres:
        a = c - w // 2
        if a < 0 or a + w + lag > len(x):
            continue
        u, z = x[a:a + w], x[a + lag:a + w + lag]
        d = np.sqrt((u * u).sum() * (z * z).sum())
        if d > 1e-20:
            o.append((u * z).sum() / d)
    return float(np.median(o)) if o else float("nan")


def _spec_match(t, p, fs, lo, hi):
    from formant_ml.engine.turbulence import average_psd
    n = min(len(t), len(p))
    f, pt = average_psd(t[:n], fs)
    _, pp = average_psd(p[:n], fs)
    m = min(len(pt), len(pp))
    f = f[:m]
    a = 10 * np.log10(pt[:m] + 1e-20)
    b = 10 * np.log10(pp[:m] + 1e-20)
    k = (a > a.max() - 60) & (f >= lo) & (f < hi)
    a, b = a[k] - a[k].mean(), b[k] - b[k].mean()
    return 100 * (1 - np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))


def _stft_db(x, n=1024, hop=256):
    w = np.hanning(n)
    fr = [20 * np.log10(np.abs(np.fft.rfft(x[i:i + n] * w)) + 1e-10)
          for i in range(0, len(x) - n, hop)]
    return np.array(fr)


def _line_persist(x, fs, live, lag=10):
    """고역 미세구조의 지연 상관 (hop 5 ms → lag 10 = 50 ms). 대역 (6-10k, 10-16k)."""
    from scipy.ndimage import median_filter
    from scipy.signal import stft
    f, _, Z = stft(x, fs, nperseg=1024, noverlap=1024 - 240)
    S = 20 * np.log10(np.abs(Z) + 1e-9)
    w = int(1000 / (f[1] - f[0])) | 1
    F = S - median_filter(S, size=(w, 1), mode="nearest")
    T = min(F.shape[1], len(live))
    out = []
    for lo, hi in ((6000, 10000), (10000, 16000)):
        sel = (f >= lo) & (f < hi)
        cs = [np.corrcoef(F[sel, i], F[sel, i + lag])[0, 1]
              for i in range(T - lag) if live[i] and live[i + lag]]
        out.append(float(np.mean(cs)) if cs else float("nan"))
    return out


def _stack1k(x, fs, live_ref, ref=None, band=(10000, 16000)):
    """대역 100 ms 국소 평균 요철의 ≈1 kHz 주파수 지연 자기상관 (MEASUREMENTS §50.14).

    `ref`(목표 파형)를 주면 합성의 그 대역이 목표의 같은 창보다 20 dB 넘게 낮은 창을 뺀다.
    반환 (값, 센 창 비율)."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import stft
    f, _, Z = stft(x, fs, nperseg=4096, noverlap=4096 - 240)
    P = np.abs(Z) ** 2
    Pr = None
    if ref is not None:
        Pr = np.abs(stft(ref, fs, nperseg=4096, noverlap=4096 - 240)[2]) ** 2
    d = f[1] - f[0]
    sel = (f >= band[0]) & (f < band[1])
    T = min(P.shape[1], len(live_ref))
    lo, hi = int(950 / d), int(1060 / d)
    acs, n_all = [], 0
    for s0 in range(0, T - 21, 10):
        if not live_ref[s0:s0 + 21].all():
            continue
        n_all += 1
        if Pr is not None:
            pb = 10 * np.log10(P[sel, s0:s0 + 21].mean() + 1e-30)
            tb = 10 * np.log10(Pr[sel, s0:s0 + 21].mean() + 1e-30)
            if pb < tb - 20.0:
                continue
        m = 10 * np.log10(P[:, s0:s0 + 21].mean(1) + 1e-30)
        r = (m - uniform_filter1d(m, int(1000 / d)))[sel]
        u = r - r.mean()
        acs.append(max(np.dot(u[:-L], u[L:]) / (u * u).sum() for L in range(lo, hi)))
    return (float(np.mean(acs)) if acs else float("nan")), len(acs) / max(n_all, 1)


def _live_4096(t, fs):
    from scipy.signal import stft
    _, _, Z = stft(t, fs, nperseg=4096, noverlap=4096 - 240)
    lv = 10 * np.log10((np.abs(Z) ** 2).sum(0) + 1e-30)
    return lv > lv.max() - 30


def _live_5ms(t, fs):
    from scipy.signal import stft
    _, _, Z = stft(t, fs, nperseg=1024, noverlap=1024 - 240)
    lv = 20 * np.log10(np.abs(Z).max(0) + 1e-9)
    return lv > lv.max() - 45


def _chop(x, fs, fm, hop):
    """마찰 프레임의 4~16 kHz 포락 요동 [dB] — 2.5 ms 창 에너지에서 40 ms 이동평균을 뺀 표준편차."""
    from scipy.ndimage import uniform_filter1d
    b = _bp(x, 4000, 16000, fs)
    w = int(0.0025 * fs)
    h = w // 2
    e = np.array([10 * np.log10(np.mean(b[i:i + w] ** 2) + 1e-20)
                  for i in range(0, len(b) - w, h)])
    c = (np.arange(len(e)) * h + w // 2) // hop
    m = np.asarray(fm, bool)[np.clip(c, 0, len(fm) - 1)]
    if not m.any():
        return float("nan")
    d = (e - uniform_filter1d(e, max(3, int(0.040 * fs / h))))[m]
    return float(np.std(d))


def measure(stem: str, f0: float) -> dict:
    import json
    from formant_ml.engine.profile import SpeakerProfile
    from formant_ml.engine.segment import fricative_mask

    t, fs = sf.read(stem + "_target.wav")
    p, _ = sf.read(stem + "_fit.wav")
    t, p = np.asarray(t, float), np.asarray(p, float)
    n = min(len(t), len(p))
    t, p = t[:n], p[:n]
    out: dict = {}
    try:
        rep = json.load(open(stem + "_report.json", encoding="utf-8"))
        out["env"], out["fine"] = rep["env"], rep["fine"]
    except Exception:
        out["env"] = out["fine"] = float("nan")
    out["spec"] = _spec_match(t, p, fs, 250, 15000)

    prof = SpeakerProfile.load(os.path.join(os.path.dirname(__file__), "..",
                                            "profiles", "yang_female.json"))
    hop = int(round(fs / 1000.0))
    fm = fricative_mask(t, fs, prof, hop)
    lvl = np.array([20 * np.log10(np.std(t[i * hop:(i + 1) * hop]) + 1e-20)
                    for i in range(len(fm))])
    voiced = (~fm) & (lvl > lvl.max() - 30)
    cv = [i * hop for i in np.flatnonzero(voiced)[::20]]

    perr = []
    for lo, hi in BANDS[:5]:
        perr.append(abs(_per(_bp(p, lo, hi, fs), fs, f0, cv)
                        - _per(_bp(t, lo, hi, fs), fs, f0, cv)))
    out["per_err"] = float(np.nanmean(perr))
    for lo, hi, key in ((8000, 12000, "sync8"), (12000, 16000, "sync12")):
        out[key] = _per(_env(_bp(p, lo, hi, fs), fs), fs, f0, cv)
        out[key + "_t"] = _per(_env(_bp(t, lo, hi, fs), fs), fs, f0, cv)

    win, h = int(0.020 * fs), int(0.005 * fs)
    cs, live = [], []
    for i in range(0, n - win, h):
        a, b = t[i:i + win], p[i:i + win]
        d = np.sqrt((a * a).sum() * (b * b).sum())
        cs.append((a * b).sum() / d if d > 1e-14 else np.nan)
        live.append(20 * np.log10(a.std() + 1e-20))
    cs, live = np.array(cs), np.array(live)
    ok = live > live.max() - 45
    out["breaks"] = int((cs[ok] < 0).sum())
    out["corr_med"] = float(np.nanmedian(cs[ok]))

    lv5 = _live_5ms(t, fs)
    out["line6"], out["line10"] = _line_persist(p, fs, lv5)
    out["line6_t"], out["line10_t"] = _line_persist(t, fs, lv5)
    out["chop"], out["chop_t"] = _chop(p, fs, fm, hop), _chop(t, fs, fm, hop)
    lv4 = _live_4096(t, fs)
    out["stack6"], _ = _stack1k(p, fs, lv4, t, (6000, 10000))
    out["stack10"], cov = _stack1k(p, fs, lv4, t, (10000, 16000))
    out["hf_alive"] = 100.0 * cov
    out["stack6_t"], _ = _stack1k(t, fs, lv4, None, (6000, 10000))
    out["stack10_t"], _ = _stack1k(t, fs, lv4, None, (10000, 16000))

    T, P = _stft_db(t), _stft_db(p)
    m = min(len(T), len(P))
    T, P = T[:m], P[:m]
    fr = np.fft.rfftfreq(1024, 1.0 / fs)
    fl = T.mean(1) > T.mean(1).max() - 45
    for lo, hi, key in ((300, 3000, "stft_lo"), (3000, 8000, "stft_mid"),
                        (8000, 16000, "stft_hi")):
        k = (fr >= lo) & (fr < hi)
        out[key] = float(np.abs(P[fl][:, k] - T[fl][:, k]).mean())

    try:
        from formant_ml.engine.control import PARAMS
        d = np.load(stem + "_track.npz", allow_pickle=True)
        v = d["values"]
        names = [str(q) for q in d["names"]]
        e = []
        for i in range(v.shape[1]):
            x = v[:, i].astype(float)
            # **경계에 고착된 파라미터는 뺀다.** 값이 거의 상수면 1e-5 짜리 요동도
            # 분산 대비로는 전부 "고주파" 가 되어 비율이 부푼다 (`oral_open` 이 상한
            # 1.0 에 0.9999 로 붙어 84 %로 나왔다). 범위의 0.1 % 도 안 움직이면 난동이 아니다.
            sp = PARAMS.get(names[i]) if i < len(names) else None
            rng = (float(sp.hi) - float(sp.lo)) if sp is not None else 1.0
            if not np.isfinite(rng) or rng <= 0:
                rng = max(abs(float(np.median(x))), 1e-9)
            if x.std() < 1e-3 * rng:
                continue
            X = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)))) ** 2
            f = np.fft.rfftfreq(len(x), 1e-3)
            e.append(100 * X[f >= 60].sum() / (X.sum() + 1e-30))
        out["ctl_mean"], out["ctl_max"] = float(np.mean(e)), float(np.max(e))
    except Exception:
        out["ctl_mean"] = out["ctl_max"] = float("nan")
    return out


F0 = {"s040": 300.0, "s101": 270.0}
COLS = [("env", "포락", "{:7.2f}"), ("fine", "정밀", "{:7.2f}"),
        ("spec", "스펙", "{:7.1f}"), ("per_err", "주기오차", "{:8.3f}"),
        ("sync8", "동기8k", "{:7.3f}"), ("sync12", "동기12k", "{:8.3f}"),
        ("breaks", "끊김", "{:5d}"), ("corr_med", "상관중앙", "{:8.3f}"),
        ("stft_lo", "오차lo", "{:7.2f}"), ("stft_mid", "오차mid", "{:8.2f}"),
        ("stft_hi", "오차hi", "{:7.2f}"), ("ctl_mean", "난동평균", "{:8.2f}"),
        ("ctl_max", "난동최대", "{:8.1f}"), ("line6", "줄6-10", "{:7.3f}"),
        ("line10", "줄10-16", "{:8.3f}"), ("stack6", "층층6-10", "{:8.3f}"), ("stack10", "층층10-16", "{:9.3f}"),
        ("hf_alive", "고역생존", "{:8.0f}"), ("chop", "치찰토막", "{:8.2f}")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="*")
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--files", nargs="*", default=["s040", "s101"])
    a = ap.parse_args()
    stems = list(a.stems)
    if a.tags:
        stems += [f"out/{t}/{f}" for f in a.files for t in a.tags]
    stems = [s for s in stems if os.path.exists(s + "_fit.wav")]
    if not stems:
        print("재 볼 결과가 없다")
        return 1
    rows = []
    for s in stems:
        lab = os.path.basename(s)
        rows.append((s, measure(s, F0.get(lab, 290.0))))
    print(f"{'결과':<14}" + "".join(f"{h:>{len(fmt.format(0)) if 'd' in fmt else len(fmt.format(0.0))}}"
                                    for _, h, fmt in COLS))
    for s, r in rows:
        cells = []
        for k, _, fmt in COLS:
            v = r.get(k, float("nan"))
            cells.append(fmt.format(v) if not (isinstance(v, float) and np.isnan(v))
                         else " " * len(fmt.format(0.0)))
        print(f"{s.replace('out/', ''):<14}" + "".join(cells))
    lab0 = os.path.basename(stems[0])
    r0 = rows[0][1]
    print(f"\n목표의 고역 동기: 8~12k {r0.get('sync8_t', float('nan')):.3f}, "
          f"12~16k {r0.get('sync12_t', float('nan')):.3f} ({lab0})")
    seen = set()
    for s, r in rows:
        lab = os.path.basename(s)
        if lab in seen:
            continue
        seen.add(lab)
        print(f"목표 {lab}: 줄 6-10k {r.get('line6_t', float('nan')):.3f}, "
              f"10-16k {r.get('line10_t', float('nan')):.3f}, 층층 6-10k {r.get('stack6_t', float('nan')):.3f} / 10-16k {r.get('stack10_t', float('nan')):.3f}, "
              f"치찰토막 {r.get('chop_t', float('nan')):.2f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
