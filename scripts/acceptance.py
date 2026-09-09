"""사용자가 건 네 가지 합격 조건을 한 번에 잰다.

    python scripts/acceptance.py out/sib/s101 out/sib/s040

조건
----
1. **세로 얼룩 없음.** 스펙트로그램의 세로 줄은 에너지가 **모든 주파수에서 동시에**
   급변할 때 생긴다. 페이드 인/아웃이 구간 안에서 되풀이되면 그렇게 보인다.
   그래서 대역별 dB 의 프레임 간 변화를 **대역 평균**해서 잰다(스펙트럼 플럭스):

       flux(t) = mean_k |dB(t,k) − dB(t−1,k)|

   자연 음성에도 성문 펄스와 파열음 버스트가 있으므로 **목표 대비**로 본다.

2. **스펙트럼 일치 — 특히 부각되는 신호(다이폴)의 추이.** 시간평균 스펙트럼만
   맞으면 안 된다. 앞니 다이폴처럼 두드러지는 성분이 **시간에 따라 어떻게
   움직이는가**가 같아야 하므로, 프레임별 무게중심과 고역 기울기의 **궤적**을
   상관과 오차로 잰다.

3. **파형 장기 추이곡선 일치.** 25 ms / 100 ms rms 포락선의 상관과 dB 오차.

4. **지터 낮음.** 적합이 내놓은 `jitter`·`shimmer` 궤적의 중앙값. 이 값이 곧
   학습 데이터로 나가는 물리량이다. 사람의 모달 발성은 지터 0.2~1 % 다.

합격선은 `THRESHOLDS` 에 있고, 근거는 각 항목의 주석에 적었다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import soundfile as sf

from formant_ml.engine import turbulence as tb
from formant_ml.engine.control import INDEX

FS = 48000.0

#: 대역 가장자리 [Hz] — 로그 간격 16 개. 200 Hz 아래는 방 잡음, 16 kHz 위는 코덱.
FLUX_EDGES = np.geomspace(200.0, 16000.0, 17)

THRESHOLDS = dict(
    # 목표의 p95 대비 합성의 p95. 1.0 이 "목표와 같은 만큼 급변한다".
    flux_p95_ratio=1.25,
    # 목표의 p99 를 넘는 합성 프레임의 비율 [%]. 목표에도 1 % 는 있으므로 여유를 준다.
    flux_over_pct=3.0,
    # 시간평균 스펙트럼의 대역 MAE [dB].
    band_mae_db=0.5,
    # 무게중심 궤적: 상관과 중앙 절대오차 [Hz].
    centroid_r=0.90, centroid_err_hz=300.0,
    # 고역(2~16 kHz) 기울기 궤적: 상관과 중앙 절대오차 [dB/oct].
    tilt_r=0.80, tilt_err_db=1.5,
    # 25 ms 포락선: 상관과 rms 오차 [dB].
    env_r=0.95, env_rms_db=2.0,
    # 지터·시머 [비율]. 사람 모달 발성 0.2~1 % / 시머 2~5 %.
    jitter=0.010, shimmer=0.050,
)


def _read(path):
    y, sr = sf.read(path)
    y = np.asarray(y.mean(1) if y.ndim > 1 else y, float)
    if sr != FS:
        from scipy.signal import resample_poly
        g = np.gcd(int(sr), int(FS))
        y = resample_poly(y, int(FS) // g, sr // g)
    return y


def _bands_db(x, n_fft=512, hop=240):
    """프레임별 대역 dB (T, K). hop 240 = 5 ms."""
    w = np.hanning(n_fft + 1)[:n_fft]
    t = max(1, 1 + (len(x) - n_fft) // hop)
    fr = np.stack([x[i * hop:i * hop + n_fft] * w for i in range(t)])
    P = np.abs(np.fft.rfft(fr, axis=-1)) ** 2
    f = np.fft.rfftfreq(n_fft, 1.0 / FS)
    out = np.empty((t, len(FLUX_EDGES) - 1))
    for k, (lo, hi) in enumerate(zip(FLUX_EDGES[:-1], FLUX_EDGES[1:])):
        m = (f >= lo) & (f < hi)
        out[:, k] = 10 * np.log10(P[:, m].sum(1) + 1e-20)
    return out


def flux(x, live):
    """대역 평균 |ΔdB| — 세로 줄의 자. `live` 는 프레임 마스크."""
    b = _bands_db(x)
    d = np.abs(np.diff(b, axis=0)).mean(1)
    m = live[:len(d)]
    return d[m] if m.any() else d


def centroid_tilt(x, live, n_fft=1024, hop=480):
    """프레임별 (무게중심 Hz, 2~16 kHz 기울기 dB/oct). 1 kHz 아래는 뺀다."""
    w = np.hanning(n_fft + 1)[:n_fft]
    t = max(1, 1 + (len(x) - n_fft) // hop)
    fr = np.stack([x[i * hop:i * hop + n_fft] * w for i in range(t)])
    P = np.abs(np.fft.rfft(fr, axis=-1)) ** 2
    f = np.fft.rfftfreq(n_fft, 1.0 / FS)
    hi = f >= 1000.0
    cen = (f[hi] * P[:, hi]).sum(1) / np.maximum(P[:, hi].sum(1), 1e-20)
    sl = (f >= 2000.0) & (f <= 16000.0)
    lf = np.log2(f[sl])
    A = np.stack([lf, np.ones_like(lf)], 1)
    db = 10 * np.log10(P[:, sl] + 1e-20)
    coef = np.linalg.lstsq(A, db.T, rcond=None)[0]
    m = live[:t]
    return cen[m], coef[0][m]


def envelope_db(x, win_ms):
    w = int(win_ms * 1e-3 * FS)
    m = len(x) // w
    return 10 * np.log10((x[:m * w] ** 2).reshape(m, w).mean(1) + 1e-20)


def _r(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 4 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def check(stem: str) -> dict:
    tgt, syn = _read(stem + "_target.wav"), _read(stem + "_fit.wav")
    n = min(len(tgt), len(syn))
    tgt, syn = tgt[:n], syn[:n]
    d = np.load(stem + "_track.npz")
    v, fm = d["values"], float(d["frame_ms"])
    ps = v[:, INDEX["p_sub"]]

    def live_at(hop_samples):
        """소리가 나는 프레임만. 무음의 −200 dB 가 상관을 부풀리는 것을 막는다."""
        step = hop_samples * 1000.0 / FS / fm
        idx = (np.arange(int(n / hop_samples)) * step).astype(int).clip(0, len(ps) - 1)
        return ps[idx] > 2.0

    out = {}
    lf = live_at(240)
    ft, fs_ = flux(tgt, lf), flux(syn, lf)
    out["flux_p95_ratio"] = float(np.percentile(fs_, 95) / max(np.percentile(ft, 95), 1e-9))
    out["flux_over_pct"] = float(100.0 * (fs_ > np.percentile(ft, 99)).mean())
    out["_flux"] = (float(np.percentile(ft, 95)), float(np.percentile(fs_, 95)))

    c = tb.compare(tgt, syn, FS)
    out["band_mae_db"] = float(c["band_mae_db"])
    lc = live_at(480)
    ct, tt = centroid_tilt(tgt, lc)
    cs, ts = centroid_tilt(syn, lc)
    out["centroid_r"] = _r(ct, cs)
    out["centroid_err_hz"] = float(np.median(np.abs(ct[:len(cs)] - cs[:len(ct)])))
    out["tilt_r"] = _r(tt, ts)
    out["tilt_err_db"] = float(np.median(np.abs(tt[:len(ts)] - ts[:len(tt)])))

    et, es = envelope_db(tgt, 25.0), envelope_db(syn, 25.0)
    k = min(len(et), len(es))
    keep = et[:k] > et[:k].max() - 60.0
    out["env_r"] = _r(et[:k][keep], es[:k][keep])
    out["env_rms_db"] = float(np.sqrt(((et[:k][keep] - es[:k][keep]) ** 2).mean()))
    e2t, e2s = envelope_db(tgt, 100.0), envelope_db(syn, 100.0)
    k2 = min(len(e2t), len(e2s))
    out["env100_r"] = _r(e2t[:k2], e2s[:k2])

    voiced = ps > 2.0
    out["jitter"] = float(np.median(v[voiced, INDEX["jitter"]]))
    out["shimmer"] = float(np.median(v[voiced, INDEX["shimmer"]]))
    return out


LOWER_IS_BETTER = {"flux_p95_ratio", "flux_over_pct", "band_mae_db", "centroid_err_hz",
                   "tilt_err_db", "env_rms_db", "jitter", "shimmer"}


def report(stem: str, res: dict) -> bool:
    print(f"\n{os.path.basename(stem)}   (목표 flux p95 {res['_flux'][0]:.2f} dB "
          f"-> 합성 {res['_flux'][1]:.2f} dB)")
    ok_all = True
    groups = (("1 세로 얼룩", ("flux_p95_ratio", "flux_over_pct")),
              ("2 스펙트럼·추이", ("band_mae_db", "centroid_r", "centroid_err_hz",
                                  "tilt_r", "tilt_err_db")),
              ("3 장기 추이곡선", ("env_r", "env_rms_db")),
              ("4 지터", ("jitter", "shimmer")))
    for title, keys in groups:
        print(f"  {title}")
        for k in keys:
            got, want = res[k], THRESHOLDS[k]
            ok = (got <= want) if k in LOWER_IS_BETTER else (got >= want)
            ok_all &= bool(ok)
            print(f"    {'OK ' if ok else '**' } {k:18s} {got:9.3f}   "
                  f"{'≤' if k in LOWER_IS_BETTER else '≥'} {want}")
    print(f"    (참고) env100_r {res['env100_r']:.3f}")
    return ok_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    allres = {}
    passed = 0
    for s in a.stems:
        r = check(s)
        allres[s] = {k: v for k, v in r.items() if not k.startswith("_")}
        passed += bool(report(s, r))
    print(f"\n합격 {passed}/{len(a.stems)}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(allres, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
