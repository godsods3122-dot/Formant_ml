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

4. **지터 낮음.** 두 가지를 같이 본다.

   (a) 적합이 내놓은 `jitter`·`shimmer` 궤적의 중앙값 — 이 값이 곧 학습 데이터로
       나가는 물리량이다. 사람의 모달 발성은 지터 0.2~1 % 다.
   (b) **실현 지터** — 합성음의 성문 펄스열에서 직접 잰 국소 지터를 **목표의 그것과
       비교**한다. (a) 는 파라미터일 뿐이고 실제로 소리에 실린 요동이 아니기
       때문이다: `f0_target` 이 펄스에서 온 비평활 트랙이라 화자의 실제 지터를 이미
       담고 있어서, `jitter`=0 에서도 목표만큼의 지터가 나온다 (MEASUREMENTS §28).
       그러니 "낮다" 가 아니라 **"목표와 같다"** 가 맞는 조건이다.

조건 2 에는 **F0 아래 초과**를 같이 잰다. 성문 소스는 k·F0 의 합이라 F0 아래에
아무것도 없어야 하는데, 제어열이 프레임마다 흔들리면 그 변조가 F0 의 아래쪽 측대역이
되어 DC 까지 접힌다 (MEASUREMENTS §26). 그 대역은 총에너지의 1 % 도 안 되므로
**멜 포락 손실에는 안 보이지만 귀에는 거칠기로 들린다** — 그래서 따로 잰다.

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
    # 실현 지터의 **목표 대비 비**. 1.0 이 "목표와 똑같이 흔들린다". 목표보다 크게
    # 흔들리면 거칠게 들리고, 작으면 기계적으로 들린다. 추정기 자체의 흩어짐이
    # 있으므로 ±35 % 를 준다 (실측: 같은 트랙 재렌더에서 1.61~1.68 %).
    jitter_ratio=1.35,
    # F0 **아래** 대역의 초과 [dB]. 하모닉이 원리적으로 없는 자리이므로 목표와 같아야
    # 한다. 측정 잡음(방·코덱·잡음제거 잔여)이 있으므로 3 dB 를 준다 — 실측에서
    # 잔물결이 살아 있으면 +15~+27 dB 가 나온다(§26.1).
    subf0_excess_db=3.0,
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


def subf0_excess(tgt, syn, live, f0_hz, n_fft=4096, hop=1024):
    """F0 **아래** 대역에서 합성이 목표보다 몇 dB 더 있는가.

    성문 소스는 k·F0 의 코사인 합이라 F0 아래에는 하모닉이 없다. 거기 남는 것은
    잡음 바닥과 **제어열 변조의 아래쪽 측대역**뿐이다. 방·코덱이 만드는 차이를
    피하려고 40 Hz 아래는 빼고, 0.75·F0 위도 뺀다 (F0 자체의 어깨가 섞인다).

    전체 레벨 차이에 오염되지 않게 **하모닉 대역(F0~4·F0)으로 정규화한 뒤** 뺀다.
    """
    w = np.hanning(n_fft + 1)[:n_fft]
    t = max(1, 1 + (len(tgt) - n_fft) // hop)
    if t < 2:
        return float("nan")
    keep = [i for i in range(t)
            if live[min(int(i * hop / 240), len(live) - 1)]]
    if len(keep) < 2:
        keep = list(range(t))
    f = np.fft.rfftfreq(n_fft, 1.0 / FS)
    lo = (f >= 40.0) & (f <= 0.75 * f0_hz)
    ref = (f >= f0_hz * 0.9) & (f <= 4.0 * f0_hz)
    out = []
    for x in (tgt, syn):
        fr = np.stack([x[i * hop:i * hop + n_fft] * w for i in keep])
        P = (np.abs(np.fft.rfft(fr, axis=-1)) ** 2).mean(0)
        out.append(10 * np.log10(P[lo].mean() / max(P[ref].mean(), 1e-30) + 1e-30))
    return float(out[1] - out[0])


def realized_jitter(x, f0_lo=70.0, f0_hi=500.0):
    """성문 펄스열에서 직접 잰 **국소 지터** [%] — mean|T_i−T_{i+1}| / mean(T) 의 중앙값.

    Praat 의 PointProcess (cc) 를 쓴다. 부분 표본 정밀도라 48 kHz 격자의 양자화가
    지터로 새지 않는다. 유성이 끊긴 자리는 주기 범위로 걸러 낸다.
    """
    try:
        import parselmouth
        from parselmouth.praat import call
    except Exception:
        return float("nan")
    snd = parselmouth.Sound(np.asarray(x, np.float64), int(FS))
    pt = snd.to_pitch(time_step=0.001, pitch_floor=max(60.0, f0_lo * 0.6),
                      pitch_ceiling=f0_hi * 1.3)
    pp = call([snd, pt], "To PointProcess (cc)")
    n = int(call(pp, "Get number of points"))
    if n < 6:
        return float("nan")
    p = np.array([call(pp, "Get time from index", i + 1) for i in range(n)])
    T = np.diff(p)
    ok = (T > 1.0 / f0_hi) & (T < 1.0 / f0_lo)
    good = ok[:-1] & ok[1:]
    if good.sum() < 10:
        return float("nan")
    d = np.abs(np.diff(T))[good]
    return float(100.0 * np.median(d / T[:-1][good]))


def _r(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 4 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def render_seed(stem: str, seed: int) -> np.ndarray | None:
    """적합 트랙을 **다른 난수 시드**로 다시 렌더한다.

    난류의 실현은 시드마다 다르고, 세로 얼룩·F0 아래 초과 같은 **분산 통계**는 그
    차이에 크게 흔들린다 (실측: 같은 설정이 flux 비 1.23 / 1.46 / 1.46).
    시드 하나로 설정을 비교하면 틀린 결론이 난다 (MEASUREMENTS §18 과 같은 교훈).
    """
    import torch

    from formant_ml.engine.control import ControlTrack
    from formant_ml.engine.profile import DEFAULT_PROFILE, SpeakerProfile
    from formant_ml.engine.voice import EngineConfig, VoiceEngine

    d = np.load(stem + "_track.npz")
    fm = float(d["frame_ms"])
    pf = "profiles/yang_female.json"
    prof = SpeakerProfile.load(pf) if os.path.exists(pf) else DEFAULT_PROFILE
    tr = ControlTrack(np.asarray(d["values"], float), frame_ms=fm)
    eng = VoiceEngine(EngineConfig(sample_rate=int(FS), frame_ms=fm,
                                   speaker="female" if prof.f0_nominal > 165 else "male",
                                   residual=False, seed=seed), prof)
    eng.reset()
    with torch.no_grad():
        y = eng(tr.to_tensor(), [], 0.0)["audio"].squeeze().numpy()
    ir = stem + "_room.npy"
    if os.path.exists(ir):
        from formant_ml.engine import room as _room
        y = _room.apply_ir(torch.as_tensor(y, dtype=torch.float32),
                           torch.as_tensor(np.load(ir), dtype=torch.float32)).numpy()
    return np.asarray(y, float)


def check(stem: str, seeds: int = 0) -> dict:
    tgt, syn = _read(stem + "_target.wav"), _read(stem + "_fit.wav")
    n = min(len(tgt), len(syn))
    tgt, syn = tgt[:n], syn[:n]
    d = np.load(stem + "_track.npz")
    v, fm = d["values"], float(d["frame_ms"])
    ps = v[:, INDEX["p_sub"]]

    # **소리가 나는 프레임의 기준은 목표의 레벨이다.** 전에는 적합 트랙의 `p_sub > 2`
    # 를 썼는데, 적합기는 무음 구간에서 `p_sub` 를 아무 데나 둔다 (손실이 그 자리에서
    # 0 이라 기울기가 없다). 실측: s040 의 목표 무음 21 프레임에서 `p_sub` 중앙이
    # 6.63 이라 **100 % 가 "소리남" 으로 셈해졌다.** 그 프레임들의 합성 레벨은
    # −82 dB 로 목표(−79 dB)보다 오히려 **조용한데**, −80 dB 짜리 대역 dB 는 잘게
    # 흔들려 세로 얼룩 지표를 통째로 오염시켰다 (s040 의 최악 구간 0.065~0.155 s 가
    # 통째로 무음이었다). 목표 정점 대비 −45 dB 를 문턱으로 쓴다.
    lvl_w = int(0.010 * FS)
    m_lvl = max(1, len(tgt) // lvl_w)
    lvl = 20 * np.log10(np.sqrt((tgt[:m_lvl * lvl_w] ** 2).reshape(m_lvl, lvl_w).mean(1)) + 1e-12)
    audible = lvl > lvl.max() - 45.0

    def live_at(hop_samples):
        """소리가 나는 프레임만. 무음의 −200 dB 가 상관을 부풀리는 것을 막는다."""
        step = hop_samples * 1000.0 / FS / fm
        idx = (np.arange(int(n / hop_samples)) * step).astype(int).clip(0, len(ps) - 1)
        li = (np.arange(int(n / hop_samples)) * hop_samples // lvl_w).clip(0, m_lvl - 1)
        return (ps[idx] > 2.0) & audible[li]

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
    # **−45 dB 다, −60 dB 가 아니다.** 이 화자의 녹음은 잡음 제거 뒤에도 무음이
    # 정점 대비 −50~−60 dB 에 있다(방 잔여). 거기까지 마스크를 열면 **목소리 모형이
    # 방 잡음을 재현하지 않는 것**을 오차로 센다 — 실측(out/fin/s040): 0.10~0.18 s
    # 에서 목표 −68~−80 dB 인데 합성이 −90 dB 로 25 dB 더 **조용해서** rms 오차가
    # 0.230 -> 4.553 dB 로 뛴다. 들리지도 않고 재현할 이유도 없는 양이다.
    # 문턱을 세로 얼룩·F0 아래 초과와 같은 −45 dB 로 맞춘다 (§30 과 같은 교훈,
    # 이번이 세 번째다 — **자를 먼저 의심할 것**).
    keep = et[:k] > et[:k].max() - 45.0
    out["env_r"] = _r(et[:k][keep], es[:k][keep])
    out["env_rms_db"] = float(np.sqrt(((et[:k][keep] - es[:k][keep]) ** 2).mean()))
    e2t, e2s = envelope_db(tgt, 100.0), envelope_db(syn, 100.0)
    k2 = min(len(e2t), len(e2s))
    out["env100_r"] = _r(e2t[:k2], e2s[:k2])

    voiced = ps > 2.0
    f0v = v[voiced, INDEX["f0_target"]]
    f0v = f0v[f0v > 0]
    f0m = float(np.median(f0v)) if len(f0v) else 200.0
    out["subf0_excess_db"] = subf0_excess(tgt, syn, lf, f0m)
    out["_f0"] = f0m
    jt, js = realized_jitter(tgt), realized_jitter(syn)
    out["_jit"] = (jt, js)
    out["jitter_ratio"] = float(js / jt) if jt and np.isfinite(jt) and jt > 0 else float("nan")
    out["jitter"] = float(np.median(v[voiced, INDEX["jitter"]]))
    out["shimmer"] = float(np.median(v[voiced, INDEX["shimmer"]]))

    # **시드에 민감한 통계는 여러 실현에서 잰다.** 아래 셋은 난류의 실현마다 크게
    # 흔들리므로 (§18) 한 벌로 설정을 비교하면 틀린 결론이 난다.
    if seeds > 1:
        keys = ("flux_p95_ratio", "flux_over_pct", "subf0_excess_db")
        acc_ = {k: [out[k]] for k in keys}
        for sd in range(1, seeds):
            y = render_seed(stem, sd)
            if y is None:
                continue
            n2 = min(len(y), len(tgt))
            g = np.sqrt((tgt[:n2] ** 2).mean() / max((y[:n2] ** 2).mean(), 1e-20))
            y = y[:n2] * g
            f2 = flux(y, lf)
            acc_["flux_p95_ratio"].append(
                float(np.percentile(f2, 95) / max(np.percentile(ft, 95), 1e-9)))
            acc_["flux_over_pct"].append(
                float(100.0 * (f2 > np.percentile(ft, 99)).mean()))
            acc_["subf0_excess_db"].append(subf0_excess(tgt[:n2], y, lf, f0m))
        for k in keys:
            a = np.array(acc_[k], float)
            out[k] = float(np.median(a))
            out["_" + k + "_rng"] = (float(a.min()), float(a.max()), len(a))
    return out


LOWER_IS_BETTER = {"flux_p95_ratio", "flux_over_pct", "band_mae_db", "centroid_err_hz",
                   "tilt_err_db", "env_rms_db", "jitter", "shimmer",
                   "subf0_excess_db"}


def report(stem: str, res: dict) -> bool:
    print(f"\n{os.path.basename(stem)}   (목표 flux p95 {res['_flux'][0]:.2f} dB "
          f"-> 합성 {res['_flux'][1]:.2f} dB)")
    ok_all = True
    groups = (("1 세로 얼룩", ("flux_p95_ratio", "flux_over_pct")),
              ("2 스펙트럼·추이", ("band_mae_db", "centroid_r", "centroid_err_hz",
                                  "tilt_r", "tilt_err_db", "subf0_excess_db")),
              ("3 장기 추이곡선", ("env_r", "env_rms_db")),
              ("4 지터", ("jitter", "shimmer", "jitter_ratio")))
    for title, keys in groups:
        print(f"  {title}")
        for k in keys:
            got, want = res[k], THRESHOLDS[k]
            if k == "jitter_ratio":
                # 1 을 가운데 두고 양쪽으로 본다 — 크면 거칠고 작으면 기계적이다.
                ok = np.isfinite(got) and (1.0 / want) <= got <= want
            else:
                ok = (got <= want) if k in LOWER_IS_BETTER else (got >= want)
            ok_all &= bool(ok)
            rel = (f"{1/want:.2f}~{want}" if k == "jitter_ratio"
                   else f"{'≤' if k in LOWER_IS_BETTER else '≥'} {want}")
            print(f"    {'OK ' if ok else '**' } {k:18s} {got:9.3f}   {rel}")
    print(f"    (참고) env100_r {res['env100_r']:.3f}   F0 중앙 {res['_f0']:.0f} Hz   "
          f"실현 지터 목표 {res['_jit'][0]:.2f} % -> 합성 {res['_jit'][1]:.2f} %")
    rng = {k[1:-4]: v for k, v in res.items() if k.endswith("_rng")}
    for k, (lo, hi, n) in rng.items():
        print(f"    (시드 {n} 벌) {k:18s} 범위 {lo:.3f} ~ {hi:.3f}")
    return ok_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--seeds", type=int, default=1,
                    help="시드 몇 벌로 잴까 (>1 이면 flux·F0 아래 초과를 중앙값으로 낸다)")
    a = ap.parse_args()
    allres = {}
    passed = 0
    for s in a.stems:
        r = check(s, seeds=a.seeds)
        allres[s] = {k: v for k, v in r.items() if not k.startswith("_")}
        passed += bool(report(s, r))
    print(f"\n합격 {passed}/{len(a.stems)}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(allres, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
