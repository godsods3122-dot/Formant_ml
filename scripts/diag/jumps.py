"""도약 자 — 합성에서 **물리적으로 튄 곳**을 찾는다 (MEASUREMENTS §52.463).

사용자: *"어떠한 것도 물리적으로 jumping 이 있으면 안 돼. 그걸 평가하는 척도는 없어?"* — 없었다. 적합 손실의 가장 짧은 창은 256 표본(5.3 ms)이라
그 안에서 에너지만 맞으면 1~3 ms 의 튐은 안 보이고, 도약 벌점은 일부 파라미터(5 ms 차분의 두꺼운 꼬리 사전)에만 있었다.

**A 층 — 들리는 튐 (출력).** 대역마다(0.1–1.5 / 1.5–6 / 6–12 / 12–20 kHz — 발음 경계의 '퍽' 은 저역에 있다) 0.5 ms 포락 e(t) [dB] 를 0.125 ms 간격으로 재고, ±`BASE_MS` 이동 중앙값 b(t) 에 대한
솟음 p = e − b 를 본다. 합성의 솟음 봉우리마다
  * **초과 솟음** Δp = p_합성 − max(p_목표, ±`MATCH_MS`) — 목표에도 같은 자리에 솟음이 있으면(파열 개방, 녹음 클릭) 빠진다,
  * **초과 수준** ΔL = e_합성 − max(e_목표, ±`MATCH_MS`) — 합성이 그 순간 목표보다 실제로 큰가 (바닥만 낮아서 솟아 보이는 것을 거른다),
  * 둘 다 문턱(`DP_MIN`, `DL_MIN`)을 넘으면 **튐** 하나. 판의 점수 = 튐 수, Σ Δp.
**B 층 — 어디서 튀었나 (기구, `--track`).** 저장된 제어열에서
  * **주기별 보정의 경계 계단**: 보정은 주기마다 따로인 푸리에 계수라 주기 i 끝 값 Σa_k(i) 와 i+1 시작 값 Σa_k(i+1) 이 다르면 한 표본에 뛴다.
    |ΔS|·눈금·게이트 를 그 주기 보정 신호의 rms 로 나눈 값을 적는다 (연속이면 0),
  * **제어값의 한 틀 계단**: 면적·폭·주파수는 로그, 나머지는 선형으로, 한 틀 변화가 ±10 틀 변화의 중앙값보다 `STEP_X` 배 넘고 그 파라미터 범위의 5 % 를 넘는 곳.

    python scripts/diag/jumps.py out/C/W7b                       # A 층 (stem_target / stem_fit)
    python scripts/diag/jumps.py 목표.wav 합성.wav                # A 층, 파일 둘
    python scripts/diag/jumps.py out/C/W7b --track               # + B 층
    python scripts/diag/jumps.py --calibrate out/C/W7b            # 눈금: 알려진 클릭을 심어 되찾는다
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.signal import butter, sosfiltfilt

BANDS = ((100.0, 1500.0), (1500.0, 6000.0), (6000.0, 12000.0), (12000.0, 20000.0))
HOP_MS = 0.125
WIN_MS = 0.5
BASE_MS = 5.0
MATCH_MS = 1.5
DP_MIN = 6.0          # 초과 솟음 [dB]
DL_MIN = 3.0          # 초과 수준 [dB]
FLOOR_DB = 60.0       # 파일 대역 최대보다 이만큼 아래는 안 본다 (녹음 바닥)
STEP_X = 8.0


def band_env(x: np.ndarray, sr: int, lo: float, hi: float) -> tuple[np.ndarray, int]:
    hi = min(hi, 0.45 * sr)
    y = sosfiltfilt(butter(4, (lo, hi), "bp", fs=sr, output="sos"), x)
    w = max(2, int(round(WIN_MS * 1e-3 * sr)))
    hop = max(1, int(round(HOP_MS * 1e-3 * sr)))
    win = np.hanning(w + 2)[1:-1]
    e = np.convolve(y * y, win / win.sum(), mode="same")[::hop]
    return 10.0 * np.log10(e + 1e-20), hop


def prominence(e: np.ndarray, hop_s: float) -> np.ndarray:
    k = int(round(2 * BASE_MS * 1e-3 / hop_s)) | 1
    return e - median_filter(e, size=k, mode="nearest")


def events(xt: np.ndarray, xs: np.ndarray, sr: int, dp_min=DP_MIN, dl_min=DL_MIN) -> list[dict]:
    """A 층: 합성에만 있는 튐의 목록 (시각, 대역, Δp, ΔL, 합성 솟음, 합성 수준)."""
    n = min(len(xt), len(xs))
    xt, xs = xt[:n], xs[:n]
    out = []
    for lo, hi in BANDS:
        et, hop = band_env(xt, sr, lo, hi)
        es, _ = band_env(xs, sr, lo, hi)
        hop_s = hop / sr
        pt, ps = prominence(et, hop_s), prominence(es, hop_s)
        m = max(1, int(round(MATCH_MS * 1e-3 / hop_s)))
        pt_n = maximum_filter1d(pt, 2 * m + 1, mode="nearest")
        et_n = maximum_filter1d(et, 2 * m + 1, mode="nearest")
        floor = max(et.max(), es.max()) - FLOOR_DB
        pk = np.flatnonzero((ps[1:-1] >= ps[:-2]) & (ps[1:-1] > ps[2:])) + 1
        last = -10 ** 9
        for i in pk[np.argsort(-ps[pk])]:
            dp, dl = ps[i] - pt_n[i], es[i] - et_n[i]
            if dp < dp_min or dl < dl_min or es[i] < floor:
                continue
            out.append(dict(t=i * hop_s, band=(lo, hi), dp=float(dp), dl=float(dl), p=float(ps[i]), level=float(es[i])))
            last = i
    # 같은 사건이 대역 여럿에 잡히면 1 ms 안은 하나로 (가장 큰 Δp)
    out.sort(key=lambda d: -d["dp"])
    merged = []
    for d in out:
        if all(abs(d["t"] - k["t"]) > 1e-3 for k in merged):
            merged.append(d)
    return sorted(merged, key=lambda d: d["t"])


def score(ev: list[dict]) -> dict:
    return dict(n=len(ev), sum_dp=float(sum(d["dp"] for d in ev)), max_dp=float(max((d["dp"] for d in ev), default=0.0)))


# ---------------------------------------------------------------- B 층
def hcorr_steps(track: str, sr: int = 48000) -> list[dict]:
    z = np.load(track, allow_pickle=False)
    if "hcorr" not in z.files or z["hcorr"].size == 0:
        return []
    H = z["hcorr"]
    sc = float(z["hcorr_scale"])
    c0 = int(z["hcorr_c0"])
    u = (z["pulse_phase"] + float(z["pulse_phi0"])) / (2 * np.pi)
    fl = np.floor(u).astype(np.int64)
    cyc = np.clip(fl - c0, 0, H.shape[2] - 1)
    b = np.flatnonzero(np.diff(fl) > 0) + 1
    s0 = H[0].sum(0)                                             # 주기마다 φ=0 (=φ=1) 의 값 Σa_k
    k = np.arange(1, H.shape[1] + 1)[:, None]
    ph = np.linspace(0.0, 1.0, 64, endpoint=False)[None, :]
    rms = np.sqrt(np.mean(((H[0][:, :, None] * np.cos(2 * np.pi * k[..., None] * ph[None])
                            + H[1][:, :, None] * np.sin(2 * np.pi * k[..., None] * ph[None])).sum(0)) ** 2, axis=-1))
    out = []
    for i in b:
        a, c = cyc[i - 1], cyc[i]
        if a == c:
            continue
        step = abs(s0[c] - s0[a]) * sc
        ref = sc * max(rms[a], rms[c], 1e-12)
        out.append(dict(t=i / sr, cyc=(int(a), int(c)), step=float(step), rel=float(step / ref)))
    return out


def control_steps(track: str) -> list[dict]:
    z = np.load(track, allow_pickle=False)
    V, names = z["values"], [str(n) for n in z["names"]]
    fm = float(z["frame_ms"])
    logk = ("a_c", "f", "bw", "front_len", "lat_bw", "nasal_f", "p_sub")
    out = []
    for j, n in enumerate(names):
        v = V[:, j].astype(np.float64)
        if np.ptp(v) < 1e-9:
            continue
        if n == "a_c" or n.startswith(("f", "bw", "nasal_f", "front_len", "lat_bw")) and n not in ("front_q", "fric_gain", "f0_scale"):
            v = np.log(np.maximum(v, 1e-6))
        d = np.abs(np.diff(v))
        ref = median_filter(d, size=21, mode="nearest") + 1e-12
        big = np.flatnonzero((d > STEP_X * ref) & (d > 0.05 * np.ptp(v)))
        for i in big:
            out.append(dict(t=(i + 1) * fm / 1000.0, name=n, x=float(d[i] / ref[i]), frac=float(d[i] / np.ptp(v))))
    return sorted(out, key=lambda d: d["t"])


# ---------------------------------------------------------------- 눈금
def calibrate(stem: str, seed: int = 7) -> None:
    """합성에 알려진 클릭(광대역 감쇠 잡음 1 ms)을 심어 A 층이 되찾는지, 목표에도 같이 심으면 빠지는지 본다."""
    xt, sr = sf.read(stem + "_target.wav", dtype="float64")
    xs, _ = sf.read(stem + "_fit.wav", dtype="float64")
    n = min(len(xt), len(xs))
    xt, xs = xt[:n], xs[:n]
    base = {round(d["t"], 4) for d in events(xt, xs, sr)}
    rng = np.random.default_rng(seed)
    L = int(0.001 * sr)
    raw = rng.standard_normal(L) * np.exp(-np.arange(L) / (0.25e-3 * sr))
    kinds = {"고역 클릭(1.5 kHz 위)": (sosfiltfilt(butter(2, 1500.0, "hp", fs=sr, output="sos"), raw), (6000.0, 12000.0)),
             "저역 퍽(1.5 kHz 아래, 3 ms)": (sosfiltfilt(butter(2, 1500.0, "lp", fs=sr, output="sos"),
                                                        np.r_[raw, np.zeros(2 * L)] * np.exp(-np.arange(3 * L) / (1e-3 * sr))), (100.0, 1500.0))}
    # 심을 자리: 소리 있는 곳 전역에 고르게 8 곳
    lv = np.convolve(xt ** 2, np.ones(int(0.01 * sr)) / int(0.01 * sr), mode="same")
    live = np.flatnonzero(10 * np.log10(lv + 1e-20) > 10 * np.log10(lv.max()) - 40)
    pos = live[(np.linspace(0.08, 0.92, 8) * (len(live) - 1)).astype(int)]
    for kind, (burst, (blo, bhi)) in kinds.items():
      Lb = len(burst)
      print(f"눈금 [{kind}]: {len(pos)} 곳에 심는다 (심기 전부터 있던 튐 {len(base)} 개는 뺀다)")
      print(f"  심은 세기 = 그 자리 합성 {blo / 1000:g}–{bhi / 1000:g} kHz 국소 바닥 대비 [dB] → 되찾은 수 / 잰 Δp 중앙 | 목표에도 같이 심었을 때 잡힌 수")
      e6, hop = band_env(xs, sr, blo, bhi)
      b6 = e6 - prominence(e6, hop / sr)
      eb, _ = band_env(burst, sr, blo, bhi)
      for rel in (3.0, 6.0, 9.0, 12.0, 18.0):
        ys, yt = xs.copy(), xt.copy()
        for p in pos:
            g = 10 ** ((b6[p // hop] + rel - eb.max()) / 20.0)
            ys[p:p + Lb] += g * burst[: n - p]
            yt[p:p + Lb] += g * burst[: n - p]
        ev = [d for d in events(xt, ys, sr) if round(d["t"], 4) not in base]
        hit = [min(ev, key=lambda d: abs(d["t"] - p / sr)) for p in pos if any(abs(d["t"] - p / sr) < 3e-3 for d in ev)]
        both = [d for d in events(yt, ys, sr) if round(d["t"], 4) not in base
                and any(abs(d["t"] - p / sr) < 3e-3 for p in pos)]
        med = np.median([d["dp"] for d in hit]) if hit else float("nan")
        print(f"  +{rel:4.0f} dB → {len(hit)}/{len(pos)}  Δp 중앙 {med:5.1f} | 목표에도 심음: {len(both)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--track", action="store_true", help="B 층: stem_track.npz 의 보정 경계 계단·제어값 계단")
    ap.add_argument("--calibrate", metavar="STEM")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    if a.calibrate:
        calibrate(a.calibrate)
        return
    if a.b:
        tgt, syn, stem = a.a, a.b, None
    else:
        stem = a.a
        tgt, syn = stem + "_target.wav", stem + "_fit.wav"
    xt, sr = sf.read(tgt, dtype="float64")
    xs, _ = sf.read(syn, dtype="float64")
    ev = events(xt, xs, sr)
    s = score(ev)
    print(f"A 층 — 합성에만 있는 튐 {s['n']} 개, Σ초과솟음 {s['sum_dp']:.0f} dB, 최대 {s['max_dp']:.1f} dB  (문턱 Δp ≥ {DP_MIN:g}, ΔL ≥ {DL_MIN:g})")
    for d in sorted(ev, key=lambda d: -d["dp"])[: a.top]:
        lo, hi = d["band"]
        print(f"   {d['t']:8.4f} s  {lo / 1000:4.1f}–{hi / 1000:<4.1f} kHz  Δp {d['dp']:5.1f}  ΔL {d['dl']:5.1f}  (솟음 {d['p']:5.1f}, 수준 {d['level']:6.1f} dB)")
    if a.track and stem:
        hs = hcorr_steps(stem + "_track.npz", sr)
        if hs:
            r = np.array([h["rel"] for h in hs])
            print(f"B 층 — 주기별 보정의 경계 계단: 경계 {len(hs)} 곳, 계단/보정 rms 중앙 {np.median(r):.2f}, 90 % {np.percentile(r, 90):.2f}, 최대 {r.max():.2f}")
            for h in sorted(hs, key=lambda h: -h["step"])[:8]:
                print(f"   {h['t']:8.4f} s  주기 {h['cyc'][0]}→{h['cyc'][1]}  계단 {h['step']:.3f} (그 주기 보정 rms 의 {h['rel']:.1f} 배)")
            for d in ev:
                near = [h for h in hs if abs(h["t"] - d["t"]) < 1.5e-3]
                d["hstep"] = max((h["rel"] for h in near), default=0.0)
        cs = control_steps(stem + "_track.npz")
        print(f"B 층 — 제어값 한 틀 계단 {len(cs)} 곳 (한 틀 변화 > 주변 중앙값 ×{STEP_X:g}, 범위의 5 % 넘음)")
        for c in cs[: a.top]:
            print(f"   {c['t']:8.3f} s  {c['name']:12s} ×{c['x']:6.0f}  (범위의 {100 * c['frac']:4.1f} %)")


if __name__ == "__main__":
    sys.exit(main())
