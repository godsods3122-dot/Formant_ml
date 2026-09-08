"""치찰음 구간의 **비조화 초과**가 어디서 오는지 가른다.

    python scripts/probe_nonharmonic.py out/long/s101_track.npz \
        --wav data/voices/yang_00000101.wav --profile profiles/yang_female.json

무엇이 문제인가
---------------
회귀 벤치 (docs/MEASUREMENTS.md §9.12) 에서 유일하게 크게 벌어진 지표다 — 마찰이 든
구간에서 **합성이 목표보다 1.7~4.2 배 덜 주기적**이다. 모음에서는 0.8 % / 0.8 % 로
정확히 일치하므로 원인이 마찰 쪽에 있다.

어떻게 가르는가
---------------
적합이 끝난 제어열을 놓고 **한 번에 하나씩만 끄거나 뭉갠 뒤 다시 렌더**해서, 비조화
비율이 얼마나 내려가는지 본다. 크게 내려가는 것이 원인이다.

* `jitter=0` / `shimmer=0` — 성문의 주기 요동. 하모닉 차수에 비례해 위상변조 지수가
  커지므로(β ∝ k) 고차 하모닉을 통째로 뭉갠다. 실측으로 광대역 바닥을 33 dB 올린
  전력이 있다 (MEASUREMENTS §7.2).
* `aspiration=0` — 성문 기식 잡음.
* `fric_gain=0` — 마찰 소스. **이걸 끄면 치찰음 자체가 사라지므로** 비조화가 줄어드는
  것은 당연하다. "얼마나" 줄어드는지를 다른 항목의 기준으로 쓴다.
* `트랙 평활` — 제어열을 시간축으로 뭉갠다. 시변 필터의 **계수 변조**가 잡음을 만드는지
  본다 (§7 이 지목했던 후보). 소스는 그대로 두고 필터만 매끄럽게 하는 셈이다.

**대조군이 반드시 필요하다.** 목표 녹음의 비조화 비율을 같은 자로 재서 나란히 놓는다 —
0 이 정답이 아니다. 사람 목소리도 마찰 구간에서는 비조화가 크다.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import soundfile as sf
import torch

from formant_ml.engine.analyze import analyze
from formant_ml.engine.control import INDEX, PARAM_NAMES, ControlTrack
from formant_ml.engine.denoise import denoise, noise_profile
from formant_ml.engine.profile import SpeakerProfile
from formant_ml.engine.voice import EngineConfig, VoiceEngine
from formant_ml.engine.waveform import decompose

HOP = 48
FS = 48000.0


def nonharmonic(sig, f0, voiced, hop=HOP):
    """유성 프레임의 비조화 에너지 비율 (%). 벤치와 같은 자."""
    _, har, _ = decompose(sig, FS, f0, hop)
    v = []
    for i in range(min(len(voiced), len(sig) // hop)):
        if not voiced[i]:
            continue
        a, b = i * hop, (i + 1) * hop
        eh = (har[a:b] ** 2).sum()
        er = ((sig[a:b] - har[a:b]) ** 2).sum()
        if eh + er > 0:
            v.append(er / (eh + er))
    return 100 * float(np.median(v)) if v else float("nan")


def smooth_cols(values, names, cols, win_ms, frame_ms):
    """제어열의 일부 열만 시간축으로 뭉갠다."""
    out = values.copy()
    k = max(1, int(round(win_ms / frame_ms)))
    if k < 2:
        return out
    w = np.hanning(k + 2)[1:-1]
    w = w / w.sum()
    for c in cols:
        i = names.index(c)
        x = out[:, i]
        out[:, i] = np.convolve(np.pad(x, k // 2, mode="edge"), w,
                                mode="same")[k // 2:k // 2 + len(x)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("track", help="copyfit 이 저장한 *_track.npz")
    ap.add_argument("--wav", required=True, help="원본 wav (목표 대조군용)")
    ap.add_argument("--profile", default="profiles/yang_female.json")
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=None)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    d = np.load(a.track, allow_pickle=True)
    values = np.asarray(d["values"], dtype=np.float64)
    frame_ms = float(d["frame_ms"])
    names = [str(x) for x in d["names"]]
    prof = SpeakerProfile.load(a.profile)

    y, sr = sf.read(a.wav)
    if y.ndim > 1:
        y = y.mean(1)
    y = denoise(np.asarray(y, float), sr, noise_profile(np.asarray(y, float), sr))
    t1 = a.t1 if a.t1 is not None else len(y) / sr
    seg = y[int(a.t0 * sr):int(t1 * sr)]
    tr0 = analyze(seg, sr, prof, HOP, t0=a.t0, full=y)
    f0 = np.asarray(tr0["f0_target"])
    voi = np.asarray(tr0.voiced).astype(bool)
    fric = np.asarray(tr0.fricative).astype(bool)
    if sr != 48000:
        from scipy.signal import resample_poly
        g = np.gcd(int(sr), 48000)
        seg = resample_poly(seg, 48000 // g, int(sr) // g)

    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=frame_ms,
                                   speaker="female" if prof.f0_nominal > 165 else "male",
                                   residual=False), prof)

    def render(v):
        eng.reset()
        ctrl = torch.as_tensor(v, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return eng(ctrl, [], 0.0)["audio"][0].cpu().numpy()

    base = render(values)
    # 합성의 레벨을 목표에 맞춘다 (비조화 비율 자체는 레벨에 불변이지만 표를 읽기 쉽게)
    n = min(len(base), len(seg))
    tgt = seg[:n]

    def variant(label, v):
        y2 = render(v)[:n]
        nh_all = nonharmonic(y2, f0, voi)
        nh_fric = nonharmonic(y2, f0, voi & _dilate(fric, 20))
        return label, nh_all, nh_fric

    rows = [("목표 (대조군)", nonharmonic(tgt, f0, voi),
             nonharmonic(tgt, f0, voi & _dilate(fric, 20))),
            variant("적합 그대로", values)]

    for label, cols in (("지터 0", ["jitter"]), ("시머 0", ["shimmer"]),
                        ("지터+시머 0", ["jitter", "shimmer"]),
                        ("기식 0", ["aspiration"]), ("마찰 0", ["fric_gain"])):
        v = values.copy()
        for c in cols:
            v[:, INDEX[c]] = 0.0
        rows.append(variant(label, v))

    for win in (5.0, 15.0, 40.0):
        v = smooth_cols(values, names,
                        ["f1", "f2", "f3", "f4", "bw1", "bw2", "bw3", "bw4",
                         "a_c", "front_len", "tract_gain", "fric_gain"],
                        win, frame_ms)
        rows.append(variant(f"트랙 평활 {win:g} ms", v))

    print(f"\n{'조건':>16s} {'비조화 전체 %':>14s} {'비조화 마찰부 %':>16s}")
    for lab, x, z in rows:
        print(f"{lab:>16s} {x:14.1f} {z:16.1f}")
    print("\n목표 행이 대조군이다 — 0 이 정답이 아니다. '마찰 0' 은 치찰음 자체를 껐으므로")
    print("당연히 내려간다. 다른 항목이 그만큼 내려가면 그게 원인이다.")


def _dilate(mask, k):
    """마스크를 앞뒤로 k 프레임 넓힌다 (마찰 경계의 전이도 포함해서 본다)."""
    m = np.asarray(mask, bool)
    if not m.any():
        return m
    out = m.copy()
    for i in range(1, k + 1):
        out[i:] |= m[:-i]
        out[:-i] |= m[i:]
    return out


if __name__ == "__main__":
    main()
