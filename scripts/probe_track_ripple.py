"""제어열의 **잔물결**을 잰다 — 분석기 초기값과 나란히 놓고 (MEASUREMENTS §26).

    python scripts/probe_track_ripple.py out/room2/s040 out/sw/r010

왜 이걸 재나
------------
`du(t) = amp(t)·Σ A_k(t)·cos(kφ(t))` 이므로 제어열이 프레임마다 흔들리면 그 변조가
반송파 F0 의 양쪽 측대역으로 퍼지고, 변조 대역이 F0 보다 넓으면 아래쪽 측대역이 DC
까지 접혀 **F0 아래 평평한 잡음 바닥**이 된다 (§26.1~26.3).

그 대역은 유성 프레임 총에너지의 1 % 도 안 되므로 **멜 포락 손실에는 안 보인다.**
귀에는 거칠기로 들린다. 그래서 손실이 아니라 이 자로 판정한다.

무엇과 비교하나
---------------
`analyze` 가 낸 초기값이다. 분석기는 이미 중앙값·이동평균으로 뭉개 놓았고 그 트랙에는
F0 아래 바닥이 없다 — 잔물결은 전적으로 적합기가 만든다 (실측 26 배 / 95 분위 131 배).
그러므로 **초기값의 곡률이 목표선**이고, 그 몇 배인지가 이 자의 값이다.

조음은 "대부분 정지, 짧은 순간 급전" 이라 곡률이 **희소하다**. 그래서 rms 보다
95 분위가 더 잘 가른다.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import soundfile as sf

from formant_ml.engine.analyze import analyze
from formant_ml.engine.control import INDEX, PARAM_NAMES
from formant_ml.engine.denoise import denoise, noise_profile
from formant_ml.engine.profile import DEFAULT_PROFILE, SpeakerProfile


def curvature(v: np.ndarray, frame_ms: float) -> np.ndarray:
    """|Δ²x|/dt² — 프레임 × 파라미터."""
    if v.shape[0] < 3:
        return np.zeros((0, v.shape[1]))
    return np.abs(np.diff(v, n=2, axis=0)) / (frame_ms * frame_ms)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--wav", default=None,
                    help="초기값 기준선을 낼 원본 wav. 없으면 기준선을 안 낸다")
    ap.add_argument("--profile", default="profiles/yang_female.json")
    ap.add_argument("--top", type=int, default=6, help="가장 심한 파라미터 몇 개를 볼까")
    a = ap.parse_args()
    prof = SpeakerProfile.load(a.profile) if os.path.exists(a.profile) else DEFAULT_PROFILE

    base = None
    if a.wav:
        y, sr = sf.read(a.wav)
        if y.ndim > 1:
            y = y.mean(1)
        y = np.asarray(y, float)
        y = denoise(y, sr, noise_profile(y, sr))
        tr0 = analyze(y, sr, prof, hop=max(1, int(round(sr / 1000.0))), t0=0.0, full=y)
        base = curvature(np.asarray(tr0.values, float), tr0.frame_ms)
        print(f"기준선 (analyze 초기값, {a.wav}):  "
              f"rms {np.sqrt((base ** 2).mean()):.3f}   95 분위 {np.percentile(base, 95):.3f}")

    print(f"\n{'':26s} {'곡률 rms':>10s} {'95 분위':>10s} {'배수(rms)':>10s} {'배수(95%)':>10s}")
    for s in a.stems:
        d = np.load(s + "_track.npz")
        c = curvature(d["values"].astype(float), float(d["frame_ms"]))
        r, q = np.sqrt((c ** 2).mean()), np.percentile(c, 95)
        if base is not None:
            br, bq = np.sqrt((base ** 2).mean()), np.percentile(base, 95)
            print(f"{os.path.basename(s):26s} {r:10.3f} {q:10.3f} "
                  f"{r / max(br, 1e-9):10.1f} {q / max(bq, 1e-9):10.1f}")
        else:
            print(f"{os.path.basename(s):26s} {r:10.3f} {q:10.3f} {'-':>10s} {'-':>10s}")
        # 가장 심한 파라미터
        per = np.percentile(c, 95, axis=0)
        top = np.argsort(per)[::-1][:a.top]
        print("      심한 파라미터: " + ", ".join(
            f"{PARAM_NAMES[i]} {per[i]:.2f}" for i in top))


if __name__ == "__main__":
    main()
