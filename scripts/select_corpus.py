"""코퍼스에서 **적합에 쓸 것만** 골라 낸다.

    python scripts/select_corpus.py --min 0.9 --max 2.6 --top 20

왜 고르는가
-----------
코퍼스 205 개(45 분)를 통째로 적합하는 것은 며칠 분량이고, 그중 상당수는 애초에
적합에 부적합하다 — 무음이 대부분이거나, 잡음 바닥이 높거나, 너무 짧다. 먼저 골라야
같은 시간에 더 많은 것을 배운다.

기준
----
* 길이 — "긴 음원" 이면서 한 번에 적합 가능한 범위 (기본 0.9~2.6 s)
* 잡음 바닥이 낮을 것
* 무음이 적을 것 (활성 프레임 비율)
* **진짜 치찰음**이 하나 이상 (치찰도 > 8 dB, 무게중심 > 6 kHz)

마지막 조건이 중요하다. 고역/저역비만으로 마찰을 세면 호흡·기식이 섞여
무게중심 중앙값이 5175 Hz 로 나온다 (프로파일 실측 8100 Hz, MEASUREMENTS §9.6).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, soundfile as sf
from formant_ml.engine.profile import SpeakerProfile
from formant_ml.engine import segment as sg
from formant_ml.engine import turbulence as tb

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--wavs", default="data/voices/*.wav")
ap.add_argument("--profile", default="profiles/yang_female.json")
ap.add_argument("--min", type=float, default=0.9, help="최소 길이 s")
ap.add_argument("--max", type=float, default=2.6, help="최대 길이 s")
ap.add_argument("--top", type=int, default=20)
ap.add_argument("--sibilant-only", action="store_true",
                help="진짜 치찰음이 든 것만")
a = ap.parse_args()

prof = SpeakerProfile.load(a.profile)
rows = []
for p in sorted(glob.glob(a.wavs)):
    info = sf.info(p)
    if not (a.min <= info.duration <= a.max):
        continue
    y, sr = sf.read(p)
    if y.ndim > 1: y = y.mean(1)
    y = np.asarray(y, float)
    w = int(0.02*sr); m = len(y)//w
    if m < 5: continue
    r = np.sqrt((y[:m*w].reshape(-1, w)**2).mean(1)+1e-20)
    db = 20*np.log10(r/ (r.max()+1e-20) + 1e-12)
    active = float((db > -35).mean())
    floor = float(np.percentile(20*np.log10(r+1e-12), 5))
    try:
        segs = sg.segments(y, sr, prof, file=os.path.basename(p))
    except Exception:
        continue
    nv = sum(1 for s in segs if s.kind == "vowel")
    nf = sum(1 for s in segs if s.kind == "fricative")
    # 치찰음이 진짜인지 (치찰도·무게중심)
    sib = 0
    for s in segs:
        if s.kind != "fricative": continue
        x = y[int(s.t0*sr):int(s.t1*sr)]
        if len(x) < int(0.05*sr): continue
        f = tb.sibilant_features(x, sr)
        if f["sibilance"] > 8 and f["centroid"] > 6000: sib += 1
    rows.append((os.path.basename(p), info.duration, active, floor, nv, nf, sib))

if a.sibilant_only:
    rows = [r for r in rows if r[6] > 0]
rows.sort(key=lambda r: (-r[6], -r[2], r[3]))
print(f"{'파일':>22s} {'초':>5s} {'활성%':>6s} {'바닥dB':>7s} {'모음':>4s} {'마찰':>4s} {'치찰':>4s}")
for r in rows[:a.top]:
    print(f"{r[0]:>22s} {r[1]:5.2f} {100*r[2]:6.1f} {r[3]:7.1f} {r[4]:4d} {r[5]:4d} {r[6]:4d}")
print(f"\n{a.min:g}~{a.max:g} s 파일 {len(rows)} 개 중 치찰음 포함 "
      f"{sum(1 for r in rows if r[6] > 0)} 개")
