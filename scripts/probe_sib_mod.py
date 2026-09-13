"""치찰음 고역 포락의 **변조 스펙트럼** — 토막이 어느 주파수에 있는가 (MEASUREMENTS §50.9).

    python scripts/probe_sib_mod.py s040 L25 L31 L34

가장 긴 마찰 구간의 4~16 kHz 포락(0.5 ms)을 로그로 잡아 5~20 / 20~60 / 60~150 / 150~450(F0) /
450~1000 Hz 대역의 변조 에너지를 목표 대비 dB 로 낸다. F0 대역이 크면 성문(성대가 안 멈춤),
20~60 Hz 가 크면 치찰음 안의 제어열 움직임이다. 목표는 `out/L25/<파일>_target.wav`.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, soundfile as sf
from census import _bp
from formant_ml.engine.profile import SpeakerProfile
from formant_ml.engine.segment import fricative_mask
lab = sys.argv[1]; tags = sys.argv[2:]
prof = SpeakerProfile.load("profiles/yang_female.json")
tg, fs = sf.read(f"out/L25/{lab}_target.wav"); tg = np.asarray(tg, float)
fm = fricative_mask(tg, fs, prof, 48)
# 가장 긴 마찰 구간
runs, s = [], None
for i, v in enumerate(list(fm) + [False]):
    if v and s is None: s = i
    if not v and s is not None: runs.append((s, i)); s = None
a, b = max(runs, key=lambda r: r[1] - r[0])
a, b = a + 10, b - 10
print(f"[{lab}] 마찰 구간 {a/1000:.3f}~{b/1000:.3f} s ({b-a} ms), 4~16 kHz 포락(0.5 ms)의 변조 에너지 [dB, 목표 대비]")
bands = ((5, 20), (20, 60), (60, 150), (150, 450), (450, 1000))
print(f"{'':8}" + "".join(f"{lo:>5}-{hi:<5}" for lo, hi in bands) + "   F0중앙")
def modspec(x):
    h = _bp(x, 4000, 16000, fs)[a * 48:b * 48]
    w = 24
    e = np.sqrt(np.convolve(h * h, np.ones(w) / w, "same"))[::w]      # 0.5 ms 포락
    e = np.log(e + 1e-12); e = e - e.mean()
    E = np.abs(np.fft.rfft(e * np.hanning(len(e)))) ** 2
    f = np.fft.rfftfreq(len(e), w / fs)
    return [10 * np.log10(E[(f >= lo) & (f < hi)].sum() + 1e-20) for lo, hi in bands]
ref = modspec(tg)
print(f"{'목표':8}" + "".join(f"{v:10.1f}" for v in ref) + "   (절대)")
for t in tags:
    y = np.asarray(sf.read(f"out/{t}/{lab}_fit.wav")[0], float)[:len(tg)]
    m = modspec(y)
    z = np.load(f"out/{t}/{lab}_track.npz", allow_pickle=True)
    names = list(z["names"]); v = z["values"]
    f0 = np.median(v[a:b, names.index("f0_target")])
    print(f"{t:8}" + "".join(f"{x - r:+10.1f}" for x, r in zip(m, ref)) + f"   {f0:.0f} Hz")
