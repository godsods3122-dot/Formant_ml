"""대역폭 과대가 **분석기(관측)** 에서 온 것인가 **적합기**가 벌린 것인가."""
import sys, numpy as np, soundfile as sf
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from formant_ml.engine.profile import SpeakerProfile
from formant_ml.engine.segment import fricative_mask
from formant_ml.engine.analyze import analyze
prof = SpeakerProfile.load("profiles/yang_female.json")
for stem in sys.argv[1:]:
    y, fs = sf.read(stem + "_target.wav"); y = np.asarray(y, float)
    tr = analyze(y, fs, prof, 48, t0=0.0, full=y)
    voi = np.asarray(tr.voiced, bool)
    fm = np.asarray(fricative_mask(y, fs, prof, 48), bool)
    m = min(len(voi), len(fm)); good = voi[:m] & ~fm[:m]
    z = np.load(stem + "_track.npz", allow_pickle=True)
    nm = [str(x) for x in z["names"]]; V = np.asarray(z["values"], float)[:m]
    print(f"\n== {stem}   유성 비마찰 {good.sum()} 프레임")
    print("        관측(분석기)            적합 결과            생리 범위")
    for k in range(1, 5):
        o = np.asarray(tr[f"bw{k}"], float)[:m][good]
        f_ = V[:, nm.index(f"bw{k}")][good]
        lim = {1: "40~90", 2: "60~120", 3: "80~160", 4: "100~200"}[k]
        print(f"   bw{k}  {np.median(o):6.0f} ({np.percentile(o,25):5.0f}~{np.percentile(o,75):5.0f})   "
              f"{np.median(f_):6.0f} ({np.percentile(f_,25):5.0f}~{np.percentile(f_,75):5.0f})   {lim} Hz"
              f"   적합/관측 {np.median(f_)/max(np.median(o),1):.2f}")
    print("   Q = f/bw (관측 -> 적합).  실제 성도는 15~40")
    for k in range(1, 5):
        o = np.asarray(tr[f"bw{k}"], float)[:m][good]
        f_ = V[:, nm.index(f"bw{k}")][good]
        ff = V[:, nm.index(f"f{k}")][good]
        fo = np.asarray(tr[f"f{k}"], float)[:m][good]
        print(f"     Q{k}  {np.median(fo/np.maximum(o,1)):5.1f} -> {np.median(ff/np.maximum(f_,1)):5.1f}")
