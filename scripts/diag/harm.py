"""**배음 차수별**로 오차를 쪼갠다 — 창이 길수록 나빠지는 이유를 특정한다.

가설 A (진폭): 배음의 크기가 틀렸다 -> 오차가 차수 k 에 무관하게 깔린다.
가설 B (주파수): F0 가 어긋났다      -> 주파수 오차가 k 에 **비례**한다 (k·ΔF0).
가설 C (번짐):   지터·F0 흔들림      -> 봉우리 폭이 k 에 비례해 넓어진다.

세 가지는 예측이 다르므로 구분된다.
"""
import sys, numpy as np, soundfile as sf
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from formant_ml.engine.profile import SpeakerProfile
from formant_ml.engine.segment import fricative_mask
from formant_ml.engine.analyze import analyze
from scipy.signal import stft
prof = SpeakerProfile.load("profiles/yang_female.json")
NW = 2048                                   # 23 Hz 분해능 — 배음이 갈라진다

def peak(S, f, fc, half=90.0):
    m = (f > fc - half) & (f < fc + half)
    if m.sum() < 3: return None
    seg, fs_ = S[m], f[m]
    i = int(np.argmax(seg))
    if i == 0 or i == len(seg) - 1: return float(seg[i]), float(fs_[i]), np.nan
    # 3 점 포물선 보간으로 봉우리 자리, 그리고 −3 dB 폭
    y0, y1, y2 = np.log(seg[i-1] + 1e-20), np.log(seg[i] + 1e-20), np.log(seg[i+1] + 1e-20)
    den = y0 - 2*y1 + y2
    d = 0.5 * (y0 - y2) / den if den < -1e-9 else 0.0
    d = float(np.clip(d, -0.5, 0.5))                  # 보간은 반 빈을 못 넘는다
    fpk = fs_[i] + d * (fs_[1] - fs_[0])
    half_lv = seg[i] / np.sqrt(2.0)
    a = i
    while a > 0 and seg[a] > half_lv: a -= 1
    b = i
    while b < len(seg) - 1 and seg[b] > half_lv: b += 1
    return float(seg[i]), float(fpk), float((b - a) * (fs_[1] - fs_[0]))

for stem in sys.argv[1:]:
    a_, fs = sf.read(stem + "_target.wav"); b_, _ = sf.read(stem + "_fit.wav")
    a_, b_ = np.asarray(a_, float), np.asarray(b_, float)
    n = min(len(a_), len(b_)); a_, b_ = a_[:n], b_[:n]
    fm = np.asarray(fricative_mask(a_, fs, prof, 48), bool)
    tr = analyze(a_, fs, prof, 48, t0=0.0, full=a_)
    voi = np.asarray(tr.voiced, bool); f0 = np.asarray(tr["f0_target"], float)
    mm = min(len(fm), len(voi), len(f0)); good = voi[:mm] & ~fm[:mm] & (f0[:mm] > 0)
    f, tt, A = stft(a_, fs, nperseg=NW, noverlap=NW - 256)
    _, _, B = stft(b_, fs, nperseg=NW, noverlap=NW - 256)
    A, B = np.abs(A), np.abs(B)
    nf = min(A.shape[1], B.shape[1])
    rows = {}
    for j in range(nf):
        t_ms = int(j * 256 / 48)
        if t_ms >= mm or not good[t_ms]: continue
        F0 = f0[t_ms]
        for k in range(1, 21):
            fc = k * F0
            if fc > 16000: break
            pa, pb = peak(A[:, j], f, fc), peak(B[:, j], f, fc)
            if pa is None or pb is None: continue
            amp = 20*np.log10((pb[0]+1e-20)/(pa[0]+1e-20))
            df = pb[1] - pa[1]
            rows.setdefault(k, []).append((amp, df, 1200*np.log2(max(pb[1],1)/max(pa[1],1)),
                                           pa[2], pb[2], abs(amp)))
    print(f"\n== {stem}   (배음 봉우리, 2048 창 = 23 Hz)")
    print("   k    Δ진폭(부호) |Δ진폭|  95%|Δ|    Δ주파수 Hz   |Δ| 센트    n")
    for k in sorted(rows):
        v = np.array(rows[k], float)
        if len(v) < 30: continue
        print(f"  {k:3d}   {np.median(v[:,0]):+8.2f}  {np.median(v[:,5]):7.2f} {np.percentile(v[:,5],95):7.2f}   "
              f"{np.median(v[:,1]):+8.1f}   {np.median(np.abs(v[:,2])):7.1f}  {len(v):5d}")
