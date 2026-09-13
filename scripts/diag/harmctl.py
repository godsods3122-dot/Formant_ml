"""대조군 — **목표 대 목표(잡음 위상만 새로)** 에서도 배음 오차가 3 dB 나오는가.

나오면 그 3 dB 는 줄일 수 없는 실현 잡음이고, 우리가 고칠 것은 그 위의 몫뿐이다.
"""
import sys, numpy as np, soundfile as sf
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from formant_ml.engine.profile import SpeakerProfile
from formant_ml.engine.segment import fricative_mask
from formant_ml.engine.analyze import analyze
from scipy.signal import stft, istft
prof = SpeakerProfile.load("profiles/yang_female.json")
NW = 2048

def rephase(x, w_fr, seed):
    rng = np.random.default_rng(seed)
    f, t_, Z = stft(x, 48000, nperseg=1024, noverlap=1024-256)
    M, P = np.abs(Z), np.angle(Z)
    w = np.asarray(w_fr, float).ravel()
    if len(w) < P.shape[1]: w = np.pad(w, (0, P.shape[1]-len(w)), mode="edge")
    w = np.broadcast_to(w[None, :P.shape[1]], P.shape)
    P2 = np.where(w > 0.5, rng.uniform(-np.pi, np.pi, P.shape), P)
    _, y = istft(M*np.exp(1j*P2), 48000, nperseg=1024, noverlap=1024-256)
    return y[:len(x)]

def harm_err(a_, b_, fs, good, f0, mm):
    f, tt, A = stft(a_, fs, nperseg=NW, noverlap=NW-256)
    _, _, B = stft(b_, fs, nperseg=NW, noverlap=NW-256)
    A, B = np.abs(A), np.abs(B)
    out = []
    for j in range(min(A.shape[1], B.shape[1])):
        t_ms = int(j*256/48)
        if t_ms >= mm or not good[t_ms]: continue
        F0 = f0[t_ms]
        for k in range(1, 26):
            fc = k*F0
            if fc > 14000: break
            m = (f > fc-90) & (f < fc+90)
            if m.sum() < 3: continue
            out.append(abs(20*np.log10((B[m,j].max()+1e-20)/(A[m,j].max()+1e-20))))
    return np.array(out)

for stem in sys.argv[1:]:
    a_, fs = sf.read(stem+"_target.wav"); b_, _ = sf.read(stem+"_fit.wav")
    a_, b_ = np.asarray(a_,float), np.asarray(b_,float)
    n=min(len(a_),len(b_)); a_,b_=a_[:n],b_[:n]
    fm = np.asarray(fricative_mask(a_, fs, prof, 48), bool)
    tr = analyze(a_, fs, prof, 48, t0=0.0, full=a_)
    voi = np.asarray(tr.voiced,bool); f0=np.asarray(tr["f0_target"],float)
    mm=min(len(fm),len(voi),len(f0)); good = voi[:mm]&~fm[:mm]&(f0[:mm]>0)
    unv = (~voi[:mm]) | fm[:mm]
    nfr = 1 + len(a_)//256
    idx = np.clip((np.arange(nfr)*256)//48, 0, mm-1)
    w_fr = unv[idx].astype(float)
    e_fit = harm_err(a_, b_, fs, good, f0, mm)
    e_ctl = harm_err(a_, rephase(a_, w_fr, 3), fs, good, f0, mm)
    e_all = harm_err(a_, rephase(a_, np.ones_like(w_fr), 4), fs, good, f0, mm)
    print(f"\n== {stem}")
    print(f"   우리 적합          |Δ진폭| 중앙 {np.median(e_fit):5.2f} dB  95 % {np.percentile(e_fit,95):5.2f}")
    print(f"   **대조: 잡음 위상만 새로**  중앙 {np.median(e_ctl):5.2f} dB  95 % {np.percentile(e_ctl,95):5.2f}")
    print(f"   참고: 전 구간 위상 새로    중앙 {np.median(e_all):5.2f} dB  95 % {np.percentile(e_all,95):5.2f}")
