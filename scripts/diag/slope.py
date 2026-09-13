"""배음 크기 오차가 **스펙트럼 기울기**와 함께 가는가.

부호 있는 오차는 0 인데 절대 오차가 3~4 dB 다. 정적인 기울기 오차가 아니라는 뜻이다.
포먼트 자리가 조금 어긋나면 **가파른 치마**에 놓인 배음만 크게 틀린다 — 그러면
|Δ진폭| 이 그 자리의 |dH/df| 에 비례해야 한다. 평탄한 곳에서도 3 dB 틀리면 딴 원인이다.
"""
import sys, numpy as np, soundfile as sf
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from formant_ml.engine.profile import SpeakerProfile
from formant_ml.engine.segment import fricative_mask
from formant_ml.engine.analyze import analyze
from scipy.signal import stft
prof = SpeakerProfile.load("profiles/yang_female.json")
NW = 2048

for stem in sys.argv[1:]:
    a_, fs = sf.read(stem + "_target.wav"); b_, _ = sf.read(stem + "_fit.wav")
    a_, b_ = np.asarray(a_, float), np.asarray(b_, float)
    n = min(len(a_), len(b_)); a_, b_ = a_[:n], b_[:n]
    fm = np.asarray(fricative_mask(a_, fs, prof, 48), bool)
    tr = analyze(a_, fs, prof, 48, t0=0.0, full=a_)
    voi = np.asarray(tr.voiced, bool); f0 = np.asarray(tr["f0_target"], float)
    mm = min(len(fm), len(voi), len(f0)); good = voi[:mm] & ~fm[:mm] & (f0[:mm] > 0)
    f, tt, A = stft(a_, fs, nperseg=NW, noverlap=NW-256)
    _, _, B = stft(b_, fs, nperseg=NW, noverlap=NW-256)
    A, B = np.abs(A), np.abs(B)
    dbA = 20*np.log10(A + 1e-12)
    # 켑스트럼 포락으로 매끈한 전달함수를 얻고 그 기울기를 잰다
    c = np.fft.irfft(np.log(A + 1e-12), axis=0); c[64:-64] = 0
    envA = np.fft.rfft(c, axis=0).real * (20/np.log(10))
    slope = np.gradient(envA, f[1]-f[0], axis=0)          # dB / Hz
    rows = []
    for j in range(min(A.shape[1], B.shape[1])):
        t_ms = int(j*256/48)
        if t_ms >= mm or not good[t_ms]: continue
        F0 = f0[t_ms]
        for k in range(1, 26):
            fc = k*F0
            if fc > 14000: break
            m = (f > fc-90) & (f < fc+90)
            if m.sum() < 3: continue
            ia = np.argmax(A[m, j]); ib = np.argmax(B[m, j])
            amp = 20*np.log10((B[m, j][ib]+1e-20)/(A[m, j][ia]+1e-20))
            sl = abs(slope[m, j][ia]) * 1000.0            # dB / kHz
            rows.append((abs(amp), sl, fc, k))
    v = np.array(rows, float)
    print(f"\n== {stem}   배음 {len(v)} 개")
    qs = np.percentile(v[:,1], [20,40,60,80])
    print("   |dH/df| 오분위        |Δ진폭| 중앙   95 %    n")
    edges = [0]+list(qs)+[1e9]
    for i in range(5):
        w = (v[:,1] >= edges[i]) & (v[:,1] < edges[i+1])
        if w.sum() < 20: continue
        print(f"   {edges[i]:7.1f}~{edges[i+1] if edges[i+1]<1e8 else 999:7.1f} dB/kHz   "
              f"{np.median(v[w,0]):7.2f}  {np.percentile(v[w,0],95):6.2f}  {int(w.sum()):6d}")
    r = np.corrcoef(v[:,1], v[:,0])[0,1]
    print(f"   상관(|기울기|, |Δ진폭|) = {r:+.3f}")
    # 평탄한 곳만 (하위 20 %) 에서도 오차가 남는가
    flat = v[:,1] < qs[0]
    print(f"   **가장 평탄한 20 % 에서도** |Δ진폭| 중앙 {np.median(v[flat,0]):.2f} dB")
