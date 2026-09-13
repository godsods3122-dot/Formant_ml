"""**어디를 고치면 몇 점인가** — 합성의 한 부분만 목표로 갈아 끼우고 env 를 다시 잰다.

추정이 아니라 직접 계산이다. 고칠 가치의 순서가 이걸로 정해진다.
"""
import sys, numpy as np, soundfile as sf
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from formant_ml.engine.fit import mel_bank, MEL_FFT
from formant_ml.engine.profile import SpeakerProfile
from formant_ml.engine.segment import fricative_mask
from formant_ml.engine.analyze import analyze
from scipy.signal import stft
prof = SpeakerProfile.load("profiles/yang_female.json")
FS = 48000
MEL = mel_bank(MEL_FFT, FS, 80).numpy()
fmel = np.array([np.argmax(MEL[i]) * (FS / MEL_FFT) for i in range(80)])

def sc(Ma, Mb):
    d = Ma - Mb
    return 100.0 * (1.0 - np.sqrt((d*d).sum()) / (np.sqrt((Ma*Ma).sum()) + 1e-9))

for stem in sys.argv[1:]:
    a_, fs = sf.read(stem+"_target.wav"); b_, _ = sf.read(stem+"_fit.wav")
    a_, b_ = np.asarray(a_,float), np.asarray(b_,float)
    n=min(len(a_),len(b_)); a_,b_=a_[:n],b_[:n]
    f,_t,A = stft(a_, FS, nperseg=MEL_FFT, noverlap=MEL_FFT-MEL_FFT//4)
    _,_,B = stft(b_, FS, nperseg=MEL_FFT, noverlap=MEL_FFT-MEL_FFT//4)
    k = min(A.shape[0], MEL.shape[1])
    Ma, Mb = MEL[:,:k]@np.abs(A[:k]), MEL[:,:k]@np.abs(B[:k])
    m = min(Ma.shape[-1], Mb.shape[-1]); Ma, Mb = Ma[...,:m], Mb[...,:m]
    fm = np.asarray(fricative_mask(a_, fs, prof, 48), bool)
    tr = analyze(a_, fs, prof, 48, t0=0.0, full=a_)
    voi = np.asarray(tr.voiced, bool)
    mm = min(len(fm), len(voi))
    idx = np.clip((np.arange(m)*(MEL_FFT//4))//48, 0, mm-1)
    vmask = voi[idx] & ~fm[idx]
    base = sc(Ma, Mb)
    print(f"\n== {stem}   지금 env {base:.2f} %")
    def what_if(name, sel_band=None, sel_time=None, factor=0.0):
        M2 = Mb.copy()
        wb = np.ones(80, bool) if sel_band is None else sel_band
        wt = np.ones(m, bool) if sel_time is None else sel_time
        ix = np.ix_(wb, wt)
        M2[ix] = Ma[ix] + factor*(Mb[ix] - Ma[ix])      # factor 0 = 완벽, 0.5 = 오차 절반
        print(f"   {name:34s} -> env {sc(Ma, M2):6.2f} %  ({sc(Ma,M2)-base:+5.2f})")
    for lo, hi in ((0,1000),(1000,3000),(3000,6000),(6000,24000)):
        wb = (fmel>=lo)&(fmel<hi)
        what_if(f"{lo//1000}-{hi//1000}k 를 완벽하게", wb, None, 0.0)
    for lo, hi in ((0,1000),(1000,3000)):
        wb = (fmel>=lo)&(fmel<hi)
        what_if(f"{lo//1000}-{hi//1000}k 오차를 **절반**으로", wb, None, 0.5)
    what_if("유성 프레임만 완벽하게", None, vmask, 0.0)
    what_if("유성 0-1k 만 완벽하게", (fmel<1000), vmask, 0.0)
    what_if("유성 0-1k 오차 절반", (fmel<1000), vmask, 0.5)
