"""잡음이 실제로 늘었는가 — 조화/비조화로 갈라서 잰다."""
import sys; sys.path.insert(0,"src")
import numpy as np, soundfile as sf
from formant_ml.engine.waveform import decompose
import parselmouth
FS=48000.0
def stats(path, f0, hop):
    x,fs=sf.read(path); x=np.asarray(x,np.float64)
    _,har,res=decompose(x,fs,f0,hop)
    def band(sig,lo,hi):
        n=1<<12; w=np.hanning(n); acc=None;c=0
        for i in range(0,len(sig)-n,n//2):
            S=np.abs(np.fft.rfft(sig[i:i+n]*w))**2
            acc=S if acc is None else acc+S; c+=1
        fr=np.fft.rfftfreq(n,1/fs); P=acc/max(c,1)
        m=(fr>=lo)&(fr<hi); return 10*np.log10(P[m].sum()+1e-30)
    out={}
    for lo,hi,nm in ((80,1000,"0.1-1k"),(1000,4000,"1-4k"),(4000,8000,"4-8k"),(8000,16000,"8-16k")):
        h=band(har,lo,hi); r=band(res,lo,hi)
        out[nm]=(h-r, r)
    tot_h=band(har,80,16000); tot_r=band(res,80,16000)
    out["전체"]=(tot_h-tot_r, tot_r)
    return out
y,sr=sf.read("data/voices/yang_00000040.wav"); y=np.asarray(y,np.float64)
snd=parselmouth.Sound(y,sr); hop=48
pt=snd.to_pitch(time_step=hop/sr, pitch_floor=90, pitch_ceiling=700)
n=len(y)//hop
f0=np.array([pt.get_value_at_time((i+0.5)*hop/sr) or 0.0 for i in range(n)])
print("조화 대 비조화 비 [dB] — 클수록 깨끗하다.  괄호는 비조화 성분의 절대 레벨")
print(f"{'':>12s} " + " ".join(f"{k:>16s}" for k in ("전체","0.1-1k","1-4k","4-8k","8-16k")))
for tag,p in (("목표","out/lad/s040_target.wav"),
              ("이전(fix)","out/fix/s040_fit.wav"),
              ("새(lad)","out/lad/s040_fit.wav"),
              ("pole","out/pole/s040_fit.wav")):
    s=stats(p,f0,hop)
    print(f"{tag:>12s} " + " ".join(f"{s[k][0]:7.1f} ({s[k][1]:6.1f})" for k in ("전체","0.1-1k","1-4k","4-8k","8-16k")))
