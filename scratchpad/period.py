"""미분 가능한 주기성 자 — F0 지연에서의 정규화 자기상관.

    r(t) = <x(t) x(t+T0)> / sqrt(<x(t)^2><x(t+T0)^2>)

Praat 의 HNR 이 쓰는 것과 같은 양이고, 텐서 연산만으로 쓸 수 있다.
잡음이 늘면 r 이 떨어진다. 대역을 나눠 재면 어느 대역이 시끄러운지도 보인다.
"""
import sys; sys.path.insert(0,"src")
import numpy as np, torch, soundfile as sf, parselmouth
FS=48000.0
def bandpass(x, lo, hi, fs=FS):
    n=len(x); X=np.fft.rfft(x); fr=np.fft.rfftfreq(n,1/fs)
    X=X*((fr>=lo)&(fr<hi)); return np.fft.irfft(X,n)
def periodicity(x, f0, hop, fs=FS, win_periods=3.0):
    """프레임별 정규화 자기상관. (T,)"""
    x=torch.as_tensor(x,dtype=torch.float64)
    T=len(f0); out=np.full(T,np.nan)
    for i in range(T):
        if not np.isfinite(f0[i]) or f0[i] < 50: continue
        lag=int(round(fs/f0[i])); w=int(win_periods*lag)
        c=i*hop
        a=c-w//2; b=a+w
        if a<0 or b+lag>len(x): continue
        u=x[a:b]; v=x[a+lag:b+lag]
        d=torch.sqrt((u*u).sum()*(v*v).sum())
        if float(d)<1e-20: continue
        out[i]=float((u*v).sum()/d)
    return out
y,sr=sf.read("data/voices/yang_00000040.wav"); y=np.asarray(y,np.float64)
hop=480   # 10 ms
snd=parselmouth.Sound(y,sr)
pt=snd.to_pitch(time_step=hop/sr, pitch_floor=90, pitch_ceiling=700)
n=len(y)//hop
f0=np.array([pt.get_value_at_time((i+0.5)*hop/sr) for i in range(n)]); f0=np.nan_to_num(f0,nan=0.0)
BANDS=((80,20000,"전대역"),(80,1000,"0.1-1k"),(1000,4000,"1-4k"),(4000,8000,"4-8k"))
print(f"{'':>12s} " + " ".join(f"{nm:>9s}" for _,_,nm in BANDS))
for tag,p in (("목표","out/lad/s040_target.wav"),("fix","out/fix/s040_fit.wav"),
              ("pole","out/pole/s040_fit.wav"),("lad","out/lad/s040_fit.wav")):
    x,fs=sf.read(p); x=np.asarray(x,np.float64)
    row=[]
    for lo,hi,_ in BANDS:
        r=periodicity(bandpass(x,lo,hi), f0, hop)
        row.append(np.nanmean(r))
    print(f"{tag:>12s} " + " ".join(f"{v:9.4f}" for v in row))
