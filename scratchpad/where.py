"""포락 일치율의 남은 오차가 **어디**에 있는가 — 시간과 주파수로 쪼갠다."""
import sys; sys.path.insert(0,"src")
import numpy as np, torch, soundfile as sf
from formant_ml.engine.fit import mel_bank
FS=48000.0; N=256
def mel_db(x, mel):
    w=torch.hann_window(N)
    S=torch.stft(torch.as_tensor(x,dtype=torch.float32).unsqueeze(0),n_fft=N,hop_length=N//4,
                 win_length=N,window=w,center=True,return_complex=True,pad_mode="reflect").abs()[0]
    M=mel[:,:S.shape[0]]@S
    return 20*torch.log10(M+1e-6)
mel=mel_bank(N,FS,48,fmax=0.9*FS/2)
import itertools
for run,stem in itertools.product(("fix","pole"),("s040","s101")):
    t,fs=sf.read(f"out/{run}/{stem}_target.wav"); y,_=sf.read(f"out/{run}/{stem}_fit.wav")
    dt=mel_db(np.asarray(t,np.float32),mel); dy=mel_db(np.asarray(y,np.float32),mel)
    T=min(dt.shape[1],dy.shape[1]); dt,dy=dt[:,:T],dy[:,:T]
    lvl=dt.mean(0); live=lvl>float(lvl.max())-45.0
    e=(dy-dt)[:,live]
    print(f"=== {run}/{stem}  유효 프레임 {int(live.sum())}/{T},  |오차| 평균 {float(e.abs().mean()):.2f} dB")
    # 멜 밴드별
    fb=(mel*torch.arange(mel.shape[1])*FS/N).sum(1)/mel.sum(1).clamp_min(1e-9)
    q=[0,8,16,24,32,40,48]
    print("  밴드 중심 [Hz]  " + " ".join(f"{float(fb[a:b].mean()):7.0f}" for a,b in zip(q,q[1:])))
    print("  |오차| [dB]     " + " ".join(f"{float(e[a:b].abs().mean()):7.2f}" for a,b in zip(q,q[1:])))
    print("  편향 [dB]       " + " ".join(f"{float(e[a:b].mean()):+7.2f}" for a,b in zip(q,q[1:])))
    # 프레임별 상위 오차
    pf=e.abs().mean(0)
    idx=torch.argsort(pf,descending=True)[:8]
    tt=np.where(live.numpy())[0]
    print("  최악 프레임 [s]: " + " ".join(f"{tt[int(i)]*(N//4)/FS:.3f}({float(pf[i]):.1f})" for i in idx))
    # 상위 10 % 프레임이 전체 오차의 몇 %
    s=torch.sort(pf,descending=True).values
    k=max(1,len(s)//10)
    print(f"  상위 10 % 프레임이 전체 |오차| 의 {float(s[:k].sum()/s.sum())*100:.1f} %")
