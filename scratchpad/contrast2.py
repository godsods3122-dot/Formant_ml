import sys; sys.path.insert(0,"src")
import numpy as np, torch, soundfile as sf
FS=48000.0
def stft_db(x,n=256):
    w=torch.hann_window(n)
    S=torch.stft(torch.as_tensor(x,dtype=torch.float32).unsqueeze(0),n_fft=n,hop_length=n//4,
                 win_length=n,window=w,center=True,return_complex=True,pad_mode="reflect")
    return 20*torch.log10(S.abs()+1e-8)[0]
def resid(db,n=256,win_hz=1500.,lo=300.,hi=12000.):
    bw=FS/n; k=int(round(win_hz/bw))|1
    env=torch.nn.functional.conv1d(db.t().unsqueeze(1),torch.ones(1,1,k)/k,padding=k//2)[:,0].t()
    f=torch.arange(db.shape[0])*bw; m=(f>=lo)&(f<hi)
    return (db-env)[m]
VAR={
 "mean|r|":      lambda r: r.abs().mean(0),
 "std":          lambda r: r.std(0),
 "골만 mean":     lambda r: r.clamp(max=0.).abs().mean(0),
 "p90-p10":      lambda r: r.quantile(0.9,0)-r.quantile(0.1,0),
 "p95-p05":      lambda r: r.quantile(0.95,0)-r.quantile(0.05,0),
}
for stem in ("out/fix/s040","out/fix/s101"):
    t,_=sf.read(f"{stem}_target.wav"); y,_=sf.read(f"{stem}_fit.wav")
    dt=stft_db(np.asarray(t,np.float32)); dy=stft_db(np.asarray(y,np.float32))
    T=min(dt.shape[1],dy.shape[1]); dt,dy=dt[:,:T],dy[:,:T]
    lvl=dt.mean(0); live=lvl>float(lvl.max())-45.0
    print(stem)
    for win in (1200.,1500.,2000.):
        rt,ry=resid(dt,win_hz=win),resid(dy,win_hz=win)
        out=[]
        for nm,fn in VAR.items():
            a,b=fn(rt)[live].mean(),fn(ry)[live].mean()
            out.append(f"{nm} {float(b/a):.3f}")
        print(f"  창{win:5.0f}  " + "  ".join(out))
