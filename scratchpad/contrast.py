"""공진 대비 통계를 시험한다 — 목표와 적합을 가르는가."""
import sys; sys.path.insert(0,"src")
import numpy as np, torch, soundfile as sf
FS=48000.0
def stft_db(x, n=256):
    w=torch.hann_window(n)
    S=torch.stft(torch.as_tensor(x,dtype=torch.float32).unsqueeze(0), n_fft=n,
                 hop_length=n//4, win_length=n, window=w, center=True,
                 return_complex=True, pad_mode="reflect")
    return 20*torch.log10(S.abs()+1e-8)[0]            # (F, T)
def contrast(db, fs=FS, n=256, win_hz=1200.0, lo=300.0, hi=12000.0):
    """국소 이동평균 대비 편차의 평균 |·|. 봉우리−골 구조가 깊을수록 크다."""
    bw=fs/n
    k=int(round(win_hz/bw))|1
    ker=torch.ones(1,1,k)/k
    x=db.t().unsqueeze(1)                              # (T,1,F)
    env=torch.nn.functional.conv1d(x, ker, padding=k//2)[:,0].t()
    r=(db-env).abs()
    f=torch.arange(db.shape[0])*bw
    m=(f>=lo)&(f<hi)
    return r[m].mean(0)                                # (T,)
for stem in ("out/fix/s040","out/fix/s101"):
    t,fs=sf.read(f"{stem}_target.wav"); y,_=sf.read(f"{stem}_fit.wav")
    dt=stft_db(np.asarray(t,dtype=np.float32)); dy=stft_db(np.asarray(y,dtype=np.float32))
    T=min(dt.shape[1],dy.shape[1]); dt,dy=dt[:,:T],dy[:,:T]
    lvl=dt.mean(0); live=lvl>float(lvl.max())-45.0
    for win in (800.,1200.,1800.):
        ct=contrast(dt,win_hz=win); cy=contrast(dy,win_hz=win)
        print(f"{stem} 창{win:5.0f} Hz  목표 {float(ct[live].mean()):5.2f}  "
              f"적합 {float(cy[live].mean()):5.2f}  비 {float(cy[live].mean()/ct[live].mean()):5.3f}")
    # 대역별
    print("   대역별 (창 1200 Hz):", end=" ")
    for lo,hi in ((300,1500),(1500,3000),(3000,5600),(5600,8000),(8000,12000)):
        ct=contrast(dt,win_hz=1200.,lo=lo,hi=hi); cy=contrast(dy,win_hz=1200.,lo=lo,hi=hi)
        print(f"{lo//1000}-{hi//1000}k {float(cy[live].mean()/ct[live].mean()):.3f}", end="  ")
    print()
