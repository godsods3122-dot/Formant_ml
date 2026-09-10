"""세 통계가 목표와 적합을 실제로 가르는가."""
import sys; sys.path.insert(0,"src")
import numpy as np, torch, soundfile as sf
FS=48000.
def stft(x,n):
    return torch.stft(torch.as_tensor(x,dtype=torch.float32).unsqueeze(0),n_fft=n,
        hop_length=n//4,win_length=n,window=torch.hann_window(n),center=True,
        return_complex=True,pad_mode="reflect")[0]
def db(S): return 20*torch.log10(S.abs()+1e-8)
def env(d,n,win_hz):
    k=int(round(win_hz/(FS/n)))|1
    return torch.nn.functional.conv1d(d.t().unsqueeze(1),torch.ones(1,1,k)/k,padding=k//2)[:,0].t()

for stem in ("out/fix/s040","out/fix/s101","out/pole/s040","out/pole/s101"):
    t,_=sf.read(f"{stem}_target.wav"); y,_=sf.read(f"{stem}_fit.wav")
    m=min(len(t),len(y)); t,y=np.asarray(t[:m],np.float32),np.asarray(y[:m],np.float32)
    print(f"=== {stem}")
    # --- (2a) 포먼트 대비, 짧은 창 ---
    n=256; dt,dy=db(stft(t,n)),db(stft(y,n))
    lvl=dt.mean(0); live=lvl>float(lvl.max())-45.0
    f=torch.arange(dt.shape[0])*FS/n; band=(f>=300)&(f<12000)
    ct=(dt-env(dt,n,2000.))[band].abs().mean(0); cy=(dy-env(dy,n,2000.))[band].abs().mean(0)
    print(f"  (2a) 포먼트 대비   목표 {float(ct[live].mean()):.3f}  적합 {float(cy[live].mean()):.3f}"
          f"  비 {float(cy[live].mean()/ct[live].mean()):.3f}")
    # --- (2b) 하모닉 첨예도, 긴 창 ---
    n=2048; Dt,Dy=db(stft(t,n)),db(stft(y,n))
    L=Dt.mean(0); LV=L>float(L.max())-45.0
    f=torch.arange(Dt.shape[0])*FS/n
    for lo,hi,w in ((100,4000,900.),(100,2000,700.)):
        b=(f>=lo)&(f<hi)
        ht=(Dt-env(Dt,n,w))[b].abs().mean(0); hy=(Dy-env(Dy,n,w))[b].abs().mean(0)
        print(f"  (2b) 하모닉 첨예 {lo}-{hi} 창{w:.0f}  목표 {float(ht[LV].mean()):.2f}  "
              f"적합 {float(hy[LV].mean()):.2f}  비 {float(hy[LV].mean()/ht[LV].mean()):.3f}")
    # --- (3) F0 아래 에너지 ---
    import parselmouth
    snd=parselmouth.Sound(np.asarray(t,np.float64),int(FS))
    pt=snd.to_pitch(time_step=(2048//4)/FS, pitch_floor=90, pitch_ceiling=700)
    T=Dt.shape[1]; f0=np.array([pt.get_value_at_time(i*(2048//4)/FS) or 0.0 for i in range(T)])
    ok=f0>50
    st=[];sy=[]
    for i in np.where(ok)[0]:
        m2=(f.numpy()>40)&(f.numpy()<0.75*f0[i])
        ref=(f.numpy()>=0.9*f0[i])&(f.numpy()<3*f0[i])
        st.append(float(Dt[m2,i].mean()-Dt[ref,i].mean())); sy.append(float(Dy[m2,i].mean()-Dy[ref,i].mean()))
    print(f"  (3) F0 아래 상대레벨  목표 {np.mean(st):+.2f} dB  적합 {np.mean(sy):+.2f} dB"
          f"  초과 {np.mean(sy)-np.mean(st):+.2f} dB   (유성 {ok.sum()}/{T})")
