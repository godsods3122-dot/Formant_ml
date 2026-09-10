"""엔진의 실제 전달함수를 측정한다 — 임펄스를 성도에 통과시켜 본다."""
import sys; sys.path.insert(0,"src")
import numpy as np, torch
from formant_ml.engine.tract import VocalTract
FS=48000.0; HOP=48
def tf(tract, form, n=1<<15):
    T=n//HOP+2
    c={}
    for k in range(1,9):
        c[f"f{k}"]=torch.full((1,T),float(form[k-1])); c[f"bw{k}"]=torch.zeros(1,T)
    for k in ("velum","nasal_z","lat_mix","lat_z1","lat_z2","lat_bw","front_f","front_g",
              "oral","a_c","fric_gain","tract_gain"):
        c.setdefault(k, torch.zeros(1,T))
    x=torch.zeros(1,n); x[0,0]=1.0
    tract._n_emit=n
    tracks=tract._formant_tracks(c)
    st={}
    y=tract._extra_cascade(tract._cascade(x,tracks,st,"f"),tracks,st)
    Y=np.fft.rfft(y[0].detach().numpy()); fr=np.fft.rfftfreq(n,1/FS)
    d=20*np.log10(abs(Y)+1e-30); return fr, d-d[np.argmin(abs(fr-100))]
import formant_ml.engine.tract as TR
f1=35000/(4*14.6); base=[(2*n-1)*f1 for n in range(1,9)]
tr=VocalTract(FS,HOP)
for f8 in (8990.,7900.,7000.):
    form=list(base); form[7]=f8
    fr,d=tf(tr,form)
    def bd(lo,hi):
        s=(fr>=lo*1000)&(fr<hi*1000); return d[s].mean()
    k=int(3000/(fr[1]-fr[0])); k+=(k+1)%2
    env=np.convolve(d,np.ones(k)/k,mode="same"); h=d-env
    m=(fr>5500)&(fr<13000)
    print(f"F8={f8:6.0f}  " + " ".join(f"{lo}-{hi}k {bd(lo,hi):6.1f}" for lo,hi in
          [(1,3),(3,5),(5,6),(6,8),(8,10),(10,12),(12,16)])
          + f"   구멍 {h[m].min():5.2f} dB @{fr[m][h[m].argmin()]:5.0f}")
