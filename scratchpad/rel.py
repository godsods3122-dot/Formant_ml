import sys; sys.path.insert(0,"src")
import numpy as np, torch
from formant_ml.engine.tube import webster_modes, loss_bandwidths, uniform_area, C_SOUND
FS=48000.0; L=14.6
fr=np.linspace(20,23990,6000); u=np.exp(-2j*np.pi*fr/FS)
def klatt(f,b):
    H=np.ones_like(u)
    for fk,bk in zip(f,b):
        r=np.exp(-np.pi*bk/FS); a1=-2*r*np.cos(2*np.pi*fk/FS); a2=r*r
        H=H*(1+a1+a2)/(1+a1*u+a2*u*u)
    return 20*np.log10(abs(H)+1e-30)
def hole(d,lo=5500,hi=13000):
    k=int(3000/(fr[1]-fr[0])); k+=(k+1)%2
    env=np.convolve(d,np.ones(k)/k,mode="same"); h=d-env
    m=(fr>lo)&(fr<hi); return h[m].min(), fr[m][h[m].argmin()]
f1=C_SOUND/(4*L); sp=C_SOUND/(2*L)
base=np.array([(2*n-1)*f1 for n in range(1,9)])
law=lambda x: 40.0+0.05*x
print(f"극 간격 c/2L = {sp:.0f} Hz.  F8 을 움직이며 6~13 kHz 최악의 구멍 [dB]")
print(f"{'F8':>7s} | {'현행: 고정 10188~ , Q≈1':>26s} | {'고정 위치 + 손실법칙':>22s} | {'F8 상대 + 손실법칙':>22s}")
ex_fix=np.array([(2*n-1)*f1 for n in range(9,128) if (2*n-1)*f1<0.60*FS/2])
ex_fix70=np.array([(2*n-1)*f1 for n in range(9,128) if (2*n-1)*f1<0.70*FS/2])
for f8 in (8990,8400,7900,7400,6900,6400):
    ff=base.copy(); ff[7]=f8
    r=[]
    fc=np.concatenate([ff,ex_fix]); bc=np.concatenate([law(ff),np.maximum(40+1.0*ex_fix,800.0)])
    r.append(hole(klatt(fc,bc)))
    fc=np.concatenate([ff,ex_fix70]); bc=np.concatenate([law(ff),law(ex_fix70)])
    r.append(hole(klatt(fc,bc)))
    ex_rel=np.array([f8+k*sp for k in range(1,64) if f8+k*sp<0.70*FS/2])
    fc=np.concatenate([ff,ex_rel]); bc=np.concatenate([law(ff),law(ex_rel)])
    r.append(hole(klatt(fc,bc)))
    print(f"{f8:7.0f} | {r[0][0]:9.2f} dB @{r[0][1]:6.0f} Hz | {r[1][0]:8.2f} @{r[1][1]:6.0f} | {r[2][0]:8.2f} @{r[2][1]:6.0f}")
print()
print("전 대역 레벨 [dB] (DC 대비),  F8=7600 인 경우")
f8=7600.0; ff=base.copy(); ff[7]=f8
ex_rel=np.array([f8+k*sp for k in range(1,64) if f8+k*sp<0.70*FS/2])
cfg={"현행": (np.concatenate([ff,ex_fix]), np.concatenate([law(ff),np.maximum(40+1.0*ex_fix,800.0)])),
     "F8상대+법칙": (np.concatenate([ff,ex_rel]), np.concatenate([law(ff),law(ex_rel)]))}
A=uniform_area(44,3.0,dtype=torch.float64)
fa,pa=webster_modes(A,L); ba=loss_bandwidths(A,L,fa,pa,wall_shape=1.3)
print(f"{'대역 [kHz]':>12s} " + "".join(f"{k:>13s}" for k in cfg))
for lo,hi in [(0.2,1),(1,3),(3,5),(5,6),(6,8),(8,10),(10,12),(12,16),(16,24)]:
    s=(fr>=lo*1000)&(fr<hi*1000)
    print(f"{lo:5.1f}-{hi:4.1f} " + "".join(f"{klatt(*cfg[k])[s].mean():13.1f}" for k in cfg))
print(f"\n고차 극 개수: 현행 {len(ex_fix)} 개, 새 안 {len(ex_rel)} 개 (biquad 총 {8+len(ex_fix)} -> {8+len(ex_rel)})")
