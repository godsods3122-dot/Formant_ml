import sys; sys.path.insert(0,"src")
import numpy as np, torch
from formant_ml.engine.tube import webster_modes, loss_bandwidths, uniform_area, C_SOUND
FS=48000.0; L=14.6
A=uniform_area(44,3.0,dtype=torch.float64)
w=2*np.pi*np.linspace(20,24000,6000); fr=w/(2*np.pi)
def db(f,b):
    s=1j*w; H=np.ones_like(s)
    for fk,bk in zip(f,b):
        sk=-np.pi*bk+2j*np.pi*fk; H=H*(sk*np.conj(sk))/((s-sk)*(s-np.conj(sk)))
    return 20*np.log10(np.abs(H)+1e-30)
f,p=webster_modes(A,L); b=loss_bandwidths(A,L,f,p,wall_shape=1.3)
f,b=f.numpy(),b.numpy(); m=f<24000; ft,bt=f[m],b[m]
ref=db(ft,bt)
f1=C_SOUND/(4*L)
f8=np.array([(2*n-1)*f1 for n in range(1,9)])
ex=np.array([(2*n-1)*f1 for n in range(9,128) if (2*n-1)*f1<0.60*FS/2])
fe=np.concatenate([f8,ex])
law=40.0+0.05*f8
sel=(fr>200)&(fr<8000)
print("현행 8극 구조가 관의 0.2~8 kHz 포락을 따라가려면 대역폭을 몇 배로 벌려야 하나")
print(f"{'배율':>6s} {'RMS 오차 dB':>11s}")
best=None
for k in np.arange(1.0,4.01,0.1):
    be=np.concatenate([law*k,np.maximum(40+1.0*ex,800.0)])
    d=db(fe,be)
    e=np.sqrt(((d-d[sel].mean())-(ref-ref[sel].mean()))[sel].__pow__(2).mean())
    if best is None or e<best[1]: best=(k,e)
    if abs(k-round(k,1))<1e-9 and abs(k*10)%5<1e-9: print(f"{k:6.1f} {e:11.2f}")
print(f"\n-> 최적 배율 {best[0]:.1f}x  (RMS {best[1]:.2f} dB)")
print(f"   실측 피팅 대역폭은 손실법칙의 1.55~2.71 배 (MEASUREMENTS §34)")
print()
# 골 깊이
def valley(d,f_,lo,hi):
    s=(fr>=lo)&(fr<hi); return d[s]
kbest=best[0]
be=np.concatenate([law*kbest,np.maximum(40+1.0*ex,800.0)])
dcur=db(fe,be)
print("포먼트 사이 골 깊이 (봉우리−골, 2.6~6.5 kHz)")
for name,d in (("관(법칙 대역폭)",ref),(f"현행 8극 ×{kbest:.1f}",dcur)):
    vs=[]
    for k in range(2,6):
        lo,hi=f8[k],f8[k+1]; s=(fr>lo)&(fr<hi)
        pk=0.5*(d[np.argmin(abs(fr-lo))]+d[np.argmin(abs(fr-hi))])
        vs.append(pk-d[s].min())
    print(f"  {name:18s} " + " ".join(f"{v:5.2f}" for v in vs) + f"   평균 {np.mean(vs):5.2f} dB")
print()
print("== 피팅된 상황: F8 이 아래로 내려가면 구멍이 생기는가 ==")
def holedepth(fs_,bs_):
    d=db(fs_,bs_); k=int(3000/(fr[1]-fr[0])); k+=(k+1)%2
    env=np.convolve(d,np.ones(k)/k,mode="same"); h=d-env
    s=(fr>6000)&(fr<12000); return h[s].min(), fr[s][h[s].argmin()]
for f8v in (8990,8200,7600,7000):
    ff=f8.copy(); ff[7]=f8v
    fe2=np.concatenate([ff,ex]); be2=np.concatenate([(40+0.05*ff)*kbest,np.maximum(40+1.0*ex,800.0)])
    dep,at=holedepth(fe2,be2)
    print(f"  F8={f8v:5.0f} Hz -> 구멍 {dep:6.2f} dB @ {at:5.0f} Hz")
dep,at=holedepth(ft,bt)
print(f"  관 (극 22 개)  -> 구멍 {dep:6.2f} dB @ {at:5.0f} Hz")
