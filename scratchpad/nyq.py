import sys; sys.path.insert(0,"src")
import numpy as np, torch
from formant_ml.engine.tube import webster_modes, loss_bandwidths, uniform_area, C_SOUND
FS=48000.0; L=14.6
A=uniform_area(44,3.0,dtype=torch.float64)
fr=np.linspace(20,23990,6000); w=2*np.pi*fr; z=np.exp(-2j*np.pi*fr/FS)
def analog(f,b):
    s=1j*w; H=np.ones_like(s)
    for fk,bk in zip(f,b):
        sk=-np.pi*bk+2j*np.pi*fk; H=H*(sk*np.conj(sk))/((s-sk)*(s-np.conj(sk)))
    return 20*np.log10(abs(H)+1e-30)
def klatt(f,b,nyq_zero):
    H=np.ones_like(z)
    for fk,bk in zip(f,b):
        r=np.exp(-np.pi*bk/FS); a1=-2*r*np.cos(2*np.pi*fk/FS); a2=r*r
        den=1+a1*z+a2*z*z
        if nyq_zero: num=((1+a1+a2)/4.0)*(1+z)**2
        else:        num=(1+a1+a2)*np.ones_like(z)
        H=H*num/den
    return 20*np.log10(abs(H)+1e-30)
f,p=webster_modes(A,L); b=loss_bandwidths(A,L,f,p,wall_shape=1.3)
f,b=f.numpy(),b.numpy(); m=f<0.995*FS/2; ft,bt=f[m],b[m]
ref=analog(ft,bt)
print(f"극 {len(ft)} 개 (최고 {ft[-1]:.0f} Hz).  아날로그 기준 대비 이산 구현의 오차")
print(f"{'대역 [kHz]':>12s} {'아날로그':>9s} {'Klatt':>9s} {'오차':>8s} {'+나이퀴영점':>11s} {'오차':>8s}")
kc=klatt(ft,bt,False); kz=klatt(ft,bt,True)
def bd(d,lo,hi):
    s=(fr>=lo*1000)&(fr<hi*1000); return d[s].mean()
for lo,hi in [(0.2,1),(1,3),(3,5),(5,8),(8,11),(11,15),(15,20),(20,23.9)]:
    a,c1,c2=bd(ref,lo,hi),bd(kc,lo,hi),bd(kz,lo,hi)
    print(f"{lo:5.1f}-{hi:4.1f} {a:9.1f} {c1:9.1f} {c1-a:8.1f} {c2:11.1f} {c2-a:8.1f}")
print()
print("포먼트 8 개만 (현행 K) 일 때 두 이산 형식의 차이")
f8=ft[:8]; b8=bt[:8]
a8=analog(f8,b8); c8=klatt(f8,b8,False); z8=klatt(f8,b8,True)
for lo,hi in [(0.2,1),(1,3),(3,5),(5,8),(8,11),(11,15)]:
    print(f"{lo:5.1f}-{hi:4.1f} {bd(a8,lo,hi):9.1f} {bd(c8,lo,hi):9.1f} {bd(c8,lo,hi)-bd(a8,lo,hi):8.1f}"
          f" {bd(z8,lo,hi):11.1f} {bd(z8,lo,hi)-bd(a8,lo,hi):8.1f}")
