import sys; sys.path.insert(0,"src")
import numpy as np, torch
from formant_ml.engine.tube import webster_modes, loss_bandwidths, uniform_area
FS=48000.0; L=14.6
A=uniform_area(44,3.0,dtype=torch.float64)
fr=np.linspace(20,23990,8000); w=2*np.pi*fr; u=np.exp(-2j*np.pi*fr/FS)
def analog(f,b):
    s=1j*w; H=np.ones_like(s)
    for fk,bk in zip(f,b):
        sk=-np.pi*bk+2j*np.pi*fk; H=H*(sk*np.conj(sk))/((s-sk)*(s-np.conj(sk)))
    return 20*np.log10(abs(H)+1e-30)
def klatt(f,b):
    H=np.ones_like(u)
    for fk,bk in zip(f,b):
        r=np.exp(-np.pi*bk/FS); a1=-2*r*np.cos(2*np.pi*fk/FS); a2=r*r
        H=H*(1+a1+a2)/(1+a1*u+a2*u*u)
    return 20*np.log10(abs(H)+1e-30)
def bilin(f,b):
    """프리워핑 쌍선형.  g = tan(pi f/fs), Q = f/B.
       num = g^2 [1,2,1],  den = [1+g/Q+g^2, 2(g^2-1), 1-g/Q+g^2].  DC 이득 1 이 자동."""
    H=np.ones_like(u)
    for fk,bk in zip(f,b):
        g=np.tan(np.pi*fk/FS); Q=fk/bk; gq=g/Q; g2=g*g
        a0=1+gq+g2; a1=2*(g2-1); a2=1-gq+g2
        H=H*(g2*(1+2*u+u*u))/(a0+a1*u+a2*u*u)
    return 20*np.log10(abs(H)+1e-30)
f,p=webster_modes(A,L); b=loss_bandwidths(A,L,f,p,wall_shape=1.3)
f,b=f.numpy(),b.numpy()
mall=f<0.995*FS/2
ref=analog(f[mall],b[mall])           # 기준: 나이퀴스트까지 전 극, 아날로그
s=(fr>200)&(fr<12000)
print("기준 = 아날로그 전 극(22).  0.2~12 kHz 에서의 오차 [dB]")
print(f"{'형식':>8s} {'상한':>7s} {'극':>3s} {'RMS(모양)':>10s} {'RMS(절대)':>10s} {'최대':>7s}")
for name,fn in (("Klatt",klatt),("쌍선형",bilin)):
    for cap in (0.55,0.70,0.85,0.995):
        m=f<cap*FS/2
        d=fn(f[m],b[m]); e=d[s]-ref[s]
        print(f"{name:>8s} {cap:7.3f} {int(m.sum()):3d} {np.sqrt(((e-e.mean())**2).mean()):10.2f}"
              f" {np.sqrt((e**2).mean()):10.2f} {np.abs(e-e.mean()).max():7.2f}")
print()
m=f<0.995*FS/2
dk,db_=klatt(f[m],b[m]),bilin(f[m],b[m])
def bd(d,lo,hi):
    ss=(fr>=lo*1000)&(fr<hi*1000); return d[ss].mean()
print(f"{'대역 [kHz]':>12s} {'아날로그':>9s} {'Klatt':>8s} {'쌍선형':>8s}")
for lo,hi in [(0.2,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,8),(8,10),(10,12),(12,16),(16,20),(20,23.9)]:
    print(f"{lo:5.1f}-{hi:4.1f} {bd(ref,lo,hi):9.1f} {bd(dk,lo,hi):8.1f} {bd(db_,lo,hi):8.1f}")
