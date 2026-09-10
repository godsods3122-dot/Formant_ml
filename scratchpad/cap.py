import sys; sys.path.insert(0,"src")
import numpy as np, torch
from formant_ml.engine.tube import webster_modes, loss_bandwidths
FS=48000.0; L=14.6; N=44
fr=np.linspace(20,23990,6000); w=2*np.pi*fr; u=np.exp(-2j*np.pi*fr/FS)
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
def area(xc,deg,lip=1.0,ph=1.0):
    """조음 면적 함수: 위치 xc 에 세기 deg 의 협착, 인두/입술 개폐 ph/lip."""
    x=(np.arange(N)+0.5)/N
    la=np.log(3.0)+ph*(0.5-x)*0.8 - deg*np.exp(-0.5*((x-xc)/0.09)**2)
    la=la+np.log(lip)*np.clip((x-0.85)/0.15,0,1)
    return torch.tensor(np.exp(la))
cases=[("/a/ 인두협착",0.25,1.4,1.0,1.0),("/i/ 경구개",0.75,2.0,1.2,-0.6),
       ("/u/ 연구개+원순",0.55,1.6,0.35,0.2),("/e/",0.65,1.1,1.1,-0.2),
       ("/o/",0.40,1.2,0.5,0.4),("중성 근사",0.5,0.05,1.0,0.0),
       ("/s/ 치경",0.93,2.6,1.3,-0.3),("/sh/ 후치경",0.82,2.3,1.4,-0.2)]
s=(fr>200)&(fr<12000)
print("Klatt 종속의 상한(극을 어디까지 채우나)별 0.2~12 kHz 모양 오차 RMS [dB]")
caps=[0.55,0.60,0.65,0.70,0.75,0.80,0.90]
print(f"{'면적함수':>16s} " + " ".join(f"{c:6.2f}" for c in caps) + "   현행(0.60,Q≈1)")
tot={c:[] for c in caps}; cur=[]
for name,xc,dg,lp,ph in cases:
    A=area(xc,dg,lp,ph)
    f,p=webster_modes(A,L); b=loss_bandwidths(A,L,f,p,wall_shape=1.3)
    f,b=f.numpy(),b.numpy()
    ref=analog(f[f<0.995*FS/2],b[f<0.995*FS/2])
    row=[]
    for c in caps:
        m=f<c*FS/2; d=klatt(f[m],b[m]); e=d[s]-ref[s]
        v=np.sqrt(((e-e.mean())**2).mean()); row.append(v); tot[c].append(v)
    # 현행: 포먼트 8 + 고차 Q≈1, 상한 0.60
    m8=np.arange(len(f))<8; me=(f<0.60*FS/2)&(~m8)
    fc=np.concatenate([f[:8],f[me]]); bc=np.concatenate([40+0.05*f[:8],np.maximum(40+1.0*f[me],800.0)])
    d=klatt(fc,bc); e=d[s]-ref[s]; cv=np.sqrt(((e-e.mean())**2).mean()); cur.append(cv)
    print(f"{name:>16s} " + " ".join(f"{v:6.2f}" for v in row) + f"   {cv:8.2f}")
print(f"{'평균':>16s} " + " ".join(f"{np.mean(tot[c]):6.2f}" for c in caps) + f"   {np.mean(cur):8.2f}")
print()
print("6~12 kHz 최악의 구멍 [dB] (국소 3 kHz 포락 대비)")
def hole(d,lo=6000,hi=12000):
    k=int(3000/(fr[1]-fr[0])); k+=(k+1)%2
    env=np.convolve(d,np.ones(k)/k,mode="same"); h=d-env
    m=(fr>lo)&(fr<hi); return h[m].min(), fr[m][h[m].argmin()]
print(f"{'면적함수':>16s} {'현행':>16s} {'상한0.70+손실법칙':>20s}")
for name,xc,dg,lp,ph in cases:
    A=area(xc,dg,lp,ph)
    f,p=webster_modes(A,L); b=loss_bandwidths(A,L,f,p,wall_shape=1.3); f,b=f.numpy(),b.numpy()
    m8=np.arange(len(f))<8; me=(f<0.60*FS/2)&(~m8)
    fc=np.concatenate([f[:8],f[me]]); bc=np.concatenate([40+0.05*f[:8],np.maximum(40+1.0*f[me],800.0)])
    h1,a1=hole(klatt(fc,bc))
    m=f<0.70*FS/2; h2,a2=hole(klatt(f[m],b[m]))
    print(f"{name:>16s} {h1:8.2f} @{a1:5.0f} {h2:12.2f} @{a2:5.0f}")
