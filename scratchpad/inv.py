"""A안 결정 시험: 면적 함수가 이 화자의 실제 포먼트에 닿는가."""
import sys; sys.path.insert(0,"src")
import numpy as np, torch
from formant_ml.engine.tube import webster_modes, loss_bandwidths
torch.set_num_threads(2)
d=np.load("out/fix/s040_track.npz"); names=[str(x) for x in d["names"]]; V=d["values"]
fi=[names.index(f"f{k}") for k in range(1,9)]
pi=names.index("p_sub") if "p_sub" in names else None
F=V[:,fi]
live=(F[:,0]>200)&(F[:,0]<1200)&(F[:,3]>2500)
print(f"유효 프레임 {live.sum()}/{len(F)}")
Fl=F[live]
# 대표 프레임 12 개 (F1-F2 공간에서 고르게)
idx=np.linspace(0,len(Fl)-1,12).astype(int)
targ=torch.tensor(Fl[idx][:,:5],dtype=torch.float64)
N=44; L=14.6
x=(torch.arange(N,dtype=torch.float64)+0.5)/N
def build(cs, M):
    B=torch.stack([torch.cos(k*np.pi*x) for k in range(M+1)])   # (M+1,N)
    return torch.exp(cs@B)*3.0
for M in (4,6,8,10):
    cs=torch.zeros(len(idx),M+1,dtype=torch.float64,requires_grad=True)
    opt=torch.optim.Adam([cs],lr=0.05)
    for it in range(1200):
        opt.zero_grad()
        A=build(cs,M).clamp_min(0.03)
        f=webster_modes(A,L,5)[0]
        loss=(((f-targ)/targ)**2).mean()+1e-4*(cs[:,1:]**2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        A=build(cs,M).clamp_min(0.03); f=webster_modes(A,L,5)[0]
        err=100*((f-targ)/targ).abs()
        print(f"M={M:2d} (계수 {M+1} 개)  F1~F5 상대오차 [%] 평균 " +
              " ".join(f"{v:.2f}" for v in err.mean(0).tolist()) +
              f"   최대 {err.max():.2f}   면적 {float(A.min()):.3f}~{float(A.max()):.2f} cm²")
# 최적 M 으로 8 개 포먼트까지
M=8; targ8=torch.tensor(Fl[idx][:,:8],dtype=torch.float64)
cs=torch.zeros(len(idx),M+1,dtype=torch.float64,requires_grad=True)
opt=torch.optim.Adam([cs],lr=0.05)
for it in range(2000):
    opt.zero_grad(); A=build(cs,M).clamp_min(0.03)
    f=webster_modes(A,L,8)[0]
    (((f-targ8)/targ8)**2).mean().backward(); opt.step()
with torch.no_grad():
    A=build(cs,M).clamp_min(0.03); f=webster_modes(A,L,8)[0]
    e=100*((f-targ8)/targ8).abs()
    print(f"\nM=8, F1~F8 상대오차 [%]: " + " ".join(f"{v:.2f}" for v in e.mean(0).tolist()))
    b=loss_bandwidths(A,L,f,webster_modes(A,L,8)[1],wall_shape=1.3)
    print(f"그때의 대역폭 평균 [Hz]: " + " ".join(f"{v:.0f}" for v in b.mean(0).tolist()))
    bi=[names.index(f"bw{k}") for k in range(1,9)]
    print(f"피팅된 대역폭 평균 [Hz]: " + " ".join(f"{v:.0f}" for v in V[live][idx][:,bi].mean(0).tolist()))
