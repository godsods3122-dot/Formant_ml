"""몸체 구동 점막 성대의 관 출력이 매개변수에 매끈한지 — 커널만 (§52.524). L(h) = Σ w·y(θ + h d) 를 h 여러 개로."""
import sys; sys.path.insert(0,"src")
import numpy as np, torch
from formant_ml.engine import tube_td as td
from formant_ml.physics import beam_membrane as BM
FS=td.FS_SIM; N=td.N_SECT; T=int(0.25*FS)
t=np.arange(T)/FS; f0=250.0; ph=2*np.pi*f0*t
A0=np.tile(np.linspace(1.5,4.5,N),(T,1))
co=BM.ns_coefs(0.15,0.3,td.VF_NSTRIP,k_scale=3.0)          # 13 계수
VR=np.zeros((T,16)); VR[:,3:16]=co[None,:]
VR[:,0]=0.004; VR[:,1]=0.0
q=np.ones(T)
cK,cB,fB,Lc=VR[:,5],VR[:,6],VR[:,9],VR[:,10]
xb=float(sys.argv[1]) if len(sys.argv)>1 else 0.04
xst=fB/(cB*q); xd=xst-xb*np.cos(ph); kd=100.0*(cB+cK)
VR[:,6]=cB+kd; VR[:,9]=fB+kd*q*xd
base={"A":A0,"L":np.full(T,15.0),"Ag":np.full(T,0.005),"Ps":np.full(T,8*td.CMH2O)*np.minimum(1,t/0.005),"Av":np.zeros(T),"ng":np.array(0.0),"nc":np.array(0.0),"VQ":q,"VR":VR,"VS":BM.vf_static_ns()}
keys=("A","L","Ag","Ps","Av","ng","nc","VQ","VR","VS")
rng=np.random.default_rng(0); w=rng.standard_normal(T)
def L(v):
    with torch.no_grad():
        y=td.tube_torch(*[torch.tensor(v[k]) for k in keys[:7]],seed=3,vf=(torch.tensor(v["VQ"]),torch.tensor(v["VR"]),torch.tensor(v["VS"])),vf_mode=3)
    return float((y.numpy()*w).sum()*1e-6), y.numpy()
L0,y0=L(base)
print("출력 rms", np.sqrt(np.mean(y0[int(0.05*FS):]**2)))
d=np.zeros_like(VR); d[:,0]=1e-3     # 쉼 변위 열에 일정 방향
for h in (1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4):
    vp=dict(base); vp["VR"]=VR+h*d; vm=dict(base); vm["VR"]=VR-h*d
    Lp,yp=L(vp); Lm,ym=L(vm)
    print(f"h {h:g}: 차분 {(Lp-Lm)/(2*h):+.5e}   출력 차 rms {np.sqrt(np.mean((yp-ym)**2)):.3e}")
