import sys; sys.path.insert(0,"src")
import numpy as np, torch
from formant_ml.engine import tube_td as td
FS=td.FS_SIM; N=td.N_SECT
rng=np.random.default_rng(2); T=2000; t=np.arange(T)/FS
A0=np.full((T,N),3.0); A0[:,25]=0.08+0.02*np.sin(2*np.pi*60*t); A0[:,24]=0.5
base={"A":A0,"L":np.full(T,15.0),"Ag":np.full(T,0.3)+0.02*np.sin(2*np.pi*40*t),"Ps":np.full(T,8*td.CMH2O)*np.minimum(1,t/0.003),"Av":np.zeros(T),"ng":np.array(0.0),"nc":np.array(0.02),"Q":np.zeros((T,N))}
keys=("A","L","Ag","Ps","Av","ng","nc","Q"); w=torch.tensor(rng.standard_normal(T))
def loss(v): return (td.tube_torch(*[v[k] for k in keys[:-1]],seed=3,Q=v["Q"])*w).sum()*1e-6
vals={k:torch.tensor(v,requires_grad=True) for k,v in base.items()}; loss(vals).backward()
d=rng.standard_normal(base["A"].shape); ad=float((vals["A"].grad.numpy()*d).sum())
for h in (1e-5,1e-6,1e-7,1e-8):
    vp={q:torch.tensor(base[q]) for q in keys}; vm={q:torch.tensor(base[q]) for q in keys}
    vp["A"]=torch.tensor(base["A"]+h*d); vm["A"]=torch.tensor(base["A"]-h*d)
    fd=(float(loss(vp))-float(loss(vm)))/(2*h); print(f"h {h:g}: 차분 {fd:.5f}  자동 {ad:.5f}  비 {fd/ad:.5f}")
