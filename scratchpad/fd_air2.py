import sys; sys.path.insert(0,"src")
import numpy as np, torch
from formant_ml.engine import tube_td as td
OLD=dict(C_SOUND=35000.0,RHO=1.14e-3,MU=1.86e-4,GAMMA=1.4,LAMBDA_TH=2.30e3,CP=1.00e7)
which=sys.argv[1]
if which=="all":
    for k,v in OLD.items(): setattr(td,k,v)
elif which!="none":
    for k in which.split(","): setattr(td,k,OLD[k])
td.NU=td.MU/td.RHO
import os
for kv in os.environ.get("TDSET","").split(","):
    if kv: k,v=kv.split("="); setattr(td,k,eval(v))
FS=td.FS_SIM; N=td.N_SECT
rng=np.random.default_rng(2); T=2000; t=np.arange(T)/FS
A0=np.full((T,N),3.0); A0[:,25]=0.08+0.02*np.sin(2*np.pi*60*t); A0[:,24]=0.5
base={"A":A0,"L":np.full(T,15.0),"Ag":np.full(T,0.3)+0.02*np.sin(2*np.pi*40*t),"Ps":np.full(T,8*td.CMH2O)*np.minimum(1,t/0.003),"Av":np.zeros(T),"ng":np.array(0.0),"nc":np.array(float(sys.argv[2]) if len(sys.argv)>2 else 0.02),"Q":np.zeros((T,N))}
keys=("A","L","Ag","Ps","Av","ng","nc","Q"); w=torch.tensor(rng.standard_normal(T))
def loss(v): return (td.tube_torch(*[v[k] for k in keys[:-1]],seed=3,Q=v["Q"])*w).sum()*1e-6
vals={k:torch.tensor(v,requires_grad=True) for k,v in base.items()}; loss(vals).backward()
d=rng.standard_normal(base["A"].shape); ad=float((vals["A"].grad.numpy()*d).sum()); h=1e-7
vp={q:torch.tensor(base[q]) for q in keys}; vm={q:torch.tensor(base[q]) for q in keys}
vp["A"]=torch.tensor(base["A"]+h*d); vm["A"]=torch.tensor(base["A"]-h*d)
fd=(float(loss(vp))-float(loss(vm)))/(2*h); print(f"{which:22s} 비 {fd/ad:.5f}")
