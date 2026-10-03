"""실제 판의 관 입력(TD_DUMP_IN)으로 커널의 매끈함 (§52.524). 사용: real_smooth.py 덤프 [열]"""
import sys; sys.path.insert(0,"src")
import numpy as np, torch
from formant_ml.engine import tube_td as td, voice_td as V
V.set_larynx(True); V.NOSE_SCALE=(0.9,0.7); V.set_nose("vtl"); td.NOISE_PLACE=True
z=np.load(sys.argv[1]); which=sys.argv[2] if len(sys.argv)>2 else "VR0"
NSC=float(sys.argv[3]) if len(sys.argv)>3 else 1.0
T=z["A"].shape[0]; n=min(T,int(0.6*td.FS_SIM))      # 앞 0.6 s
sl=lambda a: a[:n] if a.shape and a.shape[0]==T else a
base={k:sl(z[k]) for k in ("A","L","Ag","Ps","Av","Q","VQ","VR")}
VS=z["VS"]; teeth=sl(z["teeth"]) if z["teeth"].size>1 else None
rng=np.random.default_rng(0); w=rng.standard_normal(n)
def L(b):
    with torch.no_grad():
        y=td.tube_torch(*[torch.tensor(b[k]) for k in ("A","L","Ag","Ps","Av")],torch.tensor(float(z["ng"])*NSC),torch.tensor(float(z["nc"])*NSC),
                        seed=int(z["seed"]),Q=torch.tensor(b["Q"]),vf=(torch.tensor(b["VQ"]),torch.tensor(b["VR"]),torch.tensor(VS)),vf_mode=3,
                        fs=td.FS_SIM,teeth=None if teeth is None else torch.tensor(teeth))
    return float((y.numpy()*w).sum()*1e-6)
if which=="VR0":
    d=np.zeros_like(base["VR"]); d[:,0]=1e-3
    key="VR"
elif which=="A":
    d=rng.standard_normal(base["A"].shape)*0.01; key="A"
elif which=="VQ":
    d=np.full_like(base["VQ"],0.01); key="VQ"
for h in (1.0,0.3,0.1,0.03,0.01,0.003,0.001):
    bp=dict(base); bp[key]=base[key]+h*d; bm=dict(base); bm[key]=base[key]-h*d
    print(f"{which} h {h:g}: 차분 {(L(bp)-L(bm))/(2*h):+.6e}", flush=True)
