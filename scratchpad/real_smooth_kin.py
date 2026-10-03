import sys; sys.path.insert(0,"src")
import numpy as np, torch
from formant_ml.engine import tube_td as td, voice_td as V
V.set_larynx(True); V.NOSE_SCALE=(0.9,0.7); V.set_nose("vtl"); td.NOISE_PLACE=True
z=np.load(sys.argv[1]); which=sys.argv[2]; NSC=float(sys.argv[3])
T=z["A"].shape[0]; n=min(T,int(0.6*td.FS_SIM)); sl=lambda a: a[:n] if a.shape and a.shape[0]==T else a
base={k:sl(z[k]) for k in ("A","L","Ag","Ps","Av","Q")}; teeth=sl(z["teeth"]) if z["teeth"].size>1 else None
rng=np.random.default_rng(0); w=rng.standard_normal(n)
def Y(b):
    with torch.no_grad():
        return td.tube_torch(*[torch.tensor(b[k]) for k in ("A","L","Ag","Ps","Av")],torch.tensor(float(z["ng"])*NSC),torch.tensor(float(z["nc"])*NSC),
                        seed=int(z["seed"]),Q=torch.tensor(b["Q"]),fs=td.FS_SIM,teeth=None if teeth is None else torch.tensor(teeth)).numpy()
y0=Y(base)
d={"Ag":base["Ag"]*0.01,"A":rng.standard_normal(base["A"].shape)*0.01*base["A"]}[which]
for h in (1e-1,1e-2,1e-3,1e-4):
    bp=dict(base); bp[which]=base[which]+h*d; bm=dict(base); bm[which]=base[which]-h*d
    yp,ym=Y(bp),Y(bm); print(f"{which} 잡음x{NSC:g} h {h:g}: r {np.linalg.norm(yp-2*y0+ym)/max(np.linalg.norm(yp-ym),1e-30):.3e}  차분 {((yp-ym)*w).sum()/(2*h):+.5e}",flush=True)
