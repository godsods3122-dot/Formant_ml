import sys, json, numpy as np, torch
sys.path.insert(0,"src")
from formant_ml.engine import voice_td as V
V.ARTIC="w02"; V.W02_ANAT=True
sur=V._w02a()
run=sys.argv[1]
z=np.load(f"out/VF/{run}_track.npz", allow_pickle=True)
ac=json.loads(str(z["acoustic_constants"]))["speaker"]
tdp=V.TDPath(48000,48)
for i in range(13): getattr(tdp,f"w02a_u{i}").data.fill_(ac[f"td_w02a_{i}"])
a=tdp.w02_anatomy(sur).detach().numpy()
alo,ahi,ax0=sur.alo.numpy(),sur.ahi.numpy(),sur.ax0.numpy()
nm=[n for n in dir(sur) if "anat" in n.lower()]
print("attrs:", nm)
for i in range(13):
    print(f"{i:2d} fit={a[i]:8.3f}  W02={ax0[i]:8.3f}  lo={alo[i]:8.3f} hi={ahi[i]:8.3f}")
names=list(z["names"]); Vv=z["values"]
c={n: torch.tensor(Vv[None,:,names.index(n)]).double() for n in names}
c["p_sub"]=torch.zeros(1,Vv.shape[0],dtype=torch.float64)
with torch.no_grad():
    A,L=tdp._tube_frames(c)
    L=L.numpy()[0]
    print("tract length cm: mean %.2f min %.2f max %.2f"%(L.mean(),L.min(),L.max()))
    p=torch.stack([tdp._w02_smooth(c,n) for n in sur.names],-1)
    A2,L2,_=sur(p, torch.tensor(ax0).double().expand(1,Vv.shape[0],-1).to(p.dtype))
    print("W02 anatomy, same artic: tract length mean %.2f"%L2.numpy().mean())
