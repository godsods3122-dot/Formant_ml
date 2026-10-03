import sys; sys.path.insert(0,"src")
import numpy as np, torch, math
from formant_ml.engine import voice_td as V
V.PHASE_AT_CLOSURE = True
pp = torch.tensor(np.load("out/_tmp/vf/DI_kin_pp.npz")["PP"].reshape(1,-1))[:, :48000]
T = pp.shape[-1] // 48
tdp = V.TDPath(48000, 48)
dt = torch.float32 if sys.argv[1] == "f32" else torch.float64
c = {"voice_gain": torch.zeros(1,T,dtype=dt), "rd_offset": torch.full((1,T),0.2,dtype=dt), "fold_skew": torch.full((1,T),0.6,dtype=dt)}
st = {"amp": torch.full((1,T),0.8,dtype=dt), "ag_dc": torch.full((1,T),0.02,dtype=dt)}
n_sim = T*48*V.OS
def ga(dl):
    with torch.no_grad():
        tdp.log_ag_max.fill_(math.log(0.08)+dl)
        return tdp.glottal_area(c, st, pp, n_sim).double().numpy()[0]
a0=ga(0.0)
for h in (1e-2,1e-3,1e-4,1e-5):
    ap,am=ga(h),ga(-h); print(f"{sys.argv[1]} h {h:g}: r {np.linalg.norm(ap-2*a0+am)/np.linalg.norm(ap-am):.3e}")
