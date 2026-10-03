import sys; sys.path.insert(0,"src")
import numpy as np
from scipy.signal import find_peaks
from formant_ml.engine import tube_td as td
FS=td.FS_SIM; N=td.N_SECT
print("FS_SIM",FS,"N_SECT",N, "C",td.C_SOUND)
def peaks(out, lo=200., hi=9000.):
    X=np.abs(np.fft.rfft(out*np.hanning(len(out)),1<<20)); f=np.fft.rfftfreq(1<<20,1/FS)
    db=20*np.log10(X+1e-30); m=(f>lo)&(f<hi); pk,_=find_peaks(db[m],prominence=6); return f[m][pk]
for L in (14.6, 17.5):
  for walls in (False, True):
    T=int(0.25*FS); A=np.full((T,N),3.0); Ps=np.zeros(T); Ps[10:20]=1000.
    out=td.simulate(A,np.full(T,L),np.full(T,0.02),Ps,walls=walls)[0]
    f=peaks(out); Leff=L+0.85*np.sqrt(3/np.pi)
    want=[(2*k-1)*td.C_SOUND/(4*Leff) for k in range(1,len(f)+1)]
    print(f"L={L} walls={walls}: "+"  ".join(f"{a:.0f}/{b:.0f}({100*(a/b-1):+.1f}%)" for a,b in zip(f,want)))
