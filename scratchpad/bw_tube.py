import sys; sys.path.insert(0,"src")
import numpy as np
from formant_ml.engine import tube_td as td
FS=td.FS_SIM; N=td.N_SECT
if len(sys.argv)>1: td.LOSS_WET=float(sys.argv[1])
T=int(0.4*FS); A=np.full((T,N),3.0); Ps=np.zeros(T); Ps[10:20]=1000.0
out=td.simulate(A,np.full(T,15.0),np.full(T,0.0005),Ps,walls=True)[0]
X=np.abs(np.fft.rfft(out*np.hanning(T),1<<20)); f=np.fft.rfftfreq(1<<20,1/FS); db=20*np.log10(X+1e-30)
from scipy.signal import find_peaks
m=(f>200)&(f<5000); pk,_=find_peaks(db[m],prominence=6)
res=[]
for p in pk[:4]:
    i=np.flatnonzero(m)[p]; lv=db[i]-3
    lo=i
    while lo>0 and db[lo]>lv: lo-=1
    hi=i
    while hi<len(db)-1 and db[hi]>lv: hi+=1
    res.append(f"F {f[i]:.0f} Hz BW {f[hi]-f[lo]:.0f}")
print(f"LOSS_WET {td.LOSS_WET}: "+" · ".join(res))
