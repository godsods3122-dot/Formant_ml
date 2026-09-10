"""끊김이 있는 구간에서 --corr 을 잰다. 비조화 성분과 끊김 개수를 함께 본다."""
import sys, json; sys.path.insert(0,"src")
import numpy as np, torch, soundfile as sf
import formant_ml.engine.fit as F
from formant_ml.engine.analyze import analyze
from formant_ml.engine.voice import VoiceEngine, EngineConfig
from formant_ml.engine.profile import SpeakerProfile
from formant_ml.engine.waveform import decompose
import parselmouth
torch.set_num_threads(2)
prof=SpeakerProfile(**{k:v for k,v in json.load(open("profiles/yang_female.json")).items()
                       if k in SpeakerProfile.__dataclass_fields__})
y,sr=sf.read("data/voices/yang_00000040.wav"); y=np.asarray(y,np.float64)
T0,T1=0.38,0.60                       # fix 가 0.46 s 에서 상관 −0.07 로 무너지는 구간
seg=y[int(T0*sr):int(T1*sr)].copy()
tr=analyze(seg,sr,prof,48,t0=T0,full=y)
def dips(t,z,fs,win_ms=20.,hop_ms=5.):
    n,h=int(win_ms*fs/1000),int(hop_ms*fs/1000); m=min(len(t),len(z)); r=[]
    for i in range(0,m-n,h):
        a,b=t[i:i+n],z[i:i+n]; d=np.sqrt((a*a).sum()*(b*b).sum())
        r.append((a*b).sum()/d if d>1e-14 else np.nan)
    r=np.array(r); ok=np.isfinite(r)
    return int((r[ok]<0.5).sum()), int((r[ok]<0.0).sum()), float(np.nanmin(r))
def hn(x,fs,f0,hop,lo,hi):
    _,har,res=decompose(x,fs,f0,hop)
    def b(s):
        n=1<<11; w=np.hanning(n); acc=None;c=0
        for i in range(0,max(len(s)-n,1),n//2):
            q=s[i:i+n]
            if len(q)<n: break
            S=np.abs(np.fft.rfft(q*w))**2; acc=S if acc is None else acc+S; c+=1
        fr=np.fft.rfftfreq(n,1/fs); P=acc/max(c,1); m=(fr>=lo)&(fr<hi)
        return 10*np.log10(P[m].sum()+1e-30)
    return b(har)-b(res)
ptd=parselmouth.Sound(seg,sr).to_pitch(time_step=48/sr,pitch_floor=90,pitch_ceiling=700)
nd=len(seg)//48
f0d=np.nan_to_num(np.array([ptd.get_value_at_time((i+0.5)*48/sr) for i in range(nd)]),nan=0.0)
tgt_hn=[hn(seg,sr,f0d,48,*b) for b in ((80,1000),(1000,4000),(4000,8000))]
tgt_d=dips(seg,seg,sr)
print(f"{'':>26s}  {'포락':>6s} {'정밀':>6s} | {'조화÷비조화 [dB]':>20s} | {'상관<0.5':>8s} {'<0':>4s} {'최소':>7s}")
print(f"{'목표':>26s}  {'':>6s} {'':>6s} | " + " ".join(f"{v:6.1f}" for v in tgt_hn)
      + f" | {tgt_d[0]:8d} {tgt_d[1]:4d} {tgt_d[2]:7.2f}")
def run(tag, **flags):
    old={k:getattr(F,k) for k in flags}
    for k,v in flags.items(): setattr(F,k,v)
    try:
        f=F.CopySynthFitter(VoiceEngine(EngineConfig(), profile=prof), seg, sr, tr)
        rep=f.fit_staged(global_iters=60, stage_iters=40, phase_iters=80,
                         verbose=False, log_every=10**9)
        with torch.no_grad(): z=f.synth()[0].numpy().astype(np.float64)
        h=[hn(z,sr,f0d,48,*b) for b in ((80,1000),(1000,4000),(4000,8000))]
        d=dips(seg,z,sr)
        print(f"{tag:>26s}  {rep.env:6.2f} {rep.fine:6.2f} | " + " ".join(f"{v:6.1f}" for v in h)
              + f" | {d[0]:8d} {d[1]:4d} {d[2]:7.2f}")
    finally:
        for k,v in old.items(): setattr(F,k,v)
run("기준 (전부 0)", CORR_W=0.0, HNR_W=0.0)
run("--hnr 2", CORR_W=0.0, HNR_W=2.0)
run("--corr 2 --hnr 2", CORR_W=2.0, HNR_W=2.0)
run("--corr 8 --hnr 2", CORR_W=8.0, HNR_W=2.0)
