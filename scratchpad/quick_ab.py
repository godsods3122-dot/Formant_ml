"""짧은 구간으로 새 항이 최적화 중에 실제로 작동하는지 본다."""
import sys, json; sys.path.insert(0,"src")
import numpy as np, torch, soundfile as sf
import formant_ml.engine.fit as F
from formant_ml.engine.analyze import analyze
from formant_ml.engine.voice import VoiceEngine, EngineConfig
from formant_ml.engine.profile import SpeakerProfile
torch.set_num_threads(2)
prof=SpeakerProfile(**{k:v for k,v in json.load(open("profiles/yang_female.json")).items()
                       if k in SpeakerProfile.__dataclass_fields__})
y,sr=sf.read("data/voices/yang_00000040.wav"); y=np.asarray(y,np.float64)
T0,T1=0.85,1.15                      # 유성 구간
seg=y[int(T0*sr):int(T1*sr)].copy()
tr=analyze(seg,sr,prof,48,t0=T0,full=y)
def run(tag, **flags):
    old={k:getattr(F,k) for k in flags}
    for k,v in flags.items(): setattr(F,k,v)
    try:
        eng=VoiceEngine(EngineConfig(), profile=prof)
        f=F.CopySynthFitter(eng, seg, sr, tr)
        rep=f.fit_staged(global_iters=60, stage_iters=40, phase_iters=80,
                         verbose=False, log_every=10**9)
        with torch.no_grad():
            yy=f.synth()
        F.CORR_W=F.HNR_W=1.0
        cl=float(f.corr_loss(yy)); hl=float(f.hnr_loss(yy))
        per=f._periodicity(yy)
        tp=float(f.tgt_per[0][f._per_ok].mean()); sp_=float(per[0][f._per_ok].mean())
        tp1=float(f.tgt_per[1][f._per_ok].mean()); sp1=float(per[1][f._per_ok].mean())
        print(f"{tag:>22s}  포락 {rep.env:6.2f}  정밀 {rep.fine:6.2f}  "
              f"corr {cl:.4f}  hnr {hl:.4f}  주기성 {sp_:.4f}/{sp1:.4f} (목표 {tp:.4f}/{tp1:.4f})")
        return rep.env, cl, hl
    finally:
        for k,v in old.items(): setattr(F,k,v)
run("기준 (전부 0)", CORR_W=0.0, HNR_W=0.0)
run("--corr 2", CORR_W=2.0, HNR_W=0.0)
run("--hnr 2", CORR_W=0.0, HNR_W=2.0)
run("--corr 2 --hnr 2", CORR_W=2.0, HNR_W=2.0)
run("--corr 5 --hnr 5", CORR_W=5.0, HNR_W=5.0)
