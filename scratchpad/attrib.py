"""(1) 세 항이 실제 음성에서 켜지는가  (2) 무너진 창의 책임이 어느 파라미터에 있는가."""
import sys; sys.path.insert(0,"src")
import numpy as np, torch, soundfile as sf, json
import formant_ml.engine.fit as F
from formant_ml.engine.control import ControlTrack, INDEX
from formant_ml.engine.voice import VoiceEngine, EngineConfig
from formant_ml.engine.profile import SpeakerProfile
torch.set_num_threads(1)
prof=SpeakerProfile(**{k:v for k,v in json.load(open("profiles/yang_female.json")).items()
                       if k in SpeakerProfile.__dataclass_fields__})
t,fs=sf.read("out/pole/s040_target.wav")
z=np.load("out/pole/s040_track.npz"); V=z["values"]; fm=float(z["frame_ms"])
T0,T1=0.60,0.76                       # 무너진 창(0.65~0.70)을 품는 구간
i0,i1=int(T0*fs),int(T1*fs)
seg=np.asarray(t[i0:i1],dtype=np.float64)
f0,f1=int(T0*1000/fm),int(T1*1000/fm)
tr=ControlTrack(V[f0:f1].copy(), frame_ms=fm)
eng=VoiceEngine(EngineConfig(), profile=prof)

print("=== (1) 세 항이 실제 음성에서 켜지는가")
base=None
for w in ((0.,0.,0.),(1.,0.,0.),(0.,1.,0.),(0.,0.,1.)):
    F.CONT_W,F.SHARP_W,F.SUBF0_W=w
    f=F.CopySynthFitter(eng, seg, fs, tr); f.sizes=list(F.FFT_SIZES)
    l,_,_,_=f.loss()
    v=float(l.detach())
    if base is None: base=v
    print(f"  CONT={w[0]} SHARP={w[1]} SUBF0={w[2]}  손실 {v:.6f}  기여 {v-base:+.6f}")
print(f"  목표 첨예도 {float(f.tgt_sharp.mean()):.3f},  목표 F0 아래 {float(f.tgt_subf0[f._voiced_long].mean()):+.2f} dB")

print("\n=== (2) 이 구간 손실에 대한 파라미터 책임 (|∇| × step, 합)")
F.CONT_W=F.SHARP_W=F.SUBF0_W=0.0
f=F.CopySynthFitter(eng, seg, fs, tr); f.sizes=list(F.FFT_SIZES)
l,_,_,_=f.loss(); l.backward()
g=(f.w.grad.abs()*f.scale).sum(0)      # 열별 합
order=torch.argsort(g,descending=True)
tot=float(g.sum())
for i in order[:12]:
    print(f"  {f.names[int(i)]:>12s}  {float(g[int(i)]):.4e}  ({100*float(g[int(i)])/tot:5.1f} %)")
