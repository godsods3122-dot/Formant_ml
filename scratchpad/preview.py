"""out/fix 트랙을 그대로 / F5~F8 을 채워서 렌더하고 대역 오차를 비교한다."""
import sys; sys.path.insert(0,"src")
import numpy as np, torch, soundfile as sf, json
from formant_ml.engine.control import ControlTrack, INDEX
from formant_ml.engine.voice import VoiceEngine, EngineConfig
from formant_ml.engine.profile import SpeakerProfile
torch.set_num_threads(2)
prof=SpeakerProfile(**{k:v for k,v in json.load(open("profiles/yang_female.json")).items()
                       if k in SpeakerProfile.__dataclass_fields__})
SP=35000.0/(2.0*prof.tract_length_cm)
def bands(x,fs=48000):
    n=1<<13; w=np.hanning(n); out=[]
    for i in range(0,len(x)-n,n//2):
        out.append(np.abs(np.fft.rfft(x[i:i+n]*w)))
    S=np.mean(np.array(out)**2,0); fr=np.fft.rfftfreq(n,1/fs)
    return fr, 10*np.log10(S+1e-20)
EDGES=[(0.2,1),(1,2),(2,3),(3,4),(4,5.6),(5.6,8),(8,12),(12,16)]
def tab(fr,d): return [d[(fr>=a*1000)&(fr<b*1000)].mean() for a,b in EDGES]
for stem in ("s040","s101"):
    z=np.load(f"out/fix/{stem}_track.npz"); names=[str(x) for x in z["names"]]; V=z["values"]
    tgt,fs=sf.read(f"out/fix/{stem}_target.wav")
    fr,dt=bands(np.asarray(tgt,dtype=np.float64))
    T0=tab(fr,dt)
    eng=VoiceEngine(EngineConfig(), profile=prof)
    rows={}
    for tag in ("f5~f8 = 0 (기존)","f5~f8 = F4 + k·c/2L"):
        Vv=V.copy()
        if "채" in tag or "c/2L" in tag:
            f4=Vv[:,INDEX["f4"]]
            for k in range(5,9):
                f=f4+(k-4)*SP
                Vv[:,INDEX[f"f{k}"]]=f; Vv[:,INDEX[f"bw{k}"]]=40.0+0.05*f
        tr=ControlTrack(Vv, frame_ms=float(z["frame_ms"]))
        with torch.no_grad():
            y=eng.render(tr)
        fr2,dy=bands(y.astype(np.float64))
        # 전체 레벨은 맞춰서 본다
        off=np.mean(T0)-np.mean(tab(fr2,dy))
        rows[tag]=[v+off for v in tab(fr2,dy)]
    print(f"=== {stem}  대역별 오차 [dB] (합성 − 목표, 전체 레벨 정합 후)")
    print("           " + " ".join(f"{a}-{b}k".rjust(8) for a,b in EDGES))
    for tag,r in rows.items():
        e=[x-t for x,t in zip(r,T0)]
        print(f"{tag:20s} " + " ".join(f"{v:+8.1f}" for v in e) +
              f"   |오차|평균 {np.mean(np.abs(e)):.2f}")
