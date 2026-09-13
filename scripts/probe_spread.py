"""궤적을 '펴면' 무엇을 잃고 무엇을 얻나 — 적합 궤적을 물리 시간상수로 평활해 재렌더한다
(MEASUREMENTS §50.11).

    python scripts/probe_spread.py out/L25/s040

손실이 거의 안 오르는데 토막·끊김이 준다면 그 잔 움직임은 소리에 필요 없었다. 적합 뒤에 펴므로
다른 파라미터가 흔들림에 적응해 있던 몫까지 잃는다 — 적합 중 제약의 상한 추정으로 읽는다.
"""
import sys, json
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, soundfile as sf, torch
from scipy.ndimage import gaussian_filter1d
from formant_ml.engine import fit as F
from formant_ml.engine.analyze import analyze
from formant_ml.engine.profile import SpeakerProfile
from formant_ml.engine.segment import fricative_mask
from formant_ml.engine.voice import VoiceEngine, EngineConfig
from census import _chop, _line_persist, _live_5ms
torch.set_num_threads(2)
stem = sys.argv[1]
seg, sr = sf.read(stem + "_target.wav"); seg = np.asarray(seg, float)
prof = SpeakerProfile.load("profiles/yang_female.json")
track = analyze(seg, sr, prof, 48, t0=0.0, full=seg)
eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, speaker="female", residual=False), prof)
fit = F.CopySynthFitter(eng, seg, sr, track, phase_weight=1.0)
fit.pulse_weight = 0.0; fit._collect = True
z = np.load(stem + "_track.npz", allow_pickle=True)
with torch.no_grad():
    for k, v in json.loads(str(z["engine_params"])).items():
        getattr(eng.tract, k).fill_(v)
g = 10 ** (float(z["gain_db"]) / 20)
V0 = np.asarray(z["values"], float); names = [str(n) for n in z["names"]]
n = fit.target.shape[-1]
fm = fricative_mask(seg, sr, prof, 48); live = _live_5ms(seg, sr)

def smooth(V, taus, mult=1.0):
    V = V.copy()
    for k, tau in taus.items():
        if k in names:
            V[:, names.index(k)] = gaussian_filter1d(V[:, names.index(k)], tau * mult, mode="nearest")
    return V

def render(V):
    eng.reset()
    with torch.no_grad():
        y = eng(torch.as_tensor(V, dtype=torch.float32).unsqueeze(0), track.events, 0.0)["audio"][0].double().numpy()[:n] * g
    return np.pad(y, (0, n - len(y)))

def evaluate(y):
    yt = torch.as_tensor(y[None, :], dtype=fit.target.dtype)
    fit.synth = lambda want_phase=False: yt
    with torch.no_grad():
        _, sc, env_sc, _ = fit.loss()
    T = {k: float(v) for k, v in fit._terms.items()}
    T["env_db"] = (T["env"] - float(env_sc) - 0.5 * float(sc)) / 4 * 20
    T["포락%"] = 100 * (1 - float(env_sc)); T["정밀%"] = 100 * (1 - float(sc))
    w, h = int(0.020 * sr), int(0.005 * sr)
    cs, lv = [], []
    for i in range(0, n - w, h):
        a, b = seg[i:i + w], y[i:i + w]
        d = np.sqrt((a * a).sum() * (b * b).sum())
        cs.append((a * b).sum() / d if d > 1e-14 else np.nan); lv.append(20 * np.log10(a.std() + 1e-20))
    cs, lv = np.array(cs), np.array(lv)
    T["끊김"] = int((cs[lv > lv.max() - 45] < 0).sum())
    T["치찰토막"] = _chop(y, sr, fm, 48)
    T["줄10-16"] = _line_persist(y, sr, live)[1]
    return T

LAR = {k: F.PARAM_TAU_PHYS[k] for k in ("adduction", "p_sub")}
rows = [("적합 그대로", V0), ("후두만 펴기 (내전 15·폐압 30 ms)", smooth(V0, LAR)),
        ("전부 물리 τ 로 펴기", smooth(V0, F.PARAM_TAU_PHYS)), ("전부 2τ 로 펴기", smooth(V0, F.PARAM_TAU_PHYS, 2.0))]
cols = ["포락%", "정밀%", "env_db", "corr", "phase", "끊김", "치찰토막", "줄10-16"]
print(f"[{stem}]  목표 치찰토막 {_chop(seg, sr, fm, 48):.2f} dB, 줄10-16 {_line_persist(seg, sr, live)[1]:.3f}")
print(f"{'':34}" + "".join(f"{c:>9}" for c in cols))
for lab, V in rows:
    T = evaluate(render(V))
    print(f"{lab:<34}" + "".join((f"{T[c]:9d}" if isinstance(T[c], int) else f"{T[c]:9.3f}") for c in cols), flush=True)
