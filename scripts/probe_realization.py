"""잡음 **실현 과적합** 검사 — 같은 제어열을 잡음 씨앗만 바꿔 다시 렌더하고 손실을 잰다.

    python scripts/probe_realization.py out/L25/s040
    python scripts/probe_realization.py out/L32/s040 --glottis asp_am=0.3 harm_jit=0.25 onset=1500 --fric-am 0

적합은 한 씨앗(0)으로 돈다. 손실이 잡음의 **통계**만 본다면 씨앗을 바꿔도 손실이 거의
같아야 한다. 씨앗 0 에서만 낮으면 적합기가 제어열을 흔들어 **그 한 벌의 난수**를 목표의
무작위 요동에 끼워 맞춘 것이다 (MEASUREMENTS §50.6). 적합 때와 같은 성문·마찰 설정을
주어야 씨앗 0 재렌더가 저장된 `_fit.wav` 와 같아진다 — 첫 줄의 상관이 그 확인이다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from formant_ml.engine import fit as F                     # noqa: E402
from formant_ml.engine import glottis as G                 # noqa: E402
from formant_ml.engine import noise as NZ                  # noqa: E402
from formant_ml.engine.analyze import analyze             # noqa: E402
from formant_ml.engine.profile import SpeakerProfile      # noqa: E402
from formant_ml.engine.voice import EngineConfig, VoiceEngine   # noqa: E402

GL = {"asp_am": "ASP_AM_DEPTH", "harm_jit": "HARM_PHASE_JIT", "onset": "HARM_PHASE_ONSET",
      "voiced_soft": "VOICED_SOFT"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--glottis", nargs="*", default=[], metavar="이름=값",
                    help="적합 때의 성문 설정: " + ", ".join(GL))
    ap.add_argument("--fric-am", type=float, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--profile", default="profiles/yang_female.json")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    for item in a.glottis:
        k, v = item.split("=")
        setattr(G, GL[k], float(v))
    if a.fric_am is not None:
        NZ.FRIC_AM_DEPTH = float(a.fric_am)
    seg, sr = sf.read(a.stem + "_target.wav")
    seg = np.asarray(seg, float)
    prof = SpeakerProfile.load(a.profile)
    track = analyze(seg, sr, prof, int(round(sr / 1000.0)), t0=0.0, full=seg)
    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, speaker="female",
                                   residual=False), prof)
    fit = F.CopySynthFitter(eng, seg, sr, track, phase_weight=1.0)
    fit.pulse_weight = 0.0
    fit._collect = True
    z = np.load(a.stem + "_track.npz", allow_pickle=True)
    with torch.no_grad():
        for k, v in json.loads(str(z["engine_params"])).items():
            getattr(eng.tract, k).fill_(v)
    g = 10 ** (float(z["gain_db"]) / 20)
    vals = torch.as_tensor(np.asarray(z["values"], float), dtype=torch.float32).unsqueeze(0)
    n = fit.target.shape[-1]
    ref = np.asarray(sf.read(a.stem + "_fit.wav")[0], float)[:n]

    def render(seed):
        eng.cfg.seed = seed            # eng.noise 에 넣으면 reset() 이 덮어쓴다
        eng.reset()
        with torch.no_grad():
            y = eng(vals, track.events, 0.0)["audio"][0].double().cpu().numpy()[:n] * g
        return np.pad(y, (0, n - len(y)))

    def terms(y):
        yt = torch.as_tensor(y[None, :], dtype=fit.target.dtype)
        fit.synth = lambda want_phase=False: yt
        with torch.no_grad():
            _, sc, env_sc, _ = fit.loss()
        T = {k: float(v) for k, v in fit._terms.items()}
        T["env_db"] = (T["env"] - float(env_sc) - 0.5 * float(sc)) / 4 * 20
        return T

    y0 = render(a.seeds[0])
    m = min(len(y0), len(ref))
    c = np.dot(y0[:m], ref[:m]) / np.sqrt(np.dot(y0[:m], y0[:m]) * np.dot(ref[:m], ref[:m]))
    print(f"[{a.stem}] 씨앗 {a.seeds[0]} 재렌더 ↔ 저장된 _fit.wav 상관 {c:.4f} "
          f"(1 에서 멀면 설정이 적합 때와 다르다)")
    cols = ["env", "env_db", "flux", "corr", "hnr", "phase"]
    print(f"{'':10}" + "".join(f"{k:>9}" for k in cols))
    for s in a.seeds:
        T = terms(y0 if s == a.seeds[0] else render(s))
        print(f"씨앗 {s:<5}" + "".join(f"{T.get(k, np.nan):9.4f}" for k in cols), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
