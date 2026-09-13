"""'고역의 평평한 줄' 의 인과 검사 — 같은 제어열에서 고차 극 대역폭 배율만 바꿔 재렌더한다.

    python scripts/probe_ladder.py out/L25/s101

척도는 `census.py` 의 줄 지속과 같다: 고역 미세구조(dB − 주파수 1 kHz 이동중앙값)의
50 ms 지연 상관. 안개면 0 근처, 수평 줄이 서 있으면 높다 (MEASUREMENTS §50.5).
가둠(`tract.HF_BW_LIM`)을 풀어 적합값 그대로도 재현한다.
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
sys.path.insert(0, os.path.dirname(__file__))

from census import _line_persist, _live_5ms                 # noqa: E402
from formant_ml.engine import tract as TR                   # noqa: E402
from formant_ml.engine.analyze import analyze              # noqa: E402
from formant_ml.engine.profile import SpeakerProfile       # noqa: E402
from formant_ml.engine.voice import EngineConfig, VoiceEngine   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--extra", type=float, nargs="*", default=[0.0, 0.5, 0.9, -0.7],
                    help="시험할 log_extra_bw 값 (적합값은 항상 먼저 잰다)")
    ap.add_argument("--profile", default="profiles/yang_female.json")
    a = ap.parse_args()
    torch.set_num_threads(2)
    TR.HF_BW_LIM = 50.0                                     # 가둠을 푼다
    TR._extra_mult = lambda x: torch.exp(x)                 # 좁히기 금지도 푼다
    seg, sr = sf.read(a.stem + "_target.wav")
    seg = np.asarray(seg, float)
    prof = SpeakerProfile.load(a.profile)
    track = analyze(seg, sr, prof, int(round(sr / 1000.0)), t0=0.0, full=seg)
    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, speaker="female",
                                   residual=False), prof)
    z = np.load(a.stem + "_track.npz", allow_pickle=True)
    ep = json.loads(str(z["engine_params"]))
    vals = torch.as_tensor(np.asarray(z["values"], float), dtype=torch.float32).unsqueeze(0)
    live = _live_5ms(seg, sr)
    t6, t10 = _line_persist(seg, sr, live)
    print(f"[{a.stem}] 적합값 {ep}")
    print(f"  목표                     줄 6-10k {t6:.3f}  10-16k {t10:.3f}")
    for xb in [ep["log_extra_bw"]] + list(a.extra):
        with torch.no_grad():
            eng.tract.log_extra_bw.fill_(xb)
            eng.tract.log_front_bw.fill_(ep["log_front_bw"])
        eng.reset()
        with torch.no_grad():
            y = eng(vals, track.events, 0.0)["audio"][0].double().numpy()[:len(seg)]
        p6, p10 = _line_persist(y, sr, live)
        print(f"  고차 극 ×{np.exp(xb):5.2f} ({xb:+.2f})  줄 6-10k {p6:.3f}  10-16k {p10:.3f}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
