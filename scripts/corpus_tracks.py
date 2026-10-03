"""코퍼스 전체의 **분석 궤적**을 모은다 — 움직임 통계(사전 분포)의 재료 (MEASUREMENTS §52.425).

적합 결과가 아니라 분석기가 목표 음성에서 직접 읽은 값만 쓴다. 적합 궤적은 적합기의 지름길로 오염돼 있어 거기서 잰
통계는 순환 논리다.

    python scripts/corpus_tracks.py --out out/corpus_an --jobs 8
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def one(args):
    path, out, prof_path, hop = args
    dst = os.path.join(out, os.path.basename(path).replace(".wav", ".npz"))
    if os.path.exists(dst):
        return dst, "skip"
    import soundfile as sf
    from formant_ml.engine.analyze import analyze
    from formant_ml.engine.control import PARAM_NAMES
    from formant_ml.engine.profile import SpeakerProfile
    y, sr = sf.read(path)
    y = np.asarray(y, float)
    if y.ndim > 1:
        y = y.mean(1)
    prof = SpeakerProfile.load(prof_path)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        tr = analyze(y, sr, prof, hop, t0=0.0, full=y)
    vals = np.stack([np.asarray(tr[n], float) for n in PARAM_NAMES], 1)
    np.savez_compressed(dst, values=vals, names=np.array(PARAM_NAMES), voiced=np.asarray(tr.voiced, bool),
                        fricative=np.asarray(getattr(tr, "fricative", np.zeros(0)), bool),
                        bursts=np.asarray(getattr(tr, "bursts", np.zeros(0)), int), hop=hop, sr=sr)
    return dst, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavs", default="wavs/*.wav")
    ap.add_argument("--out", default="out/corpus_an")
    ap.add_argument("--profile", default="profiles/yang_female.json")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(a.wavs))
    jobs = [(f, a.out, a.profile, 48) for f in files]
    n = 0
    with ProcessPoolExecutor(a.jobs) as ex:
        for dst, st in ex.map(one, jobs):
            n += 1
            if n % 20 == 0 or st != "ok":
                print(f"{n}/{len(files)} {os.path.basename(dst)} {st}", flush=True)
    print("끝", n)


if __name__ == "__main__":
    main()
