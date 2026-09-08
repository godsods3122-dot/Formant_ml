"""코퍼스 -> 물리 factor 학습 데이터.

    python scripts/parametrize_corpus.py --wavs 'data/voices/*.wav' --out out/corpus \
        --profile profiles/yang_female.json --jobs 3

무엇을 만드는가
---------------
녹음 한 조각마다 **물리 파라미터의 시계열**(제어 트랙)을 만든다. 그것이 이 프로젝트의
학습 데이터다 — 신경망은 파형을 만들지 않고 이 손잡이만 예측하기 때문이다.

    out/<이름>/
      index.jsonl                     구간 하나당 한 줄: 출처·유형·품질 지표
      tracks/<파일>_<시작ms>-<끝ms>.npz  values (T,P) / target / synth / meta

`values` 가 학습 목표다. 열 이름은 `formant_ml.engine.control.PARAM_NAMES` 와 같은
순서이고, 행은 1 ms 프레임이다.

품질을 같이 저장하는 이유
-------------------------
적합은 구간마다 성공률이 다르다. 나쁜 적합을 학습에 그대로 먹이면 신경망이 **적합기의
실패를 배운다.** 그래서 지표를 함께 남기고 `ok` 를 표시해 둔다. 걸러 내는 것은
학습 쪽의 결정이다 — 여기서 버리지는 않는다 (왜 나빴는지 나중에 볼 수 있어야 한다).

치찰음을 어떻게 채점하는가
--------------------------
`fine`(선형 다해상도 STFT)은 난류에서 **34 % 가 상한**이다 — 같은 스펙트럼의 두 독립
실현조차 그렇다. 그 자로는 마찰이 든 구간을 채점할 수 없다. 그래서 `fine_corr`
(실현 잡음을 뺀 값)을 같이 낸다. 자세한 것은 `engine/turbulence.py` 머리말.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import soundfile as sf
import torch

from formant_ml.engine import turbulence as tb
from formant_ml.engine.analyze import analyze
from formant_ml.engine.control import PARAM_NAMES
from formant_ml.engine.denoise import denoise, noise_profile
from formant_ml.engine.fit import CopySynthFitter
from formant_ml.engine.profile import SpeakerProfile
from formant_ml.engine.segment import segments
from formant_ml.engine.voice import EngineConfig, VoiceEngine

# 이 아래면 "적합이 실패했다" 고 본다. 포락은 조음이, 보정 정밀은 파형까지가 맞는가.
OK_ENV = 85.0
OK_FINE_CORR = 70.0


def _name(path: str, t0: float, t1: float) -> str:
    return f"{os.path.splitext(os.path.basename(path))[0]}_{int(t0*1000)}-{int(t1*1000)}"


def fit_one(path: str, t0: float, t1: float, kind: str, prof: SpeakerProfile,
            out_dir: str, budget: tuple[int, int, int], patience: int,
            frame_ms: float = 1.0) -> dict:
    """구간 하나를 적합해 npz 로 남기고 요약 한 줄을 돌려준다."""
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(1)
    y = np.asarray(y, dtype=np.float64)
    # 방 잡음을 지운다. 목표에 잡음이 남으면 엔진이 **그 잡음까지** 만들려고 파라미터를
    # 비튼다 (engine/denoise.py 머리말).
    y = denoise(y, sr, noise_profile(y, sr))
    seg = y[int(t0 * sr):int(t1 * sr)]
    hop = max(1, int(round(frame_ms * sr / 1000.0)))
    track = analyze(seg, sr, prof, hop, t0=t0, full=y)
    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=frame_ms,
                                   speaker="female" if prof.f0_nominal > 165 else "male",
                                   residual=False), prof)
    fit = CopySynthFitter(eng, seg, sr, track)
    rep = fit.fit_staged(global_iters=budget[0], stage_iters=budget[1],
                         phase_iters=budget[2], verbose=False, log_every=10 ** 9,
                         patience=patience)
    out = fit.render()
    tgt = fit.target[0].numpy()
    fid = fit.fidelity()
    res = fit.result_track()
    stem = _name(path, t0, t1)
    npz = os.path.join(out_dir, "tracks", stem + ".npz")
    os.makedirs(os.path.dirname(npz), exist_ok=True)
    np.savez_compressed(
        npz, values=res.values.astype(np.float32), frame_ms=res.frame_ms,
        names=np.array(PARAM_NAMES), target=tgt.astype(np.float32),
        synth=out.astype(np.float32), sample_rate=48000,
        voiced=np.asarray(track.voiced), fricative=np.asarray(track.fricative),
        pulses=np.asarray(track.pulses))
    row = dict(stem=stem, file=os.path.basename(path), t0=t0, t1=t1, kind=kind,
               frames=int(res.values.shape[0]), npz=os.path.relpath(npz, out_dir),
               env=rep.env, fine=rep.fine, db=rep.db,
               fine_corr=fid["fine_corr"], floor=fid["floor"],
               spectrum_match=fid["spectrum_match"], gain_db=fit.gain_db())
    if kind == "fricative":
        c = tb.compare(tgt, out, 48000.0)
        row.update(centroid_err=c["centroid_err"], band_mae=c["band_mae_db"],
                   sibilance_err=c["sibilance_err"], mod_mae=c["mod_mae_pct"],
                   centroid_target=c["target"]["centroid"],
                   centroid_synth=c["synth"]["centroid"])
    row["ok"] = bool(rep.env >= OK_ENV and fid["fine_corr"] >= OK_FINE_CORR)
    return row


def _worker(args):
    (path, t0, t1, kind, prof_path, out_dir, budget, patience, threads) = args
    torch.set_num_threads(threads)
    try:
        prof = SpeakerProfile.load(prof_path)
        t = time.time()
        row = fit_one(path, t0, t1, kind, prof, out_dir, budget, patience)
        row["seconds"] = round(time.time() - t, 1)
        return row
    except Exception as e:                      # 한 구간이 죽어도 코퍼스는 계속 돈다
        return dict(stem=_name(path, t0, t1), file=os.path.basename(path),
                    t0=t0, t1=t1, kind=kind, ok=False,
                    error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc()[-1500:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wavs", default="data/voices/*.wav")
    ap.add_argument("--out", default="out/corpus")
    ap.add_argument("--profile", default="profiles/yang_female.json")
    ap.add_argument("--kinds", default="fricative,vowel")
    ap.add_argument("--limit", type=int, default=0, help="구간 수 상한 (0 = 전부)")
    ap.add_argument("--files", type=int, default=0, help="파일 수 상한 (0 = 전부)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2, help="작업 하나당 torch 스레드")
    ap.add_argument("--budget", default="200,150,800",
                    help="전역,단계,위상 반복 수")
    ap.add_argument("--patience", type=int, default=40,
                    help="이 회차 동안 손실이 안 줄면 그 단계를 끝낸다 (0 = 끔)")
    ap.add_argument("--resume", action="store_true", help="이미 있는 npz 는 건너뛴다")
    a = ap.parse_args()

    budget = tuple(int(x) for x in a.budget.split(","))
    kinds = set(a.kinds.split(","))
    prof = SpeakerProfile.load(a.profile)
    os.makedirs(a.out, exist_ok=True)
    paths = sorted(glob.glob(a.wavs))
    if a.files:
        paths = paths[:a.files]
    if not paths:
        sys.exit(f"wav 을 못 찾았다: {a.wavs}")

    print(f"파일 {len(paths)} 개에서 구간을 고른다...", flush=True)
    jobs = []
    for p in paths:
        y, sr = sf.read(p)
        if y.ndim > 1:
            y = y.mean(1)
        try:
            segs = segments(y, sr, prof, file=os.path.basename(p))
        except Exception as e:
            print(f"  {os.path.basename(p)}: 분할 실패 {e}", flush=True)
            continue
        for s in segs:
            if s.kind not in kinds:
                continue
            if a.resume and os.path.exists(
                    os.path.join(a.out, "tracks", _name(p, s.t0, s.t1) + ".npz")):
                continue
            jobs.append((p, s.t0, s.t1, s.kind, a.profile, a.out, budget,
                         a.patience, a.threads))
    if a.limit:
        jobs = jobs[:a.limit]
    n_fric = sum(1 for j in jobs if j[3] == "fricative")
    print(f"구간 {len(jobs)} 개 (마찰 {n_fric}, 모음 {len(jobs)-n_fric}), "
          f"작업 {a.jobs} 개로 돈다", flush=True)

    idx = os.path.join(a.out, "index.jsonl")
    done = 0
    t_all = time.time()
    with open(idx, "a", encoding="utf-8") as fh:
        if a.jobs > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(a.jobs) as pool:
                for row in pool.imap_unordered(_worker, jobs):
                    done += 1
                    _log(fh, row, done, len(jobs), t_all)
        else:
            for j in jobs:
                row = _worker(j)
                done += 1
                _log(fh, row, done, len(jobs), t_all)
    print(f"\n끝. 색인: {idx}")


def _log(fh, row, done, total, t_all):
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    fh.flush()
    el = time.time() - t_all
    eta = el / max(done, 1) * (total - done)
    if "error" in row:
        print(f"[{done}/{total}] {row['stem']:>34s}  실패 {row['error'][:60]}", flush=True)
    else:
        print(f"[{done}/{total}] {row['stem']:>34s} {row['kind'][:4]:>4s} "
              f"포락 {row['env']:5.1f} 보정정밀 {row['fine_corr']:5.1f} "
              f"(상한 {row['floor']:4.1f}) {'ok' if row['ok'] else '--'} "
              f"{row.get('seconds', 0):5.1f}s  남은 {eta/60:.0f}분", flush=True)


if __name__ == "__main__":
    main()
