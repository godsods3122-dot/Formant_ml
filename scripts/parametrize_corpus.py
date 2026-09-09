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
`fine`(선형 다해상도 STFT)은 독립 잡음 실현에서 약 34 % 가 기대값이다 (엄밀한 상한
아님). `fine_corr` 는 실현 분산을 차감한 휴리스틱 추정이며 미분해 값은 하한도
신뢰구간도 아니다. `ok` 는 학습용 자동 필터일 뿐 지각적 동등성 판정이 아니다.
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
from formant_ml.engine import segment
from formant_ml.engine.analyze import analyze, glottal_pulses
from formant_ml.engine.control import PARAM_NAMES
from formant_ml.engine.denoise import denoise, noise_profile
from formant_ml.engine.fit import CopySynthFitter
from formant_ml.engine.profile import SpeakerProfile
from formant_ml.engine.segment import segments
from formant_ml.engine.voice import EngineConfig, VoiceEngine

# 이 아래면 "적합이 실패했다" 고 본다. 포락은 조음이, 보정 정밀은 파형까지가 맞는가.
OK_ENV = 85.0
OK_FINE_CORR = 70.0
# 잡음량이 이 범위를 벗어나면 실패로 본다 (합성/목표 실현 분산 비, 1 이 맞음).
# ±6 dB 진폭 오차가 분산비 0.25 / 4.0 에 해당하므로 그 절반쯤에서 자른다.
OK_NOISE_LO, OK_NOISE_HI = 0.5, 2.0


def _name(path: str, t0: float, t1: float) -> str:
    return f"{os.path.splitext(os.path.basename(path))[0]}_{int(t0*1000)}-{int(t1*1000)}"


def load_clean(path: str) -> tuple[np.ndarray, int]:
    """파일을 읽고 방 잡음을 지운다. **파일당 한 번만** 한다.

    목표에 잡음이 남으면 엔진이 그 잡음까지 만들려고 파라미터를 비튼다
    (engine/denoise.py 머리말). 그런데 위너 차감은 파일 전체를 훑으므로, 한 파일의
    구간마다 다시 걸면 20 초짜리 파일에서 그 비용이 구간 수만큼 곱해진다.
    """
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(1)
    y = np.asarray(y, dtype=np.float64)
    return denoise(y, sr, noise_profile(y, sr)), sr


def fit_one(path: str, t0: float, t1: float, kind: str, prof: SpeakerProfile,
            out_dir: str, budget: tuple[int, int, int], patience: int,
            frame_ms: float = 1.0, clean: tuple[np.ndarray, int] | None = None,
            pulses: np.ndarray | None = None) -> dict:
    """구간 하나를 적합해 npz 로 남기고 요약 한 줄을 돌려준다.

    `clean` 과 `pulses` 는 같은 파일의 다른 구간과 **나눠 쓰라고** 있는 인자다. 둘 다
    파일 전체를 훑는 계산이라 구간마다 다시 하면 그만큼 곱절로 든다.
    """
    y, sr = clean if clean is not None else load_clean(path)
    seg = y[int(t0 * sr):int(t1 * sr)]
    # **말소리가 아닌 입력은 여기서 막는다.** 코퍼스에 방 잡음·숨소리 파일이 섞여
    # 있고, 그런 것도 적합은 잘 된다 — 크기 스펙트럼을 맞추는 일이니 당연하다.
    # 문제는 나오는 파라미터다: 실측에서 적합기가 방 잡음을 `p_sub` 15.8 cmH₂O
    # (외치는 값)에 `a_c` 1.19(내내 협착)로 흉내 냈다. 그대로 학습에 들어가면
    # 신경망이 그걸 목소리로 배운다 (docs/MEASUREMENTS.md §23.9).
    # **적합 성적으로는 못 거른다** — 그 파일의 포락 일치가 95.0 % 로 제일 높았다.
    sl = segment.speech_likeness(seg, float(sr))
    if sl["periodic"] < segment.SPEECH_PERIODIC_MIN:
        return dict(stem=_name(path, t0, t1), file=os.path.basename(path),
                    t0=t0, t1=t1, kind=kind, ok=False,
                    skipped="not_speech", **sl)
    hop = max(1, int(round(frame_ms * sr / 1000.0)))
    track = analyze(seg, sr, prof, hop, t0=t0, full=y, pulses=pulses)
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
               env=rep.env, fine=fid["fine"], db=rep.db,
               fine_corr=fid["fine_corr"], floor=fid["floor"],
               trust=fid["trust"], resolved=fid["resolved"],
               noise_ratio=fid["noise_ratio"],
               spectrum_match=fid["spectrum_match"], gain_db=fit.gain_db(),
               correction_status=fid.get("correction_status",
                                         "resolved" if fid["resolved"] else "unresolved"),
               score_note="clipped heuristic estimate, not a bound; trust is not confidence; "
                          "ok is an automatic filter, not perceptual equivalence")
    if kind == "fricative":
        c = tb.compare(tgt, out, 48000.0)
        row.update(centroid_err=c["centroid_err"], band_mae=c["band_mae_db"],
                   sibilance_err=c["sibilance_err"], mod_mae=c["mod_mae_pct"],
                   centroid_target=c["target"]["centroid"],
                   centroid_synth=c["synth"]["centroid"])
    # **`fine_corr` 만으로 판정하면 안 된다.** 분해가 안 된 구간(치찰음이 대개 그렇다)
    # 에서는 그 값이 미분해 추정치라 잡음 투성이 합성도 높게 나온다. 잡음량이 맞는지를
    # `noise_ratio` 로 같이 본다.
    nr = fid["noise_ratio"]
    row["ok"] = bool(rep.env >= OK_ENV
                     and fid["fine_corr"] >= OK_FINE_CORR
                     and (not np.isfinite(nr) or OK_NOISE_LO <= nr <= OK_NOISE_HI))
    return row


def _worker(args, on_row=None):
    """한 **파일**의 구간 전부를 처리한다 — 잡음 제거와 성문 펄스를 나눠 쓰기 위해.

    `on_row` 는 단일 프로세스로 돌 때만 쓴다. 파일 하나가 몇십 분씩 걸리는데 그 동안
    아무것도 안 찍히면 멈춘 것과 구별이 안 된다 (다중 프로세스에서는 피클이 안 되므로
    None 이고, 진행은 파일 단위로 보인다).
    """
    (path, segs, prof_path, out_dir, budget, patience, threads) = args
    torch.set_num_threads(threads)
    rows = []
    try:
        prof = SpeakerProfile.load(prof_path)
        clean = load_clean(path)
        pulses = glottal_pulses(clean[0], clean[1], prof)
    except Exception as e:
        return [dict(stem=_name(path, t0, t1), file=os.path.basename(path),
                     t0=t0, t1=t1, kind=kind, ok=False,
                     error=f"파일 준비 실패 {type(e).__name__}: {e}")
                for t0, t1, kind in segs]
    for t0, t1, kind in segs:
        t = time.time()
        try:
            row = fit_one(path, t0, t1, kind, prof, out_dir, budget, patience,
                          clean=clean, pulses=pulses)
            row["seconds"] = round(time.time() - t, 1)
        except Exception as e:                  # 한 구간이 죽어도 코퍼스는 계속 돈다
            row = dict(stem=_name(path, t0, t1), file=os.path.basename(path),
                       t0=t0, t1=t1, kind=kind, ok=False,
                       error=f"{type(e).__name__}: {e}",
                       traceback=traceback.format_exc()[-1500:])
        rows.append(row)
        if on_row is not None:
            on_row(row)
    return rows


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
    # **작업 단위는 파일이다.** 잡음 제거와 성문 펄스 추출은 파일 전체를 훑으므로,
    # 구간을 흩어 놓으면 그 비용이 구간 수만큼 곱해진다.
    jobs, n_seg, n_fric = [], 0, 0
    for p in paths:
        y, sr = sf.read(p)
        if y.ndim > 1:
            y = y.mean(1)
        try:
            segs = segments(y, sr, prof, file=os.path.basename(p))
        except Exception as e:
            print(f"  {os.path.basename(p)}: 분할 실패 {e}", flush=True)
            continue
        keep = []
        for s in segs:
            if s.kind not in kinds:
                continue
            if a.resume and os.path.exists(
                    os.path.join(a.out, "tracks", _name(p, s.t0, s.t1) + ".npz")):
                continue
            if a.limit and n_seg >= a.limit:
                break
            keep.append((s.t0, s.t1, s.kind))
            n_seg += 1
            n_fric += s.kind == "fricative"
        if keep:
            jobs.append((p, keep, a.profile, a.out, budget, a.patience, a.threads))
        if a.limit and n_seg >= a.limit:
            break
    print(f"구간 {n_seg} 개 (마찰 {n_fric}, 모음 {n_seg - n_fric}) / 파일 {len(jobs)} 개, "
          f"작업 {a.jobs} 개로 돈다", flush=True)
    if not jobs:
        print("할 일이 없다 (--resume 로 전부 건너뛴 것일 수 있다).")
        return

    idx = os.path.join(a.out, "index.jsonl")
    done = 0
    t_all = time.time()
    with open(idx, "a", encoding="utf-8") as fh:
        if a.jobs > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(a.jobs) as pool:
                for rows in pool.imap_unordered(_worker, jobs):
                    for row in rows:
                        done += 1
                        _log(fh, row, done, n_seg, t_all)
        else:
            state = {"n": done}

            def emit(row):
                state["n"] += 1
                _log(fh, row, state["n"], n_seg, t_all)

            for j in jobs:
                _worker(j, on_row=emit)
            done = state["n"]
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
              f"포락 {row['env']:5.1f} 보정정밀 "
              f"{row['fine_corr']:5.1f} ({'resolved' if row['resolved'] else 'unresolved estimate'}) "
              f"잡음비 {row['noise_ratio']:4.2f} "
              f"{'ok' if row['ok'] else '--'} "
              f"{row.get('seconds', 0):5.1f}s  남은 {eta/60:.0f}분", flush=True)


if __name__ == "__main__":
    main()
