"""복사합성 적합기의 회귀 벤치 — **코퍼스**의 고정 구간으로.

왜 새로 만드는가
----------------
`bench_copyfit.py` 는 `data/ref/female_yang_ilin-ilsil.wav` 의 네 구간을 쓴다. 그 파일은
레포에 없다 (`.gitignore` 의 `*.wav`). 그리고 직전 세션이 남긴 최대 약점이 바로
**"표본이 파일 하나다"** 였다 (docs/HANDOFF.md §5). 여기 구간은 orphan 브랜치의
코퍼스 205 개(45 분, 같은 화자)에서 자동 선별로 고른 것이다.

무엇을 재는가
-------------
기존 벤치의 네 지표(위상·조화 SNR·궤적 속도·비조화)에 **난류를 정직하게 재는 자**를
더했다:

* `보정정밀` — 실현 분산을 차감한 휴리스틱 추정. 미분해 값은 하한이 아니다.
  100 으로 클리핑되어도 완벽함이나 지각적 동등성을 뜻하지 않는다.
* `floor` — 실현 잡음 기여량의 추정; 유한 표본 점수의 엄밀한 상한이 아니다.
* `무게중심` / `대역MAE` — 치찰음의 조음 위치가 맞는가 (Jongman et al. 2000).
* raw target masks separate frication core, voiced overlap and observed vowel onset.
  Each contiguous run is evaluated separately; no splicing across mask gaps.
* --denoise raw|weak|current|all, --seeds 0,991,2027, --budget 2,1,0,
  --no-probe, --json out/diagnostics.json, --audio-dir out/listening.
  Seeds re-render one fixed fit; distributions are descriptive, not confidence intervals.

구간은 어떻게 골랐나
--------------------
`engine/segment.py` 로 전 코퍼스를 훑고, 마찰 중 **치찰도 > 8 dB 이고 무게중심
> 6 kHz** 인 것만 남겼다. 이 거르기 전에는 무게중심 중앙값이 5175 Hz 로 프로파일
실측(8100 Hz)과 3 kHz 어긋났다 — 호흡과 기식이 섞여 있었기 때문이다. 거른 뒤에는
7997 Hz 로 실측과 맞는다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import soundfile as sf
import torch

from formant_ml.engine import turbulence as tb
from formant_ml.engine.analyze import analyze
from formant_ml.engine.denoise import denoise, noise_profile
from formant_ml.engine.fit import CopySynthFitter
from formant_ml.engine.profile import SpeakerProfile
from formant_ml.engine.voice import EngineConfig, VoiceEngine
from formant_ml.engine.waveform import decompose, phase_error, seg_snr

FS = 48000.0
HOP = 48
GATE_DB = -25.0

# (이름, 파일, 시작 s, 끝 s, 유형)
SEGMENTS = [
    ("치찰 ㅅ (34)",   "data/voices/yang_00000034.wav", 0.510, 0.665, "sib"),
    ("치찰 ㅆ (00)",   "data/voices/yang_00000000.wav", 5.090, 5.235, "sib"),
    ("치찰+모음 (11)", "data/voices/yang_00000011.wav", 14.495, 14.725, "sib"),
    ("모음 (00)",      "data/voices/yang_00000000.wav", 16.500, 16.700, "vowel"),
]


def nonharmonic(sig, har, voi, nf):
    """Legacy voiced-only residual ratio; NOT a measure of unvoiced frication."""
    v = []
    for i in range(nf):
        a, b = i * HOP, (i + 1) * HOP
        if b > len(sig) or not voi[i]:
            continue
        eh = (har[a:b] ** 2).sum()
        er = ((sig[a:b] - har[a:b]) ** 2).sum()
        if eh + er > 0:
            v.append(er / (eh + er))
    return 100 * float(np.median(v)) if v else float("nan")


def prepare_audio(y, sr, mode):
    """Explicit preprocessing condition; weak uses over=.5, floor_db=-6."""
    if mode == "raw":
        return np.asarray(y, dtype=float).copy()
    if mode not in ("weak", "current"):
        raise ValueError(f"unknown denoise mode: {mode}")
    if len(y) <= 2048:
        raise ValueError("denoising needs a full recording longer than 2048 samples")
    kwargs = {"over": 0.5, "floor_db": -6.0} if mode == "weak" else {}
    return denoise(y, sr, noise_profile(y, sr), **kwargs)


def _resample(y, sr):
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(sr), int(FS))
    return resample_poly(y, int(FS) // g, int(sr) // g) if sr != FS else y.copy()


def bench(path, t0, t1, prof, verbose=False, iters=(200, 150, 800),
          probe=True, patience=0, denoise_mode="current", seeds=(0, 991, 2027),
          coupling=None, audio_dir=None):
    """Fit once per preprocessing/coupling condition; only re-render across seeds."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(1)
    if not 0 <= t0 < t1 <= len(y) / sr:
        raise ValueError("segment must lie inside the recording")
    raw = y.copy()
    raw_seg = raw[int(t0 * sr):int(t1 * sr)]
    raw_track = analyze(raw_seg, sr, prof, max(1, round(sr / 1000)), t0=t0, full=raw)
    raw_target = _resample(raw_seg, sr)
    masks = tb.target_regions(raw_target, FS, raw_track.voiced, raw_track.frame_ms)
    y = prepare_audio(y, sr, denoise_mode)
    seg = y[int(t0 * sr):int(t1 * sr)]
    tr = analyze(seg, sr, prof, max(1, round(sr / 1000)), t0=t0, full=y)
    cfg = EngineConfig(sample_rate=48000, frame_ms=1.0,
                       speaker="female", residual=False,
                       noise_modulation=coupling or "legacy")
    eng = VoiceEngine(cfg, prof)
    f = CopySynthFitter(eng, seg, sr, tr)
    # lr 탐침은 비용의 절반쯤이다 (전역 4 후보 + 위상 3 후보). 구간마다 맞는 값이
    # 다르므로 회귀 벤치에서는 켜 두지만, 반복 실행할 때는 끄고 고정값을 쓴다.
    kw = {} if probe else {"lr_global": 0.05, "lr_phase": 0.12}
    rep = f.fit_staged(global_iters=iters[0], stage_iters=iters[1],
                       phase_iters=iters[2], verbose=verbose, log_every=10 ** 9,
                       patience=patience, **kw)
    renders = [f.render(seed=seed) for seed in seeds]
    out = renders[0]
    tgt = f.target[0].numpy()
    n = min(len(tgt), len(out))
    tgt, out = tgt[:n], out[:n]
    f0 = np.asarray(tr["f0_target"])
    voi = np.asarray(tr.voiced).astype(bool)
    at, ht, _ = decompose(tgt, FS, f0, HOP)
    as_, hs, _ = decompose(out, FS, f0, HOP)
    nf = min(len(at), len(as_), len(voi))
    rms = np.array([np.sqrt((tgt[i * HOP:(i + 1) * HOP] ** 2).mean() + 1e-20)
                    for i in range(nf)])
    db = 20 * np.log10(rms / max(rms.max(), 1e-12) + 1e-12) if nf else np.array([])
    ph = [phase_error(at[i], as_[i])[0] for i in range(nf)
          if voi[i] and db[i] > GATE_DB and np.any(at[i])]
    ph = [x for x in ph if np.isfinite(x)]
    ft = f.result_track()
    bandwidth = min(0.5 * sr, tb.effective_bandwidth(raw_target, FS))
    evaluations = []
    for i, (seed, wave) in enumerate(zip(seeds, renders)):
        companion = renders[(i + 1) % len(renders)] if len(renders) > 1 else None
        fidelity = tb.spectral_fidelity(tgt, wave, companion, FS, f_max=bandwidth)
        fidelity["spectrum_match"] = tb.spectrum_match(tgt, wave, FS)
        evaluations.append(dict(seed=seed, fidelity=fidelity,
                                regions=tb.region_diagnostics(tgt, wave, FS, masks, bandwidth)))
    fid = evaluations[0]["fidelity"]
    cmp = tb.compare(tgt, out, FS)
    # `fine` 은 **fid 쪽 것을 쓴다.** `rep.fine` 은 손실이 쓰는 값이라 마찰 구간에서
    # 기대 스펙트럼 평활이 걸려 있다. 채점은 손실과 독립이어야 한다.
    result = dict(
        env=rep.env, **fid,
        phase=float(np.median(ph)) if ph else float("nan"),
        snr=seg_snr(ht, hs, FS),
        v1=float(np.median(np.abs(np.diff(np.asarray(ft["f1"]))))),
        v2=float(np.median(np.abs(np.diff(np.asarray(ft["f2"]))))),
        nh_synth=nonharmonic(out, hs, voi, nf),
        nh_target=nonharmonic(tgt, ht, voi, nf),
        centroid_err=cmp["centroid_err"], band_mae=cmp["band_mae_db"],
        sib_err=cmp["sibilance_err"], mod_mae=cmp["mod_mae_pct"],
        denoise=denoise_mode, coupling=coupling or "engine_default",
        fit_seed=cfg.seed, render_seeds=list(seeds), fits=1,
        score_note="fine_corr is a clipped heuristic, not a bound; trust is not statistical confidence",
        summary_note="same-fit seed distributions, not confidence intervals or perceptual equivalence",
        mask_definition=masks, bandwidth_hz=bandwidth, fit_bandwidth_hz=float(f.f_max),
        seeds=evaluations,
        fidelity_summary=tb.seed_summary([r["fidelity"] for r in evaluations]),
        region_summary={name: tb.seed_summary([r["regions"][name]["metrics"]
                                              for r in evaluations])
                        for name in masks["regions"]},
        preprocessing=dict(mode=denoise_mode,
                           over=0.5 if denoise_mode == "weak" else 1.5 if denoise_mode == "current" else None,
                           floor_db=-6 if denoise_mode == "weak" else -18 if denoise_mode == "current" else None),
        target_note="fit and metrics use the selected preprocessing target; masks always use raw target",
        preprocessing_change=tb.region_diagnostics(raw_target, _resample(seg, sr),
                                                  FS, masks, bandwidth),
    )
    if audio_dir:
        result["audio"] = write_listening(audio_dir, raw_target, tgt, renders, seeds)
    return result


def write_listening(directory, raw, target, renders, seeds):
    """Anonymous samples with a separate key; shared attenuation preserves levels."""
    os.makedirs(directory, exist_ok=True)
    candidates = [("raw_target", raw), ("fit_target", target)]
    candidates += [(f"synth_seed_{seed}", wave) for seed, wave in zip(seeds, renders)]
    gain = min(1.0, 0.95 / max(max(np.max(np.abs(w)), 1e-12) for _, w in candidates))
    order = np.random.default_rng(2718).permutation(len(candidates))
    key = {}
    for i, index in enumerate(order):
        label, wave = candidates[index]
        name = f"sample_{i + 1:02d}.wav"
        sf.write(os.path.join(directory, name), gain * wave, int(FS), subtype="PCM_24")
        key[name] = label
    manifest = dict(key=key, shared_gain=gain, sample_rate=int(FS),
                    note="Hide key.json and report metadata for informal blind listening; not a human study.")
    with open(os.path.join(directory, "key.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return dict(directory=str(directory), shared_gain=gain)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(v) for key, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def _integers(text):
    try:
        values = tuple(int(x) for x in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or min(values) < 0:
        raise argparse.ArgumentTypeError("expected nonnegative integers")
    return values


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="profiles/yang_female.json")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--quick", action="store_true", help="예산을 1/4 로 (개발용)")
    ap.add_argument("--only", default=None, help="이름에 이 문자열이 든 구간만")
    ap.add_argument("--no-probe", action="store_true",
                    help="lr 탐침을 끄고 고정값을 쓴다 (비용 절반, 반복 실행용)")
    ap.add_argument("--patience", type=int, default=0,
                    help="이 회차 동안 손실이 안 줄면 그 단계를 끝낸다 (0 = 끔)")
    ap.add_argument("--budget", type=_integers, help="global,stage,phase iterations, e.g. 2,1,0")
    ap.add_argument("--seeds", type=_integers, default=(0, 991, 2027),
                    help="same-fit render seeds; no refitting across seeds")
    ap.add_argument("--denoise", choices=("raw", "weak", "current", "all"), default="current")
    ap.add_argument("--coupling", "--noise-modulation", choices=("legacy", "lf", "all"), default=None,
                    help="engine turbulence coupling A/B; separate fits, fixed masks")
    ap.add_argument("--json", dest="json_path", help="structured report (nonfinite values become null)")
    ap.add_argument("--audio-dir", help="anonymous listening WAVs plus separate key.json")
    a = ap.parse_args(argv)
    if a.budget is not None and len(a.budget) != 3:
        ap.error("--budget needs exactly global,stage,phase iterations")
    if len(set(a.seeds)) != len(a.seeds):
        ap.error("--seeds must be unique")
    if a.patience < 0:
        ap.error("--patience must be nonnegative")
    prof = SpeakerProfile.load(a.profile)
    torch.set_num_threads(2)
    it = a.budget or ((60, 40, 250) if a.quick else (200, 150, 800))
    print(f"{'구간':>16s} {'포락':>6s} {'정밀':>6s} {'보정정밀':>9s} {'잡음비':>6s} "
          f"{'위상':>7s} {'조화SNR':>8s} {'무게중심':>9s} {'대역MAE':>8s} "
          f"{'변조MAE':>8s} {'비조화 합/목':>13s}")
    print("  * 보정정밀은 휴리스틱 추정: unresolved는 미분해이며 하한/신뢰구간이 아닙니다.")
    print("    trust is a residual-distance ratio, NOT confidence. No perceptual equivalence claim.\n")
    modes = ("raw", "weak", "current") if a.denoise == "all" else (a.denoise,)
    couplings = ("legacy", "lf") if a.coupling == "all" else (a.coupling,)
    report = dict(schema_version=1, budget=it, segments=[],
                  interpretation="Descriptive same-fit seed distributions, not confidence intervals. "
                                 "Fixed raw-target heuristic masks; selected denoise target for metrics. "
                                 "No objective metric establishes perceptual equivalence.")
    for name, path, t0, t1, kind in SEGMENTS:
        if a.only and a.only not in name:
            continue
        if not os.path.exists(path):
            print(f"{name:>16s}  (파일 없음: {path})")
            report["segments"].append(dict(name=name, path=path, status="unavailable",
                                           reason="missing corpus file"))
            continue
        for mode in modes:
            for coupling in couplings:
                t = time.time()
                audio_dir = (os.path.join(a.audio_dir, f"segment_{SEGMENTS.index((name, path, t0, t1, kind)):02d}",
                                         f"{mode}_{coupling or 'default'}") if a.audio_dir else None)
                r = bench(path, t0, t1, prof, a.verbose, it, probe=not a.no_probe,
                          patience=a.patience, denoise_mode=mode, seeds=a.seeds,
                          coupling=coupling, audio_dir=audio_dir)
                report["segments"].append(dict(name=name, path=path, t0=t0, t1=t1, kind=kind,
                                               seconds=time.time() - t, **r))
                print(f"{name:>16s} [{mode}/{coupling or 'default'}] "
                      f"{r['env']:6.2f} {r['fine']:6.2f} "
                      f"{r['fine_corr']:6.2f} ({r['correction_status']}) {r['noise_ratio']:6.2f} "
                      f"{r['phase']:6.1f}° {r['snr']:8.2f} "
                      f"{r['centroid_err']:8.0f} {r['band_mae']:8.2f} {r['mod_mae']:8.2f} "
                      f"{r['nh_synth']:6.1f}% /{r['nh_target']:5.1f}% voiced-only "
                      f"({time.time()-t:.0f}s)", flush=True)
                for region, stats in r["region_summary"].items():
                    median = stats.get("band_mae_db", {}).get("median")
                    print(f"    {region}: band MAE median={median if median is not None else 'unavailable'}; "
                          f"{len(a.seeds)} same-fit renders (details in --json)")
    if a.json_path:
        os.makedirs(os.path.dirname(os.path.abspath(a.json_path)), exist_ok=True)
        with open(a.json_path, "w", encoding="utf-8") as fh:
            json.dump(json_safe(report), fh, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
