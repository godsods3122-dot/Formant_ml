"""복사합성 — 녹음의 노이즈를 지우고, 스펙트럼이 맞을 때까지 물리 파라미터를 추적한다.

    OMP_NUM_THREADS=2 python scripts/copyfit.py data/voices/yang_00000034.wav \
        --profile profiles/yang_female.json --from 0.510 --to 0.665 --out out/fit

무엇을 하는가
    1. 잡음 프로파일 추정 -> 위너 스펙트럼 차감 (engine/denoise.py)
    2. 1 ms 프레임마다 F0·유성도·포먼트·세기·마찰 추정 -> 제어열 초기값 (engine/analyze.py)
    3. 미분가능 엔진으로 제어열을 역추정 (engine/fit.py) — 전역 스칼라 -> 성김/촘촘함 프레임별
    4. 원본·복원·차이를 wav 로, 제어열을 npz 로, 일치율 표를 표준출력으로

일치율을 어떻게 읽는가
    **포락** (멜 dB) — 조음이 맞는가. 사람이 듣는 음색에 대응한다.
    **정밀** (선형 다해상도 STFT) — F0 궤적과 성문 펄스 위치까지 맞는가.
    **보정 정밀** — 실현 분산을 추정해 뺀 진단값. 대역·변조·잡음비와 함께 읽는다.

34.5 % 는 동일 PSD 의 독립 가우시안 잡음을 비평활 STFT 크기로 비교할 때의
이론적 기준이지 모든 치찰음의 상한이 아니다. `trust` 는 분산 차감 뒤 남은 거리
비율이며 신뢰확률이 아니다. 분해 불가인 보정값은 신뢰구간의 하한도 아니다.
어떤 일치율도 인간과 구별 불가능함을 보장하지 않는다.
자세한 것은 engine/turbulence.py 머리말과 docs/MEASUREMENTS.md §9.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import soundfile as sf
import torch

from formant_ml.engine.analyze import analyze
from formant_ml.engine.control import INDEX, PARAM_NAMES
from formant_ml.engine.denoise import denoise, noise_profile, snr_report
from formant_ml.engine.fit import CopySynthFitter
from formant_ml.engine.profile import DEFAULT_PROFILE, SpeakerProfile
from formant_ml.engine.voice import EngineConfig, VoiceEngine

BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 5000),
         (5000, 8000), (8000, 12000), (12000, 24000)]


def band_db(x: np.ndarray, sr: int) -> list[float]:
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    tot = p.sum() + 1e-20
    return [10 * np.log10(p[(f >= lo) & (f < hi)].sum() / tot + 1e-20) for lo, hi in BANDS]


def fidelity_summary(fid: dict) -> str:
    score = f"{fid['fine_corr']:.2f} %" if fid["resolved"] else "분해 불가"
    return (f"보정 정밀: {score} (진단값 {fid['fine_corr']:.2f}, "
            f"잔여거리 비율 {fid['trust']:.2f}, 잡음비 {fid['noise_ratio']:.2f}, "
            f"실현 기준 추정 {fid['floor']:.1f} %, "
            f"시간평균 스펙트럼 {fid['spectrum_match']:.1f} %)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--profile", default=None, help="화자 프로파일 json")
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=None)
    ap.add_argument("--out", default="out/fit")
    ap.add_argument("--frame-ms", type=float, default=1.0)
    ap.add_argument("--global-iters", type=int, default=200)
    ap.add_argument("--stage-iters", type=int, default=100)
    ap.add_argument("--phase-iters", type=int, default=200,
                    help="마지막 위상 단계. 0 이면 끔 (크기만 맞춘다)")
    ap.add_argument("--lr-global", type=float, default=None,
                    help="생략하면 짧은 탐침으로 자동 선택 (구간마다 맞는 값이 다르다)")
    ap.add_argument("--lr-frame", type=float, default=0.04)
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--phase", type=float, default=0.0, help="복소 STFT 항 가중(2 단계용)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--patience", type=int, default=0,
                    help="이 회차 동안 손실이 안 줄면 그 단계를 끝낸다 (0 = 끔). "
                         "긴 음원에서는 켜는 편이 낫다 — 수렴한 단계에 예산을 다 쓴다")
    ap.add_argument("--room-from", default=None, metavar="STEM",
                    help="이미 있는 **마른** 적합 결과(copyfit --out 값)에서 녹음 경로 IR 을 "
                         "추정해 순방향 모형에 넣는다. 그러면 적합기가 방·마이크·코덱의 "
                         "응답을 성도·소스 파라미터로 흡수하지 않아 추정되는 물리량이 "
                         "**마른 목소리**의 것이 된다 (engine/room.py, MEASUREMENTS §23)")
    ap.add_argument("--room-taps", type=int, default=None, help="IR 탭 수 (기본 4096 = 85 ms)")
    ap.add_argument("--room-force", action="store_true",
                    help="홀드아웃에서 좋아지지 않아도 방을 넣는다")
    ap.add_argument("--room-lambda", type=float, default=None,
                    help="IR 추정의 정규화 세기 (기본 0.1). 크면 IR 이 δ 에 가까워져 "
                         "방의 시간 구조가 사라진다")
    ap.add_argument("--prior", action="append", default=None, metavar="이름=값",
                    help="사전 가중(fit.PRIOR_W) 덮어쓰기. 여러 번 줄 수 있다. "
                         "예: --prior aspiration=3.0")
    ap.add_argument("--tilt-cap", type=float, default=None,
                    help="소스 기울기(tilt)가 먹히는 상한 주파수 [Hz] "
                         "(glottis.TILT_MAX_HZ, 기본 5000). 0 이면 상한 없음 — "
                         "그러면 성문 펄스가 사각파처럼 날카로워진다 (MEASUREMENTS §25)")
    ap.add_argument("--artic-vel", type=float, default=None,
                    help="조음 속도 한계 벌점의 세기 (fit.ARTIC_VEL_W, 기본 1.0). "
                         "0 이면 끈다 — 마찰음이 유성 구간에서 1 ms 만에 켜지는 것을 "
                         "막는 항이다 (MEASUREMENTS §24)")
    ap.add_argument("--ripple", type=float, default=None,
                    help="제어열 잔물결 벌점 세기 (fit.RIPPLE_W). 조음 대역(0~20 Hz) "
                         "위에서 트랙이 흔들리는 것만 문다 — 지지직과 저역 초과가 "
                         "둘 다 여기서 온다 (docs/MEASUREMENTS.md §13, §16)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.ripple is not None or a.prior or a.artic_vel is not None:
        from formant_ml.engine import fit as _fit
        if a.ripple is not None:
            _fit.RIPPLE_W = float(a.ripple)
        if a.artic_vel is not None:
            _fit.ARTIC_VEL_W = float(a.artic_vel)
    if a.tilt_cap is not None:
        from formant_ml.engine import glottis as _g
        _g.TILT_MAX_HZ = float(a.tilt_cap)
        for item in (a.prior or ()):
            k, _, v = item.partition("=")
            if k not in _fit.PRIOR_W and k not in _fit.DEFAULT_PARAMS:
                raise SystemExit(f"--prior: 모르는 파라미터 {k!r}")
            _fit.PRIOR_W[k] = float(v)

    y, sr = sf.read(a.wav)
    if y.ndim > 1:
        y = y.mean(1)
    y = np.asarray(y, dtype=np.float64)
    if not a.no_denoise:
        noise = noise_profile(y, sr)
        yd = denoise(y, sr, noise)
        r = snr_report(y, yd, sr, noise)
        print(f"잡음 제거: 바닥 {r['noise_db_in']:.1f} -> {r['noise_db_out']:.1f} dB "
              f"(−{r['removed_db']:.1f} dB), 정점 {r['peak_db_in']:.1f} -> "
              f"{r['peak_db_out']:.1f} dB")
        y = yd

    t1 = a.t1 if a.t1 is not None else len(y) / sr
    seg = y[int(a.t0 * sr):int(t1 * sr)]
    prof = SpeakerProfile.load(a.profile) if a.profile else DEFAULT_PROFILE
    hop = max(1, int(round(a.frame_ms * sr / 1000.0)))
    track = analyze(seg, sr, prof, hop, t0=a.t0, full=y)
    print(f"구간 {a.t0:.3f}~{t1:.3f} s, {track.n_frames} 프레임 × {track.frame_ms} ms",
          flush=True)

    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=a.frame_ms,
                                   speaker="female" if prof.f0_nominal > 165 else "male",
                                   residual=False), prof)
    room_ir = None
    if a.room_from:
        from formant_ml.engine import room as _room
        # **마른 출력이 있으면 그쪽을 쓴다.** 이미 방을 넣고 적합한 결과라면
        # `_fit.wav` 는 방을 통과한 소리다 — 그걸로 다시 추정하면 방이 두 번 들어간다.
        src = (a.room_from + "_dry.wav" if os.path.exists(a.room_from + "_dry.wav")
               else a.room_from + "_fit.wav")
        dry, _ = sf.read(src)
        ref, _ = sf.read(a.room_from + "_target.wav")
        dry = dry.mean(1) if dry.ndim > 1 else dry
        ref = ref.mean(1) if ref.ndim > 1 else ref
        m = min(len(dry), len(ref))
        kw = dict(taps=a.room_taps or _room.DEFAULT_TAPS,
                  lam=a.room_lambda if a.room_lambda is not None else _room.DEFAULT_LAMBDA)
        # **못 본 절반에서 실제로 좋아지는지 먼저 본다.** IR 이 항상 도움이 되지는
        # 않는다 — 마른 모형이 이미 잘 맞는 파일에서는 역합성곱이 잡을 계통 성분이
        # 없고, 그때 IR 은 오히려 나빠지게 한다 (engine/room.holdout_gain).
        hg = _room.holdout_gain(dry[:m], ref[:m], **kw)
        room_ir = _room.estimate_ir(dry[:m], ref[:m], **kw)
        print(f"녹음 경로 IR: {src} 에서 {len(room_ir)} 탭 "
              f"({len(room_ir)/48000*1000:.0f} ms) 추정, 직접음 {_room.direct_gain(room_ir):.3f}",
              flush=True)
        print(f"  홀드아웃(앞 절반 추정 → 뒤 절반 시험): {hg['before']:.3f} -> {hg['after']:.3f}"
              f"  {'쓴다' if hg['improves'] else '**안 좋아진다**'}", flush=True)
        if not hg["improves"] and not a.room_force:
            print("  -> 방을 넣지 않는다 (--room-force 로 강제 가능)", flush=True)
            room_ir = None
    fit = CopySynthFitter(eng, seg, sr, track, phase_weight=a.phase, room_ir=room_ir)
    kw = {} if a.lr_global is None else {"lr_global": a.lr_global}
    rep = fit.fit_staged(global_iters=a.global_iters, stage_iters=a.stage_iters,
                         lr_frame=a.lr_frame, phase_iters=a.phase_iters,
                         patience=a.patience, **kw)
    print(rep)

    # 보정값은 진단용이며 청취상 동등성이나 신뢰구간을 뜻하지 않는다.
    fid = fit.fidelity()
    print(fidelity_summary(fid))
    if not fid["resolved"]:
        print("  * 분산 보정으로 편향을 판별하지 못했다. 하한·합격 판정으로 쓰지 말고 "
              "잡음비와 아래 치찰음 지문을 함께 읽을 것.")

    out = fit.render()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sf.write(a.out + "_target.wav", seg, sr)
    sf.write(a.out + "_fit.wav", out, 48000)
    if room_ir is not None:
        # 마른 출력도 같이 낸다 — 학습 데이터로 나가는 것은 이쪽의 파라미터다.
        with torch.no_grad():
            sf.write(a.out + "_dry.wav", fit.synth_dry()[0].cpu().numpy(), 48000)
        np.save(a.out + "_room.npy", room_ir)
    res = fit.result_track()
    np.savez(a.out + "_track.npz", values=res.values, frame_ms=res.frame_ms,
             names=np.array(PARAM_NAMES))
    with open(a.out + "_report.json", "w", encoding="utf-8") as f:
        json.dump({"env": rep.env, "fine": rep.fine, "per_size": rep.per_size,
                   "fine_corr": fid["fine_corr"], "floor": fid["floor"],
                   "resolved": fid["resolved"], "trust": fid["trust"],
                   "noise_ratio": fid["noise_ratio"],
                   "score_kind": "diagnostic_not_perceptual_equivalence",
                   "spectrum_match": fid["spectrum_match"],
                   "loss": rep.loss, "gain_db": fit.gain_db(),
                   "moved": {n: [x, z] for n, x, z in fit.moved()}},
                  f, ensure_ascii=False, indent=1)

    print("\n대역 에너지 (총합 대비 dB)")
    print("           " + " ".join(f"{lo // 1000}-{hi // 1000}k".rjust(6) for lo, hi in BANDS))
    print("목표      " + " ".join(f"{v:6.1f}" for v in band_db(seg, sr)))
    print("합성      " + " ".join(f"{v:6.1f}" for v in band_db(out, 48000)))
    # 치찰음 지문 — 조음 위치가 맞는가 (Jongman et al. 2000). 대역 에너지표가 맞아도
    # 무게중심이 1 kHz 어긋나면 /s/ 가 /ʃ/ 로 들린다.
    from formant_ml.engine import turbulence as tb
    c = tb.compare(fit.target[0].numpy(), out, 48000.0)
    print("\n치찰음 지문 (목표 -> 합성)")
    print(f"  무게중심 {c['target']['centroid']:7.0f} -> {c['synth']['centroid']:7.0f} Hz "
          f"({c['centroid_err']:+.0f})   봉우리 {c['target']['peak']:6.0f} -> "
          f"{c['synth']['peak']:6.0f} Hz ({c['peak_err']:+.0f})")
    print(f"  치찰도   {c['target']['sibilance']:7.1f} -> {c['synth']['sibilance']:7.1f} dB "
          f"({c['sibilance_err']:+.1f})   대역 MAE {c['band_mae_db']:.2f} dB   "
          f"변조 MAE {c['mod_mae_pct']:.1f} %p")

    # 지글거림 — **마찰 프레임만** 골라 고역 포락의 변조 지수를 목표와 나눈다.
    # 위의 "변조 MAE" 는 파일 전체를 대역별 백분율로 재므로 마찰 구간의 맥동이
    # 유성 구간에 묻힌다 (docs/MEASUREMENTS.md §13).
    ac = res.values[:, INDEX["a_c"]]
    ps = res.values[:, INDEX["p_sub"]]
    f0v = res.values[:, INDEX["f0_target"]]
    hop48 = int(round(res.frame_ms * 48000.0 / 1000.0))
    for label, sel in (("마찰", (ps > 2.0) & (ac < 0.5)),
                       ("유성", (ps > 2.0) & (ac > 1.0))):
        if sel.sum() < 20:
            continue
        f0m = float(np.median(f0v[sel]))
        mb = [(0.8 * f0m, 2.5 * f0m), (60.0, 150.0), (150.0, 400.0)]
        msk = np.repeat(sel.astype(float), hop48)
        tgt48 = fit.target[0].numpy()
        vt, lt = tb.env_modulation_index(tgt48, 48000.0, msk, mb)
        vs, ls = tb.env_modulation_index(out, 48000.0, msk, mb)
        r = vs / np.maximum(vt, 1e-30)
        print(f"  지글거림({label} {sel.mean()*100:2.0f}%, F0 {f0m:.0f} Hz)  "
              f"F0대역 {r[0]:5.2f}x  60-150 {r[1]:5.2f}x  150-400 {r[2]:5.2f}x   "
              f"대역레벨 {lt:.1f} -> {ls:.1f} dB")

    print("\n움직인 파라미터 (초기 중앙값 -> 적합 중앙값)")
    for n, x, z in fit.moved():
        if abs(z - x) > 0.05 * max(abs(x), 1e-6):
            print(f"  {n:12s} {x:10.3f} -> {z:10.3f}")
    print(f"\n결과: {a.out}_target.wav / {a.out}_fit.wav / {a.out}_track.npz")


if __name__ == "__main__":
    main()
