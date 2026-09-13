"""손실 함수가 결함을 **보기는 하는가** — 목표에 결함을 심어 항마다 반응을 잰다.

    python scripts/probe_loss_blind.py out/L25/s040 --compare out/L25/s040 out/L31/s040
    python scripts/probe_loss_blind.py out/L25/s040 --expect 10

적합을 돌리지 않는다. 목표 파형 x 에 결함 하나씩을 심은 y 를 합성 출력 자리에 끼워
손실 항 전부를 계산한다. 결함을 넣었는데 항이 안 움직이면 그 항은 그 결함에 눈이 멀었다.
실제 적합 출력의 손실을 나란히 두어 크기를 비교한다 (MEASUREMENTS §50.2).

심는 결함: 고역(>9 kHz)을 포락 맞춘 평평한 줄로 / 고정 수준 줄 추가 / 고역 −20·−6 dB /
치찰음 F0 토막(깊이 0.35·0.7) / 고역 20 ms 세로블록 ±4 dB / 치찰음 20 ms 깊은 골.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from formant_ml.engine import fit as F                     # noqa: E402
from formant_ml.engine.analyze import analyze             # noqa: E402
from formant_ml.engine.profile import SpeakerProfile      # noqa: E402
from formant_ml.engine.segment import fricative_mask      # noqa: E402
from formant_ml.engine.voice import EngineConfig, VoiceEngine   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="목표를 읽을 적합 결과 접두어 (…_target.wav)")
    ap.add_argument("--compare", nargs="*", default=[], help="손실을 나란히 잴 적합 결과 접두어")
    ap.add_argument("--profile", default="profiles/yang_female.json")
    ap.add_argument("--expect", type=float, default=None, help="NOISE_EXPECT_MS 를 바꿔 잰다")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.expect is not None:
        F.NOISE_EXPECT_MS = float(a.expect)
    seg, sr = sf.read(a.stem + "_target.wav")
    seg = np.asarray(seg, np.float64)
    prof = SpeakerProfile.load(a.profile)
    hop = int(round(sr / 1000.0))
    track = analyze(seg, sr, prof, hop, t0=0.0, full=seg)
    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, speaker="female",
                                   residual=False), prof)
    for k in ("CONT_W", "SHARP_W", "SUBF0_W", "FLUX_W", "CORR_W", "HNR_W"):
        setattr(F, k, 1.0)                  # 모든 항을 켜야 반응을 볼 수 있다
    fit = F.CopySynthFitter(eng, seg, sr, track, phase_weight=1.0)
    fit.pulse_weight = 0.0
    fit._collect = True
    fs = fit.fs
    n = fit.target.shape[-1]
    x = fit.target[0].detach().cpu().double().numpy()

    def band(sig, lo, hi):
        X = np.fft.rfft(sig)
        f = np.fft.rfftfreq(len(sig), 1 / fs)
        return np.fft.irfft(X * ((f >= lo) & (f < hi)), len(sig))

    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    f0 = np.asarray(track["f0_target"], float)
    f0s = np.interp(np.arange(n) / hop, np.arange(len(f0)), np.where(f0 > 60, f0, 0))
    fm = fricative_mask(x, fs, prof, hop)
    fms = np.interp(np.arange(n) / hop, np.arange(len(fm)), fm.astype(float))
    hf = band(x, 9000, 24000)
    lo = x - hf
    w20 = int(0.020 * fs)
    e_hf = np.sqrt(np.convolve(hf * hf, np.ones(w20) / w20, "same") + 1e-20)
    lines = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
                for f in np.arange(9500, 20001, 700.0))
    lines = lines / np.sqrt(np.mean(lines ** 2))
    hf3, hf6 = band(x, 3000, 24000), band(x, 6000, 24000)
    ph = 2 * np.pi * np.cumsum(np.where(f0s > 0, f0s, 250.0)) / fs
    gate = 0.5 * (1 + np.cos(ph))
    blk = np.repeat(10 ** (rng.uniform(-4, 4, n // w20 + 2) / 20), w20)[:n]
    dip = 1 - 0.8 * np.exp(-0.5 * (((np.arange(n) % w20) / fs - 0.010) / 0.002) ** 2)
    V = {
        "목표 그대로": x,
        "고역(>9k)을 포락 맞춘 평평한 줄로": lo + lines * e_hf,
        "고역에 고정 수준 줄 추가(-10dB)": x + lines * np.sqrt(np.mean(hf ** 2)) * 10 ** (-0.5),
        "고역(>9k) -20 dB": lo + hf * 0.1,
        "고역(>9k) -6 dB": lo + hf * 0.5,
        "치찰음 F0 토막(깊이 .35)": x - hf3 + hf3 * (1 - 0.35 * gate * fms),
        "치찰음 F0 토막(깊이 .7)": x - hf3 + hf3 * (1 - 0.7 * gate * fms),
        "고역(>6k) 20ms 세로블록 ±4dB": x - hf6 + hf6 * blk,
        "치찰음 20ms 깊은 골(-14dB)": x * (1 - fms * (1 - dip)),
    }
    for stem in a.compare:
        y, _ = sf.read(stem + "_fit.wav")
        y = np.asarray(y, float)[:n]
        V["적합 " + stem.replace("out/", "")] = np.pad(y, (0, n - len(y)))

    mel = fit.mel.detach().cpu().numpy()
    cf = mel.argmax(1) * fs / F.MEL_FFT
    B = {"<3k": cf < 3000, "3-9k": (cf >= 3000) & (cf < 9000), ">9k": cf >= 9000}

    def run(y):
        yt = torch.as_tensor(y[None, :], dtype=fit.target.dtype, device=fit.target.device)
        fit.synth = lambda want_phase=False: yt
        with torch.no_grad():
            _, sc, env_sc, _ = fit.loss()
            T = {k: float(v) for k, v in fit._terms.items()}
            raw = fit._cabs(F._stft(yt, F.MEL_FFT, fit.wins[F.MEL_FFT]))[:, :fit.bin_max[F.MEL_FFT]]
            Mp = fit.mel[:, :raw.shape[1]] @ fit._expect(raw, F.MEL_FFT)
            m = min(Mp.shape[-1], fit.tgt_M.shape[-1])
            aa = fit._soft_floor(fit._db(fit.tgt_M[..., :m]), fit.db_floor_band, 0.5)
            bb = fit._soft_floor(fit._db(Mp[..., :m]), fit.db_floor_band, 0.5)
            e = fit._soft_abs(aa - bb, 0.5)[0].mean(-1).cpu().numpy()
        T["env_db"] = (T["env"] - float(env_sc) - 0.5 * float(sc)) / 4 * 20
        for k, mk in B.items():
            T["dB" + k] = float(e[mk].mean())
        return T

    cols = ["env", "env_db", "dB<3k", "dB3-9k", "dB>9k", "flux", "corr", "hnr", "phase"]
    print(f"[{os.path.basename(a.stem)}] NOISE_EXPECT_MS={F.NOISE_EXPECT_MS:g}, 멜 {len(cf)} 대역")
    print(f"{'변형':<34}" + "".join(f"{c:>9}" for c in cols))
    for k, y in V.items():
        T = run(y)
        print(f"{k:<34}" + "".join(f"{T.get(c, float('nan')):9.4f}" for c in cols), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
