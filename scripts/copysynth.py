"""복사합성 — 녹음을 분석해 그 파라미터로 물리모델을 구동한다.

    PYTHONPATH=src python3 scripts/copysynth.py IN.wav -o out/recon.wav [--from S --to S]

왜 이걸 먼저 하는가
-------------------
음절을 손으로 작곡하면 "스펙트럼 지표는 맞는데 사람 소리가 아닌" 상태에서
빠져나올 방법이 없다. 무엇이 틀렸는지 비교할 정답이 없기 때문이다.
복사합성은 **정답이 있다** — 원본 파형. 되합성이 원본처럼 안 들리면 엔진이
틀린 것이고, 어느 파라미터가 부족한지 바로 좁혀진다.
PLAN.md 의 Phase 1 이고, 이걸 건너뛰고 Phase 3(작곡)부터 한 것이 실패의 원인이다.

경로는 **포먼트 캐스케이드**를 쓴다. 도파관은 이득 정규화가 아직 안 풀렸고
(방사 임피던스 부재), 캐스케이드는 이 레포가 목표치 ±3 % 로 검증해 두었다.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from formant_ml.analysis.acoustic import load as load16
from formant_ml.analysis.track import track_formants
from formant_ml.config import Config
from formant_ml.data.features import yin_f0
from formant_ml.models.synth import Controls, PhysicalVoiceSynth
from formant_ml.utils import save_wav


def frame_rms(y: np.ndarray, hop: int, win: int) -> np.ndarray:
    t = max(0, 1 + (len(y) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(t)[:, None]
    return np.sqrt((y[idx] ** 2).mean(1))


def analyse(y: np.ndarray, cfg: Config):
    """파형 -> 복사합성에 필요한 프레임별 파라미터."""
    sr, hop = cfg.audio.sample_rate, cfg.audio.hop_size
    win = 1024
    F, B = track_formants(y, sr, hop, win, n=cfg.filt.n_formants)
    t = len(F)
    yt = torch.from_numpy(y).float()[None]
    f0, voi = yin_f0(yt, sr, hop)
    f0, voi = f0[0, :t].numpy(), voi[0, :t].numpy()
    rms = frame_rms(y, hop, win)[:t]
    # F0 를 유성 구간 중앙값으로 메운다 (무성/무음 구간에 0 이 들어가면
    # 합성기가 위상을 못 잇는다)
    med = float(np.median(f0[voi > 0.5])) if (voi > 0.5).any() else 120.0
    f0 = np.where(voi > 0.5, f0, med)
    f0 = np.clip(f0, cfg.source.f0_min, cfg.source.f0_max)
    return dict(freq=F, bw=B, f0=f0, voicing=voi, rms=rms, t=t)


def build_controls(a: dict, cfg: Config, jitter: float, shimmer: float,
                   tilt: float, rd: float):
    t, K = a["t"], cfg.filt.n_formants
    nb = cfg.noise.n_bands

    def T(x):
        return torch.from_numpy(np.asarray(x, dtype=np.float32))

    freq = T(a["freq"]).clamp(cfg.filt.f_min, cfg.filt.f_max)[None]
    bw = T(a["bw"]).clamp(cfg.filt.bw_min, cfg.filt.bw_max)[None]
    rms = a["rms"] / max(a["rms"].max(), 1e-9)
    voi = a["voicing"]
    # 난류 소스의 스펙트럼 기울기. **평탄한 백색으로 두면 안 된다** — 노이즈
    # 경로를 재니 +13.4 dB/oct 로 치솟아(캐스케이드의 고역 극들을 그대로 통과)
    # 유성 구간에 섞인 2 % 만으로도 고역 전체를 덮어 "쨍한" 소리를 만들었다.
    # 실제 기식/마찰 소스는 고역으로 갈수록 떨어진다.
    fb = np.linspace(0.0, cfg.audio.sample_rate / 2, nb)
    shape = (1.0 / (1.0 + (fb / 700.0) ** 2)) ** 0.9
    shape = shape / shape.max()
    return Controls(
        f0=T(a["f0"]).reshape(1, t, 1),
        harmonic_amp=T(rms * voi).reshape(1, t, 1),
        rd=torch.full((1, t, 1), rd),
        formant_freq=freq, formant_bw=bw,
        formant_gain=torch.ones(1, t, K),
        # 무성 구간의 마찰 성분. 유성 구간에는 소량의 기식만 남긴다.
        noise_bands=(T(rms * (0.25 * (1.0 - voi) + 0.002 * voi)
                       ).reshape(1, t, 1) * T(shape).reshape(1, 1, nb)
                     ).contiguous(),
        # 기식은 **성문**에서 난다 -> 성도 전체를 지난다. 여기를 K*0.7 로 두면
        # 노이즈가 상위 포먼트(5~11 kHz)만 통과해 +33 dB/oct 로 치솟고,
        # 유성 구간에 섞인 2 % 만으로도 고역을 덮어 "쨍한" 소리가 된다.
        noise_entry=torch.zeros(1, t, 1),
        noise_am=torch.full((1, t, 1), 0.5),
        tilt=torch.full((1, t, 1), tilt),
        jitter=torch.full((1, t, 1), jitter),
        shimmer=torch.full((1, t, 1), shimmer),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("-o", "--out", default="out/recon.wav")
    ap.add_argument("--from", dest="t0", type=float, default=None)
    ap.add_argument("--to", dest="t1", type=float, default=None)
    ap.add_argument("--jitter", type=float, default=0.006)
    ap.add_argument("--shimmer", type=float, default=0.04)
    ap.add_argument("--tilt", type=float, default=0.0)
    ap.add_argument("--rd", type=float, default=1.2)
    args = ap.parse_args()

    cfg = Config()
    y, sr = load16(args.wav, cfg.audio.sample_rate)
    if args.t0 is not None:
        y = y[int(args.t0 * sr): int((args.t1 or len(y) / sr) * sr)]
    y = y / max(np.abs(y).max(), 1e-9)

    a = analyse(y, cfg)
    syn = PhysicalVoiceSynth(cfg, tract_mode="formant")
    c = build_controls(a, cfg, args.jitter, args.shimmer, args.tilt, args.rd)
    with torch.no_grad():
        out = syn(c)["audio"]
    save_wav(args.out, out, sr)
    print(f"{args.out}  ({a['t'] * cfg.audio.hop_size / sr:.2f}s, "
          f"{a['t']} 프레임, 유성 {float((a['voicing'] > 0.5).mean()):.2f})")


if __name__ == "__main__":
    main()
