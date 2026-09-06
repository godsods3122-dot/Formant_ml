#!/usr/bin/env python3
"""유음을 낸다 — 포먼트 궤적을 성도 캐스케이드에 통과시킨다.

    PYTHONPATH=src python scripts/gen_liquid.py -o out/liquid

근거는 `docs/LIQUID.md`, 궤적은 `formant_ml.liquid`. 경로는 사용자가 지정한
**성도 + 필터**(`tract_mode="formant"`)다 — 복사합성과 같은 경로이므로
거기서 고친 것들(고역, 소스 기울기)이 그대로 적용된다.

여기서 만드는 것은 '라 / 아라 / 을라' 세 음절이고, 각각 어두 설측음 /
모음사이 탄음 / 긴 설측음이다. **셋이 같은 코드 경로**이고 `hold_ms` 와
`body_raise` 만 다르다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from formant_ml.config import Config
from formant_ml.liquid import LiquidGesture, liquid_track
from formant_ml.models.synth import Controls, PhysicalVoiceSynth
from formant_ml.utils import save_wav

#: 이 화자의 /아/ (사용자 녹음 실측, LIQUID.md §2.1). F4 위는 추적기 값의 중앙값.
VOWEL_A = (700.0, 1200.0, 2450.0, 3300.0, 4400.0, 5300.0,
           6200.0, 7100.0, 8000.0, 8900.0, 9800.0, 10700.0)

#: 성문 소스. 복사합성이 고른 값과 같다(`scripts/copysynth.py` 의 기본값).
DEFAULT_TILT, DEFAULT_RD = 1.0, 0.6


def fant_bw(f: np.ndarray) -> np.ndarray:
    return 50.0 + 20.0 * (f / 1000.0) ** 2 + 10.0 * (f / 1000.0)


def match_envelope(y: np.ndarray, target: np.ndarray, hop: int,
                   floor: float = 1e-5) -> np.ndarray:
    """출력의 프레임 세기를 `target` 곡선에 맞춘다 (copysynth 와 같은 발상).

    **왜 두 번 렌더하는가.** 유음은 F1 을 700 -> 333 으로 내리는데 그러면
    캐스케이드 자체의 이득이 변한다. 세기를 `depth_db` 만큼 낮추라고 지시해도
    출력은 그만큼 안 낮아진다 — 실측: 지시 -8 dB 에 실현 -10.5(탄음) /
    -12.3(긴 설측음).

    필터 이득을 해석적으로 계산해 나눠 보려 했는데(소스 기울기를 1/f 로 근사),
    **과보정됐다**: -12.3 이 -3.2 가 됐다. 실제 소스(Rd=0.6, tilt=1)가 1/f 와
    충분히 다르다. 근사를 다듬는 대신 **한 번 렌더해서 재고 다시 맞춘다.**
    그러면 `depth_db` 가 정의상 실현값이 된다.

    이름과 동작이 어긋나는 파라미터는 이 레포가 반복해서 당한 함정이다
    (`bw_neutral` 이 중립이 아니었던 것, `radiation` 버퍼가 항등이었던 것).
    """
    t = len(target)
    n = t * hop
    y = np.pad(y, (0, max(0, n - len(y))))[:n]
    cur = np.sqrt((y.reshape(t, hop) ** 2).mean(axis=1))
    g = np.where(cur > floor, target / np.maximum(cur, floor), 0.0)
    # 프레임 경계에서 계단이 지지 않도록 표본률로 선형보간한다.
    gs = np.interp(np.arange(n), np.arange(t) * hop + hop / 2.0, g)
    return y * gs


def render(gesture: LiquidGesture, seconds: float, f0: float = 108.0,
           cfg: Config | None = None, seed: int = 0) -> np.ndarray:
    """모음-유음-모음 하나를 낸다."""
    cfg = cfg or Config()
    sr, hop = cfg.audio.sample_rate, cfg.audio.hop_size
    K = cfg.filt.n_formants
    t = int(seconds * sr / hop)

    vowel = np.tile(np.asarray(VOWEL_A[:K], dtype=np.float64), (t, 1))
    F, gain = liquid_track(vowel, gesture, t, hop, sr)
    B = fant_bw(F)

    # 세기 포락선: 앞뒤로 모음이 붙고 가운데가 유음이다. 발화 시작/끝은
    # 부드럽게 열고 닫는다 — 계단으로 두면 '툭' 소리가 난다.
    ms = np.arange(t) * hop / sr * 1000.0
    edge = np.clip(ms / 60.0, 0, 1) * np.clip((ms[-1] - ms) / 60.0, 0, 1)
    amp = (edge ** 2 * (3 - 2 * edge)) * gain

    # F0 는 완만히 내려간다(실측 108 -> 103).
    f0_track = np.linspace(f0, f0 * 0.95, t)

    T = lambda x: torch.tensor(np.asarray(x, dtype=np.float32))
    c = Controls(
        f0=T(f0_track).reshape(1, t, 1),
        harmonic_amp=T(amp).reshape(1, t, 1),
        rd=torch.full((1, t, 1), DEFAULT_RD),
        formant_freq=T(F).reshape(1, t, K),
        formant_bw=T(B).reshape(1, t, K),
        formant_gain=torch.ones(1, t, K),
        # 유성 기식 바닥. 복사합성과 같은 뜻이다(성문에서 나므로 성도 전체를
        # 지난다 -> noise_entry=0).
        noise_bands=(T(amp * 0.10).reshape(1, t, 1)
                     * torch.ones(1, 1, cfg.noise.n_bands)).contiguous(),
        noise_entry=torch.zeros(1, t, 1),
        noise_am=torch.full((1, t, 1), 0.5),
        tilt=torch.full((1, t, 1), DEFAULT_TILT),
        jitter=torch.full((1, t, 1), 0.002),
        shimmer=torch.full((1, t, 1), 0.02),
    )
    syn = PhysicalVoiceSynth(cfg, tract_mode="formant")
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        y = syn(c, generator=g)["audio"][0].numpy().astype(np.float64)
    y = match_envelope(y, amp, hop)
    return y / max(np.abs(y).max(), 1e-9) * 0.9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="out/liquid")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = Config()

    # 실측 토큰과 같은 길이로 낸다 (reference/README.md 의 구간 표).
    jobs = [("ara_tap", "tap", 0.66), ("ra_onset", "onset", 0.52),
            ("eulla_lateral", "lateral", 0.97)]
    for name, preset, seconds in jobs:
        g = LiquidGesture.preset(preset)
        y = render(g, seconds, cfg=cfg, seed=args.seed)
        path = os.path.join(args.out, f"{name}.wav")
        save_wav(path, torch.from_numpy(y).float(), cfg.audio.sample_rate,
                 normalize=False)
        print(f"{path}  {seconds:.2f}s  {preset}: "
              f"유지 {g.hold_ms:.0f} ms, 전체 {g.total_ms:.0f} ms, F2 {g.f2:.0f} Hz")


if __name__ == "__main__":
    main()
