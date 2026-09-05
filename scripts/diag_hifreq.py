"""고역이 '토막'으로 끊기는 원인을 재고 그린다.

    PYTHONPATH=src python3 scripts/diag_hifreq.py -o out/diag_hifreq.png

무엇을 재는가
-------------
HANDOFF_LIQUID §2 는 "상위 포먼트가 프레임마다 흔들리는데 지각과 맞는 지표를
못 찾았다" 로 끝난다. **맞는 지표는 이것이다**:

    3.5~9 kHz 켑스트럼 포락선의 **프레임간 |Δ| (dB)**

하모닉 빗살을 켑스트럼으로 지운 뒤 남는 포락선이 프레임마다 얼마나 뛰는지를
본다. 스펙트로그램에서 '토막' 으로 보이는 것이 정확히 이 값이다.
집계 지표(대역 에너지비, flux, 주기성)는 이걸 못 잡았다 — 저역이 압도해서다.

기준값(사용자 녹음): **원본 1.7~1.9 dB**. 합성이 4 dB 를 넘으면 토막이 보인다.

진단 결과 (2026-09-05)
----------------------
범인은 포먼트 궤적도, 대역폭도, 노이즈도, 지터/시머도 아니다.
`dsp/core.ltv_filter` 가 **여기신호를 240 샘플 직사각 블록으로 잘라** 각각 다른
IR 로 컨볼루션한 뒤 겹쳐 더하는 것이다. 창이 없으므로 블록 경계마다 스펙트럼이
번지고(직사각 = 주파수축 sinc), 응답이 프레임마다 다르면 그 번짐이 상쇄되지
않는다. 10 ms 격자의 타일이 그대로 보인다.

이 스크립트의 --ablate 는 그 근거를 표로 뽑는다.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import torch.nn.functional as TF

from formant_ml.analysis.acoustic import load
from formant_ml.config import Config
from formant_ml.dsp.core import fft_convolve, response_to_ir

SR = 24000
HI_LO, HI_HI = 3500.0, 9000.0


# --------------------------------------------------------------------- 지표
def envelope_db(y: np.ndarray, hop: int = 240, n_fft: int = 1024,
                lifter: int = 40) -> np.ndarray:
    """(T, F) 켑스트럼 평활 로그 스펙트럼. 하모닉 빗살을 지운 포락선."""
    t = max(1, 1 + (len(y) - n_fft) // hop)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(t)[:, None]
    D = 20 * np.log10(np.abs(np.fft.rfft(y[idx] * np.hanning(n_fft), n_fft,
                                         axis=1)) + 1e-9)
    c = np.fft.irfft(D, axis=1)
    c[:, lifter:-lifter] = 0.0
    return np.fft.rfft(c, axis=1).real


def hf_flutter(y: np.ndarray, hop: int = 240, n_fft: int = 1024):
    """고역 포락선의 프레임간 |Δ| 중앙값·p90 [dB]. 이 레포의 '토막' 지표."""
    f = np.fft.rfftfreq(n_fft, 1.0 / SR)
    k = (f >= HI_LO) & (f <= HI_HI)
    d = np.abs(np.diff(envelope_db(y, hop, n_fft)[:, k], axis=0))
    return float(np.median(d)), float(np.percentile(d, 90))


def band_db(y: np.ndarray, hop: int = 240, n_fft: int = 1024):
    """전체 대비 대역별 에너지 [dB]."""
    f = np.fft.rfftfreq(n_fft, 1.0 / SR)
    t = max(1, 1 + (len(y) - n_fft) // hop)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(t)[:, None]
    P = np.abs(np.fft.rfft(y[idx] * np.hanning(n_fft), n_fft, axis=1)) ** 2
    tot = P.sum()
    return [10 * np.log10(P[:, (f >= lo) & (f < hi)].sum() / tot + 1e-12)
            for lo, hi in ((100, 2500), (2500, 4000), (4000, 7000), (7000, 11000))]


# ------------------------------------------------------- 교차창 LTV (시제품)
def ltv_xfade(x: torch.Tensor, H: torch.Tensor, hop: int,
              ir_size: int = 512) -> torch.Tensor:
    """COLA(50 % 중첩 Hann) 교차창 시변 필터. `ltv_filter` 의 대안 시제품.

    현행 `ltv_filter` 는 창 없이 직사각 블록을 쓴다. 응답이 고정이면 완전복원이
    되지만, 프레임마다 다르면 블록 경계의 스펙트럼 번짐이 상쇄되지 않는다.
    주기적 Hann 은 50 % 중첩에서 합이 정확히 1 이라 응답이 고정이면 여전히
    완전복원이고(끝 프레임 제외), 변할 때는 부드럽게 넘어간다.
    """
    b, n = x.shape
    t = H.shape[1]
    L, n_pad = 2 * hop, t * hop
    if n < n_pad:
        x = TF.pad(x, (0, n_pad - n))
    xp = TF.pad(x[:, :n_pad], (hop, 2 * hop))
    fr = xp.unfold(1, L, hop)[:, :t]
    w = torch.hann_window(L, periodic=True, dtype=x.dtype, device=x.device)
    wet = fft_convolve(fr * w, response_to_ir(H, ir_size))
    out = torch.zeros(b, n_pad + 2 * hop + wet.shape[-1], dtype=x.dtype,
                      device=x.device)
    out = out.index_put_(
        (torch.arange(b, device=x.device)[:, None, None],
         (torch.arange(t, device=x.device)[:, None] * hop
          + torch.arange(wet.shape[-1], device=x.device)[None, :])[None]),
        wet, accumulate=True)
    d = ir_size // 2 + hop
    return out[:, d:d + n]


# ------------------------------------------------------------------- 그림
def plot(paths: dict[str, np.ndarray], out: str, fmax: float = 9000.0,
         fmin: float = 2000.0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def spec(y, n_fft=1024, hop=120):
        t = max(1, 1 + (len(y) - n_fft) // hop)
        idx = np.arange(n_fft)[None, :] + hop * np.arange(t)[:, None]
        S = np.abs(np.fft.rfft(y[idx] * np.hanning(n_fft), n_fft, axis=1)).T
        return 20 * np.log10(S + 1e-8)

    fig, ax = plt.subplots(len(paths), 1, figsize=(11, 3 * len(paths)),
                           sharex=True)
    top = None
    for a, (lab, y) in zip(np.atleast_1d(ax), paths.items()):
        y = y / max(abs(y).max(), 1e-9)
        S = spec(y)
        top = S.max() if top is None else top
        a.imshow(S, origin="lower", aspect="auto", cmap="magma",
                 vmin=top - 70, vmax=top, extent=[0, len(y) / SR, 0, SR / 2])
        a.set_ylim(fmin, fmax)
        a.set_ylabel("Hz")
        m, p = hf_flutter(y)
        a.set_title(f"{lab}   [HF flutter  med {m:.2f} / p90 {p:.2f} dB]",
                    fontsize=9, loc="left")
    np.atleast_1d(ax)[-1].set_xlabel("time (s)")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default="reference/recordings/"
                                     "ko_liquid_ra-eulla-ara_male_44k.wav")
    ap.add_argument("--from", dest="t0", type=float, default=0.50)
    ap.add_argument("--to", dest="t1", type=float, default=1.02)
    ap.add_argument("-o", "--out", default="out/diag_hifreq.png")
    ap.add_argument("--ablate", action="store_true",
                    help="어느 단이 토막을 만드는지 표로 뽑는다 (느리다)")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(__file__))
    from copysynth import analyse, build_controls, match_envelope
    import formant_ml.models.synth as S

    cfg = Config()
    hop = cfg.audio.hop_size
    y, _ = load(args.rec, SR)
    y = y[int(args.t0 * SR): int(args.t1 * SR)]
    y = y / max(abs(y).max(), 1e-9)
    a = analyse(y, cfg)
    torch.manual_seed(0)

    def render(xfade: bool, tilt: float = -12.0, **kw):
        orig = S.ltv_filter
        if xfade:
            S.ltv_filter = lambda x, H, h, ir_size=512, tail=None, \
                return_tail=False: ltv_xfade(x, H, h, ir_size)
        try:
            syn = S.PhysicalVoiceSynth(cfg, tract_mode="formant")
            c = build_controls(a, cfg, kw.get("jit", 0.002), kw.get("shim", 0.02),
                               tilt, 0.6)
            if kw.get("nonoise"):
                c.noise_bands = torch.zeros_like(c.noise_bands)
            if kw.get("freeze"):
                for nm in ("formant_freq", "formant_bw"):
                    v = getattr(c, nm)
                    setattr(c, nm, v.median(dim=1, keepdim=True).values
                            .expand_as(v).contiguous())
            with torch.no_grad():
                o = syn(c)["audio"][0].numpy().astype(np.float64)
            return match_envelope(o, a["rms"], hop)
        finally:
            S.ltv_filter = orig

    if args.ablate:
        # 그림 라벨은 ASCII 로 둔다 — matplotlib 기본 폰트에 한글이 없어서
        # 한글 라벨을 쓰면 전부 두부(tofu) 상자로 찍힌다.
        rows = [("original", "원본", y),
                ("current: rectangular-block OLA", "현행 (직사각 블록 OLA)",
                 render(False)),
                ("  + noise/jitter/shimmer off", "  + 노이즈/지터/시머 끔",
                 render(False, nonoise=1, jit=0., shim=0.)),
                ("  + formant track frozen", "  + 포먼트 궤적 고정",
                 render(False, nonoise=1, jit=0., shim=0., freeze=1)),
                ("cross-faded OLA, tilt=-12", "교차창 OLA, tilt=-12",
                 render(True)),
                ("cross-faded OLA, tilt=0", "교차창 OLA, tilt=0",
                 render(True, tilt=0.0))]
        print(f"\n{'':28s}  flut(med/p90)   0.1-2.5k  2.5-4k   4-7k   7-11k")
        for _, ko, s in rows:
            s = s / max(abs(s).max(), 1e-9)
            m, p = hf_flutter(s)
            print(f"{ko:28s}  {m:5.2f} / {p:5.2f}   "
                  + "  ".join(f"{x:7.2f}" for x in band_db(s)))
        plot({lab: s for lab, _, s in rows}, args.out)
    else:
        plot({"original": y,
              "current (rectangular-block OLA)": render(False),
              "prototype (COLA cross-faded OLA)": render(True)}, args.out)


if __name__ == "__main__":
    main()
