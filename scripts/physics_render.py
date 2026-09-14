"""물리 음원을 기존 성도에 물려 소리를 낸다 — **음원만 바꾼 A/B**.

같은 제어열(적합된 트랙), 같은 성도, 같은 잡음 가지. 다른 것은 성문 음원 하나다:

  * `lf`      — 지금 엔진의 LF 모델 (Rd 하나)
  * `physics` — `formant_ml.physics.transient` 의 모드 링다운 음원

    python scripts/physics_render.py out/M/M14/s101 --out out/PHY/s101
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from formant_ml.engine.control import INDEX, PARAM_NAMES, ControlTrack   # noqa: E402
from formant_ml.engine.tract import VocalTract                           # noqa: E402
from formant_ml.physics.transient import modal_source                    # noqa: E402


def strain_for_f0(f0, lo=232.0, hi=634.0):
    """F0 -> 늘어남. `fold_modes` 로 만든 대응표의 역함수를 선형으로 쓴다."""
    x = np.clip((np.asarray(f0, float) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return 0.02 + 0.48 * x


def build_lf_source(track_vals, names, fs, hop, rd=2.16, jitter=0.0, shimmer=0.0, seed=0):
    """**대조군** — 같은 F0 궤적을 따라 엔진의 LF 펄스(Rd 고정)를 잇는다.

    지터·시머는 물리 음원과 **같은 값**을 줘야 모양만 가르는 A/B 가 된다.
    """
    rng = np.random.default_rng(seed)
    from formant_ml.engine.glottis import lf_pulse
    f0 = track_vals[:, names.index("f0_target")]
    n = len(f0) * hop
    du = np.zeros(n)
    base = lf_pulse(rd, n=4096)
    i = 0
    while i < n:
        fr = min(i // hop, len(f0) - 1)
        if f0[fr] <= 0:
            i += hop
            continue
        per = max(int(round((fs / f0[fr]) * (1.0 + jitter * rng.normal()))), 8)
        one = np.interp(np.linspace(0, len(base) - 1, per), np.arange(len(base)), base)
        one = one * max(0.0, 1.0 + shimmer * rng.normal())
        m = min(per, n - i)
        du[i:i + m] = one[:m]
        i += per
    if np.abs(du).max() > 0:
        du = du / np.abs(du).max()
    return du, (np.abs(du) > 0).astype(float)


def build_source(track_vals, names, fs, hop, seed=0, jitter=0.0, shimmer=0.0, mode_var=0.0):
    """제어열의 F0 를 따라 물리 음원을 잇는다. 반환 (du, area)."""
    f0 = track_vals[:, names.index("f0_target")]
    n = len(f0) * hop
    du = np.zeros(n)
    area = np.zeros(n)
    # F0 가 크게 바뀌는 구간마다 나눠서 만든다 (모드 계산이 비싸다)
    seg, i = [], 0
    while i < len(f0):
        j = i
        while j + 1 < len(f0) and abs(f0[j + 1] - f0[i]) < 25.0 and j - i < 400:
            j += 1
        seg.append((i, j + 1))
        i = j + 1
    for k, (a, b) in enumerate(seg):
        fm = float(np.median(f0[a:b]))
        ns = (b - a) * hop
        if fm <= 0 or ns < hop:
            continue
        # 수렴·검증된 설정 (MEASUREMENTS §52): 완전 내전(h0 0), 절대 벌어짐 0.30 mm,
        # 적분 간격 2 배 세분. 지터·시머는 끈다 — 음원 모양만 LF 와 가르는 대조다.
        r = modal_source(strain=float(strain_for_f0(fm)), contact=0.0,
                         f0=max(fm, 60.0), seconds=ns / fs, fs=fs, seed=seed + k,
                         jitter=jitter, shimmer=shimmer, mode_var=mode_var,
                         h0=0.0, open_amp=0.30e-3, substeps=2, n_modes=12)
        m = min(ns, len(r["flow_deriv"]))
        du[a * hop:a * hop + m] = r["flow_deriv"][:m]
        area[a * hop:a * hop + m] = r["area"][:m]
    # 무성 구간은 0 으로 남는다 (잡음 가지가 따로 담당)
    if np.abs(du).max() > 0:
        du = du / np.abs(du).max()
    return du, area


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fs", type=float, default=48000.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="주기 길이 변동 (비율). 이 화자 실측 0.025 — 고역 배음만 흩는 물리적 기제")
    ap.add_argument("--shimmer", type=float, default=0.0)
    ap.add_argument("--mode-var", type=float, default=0.0)
    ap.add_argument("--source", choices=("physics", "lf"), default="physics",
                    help="lf = 같은 경로의 대조군 (엔진 LF 펄스, Rd 2.16)")
    a = ap.parse_args()

    z = np.load(a.stem + "_track.npz", allow_pickle=True)
    names = [str(x) for x in z["names"]]
    vals = np.asarray(z["values"], float)
    frame_ms = float(z["frame_ms"]) if "frame_ms" in z else 1.0
    hop = int(round(a.fs * frame_ms / 1000.0))

    if a.source == "lf":
        du, area = build_lf_source(vals, names, a.fs, hop, jitter=a.jitter,
                                   shimmer=a.shimmer, seed=a.seed)
    else:
        du, area = build_source(vals, names, a.fs, hop, seed=a.seed, jitter=a.jitter,
                                shimmer=a.shimmer, mode_var=a.mode_var)
    n = len(du)
    print(f"물리 음원 {n/a.fs:.3f} s, 유량미분 rms {du.std():.4f}, "
          f"성문 개방률 {float(np.mean(area > 0)):.3f}", flush=True)

    tr = VocalTract(a.fs, hop).double()
    T = vals.shape[0]
    ten = torch.tensor(vals, dtype=torch.float64).unsqueeze(0)
    c = {nm: ten[..., names.index(nm)] for nm in PARAM_NAMES if nm in names}
    x = torch.tensor(du, dtype=torch.float64).unsqueeze(0)
    z0 = torch.zeros_like(x)
    with torch.no_grad():
        y = tr(x, z0, z0, z0, c)["audio"][0].numpy()
    y = y / max(np.abs(y).max(), 1e-12) * 0.5
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sf.write(a.out + "_fit.wav", y, int(a.fs))
    tgt = a.stem + "_target.wav"
    if os.path.exists(tgt):
        t, sr = sf.read(tgt)
        sf.write(a.out + "_target.wav", t, sr)
    print(f"결과: {a.out}_fit.wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
