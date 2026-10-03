"""W02 (해부 적응) 의 **입술–앞니 거리** 예측기 (MEASUREMENTS §52.519).

협착 난류의 음원 모양은 제트가 나오는 자리와 앞니 사이 거리가 정한다 (VTL `TdsModel::calcNoiseSources`: 앞니 장애물 · 벽 · 양순). 예전 규칙
(`tube_td.NOISE_PLACE`, §52.503)은 앞니를 "입술 끝에서 1 cm, 명목 길이 15 cm" 로 고정해, 앞니 바로 뒤에서 나오는 무성화 '시' 의 제트를 양순 파열
(500 Hz)로 분류했다. VTL 기하에서 입술–앞니 거리는 자세마다 0.26 (E) – 1.27 cm (후치조 마찰, 입술 내밂) 로 한 칸 넘게 움직인다.

자료: `build_w02_anat_surrogate.py` 의 해부 ±35 % 조각 (`out/_tmp/w02_anat_chunks35`, 앞니 자리 `inc` [cm, 성문에서] 포함).
MLP (조음 19 + 해부 13 → 입술–앞니 거리 [cm] = 성도 길이 − inc). 입력은 대리 모형과 같은 정규화 ([−1, 1]).

사용: python scripts/build_w02_teeth.py [--epochs 40]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CHUNKS = ROOT / "out" / "_tmp" / "w02_anat_chunks35"
MODEL = ROOT / "profiles" / "w02_teeth.pt"


def load():
    fs = sorted(glob.glob(str(CHUNKS / "chunk_*.npz")))
    zs = [np.load(f) for f in fs]
    P = np.concatenate([z["P"] for z in zs]); X = np.concatenate([z["X"] for z in zs])
    D = np.concatenate([z["L"].sum(1) - z["inc"] for z in zs])
    return P, X, D


def main() -> None:
    import torch
    from formant_ml.engine.articulation import W02TeethNet
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    P, X, D = load()
    from formant_ml.engine.vtl_api import VTL, ANAT_API_DIR
    v = VTL(api_dir=ANAT_API_DIR)
    lo, hi = v.lo.copy(), v.hi.copy()
    alo, ahi = X.min(0), X.max(0)
    net = W02TeethNet(lo, hi, v.names, alo, ahi)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(dev)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(D)); nv = len(D) // 20
    va, tr = idx[:nv], idx[nv:]
    t = lambda z: torch.as_tensor(z, dtype=torch.float32, device=dev)
    Pt, Xt, Dt = t(P), t(X), t(D)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    bs = 4096
    for ep in range(a.epochs):
        net.train()
        perm = torch.as_tensor(rng.permutation(tr), device=dev)
        for k in range(0, len(perm), bs):
            b = perm[k:k + bs]
            loss = torch.nn.functional.smooth_l1_loss(net(Pt[b], Xt[b]), Dt[b], beta=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if ep % 5 == 4 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                e = (net(Pt[va], Xt[va]) - Dt[va]).abs().cpu().numpy()
            print(f"  {ep + 1:3d}  검증 |오차| 중앙 {np.median(e):.3f} · 95 % {np.percentile(e, 95):.3f} · 최대 {e.max():.3f} cm", flush=True)
    torch.save(net.cpu().pack(), MODEL)
    print(f"→ {MODEL}")
    # MRI 음소 자세 (W02 해부 그대로) 와 VTL 원본
    net.eval()
    nm, alo_, ahi_, x0 = v.anatomy_info()
    worst = 0.0
    for s in v.shape_names():
        p = v.shape(s)
        L, A_, art, inc, tts, vel = v.tube(p)
        with torch.no_grad():
            d = float(net(torch.as_tensor(p[None], dtype=torch.float32), torch.as_tensor(x0[None], dtype=torch.float32))[0])
        worst = max(worst, abs(d - (L.sum() - inc)))
    print(f"  MRI 자세 {len(v.shape_names())} 개 |오차| 최대 {worst:.3f} cm")


if __name__ == "__main__":
    main()
