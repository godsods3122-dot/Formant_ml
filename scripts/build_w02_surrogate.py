"""W02 (VocalTractLab 여성 화자) 기하의 미분 가능한 대리 모형을 만든다 (MEASUREMENTS §52.486).

1. 자료: VTL 로 (해부학 변수 19 → 면적 50 칸) 을 N 개 뽑는다 — 절반은 변수 범위 전체에서 고르게, 절반은 화자 파일의 MRI 음소 자세
   주변에서 (가우시안, 범위의 10 %). 연구개는 대리 모형의 입력에서 뺀다 (구강 면적에 거의 안 들어가고, 포트 면적은 따로 낸다).
2. 균일 28 칸 로그 면적 + 성도 길이 + 연구개 열림으로 바꾼다 (`articulation.uniform_tube`, 부피 보존).
3. MLP (정규화 변수 → 30 출력) 를 학습하고, 떼어 둔 자세에서 VTL 원본과 면적·포먼트 오차를 잰다.

사용: python scripts/build_w02_surrogate.py [--n 200000] [--workers 8] [--epochs 60]
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "out" / "_tmp" / "w02_data.npz"
MODEL = ROOT / "profiles" / "w02_surrogate.pt"
N_CELLS = 28


def _worker(args):
    seed, n, mode = args
    from formant_ml.engine.vtl_api import VTL
    v = VTL()
    rng = np.random.default_rng(seed)
    span = v.hi - v.lo
    if mode == "uniform":
        P = rng.uniform(v.lo, v.hi, (n, v.n_par))
    else:                                       # 음소 자세 주변 — 흔들림 5 / 10 / 20 % 섞기, 두 자세 사이 섞기도
        S = np.stack([v.shape(s) for s in v.shape_names()])
        i, j = rng.integers(0, len(S), n), rng.integers(0, len(S), n)
        w = rng.uniform(0, 1, (n, 1)) * (rng.uniform(0, 1, (n, 1)) < 0.5)
        sd = rng.choice([0.05, 0.10, 0.20], (n, 1))
        P = (1 - w) * S[i] + w * S[j] + rng.normal(0, 1.0, (n, v.n_par)) * sd * span
        P = np.clip(P, v.lo, v.hi)
    L = np.zeros((n, v.n_tube)); A = np.zeros((n, v.n_tube)); vel = np.zeros(n); inc = np.zeros(n)
    for i, p in enumerate(P):
        L[i], A[i], _, inc[i], _, vel[i] = v.tube(p)
    return P, L, A, vel, inc


def make_data(n: int, workers: int) -> None:
    from formant_ml.engine.vtl_api import VTL
    v = VTL()
    chunk = 5000
    jobs = [(k, chunk, "uniform" if k % 4 == 0 else "shapes") for k in range(max(4, n // chunk))]
    t = time.time()
    with Pool(workers) as pool:
        out = pool.map(_worker, jobs)
    P, L, A, vel, inc = (np.concatenate([o[k] for o in out]) for k in range(5))
    np.savez(DATA, P=P, L=L, A=A, vel=vel, inc=inc, names=np.array(v.names), lo=v.lo, hi=v.hi, neutral=v.neutral)
    print(f"자료 {len(P)} 자세, {time.time() - t:.0f} s → {DATA}", flush=True)


def train(epochs: int) -> None:
    import torch
    from formant_ml.engine.articulation import uniform_tube, W02Surrogate, AREA_EPS
    z = np.load(DATA)
    P, L, A, vel = z["P"], z["L"], z["A"], z["vel"]
    with torch.no_grad():
        Au, Lt = uniform_tube(torch.as_tensor(A).clamp_min(1e-4), torch.as_tensor(L).clamp_min(1e-3), N_CELLS)
    Y = np.concatenate([np.log(Au.numpy() + AREA_EPS), np.log(Lt.numpy())[:, None], np.log(vel + 1e-4)[:, None]], 1)
    net = W02Surrogate(lo=z["lo"], hi=z["hi"], names=list(z["names"]))
    X = torch.as_tensor(P, dtype=torch.float32)
    Yt = torch.as_tensor(Y, dtype=torch.float32)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X)); nv = len(X) // 20
    va, tr = idx[:nv], idx[nv:]
    opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bs = 1024
    for ep in range(epochs):
        net.train()
        perm = torch.as_tensor(rng.permutation(tr))
        tot = 0.0
        for b in range(0, len(perm), bs):
            j = perm[b:b + bs]
            pred = net.raw(X[j])
            d2 = (pred - Yt[j]) ** 2
            loss = d2[:, :N_CELLS].mean() + 20.0 * d2[:, N_CELLS].mean() + d2[:, N_CELLS + 1].mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(j)
        sched.step()
        if ep % 5 == 4 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                e = (net.raw(X[va]) - Yt[va]).abs()
            print(f"  세대 {ep + 1:3d}  학습 {tot / len(tr):.5f}  검증 |Δlog 면적| 중앙 {e[:, :N_CELLS].median():.4f} 99% "
                  f"{torch.quantile(e[:, :N_CELLS].flatten()[:200000], 0.99):.3f}  |Δlog 길이| 중앙 {e[:, N_CELLS].median():.4f}", flush=True)
    torch.save({"state": net.state_dict(), "lo": z["lo"], "hi": z["hi"], "names": list(z["names"])}, MODEL)
    print(f"→ {MODEL}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--skip-data", action="store_true")
    a = ap.parse_args()
    if not a.skip_data:
        make_data(a.n, a.workers)
    train(a.epochs)
