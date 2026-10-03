"""해부 적응까지 받는 W02 기하 대리 모형 (MEASUREMENTS §52.501).

`build_w02_surrogate.py` 는 VTL 여성 화자 W02 의 해부에 묶인 대리 모형을 만든다. 이 화자의 성도를 W02 해부로 역산하면 후두 칸이 3–4 배 어긋났다
(§52.499). VTL 의 해부 적응(`AnatomyParams::setFor`, GUI 해부 대화상자와 같은 절차 — 기준 화자 W02 에 대한 변형; 입술 폭·하악·어금니 높이·
구개 높이·깊이·경·연구개 길이·인두 길이·후두 길이·폭·성대 길이·구강–인두 각)을 API 에 노출해 다시 빌드했다 (`third_party/VTL2.4_anat`).

1. 자료: 해부 벡터를 W02 값의 ±`SPREAD` 안에서 뽑고 (각은 −105 ~ −90°, 25 % 는 W02 그대로), 해부마다 조음 `PER_ANAT` 개 — 절반은 **그 해부로 옮긴**
   변수 범위 전체에서 고르게, 절반은 **그 해부로 옮긴** MRI 음소 자세 주변. 조음은 그 해부의 범위로 정규화한 위치를 W02 범위로 다시 펴서 저장한다
   (같은 조음 위치 → 해부마다 다른 모양).
2. 균일 28 칸 로그 면적 + 성도 길이 + 연구개 열림 (`articulation.uniform_tube`).
3. MLP (조음 19 + 해부 13 → 30) — 떼어 둔 자세에서 VTL 원본과 잰다.

사용: python scripts/build_w02_anat_surrogate.py [--n 600000] [--workers 6] [--epochs 100]
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

DATA = ROOT / "out" / "_tmp" / "w02_anat_data.npz"
MODEL = ROOT / "profiles" / "w02_anat_surrogate.pt"
N_CELLS = 28
import os
SPREAD = float(os.environ.get("ANAT_SPREAD", "0.20"))       # 해부 뽑기 폭 (W02 값 대비) — VTL 해부 한계 안으로 자른다
PER_ANAT = 40


def anat_range():
    from formant_ml.engine.vtl_api import VTL, ANAT_API_DIR
    v = VTL(api_dir=ANAT_API_DIR)
    nm, lo, hi, x0 = v.anatomy_info()
    alo = np.maximum(lo, x0 * (1 - SPREAD))
    ahi = np.minimum(hi, x0 * (1 + SPREAD))
    alo[-1], ahi[-1] = -105.0, -90.0                    # 구강–인두 각 [도]
    return nm, alo, ahi, x0, v.lo.copy(), v.hi.copy(), v.names


def _worker(args):
    seed, n_anat = args
    from formant_ml.engine.vtl_api import VTL, ANAT_API_DIR
    v = VTL(api_dir=ANAT_API_DIR)
    nm, alo, ahi, x0, lo0, hi0, names = anat_range()
    rng = np.random.default_rng(seed)
    shapes = v.shape_names()
    out_P, out_X, out_L, out_A, out_vel = [], [], [], [], []
    out_inc, out_tts, out_art = [], [], []          # 앞니 자리 [cm, 성문에서], 혀끝 옆 높이, 칸별 조음기 (난류 음원 자리, §52.503)
    for k in range(n_anat):
        x = x0.copy() if rng.uniform() < 0.25 else rng.uniform(alo, ahi)
        v.set_anatomy(x)
        lo_a, hi_a = v.lo.copy(), v.hi.copy()
        span = np.maximum(hi_a - lo_a, 1e-9)
        S = np.stack([v.shape(s) for s in shapes])
        P = np.empty((PER_ANAT, v.n_par))
        h = PER_ANAT // 2
        P[:h] = rng.uniform(lo_a, hi_a, (h, v.n_par))
        i, j = rng.integers(0, len(S), PER_ANAT - h), rng.integers(0, len(S), PER_ANAT - h)
        w = rng.uniform(0, 1, (PER_ANAT - h, 1)) * (rng.uniform(0, 1, (PER_ANAT - h, 1)) < 0.5)
        sd = rng.choice([0.05, 0.10, 0.20], (PER_ANAT - h, 1))
        P[h:] = np.clip((1 - w) * S[i] + w * S[j] + rng.normal(0, 1.0, (PER_ANAT - h, v.n_par)) * sd * span, lo_a, hi_a)
        for p in P:
            L, A, art, inc, tts, vel = v.tube(p)
            out_L.append(L); out_A.append(A); out_vel.append(vel)
            out_inc.append(inc); out_tts.append(tts); out_art.append(art.astype(np.int8))
        out_P.append(lo0 + (P - lo_a) / span * (hi0 - lo0))    # 그 해부의 범위로 정규화한 위치 → W02 단위
        out_X.append(np.repeat(x[None], PER_ANAT, 0))
    v.set_anatomy(x0)
    return (np.concatenate(out_P), np.concatenate(out_X), np.array(out_L), np.array(out_A), np.array(out_vel),
            np.array(out_inc), np.array(out_tts), np.array(out_art))


def make_data(n: int, workers: int) -> None:
    nm, alo, ahi, x0, lo0, hi0, names = anat_range()
    n_anat = max(1, n // PER_ANAT)
    per = 25
    jobs = [(1000 + k, per) for k in range(max(1, n_anat // per))]
    t = time.time()
    with Pool(workers) as pool:
        out = pool.map(_worker, jobs)
    P, X, L, A, vel = (np.concatenate([o[k] for o in out]) for k in range(5))       # (옛 경로 — 조각 방식을 쓴다)
    np.savez(DATA, P=P, X=X, L=L, A=A, vel=vel, names=np.array(names), lo=lo0, hi=hi0,
             anames=np.array(nm), alo=alo, ahi=ahi, ax0=x0)
    print(f"자료 {len(P)} 자세 · 해부 {len(P) // PER_ANAT} 개, {time.time() - t:.0f} s → {DATA}", flush=True)


CHUNKS = ROOT / "out" / "_tmp" / os.environ.get("ANAT_CHUNKS", "w02_anat_chunks")
SEED0 = int(os.environ.get("ANAT_SEED0", "1000"))


def make_chunk(k: int, n_anat: int) -> None:
    """조각 하나를 독립 프로세스로 (Windows 의 multiprocessing Pool 이 기동에서 멈췄다 — 작업자 다섯이 27 MB·CPU 0.2 s 로 한 시간)."""
    CHUNKS.mkdir(parents=True, exist_ok=True)
    t = time.time()
    P, X, L, A, vel, inc, tts, art = _worker((SEED0 + k, n_anat))
    np.savez(CHUNKS / f"chunk_{k:03d}.npz", P=P, X=X, L=L, A=A, vel=vel, inc=inc, tts=tts, art=art)
    print(f"조각 {k}: {len(P)} 자세, {time.time() - t:.0f} s", flush=True)


def merge() -> None:
    nm, alo, ahi, x0, lo0, hi0, names = anat_range()
    dirs = [ROOT / "out" / "_tmp" / d for d in os.environ.get("ANAT_MERGE", CHUNKS.name).split(",")]
    fs = sorted(f for d in dirs for f in d.glob("chunk_*.npz"))
    zs = [np.load(f) for f in fs]
    P, X, L, A, vel = (np.concatenate([z[k] for z in zs]) for k in ("P", "X", "L", "A", "vel"))
    np.savez(DATA, P=P, X=X, L=L, A=A, vel=vel, names=np.array(names), lo=lo0, hi=hi0,
             anames=np.array(nm), alo=alo, ahi=ahi, ax0=x0)
    print(f"합침: 조각 {len(fs)}, 자세 {len(P)} → {DATA}", flush=True)


def train(epochs: int, width: int) -> None:
    import torch
    from formant_ml.engine.articulation import uniform_tube, W02AnatSurrogate, AREA_EPS
    z = np.load(DATA)
    P, X, L, A, vel = z["P"], z["X"], z["L"], z["A"], z["vel"]
    with torch.no_grad():
        Au, Lt = uniform_tube(torch.as_tensor(A).clamp_min(1e-4), torch.as_tensor(L).clamp_min(1e-3), N_CELLS)
    Y = np.concatenate([np.log(Au.numpy() + AREA_EPS), np.log(Lt.numpy())[:, None], np.log(vel + 1e-4)[:, None]], 1)
    net = W02AnatSurrogate(lo=z["lo"], hi=z["hi"], names=list(z["names"]), alo=z["alo"], ahi=z["ahi"], anames=list(z["anames"]),
                           ax0=z["ax0"], width=width)
    Pt = torch.as_tensor(P, dtype=torch.float32)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    Yt = torch.as_tensor(Y, dtype=torch.float32)
    rng = np.random.default_rng(0)
    # 검증은 **해부 단위로** 떼어 둔다 (학습에 없던 해부에서 잰다)
    na = len(P) // PER_ANAT
    ia = rng.permutation(na)
    va_a = ia[:max(1, na // 20)]
    va = (va_a[:, None] * PER_ANAT + np.arange(PER_ANAT)[None]).ravel()
    trm = np.ones(len(P), bool); trm[va] = False
    tr = np.flatnonzero(trm)
    opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bs = 2048
    for ep in range(epochs):
        net.train()
        perm = torch.as_tensor(rng.permutation(tr))
        tot = 0.0
        for b in range(0, len(perm), bs):
            j = perm[b:b + bs]
            pred = net.raw(Pt[j], Xt[j])
            d2 = (pred - Yt[j]) ** 2
            loss = d2[:, :N_CELLS].mean() + 20.0 * d2[:, N_CELLS].mean() + d2[:, N_CELLS + 1].mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(j)
        sched.step()
        if ep % 5 == 4 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                e = (net.raw(Pt[va], Xt[va]) - Yt[va]).abs()
            print(f"  세대 {ep + 1:3d}  학습 {tot / len(tr):.5f}  검증(새 해부) |Δlog 면적| 중앙 {e[:, :N_CELLS].median():.4f} 99% "
                  f"{torch.quantile(e[:, :N_CELLS].flatten()[:200000], 0.99):.3f}  |Δlog 길이| 중앙 {e[:, N_CELLS].median():.4f}", flush=True)
            torch.save(net.pack(), MODEL)
    torch.save(net.pack(), MODEL)
    print(f"→ {MODEL}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--chunk", type=int, default=None, help="조각 번호 — 이 조각만 만들고 끝낸다")
    ap.add_argument("--chunk-anat", type=int, default=300, help="조각 하나의 해부 수")
    ap.add_argument("--merge", action="store_true", help="조각을 합치고 학습한다")
    a = ap.parse_args()
    if a.chunk is not None:
        make_chunk(a.chunk, a.chunk_anat)
        sys.exit(0)
    if a.merge:
        merge()
    elif not a.skip_data:
        make_data(a.n, a.workers)
    train(a.epochs, a.width)
