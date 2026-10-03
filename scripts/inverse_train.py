"""소리 → 기관 매개 역모형 학습 (MEASUREMENTS §52.426).

`scripts/inverse_data.py` 가 만든 참값 있는 합성 음성으로 학습하고, 원본 파일 단위로 떼어 둔 검증 묶음에서
(1) 참값 상관, (2) 5 ms 움직임 분포(분위)가 참값과 맞는지 본다. 맞아야 코퍼스 실녹음에 적용해 움직임 통계를 잰다.

    python scripts/inverse_train.py --data out/inv --out out/inv_model.pt --epochs 30
    python scripts/inverse_train.py --apply out/inv_model.pt --wavs 'wavs/*.wav' --out out/corpus_inv
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

FS = 48000
HOP = 240                      # 5 ms
NFFT = 1024
NMEL = 64
TARGETS = ("adduction", "a_c", "p_sub", "velum", "rd_offset")
LOGT = {"a_c"}


def mel_fb(fmin=50.0, fmax=20000.0):
    """삼각 멜 필터뱅크 (NMEL, NFFT/2+1) — HTK 멜 눈금."""
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    f = np.fft.rfftfreq(NFFT, 1.0 / FS)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), NMEL + 2))
    fb = np.zeros((NMEL, f.size))
    for i in range(NMEL):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)), 0.0, None)
    return torch.tensor(fb, dtype=torch.float32)


_FB = None


def feats(y: np.ndarray) -> torch.Tensor:
    """(T5, NMEL+1) — 로그 멜 + 로그 에너지, 파일 최대로 정규화 (수준에 무관하게)."""
    global _FB
    if _FB is None:
        _FB = mel_fb()
    x = torch.as_tensor(y, dtype=torch.float32)
    S = torch.stft(x, NFFT, HOP, window=torch.hann_window(NFFT), return_complex=True).abs() ** 2
    M = torch.log10(_FB @ S + 1e-10).T
    M = M - M.max()
    e = torch.log10(S.sum(0) + 1e-10)
    e = (e - e.max()).unsqueeze(1)
    return torch.cat([M.clamp_min(-8), e.clamp_min(-8)], 1)


def targ5(tg: np.ndarray, n5: int) -> np.ndarray:
    t = tg.copy()
    for j, nm in enumerate(TARGETS):
        if nm in LOGT:
            t[:, j] = np.log(np.maximum(t[:, j], 1e-3))
    m = (t.shape[0] // 5) * 5
    t5 = t[:m].reshape(-1, 5, t.shape[1]).mean(1)
    if t5.shape[0] < n5:
        t5 = np.pad(t5, ((0, n5 - t5.shape[0]), (0, 0)), mode="edge")
    return t5[:n5]


class Net(nn.Module):
    def __init__(self, nin=NMEL + 1, nout=len(TARGETS), h=128):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(nin, h, 5, padding=2), nn.GELU(),
                                 nn.Conv1d(h, h, 5, padding=2), nn.GELU(),
                                 nn.Conv1d(h, h, 3, padding=1), nn.GELU())
        self.rnn = nn.GRU(h, h, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * h, nout)

    def forward(self, x):                                   # x (B, T, F)
        z = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        z, _ = self.rnn(z)
        return self.head(z)


def load_set(files):
    X, Y = [], []
    for f in files:
        d = np.load(f)
        x = feats(d["audio"])
        y = targ5(d["targets"], x.shape[0])
        X.append(x); Y.append(torch.as_tensor(y, dtype=torch.float32))
    return X, Y


def train(a):
    files = sorted(glob.glob(os.path.join(a.data, "*.npz")))
    srcs = sorted({os.path.basename(f).rsplit("_", 1)[0] for f in files})
    rng = np.random.default_rng(0)
    held = set(rng.choice(srcs, size=max(1, len(srcs) // 11), replace=False))
    tr = [f for f in files if os.path.basename(f).rsplit("_", 1)[0] not in held]
    va = [f for f in files if os.path.basename(f).rsplit("_", 1)[0] in held]
    print(f"학습 {len(tr)} / 검증 {len(va)} (원본 {len(held)} 개 떼어 둠)", flush=True)
    Xt, Yt = load_set(tr); Xv, Yv = load_set(va)
    Ycat = torch.cat(Yt, 0)
    mu, sd = Ycat.mean(0), Ycat.std(0) + 1e-6
    torch.manual_seed(0)
    net = Net(h=a.hidden)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    L = 400                                                   # 2 s 조각
    for ep in range(a.epochs):
        net.train(); perm = rng.permutation(len(Xt)); tot = 0.0
        for i in range(0, len(perm), 16):
            xb, yb = [], []
            for j in perm[i:i + 16]:
                x, y = Xt[j], Yt[j]
                s = int(rng.integers(0, max(1, x.shape[0] - L)))
                xb.append(x[s:s + L]); yb.append((y[s:s + L] - mu) / sd)
            m = min(t.shape[0] for t in xb)
            xb = torch.stack([t[:m] for t in xb]); yb = torch.stack([t[:m] for t in yb])
            loss = ((net(xb) - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            tot += float(loss.detach())
        sched.step()
        if ep % 5 == 4 or ep == a.epochs - 1:
            print(f"  에폭 {ep + 1}: 학습 {tot / max(1, len(perm) // 16):.3f}", flush=True)
    torch.save({"net": net.state_dict(), "mu": mu, "sd": sd, "hidden": a.hidden}, a.out)
    evaluate(net, mu, sd, Xv, Yv)


@torch.no_grad()
def evaluate(net, mu, sd, Xv, Yv):
    net.eval()
    P = [net(x[None])[0] * sd + mu for x in Xv]
    p, y = torch.cat(P).numpy(), torch.cat(Yv).numpy()
    print("검증 (떼어 둔 원본):  상관 | 5 ms 차분 |값| 분위 50/90/99  참 → 추정")
    for j, nm in enumerate(TARGETS):
        r = np.corrcoef(p[:, j], y[:, j])[0, 1]
        dy = np.concatenate([np.abs(np.diff(t[:, j].numpy())) for t in Yv])
        dp = np.concatenate([np.abs(np.diff(t[:, j].numpy())) for t in P])
        qy, qp = np.percentile(dy, [50, 90, 99]), np.percentile(dp, [50, 90, 99])
        print(f"  {nm:10s} r {r:+.3f} | {qy[0]:.4f} {qy[1]:.4f} {qy[2]:.4f} → {qp[0]:.4f} {qp[1]:.4f} {qp[2]:.4f}")


@torch.no_grad()
def apply(a):
    import soundfile as sf
    ck = torch.load(a.apply)
    net = Net(h=ck.get("hidden", 128)); net.load_state_dict(ck["net"]); net.eval()
    os.makedirs(a.out, exist_ok=True)
    for f in sorted(glob.glob(a.wavs)):
        y, sr = sf.read(f)
        y = np.asarray(y, float)
        if y.ndim > 1:
            y = y.mean(1)
        p = (net(feats(y)[None])[0] * ck["sd"] + ck["mu"]).numpy()
        np.save(os.path.join(a.out, os.path.basename(f)[:-4] + ".npy"), p)
    print("적용 끝", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/inv")
    ap.add_argument("--out", default="out/inv_model.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--apply", default=None)
    ap.add_argument("--wavs", default="wavs/*.wav")
    a = ap.parse_args()
    apply(a) if a.apply else train(a)
