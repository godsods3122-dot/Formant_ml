"""손실 항 줄다리기 — 저장된 판의 해에서 **어느 항이 무엇을 막는가** (MEASUREMENTS §52.472).

    python scripts/diag/tug.py out/C/W17a --win 0.60:0.70:2000:4000 --win 0.78:0.84:4000:6000 ...

`--init <판> --no-fit` 으로 해를 그대로 되살린 뒤, 한 번의 손실 평가에서 항마다 두 가지 기울기를 잰다.

1. **대역 탐침** ε: 합성 y 에 `ε · 대역통과(y) · 창` 을 더한다(ε=0 이 지금 해). 항 k 의 `−w_k ∂T_k/∂ε` 가 양이면 그 항은
   그 창·대역에 **지금 위상 그대로의 에너지를 더 원하고**, 음이면 **막는다**. 포락이 원하는데 누가 막는지가 줄다리기다.
2. **손잡이** u: 창 안 프레임의 제어(비구속 u)에 대한 `−w_k ∂T_k/∂u` 합 — 경사하강이 그 항 때문에 손잡이를 미는 방향.

가중 w_k 는 실제 적합과 같게: 균형 가중은 판의 로그(`균형 가중:` 줄)에서, 나머지는 `loss()` 가 넘기는 기본값을 가로챈다.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import runpy
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
KNOBS = ("voice_gain", "tilt", "rd_offset", "p_sub", "adduction", "f2", "bw2", "f3", "bw3", "f4", "bw4")


_KNOBS = None


class _RecBal(dict):
    """`self._bal.get(name, base)` 가 넘기는 기본 가중을 가로챈다."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.eff = {}

    def get(self, k, d=None):
        v = super().get(k, d)
        self.eff[k] = v
        return v


def _log_bal(stem):
    bal = {}
    try:
        for ln in open(stem + ".log", encoding="utf-8", errors="ignore"):
            if "균형 가중:" in ln:
                for k, v in re.findall(r"(\w+) ([-+0-9.e]+)", ln.split("균형 가중:")[1]):
                    bal.setdefault(k, float(v))
    except OSError:
        pass
    return bal


def diagnose(fit, stem, wins):
    import torch
    from formant_ml.engine import fit as F
    sr = fit.fs
    bal = _RecBal(_log_bal(stem))
    fit._bal = bal
    u_leaf = fit._u().detach().clone().requires_grad_(True)
    fit._u = lambda: u_leaf
    eps = [torch.zeros((), dtype=u_leaf.dtype, requires_grad=True) for _ in wins]
    orig = type(fit).synth

    def synth(self, *a, **k):
        out = orig(self, *a, **k)
        y = out[0] if isinstance(out, tuple) else out
        n = y.shape[-1]
        Y = torch.fft.rfft(y, dim=-1)
        f = torch.fft.rfftfreq(n, 1.0 / sr).to(y.device)
        t = torch.arange(n, device=y.device, dtype=y.dtype) / sr
        add = torch.zeros_like(y)
        for e, (t0, t1, lo, hi) in zip(eps, wins):
            bp = torch.fft.irfft(Y * ((f >= lo) & (f < hi)).to(Y.dtype), n=n, dim=-1)
            # 가장자리를 5 ms 올림 코사인으로 깎는다 — 직사각 창은 경계에서 광대역으로 번져 고역 항(hfcomb 등)이
            # 1~2 kHz 탐침에도 반응하는 가짜 줄다리기를 만든다 (처음 판에서 실제로 그랬다).
            r = 0.005
            a_ = ((t - t0) / r).clamp(0.0, 1.0)
            b_ = ((t1 - t) / r).clamp(0.0, 1.0)
            w = (0.5 - 0.5 * torch.cos(math.pi * a_)) * (0.5 - 0.5 * torch.cos(math.pi * b_))
            add = add + e * bp * w
        y = y + add
        return (y,) + tuple(out[1:]) if isinstance(out, tuple) else y
    fit.synth = synth.__get__(fit)
    fit._collect = True
    try:
        total, *_ = fit.loss()
        T = dict(fit._terms)
    finally:
        fit._collect = False
    names = fit.names
    kn = [p for p in (_KNOBS or KNOBS) if p in names]
    ki = [names.index(p) for p in kn]
    fm = fit.track.frame_ms * 1e-3
    rows = {}
    for k, term in list(T.items()) + [("(벌점·기타)", None)]:
        if k == "(벌점·기타)":
            rest = total - sum(bal.eff.get(q, 1.0 if q == "env" else 0.0) * T[q] for q in T
                               if torch.is_tensor(T[q]))
            term, w = rest, 1.0
        else:
            if not (torch.is_tensor(term) and term.requires_grad):
                continue
            w = 1.0 if k == "env" else float(bal.eff.get(k, 0.0))
            if k == "phase":
                w = float(bal.eff.get("phase", getattr(fit, "phase_weight", 0.0)))
        if w == 0.0:
            continue
        g = torch.autograd.grad(term, [u_leaf] + eps, retain_graph=True, allow_unused=True)
        gu = g[0]
        ge = [0.0 if x is None else -w * float(x) for x in g[1:]]
        gk = []
        for (t0, t1, _, _) in wins:
            a, b = int(t0 / fm), int(t1 / fm)
            gk.append([0.0 if gu is None else -w * float(gu[a:b, j].sum()) for j in ki])
        rows[k] = (w, ge, gk)
    return rows, kn


def report(rows, kn, wins):
    order = sorted(rows, key=lambda k: -max(abs(v) for v in rows[k][1]))
    for i, (t0, t1, lo, hi) in enumerate(wins):
        print(f"\n=== 창 {t0:.2f}–{t1:.2f} s, 대역 {lo / 1000:g}–{hi / 1000:g} kHz ===")
        env = rows["env"][1][i] if "env" in rows else 0.0
        print(f"  대역 탐침 −w·∂T/∂ε (양 = 더 원함, 음 = 막음).  포락 {env:+.3g}")
        tot = sum(r[1][i] for r in rows.values())
        for k in sorted(rows, key=lambda k: rows[k][1][i]):
            v = rows[k][1][i]
            if abs(v) < 0.02 * max(abs(env), 1e-12):
                continue
            print(f"    {k:12s} w={rows[k][0]:<8.3g} {v:+10.3g}  ({v / (abs(env) + 1e-30):+6.2f} × |포락|)")
        print(f"    {'합':12s} {'':10s} {tot:+10.3g}")
        print("  손잡이 −w·Σ∂T/∂u (창 안 프레임 합) — 포락 대비 큰 항만")
        print("    " + f"{'':12s}" + "".join(f"{p[:8]:>9s}" for p in kn))
        e = np.array(rows["env"][2][i]) if "env" in rows else np.zeros(len(kn))
        big = []
        for k in rows:
            g = np.array(rows[k][2][i])
            if k == "env" or np.any(np.abs(g) > 0.3 * np.abs(e) + 1e-12) and np.any(np.abs(g) > 1e-9):
                big.append(k)
        for k in ["env"] + [k for k in big if k != "env"]:
            g = np.array(rows[k][2][i])
            print("    " + f"{k:12s}" + "".join(f"{v:+9.2g}" for v in g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--win", action="append", required=True, help="t0:t1:lo_hz:hi_hz")
    ap.add_argument("--knobs", default=None, help="쉼표로 가른 손잡이 (기본: 음원·F2~F4)")
    a = ap.parse_args()
    global _KNOBS
    if a.knobs:
        _KNOBS = tuple(k.strip() for k in a.knobs.split(",") if k.strip())
    wins = [tuple(float(v) for v in w.split(":")) for w in a.win]
    sys.path.insert(0, os.path.join(ROOT, "src"))
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from formant_ml.engine import fit as F
    orig_fit = F.CopySynthFitter.fit

    def fit_then_tug(self, iters=200, *x, **k):
        rep = orig_fit(self, iters, *x, **k)
        if iters == 0:
            rows, kn = diagnose(self, a.stem, wins)
            report(rows, kn, wins)
            sys.stdout.flush()
            os._exit(0)
        return rep
    F.CopySynthFitter.fit = fit_then_tug
    args = [l.strip() for l in open(a.stem + ".argv", encoding="utf-8").read().split("\n") if l.strip()]
    args = args[:-2]
    if "--init" in args:
        args[args.index("--init") + 1] = a.stem
    else:
        args += ["--init", a.stem]
    tmp = os.path.join(ROOT, "out", "_tmp", "tug")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    sys.argv = ["scripts/copyfit.py"] + args + ["--no-fit", "--out", tmp]
    runpy.run_path(os.path.join(ROOT, "scripts", "copyfit.py"), run_name="__main__")


if __name__ == "__main__":
    main()
