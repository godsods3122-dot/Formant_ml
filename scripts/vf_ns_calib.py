"""모드 3 (보–막 점막) 성대의 **화자 보정** (MEASUREMENTS §52.530) — 판의 화자 상수 그대로 모형 자신에게서 잰다.

1. **f0(ε) 표** — 목표 f0 → 늘어남 ε 는 `voice_td.vf_ns_eps` 가 이 표를 뒤집어 쓴다. 예전 표(`VF_NS_F0_TAB`)는 덮개 결합 ×3 · 쉼 틈 0.045 mm 에서
   잰 것이라, 생리 범위로 자른 상수(덮개 결합 ≤ ×1.5, `--vf-ns-hard`)에서는 같은 ε 의 실제 음높이가 0.6 배였고(ε 0 에서 208 → 123 Hz) ε 0.3 위에서는
   폐압 7 cmH2O 로 떨지 않았다 — KWF_n3 의 음높이 붕괴(90 % 오차 1236 cents). ε 마다 폐압을 0 → `--ps-max` 로 천천히 올려 문턱 PTP 를 잡고,
   폐압 clip(1.3·PTP, 5, ps_max) 에 두어 성문 유량의 자기상관 첫 주기(최대의 85 % 를 넘는 첫 봉우리 — 이중 주기에서 두 배 주기를 잡지 않게)로 f0.
   표는 f0 가 늘어나는 점만 남긴다 (PCHIP 역함수가 단조를 요구).
2. **발성 문턱 지도** — 새 표로 f0 → ε 를 옮긴 격자 (f0 × 내전 × 수렴 × 갑상피열근) 의 PTP. `fit.pth_loss` (`--pth-map`) 가 읽는다.

대표 자세: 모음 틀 하나의 성도 (`--dump` 관 입력, `--t` 초), 내전 `--add` (기본 0.35 = 쉼 틈 아래 0.056 · 위 0.10 mm — 0.4 위는 아랫날이 눌려
떨림이 이중 주기로 갈라졌다), 수렴 0, 갑상피열근 0.25. 신경 요동 · 지터 끔.

사용: python scripts/vf_ns_calib.py 판 [--dump npz] [--t 0.10] [--hard] [--out profiles/vf/판_ns_calib.npz]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("W02A_PATH", "profiles/w02_anat_surrogate_v35.pt")
import numpy as np
import torch

from formant_ml.engine import voice_td as VT, tube_td as td
from formant_ml.engine.vf_calibration import EPS_LIMITS, model_signature, validate_pitch_table


def setup(run: str, hard: bool, over: dict | None = None):
    VT.GLOTTIS = "vf"; VT.VF_NS = True; VT.VF_NEURAL = False; VT.VF_BODY_DRIVE = False; VT.SLEW_ON = False
    VT.VF_ADD_SMOOTH_MS = 0.0; VT.VF_TA_SMOOTH_MS = 0.0
    VT.VF_NS_HARD = dict(VT.VF_NS_HARD_DEFAULT) if hard else None
    path = run if run.endswith("_track.npz") else f"out/VF/{run}_track.npz"
    with np.load(path, allow_pickle=False) as z:
        Z = dict(z)
    ac = json.loads(str(Z["acoustic_constants"]))["speaker"]
    tdp = VT.TDPath(48000, 48)
    for k, v in ac.items():
        nm = k[3:] if k.startswith("td_") else None
        if nm and hasattr(tdp, nm) and isinstance(getattr(tdp, nm), torch.Tensor) and not isinstance(v, (list, dict)):
            getattr(tdp, nm).data.fill_(float(v))
    for k, v in (over or {}).items():
        getattr(tdp, f"log_ns_{k}").data.fill_(float(np.log(v)))
    tdp._vf_jitter = lambda n, dt, dv: torch.ones(n, dtype=dt, device=dv)
    eff = {k: float(v.detach()) for k, v in tdp.ns_scales().items()}
    nfr = int(Z["values"].shape[0])
    dur = nfr * float(Z["frame_ms"]) / 1000.0 if np.ndim(Z["frame_ms"]) == 0 else nfr / 1000.0
    return tdp, eff, dur


class Bench:
    """한 성도 틀에서 모드 3 성대를 고정 자세로 모의."""

    def __init__(self, tdp, A0, L0, add=0.35, rdo=0.0, ta=0.25):
        self.tdp, self.A0, self.L0 = tdp, A0, L0
        self.add, self.rdo, self.ta = add, rdo, ta
        self.fs = td.FS_SIM

    def ug(self, ps: np.ndarray, eps: float | None = None, f0: float | None = None, add=None, rdo=None, ta=None) -> np.ndarray:
        """ps [dyn/cm²] (n,) → 성문 유량 (n,). eps 를 주면 표를 건너뛰고 그 ε, 아니면 f0 → 표."""
        n = ps.size
        nf = int(np.ceil(n / self.fs * 1000.0))
        f = lambda v: torch.full((1, nf), float(v), dtype=torch.float64)
        c = {"adduction": f(self.add if add is None else add), "rd_offset": f(self.rdo if rdo is None else rdo),
             "vf_ta": f(self.ta if ta is None else ta)}
        st = {"f0": f(250.0 if f0 is None else f0), "vf_qcorr": f(1.0)}
        orig = VT.vf_ns_eps
        if eps is not None:
            VT.vf_ns_eps = lambda x, _e=float(eps): torch.full_like(x, _e)
        try:
            with torch.no_grad():
                q, VR, ch = self.tdp.vf_inputs(c, st, n)
                z = torch.zeros(n, dtype=torch.float64)
                e6 = torch.tensor(1e-6, dtype=torch.float64)
                td.tube_torch(torch.as_tensor(np.tile(self.A0, (n, 1))), torch.full((n,), self.L0, dtype=torch.float64), ch[0].double(),
                              torch.as_tensor(ps), z, e6, e6, seed=1, Q=torch.zeros((n, self.A0.size), dtype=torch.float64),
                              vf=(q[0].double(), VR[0].double(), self.tdp.vf_statics()), return_area=True, vf_mode=3, fs=self.fs)
        finally:
            VT.vf_ns_eps = orig
        return td.TubeFn.last_rec[:, 0].copy()

    def threshold(self, ps_max: float, ramp_s: float = 1.0, **kw) -> float:
        """폐압을 0 → ps_max [cmH2O] 로 올려 떨림이 시작되는 폐압. 20 ms 창마다 직선을 빼고(올라가는 직류가 떨림으로 보이지 않게)
        떨림 폭 > 30 cm³/s 이고 직류의 15 % 이상, 그 뒤 70 % 이상의 창에서 이어지면 시작. 끝까지 없으면 nan."""
        n = int(ramp_s * self.fs)
        ramp = np.linspace(0.0, ps_max * 980.665, n)
        u = self.ug(ramp, **kw)
        W = int(0.02 * self.fs)
        fr = u[: (n // W) * W].reshape(-1, W)
        t = np.arange(W) - (W - 1) / 2.0
        slope = (fr * t).sum(1, keepdims=True) / (t * t).sum()
        r = fr - fr.mean(1, keepdims=True) - slope * t
        amp = r.max(1) - r.min(1)
        dc = fr.mean(1)
        ok = np.flatnonzero((amp > 30.0) & (amp > 0.15 * np.maximum(dc, 1.0)))
        if ok.size == 0 or (amp[ok[0]:] > 30.0).mean() < 0.7:
            return float("nan")
        return float(ramp[min(n - 1, (ok[0] + 1) * W)] / 980.665)

    def f0_at(self, p_cm: float, hold_s: float = 0.5, **kw):
        """폐압 p [cmH2O] 에 두고 뒤 0.3 s 의 (f0 Hz, 주기성, 이중 주기 여부, 유량 마루–골)."""
        n = int(hold_s * self.fs)
        ps = np.full(n, p_cm * 980.665)
        m = int(0.03 * self.fs)
        ps[:m] *= np.linspace(0.0, 1.0, m)
        u = self.ug(ps, **kw)[-int(0.3 * self.fs):]
        amp = float(np.ptp(u))
        if amp < 30.0 or not np.isfinite(u).all():
            return float("nan"), 0.0, False, amp
        x = u - u.mean()
        X = np.fft.rfft(x, 2 * x.size)
        ac = np.fft.irfft(np.abs(X) ** 2)[: x.size]
        ac /= ac[0]
        lo, hi = int(self.fs / 900), int(self.fs / 60)
        seg = ac[lo:hi]
        pk = [i for i in range(1, seg.size - 1) if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1]]
        if not pk:
            return float("nan"), 0.0, False, amp
        top = max(seg[i] for i in pk)
        k = next(i for i in pk if seg[i] >= 0.85 * top) + lo
        a_, b_, c_ = ac[k - 1], ac[k], ac[k + 1]
        kf = k + 0.5 * (a_ - c_) / (a_ - 2 * b_ + c_ + 1e-12)
        dbl = 2 * k < ac.size and ac[2 * k] > ac[k] + 0.05          # 두 배 주기가 더 닮았다 = 이중 주기
        return float(self.fs / kf), float(ac[k]), bool(dbl), amp


def f0_table(b: Bench, eps_grid, ps_max: float, log=print):
    rows = []
    for e in eps_grid:
        P = b.threshold(ps_max, eps=e)
        if not np.isfinite(P):
            rows.append((e, np.nan, np.nan, 0.0, False)); log(f"  ε {e:+.2f}: {ps_max:g} cmH2O 까지 안 떪"); continue
        p = float(np.clip(1.3 * P, 5.0, ps_max))
        f0, per, dbl, amp = b.f0_at(p, eps=e)
        rows.append((e, P, f0, per, dbl))
        log(f"  ε {e:+.2f}: 문턱 {P:4.1f} cmH2O → {p:4.1f} 에서 f0 {f0:6.1f} Hz (주기성 {per:.2f}{', 이중 주기' if dbl else ''}, 유량 폭 {amp:.0f})")
    # An inverse for modal pitch cannot use a subharmonic/weakly periodic branch.
    ok = [(e, f) for e, P, f, per, dbl in rows if np.isfinite(f) and per >= 0.8 and not dbl]
    keep, fmax = [], 0.0
    for e, f in ok:                                         # 단조 증가만 (1 % 넘게 오를 때)
        if f > fmax * 1.01:
            keep.append((e, f)); fmax = f
    return rows, np.array([k[0] for k in keep]), np.array([k[1] for k in keep])


def reference_frame(path, duration, t):
    """Read full-rate TD_DUMP_IN, with explicit support for legacy dumps."""
    with np.load(path, allow_pickle=False) as z:
        fs = float(z["fs"]) if "fs" in z.files else td.FS_SIM
        A, L = np.asarray(z["A"], float), np.asarray(z["L"], float)
        n_pre = int(z["n_pre"]) if "n_pre" in z.files else A.shape[0] - int(round(duration * fs))
    if (not np.isfinite(fs) or fs <= 0 or A.ndim != 2 or L.shape != (A.shape[0],)
            or n_pre < 0 or not 0 <= t < duration):
        raise ValueError("Reference requires full-rate TD_DUMP_IN matching the calibration utterance")
    i = n_pre + int(t * fs)
    if not 0 <= i < A.shape[0] or not np.isfinite(A[i]).all() or np.any(A[i] <= 0) or not 0 < L[i] < np.inf:
        raise ValueError("Invalid reference tract frame")
    return A[i], float(L[i])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="out/VF의 판 이름 또는 _track.npz 경로")
    ap.add_argument("--dump", default="out/_tmp/3d/n1_tdin.npz", help="관 입력 원래 표본률 기록 (A, L) — TD_DUMP_IN, TD_DUMP 아님")
    ap.add_argument("--t", type=float, default=0.10, help="성도 틀 시각 [s] (모음)")
    ap.add_argument("--hard", action="store_true", help="화자 상수를 VF_NS_HARD_DEFAULT 로 자른다 (--vf-ns-hard 판)")
    ap.add_argument("--set", default="", help="화자 상수 덮어쓰기 (배율) 'kc=1.2,len=1.1'")
    ap.add_argument("--add", type=float, default=0.35)
    ap.add_argument("--rdo", type=float, default=0.0)
    ap.add_argument("--ta", type=float, default=0.25)
    ap.add_argument("--ps-max", type=float, default=15.0, help="표 · 지도의 폐압 상한 [cmH2O]")
    ap.add_argument("--eps-min", type=float, default=EPS_LIMITS[0])
    ap.add_argument("--eps-max", type=float, default=EPS_LIMITS[1])
    ap.add_argument("--eps-step", type=float, default=0.05)
    ap.add_argument("--lr", action="store_true", help="좌우 성대를 따로 푼 판의 보정")
    ap.add_argument("--edge-um", type=float, default=None, help="닫힘 이음 폭 [μm] (beam_membrane.CLOSURE_EDGE_CM, 판의 --vf-closure-edge 와 같게)")
    ap.add_argument("--rest-modal-um", type=float, default=None, help="모달 쉼 반틈새 [μm] (voice_td.VF_R_MODAL, 판의 --vf-rest-modal 과 같게, §52.532)")
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if (not EPS_LIMITS[0] <= a.eps_min < a.eps_max <= EPS_LIMITS[1]
            or not np.isfinite(a.eps_step) or a.eps_step <= 0 or not 0 < a.ps_max < np.inf):
        ap.error("eps 범위는 [-0.3, 0.7] 안, eps-step과 ps-max는 유한한 양수여야 한다")
    from formant_ml.physics import beam_membrane as _bm
    _bm.LR_FOLDS = a.lr
    over = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.set.split(",") if kv}
    if a.edge_um is not None:
        from formant_ml.physics import beam_membrane as _bm
        _bm.CLOSURE_EDGE_CM = a.edge_um * 1e-4
    if a.rest_modal_um is not None:
        from formant_ml.engine import voice_td as _vr
        _vr.VF_R_MODAL = a.rest_modal_um * 1e-4
    tdp, eff, dur = setup(a.run, a.hard, over)
    A0, L0 = reference_frame(a.dump, dur, a.t)
    print(f"{a.run}: 실효 상수 " + " ".join(f"{k} {v:.3g}" for k, v in eff.items()) +
          f" | 틀 {a.t:.3f} s (관 {L0:.2f} cm) | 내전 {a.add} 수렴 {a.rdo} 갑상피열근 {a.ta}", flush=True)
    b = Bench(tdp, A0, L0, a.add, a.rdo, a.ta)
    t0 = time.time()
    eps_grid = np.arange(a.eps_min, a.eps_max, a.eps_step)
    eps_grid = np.r_[eps_grid[eps_grid < a.eps_max - 1e-10], a.eps_max]
    rows, et, ft = f0_table(b, eps_grid, a.ps_max, log=lambda s: print(s, flush=True))
    print(f"  표 ({time.time()-t0:.0f} s): " + " ".join(f"{e:+.2f}:{f:.0f}" for e, f in zip(et, ft)), flush=True)
    if et.size < 3:
        print("  표 점이 3 개보다 적다 — 이 상수로는 떨림 범위가 없다", flush=True)
        return 1
    validate_pitch_table(et, ft)
    out = dict(eps_tab=et, f0_tab=ft, rows=np.array([(r[0], r[1], r[2], r[3], float(r[4])) for r in rows]),
               consts=json.dumps(eff), cond=json.dumps(dict(run=a.run, dump=a.dump, t=a.t, add=a.add, rdo=a.rdo, ta=a.ta, rest_modal_um=a.rest_modal_um,
                                                             ps_max=a.ps_max, hard=a.hard, set=over, edge_um=a.edge_um,
                                                             eps_min=a.eps_min, eps_max=a.eps_max, eps_step=a.eps_step)),
               signature=json.dumps(model_signature(tdp)), reference_A=A0, reference_L=L0)
    if not a.no_map:
        VT.VF_NS_EPS_TAB, VT.VF_NS_F0_TAB = tuple(map(float, et)), tuple(map(float, ft))
        F0 = tuple(float(v) for v in np.geomspace(ft[0], ft[-1], 6))
        ADD, RDO, TA = (0.3, 0.35, 0.45, 0.55), (-1.0, 0.0, 1.0), (0.2, 0.5)
        res = np.full((len(F0), len(ADD), len(RDO), len(TA)), np.nan)
        t0 = time.time()
        for (i0, f0), (j, ad), (k, r), (l, ta) in itertools.product(enumerate(F0), enumerate(ADD), enumerate(RDO), enumerate(TA)):
            res[i0, j, k, l] = b.threshold(a.ps_max, f0=f0, add=ad, rdo=r, ta=ta)
        print(f"  문턱 지도 {res.size} 점 {time.time()-t0:.0f} s", flush=True)
        for l, ta in enumerate(TA):
            for k, r in enumerate(RDO):
                print(f"  갑상피열근 {ta}, 수렴 {r:+g}: 문턱 [cmH2O] (행 f0, 열 내전 {ADD})")
                for i0, f0 in enumerate(F0):
                    print(f"    f0 {f0:5.0f}: " + " ".join("  ---" if not np.isfinite(x) else f"{x:5.1f}" for x in res[i0, :, k, l]))
        out.update(pth=res, f0=np.array(F0), add=np.array(ADD), rdo=np.array(RDO), ta=np.array(TA))
    stem = os.path.basename(a.run).removesuffix("_track.npz")
    path = a.out or f"profiles/vf/{stem}_ns_calib.npz"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **out)
    print(f"  저장 {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
