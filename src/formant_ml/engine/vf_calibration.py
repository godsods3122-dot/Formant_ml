"""Contracts for the reduced vocal-fold model's numerical calibration tables."""
from __future__ import annotations

import json
import math

import numpy as np
import torch

EPS_LIMITS = (-0.3, 0.7)


def validate_pitch_table(eps, f0) -> None:
    e, f = np.asarray(eps, float), np.asarray(f0, float)
    if (e.ndim != 1 or f.shape != e.shape or e.size < 2
            or not np.isfinite(e).all() or not np.isfinite(f).all()
            or np.any(np.diff(e) <= 0) or np.any(np.diff(f) <= 0)
            or np.any(f <= 0) or e[0] < EPS_LIMITS[0] or e[-1] > EPS_LIMITS[1]):
        raise ValueError("NS calibration requires increasing finite eps/f0 pairs within eps [-0.3, 0.7]")


def grid_index(value: torch.Tensor, grid) -> torch.Tensor:
    """Fractional *sample index*, not distance between the axis endpoints."""
    g = torch.as_tensor(grid, dtype=value.dtype, device=value.device)
    if g.numel() == 1:
        return value * 0.0
    i = torch.searchsorted(g, value.detach().contiguous()).clamp(1, g.numel() - 1)
    v = torch.where(value < g[0], g[0], torch.where(value > g[-1], g[-1], value))
    w = (v - g[i - 1]) / (g[i] - g[i - 1])
    return i - 1 + w


def model_signature(tdp) -> dict:
    from . import voice_td as vt, tube_td as td
    from ..physics import beam_membrane as bm

    return dict(
        schema_version=1,
        consts={k: float(v.detach()) for k, v in tdp.ns_scales().items()},
        closure_edge_cm=bm.CLOSURE_EDGE_CM,
        rest={k: getattr(vt, k) for k in
              ("VF_R_MID", "VF_R_K", "VF_R_SLOPE", "VF_R_QUAD", "VF_R_FLOOR",
               "VF_R_MODAL", "VF_ADD_MODAL", "VF_CONV", "VF_CONV_PER_RD")},
        chink_cm2=(float(tdp.log_vf_chink.detach().exp()) if vt.VF_CHINK_FIX is None else vt.VF_CHINK_FIX),
        nstrip=td.VF_NSTRIP, long_modes=bm.LONG_MODES, fs_sim=td.FS_SIM,
        wall=[td.WALL_M, td.WALL_R, td.WALL_K],
        lr_folds=bm.LR_FOLDS,
        lr_parameters=([float(tdp.ns_lr_u_q.detach()), float(tdp.ns_lr_u_m.detach()),
                        vt.VF_LR_DQ_MAX, vt.VF_LR_DM_MAX] if bm.LR_FOLDS else []),
    )


def locked_names() -> tuple[str, ...]:
    from . import voice_td as vt
    from ..physics import beam_membrane as bm

    names = tuple(f"td_log_ns_{k}" for k in vt.VF_NS_INIT) + ("td_log_vf_chink",)
    return names + (("td_ns_lr_q", "td_ns_lr_m") if bm.LR_FOLDS else ())


def check_calibration(path, tdp, locked_constants, *, approximate=False, log=print) -> None:
    """Reject known stale conditions; legacy tables explicitly remain only partly checked."""
    from . import voice_td as vt

    if not vt.VF_NS or vt.GLOTTIS != "vf":
        raise ValueError("NS calibration requires --td --glottis vf --vf-ns")
    actual = model_signature(tdp)
    errors = []

    def compare(expected, current, prefix):
        if isinstance(expected, dict):
            if not isinstance(current, dict):
                errors.append(f"{prefix[:-1]}: different type")
                return
            for k in current.keys() - expected.keys():
                errors.append(f"{prefix}{k}: missing saved condition")
            for k, v in expected.items():
                if k not in current:
                    errors.append(f"{prefix}{k}: unsupported saved condition")
                else:
                    compare(v, current[k], f"{prefix}{k}.")
        elif isinstance(expected, list):
            if not isinstance(current, list) or len(expected) != len(current):
                errors.append(f"{prefix[:-1]}: different shape")
            else:
                for i, (a, b) in enumerate(zip(expected, current)):
                    compare(a, b, f"{prefix}{i}.")
        elif not math.isclose(float(expected), float(current), rel_tol=1e-6, abs_tol=1e-9):
            errors.append(f"{prefix[:-1]}: table={expected}, model={current}")

    with np.load(path, allow_pickle=False) as z:
        if "eps_tab" in z.files:
            validate_pitch_table(z["eps_tab"], z["f0_tab"])
            for name, current in (("eps_tab", vt.VF_NS_EPS_TAB), ("f0_tab", vt.VF_NS_F0_TAB)):
                saved = np.asarray(z[name], float)
                if saved.shape != np.shape(current) or not np.allclose(saved, current, rtol=1e-9, atol=1e-9):
                    errors.append(f"{name}: differs from the active pitch mapping")
        if "signature" in z.files:
            compare(json.loads(str(z["signature"])), actual, "")
        else:
            log(f"  WARNING: {path}: legacy NS table; wall/chink/LR/rest geometry are not fully recorded")
            if "consts" in z.files:
                compare(json.loads(str(z["consts"])), actual["consts"], "consts.")
            else:
                errors.append("missing speaker constants")
            cond = json.loads(str(z["cond"])) if "cond" in z.files else {}
            for key, field in (("edge_um", "closure_edge_cm"), ("rest_modal_um", "rest")):
                if cond.get(key) is not None:
                    current = actual[field] if field != "rest" else actual["rest"]["VF_R_MODAL"]
                    compare(float(cond[key]) * 1e-4, current, key + ".")
        if "pth" in z.files and "f0_tab" in z.files:
            f = np.asarray(z["f0"], float)
            tab = np.asarray(z["f0_tab"], float)
            if f.min() < tab[0] - 1.0 or f.max() > tab[-1] + 1.0:
                errors.append("threshold grid extends outside the measured pitch table")
    free = set(locked_names()) - set(locked_constants)
    if free:
        errors.append("table-dependent constants are not locked: " + ", ".join(sorted(free)))
    if errors:
        message = f"NS calibration {path}: " + "; ".join(errors)
        if not approximate:
            raise ValueError(message + ". Match the speaker/settings and use --vf-ns-lock, regenerate the table, "
                             "or explicitly use --vf-ns-calib-approx for a legacy approximation.")
        log("  WARNING: APPROXIMATE " + message)
    if vt.VF_BODY_DRIVE:
        log("  WARNING: NS threshold table measures unforced onset, not body-driven phonation")
    log("  NS calibration: fixed reference tract/TA/pressure; not a human physiological validation")


def pitch_coverage(f0, voiced=None) -> dict:
    from . import voice_td as vt

    f = np.asarray(f0, float).reshape(-1)
    if voiced is not None:
        mask = np.asarray(voiced, bool).reshape(-1)
        if mask.size != f.size:
            raise ValueError("Pitch coverage requires one voiced flag per frame")
        f = f[mask]
    if not np.isfinite(f).all() or np.any(f <= 0):
        raise ValueError("Pitch coverage requires finite positive frequencies")
    eps = vt.vf_ns_eps(torch.as_tensor(f, dtype=torch.float64)).numpy()
    lo, hi = vt.VF_NS_F0_TAB[0], vt.VF_NS_F0_TAB[-1]
    return dict(frames=int(f.size), measured_hz=[lo, hi],
                outside=int(np.sum((f < lo) | (f > hi))),
                saturated=int(np.sum((eps <= EPS_LIMITS[0]) | (eps >= EPS_LIMITS[1]))))
