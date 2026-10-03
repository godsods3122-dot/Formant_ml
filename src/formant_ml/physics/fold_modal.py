"""Linearized modal summary of the 3D vocal-fold FEM (``vocal_fold_solid``) for reduced-model calibration.

MEASUREMENTS §52.537. The coupled 3D reference (``phonation``) costs ~15-26 s per 0.21 µs step, so it cannot sit in the
fitting loop. Its solid can still *define* the fold the fitter uses: at a muscle posture (CT strain, TA activation) we
solve the prestressed static equilibrium of the actual Neo-Hookean/fiber/active FEM, assemble its exact tangent
stiffness from the same element blocks ``VocalFoldSolid._timestep`` uses (``V Bᵀ ∂P/∂F B``), the viscous tangent from
``∂P/∂Ḟ`` and the lumped mass, and report what a reduced cover-body model must reproduce:

* eigenfrequencies of the modes whose medial lateral motion is the first AP harmonic sin(πx/L) (the shape the mode-3
  strip model projects onto), with their inferior→superior medial profile and modal damping ratio,
* the static medial lateral compliance to a uniform transglottal pressure, per SI row,
* the medial bulge produced by TA activation relative to the passive posture.

Only one fold is analysed: without contact the two folds are uncoupled, so the bilateral spectrum is this one doubled.
This is a small-amplitude linearization about a posture — not self-oscillation, contact, or flow coupling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from .vocal_fold_solid import (BilateralControl, FoldControl, SolidConfig, VocalFoldSolid, nominal_fold_mesh,
                               triangle_pressure_forces)


@dataclass(frozen=True)
class FoldModes:
    eps: float
    ta: float
    freq_hz: np.ndarray            # (n,) AP-first-harmonic modes, ascending
    damping_ratio: np.ndarray      # (n,)
    medial_profile: np.ndarray     # (n, n_si+1) lateral medial amplitude per SI row (inferior → superior), max |.| = 1
    ap_purity: np.ndarray          # (n,) share of medial lateral energy in sin(πx/L)
    compliance_m_per_pa: np.ndarray  # (n_si+1,) static lateral medial displacement at mid-length per Pa (+ = lateral)
    bulge_m: np.ndarray            # (n_si+1,) medial displacement at mid-length relative to the passive posture (+ = medial)
    length_m: float
    height_m: float
    newton_residual_n: float


def build_solid(n_ap: int = 8, n_si: int = 6, cells_per_layer=(2, 2, 2), **mesh) -> VocalFoldSolid:
    meshes = [nominal_fold_mesh(side, n_ap=n_ap, n_si=n_si, cells_per_layer=cells_per_layer, **mesh)
              for side in ("left", "right")]
    return VocalFoldSolid(SolidConfig(*meshes))


class FoldLinearizer:
    """Static equilibrium and tangent operators of the *left* fold of a ``VocalFoldSolid``."""

    def __init__(self, solid: VocalFoldSolid):
        self.s = solid
        n_left = solid._offset
        self.left = torch.zeros(len(solid._X), dtype=torch.bool)
        self.left[:n_left] = True
        free = (~solid._fixed) & self.left[:, None]
        self.free = free
        self.free_idx = np.flatnonzero(free.flatten().numpy())
        self.x = solid._X.clone()
        self.ctl = (0.0, 0.0)
        left_elements = solid._element_side == 0
        self.elements = torch.nonzero(left_elements).flatten()
        self.press = solid._pressure(None, ())
        mesh = solid._meshes[0]
        self.grid = np.asarray(mesh.medial_grid)                     # (n_ap+1, n_si+1) node ids
        X = solid._X.numpy()
        self.length_m = float(np.ptp(X[:n_left, 0]))
        self.height_m = float(np.ptp(X[:n_left, 2]))
        mirror = X[n_left:] * np.array([1.0, -1.0, 1.0])
        if mirror.shape != X[:n_left].shape or not np.allclose(mirror, X[:n_left], rtol=0, atol=1e-12):
            raise ValueError("modal linearization needs node-for-node mirror-symmetric left/right meshes")

    def _mirror(self, x):
        """Place the right fold as the mirror image of the left (keeps contact registration; folds are uncoupled)."""
        n = self.s._offset
        x = x.clone()
        x[n:] = x[:n] * x.new_tensor([1.0, -1.0, 1.0])
        return x

    def control(self, eps, ta):
        c = FoldControl(ta, eps)
        return BilateralControl(c, c)

    def _tangent(self, x, ctl, viscous=False):
        """Sparse global tangent (ndof×ndof) of the left fold's elements: stiffness, or viscous if ``viscous``."""
        s = self.s
        e = self.elements
        F = (x[s._t[e, 1:]] - x[s._t[e, :1]]).transpose(1, 2) @ s._inverse[e]
        Fdot = torch.zeros_like(F)
        sub = _ElementView(s, e)
        with torch.enable_grad():
            F = F.detach().requires_grad_(True)
            Fdot = Fdot.detach().requires_grad_(True)
            pp, pv, pa, _, _ = sub.constitutive(F, Fdot, ctl)
            stress = (pv if viscous else pp + pa).reshape(-1, 9)
            wrt = Fdot if viscous else F
            rows = [torch.autograd.grad(stress[:, k].sum(), wrt, retain_graph=k < 8)[0].reshape(-1, 9) for k in range(9)]
        A = torch.stack(rows, 1).detach()
        blocks = s._volume[e, None, None] * (s._B[e].transpose(1, 2) @ A @ s._B[e])
        dofs = s._dofs[e]
        r = dofs[:, :, None].expand(-1, 12, 12).reshape(-1).numpy()
        c = dofs[:, None, :].expand(-1, 12, 12).reshape(-1).numpy()
        n = s._X.numel()
        return sp.csr_matrix((blocks.reshape(-1).numpy(), (r, c)), shape=(n, n))

    def _residual(self, x, ctl):
        f, _ = self.s._evaluate(x, torch.zeros_like(x), ctl, self.press)
        return (f[0] + f[2] + f[4]).flatten().numpy()[self.free_idx]

    def solve(self, eps, ta, *, increment=0.05, tol_n=1e-10, iters=25):
        """Prestressed equilibrium by load continuation from the last solved posture."""
        e0, a0 = self.ctl
        n_load = max(1, int(np.ceil(max(abs(eps - e0), abs(ta - a0)) / increment)))
        res = 0.0
        for k in range(1, n_load + 1):
            e, a = e0 + (eps - e0) * k / n_load, a0 + (ta - a0) * k / n_load
            ctl = self.control(e, a)
            # Affine predictor: carry the strain increment through the whole fold, not only its supports.
            prev = self.s._targets(self.control(*self.ctl))
            targets = self.s._targets(ctl)
            x = self.x + (targets - prev)
            x[self.s._fixed] = targets[self.s._fixed]
            x = self._mirror(x)
            for _ in range(iters):
                r = self._residual(x, ctl)
                res = float(np.abs(r).max())
                if res < tol_n:
                    break
                K = self._tangent(x, ctl)[self.free_idx][:, self.free_idx]
                dx = spla.spsolve(K.tocsc(), r)
                norm = float(np.linalg.norm(r))
                step = 1.0
                while True:
                    trial = x.clone()
                    trial.view(-1)[self.free_idx] += torch.as_tensor(step * dx)
                    trial = self._mirror(trial)
                    try:
                        if float(np.linalg.norm(self._residual(trial, ctl))) < norm:
                            break
                    except ValueError:
                        pass
                    step /= 2
                    if step < 1e-6:
                        raise ValueError(f"FEM equilibrium line search failed at eps {e:.3f}, ta {a:.3f} (residual {res:.3g} N)")
                x = trial
            if res >= 1e3 * tol_n:
                raise ValueError(f"FEM equilibrium did not converge at eps {e:.3f}, ta {a:.3f}: residual {res:.3g} N")
            self.x, self.ctl = x, (e, a)
        return res

    def _mid_lateral(self, u):
        """Lateral (y) displacement of the left medial grid at mid-length, per SI row; + = medial (toward +y)."""
        g = self.grid
        mid = g.shape[0] // 2
        return u.reshape(-1, 3)[g[mid], 1]

    def summary(self, n_modes=12) -> FoldModes:
        s = self.s
        ctl = self.control(*self.ctl)
        K = self._tangent(self.x, ctl)[self.free_idx][:, self.free_idx]
        K = 0.5 * (K + K.T)
        C = self._tangent(self.x, ctl, viscous=True)[self.free_idx][:, self.free_idx]
        m = s._mass.repeat_interleave(3).numpy()[self.free_idx]
        M = sp.diags(m)
        w2, V = spla.eigsh(K.tocsc(), k=n_modes, M=M.tocsc(), sigma=0.0, which="LM")
        order = np.argsort(w2)
        w2, V = w2[order], V[:, order]
        full = np.zeros((len(order), s._X.numel()))
        full[:, self.free_idx] = V.T
        g = self.grid
        xs = s._X.numpy()[g[:, 0], 0]
        shape = np.sin(np.pi * (xs - xs.min()) / np.ptp(xs))
        keep, prof, purity, zeta = [], [], [], []
        for i in range(len(order)):
            lat = full[i].reshape(-1, 3)[g, 1]                        # (n_ap+1, n_si+1)
            coef = (shape @ lat) / (shape @ shape)
            fit = np.outer(shape, coef)
            pur = float((fit ** 2).sum() / max((lat ** 2).sum(), 1e-300))
            if pur < 0.8:
                continue
            keep.append(i)
            prof.append(coef / np.abs(coef).max() * np.sign(coef[np.argmax(np.abs(coef))]))
            purity.append(pur)
            v = V[:, i]
            zeta.append(float(v @ (C @ v)) / (2.0 * np.sqrt(max(w2[i], 1e-300)) * float(v @ (m * v))))
        # Static compliance to uniform medial pressure on the left fold (positive pressure pushes laterally).
        tris = s._surfaces[0]
        p = torch.ones(len(tris), dtype=torch.float64)
        fp = triangle_pressure_forces(self.x, tris, p).flatten().numpy()[self.free_idx]
        u = np.zeros(s._X.numel())
        u[self.free_idx] = spla.spsolve(K.tocsc(), fp)
        compliance = -self._mid_lateral(u)
        bulge = np.zeros(g.shape[1])
        if self.ctl[1] != 0.0:
            here = self.x.clone()
            state = self.ctl
            passive = FoldLinearizer(s)
            passive.solve(self.ctl[0], 0.0)
            bulge = self._mid_lateral((here - passive.x).flatten().numpy())
            self.x, self.ctl = here, state
        return FoldModes(self.ctl[0], self.ctl[1], np.sqrt(np.maximum(w2[keep], 0.0)) / (2 * np.pi),
                         np.asarray(zeta), np.asarray(prof), np.asarray(purity), compliance, bulge,
                         self.length_m, self.height_m, float(np.abs(self._residual(self.x, ctl)).max()))


class _ElementView:
    """Evaluate ``VocalFoldSolid._constitutive`` on a subset of elements."""

    def __init__(self, solid, elements):
        self.s, self.e = solid, elements

    def constitutive(self, F, Fdot, ctl):
        s, e = self.s, self.e
        saved = (s._parameters, s._fib, s._element_side)
        try:
            s._parameters = {k: v[e] for k, v in saved[0].items()}
            s._fib = saved[1][e]
            s._element_side = saved[2][e]
            return s._constitutive(F, Fdot, ctl)
        finally:
            s._parameters, s._fib, s._element_side = saved
