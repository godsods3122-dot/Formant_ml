"""Cached, target-synchronous harmonic observations for offline fitting only."""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve

from .waveform import harmonic_basis


class HarmonicMagnitudeLoss:
    """Measure individual harmonic amplitudes without FFT-bin scalloping.

    The target fixes pitch, voicing, support and confidence. Projection matrices
    are cached; the synthesizer gains no parameters or additional runtime work.
    """

    WINDOW = 2048
    STRIDE = 512

    def __init__(self, target: np.ndarray, fs: float, hop: int, f0: np.ndarray,
                 voiced: np.ndarray, fricative: np.ndarray,
                 fmax: float = 3000.0, device: str = "cpu",
                 pulses: np.ndarray | None = None):
        target = np.asarray(target, dtype=np.float64)
        f0 = np.asarray(f0, dtype=np.float64)
        voiced = np.asarray(voiced, dtype=bool)
        fricative = np.asarray(fricative, dtype=bool)
        pulses = np.asarray([] if pulses is None else pulses, dtype=np.float64)
        if (target.ndim != 1 or f0.ndim != 1 or not len(f0)
                or fs <= 0 or hop <= 0 or not 0 < fmax < fs / 2
                or not np.isfinite(target).all() or not np.isfinite(f0).all()
                or (f0 < 0).any()):
            raise ValueError("invalid target, pitch or sampling grid for harmonic loss")
        if voiced.shape != f0.shape or fricative.shape != f0.shape:
            raise ValueError("harmonic masks must match the target pitch track")
        if (pulses.ndim != 1 or not np.isfinite(pulses).all()
                or (np.diff(pulses) <= 0).any()):
            raise ValueError("target pulses must be finite and strictly increasing")
        self.n_samples = len(target)
        self.fmax = float(fmax)
        self.coverage = np.zeros(len(f0), dtype=bool)
        self.projection = torch.empty(0, 0, 2, self.WINDOW, device=device)
        self.frame_indices = torch.empty(0, dtype=torch.long, device=device)
        self.weights = torch.empty(0, 0, device=device)
        self.reference_power = torch.empty(0, 0, device=device)
        self.live = torch.empty(0, 0, dtype=torch.bool, device=device)
        good = voiced & ~fricative & (f0 > 0) & (f0 <= fmax)
        if not good.any() or len(target) < self.WINDOW:
            return

        sample = np.arange(len(target))
        frame = np.minimum(sample // hop, len(f0) - 1)
        valid = f0 > 0
        pitch = np.interp(sample, (np.flatnonzero(valid) + 0.5) * hop, f0[valid])
        phase = 2 * np.pi * np.cumsum(pitch) / fs
        supported = good[frame]
        if len(pulses) >= 2:
            # --init may contain fitted F0; recorded pulses remain target data.
            times = sample / fs
            phase = np.interp(times, pulses, 2 * np.pi * np.arange(len(pulses)))
            supported &= (times >= pulses[0]) & (times <= pulses[-1])
            pitch = np.gradient(phase) * fs / (2 * np.pi)
        win = np.hanning(self.WINDOW)
        candidates = []
        for j, start in enumerate(range(0, len(target) - self.WINDOW + 1, self.STRIDE)):
            stop = start + self.WINDOW
            if not supported[start:stop].all():
                continue
            seg = target[start:stop]
            power = float(np.dot(win, seg * seg) / win.sum())
            candidates.append((j, start, power))
        if not candidates:
            return
        peak_power = max(p for _, _, p in candidates)
        rows, masks, indices = [], [], []
        for j, start, power in candidates:
            if power <= max(peak_power * 1e-6, np.finfo(float).tiny):
                continue
            stop = start + self.WINDOW
            # Every retained harmonic stays inside the measured band during the window.
            kmax = min(int(fmax / pitch[start:stop].max()), self.WINDOW // 4)
            if kmax < 1:
                continue
            ph = phase[start:stop] - phase[start + self.WINDOW // 2]
            basis = harmonic_basis(self.WINDOW, 0.0, fs, kmax, phase=ph)
            basis = np.column_stack((np.ones(self.WINDOW), basis))
            bt = basis.T * win
            gram = bt @ basis
            gram.flat[::len(gram) + 1] += 1e-8 * np.trace(gram) / len(gram)
            projection = cho_solve(cho_factor(gram), bt)
            coef = projection @ target[start:stop]
            pairs = np.stack((projection[1:kmax + 1], projection[kmax + 1:]), axis=1)
            amp2 = coef[1:kmax + 1] ** 2 + coef[kmax + 1:] ** 2
            residual = target[start:stop] - basis @ coef
            noise = float(np.dot(win, residual * residual) / win.sum())
            variance = noise * np.sum(pairs * pairs, axis=(1, 2))
            keep = (amp2 >= amp2.max() * 1e-4) & (amp2 > 10.0 * variance)
            if not keep.any():
                continue
            rows.append(pairs)
            masks.append(keep)
            indices.append(j)
            self.coverage[frame[start:stop]] = True
        if not rows:
            return

        width = max(len(row) for row in rows)
        proj = np.zeros((len(rows), width, 2, self.WINDOW), dtype=np.float32)
        live = np.zeros((len(rows), width), dtype=bool)
        for i, (row, mask) in enumerate(zip(rows, masks)):
            proj[i, :len(row)] = row
            live[i, :len(row)] = mask
        self.projection = torch.as_tensor(proj, device=device)
        self.frame_indices = torch.as_tensor(indices, dtype=torch.long, device=device)
        self.live = torch.as_tensor(live, device=device)
        weights = live / live.sum(axis=1, keepdims=True)
        self.weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
        with torch.no_grad():
            y = torch.as_tensor(target, dtype=torch.float32, device=device)
            self.reference_power = self._power(y)

    @property
    def observations(self) -> int:
        return int(self.live.sum())

    def _power(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim != 1 or y.numel() != self.n_samples:
            raise ValueError("harmonic loss requires the unchanged target sample grid")
        frames = y.unfold(0, self.WINDOW, self.STRIDE)[self.frame_indices]
        coef = torch.einsum("mkcn,mn->mkc", self.projection.to(y.dtype), frames)
        return (coef * coef).sum(-1)

    def errors_db(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim != 1 or y.numel() != self.n_samples:
            raise ValueError("harmonic loss requires the unchanged target sample grid")
        if not self.weights.numel():
            return y[:0].reshape(0, 0)
        power = self._power(y)
        reference = self.reference_power.to(y.dtype)
        floor = reference * 1e-6 + torch.finfo(y.dtype).tiny
        return (10.0 / math.log(10.0)) * (
            torch.log(power + floor) - torch.log(reference + floor))

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        db = self.errors_db(y)
        if not db.numel():
            return db.sum()
        error = torch.sqrt(db * db + 0.1 ** 2) - 0.1
        return (error * self.weights).sum() / self.weights.sum() / 5.0

    def report(self, y: torch.Tensor) -> dict:
        with torch.no_grad():
            db = self.errors_db(y).abs()
            if not db.numel():
                return dict(mae_db=None, p95_db=None, max_db=None, observations=0,
                            windows=0, by_order={})
            values = db[self.live]
            by_order = {}
            for k in range(db.shape[1]):
                v = db[:, k][self.live[:, k]]
                if v.numel():
                    by_order[str(k + 1)] = dict(mae_db=float(v.mean()),
                                               p95_db=float(torch.quantile(v, 0.95)),
                                               observations=v.numel())
            return dict(mae_db=float((db * self.weights).sum() / self.weights.sum()),
                        p95_db=float(torch.quantile(values, 0.95)),
                        max_db=float(values.max()), observations=values.numel(),
                        windows=db.shape[0], by_order=by_order)
