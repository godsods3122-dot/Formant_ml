"""Prescribed-flow, passive input-impedance proxy (cgs acoustic units).

This is not a vocal-fold oscillator or a measured anatomical two-port.
See docs/IMPEDANCE_SOURCE.md for the midpoint power balance and limitations.
The optional CPU backend has an analytic first-order adjoint, not a sample-wise
Torch graph. Missing numba is an error only when the loaded source is used.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.autograd.function import once_differentiable

from .glottis import CMH2O, RHO
from .noise import C_SOUND

try:
    from numba import njit
except ModuleNotFoundError as exc:
    if exc.name != "numba":
        raise
    njit = None

N_MODES = 3
GLOTTAL_LENGTH_CM = 0.3
TRACT_AREA_CM2 = 3.0
PULSE_FLOW_CM3_S = 100.0
VISCOUS_RESISTANCE = 1.0  # dyn*s/cm^5, a small positive incremental resistance


def require_backend() -> None:
    if njit is None:
        raise RuntimeError("The loaded glottal source requires the optional numba "
                           "backend; install formant-ml[realtime].")


def _forward(ref, mass, resistance, quadratic, mean, aa, bb, zz, ww,
             initial, coupling, r0):
    batch, samples = ref.shape
    history = np.empty((batch, samples + 1, 2 + 2 * N_MODES))
    pressure = np.empty_like(ref)
    for b in range(batch):
        history[b, 0] = initial[b]
        for n in range(samples):
            old = history[b, n]
            new = history[b, n + 1]
            ur = 0.5 * (ref[b, n] + old[1])
            m, r, k, u0 = mass[b, n], resistance[b, n], quadratic[b, n], mean[b, n]
            drive = 0.5 * m * (ref[b, n] - old[1]) + r * ur
            drive += k * ((u0 + ur) * abs(u0 + ur) - u0 * abs(u0))
            base = 0.0
            impedance = r0
            for j in range(N_MODES):
                base += aa[b, n, j] * old[2 + 2*j] + bb[b, n, j] * old[3 + 2*j]
                impedance += zz[b, n, j]
            a = m + r + coupling * impedance
            rhs = drive + m * old[0] - coupling * base + a * u0 + k * u0 * abs(u0)
            total = 2.0 * rhs / (a + math.sqrt(a*a + 4.0*k*abs(rhs)))
            mid_flow = total - u0
            new[0], new[1] = 2.0 * mid_flow - old[0], ref[b, n]
            load = r0 * mid_flow
            for j in range(N_MODES):
                p, q = old[2 + 2*j], old[3 + 2*j]
                pm = aa[b, n, j] * p + bb[b, n, j] * q + zz[b, n, j] * mid_flow
                new[2 + 2*j] = 2.0 * pm - p
                new[3 + 2*j] = q + 2.0 * ww[b, n, j] * pm
                load += pm
            pressure[b, n] = coupling * load
    return history, pressure


def _backward(gflow, gpressure, gfinal, ref, mass, resistance, quadratic, mean,
              aa, bb, zz, ww, history, coupling, r0):
    batch, samples = ref.shape
    gr, gm, gv, gk, gu = [np.zeros_like(ref) for _ in range(5)]
    ga, gb, gz, gw = [np.zeros_like(aa) for _ in range(4)]
    gi = np.empty_like(gfinal)
    for b in range(batch):
        adj = gfinal[b].copy()
        for n in range(samples - 1, -1, -1):
            old, new = history[b, n], history[b, n + 1]
            v = 0.5 * (old[0] + new[0])
            vr = 0.5 * (old[1] + ref[b, n])
            m, r, k, u0 = mass[b, n], resistance[b, n], quadratic[b, n], mean[b, n]
            gp = coupling * gpressure[b, n]
            lflow = adj[0] + gflow[b, n]
            lv = 2.0 * lflow + gp * r0
            prev = np.zeros(2 + 2 * N_MODES)
            prev[0] = -lflow
            gr[b, n] = adj[1]
            lm = np.empty(N_MODES)
            denom = m + r + 2.0 * k * abs(u0 + v) + coupling * r0
            for j in range(N_MODES):
                pm = 0.5 * (old[2 + 2*j] + new[2 + 2*j])
                lp, lq = adj[2 + 2*j], adj[3 + 2*j]
                lm[j] = 2.0 * lp + 2.0 * ww[b, n, j] * lq + gp
                gw[b, n, j] = 2.0 * pm * lq
                prev[2 + 2*j], prev[3 + 2*j] = -lp, lq
                lv += zz[b, n, j] * lm[j]
                denom += coupling * zz[b, n, j]
            implicit = -lv / denom
            gm[b, n] = implicit * (v - old[0] - 0.5 * (ref[b, n] - old[1]))
            gv[b, n] = implicit * (v - vr)
            gk[b, n] = implicit * ((u0 + v)*abs(u0 + v) - (u0 + vr)*abs(u0 + vr))
            gu[b, n] = implicit * 2.0 * k * (abs(u0 + v) - abs(u0 + vr))
            slope_ref = r + 2.0 * k * abs(u0 + vr)
            gr[b, n] -= 0.5 * implicit * (m + slope_ref)
            prev[1] = 0.5 * implicit * (m - slope_ref)
            prev[0] -= implicit * m
            for j in range(N_MODES):
                lm[j] += implicit * coupling
                ga[b, n, j] = lm[j] * old[2 + 2*j]
                gb[b, n, j] = lm[j] * old[3 + 2*j]
                gz[b, n, j] = lm[j] * v
                prev[2 + 2*j] += lm[j] * aa[b, n, j]
                prev[3 + 2*j] += lm[j] * bb[b, n, j]
            adj = prev
        gi[b] = adj
    return gr, gm, gv, gk, gu, ga, gb, gz, gw, gi


if njit is not None:
    _forward = njit(cache=True, fastmath=False)(_forward)
    _backward = njit(cache=True, fastmath=False)(_backward)


class _LoadedFlow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ref, mass, resistance, quadratic, mean, aa, bb, zz, ww,
                initial, coupling, r0):
        arrays = tuple(np.ascontiguousarray(t.detach().numpy())
                       for t in (ref, mass, resistance, quadratic, mean, aa, bb, zz, ww, initial))
        history, pressure = _forward(*arrays, coupling, r0)
        ctx.save_for_backward(ref, mass, resistance, quadratic, mean, aa, bb, zz, ww,
                              torch.from_numpy(history))
        ctx.coupling, ctx.r0 = coupling, r0
        return (torch.from_numpy(history[:, 1:, 0].copy()), torch.from_numpy(pressure),
                torch.from_numpy(history[:, -1].copy()))

    @staticmethod
    @once_differentiable
    def backward(ctx, gflow, gpressure, gfinal):
        saved = tuple(np.ascontiguousarray(t.detach().numpy()) for t in ctx.saved_tensors)
        shape = saved[0].shape
        grads = (np.zeros(shape) if gflow is None else np.ascontiguousarray(gflow.numpy()),
                 np.zeros(shape) if gpressure is None else np.ascontiguousarray(gpressure.numpy()),
                 np.zeros((shape[0], 8)) if gfinal is None else np.ascontiguousarray(gfinal.numpy()))
        result = _backward(*grads, *saved, ctx.coupling, ctx.r0)
        return (*[torch.from_numpy(a) for a in result], None, None)


def load_coefficients(tracks, sample_rate: float, length_cm: float):
    """Midpoint coefficients for positive-real modal pressure states."""
    h = 0.5 / sample_rate
    omega = torch.stack([2.0 * math.pi * f for f, _ in tracks[:N_MODES]], -1)
    gamma = torch.stack([2.0 * math.pi * bw for _, bw in tracks[:N_MODES]], -1)
    residue = 2.0 * RHO * C_SOUND**2 / (TRACT_AREA_CM2 * length_cm)
    w = h * omega
    a = 1.0 / (1.0 + h * gamma + w.square())
    return a, -w * a, h * residue * a, w


def loaded_flow(ref, mass, resistance, quadratic, mean, coefficients, initial,
                coupling: float, r0: float):
    """Return AC flow, midpoint feedback pressure, and differentiable final state.

    mass is 2*M/dt. Inputs are float64 CPU tensors; coefficients have shape
    (B,N,3), other tracks (B,N), initial state (B,8).
    """
    require_backend()
    tensors = (ref, mass, resistance, quadratic, mean, *coefficients, initial)
    if any(t.device.type != "cpu" or t.dtype != torch.float64 for t in tensors):
        raise ValueError("loaded_flow requires float64 CPU tensors")
    if (ref.ndim != 2 or ref.shape[1] == 0
            or len(coefficients) != 4
            or any(t.shape != ref.shape for t in tensors[1:5])
            or any(t.shape != (*ref.shape, N_MODES) for t in coefficients)
            or initial.shape != (ref.shape[0], 2 + 2*N_MODES)):
        raise ValueError("Invalid loaded_flow track/state shapes")
    if not math.isfinite(coupling) or not 0 <= coupling <= 1 or not math.isfinite(r0) or r0 < 0:
        raise ValueError("Loaded flow coupling must be in [0,1] and resistance nonnegative")
    if (any(not bool(torch.isfinite(t).all()) for t in tensors)
            or bool((mass <= 0).any()) or bool((resistance < 0).any())
            or bool((quadratic < 0).any())):
        raise ValueError("Loaded flow requires finite states and positive inertance/passive resistance")
    return _LoadedFlow.apply(*tensors, coupling, r0)


class ImpedanceLoadedSource:
    """No trainable parameters; instance configuration is owned by VoiceEngine."""
    def __init__(self, sample_rate: float, length_cm: float, coupling: float):
        require_backend()
        if not math.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("Loaded source requires a positive finite sample rate")
        if not math.isfinite(length_cm) or length_cm <= 0:
            raise ValueError("Loaded source requires positive finite tract_length_cm")
        self.fs, self.length_cm, self.coupling = sample_rate, length_cm, coupling
        self.r0 = 0.02 * RHO * C_SOUND / TRACT_AREA_CM2

    def __call__(self, reference, flow_scale, f0, area, p_sub, tracks, state):
        if reference.device.type != "cpu":
            raise ValueError("The loaded glottal source currently supports CPU only")
        ref, area, p_sub = reference.double(), area.double(), p_sub.double()
        initial = state.get("flow")
        if initial is None:
            initial = ref.new_zeros((ref.shape[0], 2 + 2*N_MODES))
        mass = 2.0 * self.fs * RHO * GLOTTAL_LENGTH_CM / area
        resistance = torch.full_like(ref, VISCOUS_RESISTANCE)
        quadratic = 0.5 * RHO / area.square()
        mean = area * torch.sqrt(2.0 * CMH2O * p_sub.clamp_min(0.0) / RHO + 1e-12)
        flow, pressure, final = loaded_flow(
            ref, mass, resistance, quadratic, mean,
            load_coefficients(tracks, self.fs, self.length_cm),
            initial, self.coupling, self.r0)
        previous = torch.cat((initial[:, :1], flow[:, :-1]), -1)
        du = (flow - previous) * self.fs / (flow_scale.double() * f0.double().clamp_min(1.0))
        return dict(du=du.to(reference.dtype), flow=flow, load_pressure=pressure,
                    glottal_flow=flow + mean,
                    state={"flow": final})
