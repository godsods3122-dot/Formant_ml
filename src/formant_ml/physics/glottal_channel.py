"""Conservative, compressible AP-strip chamber/throat reference model (SI).

This is not moving-boundary 3D Navier--Stokes. Cell mass, axial momentum and
total gas energy are retained, including behind a fully closed throat. Storage
geometry and conductive aperture are deliberately different inputs. A residual
cavity must be supplied as geometry, not inferred from a numerical area floor.

Mass and total energy occupy cells; axial momentum occupies staggered dual
volumes. The axial cuts are fixed in the laboratory frame. Moving sidewall volume must
come from a closed geometric construction. This module does not manufacture
that construction from a vocal-fold gap profile.

Viscous throat drag transfers kinetic energy to internal gas energy; there is
no heat sink. Thermodynamic transport and acoustic pressure work are reported
separately. The midpoint finite-volume discretization is a low-Mach reference
approximation, not a shock-capturing or validated contact-wetting model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from collections.abc import Sequence

import numpy as np

from . import air


def _array(value, shape=None, *, name, nonnegative=False):
    out = np.array(value, dtype=np.float64, copy=True)
    if shape is not None and out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    if nonnegative and np.any(out < 0):
        raise ValueError(f"{name} must be nonnegative")
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class ChannelGeometry:
    """Fixed-cut channel geometry; arrays are (AP strips, SI cells/faces).

    ``storage_volume_m3`` is actual declared gas storage, not clipped gap area.
    ``storage_face_area_m2`` includes the gas side of blocked throat plates.
    ``open_area_m2`` alone connects neighbors or reservoirs. ``face_gap_m`` is
    the actual full conductive slit gap, not residual cavity depth.

    The caller owns the sidewall pressure footprint and closed-volume sweep.
    ``provenance`` must identify the geometry and any unresolved cavity.
    """

    lengths_m: np.ndarray
    widths_m: np.ndarray
    storage_volume_m3: np.ndarray
    storage_face_area_m2: np.ndarray
    open_area_m2: np.ndarray
    face_gap_m: np.ndarray
    provenance: str

    def __post_init__(self):
        lengths = _array(self.lengths_m, name="lengths_m")
        widths = _array(self.widths_m, name="widths_m")
        if (lengths.ndim != 1 or widths.ndim != 1 or not lengths.size
                or not widths.size or np.any(lengths <= 0) or np.any(widths <= 0)):
            raise ValueError("cell lengths and strip widths must be positive vectors")
        shape = (len(widths), len(lengths))
        face_shape = (shape[0], shape[1] + 1)
        for name, value, expected in (
            ("lengths_m", lengths, lengths.shape),
            ("widths_m", widths, widths.shape),
            ("storage_volume_m3", self.storage_volume_m3, shape),
            ("storage_face_area_m2", self.storage_face_area_m2, face_shape),
            ("open_area_m2", self.open_area_m2, face_shape),
            ("face_gap_m", self.face_gap_m, face_shape),
        ):
            object.__setattr__(
                self, name, _array(value, expected, name=name, nonnegative=True)
            )
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("explicit channel geometry/cavity provenance is required")
        if np.any(self.open_area_m2 > self.storage_face_area_m2):
            raise ValueError("open throat area exceeds the declared storage face")
        if not np.allclose(self.open_area_m2,
                           widths[:, None] * self.face_gap_m, rtol=1e-12, atol=0):
            raise ValueError("open area must equal strip width times actual slit gap")
        dry = self.storage_volume_m3 == 0
        if np.any(dry) and not (
            np.all(dry) and not np.any(self.storage_face_area_m2)
            and not np.any(self.open_area_m2)
        ):
            raise ValueError("mixed dry/wet topology needs an explicit filling model")

    @property
    def shape(self):
        return self.storage_volume_m3.shape

    @property
    def dry(self):
        return not np.any(self.storage_volume_m3)


@dataclass(frozen=True)
class Reservoir:
    """Gauge pressure and incoming gas temperature at a fixed external port."""

    pressure_pa: float
    temperature_k: float = 310.15

    def __post_init__(self):
        if (not math.isfinite(self.pressure_pa)
                or self.pressure_pa + air.P_ATM <= 0):
            raise ValueError("reservoir absolute pressure must be positive and finite")
        if not math.isfinite(self.temperature_k) or self.temperature_k <= 0:
            raise ValueError("reservoir temperature must be positive and finite")


@dataclass(frozen=True)
class ChannelConfig:
    properties: air.AirProps = field(default_factory=air.props)
    viscosity: bool = True
    exit_loss: float = 0.0
    courant: float = 0.35
    max_mach: float = 0.3
    tolerance: float = 1e-14
    max_evaluations: int = 600

    def __post_init__(self):
        for name in ("rho", "c", "mu", "M", "gamma"):
            value = getattr(self.properties, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid shared air property {name}")
        if self.properties.gamma <= 1:
            raise ValueError("calorically perfect gas gamma must exceed one")
        if not math.isfinite(self.exit_loss) or self.exit_loss < 0:
            raise ValueError("exit_loss must be a finite nonnegative coefficient")
        if not 0 < self.courant <= 0.5:
            raise ValueError("courant must be in (0, 0.5]")
        if not 0 < self.max_mach < 1 / self.properties.gamma:
            raise ValueError("max_mach must keep reflecting wall pressures positive")
        if not 0 < self.tolerance < 1e-4 or self.max_evaluations < 1:
            raise ValueError("invalid nonlinear convergence settings")


@dataclass(frozen=True)
class ChannelState:
    owner: object
    geometry: ChannelGeometry
    mass_kg: np.ndarray
    momentum_kg_m_s: np.ndarray
    energy_j: np.ndarray
    time_s: float
    input_mass_kg: float
    input_shifted_energy_j: float
    wall_work_j: float


@dataclass(frozen=True)
class ChannelStep:
    time_s: float
    inlet_flow_m3_s: np.ndarray
    outlet_flow_m3_s: np.ndarray
    inlet_mass_kg_s: np.ndarray
    outlet_mass_kg_s: np.ndarray
    inlet_temperature_k: np.ndarray
    outlet_temperature_k: np.ndarray
    wall_pressure_pa: np.ndarray
    wall_work_j: float
    inlet_pressure_work_j: float
    outlet_pressure_work_j: float
    inlet_thermal_transport_j: float
    outlet_thermal_transport_j: float
    viscous_heat_j: float
    reflection_heat_j: float
    mass_residual_kg: float
    energy_residual_j: float
    iterations: int
    residual: float


class GlottalChannel:
    """Stateful compressible strip model with all-or-nothing midpoint steps."""

    def __init__(self, geometry: ChannelGeometry, config: ChannelConfig | None = None,
                 *, pressure_pa=0.0, temperature_k=None, velocity_m_s=0.0):
        self.config = config if config is not None else ChannelConfig()
        self.geometry = geometry
        self._owner = object()
        self._R = air.R_GAS / self.config.properties.M
        self._gamma = self.config.properties.gamma
        self._cv = self._R / (self._gamma - 1)
        self._temperature = self.config.properties.T_c + 273.15
        self._h_reference = (self._cv + self._R) * self._temperature
        T = self._temperature if temperature_k is None else float(temperature_k)
        reservoir = Reservoir(float(pressure_pa), T)
        rho = (air.P_ATM + reservoir.pressure_pa) / (self._R * T)
        velocity = np.broadcast_to(
            np.asarray(velocity_m_s, float), geometry.open_area_m2.shape
        )
        if not np.isfinite(velocity).all():
            raise ValueError("initial gas velocity must be finite")
        c = math.sqrt(self._gamma * self._R * T)
        if np.any(np.abs(velocity) > self.config.max_mach * c):
            raise ValueError("initial velocity exceeds low-Mach validity")
        self._mass = rho * geometry.storage_volume_m3
        self._momentum = self._dual_mass(self._mass) * velocity
        self._energy = self._mass * self._cv * T + self._kinetic(self._mass, velocity)
        self.time_s = 0.0
        self.input_mass_kg = 0.0
        self.input_shifted_energy_j = 0.0
        self.wall_work_j = 0.0

    def snapshot(self):
        return ChannelState(
            self._owner, self.geometry,
            _array(self._mass, name="mass"), _array(self._momentum, name="momentum"),
            _array(self._energy, name="energy"), self.time_s,
            self.input_mass_kg, self.input_shifted_energy_j, self.wall_work_j,
        )

    def restore(self, state: ChannelState):
        if not isinstance(state, ChannelState) or state.owner is not self._owner:
            raise ValueError("channel snapshot belongs to a different instance")
        if (state.geometry.shape != self.geometry.shape
                or state.mass_kg.shape != state.geometry.shape
                or state.energy_j.shape != state.geometry.shape
                or state.momentum_kg_m_s.shape != state.geometry.open_area_m2.shape):
            raise ValueError("channel snapshot shape mismatch")
        self._primitive(state.mass_kg, state.momentum_kg_m_s, state.energy_j,
                        state.geometry.storage_volume_m3)
        if not all(math.isfinite(x) for x in (
            state.time_s, state.input_mass_kg, state.input_shifted_energy_j,
            state.wall_work_j,
        )) or state.time_s < 0:
            raise ValueError("invalid channel snapshot ledger")
        self.geometry = state.geometry
        self._mass = state.mass_kg.copy()
        self._momentum = state.momentum_kg_m_s.copy()
        self._energy = state.energy_j.copy()
        self.time_s = state.time_s
        self.input_mass_kg = state.input_mass_kg
        self.input_shifted_energy_j = state.input_shifted_energy_j
        self.wall_work_j = state.wall_work_j

    @staticmethod
    def _dual_mass(mass):
        return np.concatenate(
            (mass[:, :1] / 2, (mass[:, :-1] + mass[:, 1:]) / 2,
             mass[:, -1:] / 2), axis=1,
        )

    @staticmethod
    def _kinetic(mass, velocity):
        return mass * (velocity[:, :-1] ** 2 + velocity[:, 1:] ** 2) / 4

    def _primitive(self, mass, momentum, energy, volume):
        if not all(np.isfinite(x).all() for x in (mass, momentum, energy, volume)):
            raise ValueError("nonfinite channel state")
        if not np.any(volume):
            if np.any(mass) or np.any(momentum) or np.any(energy):
                raise ValueError("dry geometry cannot contain gas mass or energy")
            z = np.zeros_like(volume)
            return z, np.zeros_like(momentum), z, z, z
        if np.any(mass <= 0) or np.any(volume <= 0):
            raise ValueError("gas mass and storage volume must remain positive")
        u = momentum / self._dual_mass(mass)
        internal = energy - self._kinetic(mass, u)
        if np.any(internal <= 0):
            raise ValueError("gas internal energy must remain positive")
        rho = mass / volume
        p = (self._gamma - 1) * internal / volume
        c = np.sqrt(self._gamma * p / rho)
        return rho, u, p, internal / mass, c

    def recommend_timestep(self, geometry=None):
        g = self.geometry if geometry is None else geometry
        if self.geometry.dry:
            return math.inf
        rho, u, p, e, c = self._primitive(
            self._mass, self._momentum, self._energy,
            self.geometry.storage_volume_m3,
        )
        area = g.storage_face_area_m2[:, :-1] + g.storage_face_area_m2[:, 1:]
        speed = np.maximum(np.abs(u[:, :-1]), np.abs(u[:, 1:]))
        rate = (c + speed) * area / g.storage_volume_m3
        return self.config.courant / float(rate.max())

    @staticmethod
    def _boundary(reservoir, strips):
        values = (reservoir,) * strips if isinstance(reservoir, Reservoir) else tuple(reservoir)
        if len(values) != strips or not all(isinstance(v, Reservoir) for v in values):
            raise ValueError("provide one Reservoir per AP strip, or one common Reservoir")
        return (np.array([v.pressure_pa for v in values]),
                np.array([v.temperature_k for v in values]))

    def _fluxes(self, mass, momentum, energy, g, inlet, outlet):
        rho, u, p, e, c = self._primitive(
            mass, momentum, energy, g.storage_volume_m3
        )
        c_face = np.concatenate(
            (c[:, :1], np.minimum(c[:, :-1], c[:, 1:]), c[:, -1:]), axis=1
        )
        if np.any(np.abs(u) > self.config.max_mach * c_face):
            raise ValueError("channel exceeds its declared low-Mach validity regime")
        n_strip, n_cell = g.shape
        inlet_pressure, inlet_temperature = inlet
        outlet_pressure, outlet_temperature = outlet
        q = g.open_area_m2 * u
        pf = np.empty_like(q)
        pf[:, 0] = air.P_ATM + inlet_pressure
        pf[:, -1] = air.P_ATM + outlet_pressure
        pf[:, 1:-1] = (p[:, :-1] + p[:, 1:]) / 2

        rho_up = np.empty_like(q)
        e_up = np.empty_like(q)
        positive = q[:, 1:-1] >= 0
        rho_up[:, 1:-1] = np.where(positive, rho[:, :-1], rho[:, 1:])
        e_up[:, 1:-1] = np.where(positive, e[:, :-1], e[:, 1:])
        rho_out_left = rho[:, 0] * (pf[:, 0] / p[:, 0]) ** (1 / self._gamma)
        rho_out_right = rho[:, -1] * (pf[:, -1] / p[:, -1]) ** (1 / self._gamma)
        rho_up[:, 0] = np.where(
            q[:, 0] >= 0, pf[:, 0] / (self._R * inlet_temperature), rho_out_left
        )
        rho_up[:, -1] = np.where(
            q[:, -1] <= 0, pf[:, -1] / (self._R * outlet_temperature), rho_out_right
        )
        e_up[:, 0] = np.where(
            q[:, 0] >= 0, self._cv * inlet_temperature,
            pf[:, 0] / ((self._gamma - 1) * rho_out_left)
        )
        e_up[:, -1] = np.where(
            q[:, -1] <= 0, self._cv * outlet_temperature,
            pf[:, -1] / ((self._gamma - 1) * rho_out_right)
        )
        mdot = rho_up * q
        advected = mdot * (e_up + 0.5 * u ** 2)
        total_flux = advected + pf * q

        hydraulic_length = np.r_[
            g.lengths_m[0] / 2,
            (g.lengths_m[:-1] + g.lengths_m[1:]) / 2,
            g.lengths_m[-1] / 2,
        ]
        resistance = np.zeros_like(q)
        opened = g.open_area_m2 > 0
        if self.config.viscosity:
            numerator = np.broadcast_to(
                12 * self.config.properties.mu * hydraulic_length
                / g.widths_m[:, None], q.shape
            )
            resistance[opened] = numerator[opened] / g.face_gap_m[opened] ** 3
        drop = resistance * q
        if self.config.exit_loss:
            for face in (0, -1):
                # Loss is on an exiting jet only, never twice on one stream.
                exiting = q[:, face] < 0 if face == 0 else q[:, face] > 0
                sel = exiting & opened[:, face]
                speed = q[sel, face] / g.open_area_m2[sel, face]
                drop[sel, face] += (
                    0.5 * self.config.exit_loss * rho_up[sel, face]
                    * speed * np.abs(speed)
                )

        dp = np.concatenate(
            (pf[:, :1] - p[:, :1], p[:, :-1] - p[:, 1:],
             p[:, -1:] - pf[:, -1:]), axis=1,
        )
        center_momentum_flux = (
            (mdot[:, :-1] + mdot[:, 1:]) * (u[:, :-1] + u[:, 1:]) / 4
        )
        momentum_in = np.concatenate(
            ((mdot[:, 0] * u[:, 0])[:, None], center_momentum_flux), axis=1
        )
        momentum_out = np.concatenate(
            (center_momentum_flux, (mdot[:, -1] * u[:, -1])[:, None]), axis=1
        )
        blocked = g.storage_face_area_m2 - g.open_area_m2
        rho_face = self._dual_mass(mass) / self._dual_mass(g.storage_volume_m3)
        reflection_force = 2 * rho_face * c_face * blocked * u
        rhs_momentum = (
            momentum_in - momentum_out + (dp - drop) * g.open_area_m2
            - reflection_force
        )
        reflection = reflection_force * u
        shifted_flux = total_flux - self._h_reference * mdot
        gauge_work = (pf - air.P_ATM) * q
        thermal_flux = shifted_flux - gauge_work
        return {
            "mass": mdot[:, :-1] - mdot[:, 1:],
            "momentum": rhs_momentum,
            "energy": total_flux[:, :-1] - total_flux[:, 1:],
            "q": q, "mdot": mdot, "p": p,
            "temperature": e_up / self._cv,
            "shifted_flux": shifted_flux, "gauge_work": gauge_work,
            "thermal_flux": thermal_flux,
            "viscous_power": float(np.sum(drop * q)),
            "reflection_power": float(reflection.sum()),
        }

    def advance(self, dt_s, inlet: Reservoir | Sequence[Reservoir],
                outlet: Reservoir | Sequence[Reservoir], *,
                geometry: ChannelGeometry | None = None):
        """Advance atomically with constant port pressure over the interval.

        Moving geometry is a supplied fixed-cut closed-volume construction.
        The returned per-cell pressure is conjugate to its volume change.
        The anatomy adapter, not this solver, must map that work to real walls.
        """
        dt = float(dt_s)
        g1 = self.geometry if geometry is None else geometry
        g0 = self.geometry
        inlet = self._boundary(inlet, g0.shape[0])
        outlet = self._boundary(outlet, g0.shape[0])
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt_s must be finite and positive")
        if (g1.shape != g0.shape or not np.array_equal(g1.lengths_m, g0.lengths_m)
                or not np.array_equal(g1.widths_m, g0.widths_m)):
            raise ValueError("channel axial/AP cuts must stay fixed")
        if g0.dry or g1.dry:
            if not (g0.dry and g1.dry):
                raise ValueError("wet/dry filling requires explicit retained cavity geometry")
            z = np.zeros(g0.shape[0])
            self.time_s += dt
            self.geometry = g1
            return ChannelStep(self.time_s, z.copy(), z.copy(), z.copy(), z.copy(),
                               inlet[1].copy(), outlet[1].copy(),
                               np.zeros(g0.shape), *([0.0] * 9), 0, 0.0)
        if dt > min(self.recommend_timestep(), self.recommend_timestep(g1)) * (1 + 1e-12):
            raise ValueError("channel timestep exceeds the current acoustic-volume bound")
        gm = ChannelGeometry(
            g0.lengths_m, g0.widths_m,
            (g0.storage_volume_m3 + g1.storage_volume_m3) / 2,
            (g0.storage_face_area_m2 + g1.storage_face_area_m2) / 2,
            (g0.open_area_m2 + g1.open_area_m2) / 2,
            (g0.face_gap_m + g1.face_gap_m) / 2,
            g1.provenance,
        )
        dm_volume = g1.storage_volume_m3 - g0.storage_volume_m3
        m0, j0, E0 = self._mass, self._momentum, self._energy
        rho0, u0, p0, e0, c0 = self._primitive(m0, j0, E0, g0.storage_volume_m3)
        internal0 = m0 * e0
        j_scale = self._dual_mass(m0) * np.concatenate(
            (c0[:, :1], (c0[:, :-1] + c0[:, 1:]) / 2, c0[:, -1:]), axis=1
        )
        shape = g0.shape
        n = m0.size
        n_momentum = j0.size
        initial = np.r_[np.zeros(n), (j0 / j_scale).ravel(), np.zeros(n)]

        def decode(x):
            with np.errstate(over="raise", invalid="raise"):
                mass = m0 * np.exp(x[:n].reshape(shape))
                momentum = j_scale * x[n:n + n_momentum].reshape(j0.shape)
                internal = internal0 * np.exp(x[n + n_momentum:].reshape(shape))
                velocity = momentum / self._dual_mass(mass)
                energy = internal + self._kinetic(mass, velocity)
            return mass, momentum, energy

        def evaluate(x):
            m1, j1, E1 = decode(x)
            f = self._fluxes((m0 + m1) / 2, (j0 + j1) / 2, (E0 + E1) / 2,
                             gm, inlet, outlet)
            residual = np.r_[
                ((m1 - m0 - dt * f["mass"]) / m0).ravel(),
                ((j1 - j0 - dt * f["momentum"]) / j_scale).ravel(),
                ((E1 - E0 - dt * f["energy"] + f["p"] * dm_volume) / E0).ravel(),
            ]
            return residual, f

        x = initial
        residual, flux = evaluate(x)
        evaluations = 1
        error = float(np.max(np.abs(residual)))
        while error > self.config.tolerance:
            if evaluations + 2 * x.size + 1 > self.config.max_evaluations:
                break
            jacobian = np.empty((x.size, x.size))
            # Absolute increments remain resolvable when velocity is near zero.
            increment = 1e-5
            for column in range(x.size):
                offset = np.zeros_like(x)
                offset[column] = increment
                jacobian[:, column] = (
                    evaluate(x + offset)[0] - evaluate(x - offset)[0]
                ) / (2 * increment)
            evaluations += 2 * x.size
            delta = np.linalg.solve(jacobian, -residual)
            scale = min(1.0, 0.5 / max(float(np.max(np.abs(delta))), 0.5))
            improved = False
            for _ in range(12):
                candidate = x + scale * delta
                candidate_residual, candidate_flux = evaluate(candidate)
                evaluations += 1
                candidate_error = float(np.max(np.abs(candidate_residual)))
                if candidate_error < error:
                    x, residual, flux = candidate, candidate_residual, candidate_flux
                    error = candidate_error
                    improved = True
                    break
                scale /= 2
                if evaluations >= self.config.max_evaluations:
                    break
            if not improved:
                break
        if not math.isfinite(error) or error > self.config.tolerance:
            raise RuntimeError(
                f"channel midpoint solve rejected at {self.time_s:g}s: "
                f"residual={error:.3g}, evaluations={evaluations}"
            )
        m1, j1, E1 = decode(x)
        self._primitive(m1, j1, E1, g1.storage_volume_m3)
        pressure = flux["p"] - air.P_ATM
        wall_work = float(np.sum(pressure * dm_volume))
        input_mass = dt * float(np.sum(flux["mdot"][:, 0] - flux["mdot"][:, -1]))
        input_energy = dt * float(np.sum(
            flux["shifted_flux"][:, 0] - flux["shifted_flux"][:, -1]
        ))
        delta_energy = float(np.sum(
            E1 - E0 + air.P_ATM * dm_volume - self._h_reference * (m1 - m0)
        ))
        step = ChannelStep(
            self.time_s + dt,
            flux["q"][:, 0].copy(), flux["q"][:, -1].copy(),
            flux["mdot"][:, 0].copy(), flux["mdot"][:, -1].copy(),
            flux["temperature"][:, 0].copy(), flux["temperature"][:, -1].copy(),
            pressure.copy(), wall_work,
            dt * float(flux["gauge_work"][:, 0].sum()),
            dt * float(flux["gauge_work"][:, -1].sum()),
            dt * float(flux["thermal_flux"][:, 0].sum()),
            dt * float(flux["thermal_flux"][:, -1].sum()),
            dt * flux["viscous_power"], dt * flux["reflection_power"],
            float(np.sum(m1 - m0)) - input_mass,
            delta_energy - input_energy + wall_work, evaluations, error,
        )
        self.geometry = g1
        self._mass, self._momentum, self._energy = m1, j1, E1
        self.time_s += dt
        self.input_mass_kg += input_mass
        self.input_shifted_energy_j += input_energy
        self.wall_work_j += wall_work
        return step

    def diagnostics(self):
        rho, u, p, e, c = self._primitive(
            self._mass, self._momentum, self._energy,
            self.geometry.storage_volume_m3,
        )
        kinetic = float(np.sum(self._kinetic(self._mass, u)))
        return {
            "time_s": self.time_s,
            "mass_kg": self._mass.copy(),
            "momentum_kg_m_s": self._momentum.copy(),
            "pressure_pa": p - air.P_ATM if not self.geometry.dry else p,
            "temperature_k": e / self._cv,
            "velocity_m_s": (u[:, :-1] + u[:, 1:]) / 2,
            "face_velocity_m_s": u,
            "total_energy_j": float(self._energy.sum()),
            "kinetic_energy_j": kinetic,
            "internal_energy_j": float(self._energy.sum()) - kinetic,
            "shifted_energy_j": float(np.sum(
                self._energy + air.P_ATM * self.geometry.storage_volume_m3
                - self._h_reference * self._mass
            )),
            "input_mass_kg": self.input_mass_kg,
            "input_shifted_energy_j": self.input_shifted_energy_j,
            "wall_work_j": self.wall_work_j,
            "geometry_provenance": self.geometry.provenance,
            "assumptions": (
                "fixed axial/AP cuts; independent strips; calorically perfect gas",
                "reference-temperature transport; no thermal wall sink",
                "staggered midpoint low-Mach FV; unresolved reflecting throat plates",
                "cavity footprint and solid mapping are caller-owned",
            ),
        }
