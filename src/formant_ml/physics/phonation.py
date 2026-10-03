"""Finite respiratory supply for the optional coupled forward path (SI).

The reservoir is isothermal with an explicit heat ledger, not a pressure target
controller. Its actual mass and compliant geometric volume are distinct from
reference-density air expenditure. Recoil reuses the legacy curve without
calling its automatically compensated ``lungs.run``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import math

import numpy as np
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar

from . import air, lungs
from .acoustic_domain import AcousticDomain, AcousticSnapshot
from .glottal_channel import ChannelGeometry, ChannelState, GlottalChannel, Reservoir
from .vocal_fold_solid import (
    BilateralControl, FoldControl, PressurePatch, SolidState, VocalFoldSolid, _slice_curve,
)


@dataclass(frozen=True)
class LungConfig:
    frc_volume_m3: float = 2e-3
    vital_capacity_m3: float = lungs.VC * 1e-3
    initial_fraction: float = 0.6
    initial_muscle_pressure_pa: float = 0.0
    properties: air.AirProps = field(default_factory=air.props)

    def __post_init__(self):
        for name in ("frc_volume_m3", "vital_capacity_m3"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.frc_volume_m3 <= 0.25 * self.vital_capacity_m3:
            raise ValueError("the lower recoil-curve volume must remain positive")
        if not -0.25 <= self.initial_fraction <= 1:
            raise ValueError("initial_fraction lies outside the existing recoil curve")
        if not math.isfinite(self.initial_muscle_pressure_pa):
            raise ValueError("muscle pressure must be finite")


@dataclass(frozen=True)
class LungState:
    owner: object
    time_s: float
    volume_m3: float
    mass_kg: float
    muscle_pressure_pa: float
    exhaled_mass_kg: float
    muscle_work_j: float
    heat_into_gas_j: float
    port_pressure_work_j: float


@dataclass(frozen=True)
class LungStep:
    time_s: float
    pressure_pa: float
    work_pressure_pa: float
    outlet_volume_flow_m3_s: float
    outlet_mass_flow_kg_s: float
    geometric_volume_change_m3: float
    reference_air_expenditure_m3: float
    recoil_energy_change_j: float
    gas_internal_energy_change_j: float
    muscle_work_j: float
    heat_into_gas_j: float
    outlet_enthalpy_j: float
    port_pressure_work_j: float
    first_law_residual_j: float


class LungReservoir:
    """Quasistatic compliant isothermal alveolar reservoir, with atomic steps.

    The mechanical law is p_alveolar = p_recoil(V) + p_muscle. The ideal-gas
    mass relation is solved simultaneously, so a changing muscle pressure does
    not silently create mass and net flow is not confused with wall motion.
    Nominal volumes are reference assumptions, not patient measurements.
    """

    def __init__(self, config: LungConfig | None = None):
        self.config = LungConfig() if config is None else config
        self._owner = object()
        self._R = air.R_GAS / self.config.properties.M
        self._T = self.config.properties.T_c + 273.15
        self._cv = self._R / (self.config.properties.gamma - 1)
        self._cp = self._cv + self._R
        self.time_s = 0.0
        self.volume_m3 = (
            self.config.frc_volume_m3
            + self.config.initial_fraction * self.config.vital_capacity_m3
        )
        self.muscle_pressure_pa = self.config.initial_muscle_pressure_pa
        self.mass_kg = self._mass_at(self.volume_m3, self.muscle_pressure_pa)
        self.exhaled_mass_kg = 0.0
        self.muscle_work_j = 0.0
        self.heat_into_gas_j = 0.0
        self.port_pressure_work_j = 0.0

    @property
    def minimum_volume_m3(self):
        return self.config.frc_volume_m3 - 0.25 * self.config.vital_capacity_m3

    @property
    def maximum_volume_m3(self):
        return self.config.frc_volume_m3 + self.config.vital_capacity_m3

    def _fraction(self, volume):
        f = (volume - self.config.frc_volume_m3) / self.config.vital_capacity_m3
        if not -0.25 - 1e-14 <= f <= 1 + 1e-14:
            raise ValueError("lung volume lies outside the declared recoil curve")
        return f

    def recoil_pressure_pa(self, volume_m3=None):
        volume = self.volume_m3 if volume_m3 is None else float(volume_m3)
        return lungs.CM_H2O * lungs.relaxation_pressure(self._fraction(volume))

    def recoil_energy_j(self, volume_m3=None):
        volume = self.volume_m3 if volume_m3 is None else float(volume_m3)
        f = self._fraction(volume)
        integral = (
            35 * f ** 2.6 / 2.6 if f >= 0
            else 20 * (-f) ** 2.3 / (0.25 ** 1.3 * 2.3)
        )
        return lungs.CM_H2O * self.config.vital_capacity_m3 * integral

    def pressure_pa(self):
        return self.recoil_pressure_pa() + self.muscle_pressure_pa

    def _mass_at(self, volume, muscle_pressure):
        pressure = air.P_ATM + self.recoil_pressure_pa(volume) + muscle_pressure
        if pressure <= 0 or not math.isfinite(pressure):
            raise ValueError("alveolar absolute pressure must remain positive")
        return pressure * volume / (self._R * self._T)

    def snapshot(self):
        return LungState(
            self._owner, self.time_s, self.volume_m3, self.mass_kg,
            self.muscle_pressure_pa, self.exhaled_mass_kg, self.muscle_work_j,
            self.heat_into_gas_j, self.port_pressure_work_j,
        )

    def restore(self, state: LungState):
        if not isinstance(state, LungState) or state.owner is not self._owner:
            raise ValueError("lung snapshot belongs to a different instance")
        fields = (
            "time_s", "volume_m3", "mass_kg", "muscle_pressure_pa",
            "exhaled_mass_kg", "muscle_work_j", "heat_into_gas_j",
            "port_pressure_work_j",
        )
        if any(not math.isfinite(getattr(state, name)) for name in fields):
            raise ValueError("lung snapshot contains nonfinite state")
        expected_mass = self._mass_at(state.volume_m3, state.muscle_pressure_pa)
        if state.time_s < 0 or not math.isclose(expected_mass, state.mass_kg, rel_tol=1e-12):
            raise ValueError("lung snapshot violates time or mass/pressure equilibrium")
        for name in fields:
            setattr(self, name, getattr(state, name))

    def advance(self, dt_s, outlet_mass_flow_kg_s, *, muscle_pressure_pa=None,
                inlet_temperature_k=None):
        """Spend actual signed mass; controls specify muscle pressure, not p_target."""
        temperature = self._T if inlet_temperature_k is None else inlet_temperature_k
        return self.advance_flows(
            dt_s, ((outlet_mass_flow_kg_s, temperature),),
            muscle_pressure_pa=muscle_pressure_pa,
        )

    def advance_flows(self, dt_s, mass_temperature_flows, *, muscle_pressure_pa=None):
        """Multiple simultaneous signed exchanges, including returning warm gas."""
        dt = float(dt_s)
        flows = tuple((float(m), float(T)) for m, T in mass_temperature_flows)
        if not flows or any(not math.isfinite(m) or not math.isfinite(T) or T <= 0
                            for m, T in flows):
            raise ValueError("lung exchanges require finite mass rates and positive temperatures")
        mass_flow = sum(m for m, _ in flows)
        muscle = (self.muscle_pressure_pa if muscle_pressure_pa is None
                  else float(muscle_pressure_pa))
        if not all(math.isfinite(x) for x in (dt, mass_flow, muscle)) or dt <= 0:
            raise ValueError("lung dt, mass flow and muscle pressure must be finite; dt > 0")
        expelled = dt * mass_flow
        mass = self.mass_kg - expelled
        lo, hi = self.minimum_volume_m3, self.maximum_volume_m3
        mass_lo, mass_hi = self._mass_at(lo, muscle), self._mass_at(hi, muscle)
        if not mass_lo <= mass <= mass_hi:
            raise ValueError("actual air expenditure exceeds the declared lung volume range")
        volume = brentq(
            lambda v: self._mass_at(v, muscle) - mass, lo, hi,
            xtol=1e-16, rtol=1e-14,
        )
        dv = volume - self.volume_m3
        recoil_change = self.recoil_energy_j(volume) - self.recoil_energy_j()
        recoil_mean = (
            recoil_change / dv if abs(dv) > 1e-14 * self.config.vital_capacity_m3
            else self.recoil_pressure_pa((volume + self.volume_m3) / 2)
        )
        muscle_mean = (muscle + self.muscle_pressure_pa) / 2
        work_pressure = recoil_mean + muscle_mean
        temperatures = [self._T if m >= 0 else T for m, T in flows]
        q = sum(m * self._R * T / (air.P_ATM + work_pressure)
                for (m, _), T in zip(flows, temperatures))
        internal_change = (mass - self.mass_kg) * self._cv * self._T
        enthalpy_out = dt * sum(m * self._cp * T
                               for (m, _), T in zip(flows, temperatures))
        boundary_work = air.P_ATM * dv + recoil_change + muscle_mean * dv
        heat = internal_change + boundary_work + enthalpy_out
        muscle_work = -muscle_mean * dv
        port_work = dt * work_pressure * q
        result = LungStep(
            self.time_s + dt, self.recoil_pressure_pa(volume) + muscle,
            work_pressure, q, mass_flow, dv,
            expelled / self.config.properties.rho, recoil_change,
            internal_change, muscle_work, heat, enthalpy_out, port_work,
            internal_change - heat + boundary_work + enthalpy_out,
        )
        self.volume_m3, self.mass_kg = volume, mass
        self.muscle_pressure_pa = muscle
        self.time_s += dt
        self.exhaled_mass_kg += expelled
        self.muscle_work_j += muscle_work
        self.heat_into_gas_j += heat
        self.port_pressure_work_j += port_work
        return result

    def diagnostics(self):
        return {
            "time_s": self.time_s,
            "volume_m3": self.volume_m3,
            "mass_kg": self.mass_kg,
            "pressure_pa": self.pressure_pa(),
            "muscle_pressure_pa": self.muscle_pressure_pa,
            "recoil_energy_j": self.recoil_energy_j(),
            "gas_internal_energy_j": self.mass_kg * self._cv * self._T,
            "exhaled_mass_kg": self.exhaled_mass_kg,
            "reference_air_expenditure_m3": (
                self.exhaled_mass_kg / self.config.properties.rho
            ),
            "muscle_work_j": self.muscle_work_j,
            "heat_into_gas_j": self.heat_into_gas_j,
            "port_pressure_work_j": self.port_pressure_work_j,
            "assumptions": (
                "existing nominal recoil curve; no automatic target compensation",
                "isothermal compliant ideal-gas reservoir with explicit heat exchange",
                "geometric lung volume differs from reference-density expenditure",
            ),
        }


def _clip_polygon(polygon, coefficients):
    """Convex planar clipping by a*x+b*z+c >= 0."""
    if not len(polygon):
        return polygon
    values = polygon @ coefficients[:2] + coefficients[2]
    output = []
    for index, point in enumerate(polygon):
        next_index = (index + 1) % len(polygon)
        next_point = polygon[next_index]
        a, b = values[index], values[next_index]
        if a >= 0:
            output.append(point)
        if (a < 0 <= b) or (b < 0 <= a):
            output.append(point + (next_point - point) * (a / (a - b)))
    result = np.asarray(output, dtype=float).reshape(-1, 2)
    if not len(result):
        return result
    tolerance = 64 * np.finfo(float).eps * max(float(np.max(np.abs(result))), 1e-30)
    unique = []
    for point in result:
        if not unique or np.linalg.norm(point - unique[-1]) > tolerance:
            unique.append(point)
    if len(unique) > 1 and np.linalg.norm(unique[-1] - unique[0]) <= tolerance:
        unique.pop()
    return np.asarray(unique).reshape(-1, 2)


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _polygon_area(polygon):
    if len(polygon) < 3:
        return 0.0
    origin = polygon[0]
    return abs(sum(_cross2(polygon[i] - origin, polygon[i + 1] - origin)
                   for i in range(1, len(polygon) - 1))) / 2


def _intersect_triangle(polygon, triangle):
    orientation = np.sign(_cross2(triangle[1] - triangle[0], triangle[2] - triangle[0]))
    if orientation == 0:
        raise ValueError("medial triangle cannot be projected as a lateral graph")
    result = polygon
    for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
        edge = b - a
        coefficients = orientation * np.array([-edge[1], edge[0], -_cross2(edge, a)])
        result = _clip_polygon(result, coefficients)
    return result


def _wet_regions_connected(regions):
    """A shared zero-gap contact line does not connect gas pockets."""
    if not regions:
        return True
    # With complete projected coverage and strictly positive gap everywhere,
    # the rectangular cell cannot contain a contact-separated component.
    if all(
        np.min(poly @ gap[:2] + gap[2])
        > 64 * np.finfo(float).eps * np.max(
            np.abs(poly) @ np.abs(gap[:2]) + abs(gap[2])
        )
        for poly, gap in regions
    ):
        return True
    neighbors = [set() for _ in regions]
    coordinate_scale = max(float(np.max(np.abs(p))) for p, _ in regions)
    tolerance = 64 * np.finfo(float).eps * max(coordinate_scale, 1e-12)
    for i, (first, first_gap) in enumerate(regions):
        for j in range(i):
            second, second_gap = regions[j]
            connected = False
            for a, b in zip(first, np.roll(first, -1, axis=0)):
                direction = b - a
                length = float(np.linalg.norm(direction))
                if length <= tolerance:
                    continue
                unit = direction / length
                for c, d in zip(second, np.roll(second, -1, axis=0)):
                    if (abs(_cross2(unit, c - a)) > tolerance
                            or abs(_cross2(unit, d - a)) > tolerance):
                        continue
                    lo = max(0.0, min(float((c - a) @ unit), float((d - a) @ unit)))
                    hi = min(length, max(float((c - a) @ unit), float((d - a) @ unit)))
                    if hi - lo <= tolerance:
                        continue
                    midpoint = np.r_[a + (lo + hi) * unit / 2, 1.0]
                    gap = min(float(midpoint @ first_gap), float(midpoint @ second_gap))
                    roundoff = 64 * np.finfo(float).eps * max(
                        float(np.abs(midpoint * first_gap).sum()),
                        float(np.abs(midpoint * second_gap).sum()), 1e-30,
                    )
                    if gap > roundoff:
                        connected = True
                        break
                if connected:
                    break
            if connected:
                neighbors[i].add(j)
                neighbors[j].add(i)
    visited, pending = set(), [0]
    while pending:
        current = pending.pop()
        if current not in visited:
            visited.add(current)
            pending.extend(neighbors[current] - visited)
    return len(visited) == len(regions)


@dataclass(frozen=True)
class CellPressurePatch:
    strip: int
    cell: int
    patch: PressurePatch


@dataclass(frozen=True)
class MappedChannel:
    geometry: ChannelGeometry
    patches: tuple[CellPressurePatch, ...]
    projected_coverage_m2: np.ndarray

    def pressure_patches(self, pressure_pa):
        pressure = np.asarray(pressure_pa, float)
        if pressure.shape != self.geometry.shape or not np.isfinite(pressure).all():
            raise ValueError("cell pressure array has wrong shape or nonfinite values")
        return tuple(
            PressurePatch(
                item.patch.side, item.patch.triangle_nodes,
                float(pressure[item.strip, item.cell]), item.patch.barycentric_vertices,
            )
            for item in self.patches
        )

    def swept_cell_volume_m3(self, solid, start, end):
        """Frozen material-patch wall sweep; NOT exact Eulerian remapping."""
        patches = tuple(item.patch for item in self.patches)
        sweep = solid.pressure_patch_sweep(start, end, patches)
        volume = np.zeros(self.geometry.shape)
        for item, outward in zip(self.patches, sweep.outward_swept_volume_m3):
            volume[item.strip, item.cell] -= outward
        return volume


class EulerianGlottis:
    """Exact instantaneous wet-volume clipping between two triangular graphs.

    All AP/SI cut planes are fixed. The positive gap is integrated over the
    overlay of BOTH projected triangulations, including genuine contact
    boundaries. No scalar residual volume is added. Actual recesses must exist
    in the supplied solid mesh if a closing cell needs retained gas storage.

    PressurePatch weights are frozen material coordinates for one solid step.
    Their swept work must be compared against independently recaptured volume
    under time refinement; changing clipping weights are not Simpson-exact.
    AP crossflow is excluded. Optional neck bands reduce each connection to
    its minimum actual cross-sectional area, a stated chamber/throat model.
    """

    def __init__(self, ap_edges_m, si_edges_m, *, provenance, neck_half_width_m=0.0):
        self.ap_edges_m = np.array(ap_edges_m, dtype=float, copy=True)
        self.si_edges_m = np.array(si_edges_m, dtype=float, copy=True)
        for edges in (self.ap_edges_m, self.si_edges_m):
            if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
                    or np.any(np.diff(edges) <= 0)):
                raise ValueError("Eulerian edges must be finite increasing vectors")
            edges.setflags(write=False)
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError("actual solid/cavity geometry provenance is required")
        self.provenance = provenance
        self.neck_half_width_m = float(neck_half_width_m)
        if (not math.isfinite(self.neck_half_width_m) or self.neck_half_width_m < 0
                or self.neck_half_width_m >= np.min(np.diff(self.si_edges_m)) / 2):
            raise ValueError("neck bands must be nonnegative and nonoverlapping")

    def capture(self, solid, *, positions_m=None):
        left, right = solid.surface("left"), solid.surface("right")
        if positions_m is None:
            xyz = (left.coordinates_m, right.coordinates_m)
        else:
            positions = np.asarray(positions_m, float)
            n_left = len(left.coordinates_m)
            if (positions.shape != (n_left + len(right.coordinates_m), 3)
                    or not np.isfinite(positions).all()):
                raise ValueError("geometry positions do not match the real solid")
            xyz = (positions[:n_left], positions[n_left:])
        tri_nodes = (left.triangles, right.triangles)
        triangles = tuple(x[t] for x, t in zip(xyz, tri_nodes))
        projected = tuple(t[:, :, (0, 2)] for t in triangles)
        planes, barycentric_maps = [], []
        for tri in triangles:
            matrix = np.concatenate(
                (tri[:, :, (0, 2)], np.ones((len(tri), 3, 1))), axis=2
            )
            if np.any(np.abs(np.linalg.det(matrix)) < 1e-24):
                raise ValueError("degenerate or overhanging medial graph")
            inverse = np.linalg.inv(matrix)
            planes.append(np.einsum("tij,tj->ti", inverse, tri[:, :, 1]))
            barycentric_maps.append(inverse)
        widths, lengths = np.diff(self.ap_edges_m), np.diff(self.si_edges_m)
        shape = (len(widths), len(lengths))
        volume, coverage = np.zeros(shape), np.zeros(shape)
        patches = []
        regions = [[[] for _ in lengths] for _ in widths]
        for il, ltri in enumerate(projected[0]):
            for ir, rtri in enumerate(projected[1]):
                if (np.any(ltri.max(0) < rtri.min(0))
                        or np.any(rtri.max(0) < ltri.min(0))):
                    continue
                overlap = _intersect_triangle(ltri, rtri)
                if _polygon_area(overlap) == 0:
                    continue
                gap_plane = planes[1][ir] - planes[0][il]
                for strip, (x0, x1) in enumerate(zip(self.ap_edges_m[:-1], self.ap_edges_m[1:])):
                    for cell, (z0, z1) in enumerate(zip(self.si_edges_m[:-1], self.si_edges_m[1:])):
                        poly = overlap
                        for halfplane in (
                            (1, 0, -x0), (-1, 0, x1), (0, 1, -z0), (0, -1, z1),
                        ):
                            poly = _clip_polygon(poly, np.asarray(halfplane))
                        coverage[strip, cell] += _polygon_area(poly)
                        wet = _clip_polygon(poly, gap_plane)
                        if (len(wet) >= 3 and _polygon_area(wet) > 0
                                and np.max(wet @ gap_plane[:2] + gap_plane[2]) > 0):
                            regions[strip][cell].append((wet, gap_plane))
                        for k in range(1, len(wet) - 1):
                            piece = np.asarray([wet[0], wet[k], wet[k + 1]])
                            area = _polygon_area(piece)
                            if area == 0:
                                continue
                            points = np.column_stack((piece, np.ones(3)))
                            gaps = points @ gap_plane
                            if np.max(gaps) <= 0:
                                continue
                            volume[strip, cell] += area * float(np.mean(gaps))
                            for side_index, triangle_index in ((0, il), (1, ir)):
                                B = points @ barycentric_maps[side_index][triangle_index]
                                if np.any(B < -1e-10) or np.any(B > 1 + 1e-10):
                                    raise ValueError("wet clipping left its parent triangle")
                                # Remove only barycentric arithmetic roundoff.
                                B = np.clip(B, 0, 1)
                                B /= B.sum(1)[:, None]
                                determinant = np.linalg.det(B)
                                if abs(determinant) < 1e-14:
                                    continue
                                if determinant < 0:
                                    B = B[[0, 2, 1]]
                                patch = PressurePatch(
                                    "left" if side_index == 0 else "right",
                                    tuple(tri_nodes[side_index][triangle_index]), 0.0, B,
                                )
                                patches.append(CellPressurePatch(strip, cell, patch))
        expected_coverage = widths[:, None] * lengths[None, :]
        if not np.allclose(coverage, expected_coverage, rtol=1e-9, atol=1e-20):
            raise ValueError("fixed channel cuts lack complete single-valued medial coverage")
        if np.any(volume < 0):
            raise ValueError("wet-volume integration produced a negative volume")
        for strip in range(len(widths)):
            for cell in range(len(lengths)):
                if not _wet_regions_connected(regions[strip][cell]):
                    raise ValueError(
                        f"disconnected gas pockets in strip {strip}, cell {cell}; "
                        "component-specific storage or different fixed cuts are required"
                    )

        def area_at(z, strip):
            curves = [_slice_curve(t, z) for t in triangles]
            x0, x1 = self.ap_edges_m[strip:strip + 2]
            if any(curve[0, 0] > x0 + 1e-12 or curve[-1, 0] < x1 - 1e-12 for curve in curves):
                raise ValueError("throat plane lacks full AP coverage")
            knots = np.unique(np.concatenate(
                [np.array([x0, x1])] + [
                    curve[(curve[:, 0] > x0) & (curve[:, 0] < x1), 0]
                    for curve in curves
                ]
            ))
            gap = (np.interp(knots, curves[1][:, 0], curves[1][:, 1])
                   - np.interp(knots, curves[0][:, 0], curves[0][:, 1]))
            crossing = gap[:-1] * gap[1:] < 0
            roots = knots[:-1][crossing] - (
                gap[:-1][crossing] * np.diff(knots)[crossing] / np.diff(gap)[crossing]
            )
            full = np.unique(np.r_[knots, roots])
            positive = np.maximum(np.interp(full, knots, gap), 0)
            return float(np.sum(np.diff(full) * (positive[:-1] + positive[1:]) / 2))

        opened = np.zeros((len(widths), len(lengths) + 1))
        vertex_levels = np.unique(np.concatenate([t[:, :, 2].ravel() for t in triangles]))
        for strip in range(len(widths)):
            for face, z in enumerate(self.si_edges_m):
                opened[strip, face] = area_at(z, strip)
                half = self.neck_half_width_m
                if half and 0 < face < len(lengths):
                    knots = np.unique(np.r_[
                        z - half, z, z + half,
                        vertex_levels[(vertex_levels > z - half) & (vertex_levels < z + half)],
                    ])
                    candidates = [area_at(k, strip) for k in knots]
                    for a, b in zip(knots[:-1], knots[1:]):
                        result = minimize_scalar(
                            lambda level: area_at(level, strip),
                            bounds=(a, b), method="bounded",
                            options={"xatol": 1e-12},
                        )
                        if not result.success:
                            raise RuntimeError("geometric throat-area search did not converge")
                        candidates.append(float(result.fun))
                    opened[strip, face] = min(candidates)
        mean_area = volume / lengths[None, :]
        chamber_faces = np.concatenate(
            (mean_area[:, :1], (mean_area[:, :-1] + mean_area[:, 1:]) / 2,
             mean_area[:, -1:]), axis=1,
        )
        # Storage faces are effective chamber sections, never conductive floors.
        storage_faces = np.maximum(chamber_faces, opened)
        g = ChannelGeometry(
            lengths, widths, volume, storage_faces, opened,
            opened / widths[:, None],
            self.provenance + "; actual wet triangle volumes; fixed Eulerian cuts; "
            f"neck_half_width_m={self.neck_half_width_m}",
        )
        return MappedChannel(g, tuple(patches), coverage)


@dataclass(frozen=True)
class PortWallPatch:
    """Caller-declared wetted physical wall in a compact port plenum.

    The material patch must remain wholly outside the channel SI cuts. Its
    solid-outward swept volume is injected into the associated acoustic port.
    This is a physical solid face, never an artificial fluid cut-plane load.
    """

    end: str
    strip: int
    patch: PressurePatch

    def __post_init__(self):
        if self.end not in ("upstream", "downstream"):
            raise ValueError("plenum wall end must be upstream or downstream")
        if not isinstance(self.strip, int) or self.strip < 0:
            raise ValueError("plenum wall strip must be a nonnegative integer")


@dataclass(frozen=True)
class CouplingConfig:
    subcycles: int = 1
    max_subcycles: int = 64
    max_iterations: int = 50
    pressure_tolerance_pa: float = 1e-4
    position_tolerance_m: float = 1e-11
    volume_tolerance_m3: float = 1e-17
    work_tolerance_j: float = 1e-13
    relaxation: float = 0.6
    maximum_port_pressure_fraction: float = 0.05

    def __post_init__(self):
        if (any(isinstance(v, bool) or not isinstance(v, int) for v in
                (self.subcycles, self.max_subcycles, self.max_iterations))
                or self.subcycles < 1 or self.max_subcycles < self.subcycles
                or self.max_iterations < 1):
            raise ValueError("invalid coupling iteration/subcycle limits")
        for name in ("pressure_tolerance_pa", "position_tolerance_m",
                     "volume_tolerance_m3", "work_tolerance_j"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.relaxation <= 1:
            raise ValueError("coupling relaxation must lie in (0,1]")
        if not 0 < self.maximum_port_pressure_fraction < 1:
            raise ValueError("declare a positive small-signal port validity fraction")


@dataclass(frozen=True)
class SupplyState:
    owner: object
    time_s: float
    flow_m3_s: float
    dissipated_j: float
    input_work_j: float


@dataclass(frozen=True)
class SupplyStep:
    flow_m3_s: float
    inlet_work_j: float
    outlet_work_j: float
    dissipation_j: float
    energy_residual_j: float


class SupplyDuct:
    """Fixed circular incompressible supply inertance and Poiseuille resistance.

    The two ends carry the same signed volume flow. Coefficients use the shared
    reference air density; this is an explicit low-Mach lumped duct, not an
    acoustic delay filter or a second acoustic cavity.
    """

    def __init__(self, length_m, radius_m, *, properties=None):
        if not all(math.isfinite(v) and v > 0 for v in (length_m, radius_m)):
            raise ValueError("supply length and radius must be positive SI dimensions")
        self.properties = air.props() if properties is None else properties
        self.length_m, self.radius_m = float(length_m), float(radius_m)
        area = math.pi * radius_m ** 2
        self.inertance = self.properties.rho * length_m / area
        self.resistance = 8 * math.pi * self.properties.mu * length_m / area ** 2
        self._owner = object()
        self.time_s = self.flow_m3_s = self.dissipated_j = self.input_work_j = 0.0

    def snapshot(self):
        return SupplyState(self._owner, self.time_s, self.flow_m3_s,
                           self.dissipated_j, self.input_work_j)

    def restore(self, state):
        if not isinstance(state, SupplyState) or state.owner is not self._owner:
            raise ValueError("supply snapshot belongs to a different instance")
        values = (state.time_s, state.flow_m3_s, state.dissipated_j, state.input_work_j)
        if not all(math.isfinite(v) for v in values) or state.time_s < 0 or state.dissipated_j < 0:
            raise ValueError("invalid supply snapshot")
        self.time_s, self.flow_m3_s, self.dissipated_j, self.input_work_j = values

    def advance(self, dt_s, inlet_pressure_pa, outlet_pressure_pa):
        dt, pin, pout = float(dt_s), float(inlet_pressure_pa), float(outlet_pressure_pa)
        if not all(math.isfinite(v) for v in (dt, pin, pout)) or dt <= 0:
            raise ValueError("supply step requires finite pressures and positive dt")
        old = self.flow_m3_s
        new = ((self.inertance / dt - self.resistance / 2) * old + pin - pout) / (
            self.inertance / dt + self.resistance / 2
        )
        q = (old + new) / 2
        win, wout = dt * pin * q, dt * pout * q
        dissipation = dt * self.resistance * q * q
        delta = self.inertance * (new * new - old * old) / 2
        result = SupplyStep(q, win, wout, dissipation, delta + dissipation - win + wout)
        self.flow_m3_s = new
        self.time_s += dt
        self.dissipated_j += dissipation
        self.input_work_j += win - wout
        return result

    def diagnostics(self):
        return {
            "length_m": self.length_m, "radius_m": self.radius_m,
            "flow_m3_s": self.flow_m3_s,
            "kinetic_energy_j": self.inertance * self.flow_m3_s ** 2 / 2,
            "dissipated_j": self.dissipated_j, "input_work_j": self.input_work_j,
            "assumption": "fixed circular reference-density incompressible supply duct",
        }


@dataclass(frozen=True)
class PhonationState:
    owner: object
    solid: SolidState
    channel: ChannelState
    lung: LungState
    downstream: AcousticSnapshot
    upstream: AcousticSnapshot | None
    supply: SupplyState | None
    steps: int


@dataclass(frozen=True)
class PhonationStep:
    time_s: float
    iterations: int
    subcycles: int
    pressure_residual_pa: float
    position_residual_m: float
    volume_residual_m3: float
    interface_work_residual_j: float
    inlet_flow_m3_s: np.ndarray
    outlet_flow_m3_s: np.ndarray
    lung_mass_flow_kg_s: float
    downstream_pressure_pa: np.ndarray
    wall_pressure_pa: np.ndarray
    channel_wall_work_j: float
    upstream_wall_work_j: float
    downstream_wall_work_j: float
    solid_pressure_work_j: float
    thermal_transport_j: float


class CouplingError(RuntimeError):
    """A rejected macro interval; every component has been restored."""


class _RefineSubcycles(RuntimeError):
    pass


def _interpolate_control(first, last, fraction):
    def side(a, b):
        values = {}
        for item in fields(FoldControl):
            av, bv = getattr(a, item.name), getattr(b, item.name)
            if isinstance(av, tuple):
                values[item.name] = tuple(
                    (1 - fraction) * x + fraction * y for x, y in zip(av, bv)
                )
            else:
                values[item.name] = (1 - fraction) * av + fraction * bv
        return FoldControl(**values)
    return BilateralControl(side(first.left, last.left), side(first.right, last.right))


class PhonationSystem:
    """Real solid/channel/lung/acoustic feedback on one physical timeline.

    Each predictor/corrector pass restores the whole macro-start snapshot.
    Numerical relaxation only converges the interface equations; it does not
    filter the physical state. Failure, including unsupported pocket topology,
    restores every component. The acoustic field receives mean interval flows
    and returns its own conjugate work pressure.

    Optional subglottal acoustics has two real ports: the passive supply duct
    injects at its lung port, while glottal inflow is withdrawn at its glottis
    ports. Channel inlet pressure is the actual domain pressure, never lung
    pressure added a second time. Additional plenum wall patches are explicit
    caller-specified wetted material faces.
    """

    def __init__(self, solid: VocalFoldSolid, channel: GlottalChannel,
                 mapper: EulerianGlottis, lung: LungReservoir,
                 downstream: AcousticDomain, downstream_ports, *,
                 wall_patches=(), config: CouplingConfig | None = None,
                 upstream: AcousticDomain | None = None, upstream_ports=(),
                 upstream_lung_port=None, supply: SupplyDuct | None = None):
        self.solid, self.channel, self.mapper = solid, channel, mapper
        self.lung, self.downstream = lung, downstream
        self.upstream, self.supply = upstream, supply
        self.upstream_ports = tuple(upstream_ports)
        self.upstream_lung_port = upstream_lung_port
        self.config = CouplingConfig() if config is None else config
        self.downstream_ports = tuple(downstream_ports)
        strips = channel.geometry.shape[0]
        if (len(self.downstream_ports) != strips
                or any(p not in downstream.port_names for p in self.downstream_ports)):
            raise ValueError("provide one known downstream port per AP strip")
        self.wall_patches = tuple(wall_patches)
        if any(w.strip >= strips for w in self.wall_patches):
            raise ValueError("plenum wall references an unknown strip")
        self._left_count = len(solid.surface("left").coordinates_m)
        self._owner = object()
        self.steps = 0
        self._dt = downstream.dt
        self._T = channel.config.properties.T_c + 273.15
        self._R = air.R_GAS / channel.config.properties.M
        if not math.isclose(lung.config.properties.T_c, channel.config.properties.T_c,
                            abs_tol=1e-12):
            raise ValueError("lung and channel must share the reference gas temperature")
        if not math.isclose(downstream.rho_kg_m3, channel.config.properties.rho,
                            rel_tol=1e-10):
            raise ValueError("channel and acoustic domain must share air.props composition")
        if (upstream is None) != (supply is None):
            raise ValueError("an upstream domain requires its physical supply duct")
        if upstream is None and (self.upstream_ports or upstream_lung_port is not None):
            raise ValueError("upstream port names require an actual upstream domain")
        for component in (lung, supply):
            if component is None:
                continue
            properties = (component.config.properties if isinstance(component, LungReservoir)
                          else component.properties)
            if any(not math.isclose(getattr(properties, name),
                                    getattr(channel.config.properties, name), rel_tol=1e-10)
                   for name in ("rho", "c", "M", "gamma", "mu")):
                raise ValueError("lung, supply and channel must share the gas properties")
        if not math.isclose(downstream.sound_speed_m_s, channel.config.properties.c,
                            rel_tol=1e-10):
            raise ValueError("downstream must share the channel sound speed")
        if upstream is not None:
            if (len(self.upstream_ports) != strips
                    or any(p not in upstream.port_names for p in self.upstream_ports)
                    or upstream_lung_port not in upstream.port_names
                    or upstream_lung_port in self.upstream_ports):
                raise ValueError("upstream requires distinct lung and glottis ports")
            if not math.isclose(upstream.dt, downstream.dt, rel_tol=1e-12, abs_tol=0):
                raise ValueError("construct upstream/downstream with the same accepted clock")
            if not math.isclose(upstream.rho_kg_m3, channel.config.properties.rho,
                                rel_tol=1e-10):
                raise ValueError("upstream must share the channel air properties")
            if not math.isclose(upstream.sound_speed_m_s, channel.config.properties.c,
                                rel_tol=1e-10):
                raise ValueError("upstream must share the channel sound speed")
        times = (solid.time_s, channel.time_s, lung.time_s, downstream.time_s)
        if upstream is not None:
            times += (upstream.time_s, supply.time_s)
        if max(times) - min(times) > 1e-13:
            raise ValueError("components must start on a shared physical clock")
        actual = mapper.capture(solid).geometry
        for name in (
            "lengths_m", "widths_m", "storage_volume_m3", "storage_face_area_m2",
            "open_area_m2", "face_gap_m",
        ):
            expected, supplied = getattr(actual, name), getattr(channel.geometry, name)
            if (expected.shape != supplied.shape
                    or not np.allclose(expected, supplied, rtol=1e-12, atol=0)):
                raise ValueError(f"channel geometry {name} does not match the real mapped solid")

    @property
    def dt(self):
        return self._dt

    @property
    def time_s(self):
        return self.downstream.time_s

    def snapshot(self):
        return PhonationState(
            self._owner, self.solid.save_state(), self.channel.snapshot(),
            self.lung.snapshot(), self.downstream.snapshot(),
            None if self.upstream is None else self.upstream.snapshot(),
            None if self.supply is None else self.supply.snapshot(), self.steps,
        )

    def restore(self, state):
        if not isinstance(state, PhonationState) or state.owner is not self._owner:
            raise ValueError("phonation snapshot belongs to a different instance")
        if (not isinstance(state.steps, int) or state.steps < 0
                or (self.upstream is None) != (state.upstream is None)
                or (self.supply is None) != (state.supply is None)):
            raise ValueError("phonation snapshot component/count mismatch")
        previous = self.snapshot()
        try:
            self._restore_components(state)
            times = (self.solid.time_s, self.channel.time_s,
                     self.lung.time_s, self.downstream.time_s)
            if self.upstream is not None:
                times += (self.upstream.time_s, self.supply.time_s)
            if max(times) - min(times) > 1e-13:
                raise ValueError("snapshot components have different physical clocks")
        except (ValueError, RuntimeError, FloatingPointError):
            self._restore_components(previous)
            raise

    def _restore_components(self, state):
        self.solid.reset(state.solid)
        self.channel.restore(state.channel)
        self.lung.restore(state.lung)
        self.downstream.restore(state.downstream)
        if self.upstream is not None:
            self.upstream.restore(state.upstream)
            self.supply.restore(state.supply)
        self.steps = state.steps

    def _port_wall_loads(self, positions, upstream, downstream):
        result = []
        for wall in self.wall_patches:
            offset = 0 if wall.patch.side == "left" else self._left_count
            nodes = np.asarray(wall.patch.triangle_nodes) + offset
            points = np.asarray(wall.patch.barycentric_vertices) @ positions[nodes]
            if wall.end == "upstream":
                if points[:, 2].max() > self.mapper.si_edges_m[0] + 1e-12:
                    raise ValueError("upstream physical wall entered the channel cut region")
                pressure = upstream[wall.strip]
            else:
                if points[:, 2].min() < self.mapper.si_edges_m[-1] - 1e-12:
                    raise ValueError("downstream physical wall entered the channel cut region")
                pressure = downstream[wall.strip]
            result.append(PressurePatch(
                wall.patch.side, wall.patch.triangle_nodes, float(pressure),
                wall.patch.barycentric_vertices,
            ))
        return tuple(result)

    def _check_port_pressure(self, pressure):
        if (not np.isfinite(pressure).all()
                or np.max(np.abs(pressure)) > (
                    air.P_ATM * self.config.maximum_port_pressure_fraction
                )):
            raise ValueError("port pressure exceeds the declared linear-acoustic regime")

    def advance(self, *, control: BilateralControl | None = None,
                muscle_pressure_pa=None):
        initial = self.snapshot()
        target_control = initial.solid.control if control is None else control
        target_muscle = (self.lung.muscle_pressure_pa if muscle_pressure_pa is None
                         else float(muscle_pressure_pa))
        if not math.isfinite(target_muscle):
            raise ValueError("muscle pressure must be finite")
        bound = min(self.channel.recommend_timestep(), self.solid.recommend_timestep())
        subcycles = max(self.config.subcycles, int(math.ceil(self.dt / bound)))
        try:
            while subcycles <= self.config.max_subcycles:
                try:
                    return self._interval(initial, target_control, target_muscle, subcycles)
                except _RefineSubcycles:
                    self.restore(initial)
                    subcycles *= 2
            raise CouplingError("geometric work/timestep subcycle refinement exhausted")
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            self.restore(initial)
            if isinstance(error, CouplingError):
                raise
            raise CouplingError(f"macro step rejected at {self.time_s:g}s: {error}") from error

    def _interval(self, initial, target_control, target_muscle, subcycles):
        dt = self.dt / subcycles
        shape = self.channel.geometry.shape
        strips = shape[0]
        pressure_guess = np.repeat(
            self.channel.diagnostics()["pressure_pa"][None, :, :], subcycles, axis=0
        )
        position_guess = np.repeat(
            initial.solid.positions_m[None, :, :], subcycles, axis=0
        )
        upstream_guess = np.full((subcycles, strips), self.lung.pressure_pa())
        source_in_guess = self.lung.pressure_pa()
        source_out_guess = 0.0
        if self.upstream is not None:
            upstream_guess[:] = [
                self.upstream.pressure_pa(port) for port in self.upstream_ports
            ]
            source_out_guess = self.upstream.pressure_pa(self.upstream_lung_port)
        downstream_guess = np.array([
            self.downstream.pressure_pa(port) for port in self.downstream_ports
        ])
        alpha = self.config.relaxation
        for iteration in range(1, self.config.max_iterations + 1):
            self.restore(initial)
            cell_pressure, positions, upstream_pressure = [], [], []
            q_in, q_out = np.zeros(strips), np.zeros(strips)
            s_up, s_down = np.zeros(strips), np.zeros(strips)
            channel_wall_work = solid_work = wall_up_work = wall_down_work = 0.0
            channel_in_work = channel_out_work = lung_port_work = 0.0
            thermal_transport = expelled_mass = 0.0
            volume_error = 0.0
            for k in range(subcycles):
                self._check_port_pressure(upstream_guess[k])
                self._check_port_pressure(downstream_guess)
                solid_before = self.solid.save_state()
                midpoint = (solid_before.positions_m + position_guess[k]) / 2
                mapped = self.mapper.capture(self.solid, positions_m=midpoint)
                patches = mapped.pressure_patches(pressure_guess[k])
                extra = self._port_wall_loads(
                    midpoint, upstream_guess[k], downstream_guess
                )
                next_control = _interpolate_control(
                    initial.solid.control, target_control, (k + 1) / subcycles
                )
                if dt > self.solid.recommend_timestep(
                    next_control, pressure_patches=patches + extra
                ) * (1 + 1e-12):
                    raise _RefineSubcycles()
                old_volume = self.channel.geometry.storage_volume_m3
                solid_step = self.solid.step(
                    dt, next_control, pressure_patches=patches + extra
                )
                solid_after = self.solid.save_state()
                geometry = self.mapper.capture(self.solid).geometry
                if dt > self.channel.recommend_timestep(geometry) * (1 + 1e-12):
                    raise _RefineSubcycles()
                gas = self.channel.advance(
                    dt,
                    tuple(Reservoir(float(p), self._T) for p in upstream_guess[k]),
                    tuple(Reservoir(float(p), self._T) for p in downstream_guess),
                    geometry=geometry,
                )
                swept = np.zeros(shape)
                for item, s in zip(mapped.patches,
                                   solid_step.patch_sweep.outward_swept_volume_m3):
                    swept[item.strip, item.cell] -= s
                volume_error = max(
                    volume_error,
                    float(np.max(np.abs(geometry.storage_volume_m3 - old_volume - swept))),
                )
                up_s, down_s = np.zeros(strips), np.zeros(strips)
                extra_sweep = solid_step.patch_sweep.outward_swept_volume_m3[len(patches):]
                for wall, s in zip(self.wall_patches, extra_sweep):
                    (up_s if wall.end == "upstream" else down_s)[wall.strip] += s
                s_up += up_s
                s_down += down_s
                wall_up_work -= float(upstream_guess[k] @ up_s)
                wall_down_work -= float(downstream_guess @ down_s)
                rho_up = (air.P_ATM + upstream_guess[k]) / (self._R * self._T)
                flows = list(zip(gas.inlet_mass_kg_s, gas.inlet_temperature_k))
                flows.extend(zip(-rho_up * up_s / dt, np.full(strips, self._T)))
                muscle = (
                    initial.lung.muscle_pressure_pa
                    + (target_muscle - initial.lung.muscle_pressure_pa)
                    * (k + 1) / subcycles
                )
                if self.upstream is None:
                    lung_step = self.lung.advance_flows(
                        dt, flows, muscle_pressure_pa=muscle
                    )
                    upstream_pressure.append(np.full(strips, lung_step.work_pressure_pa))
                    expelled_mass += dt * lung_step.outlet_mass_flow_kg_s
                    lung_port_work += lung_step.port_pressure_work_j
                q_in += gas.inlet_flow_m3_s * dt
                q_out += gas.outlet_flow_m3_s * dt
                channel_wall_work += gas.wall_work_j
                channel_in_work += gas.inlet_pressure_work_j
                channel_out_work += gas.outlet_pressure_work_j
                solid_work += solid_after.work.pressure_j - solid_before.work.pressure_j
                thermal_transport += (
                    gas.inlet_thermal_transport_j - gas.outlet_thermal_transport_j
                )
                cell_pressure.append(gas.wall_pressure_pa)
                positions.append(solid_after.positions_m)
            injection = {}
            for port, volume in zip(self.downstream_ports, q_out + s_down):
                injection[port] = injection.get(port, 0.0) + float(volume / self.dt)
            acoustic_before_work = self.downstream.diagnostics()["input_work_j"]
            acoustic_step = self.downstream.advance(injection)
            acoustic_work = (
                self.downstream.diagnostics()["input_work_j"] - acoustic_before_work
            )
            new_downstream = np.array([
                acoustic_step.work_pressure_pa[port] for port in self.downstream_ports
            ])
            new_pressure = np.asarray(cell_pressure)
            new_positions = np.asarray(positions)
            new_upstream = np.asarray(upstream_pressure)
            source_error = source_work_error = 0.0
            if self.upstream is not None:
                self._check_port_pressure(np.array([source_in_guess, source_out_guess]))
                supply_step = self.supply.advance(
                    self.dt, source_in_guess, source_out_guess
                )
                source_density = (air.P_ATM + source_in_guess) / (self._R * self._T)
                lung_step = self.lung.advance(
                    self.dt, source_density * supply_step.flow_m3_s,
                    muscle_pressure_pa=target_muscle,
                )
                expelled_mass = self.dt * lung_step.outlet_mass_flow_kg_s
                lung_port_work = lung_step.port_pressure_work_j
                upstream_injection = {self.upstream_lung_port: supply_step.flow_m3_s}
                for port, volume in zip(self.upstream_ports, -q_in + s_up):
                    upstream_injection[port] = (
                        upstream_injection.get(port, 0.0) + float(volume / self.dt)
                    )
                old_work = self.upstream.diagnostics()["input_work_j"]
                upstream_step = self.upstream.advance(upstream_injection)
                upstream_work = self.upstream.diagnostics()["input_work_j"] - old_work
                new_upstream = np.repeat(np.array([[
                    upstream_step.work_pressure_pa[port] for port in self.upstream_ports
                ]]), subcycles, axis=0)
                new_source_in = lung_step.work_pressure_pa
                new_source_out = upstream_step.work_pressure_pa[self.upstream_lung_port]
                source_error = max(abs(new_source_in - source_in_guess),
                                   abs(new_source_out - source_out_guess))
                source_work_error = max(
                    abs(lung_port_work - supply_step.inlet_work_j),
                    abs(upstream_work + channel_in_work + wall_up_work
                        - supply_step.outlet_work_j),
                    abs(supply_step.energy_residual_j),
                )
            pressure_error = max(
                float(np.max(np.abs(new_pressure - pressure_guess))),
                float(np.max(np.abs(new_upstream - upstream_guess))),
                float(np.max(np.abs(new_downstream - downstream_guess))),
                source_error,
            )
            position_error = float(np.max(np.abs(new_positions - position_guess)))
            work_error = max(
                abs(solid_work - channel_wall_work - wall_up_work - wall_down_work),
                abs(acoustic_work + wall_down_work - channel_out_work),
                (abs(lung_port_work - channel_in_work - wall_up_work)
                 if self.upstream is None else source_work_error),
            )
            converged = (
                pressure_error <= self.config.pressure_tolerance_pa
                and position_error <= self.config.position_tolerance_m
            )
            if converged and volume_error > self.config.volume_tolerance_m3:
                raise _RefineSubcycles()
            if (converged and volume_error <= self.config.volume_tolerance_m3
                    and work_error <= self.config.work_tolerance_j):
                times = (self.solid.time_s, self.channel.time_s,
                         self.lung.time_s, self.downstream.time_s)
                if self.upstream is not None:
                    times += (self.upstream.time_s, self.supply.time_s)
                if max(times) - min(times) > 1e-13:
                    raise CouplingError("component physical clocks diverged")
                self.steps += 1
                return PhonationStep(
                    self.time_s, iteration, subcycles, pressure_error, position_error,
                    volume_error, work_error, q_in / self.dt, q_out / self.dt,
                    expelled_mass / self.dt, new_downstream.copy(), new_pressure[-1].copy(),
                    channel_wall_work, wall_up_work, wall_down_work, solid_work,
                    thermal_transport,
                )
            pressure_guess += alpha * (new_pressure - pressure_guess)
            position_guess += alpha * (new_positions - position_guess)
            upstream_guess += alpha * (new_upstream - upstream_guess)
            downstream_guess += alpha * (new_downstream - downstream_guess)
            if self.upstream is not None:
                source_in_guess += alpha * (new_source_in - source_in_guess)
                source_out_guess += alpha * (new_source_out - source_out_guess)
        raise CouplingError(
            f"interface iteration exhausted: pressure={pressure_error:g} Pa, "
            f"position={position_error:g} m, volume={volume_error:g} m3, "
            f"work={work_error:g} J"
        )
