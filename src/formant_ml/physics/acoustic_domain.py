"""Stateful SI coupling around the existing 3D linear acoustic field solver.

Public units: m, s, Pa, m3/s, kg. The underlying Grid/Sim are CGS; conversion
occurs at this boundary (Pa = 0.1 dyn/cm2, m3/s = 1e-6 cm3/s). This is a
small-signal, fixed-pose reference component, not a turbulence or moving-mesh
solver. Exterior receivers sample the field itself, with no radiation filter.

At step n, pressure and structural displacement are at n*dt, fluid/structural
velocities at (n-1/2)*dt. ``advance`` takes interval volume flows, positive INTO
this domain, and returns pressures at (n+1)*dt plus arithmetic endpoint means
for power-conjugate work. Reverse flow is retained. A glottal coupler supplies
opposite signed flows to its subglottal and supraglottal ports.

Each wall pair is one caller-defined, uniform-displacement patch mode, NOT a
3D tissue solid: m*x'' + r*x' + k*x = p_inside - p_outside, with areal
coefficients in kg/m2, Pa*s/m, Pa/m. Positive displacement removes inner air
volume and adds equal outer volume. Pressure averages use the same physical
areas as flux distribution. Only the corresponding legacy STRUCTURAL sink is
disabled; the distinct viscothermal boundary layer remains once per wetted
face when enabled. No nearest-air or nearest-surface pairing is inferred.

The closed, BL-free lossless-air/passive-pair scheme uses leapfrog with centered
structural damping. Its modified staggered energy is reported, including the
cross term, rather than claiming the naive simultaneous quadratic is constant.
The dt bound is conservative: the air graph Gershgorin bound plus the traces
of positive rank-one structural operators bounds the coupled spectral radius.
No state clamp, area floor, or physical cut-volume inflation is applied.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch

from . import fdtd3d
from .airway_geometry import _mask, validate_grid
from .tract3d import Grid, TISSUE


@dataclass(frozen=True)
class VolumeFlowPort:
    """Source support and optional nonnegative volume-density profile.

    ``mask`` is a boolean Grid-shaped air mask. ``density_weights`` has the same
    shape; its units cancel. Per-cell volume fraction is density*physical volume
    divided by its sum. The exact same fraction observes pressure. No separate
    source or receiver gain exists.
    """
    mask: np.ndarray
    density_weights: np.ndarray | None = None


@dataclass(frozen=True)
class WallPatch:
    index: int
    cell: tuple[int, int, int]
    position_m: tuple[float, float, float]
    outward_normal: tuple[float, float, float]
    area_m2: float
    tissue: str
    exterior: bool


@dataclass(frozen=True)
class WallPair:
    name: str
    inside: tuple[int, ...]
    outside: tuple[int, ...]
    mass_kg_m2: float
    stiffness_pa_m: float
    damping_pa_s_m: float = 0.0


@dataclass(frozen=True)
class StepResult:
    time_s: float
    pressure_pa: dict[str, float]
    work_pressure_pa: dict[str, float]
    wall_flow_m3_s: dict[str, float]


@dataclass(frozen=True)
class AcousticSnapshot:
    """Opaque, instance-specific checkpoint; includes PML, BL, and wall memories."""
    owner: object
    field: tuple[torch.Tensor, ...]
    displacement: torch.Tensor
    velocity: torch.Tensor
    steps: int
    input_volume: dict[str, float]
    input_work: float
    pair_dissipation: float
    bl_removed_volume: float
    local_wall_removed_volume: float


def _patches(sim: fdtd3d.Sim) -> tuple[WallPatch, ...]:
    cells = sim.wall_cell.cpu().numpy()
    areas = sim.wall_face_area_cgs * 1e-4
    return tuple(WallPatch(i, tuple(int(x) for x in np.unravel_index(cell, sim.g.shape)),
                           tuple(sim.wall_pos[i] * 0.01), tuple(sim.wall_dir[i]),
                           float(areas[i]), TISSUE[sim.wall_tis[i]],
                           bool(sim.g.exterior.ravel()[cell])) for i, cell in enumerate(cells))


def wall_patches(grid: Grid) -> tuple[WallPatch, ...]:
    """Enumerate stable patch indices using exactly Sim's wall ordering (SI output).

    Indices are valid only for this unchanged Grid. The caller supplies anatomical
    correspondence; this function does not find or invent a pairing.
    """
    validate_grid(grid)
    sim = fdtd3d.Sim(grid, walls="rigid", boundary_layer=False, sponge_cells=0,
                     device="cpu", dtype=torch.float64, physical_cut_volume=True)
    return _patches(sim)


class AcousticDomain:
    """Persistent multiport domain; ``dt`` is the actual stable step, never requested dt.

    ``max_dt_s`` optionally caps that step for a shared coupling clock. It is
    positive and finite, and never overrides a smaller physical stability bound.
    Unpaired ``walls='soft'`` retain the provisional legacy local approximation,
    including no paired exterior load. ``walls='rigid'`` leaves unpaired faces
    rigid, without disabling explicitly supplied WallPairs. PML and BL preserve
    their legacy approximate discretizations; the exact energy certificate is
    limited to no PML/BL and no unpaired soft walls.
    """

    def __init__(self, grid: Grid, ports: Mapping[str, VolumeFlowPort], *,
                 wall_pairs: Sequence[WallPair] = (), courant: float = 0.5,
                 T_c: float = 35.5, rh: float = 1.0, walls: str = "rigid",
                 boundary_layer: bool = True, sponge_cells: int = 20,
                 sponge_sigma: float | None = None, device: str = "cpu",
                 dtype: torch.dtype = torch.float64, max_dt_s: float | None = None):
        validate_grid(grid)
        if not np.isfinite(courant) or not 0 < courant <= 1:
            raise ValueError("courant must be finite and in (0, 1]")
        if walls not in ("soft", "rigid"):
            raise ValueError("walls must be 'soft' or 'rigid'")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("AcousticDomain requires float32 or float64")
        if max_dt_s is not None and (not np.isfinite(max_dt_s) or max_dt_s <= 0):
            raise ValueError("max_dt_s must be positive and finite")
        if not isinstance(sponge_cells, int) or sponge_cells < 0:
            raise ValueError("sponge_cells must be a nonnegative integer")
        if sponge_sigma is not None and (not np.isfinite(sponge_sigma) or sponge_sigma < 0):
            raise ValueError("sponge_sigma must be finite and nonnegative")
        if not ports or any(not isinstance(n, str) or not n for n in ports):
            raise ValueError("Provide at least one nonempty named volume-flow port")
        self.grid = deepcopy(grid)
        self._owner = object()
        options = dict(T_c=T_c, rh=rh, walls=walls, boundary_layer=boundary_layer,
                       sponge_cells=sponge_cells, sponge_sigma=sponge_sigma,
                       device=device, dtype=dtype, physical_cut_volume=True)
        sim = fdtd3d.Sim(self.grid, courant=courant, **options)
        self.patches = _patches(sim)
        self.h_m = grid.h * 0.01
        self.rho_kg_m3, self.sound_speed_m_s = sim.rho * 1000, sim.c * 0.01
        self.bulk_modulus_pa = self.rho_kg_m3 * self.sound_speed_m_s ** 2
        vf = np.asarray(grid.meta["vf"], float) if sim.cut else grid.air.astype(float)
        volume = vf * self.h_m ** 3
        self._volume_np = volume
        self.wall_pairs = deepcopy(tuple(wall_pairs))
        selected, pair_data, names = [], [], set()
        bound_structural = 0.0
        for pair in self.wall_pairs:
            if not isinstance(pair.name, str) or not pair.name or pair.name in names:
                raise ValueError("Wall pair names must be nonempty and unique")
            names.add(pair.name)
            m, k, r = pair.mass_kg_m2, pair.stiffness_pa_m, pair.damping_pa_s_m
            if not np.isfinite([m, k, r]).all() or m <= 0 or k <= 0 or r < 0:
                raise ValueError("Wall mass/stiffness must be positive; damping nonnegative")
            sides = []
            for ids, exterior in ((pair.inside, False), (pair.outside, True)):
                ids = np.asarray(ids)
                if (ids.ndim != 1 or not ids.size or not np.issubdtype(ids.dtype, np.integer)
                        or np.any((ids < 0) | (ids >= len(self.patches)))):
                    raise ValueError("WallPair must reference nonempty valid patch indices")
                side = [self.patches[int(i)] for i in ids]
                if any(p.exterior != exterior for p in side):
                    raise ValueError("WallPair inside/outside must reference internal/exterior air")
                if any(p.tissue in ("rigid", "baffle", "glottis_end", "none") for p in side):
                    raise ValueError("Cannot pair rigid palate, teeth, baffle or glottal end")
                if any(p.area_m2 <= 0 for p in side):
                    raise ValueError("Paired wall areas must be positive")
                selected.extend(int(i) for i in ids)
                cells = np.array([np.ravel_multi_index(p.cell, grid.shape) for p in side])
                areas = np.array([p.area_m2 for p in side])
                sides.append((cells, areas))
            area = float(sides[0][1].sum())
            if not np.isclose(area, sides[1][1].sum(), rtol=1e-12, atol=0):
                raise ValueError("Paired surfaces must have equal physical area")
            # Aggregate multiple faces belonging to one cell before the rank-one norm.
            signed_area = np.zeros(grid.air.size)
            for sign, (cells, areas) in zip((-1, 1), sides):
                np.add.at(signed_area, cells, sign * areas)
            nz = np.flatnonzero(signed_area)
            bound_structural += (self.bulk_modulus_pa * np.sum(
                signed_area[nz] ** 2 / volume.ravel()[nz]) / (area * m) + k / m)
            pair_data.append((sides, area))
        if len(set(selected)) != len(selected):
            raise ValueError("A wall patch cannot occur twice or belong to multiple pairs")
        selected_set = set(selected)
        self._local_m, self._local_k = np.zeros(sim.n_wall), np.zeros(sim.n_wall)
        if walls == "soft":
            for p in self.patches:
                if p.index in selected_set or p.tissue not in fdtd3d.TISSUE_WALL:
                    continue
                th, f, _ = fdtd3d.TISSUE_WALL[p.tissue]
                mass = fdtd3d.RHO_TISSUE * th * 10 * p.area_m2
                stiffness = mass * (2 * math.pi * f) ** 2
                self._local_m[p.index], self._local_k[p.index] = mass, stiffness
                if mass > 0:
                    bound_structural += self.bulk_modulus_pa * p.area_m2 ** 2 / (
                        mass * volume[p.cell]) + stiffness / mass
        # 12*c^2/(h^2*min(vf)) bounds the physical-volume Cartesian graph.
        bound_air = 12 * self.sound_speed_m_s ** 2 / (self.h_m ** 2 * vf[grid.air].min())
        limit = 1.9 / math.sqrt(bound_air + bound_structural)
        actual_dt = min(sim.dt, limit, max_dt_s if max_dt_s is not None else sim.dt)
        bl_rate = None
        if boundary_layer and sim.n_wall:
            area_per_cell = np.bincount(sim.wall_cell.cpu().numpy(),
                                       weights=sim.wall_face_area_cgs,
                                       minlength=grid.air.size).reshape(grid.shape)
            rates = sim.rho * sim.c ** 2 * sim.bl_C * area_per_cell[
                grid.air] / (vf[grid.air] * grid.h ** 3)
            bl_rate = float(rates.max())
        # BL fitting depends on dt. Accept only the coefficients of the actual
        # final Sim, with margin in each pressure/reservoir update, not a stale fit.
        for _ in range(32):
            actual_courant = actual_dt * sim.c * math.sqrt(3) / grid.h
            self._sim = fdtd3d.Sim(self.grid, courant=actual_courant,
                                  inactive_walls=selected, **options)
            if bl_rate is None:
                break
            load = max(self._sim.dt * bl_rate * sum(self._sim.bl_w), max(self._sim.bl_a))
            if not np.isfinite(load) or load < 0:
                raise RuntimeError("Nonfinite or invalid boundary-layer timestep bound")
            if load <= .25:
                break
            actual_dt = self._sim.dt * min(.5, .2 / load)
            if not np.isfinite(actual_dt) or actual_dt <= 0:
                raise RuntimeError("Boundary-layer timestep refinement underflowed")
        else:
            raise RuntimeError("No stable boundary-layer timestep after 32 coefficient refits")
        self.requested_courant = courant
        self._volume = self._tensor(volume)
        self._ports = {}
        for name, port in ports.items():
            mask = _mask(port.mask, grid.shape, f"port {name}")
            if np.any(mask & ~grid.air):
                raise ValueError(f"Port {name} includes solid")
            density = np.ones(grid.shape) if port.density_weights is None else np.asarray(port.density_weights)
            if (density.shape != grid.shape or not np.isfinite(density).all()
                    or np.any(density < 0) or not np.any(density[mask] > 0)):
                raise ValueError(f"Port {name} needs finite nonnegative nonzero density weights")
            cells = np.flatnonzero(mask & (density > 0))
            weights = density.ravel()[cells] * volume.ravel()[cells]
            weights /= weights.sum()
            self._check_undamped(cells)
            self._ports[name] = (self._indices(cells), self._tensor(weights))
        self._pair_data = []
        for sides, area in pair_data:
            converted = []
            for cells, areas in sides:
                self._check_undamped(cells)
                converted.append((self._indices(cells), self._tensor(areas / areas.sum())))
            self._pair_data.append((converted, area))
        self._x = self._tensor(np.zeros(len(wall_pairs)))
        self._v = self._tensor(np.zeros(len(wall_pairs)))
        self._steps = 0
        self._input_volume = dict.fromkeys(self._ports, 0.0)
        self._input_work = self._pair_dissipation = 0.0
        self._bl_removed_volume = self._local_wall_removed_volume = 0.0

    def _tensor(self, values):
        return torch.as_tensor(values, device=self._sim.dev, dtype=self._sim.dtype)

    def _indices(self, values):
        return torch.as_tensor(values, device=self._sim.dev, dtype=torch.long)

    def _check_undamped(self, cells):
        ids = self._indices(cells)
        if any(torch.any(d.reshape(-1)[ids] != 1).item()
               for d in (self._sim.dpx, self._sim.dpy, self._sim.dpz)):
            raise ValueError("Ports and wall pairs must lie outside nonzero PML damping")

    @property
    def dt(self) -> float:
        """Fixed field timestep [s]; reconstruct with max_dt_s to change the clock."""
        return self._sim.dt

    @property
    def time_s(self) -> float:
        return self._steps * self.dt

    @property
    def port_names(self) -> tuple[str, ...]:
        return tuple(self._ports)

    def pressure_pa(self, port: str) -> float:
        """Reciprocal volume-weighted port pressure at the current integer time."""
        cells, weights = self._ports[port]
        return float(torch.dot(self._sim.p.reshape(-1)[cells], weights).item() * 0.1)

    def sample_pressure_pa(self, points_m, *, exterior_only: bool = True) -> np.ndarray:
        """Nearest containing-cell samples, not a fitted/interpolated radiation path."""
        points = np.asarray(points_m, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Receivers must have shape (n, 3) and finite SI coordinates")
        result = []
        for point in points:
            idx = self.grid.index(point * 100)
            if any(i < 0 or i >= n for i, n in zip(idx, self.grid.shape)) or not self.grid.air[idx]:
                raise ValueError("Receiver must lie in an air cell inside the grid")
            if exterior_only and not self.grid.exterior[idx]:
                raise ValueError("Exterior receiver lies inside the airway")
            result.append(self._sim.p[idx].item() * 0.1)
        return np.asarray(result)

    @torch.no_grad()
    def advance(self, flows_m3_s: Mapping[str, float]) -> StepResult:
        """Advance once. Missing known ports mean zero; unknown/nonfinite flows fail."""
        unknown = flows_m3_s.keys() - self._ports.keys()
        if unknown:
            raise ValueError(f"Unknown volume-flow ports: {sorted(unknown)}")
        flows = {n: float(flows_m3_s.get(n, 0.0)) for n in self._ports}
        if not np.isfinite(list(flows.values())).all():
            raise ValueError("Volume flows must be finite")
        before = {n: self.pressure_pa(n) for n in self._ports}
        cells_all, flows_all, pair_flows = [], [], {}
        pf = self._sim.p.reshape(-1) * 0.1
        for n, (cells, weights) in self._ports.items():
            cells_all.append(cells)
            flows_all.append(weights * (flows[n] * 1e6))
        for i, (pair, (sides, area)) in enumerate(zip(self.wall_pairs, self._pair_data)):
            pin, pout = [torch.dot(pf[cells], weights) for cells, weights in sides]
            m, k, r = pair.mass_kg_m2, pair.stiffness_pa_m, pair.damping_pa_s_m
            old_v = self._v[i].clone()
            new_v = ((m / self.dt - r / 2) * old_v + pin - pout - k * self._x[i]) / (
                m / self.dt + r / 2)
            self._v[i] = new_v
            self._x[i] += self.dt * new_v
            q = area * new_v
            pair_flows[pair.name] = float(q.item())
            self._pair_dissipation += float((self.dt * area * r * ((old_v + new_v) / 2) ** 2).item())
            for sign, (cells, weights) in zip((-1, 1), sides):
                cells_all.append(cells)
                flows_all.append(weights * (sign * q * 1e6))
        sim = self._sim
        if sim.bl and sim.n_wall:
            pw = sim.p.reshape(-1)[sim.wall_cell]
            vbl = sum(w * (pw - phi) for w, phi in zip(sim.bl_w, sim.phi)) * sim.bl_C
            self._bl_removed_volume += float(
                (vbl * sim.wall_cos).sum().item() * sim.g.h ** 2 * self.dt * 1e-6)
        sim.step(0.0, cell_flows=(torch.cat(cells_all), torch.cat(flows_all)))
        self._local_wall_removed_volume += float(
            (sim.vw * sim.wall_cos).sum().item() * sim.g.h ** 2 * self.dt * 1e-6)
        self._steps += 1
        after = {n: self.pressure_pa(n) for n in self._ports}
        work_p = {n: (before[n] + after[n]) / 2 for n in self._ports}
        for n, q in flows.items():
            self._input_volume[n] += self.dt * q
            self._input_work += self.dt * q * work_p[n]
        if not all(torch.isfinite(t).all().item() for t in (*sim._state(), self._x, self._v)):
            raise FloatingPointError("Acoustic state became nonfinite")
        return StepResult(self.time_s, after, work_p, pair_flows)

    def snapshot(self) -> AcousticSnapshot:
        return AcousticSnapshot(self._owner, tuple(t.clone() for t in self._sim._state()),
                                self._x.clone(), self._v.clone(), self._steps,
                                self._input_volume.copy(), self._input_work,
                                self._pair_dissipation, self._bl_removed_volume,
                                self._local_wall_removed_volume)

    @torch.no_grad()
    def restore(self, snapshot: AcousticSnapshot) -> None:
        if snapshot.owner is not self._owner:
            raise ValueError("Snapshot belongs to a different acoustic domain")
        for target, value in zip(self._sim._state(), snapshot.field):
            target.copy_(value)
        self._x.copy_(snapshot.displacement)
        self._v.copy_(snapshot.velocity)
        self._steps = snapshot.steps
        self._input_volume = snapshot.input_volume.copy()
        self._input_work, self._pair_dissipation = snapshot.input_work, snapshot.pair_dissipation
        self._bl_removed_volume = snapshot.bl_removed_volume
        self._local_wall_removed_volume = snapshot.local_wall_removed_volume

    def diagnostics(self) -> dict:
        """Physical and modified energies [J], signed volumes [m3], and coverage.

        Modified energy is a passive certificate only when ``energy_certified``
        is true. BL reservoirs/PML are not included in that invariant. Compression
        volume is integral(p/B dV); it equals net injected volume in a closed
        rigid/pair-only domain (pair displacements cancel globally).
        Surface coverage counts BOTH paired wetted sides. Removed BL volume is
        signed storage/flow, not dissipated energy. Local wall energy is separate
        and excluded from the pair-only staggered energy certificate.
        """
        s, h, rho, bulk = self._sim, self.h_m, self.rho_kg_m3, self.bulk_modulus_pa
        p = s.p * 0.1
        acoustic = float((0.5 / bulk * self._volume * p ** 2).sum().item())
        cross = 0.0
        for axis, (u, a) in enumerate(zip((s.ux, s.uy, s.uz), (s.ax, s.ay, s.az))):
            velocity = u * 0.01
            acoustic += float((0.5 * rho * h ** 3 * a * velocity ** 2).sum().item())
            lo, hi, mid = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
            lo[axis], hi[axis], mid[axis] = slice(None, -1), slice(1, None), slice(1, -1)
            cross -= float((self.dt / 2 * h ** 2 * a[tuple(mid)] * velocity[tuple(mid)]
                            * (p[tuple(hi)] - p[tuple(lo)])).sum().item())
        wall_energy = 0.0
        for i, (pair, (sides, area)) in enumerate(zip(self.wall_pairs, self._pair_data)):
            dp = sum(sign * torch.dot(p.reshape(-1)[cells], weights)
                     for sign, (cells, weights) in zip((1, -1), sides))
            x, v = self._x[i], self._v[i]
            wall_energy += float((0.5 * area * (
                pair.mass_kg_m2 * v ** 2 + pair.stiffness_pa_m * x ** 2)).item())
            cross += float((self.dt / 2 * area * v * (dp - pair.stiffness_pa_m * x)).item())
        local_energy = float((0.5 * self._tensor(self._local_m) * (s.vw * 0.01) ** 2
                              + 0.5 * self._tensor(self._local_k) * (s.xw * 0.01) ** 2).sum().item())
        selected = {i for pair in self.wall_pairs for i in (*pair.inside, *pair.outside)}
        soft = [p for p in self.patches if p.tissue in fdtd3d.TISSUE_WALL]
        soft_area = sum(p.area_m2 for p in soft)
        paired_area = sum(p.area_m2 for p in soft if p.index in selected)
        pml = any(torch.any(d != 1).item() for d in (s.dpx, s.dpy, s.dpz))
        return dict(time_s=self.time_s, dt_s=self.dt, acoustic_energy_j=acoustic,
                    paired_wall_energy_j=wall_energy, local_wall_energy_j=local_energy,
                    staggered_energy_j=acoustic + wall_energy + cross,
                    energy_certified=not pml and not s.bl and not bool(s.soft.any()),
                    compression_volume_m3=float((self._volume * p / bulk).sum().item()),
                    input_volume_m3=self._input_volume.copy(), input_work_j=self._input_work,
                    pair_dissipation_j=self._pair_dissipation,
                    boundary_layer_removed_volume_m3=self._bl_removed_volume,
                    local_wall_removed_volume_m3=self._local_wall_removed_volume,
                    paired_surface_area_m2=paired_area, unpaired_surface_area_m2=soft_area - paired_area,
                    paired_soft_area_fraction=paired_area / soft_area if soft_area else 0.0,
                    wall_displacement_m={q.name: self._x[i].item() for i, q in enumerate(self.wall_pairs)},
                    wall_velocity_m_s={q.name: self._v[i].item() for i, q in enumerate(self.wall_pairs)},
                    boundary_layer_enabled=s.bl, pml_enabled=pml,
                    physical_air_volume_m3=float(self._volume_np.sum()),
                    fixed_pose=True)
