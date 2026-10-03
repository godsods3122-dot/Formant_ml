"""Optional three-dimensional vocal-fold solid, entirely in SI units.

Axes are x = anterior -> posterior (AP), y = left -> right (lateral),
z = inferior -> superior (SI). Neither side is a symmetry boundary. This is
a forward mechanics component, NOT a waveform, fitted anatomy, or F0 driver.

Constitutive assumptions
-----------------------
The matrix uses the compressible Neo-Hookean energy already used by
``softtissue``: mu/2*(F:F-3)-mu*log(J)+lambda/2*log(J)**2.
Finite bulk modulus approximates incompressibility. Linear displacement tets
can volumetrically lock; refinement and a future mixed formulation are needed
before claiming quantitatively converged near-incompressible human mechanics.

The tension-only longitudinal fiber energy is
Wf(q)=k*q**2/2+s2*(expm1(B*q)/B-q-B*q**2/2), q=max(|F*d|-1,0).
Its derivative is *nominal fiber stress*, not an added total Cauchy tissue
stress. Nominal defaults adapt the exponential coefficients and zero-strain
tangent in ``fold_rules`` (Hunter & Titze 2004; Titze 2006; Titze & Story 2002).
The matrix's small-strain uniaxial Young modulus is subtracted from that
tangent, and source prestress is removed. This is a stress-free constitutive
adaptation, not a claim to reproduce the source's complete tensile curve.
The fiber Cauchy stress is |F*d|*Wf'/J. TA activation uses the source active
force-length curve as Cauchy stress; P_active=J*sigma_active*F**(-T).
Kelvin-Voigt sigma_v=2*eta*sym(Fdot*F**(-1)) dissipates nonnegative power.
No constitutive Jacobian is clipped.

The block geometry and tissue coefficients are provisional model-source data,
not human population bounds or a personalized reconstruction. Cartilage
controls are prescribed support displacement/strain; cartilage mechanics,
fluid/tract coupling, fitting and physiological validation are out of scope.

Contact is deliberately limited: registered material nodes, fixed lateral
normal and *reference* nodal area, frictionless unilateral penalty, closing
damping only, no adhesion. Excess tangential slip or rotated/folded surfaces
raises an error, rather than silently matching the wrong tissue under shear.
This is NOT general sliding or large-rotation contact.

Pressure is positive inward on the *current* outward-wound medial triangles,
one constant Pa value per triangle. ``surface`` returns the identical area
vectors, centroids and 1/3 nodal shape weights used for traction. At any frozen
geometry its exact virtual work is -sum(p*A_normal dot mean(delta_x)).
During a step the pressure is held constant per triangle. Its follower force
uses the Simpson-exact mean area vector along the linear nodal trajectory,
solved by a bounded pressure-only fixed-point iteration. Thus pressure work
is exactly -sum(p*solid_outward_sweep), up to arithmetic/iteration tolerance.
Medial sweeps are only the moving WALL contribution to gas volume: moving
AP/SI caps are the caller's responsibility. Do not count both wall sweep and
the same gap-profile volume change. End supports can move with controls, and
inferior/superior medial edges are free by default.
Cross-sections intersect these same current triangles with z=constant,
integrating the positive lateral gap along AP, including its zero crossings.
Unsupported overhangs or unequal AP coverage fail explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Literal

import numpy as np
import torch

from . import fold_rules
from .softtissue import (
    box_tets,
    tetrahedral_nodal_forces,
    triangle_area_vectors,
    triangle_pressure_forces,
)

Side = Literal["left", "right"]


def _finite(name, value, *, minimum=None, strict=False):
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and (value <= minimum if strict else value < minimum):
        raise ValueError(f"{name} must be {'>' if strict else '>='} {minimum}")


def _array(name, value, shape, *, integer=False, boolean=False):
    a = np.asarray(value)
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
    if integer and not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"{name} must contain integer indices")
    if boolean and a.dtype != np.bool_:
        raise ValueError(f"{name} must be boolean")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must contain finite values")
    return a.copy()


@dataclass(frozen=True)
class SolidMaterial:
    """Matrix/fiber parameters. Stresses Pa, density kg/m³, viscosity Pa*s."""

    matrix_mu_pa: float
    matrix_bulk_pa: float
    density_kg_m3: float = 1040.0
    viscosity_pa_s: float = 1.0
    fiber_linear_pa: float = 0.0
    fiber_exponential_pa: float = 0.0
    fiber_exponent: float = 6.0
    active_max_pa: float = 0.0
    active_optimal_strain: float = 0.3
    active_width: float = 1.0

    def __post_init__(self):
        for name in ("matrix_mu_pa", "matrix_bulk_pa", "density_kg_m3", "fiber_exponent"):
            _finite(name, getattr(self, name), minimum=0, strict=True)
        if self.matrix_bulk_pa <= 2 * self.matrix_mu_pa / 3:
            raise ValueError("bulk modulus must exceed 2*mu/3 (positive Lame lambda)")
        for name in ("viscosity_pa_s", "fiber_linear_pa", "fiber_exponential_pa",
                     "active_max_pa", "active_width"):
            _finite(name, getattr(self, name), minimum=0)
        _finite("active_optimal_strain", self.active_optimal_strain)


def nominal_materials(tissue: str = "serry2026") -> tuple[SolidMaterial, ...]:
    """Cover, ligament, TA; nominal mu and tensile rules, nu=0.49, eta=1 Pa*s.

    Ligament matrix mu=500 Pa and viscosity are provisional. The source fiber
    tangent is floored at zero *during parameter construction* if the matrix
    already supplies it; this is not deformation or stress stabilization.
    """
    if tissue not in fold_rules.TISSUE_SETS:
        raise ValueError(f"unknown tissue source {tissue!r}")
    result = []
    for name, mu in (("muc", 500.0), ("lig", 500.0), ("ta", 1000.0)):
        p = fold_rules.TISSUE_SETS[tissue][name]
        bulk = 2 * mu * 1.49 / (3 * 0.02)
        young = 9 * bulk * mu / (3 * bulk + mu)
        fiber = max(0.0, float(fold_rules.passive_modulus(0.0, p)) - young)
        result.append(SolidMaterial(mu, bulk, fold_rules.RHO, 1.0, fiber,
                                    p[1], p[4], p[5], p[6], p[7]))
    return tuple(result)


@dataclass(frozen=True)
class FoldMesh:
    """Caller mesh: positive-oriented tets, per-element materials/unit fibers.

    All indices are side-local. Support masks (N,3) fix individual DOFs.
    ``medial_grid`` is optional (n_AP,n_SI) registered node IDs for contact;
    pressure and geometric slicing need only outward-wound medial triangles.
    Layer interfaces must use shared node IDs, not coincident separate bodies.
    """

    reference_m: np.ndarray
    tetrahedra: np.ndarray
    material_ids: np.ndarray
    fiber_directions: np.ndarray
    medial_triangles: np.ndarray
    anterior_support: np.ndarray
    posterior_support: np.ndarray
    lateral_support: np.ndarray
    medial_grid: np.ndarray | None = None


def nominal_fold_mesh(
    side: Side, *, n_ap: int = 4, n_si: int = 2,
    cells_per_layer: tuple[int, int, int] = (1, 1, 1),
    length_m: float = 0.01, height_m: float = 0.002,
    layer_depths_m: tuple[float, float, float] = (0.0015, 0.0015, 0.003),
    half_gap_m: float = 0.0002, material_ids: tuple[int, int, int] = (0, 1, 2),
    fix_anterior: bool = True, fix_posterior: bool = True, fix_lateral: bool = True,
) -> FoldMesh:
    """Conforming, explicitly provisional three-layer block (no fitted anatomy).

    Default dimensions follow ``fold_rules.DIMS['female']`` nominal values,
    without interpreting the label as a population bound. Cover is medial.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    if len(cells_per_layer) != 3 or len(layer_depths_m) != 3 or len(material_ids) != 3:
        raise ValueError("exactly three cover/ligament/TA regions are required")
    for n in (n_ap, n_si, *cells_per_layer):
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("mesh cell counts must be positive integers")
    for value in (length_m, height_m, *layer_depths_m):
        _finite("mesh dimension", value, minimum=0, strict=True)
    _finite("half_gap_m", half_gap_m, minimum=0)
    ny = sum(cells_per_layer)
    x, tets, idx = box_tets(n_ap, ny, n_si, (length_m, float(ny), height_m))
    depth = [0.0]
    for n, d in zip(cells_per_layer, layer_depths_m):
        depth.extend(np.linspace(depth[-1], depth[-1] + d, n + 1)[1:])
    layer_coordinate = x[:, 1].astype(int)
    sign = -1 if side == "left" else 1
    x[:, 1] = sign * (half_gap_m + np.asarray(depth)[layer_coordinate])
    neg = np.linalg.det(x[tets[:, 1:]] - x[tets[:, :1]]) < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    cell_region = np.repeat(np.asarray(material_ids), cells_per_layer)
    regions = cell_region[layer_coordinate[tets].min(axis=1)]
    grid = np.array([[idx(i, 0, k) for k in range(n_si + 1)] for i in range(n_ap + 1)])
    tris = []
    for i in range(n_ap):
        for k in range(n_si):
            a, b, c, d = grid[i, k], grid[i + 1, k], grid[i + 1, k + 1], grid[i, k + 1]
            # box_tets uses the (i,k) -> (i+1,k+1) diagonal on this face.
            tris.extend(((a, b, c), (a, c, d)))
    tris = np.asarray(tris)
    if side == "left":
        tris = tris[:, [0, 2, 1]]
    anterior = np.repeat((x[:, 0] == 0)[:, None], 3, axis=1) & fix_anterior
    posterior = np.repeat((x[:, 0] == length_m)[:, None], 3, axis=1) & fix_posterior
    lateral = np.repeat((layer_coordinate == ny)[:, None], 3, axis=1) & fix_lateral
    fiber = np.tile([1.0, 0.0, 0.0], (len(tets), 1))
    return FoldMesh(x, tets, regions, fiber, tris, anterior, posterior, lateral, grid)


@dataclass(frozen=True)
class ContactConfig:
    enabled: bool = True
    stiffness_pa_per_m: float = 5e7
    damping_pa_s_per_m: float = 50.0
    max_tangential_slip_m: float = 2e-5
    minimum_lateral_normal: float = 0.5

    def __post_init__(self):
        for name in ("stiffness_pa_per_m", "max_tangential_slip_m"):
            _finite(name, getattr(self, name), minimum=0, strict=True)
        _finite("damping_pa_s_per_m", self.damping_pa_s_per_m, minimum=0)
        if not 0 < self.minimum_lateral_normal <= 1:
            raise ValueError("minimum_lateral_normal must be in (0,1]")


@dataclass(frozen=True)
class SolidConfig:
    left: FoldMesh
    right: FoldMesh
    materials: tuple[SolidMaterial, ...] = field(default_factory=nominal_materials)
    contact: ContactConfig = field(default_factory=ContactConfig)
    timestep_safety: float = 0.25
    pressure_max_iterations: int = 20
    pressure_tolerance_m: float = 1e-13

    def __post_init__(self):
        if not self.materials or not all(isinstance(m, SolidMaterial) for m in self.materials):
            raise ValueError("materials must be a nonempty sequence of SolidMaterial")
        if not 0 < self.timestep_safety <= 0.5:
            raise ValueError("timestep_safety must be in (0,0.5]")
        if (isinstance(self.pressure_max_iterations, bool)
                or not isinstance(self.pressure_max_iterations, int)
                or self.pressure_max_iterations < 1):
            raise ValueError("pressure_max_iterations must be a positive integer")
        _finite("pressure_tolerance_m", self.pressure_tolerance_m, minimum=0, strict=True)


@dataclass(frozen=True)
class FoldControl:
    """Support targets, not imposed tissue oscillations.

    AP strain adds strain*(X-x_anterior) to support x; anterior/posterior
    displacement vectors interpolate in reference AP. Lateral displacement
    adds on lateral support DOFs, including intersections with end supports.
    Controls specify the endpoint posture of a step; callers must ramp posture
    continuously. Velocity on constrained DOFs is the actual displacement/dt.
    """

    ta_activation: float = 0.0
    longitudinal_strain: float = 0.0
    anterior_displacement_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    posterior_displacement_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lateral_displacement_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if not 0 <= self.ta_activation <= 1:
            raise ValueError("TA activation must be in [0,1]")
        _finite("longitudinal_strain", self.longitudinal_strain, minimum=-1, strict=True)
        for name in ("anterior_displacement_m", "posterior_displacement_m",
                     "lateral_displacement_m"):
            _array(name, getattr(self, name), (3,))


@dataclass(frozen=True)
class BilateralControl:
    left: FoldControl = field(default_factory=FoldControl)
    right: FoldControl = field(default_factory=FoldControl)


@dataclass(frozen=True)
class MedialPressure:
    """One inward-positive [Pa] value per side-local medial triangle."""

    left_pa: np.ndarray
    right_pa: np.ndarray


@dataclass(frozen=True)
class MedialSurface:
    coordinates_m: np.ndarray
    velocities_m_s: np.ndarray
    triangles: np.ndarray
    centroids_m: np.ndarray
    area_vectors_m2: np.ndarray
    quadrature_area_m2: np.ndarray
    shape_weights: np.ndarray


@dataclass(frozen=True)
class SurfaceSweep:
    """Exact oriented sweep [m³/triangle] along linear vertex trajectories.

    Outward is from the solid, so positive gas pressure does NEGATIVE work
    on positive outward sweep. ``-outward_swept_volume_m3`` is this wall's
    contribution to gas volume change; it excludes moving AP/SI end caps.
    Rigid translation has centroid displacement equal to its translation;
    rotation and deformation also change the mean oriented area.
    """

    mean_area_vectors_m2: np.ndarray
    centroid_displacement_m: np.ndarray
    outward_swept_volume_m3: np.ndarray


@dataclass(frozen=True)
class GapSlice:
    si_m: float
    ap_m: np.ndarray
    signed_gap_m: np.ndarray
    ap_quadrature_m: np.ndarray
    area_m2: float


@dataclass(frozen=True)
class GlottalProfile:
    si_m: np.ndarray
    area_m2: np.ndarray
    slices: tuple[GapSlice, ...]


@dataclass(frozen=True)
class WorkTotals:
    viscous_dissipated_j: float = 0.0
    contact_dissipated_j: float = 0.0
    pressure_j: float = 0.0
    active_j: float = 0.0
    support_j: float = 0.0


@dataclass(frozen=True)
class SolidState:
    mesh_signature: str
    time_s: float
    positions_m: np.ndarray
    velocities_m_s: np.ndarray
    control: BilateralControl
    work: WorkTotals
    initial_mechanical_j: float


@dataclass(frozen=True)
class SolidDiagnostics:
    kinetic_j: float
    strain_j: float
    contact_j: float
    viscous_power_w: float
    contact_dissipation_w: float
    minimum_jacobian: float
    work: WorkTotals
    energy_balance_residual_j: float


@dataclass(frozen=True)
class SolidForces:
    """Global node ordering is all left nodes followed by all right nodes."""

    passive_n: np.ndarray
    viscous_n: np.ndarray
    active_n: np.ndarray
    pressure_n: np.ndarray
    contact_n: np.ndarray
    total_n: np.ndarray


@dataclass(frozen=True)
class SolidStep:
    time_s: float
    diagnostics: SolidDiagnostics
    support_reaction_n: np.ndarray
    left_sweep: SurfaceSweep
    right_sweep: SurfaceSweep
    pressure_iterations: int
    pressure_residual_m: float


def _tensor(a, dtype=torch.float64):
    return torch.tensor(np.asarray(a).copy(), dtype=dtype)


def _sweep_tensors(start, end, triangles):
    areas = [triangle_area_vectors(x, triangles) for x in (start, (start + end) / 2, end)]
    if any(not torch.isfinite(a).all() or torch.any(torch.linalg.norm(a, dim=1) <= 0) for a in areas):
        raise ValueError("nonfinite/degenerate triangle along surface sweep")
    mean_area = (areas[0] + 4 * areas[1] + areas[2]) / 6
    displacement = (end[triangles] - start[triangles]).mean(dim=1)
    return mean_area, displacement, (mean_area * displacement).sum(dim=1)


def triangle_surface_sweep(start_m: np.ndarray, end_m: np.ndarray, triangles: np.ndarray) -> SurfaceSweep:
    """Sweep arbitrary consistently oriented triangles, including closed meshes.

    Area is quadratic and centroid velocity constant in trajectory parameter,
    making Simpson quadrature exact. Sum of sweeps on a CLOSED triangulated
    polyhedron equals its signed endpoint volume change. The individual
    medial surfaces of a fold are NOT closed polyhedra.
    """
    start = np.asarray(start_m)
    if start.ndim != 2 or start.shape[1] != 3:
        raise ValueError("start_m must have shape (N,3)")
    start = _tensor(_array("start_m", start, start.shape))
    end = _tensor(_array("end_m", end_m, tuple(start.shape)))
    tris = np.asarray(triangles)
    if tris.ndim != 2 or tris.shape[1] != 3 or not len(tris):
        raise ValueError("triangles must be nonempty (K,3)")
    tris = _array("triangles", tris, tris.shape, integer=True)
    if tris.min() < 0 or tris.max() >= len(start):
        raise ValueError("triangle node index out of range")
    return SurfaceSweep(*(a.numpy() for a in _sweep_tensors(start, end, _tensor(tris, torch.long))))


def _checked_mesh(mesh, side, n_materials):
    x = np.asarray(mesh.reference_m)
    t = np.asarray(mesh.tetrahedra)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) < 4:
        raise ValueError("reference_m must be (N,3), N>=4")
    if t.ndim != 2 or t.shape[1] != 4 or not len(t):
        raise ValueError("tetrahedra must be nonempty (M,4)")
    x = _array("reference_m", x, x.shape).astype(float)
    if len(np.unique(x, axis=0)) != len(x):
        raise ValueError("coincident separate nodes are unsupported; layer interfaces must share node IDs")
    t = _array("tetrahedra", t, t.shape, integer=True)
    if t.min() < 0 or t.max() >= len(x):
        raise ValueError("tetrahedron node index out of range")
    if len(np.unique(t)) != len(x):
        raise ValueError("every node must belong to a tetrahedron")
    if len(np.unique(np.sort(t, axis=1), axis=0)) != len(t):
        raise ValueError("duplicate tetrahedra")
    determinant = np.linalg.det(x[t[:, 1:]] - x[t[:, :1]])
    if not np.isfinite(determinant).all() or np.any(determinant <= 0):
        raise ValueError("reference tetrahedra must have finite positive orientation/volume")
    ids = _array("material_ids", mesh.material_ids, (len(t),), integer=True)
    if ids.min() < 0 or ids.max() >= n_materials:
        raise ValueError("material index out of range")
    fib = _array("fiber_directions", mesh.fiber_directions, (len(t), 3))
    if not np.allclose(np.linalg.norm(fib, axis=1), 1, atol=1e-10, rtol=0):
        raise ValueError("fiber directions must be unit vectors")
    masks = [_array(name, getattr(mesh, name), x.shape, boolean=True)
             for name in ("anterior_support", "posterior_support", "lateral_support")]
    tris = np.asarray(mesh.medial_triangles)
    if tris.ndim != 2 or tris.shape[1] != 3 or not len(tris):
        raise ValueError("medial_triangles must be nonempty (K,3)")
    tris = _array("medial_triangles", tris, tris.shape, integer=True)
    if tris.min() < 0 or tris.max() >= len(x):
        raise ValueError("medial triangle index out of range")
    faces, owners = {}, {}
    for element, tet in enumerate(t):
        for k in range(4):
            key = tuple(sorted(np.delete(tet, k)))
            faces.setdefault(key, []).append(tet[k])
            owners.setdefault(key, []).append(element)
    seen = set()
    for tri in tris:
        key = tuple(sorted(tri))
        if key in seen or len(faces.get(key, [])) != 1:
            raise ValueError("medial triangles must be unique exterior tetrahedron faces")
        seen.add(key)
        normal = np.cross(x[tri[1]] - x[tri[0]], x[tri[2]] - x[tri[0]])
        if np.dot(normal, x[faces[key][0]] - x[tri[0]]) >= 0:
            raise ValueError("medial triangles must be outward wound")
        if normal[1] * (1 if side == "left" else -1) <= 0:
            raise ValueError("medial surface must face the opposite side along y")
    if any(len(adjacent) > 2 for adjacent in faces.values()):
        raise ValueError("nonmanifold tetrahedral mesh")
    neighbors = [[] for _ in t]
    for face, elements in owners.items():
        if len(elements) != 2:
            continue
        a, b, c = x[list(face)]
        normal = np.cross(b - a, c - a)
        opposite = x[faces[face]] - a
        if np.prod(opposite @ normal) >= 0:
            raise ValueError("adjacent tetrahedra overlap on the same side of an interface")
        i, j = elements
        neighbors[i].append(j)
        neighbors[j].append(i)
    visited, pending = {0}, [0]
    while pending:
        for neighbor in neighbors[pending.pop()]:
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    if len(visited) != len(t):
        raise ValueError("fold tetrahedra must be face-connected with shared layer interface nodes")
    grid = None
    if mesh.medial_grid is not None:
        grid = np.asarray(mesh.medial_grid)
        if grid.ndim != 2 or min(grid.shape) < 2:
            raise ValueError("medial_grid must be (n_AP,n_SI), both >=2")
        grid = _array("medial_grid", grid, grid.shape, integer=True)
        if len(np.unique(grid)) != grid.size or not np.array_equal(np.sort(grid.ravel()), np.unique(tris)):
            raise ValueError("medial_grid must enumerate each medial node exactly once")
    if np.ptp(x[:, 0]) <= 0:
        raise ValueError("mesh must have positive AP extent")
    return FoldMesh(x, t, ids, fib, tris, *masks, grid)


class VocalFoldSolid:
    """Stateful CPU/float64 reference FEM; no connection to online/reduced paths.

    ``recommend_timestep`` bounds the mass-normalized assembled local tangent
    row sums (including active/geometric terms and pressure follower loads),
    viscous damping and potential contact onset. It is a conservative *local*
    explicit stability estimate, not a guarantee for arbitrary posture jumps
    or a proof of nonlinear stability. Every step also checks the endpoint
    bound and all Jacobians, and is atomic on failure. No adaptive clipping.

    Kick-drift (symplectic Euler) is first order except for the work-consistent
    pressure path integral. Dissipation uses beginning-of-step nonnegative
    physical power; active/support work uses force dot actual displacement.
    ``energy_balance_residual_j`` exposes discretization error, not a hidden
    energy correction.
    """

    def __init__(self, config: SolidConfig | None = None):
        self.config = config or SolidConfig(nominal_fold_mesh("left"), nominal_fold_mesh("right"))
        meshes = [_checked_mesh(getattr(self.config, side), side, len(self.config.materials))
                  for side in ("left", "right")]
        self._meshes = tuple(meshes)
        self._offset = len(meshes[0].reference_m)
        self._slices = (slice(0, self._offset), slice(self._offset, None))
        self._X = _tensor(np.concatenate([m.reference_m for m in meshes]))
        self._t = _tensor(np.concatenate([m.tetrahedra + o for m, o in zip(meshes, (0, self._offset))]),
                          torch.long)
        self._fib = _tensor(np.concatenate([m.fiber_directions for m in meshes]))
        self._ids = np.concatenate([m.material_ids for m in meshes])
        self._element_side = _tensor(np.repeat([0, 1], [len(m.tetrahedra) for m in meshes]), torch.long)
        self._surfaces = tuple(_tensor(m.medial_triangles + o, torch.long)
                               for m, o in zip(meshes, (0, self._offset)))
        self._fixed = _tensor(np.concatenate([m.anterior_support | m.posterior_support | m.lateral_support
                                             for m in meshes]), torch.bool)
        dm = (self._X[self._t[:, 1:]] - self._X[self._t[:, :1]]).transpose(1, 2)
        self._inverse = torch.linalg.inv(dm)
        self._volume = torch.linalg.det(dm) / 6
        self._parameters = {
            name: _tensor([getattr(self.config.materials[i], name) for i in self._ids])
            for name in SolidMaterial.__dataclass_fields__
        }
        self._mass = torch.zeros(len(self._X), dtype=torch.float64)
        self._mass.index_add_(0, self._t.flatten(),
                              (self._parameters["density_kg_m3"] * self._volume / 4).repeat_interleave(4))
        if not torch.isfinite(self._mass).all() or torch.any(self._mass <= 0):
            raise ValueError("nodal mass must be finite and positive")
        grad = torch.cat((-self._inverse.sum(dim=1, keepdim=True), self._inverse), dim=1)
        self._B = torch.einsum("ac,mnb->mabnc", torch.eye(3, dtype=torch.float64), grad).reshape(-1, 9, 12)
        self._dofs = (self._t[:, :, None] * 3 + torch.arange(3)).reshape(-1, 12)
        self._pair_left = self._pair_right = torch.empty(0, dtype=torch.long)
        self._pair_area = torch.empty(0, dtype=torch.float64)
        self._prepare_contact()
        digest = hashlib.sha256()
        for value in (self._X, self._t, self._fib, self._fixed, *self._surfaces,
                      self._pair_left, self._pair_right, self._pair_area, _tensor(self._ids, torch.long)):
            digest.update(value.numpy().tobytes())
        for mesh in meshes:
            for mask in (mesh.anterior_support, mesh.posterior_support, mesh.lateral_support):
                digest.update(mask.tobytes())
        digest.update(repr((self.config.materials, self.config.contact, self.config.timestep_safety,
                            self.config.pressure_max_iterations, self.config.pressure_tolerance_m)).encode())
        self._signature = digest.hexdigest()
        self.reset()

    @property
    def positions_m(self) -> np.ndarray:
        return self._x.numpy().copy()

    @property
    def velocities_m_s(self) -> np.ndarray:
        return self._v.numpy().copy()

    @property
    def nodal_mass_kg(self) -> np.ndarray:
        return self._mass.numpy().copy()

    @property
    def element_volume_m3(self) -> np.ndarray:
        return self._volume.numpy().copy()

    def _prepare_contact(self):
        if not self.config.contact.enabled:
            return
        l, r = self._meshes
        if l.medial_grid is None or r.medial_grid is None or l.medial_grid.shape != r.medial_grid.shape:
            raise ValueError("matching contact requires same-shaped registered medial_grid on both sides")
        il = _tensor(l.medial_grid.ravel(), torch.long)
        ir = _tensor(r.medial_grid.ravel() + self._offset, torch.long)
        if not torch.allclose(self._X[il][:, [0, 2]], self._X[ir][:, [0, 2]], rtol=0, atol=1e-12):
            raise ValueError("contact grids must have identical reference AP/SI registration")
        areas = []
        for tris, nodes in zip(self._surfaces, (il, ir)):
            a = torch.zeros(len(self._X), dtype=torch.float64)
            face = torch.linalg.norm(triangle_area_vectors(self._X, tris), dim=1)
            a.index_add_(0, tris.flatten(), (face / 3).repeat_interleave(3))
            areas.append(a[nodes])
        if not torch.allclose(*areas, rtol=1e-8, atol=1e-16):
            raise ValueError("matching contact requires equal reference nodal surface quadrature")
        self._pair_left, self._pair_right, self._pair_area = il, ir, areas[0]

    def _kinematics(self, x, v):
        F = (x[self._t[:, 1:]] - x[self._t[:, :1]]).transpose(1, 2) @ self._inverse
        Fdot = (v[self._t[:, 1:]] - v[self._t[:, :1]]).transpose(1, 2) @ self._inverse
        J = torch.linalg.det(F)
        if not torch.isfinite(F).all() or not torch.isfinite(Fdot).all() or not torch.isfinite(J).all() or torch.any(J <= 0):
            raise ValueError("solid state has nonfinite kinematics or nonpositive deformation Jacobian")
        return F, Fdot, J

    def _constitutive(self, F, Fdot, control):
        p = self._parameters
        J = torch.linalg.det(F)
        inverse = torch.linalg.inv(F)
        inverse_t = inverse.transpose(1, 2)
        logj = torch.log(J)
        mu = p["matrix_mu_pa"]
        lam = p["matrix_bulk_pa"] - 2 * mu / 3
        passive = mu[:, None, None] * (F - inverse_t) + (lam * logj)[:, None, None] * inverse_t
        energy = 0.5 * mu * ((F * F).sum((1, 2)) - 3) - mu * logj + 0.5 * lam * logj**2
        b = (F @ self._fib[:, :, None]).squeeze(2)
        stretch = torch.linalg.norm(b, dim=1)
        q = torch.clamp_min(stretch - 1, 0)
        exponent = p["fiber_exponent"]
        e = exponent * q
        # Series avoid cancellation in the small-strain energy, not stabilization.
        remainder2 = torch.where(torch.abs(e) < 1e-3,
                                 e**3 / 6 + e**4 / 24 + e**5 / 120 + e**6 / 720,
                                 torch.expm1(e) - e - e**2 / 2)
        energy = energy + 0.5 * p["fiber_linear_pa"] * q**2 + p["fiber_exponential_pa"] * remainder2 / exponent
        stress = p["fiber_linear_pa"] * q + p["fiber_exponential_pa"] * (torch.expm1(e) - e)
        dyad = b[:, :, None] * self._fib[:, None, :]
        passive = passive + (stress / stretch)[:, None, None] * dyad
        activation = _tensor([control.left.ta_activation, control.right.ta_activation])[self._element_side]
        active_sigma = activation * p["active_max_pa"] * torch.clamp_min(
            1 - p["active_width"] * (stretch - 1 - p["active_optimal_strain"])**2, 0)
        active = (J * active_sigma / stretch**2)[:, None, None] * dyad
        L = Fdot @ inverse
        D = 0.5 * (L + L.transpose(1, 2))
        viscous = (2 * J * p["viscosity_pa_s"])[:, None, None] * D @ inverse_t
        power = 2 * J * p["viscosity_pa_s"] * (D * D).sum((1, 2))
        for value in (passive, active, viscous, energy, power):
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite constitutive stress/energy (outside representable material response)")
        return passive, viscous, active, energy, power

    def _check_surfaces(self, x):
        for side, tris in zip(("left", "right"), self._surfaces):
            vec = triangle_area_vectors(x, tris)
            norm = torch.linalg.norm(vec, dim=1)
            if not torch.isfinite(vec).all() or torch.any(norm <= 0):
                raise ValueError("degenerate or nonfinite medial surface")
            lateral = vec[:, 1] * (1 if side == "left" else -1) / norm
            threshold = self.config.contact.minimum_lateral_normal if self.config.contact.enabled else 0
            if torch.any(lateral <= threshold):
                raise ValueError("medial surface folded/rotated outside supported lateral graph/contact range")
        if self.config.contact.enabled:
            delta = x[self._pair_right] - x[self._pair_left]
            slip = torch.linalg.norm(delta[:, [0, 2]], dim=1)
            if torch.any(slip > self.config.contact.max_tangential_slip_m):
                raise ValueError("contact registration exceeded tangential slip limit; sliding contact is unsupported")

    def _contact(self, x, v):
        force = torch.zeros_like(x)
        zero = x.new_tensor(0)
        if not self.config.contact.enabled:
            return force, zero, zero
        c = self.config.contact
        il, ir, a = self._pair_left, self._pair_right, self._pair_area
        gap = x[ir, 1] - x[il, 1]
        speed = v[ir, 1] - v[il, 1]
        penetration = torch.clamp_min(-gap, 0)
        closing = torch.clamp_max(speed, 0) * (gap <= 0)
        magnitude = a * (c.stiffness_pa_per_m * penetration - c.damping_pa_s_per_m * closing)
        force[:, 1].index_add_(0, il, -magnitude)
        force[:, 1].index_add_(0, ir, magnitude)
        energy = 0.5 * c.stiffness_pa_per_m * (a * penetration**2).sum()
        power = c.damping_pa_s_per_m * (a * closing**2).sum()
        return force, energy, power

    def _pressure(self, pressure):
        if pressure is None:
            return tuple(torch.zeros(len(t), dtype=torch.float64) for t in self._surfaces)
        if not isinstance(pressure, MedialPressure):
            raise ValueError("pressure must be MedialPressure or None")
        return tuple(_tensor(_array(name, getattr(pressure, name), (len(t),)))
                     for name, t in zip(("left_pa", "right_pa"), self._surfaces))

    def _evaluate(self, x, v, control, pressure):
        F, Fdot, J = self._kinematics(x, v)
        if not torch.isfinite((self._mass[:, None] * v**2).sum()):
            raise ValueError("nonfinite kinetic energy")
        self._check_surfaces(x)
        pp, pv, pa, energy, power = self._constitutive(F, Fdot, control)
        forces = [tetrahedral_nodal_forces(stress, self._volume, self._inverse, self._t, len(x))
                  for stress in (pp, pv, pa)]
        fp = sum((triangle_pressure_forces(x, tri, p) for tri, p in zip(self._surfaces, pressure)),
                 torch.zeros_like(x))
        fc, ec, pc = self._contact(x, v)
        forces.extend((fp, fc))
        if not all(torch.isfinite(f).all() for f in forces):
            raise ValueError("nonfinite assembled force")
        values = (float((self._volume * energy).sum()), float(ec),
                  float((self._volume * power).sum()), float(pc), float(J.min()))
        if not np.isfinite(values).all():
            raise ValueError("nonfinite assembled energy/power")
        return forces, values

    def forces(self, control: BilateralControl | None = None, pressure: MedialPressure | None = None) -> SolidForces:
        """Instantaneous unconstrained forces (including reactions' opposite)."""
        control = self._control if control is None else self._checked_control(control)
        f, _ = self._evaluate(self._x, self._v, control, self._pressure(pressure))
        return SolidForces(*(a.numpy().copy() for a in (*f, sum(f))))

    @staticmethod
    def _checked_control(control):
        if not isinstance(control, BilateralControl) or not isinstance(control.left, FoldControl) or not isinstance(control.right, FoldControl):
            raise ValueError("control must be BilateralControl of FoldControl objects")
        return control

    def _row_bound(self, blocks, dofs):
        masses = self._mass.repeat_interleave(3)[dofs]
        free = (~self._fixed).flatten()[dofs]
        normalized = torch.abs(blocks) / torch.sqrt(masses[:, :, None] * masses[:, None, :])
        normalized = normalized * free[:, :, None] * free[:, None, :]
        rows = torch.zeros(self._X.numel(), dtype=torch.float64)
        rows.index_add_(0, dofs.flatten(), normalized.sum(2).flatten())
        return rows

    def _timestep(self, x, v, control, pressure):
        F, Fdot, _ = self._kinematics(x, v)
        self._check_surfaces(x)
        with torch.enable_grad():
            F = F.detach().requires_grad_(True)
            Fdot = Fdot.detach().requires_grad_(True)
            pp, pv, pa, _, _ = self._constitutive(F, Fdot, control)
            stress = (pp + pv + pa).reshape(-1, 9)
            stiffness, damping = [], []
            for k in range(9):
                a, c = torch.autograd.grad(stress[:, k].sum(), (F, Fdot), retain_graph=k < 8)
                stiffness.append(a.reshape(-1, 9))
                damping.append(c.reshape(-1, 9))
            a = torch.stack(stiffness, dim=1).detach()
            c = torch.stack(damping, dim=1).detach()
            kblock = self._volume[:, None, None] * (self._B.transpose(1, 2) @ a @ self._B)
            cblock = self._volume[:, None, None] * (self._B.transpose(1, 2) @ c @ self._B)
            kr = self._row_bound(kblock, self._dofs)
            cr = self._row_bound(cblock, self._dofs)
            for tris, p in zip(self._surfaces, pressure):
                if not torch.any(p != 0):
                    continue
                coords = x[tris].detach().requires_grad_(True)
                nodal = -p[:, None] / 6 * torch.cross(coords[:, 1] - coords[:, 0],
                                                     coords[:, 2] - coords[:, 0], dim=1)
                jac = torch.stack([torch.autograd.grad(nodal[:, k].sum(), coords, retain_graph=k < 2)[0].reshape(-1, 9)
                                   for k in range(3)], dim=1).detach().repeat(1, 3, 1)
                dofs = (tris[:, :, None] * 3 + torch.arange(3)).reshape(-1, 9)
                kr += self._row_bound(jac, dofs)
        if self.config.contact.enabled:
            dofs = torch.stack((3 * self._pair_left + 1, 3 * self._pair_right + 1), dim=1)
            template = self._pair_area[:, None, None] * _tensor([[1, -1], [-1, 1]])
            kr += self._row_bound(template * self.config.contact.stiffness_pa_per_m, dofs)
            cr += self._row_bound(template * self.config.contact.damping_pa_s_per_m, dofs)
        kmax, cmax = float(kr.max()), float(cr.max())
        if not np.isfinite(kmax) or not np.isfinite(cmax):
            raise ValueError("nonfinite tangent stability bound")
        denominator = np.sqrt(kmax + 0.25 * cmax**2) + 0.5 * cmax
        return float("inf") if denominator == 0 else self.config.timestep_safety * 2 / denominator

    def recommend_timestep(self, control: BilateralControl | None = None,
                           pressure: MedialPressure | None = None) -> float:
        """Local maximum step [s]; includes viscosity/contact and pressure.

        A changed support posture can tighten this bound once its prescribed
        velocity is known in ``step``. Use small continuous posture increments.
        """
        control = self._control if control is None else self._checked_control(control)
        return self._timestep(self._x, self._v, control, self._pressure(pressure))

    def _targets(self, control):
        targets = self._X.clone()
        for mesh, sl, ctl in zip(self._meshes, self._slices, (control.left, control.right)):
            x = self._X[sl]
            ap = x[:, 0] - x[:, 0].min()
            fraction = ap / (x[:, 0].max() - x[:, 0].min())
            offset = (1 - fraction[:, None]) * _tensor(ctl.anterior_displacement_m)
            offset += fraction[:, None] * _tensor(ctl.posterior_displacement_m)
            offset[:, 0] += ctl.longitudinal_strain * ap
            offset += _tensor(mesh.lateral_support, torch.bool) * _tensor(ctl.lateral_displacement_m)
            targets[sl] += offset
        return targets

    def _check_path(self, start, end):
        """Reject an inverted intermediate solid even if both endpoints are valid."""
        f0, _, _ = self._kinematics(start, torch.zeros_like(start))
        f1, _, _ = self._kinematics(end, torch.zeros_like(end))
        df = f1 - f0
        # A perturbation norm <1 proves invertibility along the entire segment.
        bound = torch.linalg.matrix_norm(df @ torch.linalg.inv(f0), ord="fro", dim=(1, 2))
        for a, b in zip(f0[bound >= 1].numpy(), df[bound >= 1].numpy()):
            coefficients = [np.linalg.det(a), 0.0, 0.0, np.linalg.det(b)]
            for col in range(3):
                one, two = a.copy(), b.copy()
                one[:, col], two[:, col] = b[:, col], a[:, col]
                coefficients[1] += np.linalg.det(one)
                coefficients[2] += np.linalg.det(two)
            extrema = np.roots([3 * coefficients[3], 2 * coefficients[2], coefficients[1]])
            for t in extrema:
                if abs(t.imag) < 1e-12 and 0 < t.real < 1:
                    if np.linalg.det(a + t.real * b) <= 0:
                        raise ValueError("nonpositive deformation Jacobian along nodal trajectory")

    def _pressure_advance(self, dt, velocity, targets, forces, pressure):
        other = sum(forces[:3]) + forces[4]

        def advance(fp):
            v = velocity + dt * (other + fp) / self._mass[:, None]
            v[self._fixed] = velocity[self._fixed]
            x = self._x + dt * v
            x[self._fixed] = targets[self._fixed]
            return x, v

        x, v = advance(forces[3])
        if not any(torch.any(p != 0) for p in pressure):
            sweeps = tuple(_sweep_tensors(self._x, x, tri) for tri in self._surfaces)
            return x, v, forces[3], sweeps, 0, 0.0
        for iteration in range(1, self.config.pressure_max_iterations + 1):
            sweeps = tuple(_sweep_tensors(self._x, x, tri) for tri in self._surfaces)
            fp = torch.zeros_like(x)
            for tri, p, (area, _, _) in zip(self._surfaces, pressure, sweeps):
                nodal = -p[:, None] * area / 3
                for k in range(3):
                    fp.index_add_(0, tri[:, k], nodal)
            proposed_x, proposed_v = advance(fp)
            residual = float(torch.abs(proposed_x - x).max())
            if residual <= self.config.pressure_tolerance_m:
                return x, v, fp, sweeps, iteration, residual
            x, v = proposed_x, proposed_v
        raise ValueError("pressure follower-traction fixed point did not converge; retry smaller dt")

    def step(self, dt_s: float, control: BilateralControl | None = None,
             pressure: MedialPressure | None = None) -> SolidStep:
        """Advance once, atomically. Pressure omitted means zero; controls persist."""
        _finite("dt_s", dt_s, minimum=0, strict=True)
        control = self._control if control is None else self._checked_control(control)
        pressure = self._pressure(pressure)
        targets = self._targets(control)
        velocity = self._v.clone()
        velocity[self._fixed] = (targets[self._fixed] - self._x[self._fixed]) / dt_s
        limit = self._timestep(self._x, velocity, control, pressure)
        if dt_s > limit * (1 + 1e-12):
            raise ValueError(f"dt_s={dt_s:g} exceeds local stable recommendation {limit:g}")
        f, values = self._evaluate(self._x, velocity, control, pressure)
        new_x, new_v, f[3], sweeps, iterations, residual = self._pressure_advance(
            dt_s, velocity, targets, f, pressure)
        self._check_path(self._x, new_x)
        _, endpoint = self._evaluate(new_x, new_v, control, pressure)
        endpoint_limit = self._timestep(new_x, new_v, control, pressure)
        if dt_s > endpoint_limit * (1 + 1e-12):
            raise ValueError(f"candidate tightens stable recommendation to {endpoint_limit:g}; retry smaller dt")
        total = sum(f)
        dx = new_x - self._x
        reaction = torch.zeros_like(total)
        reaction[self._fixed] = (((new_v - self._v) * self._mass[:, None] / dt_s) - total)[self._fixed]
        support_work = (0.5 * self._mass[:, None] * (new_v**2 - self._v**2) - total * dx)[self._fixed].sum()
        old = self._work
        work = WorkTotals(old.viscous_dissipated_j + dt_s * values[2],
                          old.contact_dissipated_j + dt_s * values[3],
                          old.pressure_j + float((f[3] * dx).sum()),
                          old.active_j + float((f[2] * dx).sum()),
                          old.support_j + float(support_work))
        if not all(np.isfinite(getattr(work, k)) for k in WorkTotals.__dataclass_fields__):
            raise ValueError("nonfinite accumulated work")
        _finite("next time", self.time_s + dt_s)
        self._x, self._v, self._control, self._work = new_x, new_v, control, work
        self.time_s += dt_s
        outputs = tuple(SurfaceSweep(*(a.numpy().copy() for a in sweep)) for sweep in sweeps)
        return SolidStep(self.time_s, self._diagnostics(endpoint), reaction.numpy().copy(),
                         *outputs, iterations, residual)

    def _diagnostics(self, values):
        strain, contact, viscous, contact_power, minj = values
        kinetic = float((0.5 * self._mass[:, None] * self._v**2).sum())
        w = self._work
        residual = (kinetic + strain + contact - self._initial_energy
                    + w.viscous_dissipated_j + w.contact_dissipated_j
                    - w.pressure_j - w.active_j - w.support_j)
        return SolidDiagnostics(kinetic, strain, contact, viscous, contact_power, minj, w, residual)

    def diagnostics(self) -> SolidDiagnostics:
        _, values = self._evaluate(self._x, self._v, self._control, self._pressure(None))
        return self._diagnostics(values)

    def save_state(self) -> SolidState:
        """Owned numpy snapshots; reset accepts only this mesh/material contract."""
        return SolidState(self._signature, self.time_s, self.positions_m, self.velocities_m_s,
                          self._control, self._work, self._initial_energy)

    def initialize(self, positions_m: np.ndarray, velocities_m_s: np.ndarray | None = None,
                   control: BilateralControl | None = None) -> None:
        """Set initial conditions with a fresh time/work/energy ledger.

        Initial positions must satisfy the specified support posture. Use
        ``reset(snapshot)`` instead to resume existing work and time history.
        """
        x = _array("positions_m", positions_m, tuple(self._X.shape))
        v = np.zeros_like(x) if velocities_m_s is None else velocities_m_s
        ctl = BilateralControl() if control is None else self._checked_control(control)
        self.reset(SolidState(self._signature, 0.0, x, v, ctl, WorkTotals(), 0.0))
        diagnostics = self.diagnostics()
        self._initial_energy = diagnostics.kinetic_j + diagnostics.strain_j + diagnostics.contact_j

    def reset(self, state: SolidState | None = None) -> None:
        """Reset to rest or restore a snapshot, validating it before mutation."""
        if state is None:
            x, v, control, work, time = self._X.clone(), torch.zeros_like(self._X), BilateralControl(), WorkTotals(), 0.0
            initial = None
        else:
            if not isinstance(state, SolidState) or state.mesh_signature != self._signature:
                raise ValueError("state mesh/material signature mismatch")
            x = _tensor(_array("positions_m", state.positions_m, tuple(self._X.shape)))
            v = _tensor(_array("velocities_m_s", state.velocities_m_s, tuple(self._X.shape)))
            control = self._checked_control(state.control)
            _finite("time_s", state.time_s, minimum=0)
            _finite("initial_mechanical_j", state.initial_mechanical_j, minimum=0)
            if not isinstance(state.work, WorkTotals):
                raise ValueError("state work must be WorkTotals")
            for name in WorkTotals.__dataclass_fields__:
                _finite(name, getattr(state.work, name), minimum=0 if "dissipated" in name else None)
            work, time, initial = state.work, state.time_s, state.initial_mechanical_j
        if not torch.allclose(x[self._fixed], self._targets(control)[self._fixed], rtol=0, atol=1e-12):
            raise ValueError("state positions do not satisfy prescribed support posture")
        _, values = self._evaluate(x, v, control, self._pressure(None))
        mechanical = float((0.5 * self._mass[:, None] * v**2).sum()) + values[0] + values[1]
        _finite("mechanical energy", mechanical)
        self._x, self._v, self._control, self._work, self.time_s = x, v, control, work, time
        self._initial_energy = mechanical if initial is None else initial

    def surface(self, side: Side) -> MedialSurface:
        """Side-local triangle indices into the returned full side coordinates."""
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        i = 0 if side == "left" else 1
        tris, sl = self._surfaces[i], self._slices[i]
        area = triangle_area_vectors(self._x, tris).numpy()
        return MedialSurface(self._x[sl].numpy().copy(), self._v[sl].numpy().copy(),
                             self._meshes[i].medial_triangles.copy(),
                             self._x[tris].mean(1).numpy(), area,
                             np.linalg.norm(area, axis=1), np.full((len(tris), 3), 1 / 3))

    def surface_sweep(self, start: SolidState, end: SolidState, side: Side) -> SurfaceSweep:
        """Replay a snapshot-to-snapshot medial sweep without modifying state."""
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        for state in (start, end):
            if not isinstance(state, SolidState) or state.mesh_signature != self._signature:
                raise ValueError("sweep state mesh/material signature mismatch")
            _array("positions_m", state.positions_m, tuple(self._X.shape))
        i = 0 if side == "left" else 1
        sl = self._slices[i]
        return triangle_surface_sweep(start.positions_m[sl], end.positions_m[sl],
                                      self._meshes[i].medial_triangles)

    def gap_profile(self, si_m: np.ndarray | None = None) -> GlottalProfile:
        """Actual horizontal slices, preserving AP detail (no first-mode projection).

        Default planes use the union of medial vertex SI levels in the common
        SI extent. AP coverage must agree between sides; no silent truncation.
        Exact piecewise-linear positive-gap integration adds closure crossings.
        """
        self._check_surfaces(self._x)
        surfaces = [self._x[t].numpy() for t in self._surfaces]
        low = max(s[:, :, 2].min() for s in surfaces)
        high = min(s[:, :, 2].max() for s in surfaces)
        if high <= low:
            raise ValueError("medial surfaces have no common SI extent")
        if si_m is None:
            z = np.unique(np.concatenate([s[:, :, 2].ravel() for s in surfaces]))
            z = z[(z >= low) & (z <= high)]
        else:
            z = np.asarray(si_m, dtype=float)
            if z.ndim != 1 or not len(z) or not np.isfinite(z).all() or np.any(np.diff(z) <= 0):
                raise ValueError("si_m must be nonempty finite strictly increasing planes")
            if z[0] < low - 1e-12 or z[-1] > high + 1e-12:
                raise ValueError("SI plane outside shared medial extent")
        slices = []
        for plane in z:
            curves = [_slice_curve(s, plane) for s in surfaces]
            left, right = curves
            if not np.allclose(left[[0, -1], 0], right[[0, -1], 0], rtol=0, atol=1e-10):
                raise ValueError("unequal AP coverage; explicit fluid end-boundary geometry is required")
            ap = np.unique(np.concatenate((left[:, 0], right[:, 0])))
            gap = np.interp(ap, right[:, 0], right[:, 1]) - np.interp(ap, left[:, 0], left[:, 1])
            crossing = gap[:-1] * gap[1:] < 0
            roots = ap[:-1][crossing] - gap[:-1][crossing] * np.diff(ap)[crossing] / np.diff(gap)[crossing]
            if len(roots):
                combined = np.unique(np.concatenate((ap, roots)))
                gap = np.interp(combined, ap, gap)
                ap = combined
            weights = np.zeros_like(ap)
            weights[:-1] += np.diff(ap) / 2
            weights[1:] += np.diff(ap) / 2
            area = float(weights @ np.maximum(gap, 0))
            slices.append(GapSlice(float(plane), ap, gap, weights, area))
        return GlottalProfile(z.copy(), np.asarray([s.area_m2 for s in slices]), tuple(slices))


def _slice_curve(triangles, plane):
    """Intersect a triangulated lateral graph with a horizontal SI plane."""
    segments = []
    tolerance = 1e-12
    for tri in triangles:
        points = []
        for k in range(3):
            a, b = tri[k], tri[(k + 1) % 3]
            da, db = a[2] - plane, b[2] - plane
            if abs(da) <= tolerance:
                points.append(a[:2])
            if da * db < 0:
                points.append((a + (b - a) * (-da / (db - da)))[:2])
        if len(points) < 2:
            continue
        points = np.asarray(points)
        a, b = points[np.argmin(points[:, 0])], points[np.argmax(points[:, 0])]
        if b[0] - a[0] > tolerance:
            segments.append((a, b))
    if not segments:
        raise ValueError("SI plane has no resolvable medial AP curve")
    knots = np.unique(np.concatenate([np.asarray(s)[:, 0] for s in segments]))
    values = []
    for ap in knots:
        ys = [a[1] + (b[1] - a[1]) * (ap - a[0]) / (b[0] - a[0])
              for a, b in segments if a[0] - tolerance <= ap <= b[0] + tolerance]
        if not ys or np.ptp(ys) > 1e-10:
            raise ValueError("medial slice is disconnected or multivalued along AP")
        values.append(np.mean(ys))
    for a, b in zip(knots[:-1], knots[1:]):
        mid = (a + b) / 2
        ys = [u[1] + (w[1] - u[1]) * (mid - u[0]) / (w[0] - u[0])
              for u, w in segments if u[0] - tolerance <= mid <= w[0] + tolerance]
        if not ys or np.ptp(ys) > 1e-10:
            raise ValueError("medial slice is disconnected or multivalued along AP")
    return np.column_stack((knots, values))
