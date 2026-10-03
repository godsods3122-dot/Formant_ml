"""Independent mechanics checks, not validation against human anatomy/audio."""
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.linalg import eigh

from formant_ml.physics import softtissue as st
from formant_ml.physics.vocal_fold_solid import (
    BilateralControl,
    ContactConfig,
    FoldControl,
    MedialPressure,
    PressurePatch,
    SolidConfig,
    SolidMaterial,
    VocalFoldSolid,
    nominal_fold_mesh,
    nominal_materials,
    triangle_surface_sweep,
)


@pytest.fixture(scope="module", autouse=True)
def two_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def make_solid(*, n_ap=2, n_si=1, contact=True, free=False, materials=None, half_gap=2e-4):
    meshes = [nominal_fold_mesh(side, n_ap=n_ap, n_si=n_si, half_gap_m=half_gap,
                               fix_anterior=not free, fix_posterior=not free, fix_lateral=not free)
              for side in ("left", "right")]
    return VocalFoldSolid(SolidConfig(*meshes, materials=materials or nominal_materials(),
                                     contact=ContactConfig(enabled=contact)))


def load_state(s, x=None, v=None):
    state = s.save_state()
    s.initialize(state.positions_m if x is None else x, state.velocities_m_s if v is None else v,
                 state.control)


def pressure(s, left=100, right=100):
    return MedialPressure(np.full(len(s.surface("left").triangles), left, dtype=float),
                          np.full(len(s.surface("right").triangles), right, dtype=float))


def total_energy(s):
    d = s.diagnostics()
    return d.kinetic_j + d.strain_j + d.contact_j


def test_independent_mass_material_assembly_and_shared_interfaces():
    mats = tuple(replace(m, density_kg_m3=rho) for m, rho in zip(nominal_materials(), (900, 1050, 1100)))
    s = make_solid(materials=mats)
    expected_mass = np.zeros(len(s.positions_m))
    expected_total = 0
    for mesh, offset in zip(s._meshes, (0, s._offset)):
        for tet, region in zip(mesh.tetrahedra, mesh.material_ids):
            p = mesh.reference_m[tet]
            volume = np.dot(p[1] - p[0], np.cross(p[2] - p[0], p[3] - p[0])) / 6
            expected_mass[tet + offset] += mats[region].density_kg_m3 * volume / 4
            expected_total += volume * mats[region].density_kg_m3
        for region in (0, 1):
            a = np.unique(mesh.tetrahedra[mesh.material_ids == region])
            b = np.unique(mesh.tetrahedra[mesh.material_ids == region + 1])
            shared = np.intersect1d(a, b)
            assert len(shared) == 3 * 2
            assert len(np.unique(mesh.reference_m, axis=0)) == len(mesh.reference_m)
    np.testing.assert_allclose(s.nodal_mass_kg, expected_mass, rtol=1e-14)
    assert s.nodal_mass_kg.sum() == pytest.approx(expected_total, rel=1e-14)
    analytic = 2 * 0.01 * 0.002 * np.dot([0.0015, 0.0015, 0.003], [900, 1050, 1100])
    assert expected_total == pytest.approx(analytic, rel=1e-14)


def test_passive_energy_gradient_balance_and_positive_stiffness():
    s = make_solid(contact=False, free=True)
    state = s.save_state()
    F = np.array([[1.03, 0.01, 0.004], [0, 0.99, 0.003], [0, 0, 0.985]])
    x = state.positions_m @ F.T
    load_state(s, x)
    f = s.forces().passive_n
    np.testing.assert_allclose(f.sum(axis=0), 0, atol=1e-15)
    np.testing.assert_allclose(np.cross(x, f).sum(axis=0), 0, atol=1e-16)
    rng = np.random.default_rng(2002)
    direction = rng.normal(size=x.shape)
    direction /= np.linalg.norm(direction)
    h = 1e-8
    load_state(s, x + h * direction)
    eplus = s.diagnostics().strain_j
    load_state(s, x - h * direction)
    eminus = s.diagnostics().strain_j
    assert -(f * direction).sum() == pytest.approx((eplus - eminus) / (2 * h), rel=2e-6, abs=1e-10)
    s.reset()
    load_state(s, state.positions_m + h * direction)
    assert s.diagnostics().strain_j > 0
    assert -(s.forces().passive_n * direction).sum() > 0


def test_rest_without_activation_does_not_create_energy():
    s = make_solid()
    dt = s.recommend_timestep() * 0.3
    for _ in range(8):
        s.step(dt)
    assert total_energy(s) < 1e-25
    assert s.diagnostics().work.active_j == 0
    np.testing.assert_allclose(s.forces().active_n, 0, atol=0)


@pytest.mark.parametrize("degrees", [1.0, 7.0, 23.0, -41.0, 89.0])
def test_free_rigid_rotation_own_snapshot_reset_and_replay(degrees):
    s = make_solid(contact=False, free=True)
    angle = np.deg2rad(degrees)
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    s.initialize(s.positions_m @ rotation.T)
    initial = s.save_state()
    assert s.diagnostics().minimum_jacobian == pytest.approx(1.0, abs=1e-13)
    assert 0 <= initial.initial_mechanical_j < s._energy_roundoff_j
    s.reset(initial)
    assert s.save_state().initial_mechanical_j == initial.initial_mechanical_j
    np.testing.assert_array_equal(s.positions_m, initial.positions_m)
    np.testing.assert_array_equal(s.velocities_m_s, initial.velocities_m_s)
    p = pressure(s, 150, 50)
    dt = 0.1 * s.recommend_timestep(pressure=p)
    s.step(dt, pressure=p)
    evolved = s.save_state()
    s.reset(initial)
    s.step(dt, pressure=p)
    replay = s.save_state()
    np.testing.assert_array_equal(replay.positions_m, evolved.positions_m)
    np.testing.assert_array_equal(replay.velocities_m_s, evolved.velocities_m_s)
    assert replay.work == evolved.work
    assert replay.time_s == evolved.time_s
    assert replay.initial_mechanical_j == initial.initial_mechanical_j


def test_historical_roundoff_baseline_preserved_and_negative_energy_rejected():
    s = make_solid(contact=False, free=True)
    p = pressure(s)
    s.step(s.recommend_timestep(pressure=p) * 0.1, pressure=p)
    # Baseline produced by the old trace/log evaluation at a 1-degree rotation.
    historical = replace(s.save_state(), initial_mechanical_j=-1.8873791418627416e-20)
    s.reset(historical)
    restored = s.save_state()
    assert restored.initial_mechanical_j == historical.initial_mechanical_j
    assert restored.work == historical.work
    assert restored.time_s == historical.time_s
    np.testing.assert_array_equal(restored.positions_m, historical.positions_m)
    np.testing.assert_array_equal(restored.velocities_m_s, historical.velocities_m_s)
    with pytest.raises(ValueError, match="initial_mechanical_j"):
        s.reset(replace(historical, initial_mechanical_j=-100 * s._energy_roundoff_j))
    after = s.save_state()
    assert after.initial_mechanical_j == restored.initial_mechanical_j
    assert after.work == restored.work
    assert after.time_s == restored.time_s
    np.testing.assert_array_equal(after.positions_m, restored.positions_m)
    np.testing.assert_array_equal(after.velocities_m_s, restored.velocities_m_s)


def test_active_stress_measure_and_side_specific_activation():
    s = make_solid(contact=False, free=True)
    F = torch.diag(torch.tensor([1.1, 0.96, 0.97], dtype=torch.float64)).expand(len(s._t), -1, -1)
    controls = BilateralControl(left=FoldControl(ta_activation=0.4))
    _, _, active, _, _ = s._constitutive(F, torch.zeros_like(F), controls)
    sigma = active @ F.transpose(1, 2) / torch.linalg.det(F)[:, None, None]
    p = s._parameters
    expected = 0.4 * p["active_max_pa"] * torch.clamp_min(
        1 - p["active_width"] * (0.1 - p["active_optimal_strain"])**2, 0)
    expected[s._element_side == 1] = 0
    torch.testing.assert_close(sigma[:, 0, 0], expected)
    torch.testing.assert_close(sigma[:, 1:], torch.zeros_like(sigma[:, 1:]))
    f = s.forces(controls).active_n
    assert np.linalg.norm(f[:s._offset]) > 0
    np.testing.assert_array_equal(f[s._offset:], 0)
    assert np.sum(f * s.velocities_m_s) == 0


def test_viscous_force_work_and_unclipped_small_positive_jacobian():
    s = make_solid(contact=False, free=True)
    x = s.positions_m
    x[:, 0] *= 0.1
    v = np.random.default_rng(99).normal(size=x.shape) * 0.01
    load_state(s, x, v)
    d = s.diagnostics()
    assert d.minimum_jacobian == pytest.approx(0.1, rel=1e-13)
    assert d.viscous_power_w > 0
    force = s.forces()
    assert np.sum(force.viscous_n * v) == pytest.approx(-d.viscous_power_w, rel=1e-13)
    F, Fdot, _ = s._kinematics(s._x, s._v)
    passive, _, _, _, _ = s._constitutive(F, Fdot, s.save_state().control)
    mu = s._parameters["matrix_mu_pa"]
    lam = s._parameters["matrix_bulk_pa"] - 2 * mu / 3
    expected = mu * (0.1 - 10) + lam * np.log(0.1) * 10
    torch.testing.assert_close(passive[:, 0, 0], expected, atol=1e-8, rtol=1e-12)


def test_pressure_quadrature_total_load_and_virtual_work():
    s = make_solid(contact=False, free=True)
    x = s.positions_m
    x[:, 0] *= 1.04
    x[:, 2] *= 0.97
    load_state(s, x)
    rng = np.random.default_rng(101)
    p = MedialPressure(rng.uniform(-500, 2000, len(s.surface("left").triangles)),
                       rng.uniform(-500, 2000, len(s.surface("right").triangles)))
    f = s.forces(pressure=p).pressure_n
    virtual = rng.normal(size=x.shape) * 1e-6
    work = 0
    total = np.zeros(3)
    for side, sl, values in zip(("left", "right"), s._slices, (p.left_pa, p.right_pa)):
        surface = s.surface(side)
        triangles = surface.coordinates_m[surface.triangles]
        independent_area = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) / 2
        np.testing.assert_allclose(surface.area_vectors_m2, independent_area, atol=0)
        np.testing.assert_allclose(surface.centroids_m, triangles.mean(axis=1))
        nodal_displacement = virtual[sl][surface.triangles].mean(axis=1)
        work -= np.sum(values[:, None] * independent_area * nodal_displacement)
        total -= np.sum(values[:, None] * independent_area, axis=0)
    assert np.sum(f * virtual) == pytest.approx(work, rel=1e-14, abs=1e-20)
    np.testing.assert_allclose(f.sum(axis=0), total, atol=1e-16)


@pytest.mark.parametrize("motion", ["translation", "rotation", "deformation"])
def test_closed_polyhedron_sweep_equals_independent_signed_volume_change(motion):
    x = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.02, 0], [0, 0, 0.015]])
    tris = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    if motion == "translation":
        y = x + [0.003, -0.002, 0.001]
    elif motion == "rotation":
        angle = 0.7
        transform = np.array([[np.cos(angle), -np.sin(angle), 0],
                              [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        y = x @ transform.T
    else:
        transform = np.array([[1.2, 0.3, 0], [0, 0.9, 0.1], [0.1, 0, 1.1]])
        y = x @ transform.T + [0.001, -0.002, 0.003]
    result = triangle_surface_sweep(x, y, tris)
    v0 = np.linalg.det((x[1:] - x[0]).T) / 6
    v1 = np.linalg.det((y[1:] - y[0]).T) / 6
    assert result.outward_swept_volume_m3.sum() == pytest.approx(v1 - v0, rel=1e-13, abs=2e-21)
    if motion == "translation":
        expected = np.tile([0.003, -0.002, 0.001], (len(tris), 1))
        np.testing.assert_allclose(result.centroid_displacement_m, expected, atol=1e-18)
    reverse = triangle_surface_sweep(y, x, tris)
    np.testing.assert_allclose(reverse.outward_swept_volume_m3, -result.outward_swept_volume_m3, atol=1e-21)


def test_pressure_step_exact_swept_work_and_snapshot_replay():
    s = make_solid(n_ap=3, n_si=2, contact=False)
    surface = s.surface("left")
    p = MedialPressure(np.linspace(100, 1200, len(surface.triangles)),
                       np.linspace(300, 50, len(s.surface("right").triangles)))
    start = s.save_state()
    dt = s.recommend_timestep(pressure=p) * 0.3
    result = s.step(dt, pressure=p)
    end = s.save_state()
    exact_work = -p.left_pa @ result.left_sweep.outward_swept_volume_m3
    exact_work -= p.right_pa @ result.right_sweep.outward_swept_volume_m3
    assert end.work.pressure_j == pytest.approx(exact_work, rel=2e-13, abs=1e-22)
    for side, sweep in (("left", result.left_sweep), ("right", result.right_sweep)):
        replay = s.surface_sweep(start, end, side)
        np.testing.assert_array_equal(replay.outward_swept_volume_m3, sweep.outward_swept_volume_m3)
    assert result.pressure_iterations >= 1
    assert result.pressure_residual_m <= s.config.pressure_tolerance_m
    s.reset(start)
    repeated = s.step(dt, pressure=p)
    np.testing.assert_array_equal(s.positions_m, end.positions_m)
    assert repeated.diagnostics.work.pressure_j == end.work.pressure_j


def test_pressure_fixed_point_failure_is_atomic():
    template = make_solid(n_ap=3, n_si=2, contact=False)
    s = VocalFoldSolid(replace(template.config, pressure_max_iterations=1, pressure_tolerance_m=1e-25))
    p = pressure(s, 2000, 50)
    start = s.save_state()
    with pytest.raises(ValueError, match="fixed point"):
        s.step(s.recommend_timestep(pressure=p) * 0.3, pressure=p)
    assert s.time_s == start.time_s
    assert s.save_state().work == start.work
    np.testing.assert_array_equal(s.positions_m, start.positions_m)
    np.testing.assert_array_equal(s.velocities_m_s, start.velocities_m_s)


def _inferior_triangle(s, side="left"):
    i = 0 if side == "left" else 1
    x = s._meshes[i].reference_m
    return next(tuple(t) for t in s.boundary_triangles(side) if np.all(x[t, 2] == 0))


def _quarter_patches(side, triangle, value):
    vertices = (
        ((1, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5)),
        ((0.5, 0.5, 0), (0, 1, 0), (0, 0.5, 0.5)),
        ((0.5, 0, 0.5), (0, 0.5, 0.5), (0, 0, 1)),
        ((0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5)),
    )
    return tuple(PressurePatch(side, triangle, value, v) for v in vertices)


def test_patch_actual_geometry_nodal_quadrature_and_virtual_work():
    s = make_solid(contact=False, free=True)
    x = s.positions_m @ np.array([[1.04, 0.02, 0], [0.01, 0.98, 0.02], [0, 0.01, 1.03]]).T
    s.initialize(x)
    triangle = tuple(s.surface("right").triangles[-1])
    patches = (PressurePatch("left", _inferior_triangle(s), 1300.0),
               PressurePatch("right", triangle, -250.0,
                             ((0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.2, 0.1, 0.7))))
    actual = s.forces(pressure_patches=patches).pressure_n
    expected = np.zeros_like(x)
    delta = np.random.default_rng(140).normal(size=x.shape) * 1e-6
    work = 0.0
    for patch in patches:
        nodes = np.asarray(patch.triangle_nodes) + (s._offset if patch.side == "right" else 0)
        barycentric = np.asarray(patch.barycentric_vertices)
        vertices = barycentric @ x[nodes]
        area = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]) / 2
        local_force = np.tile(-patch.pressure_pa * area / 3, (3, 1))
        expected[nodes] += barycentric.T @ local_force
        work += np.sum(local_force * (barycentric @ delta[nodes]))
    np.testing.assert_allclose(actual, expected, atol=1e-17, rtol=1e-14)
    assert np.sum(actual * delta) == pytest.approx(work, rel=1e-14)
    np.testing.assert_allclose(s.forces(pressure_patches=patches[::-1]).pressure_n, expected, atol=1e-17)


def test_partitioned_patches_equal_full_triangle_and_share_edges_without_overlap():
    s = make_solid(contact=False, free=True)
    triangle = tuple(s.surface("left").triangles[1])
    quarters = _quarter_patches("left", triangle, 1700.0)
    full = (PressurePatch("left", triangle, 1700.0),)
    medial = pressure(s, 0, 0)
    medial.left_pa[1] = 1700.0
    np.testing.assert_allclose(s.forces(pressure=medial).pressure_n,
                               s.forces(pressure_patches=full).pressure_n, atol=1e-17, rtol=1e-14)
    np.testing.assert_allclose(s.forces(pressure_patches=quarters).pressure_n,
                               s.forces(pressure_patches=full).pressure_n, atol=1e-17, rtol=1e-14)
    assert s.recommend_timestep(pressure_patches=quarters) == pytest.approx(
        s.recommend_timestep(pressure_patches=full), rel=1e-14)
    start = s.save_state()
    x = s.positions_m
    x[:s._offset, 2] += 0.02 * x[:s._offset, 0]
    x[:, 0] *= 1.03
    s.initialize(x)
    end = s.save_state()
    quarter_sweep = s.pressure_patch_sweep(start, end, quarters)
    full_sweep = s.pressure_patch_sweep(start, end, full)
    assert quarter_sweep.outward_swept_volume_m3.sum() == pytest.approx(
        full_sweep.outward_swept_volume_m3[0], rel=1e-14, abs=1e-24)
    reversed_sweep = s.pressure_patch_sweep(start, end, quarters[::-1])
    np.testing.assert_array_equal(reversed_sweep.outward_swept_volume_m3,
                                  quarter_sweep.outward_swept_volume_m3[::-1])


def test_nonbinary_shared_edge_partition_has_no_interior_overlap():
    s = make_solid(contact=False)
    triangle = tuple(s.surface("left").triangles[0])
    t = 0.37
    a, b, c = (1, 0, 0), (0, 1, 0), (0, 0, 1)
    p, q = (1 - t, t, 0), (1 - t, 0, t)
    patches = tuple(PressurePatch("left", triangle, 500, vertices)
                    for vertices in ((a, p, q), (p, b, c), (p, c, q)))
    np.testing.assert_allclose(s.forces(pressure_patches=patches).pressure_n,
                               s.forces(pressure_patches=(PressurePatch("left", triangle, 500),)).pressure_n,
                               atol=1e-17, rtol=1e-14)


def test_patch_step_and_fixed_material_sweep_work_in_input_order():
    s = make_solid(n_ap=3, n_si=2, contact=False, free=True)
    triangle = tuple(s.surface("right").triangles[-2])
    patches = (PressurePatch("left", _inferior_triangle(s), 700.0),
               _quarter_patches("right", triangle, 250.0)[2])
    start = s.save_state()
    dt = s.recommend_timestep(pressure_patches=patches) * 0.3
    result = s.step(dt, pressure_patches=patches)
    end = s.save_state()
    assert result.patch_sweep is not None
    replay = s.pressure_patch_sweep(start, end, patches)
    np.testing.assert_array_equal(result.patch_sweep.outward_swept_volume_m3,
                                  replay.outward_swept_volume_m3)
    assert end.work.pressure_j == pytest.approx(
        -np.array([p.pressure_pa for p in patches]) @ replay.outward_swept_volume_m3,
        rel=1e-13, abs=1e-23)
    for j, patch in enumerate(patches):
        nodes = np.asarray(patch.triangle_nodes) + (s._offset if patch.side == "right" else 0)
        weights = np.asarray(patch.barycentric_vertices)
        before, after = weights @ start.positions_m[nodes], weights @ end.positions_m[nodes]
        independent = triangle_surface_sweep(before, after, np.array([[0, 1, 2]]))
        np.testing.assert_allclose(replay.mean_area_vectors_m2[j], independent.mean_area_vectors_m2[0],
                                   rtol=1e-13, atol=1e-21)
        assert replay.outward_swept_volume_m3[j] == pytest.approx(
            independent.outward_swept_volume_m3[0], rel=1e-9, abs=1e-22)
    s.reset(start)
    s.step(dt, pressure_patches=patches)
    np.testing.assert_array_equal(s.positions_m, end.positions_m)


def test_closed_boundary_patches_match_independent_tetrahedral_volume_change():
    s = make_solid(contact=False, free=True)
    patches = tuple(PressurePatch(side, tuple(tri), 500.0) for side in ("left", "right")
                    for tri in s.boundary_triangles(side))
    start = s.save_state()
    np.testing.assert_allclose(s.forces(pressure_patches=patches).pressure_n.sum(axis=0), 0, atol=1e-16)
    x = s.positions_m @ np.array([[1.03, 0.01, 0], [0, 0.99, 0.01], [0, 0, 1.02]]).T
    s.initialize(x)
    result = s.pressure_patch_sweep(start, s.save_state(), patches)
    expected = 0.0
    for mesh, offset in zip(s._meshes, (0, s._offset)):
        initial = start.positions_m[mesh.tetrahedra + offset]
        final = x[mesh.tetrahedra + offset]
        expected += (np.linalg.det(final[:, 1:] - final[:, :1]).sum()
                     - np.linalg.det(initial[:, 1:] - initial[:, :1]).sum()) / 6
    assert result.outward_swept_volume_m3.sum() == pytest.approx(expected, rel=1e-12, abs=1e-22)


def test_patch_overlap_rejects_actual_interior_not_just_excess_area():
    s = make_solid(contact=False)
    triangle = tuple(s.surface("left").triangles[0])
    quarter = _quarter_patches("left", triangle, 100.0)[0]
    shifted = PressurePatch("left", triangle, 100.0,
                            ((0.8, 0.1, 0.1), (0.3, 0.6, 0.1), (0.3, 0.1, 0.6)))
    assert np.linalg.det(quarter.barycentric_vertices) + np.linalg.det(shifted.barycentric_vertices) < 1
    with pytest.raises(ValueError, match="overlap"):
        s.forces(pressure_patches=(quarter, shifted))
    rotated = PressurePatch("left", triangle[1:] + triangle[:1], 100.0,
                            np.asarray(quarter.barycentric_vertices)[:, [1, 2, 0]])
    before = s.save_state()
    with pytest.raises(ValueError, match="overlap"):
        s.step(1e-7, pressure_patches=(quarter, rotated))
    np.testing.assert_array_equal(s.positions_m, before.positions_m)
    np.testing.assert_array_equal(s.velocities_m_s, before.velocities_m_s)
    assert s.save_state().work == before.work
    assert s.time_s == before.time_s
    with pytest.raises(ValueError, match="whole-medial"):
        s.forces(pressure=pressure(s), pressure_patches=(quarter,))


@pytest.mark.parametrize("vertices", [
    ((1, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((1, 0, 0), (1, 0, 0), (0, 0, 1)),
    ((1, 0, 0), (-0.1, 1.1, 0), (0, 0, 1)),
    ((1, 0, 0), (0, 0.9, 0), (0, 0, 1)),
    ((float("nan"), 0, 0), (0, 1, 0), (0, 0, 1)),
])
def test_invalid_patch_barycentric_geometry(vertices):
    with pytest.raises(ValueError):
        PressurePatch("left", (0, 1, 2), 100, vertices)


def test_patch_exterior_winding_immutability_and_stability_checks():
    s = make_solid(contact=False, free=True)
    triangle = tuple(s.surface("left").triangles[0])
    with pytest.raises(ValueError, match="outward"):
        s.forces(pressure_patches=(PressurePatch("left", triangle[::-1], 100),))
    with pytest.raises(ValueError, match="exterior"):
        s.forces(pressure_patches=(PressurePatch("left", (10000, 10001, 10002), 100),))
    owners = {}
    for t in s._meshes[0].tetrahedra:
        for k in range(4):
            face = tuple(sorted(np.delete(t, k)))
            owners[face] = owners.get(face, 0) + 1
    interior = next(face for face, count in owners.items() if count == 2)
    with pytest.raises(ValueError, match="exterior"):
        s.forces(pressure_patches=(PressurePatch("left", interior, 100),))
    vertices = np.eye(3)
    patch = PressurePatch("left", triangle, 1e11, vertices)
    vertices[:] = 0
    assert patch.barycentric_vertices == ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert s.recommend_timestep(pressure_patches=(patch,)) < s.recommend_timestep() / 10
    before = s.save_state()
    with pytest.raises(ValueError, match="stable"):
        s.step(s.recommend_timestep(), pressure_patches=(patch,))
    np.testing.assert_array_equal(s.positions_m, before.positions_m)
    with pytest.raises(ValueError):
        PressurePatch("left", triangle, float("inf"))
    with pytest.raises(ValueError, match="real"):
        PressurePatch("left", triangle, 100 + 20j)
    with pytest.raises(ValueError, match="real"):
        PressurePatch("left", triangle, 100, np.eye(3, dtype=complex))


def test_contact_action_reaction_energy_damping_and_separation():
    s = make_solid(free=True, half_gap=1e-5)
    x = s.positions_m
    x[:s._offset, 1] += 2e-5
    x[s._offset:, 1] -= 2e-5
    v = np.zeros_like(x)
    v[:s._offset, 1] = 0.01
    v[s._offset:, 1] = -0.02
    load_state(s, x, v)
    f = s.forces().contact_n
    np.testing.assert_allclose(f.sum(axis=0), 0, atol=1e-16)
    np.testing.assert_allclose(f[s._pair_left, 1], -f[s._pair_right, 1])
    area = 0.01 * 0.002
    k, c, penetration = 5e7, 50.0, 2e-5
    assert -f[:s._offset, 1].sum() == pytest.approx(area * (k * penetration + c * 0.03))
    assert s.diagnostics().contact_j == pytest.approx(0.5 * k * area * penetration**2)
    assert s.diagnostics().contact_dissipation_w == pytest.approx(c * area * 0.03**2)
    load_state(s, x, -v)
    assert s.diagnostics().contact_dissipation_w == 0
    assert -s.forces().contact_n[:s._offset, 1].sum() == pytest.approx(area * k * penetration)
    s.reset()
    np.testing.assert_array_equal(s.forces().contact_n, 0)
    assert s.diagnostics().contact_j == 0


def test_contact_potential_gradient():
    s = make_solid(free=True, half_gap=1e-5)
    x = s.positions_m
    x[:s._offset, 1] += 2e-5
    x[s._offset:, 1] -= 2e-5
    load_state(s, x)
    f = s.forces().contact_n
    direction = np.zeros_like(x)
    direction[:s._offset, 1] = 1
    h = 1e-8
    load_state(s, x + h * direction)
    plus = s.diagnostics().contact_j
    load_state(s, x - h * direction)
    minus = s.diagnostics().contact_j
    assert -(f * direction).sum() == pytest.approx((plus - minus) / (2 * h), rel=1e-10)


def test_shear_registration_rejected_atomically():
    s = make_solid(free=True)
    original = s.save_state()
    x = s.positions_m
    x[:s._offset, 0] += 3e-5
    with pytest.raises(ValueError, match="registration"):
        load_state(s, x)
    np.testing.assert_array_equal(s.positions_m, original.positions_m)
    assert s.time_s == original.time_s


def test_gap_slices_use_current_geometry_and_exact_closure_crossing():
    s = make_solid(contact=False, free=True)
    x = s.positions_m
    # Translate columns, leaving Jacobian=1: an asymmetric AP/SI zipper.
    for side, sl in enumerate(s._slices):
        if side == 0:
            x[sl, 1] += 6e-4 * x[sl, 0] / 0.01 + 1e-4 * x[sl, 2] / 0.002
    load_state(s, x)
    profile = s.gap_profile(np.array([0.0, 0.001, 0.002]))
    for section in profile.slices:
        g0 = 4e-4 - 1e-4 * section.si_m / 0.002
        root = g0 / 0.06
        expected = 0.5 * g0 * root
        assert section.area_m2 == pytest.approx(expected, rel=1e-12)
        assert np.min(section.signed_gap_m) < 0
        assert np.any(np.isclose(section.ap_m, root, atol=1e-12, rtol=0))
        assert section.ap_quadrature_m.sum() == pytest.approx(0.01)
        assert section.area_m2 == pytest.approx(section.ap_quadrature_m @ np.maximum(section.signed_gap_m, 0))


def test_independent_bilateral_and_ap_responses_preserve_state():
    s = make_solid(n_ap=4, n_si=2, contact=False)
    surf = s.surface("left")
    p = MedialPressure(np.where(surf.centroids_m[:, 0] < 0.004, 500.0, 0.0),
                       np.zeros(len(s.surface("right").triangles)))
    before = s.positions_m
    dt = s.recommend_timestep(pressure=p) * 0.1
    s.step(dt, pressure=p)
    after = s.positions_m
    assert np.linalg.norm(after[:s._offset] - before[:s._offset]) > 0
    np.testing.assert_allclose(after[s._offset:], before[s._offset:], atol=1e-18)
    grid = s._meshes[0].medial_grid
    displacement = after[grid, 1] - before[grid, 1]
    assert np.linalg.norm(displacement[1]) > 10 * np.linalg.norm(displacement[-2])
    snapshot = s.save_state()
    s.step(dt)
    assert np.linalg.norm(s.positions_m - after) > 0
    restored = s.positions_m
    s.reset(snapshot)
    s.step(dt)
    np.testing.assert_array_equal(s.positions_m, restored)
    # Returned state/position arrays cannot mutate the solver by aliasing.
    snapshot.positions_m[:] = 0
    copy = s.positions_m
    copy[:] = 0
    assert s.positions_m.max() > 0


def test_physiological_posture_and_all_three_motion_dofs():
    s = make_solid(contact=False)
    control = BilateralControl(left=FoldControl(longitudinal_strain=1e-5,
                                                posterior_displacement_m=(0, -2e-8, 1e-8)))
    start = s.positions_m
    dt = s.recommend_timestep(control) * 0.1
    s.step(dt, control)
    post = s._meshes[0].posterior_support[:, 0]
    displacement = s.positions_m[:s._offset][post] - start[:s._offset][post]
    np.testing.assert_allclose(displacement, np.tile([1e-7, -2e-8, 1e-8], (post.sum(), 1)), atol=1e-18)
    np.testing.assert_allclose(s.velocities_m_s[:s._offset][post], displacement / dt)
    assert s.save_state().control == control
    free_solid = make_solid(contact=False)
    v = np.zeros_like(free_solid.positions_m)
    free = ~free_solid._fixed.numpy()
    v[free] = np.broadcast_to([0.002, -0.003, 0.004], v.shape)[free]
    load_state(free_solid, v=v)
    before = free_solid.positions_m
    free_solid.step(free_solid.recommend_timestep() * 0.1)
    assert np.all(np.linalg.norm(free_solid.positions_m - before, axis=0) > 0)


def test_inverted_intermediate_path_rejected_despite_positive_endpoints():
    s = make_solid(contact=False, free=True)
    start = s._x.clone()
    end = start.clone()
    end[:, [0, 2]] *= -1  # 180-degree rigid rotation about lateral axis.
    s._kinematics(end, torch.zeros_like(end))
    with pytest.raises(ValueError, match="trajectory"):
        s._check_path(start, end)


def test_time_limit_accounts_for_viscosity_contact_and_current_deformation():
    base_m = tuple(replace(m, viscosity_pa_s=0) for m in nominal_materials())
    base = make_solid(materials=base_m, contact=False)
    viscous = make_solid(materials=tuple(replace(m, viscosity_pa_s=500) for m in base_m), contact=False)
    assert viscous.recommend_timestep() < base.recommend_timestep() / 10
    hard = VocalFoldSolid(replace(base.config, contact=ContactConfig(stiffness_pa_per_m=5e11,
                                                                   damping_pa_s_per_m=1e5)))
    assert hard.recommend_timestep() < base.recommend_timestep() / 10
    p = pressure(base, left=1e9, right=0)
    assert base.recommend_timestep(pressure=p) < base.recommend_timestep() / 10
    stretched = make_solid(contact=False, free=True)
    dt = stretched.recommend_timestep()
    x = stretched.positions_m
    x[:, 0] *= 1.8
    load_state(stretched, x)
    assert stretched.recommend_timestep() < dt
    snap = base.save_state()
    with pytest.raises(ValueError, match="stable"):
        base.step(base.recommend_timestep() * 2)
    np.testing.assert_array_equal(base.positions_m, snap.positions_m)
    assert base.time_s == 0


@pytest.mark.parametrize("bad", [0, -1e-6, float("nan"), float("inf")])
def test_invalid_timestep(bad):
    with pytest.raises(ValueError):
        make_solid().step(bad)


def test_invalid_mesh_material_pressure_and_inversion():
    with pytest.raises(ValueError):
        SolidMaterial(500, 100)
    with pytest.raises(ValueError):
        FoldControl(ta_activation=1.1)
    with pytest.raises(ValueError):
        nominal_fold_mesh("left", n_ap=0)
    mesh = nominal_fold_mesh("left")
    with pytest.raises(ValueError, match="orientation"):
        VocalFoldSolid(SolidConfig(replace(mesh, tetrahedra=mesh.tetrahedra[:, [0, 2, 1, 3]]),
                                   nominal_fold_mesh("right")))
    with pytest.raises(ValueError, match="outward"):
        VocalFoldSolid(SolidConfig(replace(mesh, medial_triangles=mesh.medial_triangles[:, [0, 2, 1]]),
                                   nominal_fold_mesh("right")))
    s = make_solid(contact=False, free=True)
    x = s.positions_m
    x[:, 0] *= -1
    with pytest.raises(ValueError, match="Jacobian"):
        load_state(s, x)
    with pytest.raises(ValueError, match="shape"):
        s.forces(pressure=MedialPressure(np.array([100.0]), np.array([100.0])))
    p = pressure(s)
    p.left_pa[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        s.forces(pressure=p)
    with pytest.raises(ValueError, match="signature"):
        make_solid(n_ap=3).reset(s.save_state())
    bad_regions = s._meshes[0].material_ids.copy()
    bad_regions[:] = 2
    other = VocalFoldSolid(replace(s.config, left=replace(s._meshes[0], material_ids=bad_regions)))
    with pytest.raises(ValueError, match="signature"):
        other.reset(s.save_state())


def test_invalid_candidate_step_keeps_position_velocity_and_history():
    mats = tuple(replace(m, viscosity_pa_s=0) for m in nominal_materials())
    s = make_solid(contact=False, free=True, materials=mats)
    dt = 0.1 * s.recommend_timestep()
    v = np.zeros_like(s.positions_m)
    v[:, 0] = -2 * s.positions_m[:, 0] / dt
    s.initialize(s.positions_m, v)
    before = s.save_state()
    with pytest.raises(ValueError):
        s.step(dt)
    np.testing.assert_array_equal(s.positions_m, before.positions_m)
    np.testing.assert_array_equal(s.velocities_m_s, before.velocities_m_s)
    assert s.time_s == before.time_s
    assert s.save_state().work == before.work


def _ringdown_run(dt_fraction, duration=0.0012):
    mats = tuple(replace(m, viscosity_pa_s=3) for m in nominal_materials())
    s = make_solid(contact=False, materials=mats)
    x = s.positions_m
    free = ~s._fixed.numpy()
    x[:, 1] += free[:, 1] * 1e-6 * np.sin(np.pi * x[:, 0] / 0.01)
    load_state(s, x)
    initial = total_energy(s)
    limit = s.recommend_timestep()
    count = int(np.ceil(duration / (limit * dt_fraction)))
    dt = duration / count
    for _ in range(count):
        s.step(dt)
    return s, initial


def test_passive_ringdown_and_first_order_time_convergence():
    coarse, initial = _ringdown_run(0.6)
    medium, _ = _ringdown_run(0.3)
    fine, _ = _ringdown_run(0.15)
    assert 0 < total_energy(fine) < initial * 0.8
    assert fine.diagnostics().work.viscous_dissipated_j > 0
    error_coarse = np.linalg.norm(coarse.positions_m - fine.positions_m)
    error_medium = np.linalg.norm(medium.positions_m - fine.positions_m)
    assert error_medium < error_coarse * 0.6
    assert np.isfinite(fine.velocities_m_s).all()
    assert abs(fine.diagnostics().energy_balance_residual_j) < initial * 0.03
    assert abs(fine.diagnostics().energy_balance_residual_j) < abs(coarse.diagnostics().energy_balance_residual_j)


def _linear_stiffness(mesh, mu, bulk):
    """Independent small-strain isotropic tetrahedral assembly."""
    n = len(mesh.reference_m)
    stiffness = np.zeros((3 * n, 3 * n))
    mass = np.zeros(n)
    for tet, region in zip(mesh.tetrahedra, mesh.material_ids):
        shear = mu if np.ndim(mu) == 0 else mu[region]
        compression = bulk if np.ndim(bulk) == 0 else bulk[region]
        lam = compression - 2 * shear / 3
        vertices = mesh.reference_m[tet]
        edge = (vertices[1:] - vertices[0]).T
        volume = np.linalg.det(edge) / 6
        inverse = np.linalg.inv(edge)
        grad = np.vstack((-inverse.sum(axis=0), inverse))
        mass[tet] += 1040 * volume / 4
        for i in range(4):
            for j in range(4):
                block = volume * (shear * np.eye(3) * np.dot(grad[i], grad[j])
                                  + lam * np.outer(grad[i], grad[j])
                                  + shear * np.outer(grad[j], grad[i]))
                stiffness[3 * tet[i]:3 * tet[i] + 3, 3 * tet[j]:3 * tet[j] + 3] += block
    return stiffness, np.repeat(mass, 3)


def test_mesh_refinement_and_independent_linear_material_response():
    mu, bulk, length = 500.0, 50000.0, 0.01
    material = SolidMaterial(mu, bulk, viscosity_pa_s=0)
    errors = []
    for n_ap in (2, 4, 8):
        mesh = nominal_fold_mesh("left", n_ap=n_ap, n_si=1, fix_lateral=False)
        k, mass = _linear_stiffness(mesh, mu, bulk)
        # A manufactured shear family: constrain AP/SI DOFs, retain all lateral DOFs.
        free = np.zeros_like(mesh.reference_m, dtype=bool)
        free[:, 1] = (mesh.reference_m[:, 0] > 0) & (mesh.reference_m[:, 0] < length)
        free = free.ravel()
        w2 = eigh(k[np.ix_(free, free)], np.diag(mass[free]), subset_by_index=[0, 0])[0][0]
        exact = mu / 1040 * (np.pi / length)**2
        errors.append(abs(w2 - exact) / exact)
        assert w2 > 0
        solid = VocalFoldSolid(SolidConfig(mesh, nominal_fold_mesh("right", n_ap=n_ap, n_si=1,
                                                                 fix_lateral=False),
                                          (material,) * 3, ContactConfig(enabled=False)))
        rng = np.random.default_rng(123)
        displacement = rng.normal(size=mesh.reference_m.shape) * 1e-10
        displacement[mesh.anterior_support | mesh.posterior_support | mesh.lateral_support] = 0
        x = solid.positions_m
        x[:solid._offset] += displacement
        load_state(solid, x)
        force = solid.forces().passive_n[:solid._offset].ravel()
        np.testing.assert_allclose(force, -k @ displacement.ravel(), atol=5e-11, rtol=1e-3)
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 0.08


def test_distinct_region_stiffness_and_invalid_support_snapshot():
    mu, bulk = [400.0, 700.0, 1200.0], [40000.0, 60000.0, 90000.0]
    materials = tuple(SolidMaterial(m, k, viscosity_pa_s=0) for m, k in zip(mu, bulk))
    s = make_solid(contact=False, materials=materials)
    mesh = s._meshes[0]
    k, _ = _linear_stiffness(mesh, mu, bulk)
    rng = np.random.default_rng(34)
    displacement = rng.normal(size=mesh.reference_m.shape) * 1e-10
    displacement[mesh.anterior_support | mesh.posterior_support | mesh.lateral_support] = 0
    x = s.positions_m
    x[:s._offset] += displacement
    load_state(s, x)
    force = s.forces().passive_n[:s._offset].ravel()
    np.testing.assert_allclose(force, -k @ displacement.ravel(), atol=1e-11, rtol=1e-3)
    state = s.save_state()
    state.positions_m[0, 0] += 1e-5
    before = s.positions_m
    with pytest.raises(ValueError, match="support posture"):
        s.reset(state)
    np.testing.assert_array_equal(s.positions_m, before)


def test_shared_kernels_preserve_legacy_fem_and_lips():
    x, tet, _ = st.box_tets(1, 1, 1, (0.002, 0.003, 0.004))
    material = st.Material(E=40000, nu=0.49, eta=3)
    body = st.Body(x, tet, material, np.zeros(len(x), dtype=bool),
                   surf_tris=np.array([[0, 1, 3]]))
    fem = st.FEM([body], device="cpu")
    fem.x[:, 0] *= 1.02
    fem.v[:, 1] = 0.1 * fem.x[:, 0]
    D = (fem.x[fem.tets[:, 1:]] - fem.x[fem.tets[:, :1]]).transpose(1, 2)
    F = D @ fem.Dm_inv
    inverse = torch.linalg.inv(F)
    J = torch.linalg.det(F).clamp_min(0.2)
    P = fem.mu[:, None, None] * (F - inverse.transpose(1, 2))
    P += (fem.lam * torch.log(J))[:, None, None] * inverse.transpose(1, 2)
    Fdot = (fem.v[fem.tets[:, 1:]] - fem.v[fem.tets[:, :1]]).transpose(1, 2) @ fem.Dm_inv
    L = Fdot @ inverse
    P += J[:, None, None] * (fem.eta[:, None, None] * (L + L.transpose(1, 2))) @ inverse.transpose(1, 2)
    expected = torch.zeros_like(fem.x)
    for j, t in enumerate(fem.tets):
        h = -fem.V0[j] * P[j] @ fem.Dm_inv[j].T
        expected[t[1:]] += h.T
        expected[t[0]] -= h.sum(1)
    tri = fem.surf[0]
    area = torch.cross(fem.x[tri[1]] - fem.x[tri[0]], fem.x[tri[2]] - fem.x[tri[0]], dim=0) / 2
    expected[tri] -= 100 * area / 3
    torch.testing.assert_close(fem.forces(p_surf=torch.tensor([100.0])), expected, atol=1e-14, rtol=1e-12)
    from formant_ml.physics.lips import Lips
    lips = Lips(device="cpu")
    original = lips.fem.x.clone()
    lips.step(lips.fem.dt_stable * 0.1, np.zeros(9))
    assert torch.isfinite(lips.fem.x).all()
    torch.testing.assert_close(lips.fem.x, original, atol=1e-8, rtol=1e-6)
