"""Respiratory state tests; real solid/acoustic coupling tests follow its adapter."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from formant_ml.physics import air, lungs
from formant_ml.physics.acoustic_domain import AcousticDomain, VolumeFlowPort
from formant_ml.physics.glottal_channel import GlottalChannel, Reservoir
from formant_ml.physics.phonation import (
    CouplingConfig, CouplingError, EulerianGlottis, LungConfig, LungReservoir,
    PhonationSystem, PortWallPatch, SupplyDuct,
)
from formant_ml.physics.tract3d import Grid, T_ID
from formant_ml.physics.vocal_fold_solid import (
    PressurePatch, SolidConfig, VocalFoldSolid, nominal_fold_mesh,
)

torch.set_num_threads(2)


def test_actual_lung_mass_expenditure_drops_recoil_without_compensation():
    lung = LungReservoir()
    initial = lung.snapshot()
    mass_flow = air.props().rho * 2e-4
    step = lung.advance(0.5, mass_flow)
    assert lung.mass_kg == pytest.approx(initial.mass_kg - 0.5 * mass_flow)
    assert lung.volume_m3 < initial.volume_m3
    assert lung.pressure_pa() < (
        lung.recoil_pressure_pa(initial.volume_m3) + initial.muscle_pressure_pa
    )
    assert lung.muscle_pressure_pa == 0
    assert step.reference_air_expenditure_m3 == pytest.approx(1e-4)
    # Gas decompresses while the compliant lung shrinks, so these are distinct.
    assert not np.isclose(-step.geometric_volume_change_m3,
                          step.reference_air_expenditure_m3, rtol=1e-4, atol=0)


def test_recoil_energy_derivative_matches_independent_existing_curve():
    lung = LungReservoir()
    for f in (-0.2, -0.05, 0.05, 0.5, 0.9):
        v = lung.config.frc_volume_m3 + f * lung.config.vital_capacity_m3
        dv = 1e-9
        derivative = (lung.recoil_energy_j(v + dv) - lung.recoil_energy_j(v - dv)) / (2 * dv)
        expected = lungs.relaxation_pressure(f) * lungs.CM_H2O
        assert derivative == pytest.approx(expected, rel=2e-8)


def test_isothermal_first_law_and_mass_equilibrium():
    lung = LungReservoir(LungConfig(initial_muscle_pressure_pa=200))
    before = lung.snapshot()
    step = lung.advance(0.1, 1e-4, muscle_pressure_pa=500)
    props = air.props()
    R = air.R_GAS / props.M
    T = props.T_c + 273.15
    expected_mass = (
        (air.P_ATM + lung.pressure_pa()) * lung.volume_m3 / (R * T)
    )
    assert lung.mass_kg == pytest.approx(expected_mass, rel=2e-13)
    cv = R / (props.gamma - 1)
    expected_heat = (
        (lung.mass_kg - before.mass_kg) * cv * T
        + air.P_ATM * (lung.volume_m3 - before.volume_m3)
        + lung.recoil_energy_j() - lung.recoil_energy_j(before.volume_m3)
        + 350 * (lung.volume_m3 - before.volume_m3)
        + step.outlet_mass_flow_kg_s * 0.1 * (cv + R) * T
    )
    assert step.heat_into_gas_j == pytest.approx(expected_heat, abs=2e-14)
    assert abs(step.first_law_residual_j) < 1e-14


def test_zero_flow_muscle_change_does_not_create_gas_and_signed_inhalation():
    lung = LungReservoir()
    mass = lung.mass_kg
    volume = lung.volume_m3
    step = lung.advance(0.1, 0, muscle_pressure_pa=500)
    assert lung.mass_kg == mass
    assert lung.volume_m3 < volume
    assert step.outlet_volume_flow_m3_s == 0
    small = lung.advance(0.1, -1e-4)
    assert small.outlet_volume_flow_m3_s < 0
    assert lung.mass_kg > mass


def test_lung_snapshot_replay_budget_failure_and_wrong_owner_are_atomic():
    lung = LungReservoir()
    start = lung.snapshot()
    first = lung.advance(0.1, 1e-4)
    end = lung.snapshot()
    lung.restore(start)
    assert lung.advance(0.1, 1e-4) == first
    assert lung.snapshot() == end
    with pytest.raises(ValueError, match="expenditure"):
        lung.advance(100, 1)
    assert lung.snapshot() == end
    with pytest.raises(ValueError, match="equilibrium"):
        lung.restore(replace(end, mass_kg=end.mass_kg * 2))
    assert lung.snapshot() == end
    with pytest.raises(ValueError, match="different instance"):
        LungReservoir().restore(end)


def reference_solid(*, half_gap=5e-5, recess=0.0, two_pockets=False):
    meshes = []
    for side in ("left", "right"):
        mesh = nominal_fold_mesh(
            side, n_ap=2, n_si=4 if two_pockets else 2, half_gap_m=half_gap,
            fix_anterior=False, fix_posterior=False, fix_lateral=False,
        )
        coordinates = mesh.reference_m.copy()
        if recess:
            sign = -1 if side == "left" else 1
            depth = np.abs(coordinates[:, 1]) - half_gap
            shape = 1 - np.abs(coordinates[:, 2] - 0.001) / 0.001
            if two_pockets:
                shape = np.rint(coordinates[:, 2] / 0.0005).astype(int) % 2
            coordinates[:, 1] += sign * recess * shape * (1 - depth / 0.006)
            mesh = replace(mesh, reference_m=coordinates)
        meshes.append(mesh)
    return VocalFoldSolid(SolidConfig(*meshes))


def test_actual_solid_clipping_matches_independent_prism_volume_and_pressure_work():
    solid = reference_solid()
    mapper = EulerianGlottis(
        [0.001, 0.005, 0.009], [0.0002, 0.001, 0.0018],
        provenance="test planar walls, no unresolved cavity",
    )
    first = mapper.capture(solid)
    expected = 0.004 * 0.0008 * 1e-4
    np.testing.assert_allclose(first.geometry.storage_volume_m3, expected, rtol=1e-12)
    start = solid.save_state()
    positions = start.positions_m.copy()
    n_left = len(solid.surface("left").coordinates_m)
    positions[:n_left, 1] -= 1e-6
    positions[n_left:, 1] += 1e-6
    solid.initialize(positions)
    end = solid.save_state()
    second = mapper.capture(solid)
    sweep = first.swept_cell_volume_m3(solid, start, end)
    np.testing.assert_allclose(
        second.geometry.storage_volume_m3 - first.geometry.storage_volume_m3,
        sweep, rtol=2e-12, atol=1e-24,
    )
    # The real solid validates the mapped disjoint exterior pressure patches.
    forces = solid.forces(pressure_patches=second.pressure_patches(np.ones((2, 2))))
    assert np.isfinite(forces.pressure_n).all()


def test_geometric_recess_keeps_real_wet_storage_but_never_opens_closed_throats():
    solid = reference_solid(half_gap=0, recess=2e-5)
    mapper = EulerianGlottis(
        [0, 0.005, 0.01], [0, 0.002],
        provenance="synthetic triangular medial recess, 20um per wall; not human data",
    )
    mapped = mapper.capture(solid)
    # Triangular full-gap profile: area = height*(2*recess)/2.
    expected = 0.005 * 0.002 * 4e-5 / 2
    np.testing.assert_allclose(mapped.geometry.storage_volume_m3, expected, rtol=1e-12)
    assert np.count_nonzero(mapped.geometry.open_area_m2) == 0
    assert mapped.patches
    solid.forces(pressure_patches=mapped.pressure_patches(np.full((2, 1), 100)))
    dry = mapper.capture(reference_solid(half_gap=0))
    assert dry.geometry.dry
    assert not dry.patches


def test_eulerian_patch_motion_has_measured_convergent_remapping_residual():
    solid = reference_solid()
    mapper = EulerianGlottis(
        [0.001, 0.005, 0.009], [0.0002, 0.001, 0.0018],
        provenance="test independently moving Eulerian clipping weights",
    )
    initial = solid.save_state()
    n_left = len(solid.surface("left").coordinates_m)
    base_volume = mapper.capture(solid).geometry.storage_volume_m3
    errors = []
    for scale in (1.0, 0.5, 0.25):
        displacement = np.zeros_like(initial.positions_m)
        X, Y, Z = initial.positions_m.T
        shape = np.sin(np.pi * X / 0.01) * np.sin(np.pi * Z / 0.002)
        displacement[:, 0] = scale * 1e-4 * shape
        displacement[:, 2] = scale * 1e-4 * shape
        displacement[:n_left, 1] = -scale * 2e-5 * shape[:n_left]
        displacement[n_left:, 1] = scale * 2e-5 * shape[n_left:]
        positions = initial.positions_m + displacement
        solid.initialize(positions)
        end = solid.save_state()
        midpoint = mapper.capture(solid, positions_m=(initial.positions_m + positions) / 2)
        end_volume = mapper.capture(solid).geometry.storage_volume_m3
        sweep = midpoint.swept_cell_volume_m3(solid, initial, end)
        errors.append(float(np.max(np.abs(end_volume - base_volume - sweep))))
    assert errors[1] < errors[0] / 3
    assert errors[2] < errors[1] / 3
    assert errors[2] < 1e-13


def test_contact_separated_pockets_are_rejected_or_kept_as_independent_states():
    solid = reference_solid(half_gap=0, recess=2e-5, two_pockets=True)
    before = solid.save_state()
    wrong = EulerianGlottis(
        [0, 0.01], [0, 0.002], provenance="two real disconnected reference recesses",
    )
    with pytest.raises(ValueError, match="disconnected gas pockets"):
        wrong.capture(solid)
    np.testing.assert_array_equal(solid.save_state().positions_m, before.positions_m)
    correct = EulerianGlottis(
        [0, 0.01], [0, 0.001, 0.002],
        provenance="two reference recesses separated by an exactly closed ridge",
    )
    first = correct.capture(solid)
    channel = GlottalChannel(first.geometry)
    gas_before = channel.snapshot()
    positions = before.positions_m.copy()
    lower_recess = np.isclose(positions[:, 2], 0.0005, atol=1e-15, rtol=0)
    positions[lower_recess, 1] *= 0.5
    solid.initialize(positions)
    second = correct.capture(solid)
    step = channel.advance(1e-8, Reservoir(0), Reservoir(0), geometry=second.geometry)
    assert np.all(second.geometry.open_area_m2 == 0)
    assert step.wall_pressure_pa[0, 0] > step.wall_pressure_pa[0, 1]
    assert channel.snapshot().mass_kg[0, 1] == gas_before.mass_kg[0, 1]
    assert channel.snapshot().energy_j[0, 1] == gas_before.energy_j[0, 1]


def acoustic_box(h_cm=0.2, dt=1e-7, two_ports=False):
    mask = np.ones((3, 3, 3), bool)
    openings, walls = [], []
    for axis in range(3):
        shape = list(mask.shape)
        shape[axis] += 1
        opened = np.ones(shape)
        wall = np.zeros(shape, np.int8)
        for index in (0, -1):
            sl = [slice(None)] * 3
            sl[axis] = index
            opened[tuple(sl)] = 0
            wall[tuple(sl)] = T_ID["rigid"]
        openings.append(opened)
        walls.append(wall)
    inlet = np.zeros_like(mask)
    inlet[1, 1, 1] = True
    grid = Grid(
        h=h_cm, origin=np.zeros(3), air=mask,
        ax=openings[0], ay=openings[1], az=openings[2],
        wx=walls[0], wy=walls[1], wz=walls[2],
        cx=np.ones_like(openings[0]), cy=np.ones_like(openings[1]),
        cz=np.ones_like(openings[2]), inlet=inlet, exterior=mask.copy(),
        exit_point=np.zeros(2), exit_dir=np.array([1., 0.]),
        area_1d=np.zeros(1), pos_1d=np.zeros(1),
    )
    ports = {"glottis": VolumeFlowPort(inlet)}
    if two_ports:
        lung_mask = np.zeros_like(mask)
        lung_mask[0, 1, 1] = True
        ports["lung"] = VolumeFlowPort(lung_mask)
    return AcousticDomain(
        grid, ports, T_c=37, rh=1,
        boundary_layer=False, sponge_cells=0, max_dt_s=dt,
    )


def actual_system(*, pressure=200.0, h_cm=0.2, dt=1e-7, config=None, upstream=False):
    solid = VocalFoldSolid(SolidConfig(
        nominal_fold_mesh("left", n_ap=2, n_si=2, half_gap_m=5e-5),
        nominal_fold_mesh("right", n_ap=2, n_si=2, half_gap_m=5e-5),
    ))
    mapper = EulerianGlottis(
        [0.001, 0.009], [0.0002, 0.0018],
        provenance="unfitted nominal bilateral solid, fixed central reference window",
    )
    channel = GlottalChannel(mapper.capture(solid).geometry)
    lung = LungReservoir(LungConfig(initial_fraction=0, initial_muscle_pressure_pa=pressure))
    wall_patches = []
    for side in ("left", "right"):
        coordinates = solid.surface(side).coordinates_m
        for triangle in solid.boundary_triangles(side):
            points = coordinates[triangle]
            if np.max(np.abs(points[:, 1])) > 0.00155:
                continue
            if np.all(points[:, 2] == 0):
                end = "upstream"
            elif np.all(points[:, 2] == 0.002):
                end = "downstream"
            else:
                continue
            wall_patches.append(PortWallPatch(
                end, 0, PressurePatch(side, tuple(triangle), 0),
            ))
    return PhonationSystem(
        solid, channel, mapper, lung, acoustic_box(h_cm, dt), ("glottis",),
        wall_patches=wall_patches, config=config,
        upstream=acoustic_box(h_cm, dt, two_ports=True) if upstream else None,
        upstream_ports=("glottis",) if upstream else (),
        upstream_lung_port="lung" if upstream else None,
        supply=SupplyDuct(0.05, 0.004) if upstream else None,
    )


def test_real_components_exchange_pressure_flow_shape_work_and_replay():
    system = actual_system()
    initial = system.snapshot()
    result = system.advance()
    assert result.iterations > 1
    assert result.inlet_flow_m3_s[0] > 0
    assert np.linalg.norm(system.solid.velocities_m_s) > 0
    assert abs(result.upstream_wall_work_j) > 0
    assert result.interface_work_residual_j < system.config.work_tolerance_j
    assert result.volume_residual_m3 < system.config.volume_tolerance_m3
    final = system.snapshot()
    system.restore(initial)
    replay = system.advance()
    np.testing.assert_array_equal(replay.inlet_flow_m3_s, result.inlet_flow_m3_s)
    np.testing.assert_array_equal(system.solid.positions_m, final.solid.positions_m)
    np.testing.assert_array_equal(system.channel.snapshot().energy_j, final.channel.energy_j)
    assert system.lung.snapshot() == final.lung


def test_pressure_changes_real_tissue_motion_without_an_oscillation_driver():
    off, on = actual_system(pressure=0), actual_system(pressure=200)
    off.advance()
    on.advance()
    assert np.max(np.abs(off.solid.velocities_m_s)) < 1e-14
    assert np.max(np.abs(on.solid.velocities_m_s)) > 1e-9


def test_rejected_coupling_iteration_restores_every_real_component():
    system = actual_system(config=CouplingConfig(max_iterations=1))
    before = system.snapshot()
    with pytest.raises(CouplingError, match="iteration exhausted"):
        system.advance()
    after = system.snapshot()
    np.testing.assert_array_equal(after.solid.positions_m, before.solid.positions_m)
    np.testing.assert_array_equal(after.channel.mass_kg, before.channel.mass_kg)
    assert after.lung == before.lung
    assert all(torch.equal(a, b) for a, b in
               zip(after.downstream.field, before.downstream.field))
    assert after.steps == before.steps


def test_supply_duct_inertial_limit_signed_flow_and_passive_work():
    duct = SupplyDuct(0.05, 0.004)
    result = duct.advance(1e-7, 100, 0)
    expected = 100 * 1e-7 / duct.inertance
    assert duct.flow_m3_s == pytest.approx(expected, rel=1e-5)
    assert abs(result.energy_residual_j) < 1e-20
    before = duct.diagnostics()["kinetic_energy_j"]
    off = duct.advance(1e-3, 0, 0)
    assert duct.diagnostics()["kinetic_energy_j"] < before
    assert before - duct.diagnostics()["kinetic_energy_j"] == pytest.approx(off.dissipation_j)
    reverse = SupplyDuct(0.05, 0.004)
    reverse.advance(1e-7, -100, 0)
    assert reverse.flow_m3_s == pytest.approx(-expected, rel=1e-5)


def test_real_two_port_upstream_load_counts_lung_flow_not_glottal_flow():
    system = actual_system(upstream=True)
    before = system.snapshot()
    result = system.advance()
    after = system.snapshot()
    spent = before.lung.mass_kg - after.lung.mass_kg
    assert spent == pytest.approx(result.lung_mass_flow_kg_s * system.dt, rel=1e-4)
    assert spent > 0
    # Some delivered air compresses the actual upstream domain.
    assert system.upstream.diagnostics()["compression_volume_m3"] > 0
    assert system.upstream.pressure_pa("glottis") < system.lung.pressure_pa() / 10
    assert result.interface_work_residual_j < system.config.work_tolerance_j
    system.restore(before)
    replay = system.advance()
    np.testing.assert_array_equal(replay.inlet_flow_m3_s, result.inlet_flow_m3_s)


def test_actual_acoustic_backpressure_changes_channel_flow():
    small = actual_system(h_cm=0.025)
    large = actual_system(h_cm=0.4)
    for _ in range(8):
        a, b = small.advance(), large.advance()
    assert abs(a.downstream_pressure_pa[0]) > abs(b.downstream_pressure_pa[0]) * 2
    assert abs(a.outlet_flow_m3_s[0] - b.outlet_flow_m3_s[0]) > 1e-12
