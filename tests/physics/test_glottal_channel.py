"""Independent limits for the explicitly supplied reference chamber geometry."""
from dataclasses import replace

import numpy as np
import pytest

from formant_ml.physics import air
from formant_ml.physics.glottal_channel import (
    ChannelConfig, ChannelGeometry, GlottalChannel, Reservoir, parallel_plate_shear,
)


def chamber(gap=1e-4, *, pocket=0.0, cells=3, strips=1, closed=False,
            length=1e-3, width=1e-3):
    """Rectangular gas chambers with explicit recessed slabs and throat plates.

    Gas boxes occupy x=[strip*width,(strip+1)*width], y=[0,gap+pocket],
    z=[cell*length/cells,(cell+1)*length/cells]. Piston faces are the
    y=gap+pocket walls. The separate throat plate opens y=[0,gap] only.
    This is test apparatus, not a vocal-fold anatomy adapter.
    """
    face = (strips, cells + 1)
    open_gap = 0.0 if closed else gap
    return ChannelGeometry(
        np.full(cells, length / cells), np.full(strips, width),
        np.full((strips, cells), width * (gap + pocket) * length / cells),
        np.full(face, width * (gap + pocket)),
        np.full(face, width * open_gap), np.full(face, open_gap),
        "Synthetic rectangular gas boxes with specified recessed slab; "
        f"width={width}m length={length}m gap={gap}m pocket={pocket}m; "
        "piston footprint is the full x-z face; no human anatomical provenance.",
        wet_wall_area_m2=np.full(
            (strips, cells), width * length / cells if gap + pocket > 0 else 0
        ),
    )


def advance_for(channel, duration, inlet, outlet):
    n = int(np.ceil(duration / channel.recommend_timestep()))
    dt = duration / n
    results = [channel.advance(dt, inlet, outlet) for _ in range(n)]
    return results


def test_independent_steady_slit_poiseuille_and_viscous_wall_pressure():
    g = chamber(gap=20e-6, cells=3)
    channel = GlottalChannel(g)
    steps = advance_for(channel, 1.5e-4, Reservoir(1.0), Reservoir(0.0))
    mu = air.props().mu
    expected = 1e-3 * (20e-6) ** 3 / (12 * mu * 1e-3)
    q = steps[-1].outlet_flow_m3_s[0]
    assert q == pytest.approx(expected, rel=0.015)
    pressures = steps[-1].wall_pressure_pa[0]
    assert np.all(np.diff(pressures) < 0)
    assert pressures == pytest.approx([5 / 6, 1 / 2, 1 / 6], abs=0.03)
    assert max(abs(s.energy_residual_j) for s in steps) < 2e-17


def test_signed_reversal_and_independent_inertial_limit():
    g = chamber(gap=2e-4, cells=1)
    config = ChannelConfig(viscosity=False)
    outputs = []
    duration = 2e-6
    for sign in (1, -1):
        channel = GlottalChannel(g, config)
        results = advance_for(
            channel, duration, Reservoir(sign * 0.5), Reservoir(-sign * 0.5)
        )
        u = channel.diagnostics()["velocity_m_s"][0, 0]
        # Newton's law on a uniform gas slug: rho*length*du/dt = dp.
        expected = sign * duration / (air.props().rho * 1e-3)
        assert u == pytest.approx(expected, rel=2e-5)
        assert np.sign(results[-1].outlet_flow_m3_s[0]) == sign
        outputs.append(u)
    assert outputs[0] == pytest.approx(-outputs[1], rel=2e-5)


def test_sealed_adiabatic_piston_independent_eos_and_work_refinement():
    errors = []
    for n in (20, 40):
        g0 = chamber(gap=80e-6, pocket=20e-6, cells=1, closed=True)
        channel = GlottalChannel(g0, ChannelConfig(viscosity=False))
        initial = channel.diagnostics()
        mass0 = initial["mass_kg"].sum()
        E0 = initial["total_energy_j"]
        absolute_work = 0.0
        previous_volume = g0.storage_volume_m3.sum()
        for gap in np.linspace(80e-6, 0, n + 1)[1:]:
            g = chamber(gap=gap, pocket=20e-6, cells=1, closed=True)
            result = channel.advance(
                1e-8, Reservoir(0), Reservoir(0), geometry=g
            )
            volume = g.storage_volume_m3.sum()
            absolute_work += result.wall_work_j + air.P_ATM * (volume - previous_volume)
            previous_volume = volume
            assert result.inlet_flow_m3_s[0] == 0
            assert result.outlet_flow_m3_s[0] == 0
            assert abs(result.mass_residual_kg) < 1e-24
            assert abs(result.energy_residual_j) < 3e-17
        diag = channel.diagnostics()
        expected_p = air.P_ATM * 5 ** air.props().gamma
        errors.append(abs((diag["pressure_pa"][0, 0] + air.P_ATM) / expected_p - 1))
        assert diag["mass_kg"].sum() == pytest.approx(mass0, rel=1e-13)
        assert diag["total_energy_j"] - E0 == pytest.approx(-absolute_work, abs=3e-17)
        for gap in np.linspace(0, 80e-6, n + 1)[1:]:
            channel.advance(
                1e-8, Reservoir(0), Reservoir(0),
                geometry=chamber(gap=gap, pocket=20e-6, cells=1, closed=True),
            )
        assert channel.diagnostics()["total_energy_j"] == pytest.approx(E0, rel=2e-13)
    assert errors[1] < errors[0] / 3.7
    assert errors[1] < 0.001


def test_actual_closing_reopening_retains_mass_energy_and_zero_throat():
    channel = GlottalChannel(chamber(gap=2e-6, pocket=1e-4, cells=1))
    original = channel.snapshot()
    for gap in np.linspace(2e-6, 0, 41)[1:]:
        result = channel.advance(
            1e-8, Reservoir(0), Reservoir(0),
            geometry=chamber(gap=gap, pocket=1e-4, cells=1),
        )
        assert abs(result.energy_residual_j) < 1e-16
    sealed = channel.snapshot()
    assert channel.geometry.open_area_m2.sum() == 0
    assert sealed.mass_kg.sum() > 0
    hold = channel.advance(1e-8, Reservoir(500), Reservoir(-500))
    assert hold.inlet_flow_m3_s[0] == hold.outlet_flow_m3_s[0] == 0
    np.testing.assert_array_equal(channel.snapshot().mass_kg, sealed.mass_kg)
    # Retained closing momentum now shears into heat inside the sealed pocket,
    # so total energy is held to the declared nonlinear solve tolerance.
    assert hold.viscous_heat_j > 0
    np.testing.assert_allclose(
        channel.snapshot().energy_j, sealed.energy_j, rtol=0,
        atol=channel.config.tolerance * float(sealed.energy_j.sum()),
    )
    reopened = channel.advance(
        1e-8, Reservoir(0), Reservoir(0),
        geometry=chamber(gap=2e-6, pocket=1e-4, cells=1),
    )
    assert channel.snapshot().mass_kg.sum() > 0
    assert abs(reopened.energy_residual_j) < 1e-16
    assert reopened.inlet_flow_m3_s[0] < 0
    assert reopened.outlet_flow_m3_s[0] > 0
    assert channel.snapshot().time_s > original.time_s


def test_closed_initial_momentum_becomes_internal_heat_not_deleted_energy():
    channel = GlottalChannel(
        chamber(gap=0, pocket=1e-4, cells=1),
        ChannelConfig(viscosity=False), velocity_m_s=0.01,
    )
    d0 = channel.diagnostics()
    result = channel.advance(channel.recommend_timestep() * 0.5, Reservoir(0), Reservoir(0))
    d1 = channel.diagnostics()
    assert d1["kinetic_energy_j"] < d0["kinetic_energy_j"]
    assert d1["internal_energy_j"] > d0["internal_energy_j"]
    assert d1["total_energy_j"] == pytest.approx(d0["total_energy_j"], abs=2e-19)
    assert d0["kinetic_energy_j"] - d1["kinetic_energy_j"] == pytest.approx(
        result.reflection_heat_j, rel=1e-5, abs=2e-20
    )
    assert result.inlet_flow_m3_s[0] == result.outlet_flow_m3_s[0] == 0


def test_snapshot_replay_and_rejection_are_atomic():
    channel = GlottalChannel(chamber(), velocity_m_s=0.01)
    before = channel.snapshot()
    dt = channel.recommend_timestep() * 0.5
    first = channel.advance(dt, Reservoir(1), Reservoir(0))
    after = channel.snapshot()
    channel.restore(before)
    second = channel.advance(dt, Reservoir(1), Reservoir(0))
    np.testing.assert_array_equal(first.inlet_flow_m3_s, second.inlet_flow_m3_s)
    np.testing.assert_array_equal(channel.snapshot().energy_j, after.energy_j)
    snapshot = channel.snapshot()
    with pytest.raises(ValueError, match="timestep"):
        channel.advance(channel.recommend_timestep() * 2, Reservoir(1), Reservoir(0))
    np.testing.assert_array_equal(channel.snapshot().mass_kg, snapshot.mass_kg)
    assert channel.time_s == snapshot.time_s
    with pytest.raises(ValueError, match="different instance"):
        GlottalChannel(chamber()).restore(snapshot)
    with pytest.raises(ValueError, match="shape"):
        channel.restore(replace(snapshot, mass_kg=np.zeros(1)))


def test_dry_closed_equilibrium_has_no_air_floor_or_implicit_refilling():
    channel = GlottalChannel(chamber(gap=0, cells=1))
    result = channel.advance(1e-3, Reservoir(1000), Reservoir(-1000))
    assert result.inlet_flow_m3_s[0] == result.outlet_flow_m3_s[0] == 0
    assert channel.diagnostics()["mass_kg"].sum() == 0
    with pytest.raises(ValueError, match="filling"):
        channel.advance(1e-8, Reservoir(0), Reservoir(0),
                        geometry=chamber(gap=1e-5, cells=1))


def test_geometry_requires_explicit_nonconducting_storage_and_real_gap():
    g = chamber(pocket=1e-4, closed=True)
    assert g.storage_volume_m3.sum() > 0
    assert g.open_area_m2.sum() == 0
    with pytest.raises(ValueError, match="provenance"):
        replace(g, provenance="")
    with pytest.raises(ValueError, match="actual slit gap"):
        replace(g, face_gap_m=np.full_like(g.face_gap_m, 1e-5))


def test_failed_nonlinear_iteration_does_not_mutate_any_state():
    channel = GlottalChannel(chamber(), ChannelConfig(max_evaluations=1))
    before = channel.snapshot()
    with pytest.raises(RuntimeError, match="midpoint solve rejected"):
        channel.advance(channel.recommend_timestep() / 2, Reservoir(10), Reservoir(0))
    after = channel.snapshot()
    for name in ("mass_kg", "momentum_kg_m_s", "energy_j"):
        np.testing.assert_array_equal(getattr(after, name), getattr(before, name))
    assert after.time_s == before.time_s
    assert after.input_mass_kg == before.input_mass_kg
    assert after.input_shifted_energy_j == before.input_shifted_energy_j
    assert after.wall_work_j == before.wall_work_j


def test_reference_cavity_size_changes_trapped_pressure_without_any_leak():
    pressures = []
    for pocket in (20e-6, 40e-6):
        channel = GlottalChannel(chamber(gap=40e-6, pocket=pocket, closed=True, cells=1))
        initial_mass = channel.diagnostics()["mass_kg"].sum()
        for gap in np.linspace(40e-6, 0, 41)[1:]:
            result = channel.advance(
                1e-8, Reservoir(0), Reservoir(0),
                geometry=chamber(gap=gap, pocket=pocket, closed=True, cells=1),
            )
            assert result.inlet_flow_m3_s[0] == result.outlet_flow_m3_s[0] == 0
        pressure = channel.diagnostics()["pressure_pa"][0, 0] + air.P_ATM
        expected = air.P_ATM * ((40e-6 + pocket) / pocket) ** air.props().gamma
        assert pressure == pytest.approx(expected, rel=4e-4)
        assert channel.diagnostics()["mass_kg"].sum() == pytest.approx(initial_mass)
        pressures.append(pressure)
    assert pressures[0] > pressures[1]


def test_closed_pockets_with_nonzero_momentum_are_independent():
    outputs = []
    for left_volume_scale in (1.0, 1.02, 0.98):
        geometry = chamber(gap=0, pocket=1e-3, cells=2, length=2e-3)
        channel = GlottalChannel(geometry, velocity_m_s=np.array([[0., 20., 0.]]))
        volume = geometry.storage_volume_m3.copy()
        volume[0, 0] *= left_volume_scale
        step = channel.advance(
            1e-7, Reservoir(0), Reservoir(0),
            geometry=replace(geometry, storage_volume_m3=volume),
        )
        state = channel.snapshot()
        outputs.append((
            channel.diagnostics()["pressure_pa"][0, 1],
            state.mass_kg[0, 1], state.energy_j[0, 1],
            state.momentum_kg_m_s[0, 1].copy(),
        ))
        assert np.max(np.abs(step.momentum_residual_kg_m_s)) < 1e-20
    for pressure, mass, energy, momentum in outputs[1:]:
        assert pressure == pytest.approx(outputs[0][0], abs=1e-8)
        assert mass == outputs[0][1]
        assert energy == pytest.approx(outputs[0][2], abs=5e-18)
        np.testing.assert_allclose(momentum, outputs[0][3], atol=1e-20, rtol=0)


def test_closing_and_reopening_do_not_delete_one_sided_momentum():
    opened = chamber(gap=1e-3, cells=2, length=2e-3)
    closed = replace(opened, open_area_m2=np.zeros((1, 3)), face_gap_m=np.zeros((1, 3)))
    channel = GlottalChannel(opened, velocity_m_s=np.array([[0., 0.02, 0.]]))
    for geometry in (closed, closed, opened):
        before = channel.snapshot()
        step = channel.advance(1e-8, Reservoir(0), Reservoir(0), geometry=geometry)
        after = channel.snapshot()
        assert np.linalg.norm(after.momentum_kg_m_s) > 0
        change = (after.momentum_kg_m_s - before.momentum_kg_m_s).sum(axis=(1, 2))
        np.testing.assert_allclose(
            change, step.input_impulse_n_s - step.wall_impulse_n_s,
            atol=1e-21, rtol=1e-8,
        )
        assert abs(step.energy_residual_j) < 1e-17


def test_slit_shear_translation_couette_heat_and_signed_relative_flow():
    mu, area, gap = 2e-5, 1e-6, 1e-4
    fluid = np.array([0.2, -0.3])
    same = parallel_plate_shear(
        fluid, np.array([fluid, fluid]), viscosity_pa_s=mu,
        wet_area_m2=area, wet_gap_m=gap,
    )
    np.testing.assert_array_equal(same.wall_force_n, np.zeros((2, 2)))
    assert same.heat_w == 0
    wall = np.array([[0.1, 0.2], [-0.1, -0.2]])
    opposing = parallel_plate_shear(
        np.zeros(2), wall, viscosity_pa_s=mu, wet_area_m2=area, wet_gap_m=gap,
    )
    expected_heat = mu * area / gap * ((0.2) ** 2 + (0.4) ** 2)
    assert opposing.heat_w == pytest.approx(expected_heat, rel=1e-14)
    assert opposing.wall_power_w == pytest.approx(-expected_heat, rel=1e-14)
    forward = parallel_plate_shear(
        np.array([0.0, 0.1]), np.zeros((2, 2)), viscosity_pa_s=mu,
        wet_area_m2=area, wet_gap_m=gap,
    )
    reverse = parallel_plate_shear(
        np.array([0.0, -0.1]), np.zeros((2, 2)), viscosity_pa_s=mu,
        wet_area_m2=area, wet_gap_m=gap,
    )
    np.testing.assert_allclose(forward.wall_force_n, -reverse.wall_force_n, rtol=0, atol=0)
    assert forward.heat_w == reverse.heat_w


def test_closed_wet_pocket_keeps_moving_wall_shear_and_energy_work():
    channel = GlottalChannel(chamber(gap=0, pocket=1e-4, cells=1))
    initial = channel.diagnostics()
    walls = np.array([[[[0.1, 0.2], [-0.1, -0.2]]]])
    step = channel.advance(
        1e-8, Reservoir(0), Reservoir(0), wall_velocity_m_s=walls,
    )
    assert step.inlet_flow_m3_s[0] == step.outlet_flow_m3_s[0] == 0
    assert step.viscous_heat_j > 0
    assert step.shear_wall_work_j < 0
    assert channel.diagnostics()["total_energy_j"] - initial["total_energy_j"] == pytest.approx(
        -step.tangential_wall_work_j, abs=2e-19,
    )
    assert abs(step.energy_residual_j) < 2e-19
    translating = GlottalChannel(
        chamber(gap=0, pocket=1e-4, cells=1), velocity_m_s=0.1
    )
    translation = np.zeros((1, 1, 2, 2))
    translation[..., 1] = 0.1
    result = translating.advance(
        1e-8, Reservoir(0), Reservoir(0), wall_velocity_m_s=translation,
    )
    assert result.viscous_heat_j < 1e-30
    assert result.reflection_heat_j < 1e-30
    np.testing.assert_allclose(result.wall_force_n, 0, atol=1e-20)
