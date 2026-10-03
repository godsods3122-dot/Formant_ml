"""Small CPU tests of physical work, conservation, loading, and independent solutions."""
import math

import numpy as np
import pytest
import torch

from formant_ml.physics import fdtd3d
from formant_ml.physics.acoustic_domain import (
    AcousticDomain, VolumeFlowPort, WallPair, wall_patches,
)
from formant_ml.physics.tract3d import Grid, T_ID

torch.set_num_threads(2)


def grid_from_mask(air, *, h=1.0, exterior=None, tissue="rigid"):
    air = np.asarray(air, bool)
    a, w, c = [], [], []
    for axis in range(3):
        shape = list(air.shape)
        shape[axis] += 1
        left, right = np.zeros(shape, bool), np.zeros(shape, bool)
        sl = [slice(None)] * 3
        sl[axis] = slice(1, None)
        left[tuple(sl)] = air
        sl[axis] = slice(None, -1)
        right[tuple(sl)] = air
        a.append((left & right).astype(float))
        w.append(np.where(left ^ right, T_ID[tissue], 0).astype(np.int8))
        c.append(np.ones(shape))
    inlet = np.zeros_like(air)
    inlet[tuple(np.argwhere(air)[0])] = True
    return Grid(h=h, origin=np.zeros(3), air=air, ax=a[0], ay=a[1], az=a[2],
                wx=w[0], wy=w[1], wz=w[2], cx=c[0], cy=c[1], cz=c[2],
                inlet=inlet, exterior=np.zeros_like(air) if exterior is None else exterior,
                exit_point=np.zeros(2), exit_dir=np.array([1., 0.]),
                area_1d=np.zeros(1), pos_1d=np.zeros(1))


def domain(g, ports=None, **kwargs):
    return AcousticDomain(g, {"in": VolumeFlowPort(g.inlet)} if ports is None else ports,
                          boundary_layer=False, sponge_cells=0, **kwargs)


def two_cavities(*, damping=0.0, courant=0.4, mass=0.02, boundary_layer=False, walls="rigid"):
    air = np.zeros((5, 3, 3), bool)
    air[1, 1, 1] = air[3, 1, 1] = True
    ext = np.zeros_like(air)
    ext[3, 1, 1] = True
    g = grid_from_mask(air, exterior=ext)
    g.wx[2, 1, 1] = g.wx[3, 1, 1] = T_ID["cheek"]
    patches = wall_patches(g)
    inner = tuple(p.index for p in patches if p.tissue == "cheek" and not p.exterior)
    outer = tuple(p.index for p in patches if p.tissue == "cheek" and p.exterior)
    pair = WallPair("cheek", inner, outer, mass, 1000., damping)
    ports = {"inside": VolumeFlowPort(g.inlet), "outside": VolumeFlowPort(ext)}
    d = AcousticDomain(g, ports, wall_pairs=[pair], courant=courant,
                       boundary_layer=boundary_layer, sponge_cells=0, walls=walls)
    return g, d, pair


def test_si_conversion_conservation_signed_flow_and_legacy_equivalence():
    g = grid_from_mask(np.ones((4, 3, 3), bool))
    d = domain(g)
    old = fdtd3d.Sim(g, courant=d.dt * d.sound_speed_m_s * math.sqrt(3) / d.h_m,
                     walls="rigid", boundary_layer=False, sponge_cells=0,
                     device="cpu", dtype=torch.float64)
    q = 2e-7
    result = d.advance({"in": q})
    old.step(q * 1e6)
    assert result.pressure_pa["in"] == pytest.approx(old.p[g.inlet].mean().item() * .1)
    assert result.pressure_pa["in"] == pytest.approx(d.bulk_modulus_pa * d.dt * q / d.h_m ** 3)
    np.testing.assert_allclose(d._sim.p.numpy(), old.p.numpy(), atol=1e-12)
    for i in range(80):
        d.advance({"in": -q if i % 2 else q / 2})
    diag = d.diagnostics()
    assert diag["compression_volume_m3"] == pytest.approx(diag["input_volume_m3"]["in"], rel=1e-12)
    assert diag["staggered_energy_j"] == pytest.approx(diag["input_work_j"], rel=2e-12)


def test_reciprocal_multiport_impulse_responses_and_power_weights():
    g = grid_from_mask(np.ones((7, 4, 3), bool))
    a, b = np.zeros_like(g.air), np.zeros_like(g.air)
    a[1, 1:3, 1] = True
    b[5, 1:3, 1] = True
    weights = np.ones(g.shape)
    weights[1, 1, 1], weights[5, 1, 1] = 3, 2
    ports = {"a": VolumeFlowPort(a, weights), "b": VolumeFlowPort(b, weights)}
    d = domain(g, ports)
    initial = d.snapshot()
    h_ab, h_ba = [], []
    for source, receiver, response in (("a", "b", h_ab), ("b", "a", h_ba)):
        d.restore(initial)
        for n in range(90):
            response.append(d.advance({source: 1e-7 if n == 0 else 0}).pressure_pa[receiver])
    np.testing.assert_allclose(h_ab, h_ba, atol=2e-14, rtol=1e-11)
    d.restore(initial)
    r = d.advance({"a": 2e-7, "b": -1e-7})
    e = d.diagnostics()
    expected = d.dt * (2e-7 * r.work_pressure_pa["a"] - 1e-7 * r.work_pressure_pa["b"])
    assert e["staggered_energy_j"] == pytest.approx(expected, rel=2e-12)


def test_physical_cut_volume_no_inflation_or_source_floor():
    g = grid_from_mask(np.ones((3, 2, 2), bool))
    patches = wall_patches(g)
    vf = np.ones(g.shape)
    vf[0, 0, 0] = 0.02
    g.meta = dict(cut=True, vf=vf, walls=dict(
        cell=np.array([np.ravel_multi_index(p.cell, g.shape) for p in patches]),
        area=np.array([p.area_m2 * 1e4 for p in patches]),
        tis=np.array([T_ID[p.tissue] for p in patches]),
        pos=np.array([p.position_m for p in patches]) * 100,
        dir=np.array([p.outward_normal for p in patches])))
    mask = g.inlet.copy()
    mask[1, 0, 0] = True
    d = domain(g, {"in": VolumeFlowPort(mask)}, courant=.9)
    np.testing.assert_array_equal(d._sim.veff, vf)
    d.advance({"in": 1e-7})
    p = d._sim.p.numpy() * .1
    assert p[0, 0, 0] == pytest.approx(p[1, 0, 0], rel=2e-12)
    e = d.diagnostics()
    assert e["compression_volume_m3"] == pytest.approx(d.dt * 1e-7, rel=2e-12)
    for _ in range(300):
        d.advance({})
    assert d.diagnostics()["staggered_energy_j"] == pytest.approx(e["staggered_energy_j"], rel=2e-12)


@pytest.mark.parametrize("damping", [0., 100.])
def test_pair_passive_staggered_energy_opposite_flow_and_source_work(damping):
    g, d, _ = two_cavities(damping=damping)
    d.advance({"inside": 1e-7})
    initial = d.diagnostics()
    energies = []
    for _ in range(600):
        before_inner = d.pressure_pa("inside")
        before_outer = d.pressure_pa("outside")
        result = d.advance({})
        q = result.wall_flow_m3_s["cheek"]
        scale = d.dt * d.bulk_modulus_pa / d.h_m ** 3
        assert result.pressure_pa["inside"] - before_inner == pytest.approx(-scale * q, abs=1e-13)
        assert result.pressure_pa["outside"] - before_outer == pytest.approx(scale * q, abs=1e-13)
        diag = d.diagnostics()
        energies.append(diag["staggered_energy_j"])
        assert diag["compression_volume_m3"] == pytest.approx(initial["compression_volume_m3"], rel=2e-12)
        assert diag["staggered_energy_j"] + diag["pair_dissipation_j"] == pytest.approx(
            initial["staggered_energy_j"], rel=2e-11)
    assert max(energies) <= initial["staggered_energy_j"] * (1 + 1e-11)
    if damping:
        assert energies[-1] < initial["staggered_energy_j"] * .6
    else:
        assert min(energies) > initial["staggered_energy_j"] * (1 - 1e-11)
    assert d.diagnostics()["energy_certified"]
    point = (np.array([3, 1, 1]) + .5) * d.h_m
    assert d.sample_pressure_pa([point])[0] == d.pressure_pa("outside")


def test_pair_coupled_field_energy_with_propagating_air_faces():
    air = np.zeros((9, 5, 5), bool)
    air[1:4, 1:4, 1:4] = air[5:8, 1:4, 1:4] = True
    ext = np.zeros_like(air)
    ext[5:8] = air[5:8]
    g = grid_from_mask(air, exterior=ext)
    g.wx[4, 2, 2] = g.wx[5, 2, 2] = T_ID["cheek"]
    patches = wall_patches(g)
    pair = WallPair("wall", tuple(p.index for p in patches if p.tissue == "cheek" and not p.exterior),
                    tuple(p.index for p in patches if p.tissue == "cheek" and p.exterior), .001, 20000., 2.)
    d = domain(g, wall_pairs=[pair], courant=.9)
    d.advance({"in": 1e-7})
    e0 = d.diagnostics()["staggered_energy_j"]
    for _ in range(400):
        d.advance({})
    diag = d.diagnostics()
    assert diag["staggered_energy_j"] + diag["pair_dissipation_j"] == pytest.approx(e0, rel=2e-12)
    assert np.max(np.abs(d._sim.p.numpy()[ext])) > 1e-8


def test_independent_two_cavity_solution_and_second_order_dt_convergence():
    errors = []
    for courant in (.6, .3, .15):
        _, d, pair = two_cavities(courant=courant)
        d.advance({"inside": 1e-7})
        dp0 = d.pressure_pa("inside")
        area, volume = d.h_m ** 2, d.h_m ** 3
        air_stiffness = 2 * d.bulk_modulus_pa * area / volume
        omega = math.sqrt((pair.stiffness_pa_m + air_stiffness) / pair.mass_kg_m2)
        # Initialize the staggered half-time velocity from the independent
        # continuous solution with x(0)=x'(0)=0 and imposed initial pressure.
        d._v[0] = -dp0 / (pair.mass_kg_m2 * omega) * math.sin(omega * d.dt / 2)
        n = round(1.7 * math.pi / (omega * d.dt))
        for _ in range(n):
            d.advance({})
        exact_x = dp0 / (pair.mass_kg_m2 * omega ** 2) * (1 - math.cos(omega * n * d.dt))
        numerical_x = d.diagnostics()["wall_displacement_m"]["cheek"]
        errors.append(abs((numerical_x - exact_x) / (dp0 / (pair.mass_kg_m2 * omega ** 2))))
    assert errors[-1] < .002
    assert errors[0] / errors[1] > 3
    assert errors[1] / errors[2] > 3


def test_multiport_work_balance_while_driving_a_damped_pair():
    _, d, _ = two_cavities(damping=10)
    for n in range(400):
        d.advance({"inside": 1e-7 * math.sin(n / 11), "outside": -2e-8 * math.cos(n / 7)})
    e = d.diagnostics()
    assert e["staggered_energy_j"] + e["pair_dissipation_j"] == pytest.approx(
        e["input_work_j"], rel=3e-12)
    assert e["compression_volume_m3"] == pytest.approx(sum(e["input_volume_m3"].values()), rel=3e-12)


def test_rigid_vs_soft_feedback_and_snapshot_replay():
    g, soft, _ = two_cavities()
    rigid = domain(g, {"inside": VolumeFlowPort(g.inlet)}, courant=soft.dt * soft.sound_speed_m_s * math.sqrt(3) / soft.h_m)
    soft.advance({"inside": 1e-7})
    rigid.advance({"inside": 1e-7})
    saved = soft.snapshot()
    out = [soft.advance({"inside": -2e-8 if n < 3 else 0}) for n in range(15)]
    after = soft.diagnostics()
    soft.restore(saved)
    assert out == [soft.advance({"inside": -2e-8 if n < 3 else 0}) for n in range(15)]
    assert after == soft.diagnostics()
    rigid.advance({})
    assert soft.pressure_pa("outside") != 0
    assert soft.pressure_pa("inside") != pytest.approx(rigid.pressure_pa("inside"))
    with pytest.raises(ValueError, match="different"):
        rigid.restore(saved)


def test_pair_retains_distinct_boundary_layers_and_no_local_structural_sink():
    _, d, pair = two_cavities(boundary_layer=True, walls="soft")
    assert not d._sim.soft[list(pair.inside + pair.outside)].any()
    d.advance({"inside": 1e-7})
    state = d.snapshot()
    for _ in range(15):
        d.advance({})
    expected = d._sim.p.clone()
    e = d.diagnostics()
    assert e["boundary_layer_removed_volume_m3"] != 0
    assert e["local_wall_removed_volume_m3"] == 0
    assert e["paired_soft_area_fraction"] == pytest.approx(1)
    assert not e["energy_certified"]
    d.restore(state)
    for _ in range(15):
        d.advance({})
    torch.testing.assert_close(d._sim.p, expected, rtol=0, atol=0)


def test_exterior_loading_returns_pressure_and_reverse_wall_flow():
    _, d, _ = two_cavities()
    d.advance({"outside": 1e-7})
    result = d.advance({})
    assert result.wall_flow_m3_s["cheek"] < 0
    assert result.pressure_pa["inside"] > 0


def test_unpaired_local_approximation_and_coverage_remain_explicit():
    g, _, _ = two_cavities()
    g.wy[1, 1, 1] = T_ID["tongue"]
    patches = wall_patches(g)
    pair = WallPair("cheek", tuple(p.index for p in patches if p.tissue == "cheek" and not p.exterior),
                    tuple(p.index for p in patches if p.tissue == "cheek" and p.exterior), .02, 1000.)
    d = domain(g, wall_pairs=[pair], walls="soft")
    d.advance({"in": 1e-7})
    for _ in range(20):
        d.advance({})
    e = d.diagnostics()
    assert e["paired_soft_area_fraction"] == pytest.approx(2 / 3)
    assert e["local_wall_removed_volume_m3"] > 0
    assert e["local_wall_energy_j"] > 0
    assert not e["energy_certified"]


def test_pml_source_off_decay_and_receiver_validation():
    air = np.ones((18, 18, 18), bool)
    g = grid_from_mask(air, exterior=air)
    g.inlet[:] = False
    g.inlet[9, 9, 9] = True
    d = AcousticDomain(g, {"in": VolumeFlowPort(g.inlet)}, boundary_layer=False,
                       sponge_cells=4, courant=.8)
    for n in range(30):
        d.advance({"in": 1e-7 * math.sin(2 * math.pi * n / 30)})
    initial = d.diagnostics()["acoustic_energy_j"]
    for _ in range(240):
        d.advance({})
    assert d.diagnostics()["acoustic_energy_j"] < initial * .03
    with pytest.raises(ValueError, match="Receiver"):
        d.sample_pressure_pa([[-1, 0, 0]])
    g.inlet[:] = False
    g.inlet[0, 0, 0] = True
    with pytest.raises(ValueError, match="PML"):
        AcousticDomain(g, {"in": VolumeFlowPort(g.inlet)}, sponge_cells=4, boundary_layer=False)


def test_invalid_ports_and_pairs_fail_explicitly():
    g, d, pair = two_cavities()
    with pytest.raises(ValueError, match="Unknown"):
        d.advance({"wrong": 1})
    with pytest.raises(ValueError, match="finite"):
        d.advance({"inside": float("nan")})
    with pytest.raises(ValueError, match="nonnegative"):
        domain(g, {"bad": VolumeFlowPort(g.inlet, -np.ones(g.shape))})
    bad = np.ones(g.shape, bool)
    with pytest.raises(ValueError, match="solid"):
        domain(g, {"bad": VolumeFlowPort(bad)})
    with pytest.raises(ValueError, match="twice"):
        domain(g, wall_pairs=[pair, WallPair("duplicate", pair.inside, pair.outside, 1, 1)])
    with pytest.raises(ValueError, match="positive"):
        domain(g, wall_pairs=[WallPair("bad", pair.inside, pair.outside, 0, 1)])


def test_shared_clock_cap_is_exact_fixed_and_validated():
    g = grid_from_mask(np.ones((3, 3, 3), bool))
    d = domain(g)
    h = d.dt / 7
    capped = domain(g, max_dt_s=h)
    assert capped.dt == pytest.approx(h, rel=1e-15)
    with pytest.raises(AttributeError):
        capped.dt = 2 * h
    for _ in range(5):
        capped.advance({})
    assert capped.time_s == pytest.approx(5 * h, rel=1e-15)
    for invalid in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="max_dt_s"):
            domain(g, max_dt_s=invalid)


@pytest.mark.parametrize("sparse_flow", [False, True])
def test_cpu_tensor_step_parity_for_legacy_and_new_sparse_flow(sparse_flow):
    g, _, pair = two_cavities()
    for inactive in ((), pair.inside + pair.outside):
        a, b = [fdtd3d.Sim(g, device="cpu", dtype=torch.float64, walls="soft",
                          boundary_layer=True, sponge_cells=0, inactive_walls=inactive) for _ in range(2)]
        b._prep_t()
        ids = torch.tensor([np.ravel_multi_index((3, 1, 1), g.shape)])
        for n in range(30):
            q = .1 * math.sin(n / 10)
            cell_flows = (ids, torch.tensor([-q / 2], dtype=torch.float64)) if sparse_flow else None
            a.step(q, cell_flows=cell_flows)
            b._step_t(torch.tensor(q, dtype=torch.float64), cell_flows=cell_flows)
        for x, y in zip(a._state(), b._state()):
            torch.testing.assert_close(x, y, rtol=1e-12, atol=1e-14)
