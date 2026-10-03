"""Nominal voxel fixtures only: no claim of subject-specific nasal anatomy."""
import numpy as np
import pytest

from formant_ml.physics.airway_geometry import build_airway_grid, component_labels, validate_grid
from formant_ml.physics.acoustic_domain import AcousticDomain, VolumeFlowPort
from formant_ml.physics.tract3d import T_ID
from .test_acoustic_domain import grid_from_mask


def anatomy():
    shape = (10, 9, 7)
    oral, left, right, ext, vp = [np.zeros(shape, bool) for _ in range(5)]
    oral[1:7, 1:3, 3] = True
    left[3:7, 5:7, 1] = True
    right[3:7, 5:7, 5] = True
    vp[2:4, 3:5, 1:6] = True
    vp[2:4, 2, 1:6] = True
    vp &= ~oral
    # Vertical nasal inlets are part of the connector.
    vp[3, 4, 1] = vp[3, 4, 5] = True
    ext[7:9, 1:8, :] = True
    g = grid_from_mask(oral | ext, exterior=ext, tissue="cheek")
    g.inlet[:] = False
    g.inlet[1, 1:3, 3] = True
    return g, dict(left_nasal=left, right_nasal=right, exterior=ext,
                   velopharyngeal=vp, solid_tissue=np.full(shape, T_ID["cheek"]))


def test_fixed_bilateral_nasal_open_closed_and_fractional_aperture():
    g, args = anatomy()
    opened = build_airway_grid(g, **args)
    closed = build_airway_grid(g, **args, aperture_fraction=0)
    half = build_airway_grid(g, **args, aperture_fraction=.5)
    assert not closed.air[args["velopharyngeal"]].any()
    assert any(np.any(f == .5) for f in (half.ax, half.ay, half.az))
    assert len(np.unique(component_labels(opened)[opened.air])) == 1
    internal = closed.air & ~closed.exterior
    assert len(np.unique(component_labels(closed, internal)[internal])) == 3
    assert opened.meta["fixed_pose"]
    assert np.array_equal(g.inlet, opened.inlet)
    assert np.any(opened.wx == T_ID["cheek"])
    assert np.any(closed.wy == T_ID["velum"])


def test_closure_blocks_internal_pressure_before_exterior_path_arrives():
    g, args = anatomy()
    # Mouth/nares are well separated from posterior VP in this nominal fixture.
    receiver = np.zeros(g.shape, bool)
    receiver[3, 5, 1] = True
    responses = []
    for fraction in (0, 1):
        airway = build_airway_grid(g, **args, aperture_fraction=fraction)
        src = np.zeros(g.shape, bool)
        src[2, 2, 3] = True
        d = AcousticDomain(airway, {"src": VolumeFlowPort(src), "nose": VolumeFlowPort(receiver)},
                           boundary_layer=False, sponge_cells=0)
        trace = [d.advance({"src": 1e-7 if n == 0 else 0}).pressure_pa["nose"] for n in range(9)]
        responses.append(trace)
    assert np.max(np.abs(responses[0])) == 0
    assert np.max(np.abs(responses[1])) > 0


def test_preserves_oral_fractional_faces_and_rigid_labels():
    g, args = anatomy()
    g.ax[2, 1, 3] = .37
    g.wz[1, 1, 3] = T_ID["rigid"]
    out = build_airway_grid(g, **args)
    assert out.ax[2, 1, 3] == .37
    assert out.wz[1, 1, 3] == T_ID["rigid"]
    assert g.air[args["left_nasal"]].sum() == 0
    assert out.meta["n_air"] == out.air.sum()


def test_aperture_survives_integer_input_faces_and_preserves_supplied_new_tissue():
    g, args = anatomy()
    g.ax, g.ay, g.az = (f.astype(np.int8) for f in (g.ax, g.ay, g.az))
    args["solid_tissue"][:] = T_ID["pharynx"]
    half = build_airway_grid(g, **args, aperture_fraction=.5)
    assert any(np.any(f == .5) for f in (half.ax, half.ay, half.az))
    assert any(np.any(w == T_ID["pharynx"]) for w in (half.wx, half.wy, half.wz))
    assert not any(np.any(w == T_ID["velum"]) for w in (half.wx, half.wy, half.wz))


def test_rejects_malformed_connectivity_masks_and_rigid_carving():
    g, args = anatomy()
    with pytest.raises(ValueError, match="boolean"):
        build_airway_grid(g, **dict(args, left_nasal=args["left_nasal"].astype(float)))
    bad = args["left_nasal"].copy()
    bad[1, 1, 3] = True
    with pytest.raises(ValueError, match="disjoint"):
        build_airway_grid(g, **dict(args, left_nasal=bad))
    bad = args["left_nasal"].copy()
    bad[5, 5:7, 1] = False
    with pytest.raises(ValueError, match="face-connected"):
        build_airway_grid(g, **dict(args, left_nasal=bad))
    g.wy[2, 3, 3] = T_ID["rigid"]
    with pytest.raises(ValueError, match="rigid"):
        build_airway_grid(g, **args)


def test_rejects_open_faces_to_solid_and_bad_fraction():
    g, args = anatomy()
    with pytest.raises(ValueError, match="aperture_fraction"):
        build_airway_grid(g, **args, aperture_fraction=-.1)
    g.ax[0, 1, 3] = 1
    with pytest.raises(ValueError, match="Open faces"):
        validate_grid(g)
