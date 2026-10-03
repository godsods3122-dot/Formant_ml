"""Direct, unfitted forward execution of the optional coupled reference model.

Example: python scripts\\phonate3d.py --config examples\\phonation3d_reference.json

The included scene is synthetic apparatus, NOT measured human anatomy. Raw
receiver pressure is retained in Pa. No original recording, F0 track, noise,
normalization, fitted filter or prescribed tissue oscillation is consumed.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import soundfile as sf
import torch
import scipy
from scipy.signal import resample_poly
from fractions import Fraction

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formant_ml.physics import air
from formant_ml.physics.acoustic_domain import AcousticDomain, VolumeFlowPort, WallPair, wall_patches
from formant_ml.physics.airway_geometry import build_airway_grid
from formant_ml.physics.glottal_channel import ChannelConfig, GlottalChannel
from formant_ml.physics.phonation import (
    CouplingConfig, EulerianGlottis, LungConfig, LungReservoir, PhonationSystem,
    PortWallPatch, SupplyDuct, _interpolate_control,
)
from formant_ml.physics.tract3d import Grid, T_ID
from formant_ml.physics.vocal_fold_solid import (
    BilateralControl, FoldControl, FoldMesh, PressurePatch, SolidConfig,
    SolidMaterial, VocalFoldSolid, nominal_fold_mesh, nominal_materials,
)


def json_value(value):
    if is_dataclass(value):
        return {f.name: json_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def grid_from_mask(mask, exterior, inlet, tissue, *, h_m, origin_m):
    """Explicit voxel faces; no inferred nasal connections or hidden area floors."""
    openings, wall_ids, area_factors = [], [], []
    for axis in range(3):
        shape = list(mask.shape)
        shape[axis] += 1
        left, right = np.zeros(shape, bool), np.zeros(shape, bool)
        low_tissue, high_tissue = np.zeros(shape, np.int8), np.zeros(shape, np.int8)
        sl = [slice(None)] * 3
        sl[axis] = slice(1, None)
        left[tuple(sl)], low_tissue[tuple(sl)] = mask, tissue
        sl[axis] = slice(None, -1)
        right[tuple(sl)], high_tissue[tuple(sl)] = mask, tissue
        opened = (left & right).astype(float)
        wall = np.where(left ^ right, np.where(left, high_tissue, low_tissue), 0)
        wall[(left ^ right) & (wall == 0)] = T_ID["rigid"]
        openings.append(opened)
        wall_ids.append(wall.astype(np.int8))
        area_factors.append(np.ones(shape))
    return Grid(
        h=h_m * 100, origin=np.asarray(origin_m) * 100, air=mask.copy(),
        ax=openings[0], ay=openings[1], az=openings[2],
        wx=wall_ids[0], wy=wall_ids[1], wz=wall_ids[2],
        cx=area_factors[0], cy=area_factors[1], cz=area_factors[2],
        inlet=inlet.copy(), exterior=exterior.copy(), exit_point=np.zeros(2),
        exit_dir=np.array([1., 0.]), area_1d=np.zeros(1), pos_1d=np.zeros(1),
    )


def reference_airway(config):
    """Small oral tube, two nasal branches, VP connector and surrounding air."""
    h = float(config.get("cell_size_m", 0.002))
    shape = (19, 19, 20)
    head = np.zeros(shape, bool)
    head[4:15, 4:15, :12] = True
    exterior = ~head
    oral = np.zeros(shape, bool)
    oral[5:10, 8, 3:12] = True
    left, right, vp = (np.zeros(shape, bool) for _ in range(3))
    left[5, 5, 6:12] = True
    right[5, 11, 6:12] = True
    vp[10, 5:12, 5] = True
    vp[5:11, 5, 5] = vp[5:11, 11, 5] = True
    tissue = np.full(shape, T_ID["cheek"], np.int8)
    # A specified rigid separator, not a claim about personalized histology.
    tissue[5:10, 7, 7:12] = T_ID["rigid"]
    inlet = np.zeros(shape, bool)
    inlet[5:10, 8, 3] = True
    origin = np.array([-5 * h, -8.5 * h, -2 * h])
    old = grid_from_mask(oral | exterior, exterior, inlet, tissue, h_m=h, origin_m=origin)
    grid = build_airway_grid(
        old, left_nasal=left, right_nasal=right, exterior=exterior,
        velopharyngeal=vp, solid_tissue=tissue,
        aperture_fraction=float(config.get("vp_aperture_fraction", 1)),
    )
    patches = wall_patches(grid)
    inside = tuple(p.index for p in patches
                   if p.cell == (5, 8, 8) and p.outward_normal == (-1., 0., 0.))
    outside = tuple(p.index for p in patches
                    if p.cell == (3, 8, 8) and p.outward_normal == (1., 0., 0.))
    if len(inside) != 1 or len(outside) != 1:
        raise ValueError("reference wall correspondence does not match its supplied geometry")
    pair = WallPair("reference_cheek", inside, outside, 2.08, 2.5e6, 100.0)
    receiver = origin + (np.array([[7, 8, 15], [2, 8, 8]]) + 0.5) * h
    provenance = {
        "origin": "synthetic small test apparatus; not measured/personalized human anatomy",
        "cell_size_m": h, "grid_shape": shape, "origin_m": origin,
        "oral_length_m": 9 * h, "nasal_branch_length_m": 6 * h,
        "wall_pair": pair,
        "radiation": "mouth, both nasal exits and paired cheek use the same exterior field",
    }
    return grid, {"glottis": VolumeFlowPort(inlet)}, (pair,), receiver, provenance


def reference_solid(config):
    depth = float(config.get("recess_depth_m", 2e-5))
    if not math.isfinite(depth) or depth < 0:
        raise ValueError("reference recess depth must be finite and nonnegative")
    half_gap = float(config.get("half_gap_m", 5e-5))
    meshes = []
    for side in ("left", "right"):
        mesh = nominal_fold_mesh(
            side, n_ap=int(config.get("n_ap", 2)), n_si=int(config.get("n_si", 2)),
            half_gap_m=half_gap,
        )
        coordinates = mesh.reference_m.copy()
        profile = 1 - np.abs(coordinates[:, 2] - 0.001) / 0.001
        through_depth = (np.abs(coordinates[:, 1]) - half_gap) / 0.006
        sign = -1 if side == "left" else 1
        coordinates[:, 1] += sign * depth * profile * (1 - through_depth)
        meshes.append(replace(mesh, reference_m=coordinates))
    return VocalFoldSolid(SolidConfig(*meshes))


def load_grid_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        values = {}
        for f in fields(Grid):
            if f.name == "meta":
                values["meta"] = json.loads(str(archive["meta_json"])) if "meta_json" in archive else {}
            else:
                key = "grid_" + f.name
                if key not in archive:
                    raise ValueError(f"supplied scene archive is missing {key}")
                value = archive[key].copy()
                values[f.name] = float(value) if f.name == "h" else value
        ports = {k.removeprefix("port_"): VolumeFlowPort(archive[k].copy())
                 for k in archive.files if k.startswith("port_")}
        receivers = archive["receiver_points_m"].copy()
    return Grid(**values), ports, receivers


def load_solid_archive(path, material_config):
    meshes = []
    with np.load(path, allow_pickle=False) as archive:
        for side in ("left", "right"):
            values = {}
            for f in fields(FoldMesh):
                key = side + "_" + f.name
                if key not in archive and f.name == "medial_grid":
                    values[f.name] = None
                elif key not in archive:
                    raise ValueError(f"supplied solid archive is missing {key}")
                else:
                    values[f.name] = archive[key].copy()
            meshes.append(FoldMesh(**values))
    materials = (tuple(SolidMaterial(**m) for m in material_config)
                 if material_config is not None else nominal_materials())
    return VocalFoldSolid(SolidConfig(*meshes, materials=materials))


def declared_end_walls(solid, mapper, explicit):
    if explicit is not None:
        return tuple(PortWallPatch(
            w["end"], int(w["strip"]),
            PressurePatch(w["side"], tuple(w["triangle_nodes"]), 0,
                          w.get("barycentric_vertices", np.eye(3))),
        ) for w in explicit)
    patches = []
    for side in ("left", "right"):
        coordinates = solid.surface(side).coordinates_m
        for triangle in solid.boundary_triangles(side):
            xyz = coordinates[triangle]
            if np.max(np.abs(xyz[:, 1])) > 0.0016:
                continue
            if np.all(xyz[:, 2] == 0):
                end = "upstream"
            elif np.all(xyz[:, 2] == 0.002):
                end = "downstream"
            else:
                continue
            patches.append(PortWallPatch(end, 0, PressurePatch(side, tuple(triangle), 0)))
    return tuple(patches)


def build_system(config, base_directory):
    properties = air.props(**config.get("air", {}))
    scene = config.get("scene", {})
    assets = {}

    def asset(name):
        path = (base_directory / config[name]).resolve()
        assets[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    if "scene_npz" in config:
        grid, ports, receivers = load_grid_archive(asset("scene_npz"))
        pairs = tuple(WallPair(**w) for w in config.get("wall_pairs", []))
        if not config.get("anatomy_provenance"):
            raise ValueError("supplied anatomy requires anatomy_provenance")
        provenance = {"origin": config["anatomy_provenance"]}
    else:
        grid, ports, pairs, receivers, provenance = reference_airway(scene)
    if "solid_mesh_npz" in config:
        solid = load_solid_archive(asset("solid_mesh_npz"), config.get("materials"))
        if "port_wall_patches" not in config:
            raise ValueError("supplied mesh requires explicit port_wall_patches (possibly empty)")
    else:
        solid = reference_solid(config.get("solid", {}))
    mapper = EulerianGlottis(
        config.get("ap_edges_m", [0.001, 0.009]),
        config.get("si_edges_m", [0.0002, 0.0018]),
        provenance=str(provenance["origin"]) + "; actual supplied/reference solid recess geometry",
        neck_half_width_m=float(config.get("neck_half_width_m", 0)),
    )
    channel = GlottalChannel(
        mapper.capture(solid).geometry,
        ChannelConfig(properties=properties, **config.get("channel", {})),
    )
    lung = LungReservoir(LungConfig(properties=properties, **config.get("lung", {})))
    options = dict(
        T_c=properties.T_c, rh=float(config.get("air", {}).get("rh", 1)),
        boundary_layer=bool(scene.get("boundary_layer", True)),
        sponge_cells=int(scene.get("sponge_cells", 2)), walls="rigid", device="cpu",
        dtype=torch.float64, max_dt_s=float(config.get("max_dt_s", 5e-7)),
    )
    downstream = AcousticDomain(grid, ports, wall_pairs=pairs, **options)
    upstream, supply = None, None
    upstream_ports, lung_port = (), None
    if "upstream_npz" in config:
        up_grid, up_ports, _ = load_grid_archive(asset("upstream_npz"))
        upstream = AcousticDomain(up_grid, up_ports, **options)
        # Actual accepted clocks can be smaller than a requested cap.
        for _ in range(16):
            common = min(downstream.dt, upstream.dt)
            downstream = AcousticDomain(grid, ports, wall_pairs=pairs, **dict(options, max_dt_s=common))
            upstream = AcousticDomain(up_grid, up_ports, **dict(options, max_dt_s=common))
            if math.isclose(downstream.dt, upstream.dt, rel_tol=1e-12, abs_tol=0):
                break
        else:
            raise ValueError("upstream/downstream common physical clock did not converge")
        upstream_ports = tuple(config["upstream_glottis_ports"])
        lung_port = config["upstream_lung_port"]
        supply = SupplyDuct(properties=properties, **config["supply_duct"])
    downstream_names = tuple(config.get("downstream_ports", ["glottis"]))
    end_walls = declared_end_walls(solid, mapper, config.get("port_wall_patches"))
    system = PhonationSystem(
        solid, channel, mapper, lung, downstream, downstream_names,
        wall_patches=end_walls, config=CouplingConfig(**config.get("coupling", {})),
        upstream=upstream, upstream_ports=upstream_ports,
        upstream_lung_port=lung_port, supply=supply,
    )
    provenance.update({
        "solid_reference": config.get("solid", {}),
        "ap_edges_m": mapper.ap_edges_m, "si_edges_m": mapper.si_edges_m,
        "neck_half_width_m": mapper.neck_half_width_m,
        "port_wall_patches": end_walls, "assets_sha256": assets,
        "remaining_limits": [
            "fixed tract pose; synthetic reference is not measured human anatomy",
            "independent AP strips; compressible low-Mach chamber/throat approximation",
            "unresolved turbulent moving-boundary CFD, mucosal wetting and general contact topology",
            "disconnected pockets within a bin or collapsed retained storage reject",
            "linear acoustic entropy/mean-density transport is a declared reservoir approximation",
            "compact plenum patches assume the declared material faces remain wetted",
            "no held-out human or perceptual identity validation",
        ],
    })
    return system, np.asarray(config.get("receiver_points_m", receivers), float), provenance


def trajectory(config):
    knots = config.get("controls", [{"time_s": 0.0, "muscle_pressure_pa": 0.0}])
    times, muscles, controls = [], [], []
    for knot in knots:
        times.append(float(knot["time_s"]))
        muscles.append(float(knot["muscle_pressure_pa"]))
        controls.append(BilateralControl(
            FoldControl(**knot.get("left", {})), FoldControl(**knot.get("right", {})),
        ))
    if (not times or times[0] != 0 or not np.isfinite(times).all()
            or not np.isfinite(muscles).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("controls require finite strictly increasing knots starting at zero")

    def at(t):
        index = min(max(int(np.searchsorted(times, t, side="right")) - 1, 0), len(times) - 1)
        if index == len(times) - 1:
            return controls[index], muscles[index]
        fraction = (t - times[index]) / (times[index + 1] - times[index])
        return (
            _interpolate_control(controls[index], controls[index + 1], fraction),
            (1 - fraction) * muscles[index] + fraction * muscles[index + 1],
        )
    return at


def _pack(value, arrays, name):
    if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
        arrays[name] = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        return {"array": name}
    if is_dataclass(value):
        return {"fields": {f.name: _pack(getattr(value, f.name), arrays, name + "." + f.name)
                           for f in fields(value) if f.name != "owner"}}
    if isinstance(value, tuple):
        return {"tuple": [_pack(v, arrays, name + "." + str(i)) for i, v in enumerate(value)]}
    if isinstance(value, dict):
        return {"dict": {k: _pack(v, arrays, name + "." + k) for k, v in value.items()}}
    return json_value(value)


def _unpack(template, encoded, arrays):
    if isinstance(template, np.ndarray) or isinstance(template, torch.Tensor):
        value = np.array(arrays[encoded["array"]], copy=True)
        if value.shape != template.shape or value.dtype != (
            template.detach().cpu().numpy().dtype if isinstance(template, torch.Tensor)
            else template.dtype
        ):
            raise ValueError("checkpoint array shape or dtype does not match its configuration")
        return (torch.as_tensor(value, device=template.device, dtype=template.dtype)
                if isinstance(template, torch.Tensor) else value)
    if is_dataclass(template):
        names = {f.name for f in fields(template) if f.name != "owner"}
        if names != set(encoded["fields"]):
            raise ValueError("checkpoint dataclass schema mismatch")
        return replace(template, **{
            name: _unpack(getattr(template, name), encoded["fields"][name], arrays)
            for name in names
        })
    if isinstance(template, tuple):
        values = encoded["tuple"]
        if len(values) != len(template):
            raise ValueError("checkpoint tuple schema mismatch")
        return tuple(_unpack(t, v, arrays) for t, v in zip(template, values))
    if isinstance(template, dict):
        values = encoded["dict"]
        if set(values) != set(template):
            raise ValueError("checkpoint mapping schema mismatch")
        return {k: _unpack(template[k], values[k], arrays) for k in template}
    if template is None and encoded is not None:
        raise ValueError("checkpoint optional component mismatch")
    return encoded


def source_signature(config, provenance):
    root = Path(__file__).resolve().parents[1]
    names = (
        Path(__file__), root / "src" / "formant_ml" / "physics" / "phonation.py",
        root / "src" / "formant_ml" / "physics" / "glottal_channel.py",
        root / "src" / "formant_ml" / "physics" / "vocal_fold_solid.py",
        root / "src" / "formant_ml" / "physics" / "acoustic_domain.py",
        root / "src" / "formant_ml" / "physics" / "fdtd3d.py",
        root / "src" / "formant_ml" / "physics" / "softtissue.py",
        root / "src" / "formant_ml" / "physics" / "airway_geometry.py",
        root / "src" / "formant_ml" / "physics" / "air.py",
        root / "src" / "formant_ml" / "physics" / "lungs.py",
        root / "src" / "formant_ml" / "physics" / "fold_rules.py",
        root / "src" / "formant_ml" / "physics" / "tract3d.py",
    )
    return {
        "configuration_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in names},
        "assets_sha256": provenance["assets_sha256"],
        "runtime": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "torch": torch.__version__,
        },
    }


def save_checkpoint(path, system, signature):
    arrays = {}
    state = _pack(system.snapshot(), arrays, "state")
    arrays["checkpoint_json"] = np.asarray(json.dumps({"signature": signature, "state": state}))
    np.savez_compressed(path, **arrays)


def load_checkpoint(path, system, signature):
    with np.load(path, allow_pickle=False) as archive:
        document = json.loads(str(archive["checkpoint_json"]))
        if document["signature"] != signature:
            raise ValueError("checkpoint configuration, anatomy or source version mismatch")
        state = _unpack(system.snapshot(), document["state"], archive)
    system.restore(state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("out") / "phonation3d")
    parser.add_argument("--steps", type=int, help="override number of physical acoustic ticks")
    parser.add_argument("--resume", type=Path, help="validated non-pickle component checkpoint")
    parser.add_argument("--wav", action="store_true", help="export using explicit configured Pa/full-scale")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(int(config.get("torch_threads", 2)))
    system, receivers, provenance = build_system(config, args.config.resolve().parent)
    signature = source_signature(config, provenance)
    if args.resume:
        load_checkpoint(args.resume, system, signature)
    at = trajectory(config)
    steps = args.steps
    if steps is None:
        duration = float(config.get("duration_s", 1e-4))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration_s must be finite and positive")
        steps = int(math.ceil(duration / system.dt))
    if steps < 1:
        raise ValueError("steps must be positive")
    pre_roll = config.get("pre_roll_steps", 0)
    if not isinstance(pre_roll, int) or isinstance(pre_roll, bool) or pre_roll < 0:
        raise ValueError("pre_roll_steps must be a nonnegative integer")
    if not args.resume:
        for _ in range(pre_roll):
            control, muscle = at(0)
            system.advance(control=control, muscle_pressure_pa=muscle)
    args.out.mkdir(parents=True, exist_ok=True)
    records = {}

    def record(name, value):
        records.setdefault(name, []).append(np.asarray(value).copy())

    start_time = system.time_s
    for _ in range(steps):
        control, muscle = at(max(0.0, system.time_s + system.dt - pre_roll * system.dt))
        result = system.advance(control=control, muscle_pressure_pa=muscle)
        record("time_s", system.time_s)
        record("pressure_pa", system.downstream.sample_pressure_pa(receivers))
        for f in fields(result):
            if f.name != "time_s":
                record(f.name, getattr(result, f.name))
        record("solid_positions_m", system.solid.positions_m)
        record("solid_velocities_m_s", system.solid.velocities_m_s)
        gas = system.channel.snapshot()
        record("channel_mass_kg", gas.mass_kg)
        record("channel_momentum_kg_m_s", gas.momentum_kg_m_s)
        record("channel_energy_j", gas.energy_j)
        record("channel_storage_volume_m3", gas.geometry.storage_volume_m3)
        record("channel_face_gap_m", gas.geometry.face_gap_m)
        record("channel_open_area_m2", gas.geometry.open_area_m2)
        record("lung_volume_m3", system.lung.volume_m3)
        record("lung_mass_kg", system.lung.mass_kg)
        solid = system.solid.diagnostics()
        record("solid_contact_energy_j", solid.contact_j)
        record("solid_kinetic_energy_j", solid.kinetic_j)
        record("solid_strain_energy_j", solid.strain_j)
        record("solid_minimum_jacobian", solid.minimum_jacobian)
        record("acoustic_staggered_energy_j", system.downstream.diagnostics()["staggered_energy_j"])
    output = {k: np.asarray(v) for k, v in records.items()}
    metadata = {
        "schema_version": 1, "config": config, "signature": signature,
        "dt_s": system.dt, "start_time_s": start_time, "end_time_s": system.time_s,
        "control_time_origin_s": pre_roll * system.dt,
        "receivers_m": receivers, "provenance": provenance,
        "channel": system.channel.diagnostics(), "lung": system.lung.diagnostics(),
        "downstream": system.downstream.diagnostics(),
        "upstream": None if system.upstream is None else system.upstream.diagnostics(),
        "supply": None if system.supply is None else system.supply.diagnostics(),
        "solid": system.solid.diagnostics(),
        "oscillation_assessment": (
            "No sustained-phonation or human-identity certificate is inferred from a rendered transient."
        ),
    }
    np.savez_compressed(args.out / "forward.npz", **output)
    save_checkpoint(args.out / "checkpoint.npz", system, signature)
    if args.wav:
        scale = float(config["wav"]["pa_per_full_scale"])
        rate = int(config["wav"].get("sample_rate_hz", 48000))
        if not math.isfinite(scale) or scale <= 0 or rate <= 0:
            raise ValueError("WAV requires positive explicit Pa/full-scale and sample rate")
        ratio = Fraction(rate * system.dt).limit_denominator(1_000_000)
        samples = resample_poly(output["pressure_pa"], ratio.numerator, ratio.denominator, axis=0)
        samples = samples / scale
        if np.max(np.abs(samples)) > 1:
            raise ValueError("declared WAV pressure scale would overrange; raw Pa files are preserved")
        sf.write(args.out / "receiver.wav", samples, rate, subtype="FLOAT")
        metadata["wav_export"] = {
            "pa_per_full_scale": scale, "sample_rate_hz": rate,
            "resampling": "explicit polyphase anti-alias conversion; raw Pa unchanged",
            "rate_ratio": [ratio.numerator, ratio.denominator],
        }
    (args.out / "metadata.json").write_text(
        json.dumps(json_value(metadata), ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.out), "steps": steps, "dt_s": system.dt,
        "peak_receiver_pa": float(np.max(np.abs(output["pressure_pa"]))),
        "maximum_interface_work_residual_j": float(np.max(output["interface_work_residual_j"])),
        "assessment": "forward reference calculation, not validated human phonation",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
