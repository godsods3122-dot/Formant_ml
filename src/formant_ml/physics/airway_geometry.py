"""Fixed-pose, explicitly supplied oral/nasal/exterior voxel anatomy.

No scan data or personalized anatomy is supplied here. The import path is
``tract3d.build_grid(profiles, ...) -> build_airway_grid(...)``; masks must
already share that Grid's frame and extent. Grid storage remains legacy CGS.
There is no dynamic tongue, lip, or velum motion and no artificial area floor.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .tract3d import Grid, T_ID, TISSUE


def _mask(value, shape, name, *, nonempty=True):
    a = np.asarray(value)
    if a.shape != shape or a.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean mask of shape {shape}")
    if nonempty and not a.any():
        raise ValueError(f"{name} must not be empty")
    return a


def _slices(axis):
    lo, hi, mid = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
    lo[axis], hi[axis], mid[axis] = slice(None, -1), slice(1, None), slice(1, -1)
    return tuple(lo), tuple(hi), tuple(mid)


def validate_grid(g: Grid) -> None:
    """Reject malformed geometry before allocating a stateful acoustic domain."""
    if not np.isfinite(g.h) or g.h <= 0:
        raise ValueError("Grid.h must be positive and finite [cm]")
    if np.shape(g.origin) != (3,) or not np.isfinite(g.origin).all():
        raise ValueError("Grid.origin must contain three finite coordinates [cm]")
    if np.ndim(g.air) != 3 or min(g.air.shape) < 1:
        raise ValueError("Grid.air must be a nonempty 3D mask")
    _mask(g.air, g.shape, "air")
    for name in ("inlet", "exterior"):
        a = _mask(getattr(g, name), g.shape, name, nonempty=name == "inlet")
        if np.any(a & ~g.air):
            raise ValueError(f"{name} must be contained in air")
    for axis, (f, w, c) in enumerate(zip((g.ax, g.ay, g.az),
                                         (g.wx, g.wy, g.wz), (g.cx, g.cy, g.cz))):
        shape = list(g.shape)
        shape[axis] += 1
        if any(np.shape(a) != tuple(shape) for a in (f, w, c)):
            raise ValueError("Face arrays have inconsistent shapes")
        if not np.isfinite(f).all() or np.any((f < 0) | (f > 1)):
            raise ValueError("Open face fractions must be finite and in [0, 1]")
        if not np.issubdtype(w.dtype, np.integer) or np.any((w < 0) | (w >= len(TISSUE))):
            raise ValueError("Invalid wall tissue IDs")
        if not np.isfinite(c).all() or np.any(c < 0):
            raise ValueError("Wall area factors must be finite and nonnegative")
        left, right = np.zeros(shape, bool), np.zeros(shape, bool)
        sl = [slice(None)] * 3
        sl[axis] = slice(1, None)
        left[tuple(sl)] = g.air
        sl[axis] = slice(None, -1)
        right[tuple(sl)] = g.air
        if np.any((f > 0) & ~(left & right)):
            raise ValueError("Open faces must join two air cells, not solid or outside the grid")
        if np.any((w > 0) & ~(left ^ right)):
            raise ValueError("Wall faces must separate one air cell from solid")
    if g.meta.get("cut", False):
        vf = np.asarray(g.meta.get("vf"))
        if (vf.shape != g.shape or not np.isfinite(vf).all()
                or np.any((vf < 0) | (vf > 1)) or not np.array_equal(vf > 0, g.air)):
            raise ValueError("Cut-cell vf must give the physical air fraction in [0, 1]")
        walls = g.meta.get("walls", {})
        cell, area, tis = (np.asarray(walls.get(k, [])) for k in ("cell", "area", "tis"))
        if cell.ndim != 1:
            raise ValueError("Malformed cut wall cells")
        n = len(cell)
        if (cell.shape != (n,) or area.shape != (n,) or tis.shape != (n,)
                or not np.issubdtype(cell.dtype, np.integer)
                or np.any((cell < 0) | (cell >= g.air.size))):
            raise ValueError("Malformed cut wall cells")
        if (not np.isfinite(area).all() or np.any(area <= 0)
                or not g.air.ravel()[cell].all()
                or not np.issubdtype(tis.dtype, np.integer)
                or np.any((tis <= 0) | (tis >= len(TISSUE)))):
            raise ValueError("Malformed cut wall areas or tissue IDs")
        for key in ("pos", "dir"):
            a = np.asarray(walls.get(key))
            if a.shape != (n, 3) or not np.isfinite(a).all():
                raise ValueError(f"Malformed cut wall {key}")


def component_labels(g: Grid, mask: np.ndarray | None = None) -> np.ndarray:
    """Connected air components using actual positive open faces, not adjacency alone."""
    mask = g.air if mask is None else _mask(mask, g.shape, "component mask")
    if np.any(mask & ~g.air):
        raise ValueError("Component mask includes solid")
    ids = np.full(g.shape, -1, dtype=np.int64)
    ids[mask] = np.arange(mask.sum())
    rows, cols = [], []
    for axis, f in enumerate((g.ax, g.ay, g.az)):
        lo, hi, mid = _slices(axis)
        keep = mask[lo] & mask[hi] & (f[mid] > 0)
        rows.append(ids[lo][keep])
        cols.append(ids[hi][keep])
    r, c = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(r)), (r, c)), shape=(mask.sum(), mask.sum())).tocsr()
    _, labels = connected_components(graph, directed=False)
    out = np.full(g.shape, -1, dtype=np.int64)
    out[mask] = labels
    return out


def build_airway_grid(oral_grid: Grid, *, left_nasal: np.ndarray,
                     right_nasal: np.ndarray, exterior: np.ndarray,
                     velopharyngeal: np.ndarray, solid_tissue: np.ndarray,
                     aperture_fraction: float = 1.0) -> Grid:
    """Combine supplied 3D masks in one fixed anatomical pose.

    All masks use ``oral_grid`` coordinates. Oral air excludes its old exterior,
    which is replaced by ``exterior``. Four supplied regions and oral air must be
    disjoint. Each branch must meet both the VP connector and common exterior;
    oral/nasal contacts outside the connector are rejected. The connector must
    not meet exterior. ``solid_tissue`` supplies actual labels for new solid
    boundaries (no inferred nasal histology); existing oral labels/areas persist.

    ``aperture_fraction`` scales only oral-facing connector open-face area.
    Zero removes the entire connector from air, exactly isolating the internal
    nasal path; nasal and oral mouths can still communicate through exterior.
    This is a fixed aperture parameter, not simulated velar tissue motion.
    Voxelized new surfaces use staircase area h**2, not invented subvoxel normals.
    Cut-cell oral imports are rejected rather than discarding their geometry.
    """
    validate_grid(oral_grid)
    if oral_grid.meta.get("cut", False):
        raise ValueError("Airway mask assembly needs a voxel oral Grid, not cut-cell metadata")
    if not np.isfinite(aperture_fraction) or not 0 <= aperture_fraction <= 1:
        raise ValueError("aperture_fraction must be finite and in [0, 1]")
    oral = oral_grid.air & ~oral_grid.exterior
    regions = [oral] + [_mask(v, oral_grid.shape, n) for v, n in (
        (left_nasal, "left_nasal"), (right_nasal, "right_nasal"),
        (exterior, "exterior"), (velopharyngeal, "velopharyngeal"))]
    if not oral.any() or np.any(sum(a.astype(int) for a in regions) > 1):
        raise ValueError("Oral, nasal, exterior and VP regions must be nonempty and disjoint")
    if np.any(oral_grid.inlet & ~oral):
        raise ValueError("Oral inlet must lie inside oral air")
    tissue = np.asarray(solid_tissue)
    if (tissue.shape != oral_grid.shape or not np.issubdtype(tissue.dtype, np.integer)
            or np.any((tissue < 0) | (tissue >= len(TISSUE)))):
        raise ValueError("solid_tissue must contain valid tissue IDs in the grid frame")
    labels = np.zeros(oral_grid.shape, np.int8)
    for i, a in enumerate(regions, 1):
        labels[a] = i
    contacts = set()
    for axis in range(3):
        lo, hi, _ = _slices(axis)
        a, b = labels[lo], labels[hi]
        for x, y in np.unique(np.stack([a.ravel(), b.ravel()], 1), axis=0):
            if x and y and x != y:
                contacts.add(tuple(sorted((int(x), int(y)))))
    required = {(1, 4), (2, 4), (3, 4), (1, 5), (2, 5), (3, 5)}
    if not required.issubset(contacts) or contacts - required:
        raise ValueError("Malformed branch/outlet/VP connectivity or an internal bypass")
    air = np.logical_or.reduce(regions if aperture_fraction > 0 else regions[:-1])
    faces, walls, areas = [], [], []
    for axis, (oldf, oldw, oldc) in enumerate(zip(
            (oral_grid.ax, oral_grid.ay, oral_grid.az),
            (oral_grid.wx, oral_grid.wy, oral_grid.wz),
            (oral_grid.cx, oral_grid.cy, oral_grid.cz))):
        lo, hi, mid = _slices(axis)
        f = np.zeros_like(oldf, dtype=float)
        w, c = np.zeros_like(oldw), np.ones_like(oldc, dtype=float)
        both = air[lo] & air[hi]
        preserved = oral_grid.air[lo] & oral_grid.air[hi] & both
        fi = np.where(preserved, oldf[mid], both.astype(float))
        gate = ((labels[lo] == 1) & (labels[hi] == 5)) | ((labels[lo] == 5) & (labels[hi] == 1))
        fi[gate] *= aperture_fraction
        f[mid] = fi
        boundary = air[lo] ^ air[hi]
        wi = np.where(air[lo], tissue[hi], tissue[lo]).copy()
        if aperture_fraction == 0:
            wi[(labels[lo] == 5) | (labels[hi] == 5)] = T_ID["velum"]
        retain = boundary & (oldw[mid] > 0)
        wi[retain] = oldw[mid][retain]
        if np.any(boundary & (wi == 0)):
            raise ValueError("New solid boundary has no supplied tissue label")
        w[mid] = np.where(boundary, wi, 0)
        c[mid] = np.where(retain, oldc[mid], 1.0)
        if np.any(both & (oldw[mid] == T_ID["rigid"])):
            raise ValueError("New air would carve through a supplied rigid palate/teeth surface")
        # Outer box faces are closed, but their supplied tissue remains explicit.
        for face_index, cell_index in ((0, 0), (-1, -1)):
            fs, cs = [slice(None)] * 3, [slice(None)] * 3
            fs[axis], cs[axis] = face_index, cell_index
            fs, cs = tuple(fs), tuple(cs)
            w[fs] = np.where(air[cs], oldw[fs], 0)
            c[fs] = oldc[fs]
        faces.append(f)
        walls.append(w)
        areas.append(c)
    meta = dict(oral_grid.meta, fixed_pose=True, anatomy_provenance="caller-supplied voxel masks",
                vp_aperture_fraction=float(aperture_fraction),
                n_air=int(air.sum()), n_tract=int((air & ~exterior).sum()),
                vol_tract=float((air & ~exterior).sum() * oral_grid.h ** 3),
                left_nasal=left_nasal.copy(), right_nasal=right_nasal.copy(),
                velopharyngeal=velopharyngeal.copy())
    g = replace(oral_grid, air=air, inlet=oral_grid.inlet.copy(), exterior=exterior.copy(),
                ax=faces[0], ay=faces[1], az=faces[2], wx=walls[0], wy=walls[1], wz=walls[2],
                cx=areas[0], cy=areas[1], cz=areas[2], meta=meta)
    validate_grid(g)
    for name, mask in zip(("oral", "left_nasal", "right_nasal", "exterior", "VP"), regions):
        if name == "VP" and aperture_fraction == 0:
            continue
        if len(np.unique(component_labels(g, mask)[mask])) != 1:
            raise ValueError(f"{name} must be one face-connected region")
    for mask in (oral, left_nasal, right_nasal):
        combined = component_labels(g, mask | exterior)
        if len(np.unique(combined[mask | exterior])) != 1:
            raise ValueError("Each airway must have an open outlet into the common exterior")
    internal = air & ~exterior
    n_internal = len(np.unique(component_labels(g, internal)[internal]))
    if n_internal != (1 if aperture_fraction > 0 else 3):
        raise ValueError("VP closure/opening does not produce the required internal connectivity")
    return g
