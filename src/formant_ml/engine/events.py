"""Observation-supported, per-control articulation steps (NumPy only)."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np

from .control import INDEX, PARAMS, ControlTrack


@dataclass(frozen=True)
class ControlEvents:
    """One basis column and one learned scalar per supported control/event pair.

    ``basis`` is float64 (T, E); ``columns`` and ``indices`` are int64 (E,).
    Columns index the supplied ``names`` and may repeat. Indices identify the
    first-changed-frame coordinate of each step's midpoint. ``jumps`` is float64
    (E,), containing positive raw-coordinate excursions, not detection scores.
    Events are ordered by midpoint frame, then control column.
    """

    basis: np.ndarray
    columns: np.ndarray
    indices: np.ndarray
    jumps: np.ndarray


def _control_names(names: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of control names")
    try:
        result = tuple(names)
    except TypeError as exc:
        raise ValueError(f"{label} must be a sequence of control names") from exc
    if any(not isinstance(name, str) or name not in PARAMS for name in result):
        raise ValueError(f"{label} contains an unknown control name")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicate control names")
    return result


def _finite_array(values: np.ndarray, label: str) -> np.ndarray:
    try:
        array = np.asarray(values)
        if np.iscomplexobj(array):
            raise ValueError("complex values are not supported")
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain real numeric values") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return array


def _nonnegative(value: float, label: str, *, positive: bool = False) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be finite and {bound}")
    return value


def build_control_events(
    track: ControlTrack,
    names: Sequence[str],
    observed_raw: np.ndarray,
    *,
    params: Sequence[str],
    rate_pct: float,
    min_gap_ms: float,
    min_jump: float,
    fade_ms: float,
) -> ControlEvents:
    """Build fixed logistic steps only where the corresponding control moves.

    ``observed_raw`` must have shape (track.n_frames, len(names)), in the
    caller's initial, unsmoothed fitting coordinates. Its run-endpoint
    excursions supply amplitude caps; this function does not reparameterize it.

    Logarithmic/strictly positive-range controls use log-relative changes.
    Other bounded controls use change divided by their PARAMS physical span,
    including legal zero values. Nonpositive log pairs break runs, so off/default
    markers neither create events nor join otherwise separate transitions.

    ``rate_pct`` is percent change per ms (zero imposes no extra rate cutoff).
    Nearby candidate changes merge within ``min_gap_ms`` only in the same
    direction, including intervening slower changes. ``min_jump`` thresholds
    their cumulative absolute detection-coordinate change. Any reversal splits
    a run, even below the rate cutoff. Midpoints are weighted by that observation
    change. ``fade_ms`` is the logistic 10--90% rise time, with a minimum scale
    of 0.25 frames. Tracks shorter than four frames have no events, matching the
    fitter's observation detector. Configuration comes from the caller.
    """
    names = _control_names(names, "names")
    enabled = set(_control_names(params, "params"))
    if not isinstance(track, ControlTrack):
        raise ValueError("track must be a ControlTrack")
    values = _finite_array(track.values, "track.values")
    if values.ndim != 2 or values.shape[1] != len(PARAMS):
        raise ValueError(f"track.values must have shape (T, {len(PARAMS)})")
    frames = values.shape[0]
    raw = _finite_array(observed_raw, "observed_raw")
    if raw.shape != (frames, len(names)):
        raise ValueError("observed_raw must have shape (T, len(names))")
    frame_ms = _nonnegative(track.frame_ms, "frame_ms", positive=True)
    rate_pct = _nonnegative(rate_pct, "rate_pct")
    min_gap_ms = _nonnegative(min_gap_ms, "min_gap_ms")
    min_jump = _nonnegative(min_jump, "min_jump")
    fade_ms = _nonnegative(fade_ms, "fade_ms")

    def empty() -> ControlEvents:
        return ControlEvents(
            np.empty((frames, 0), dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )

    if frames < 4 or not names or not enabled:
        return empty()
    gap = max(1, int(round(min(min_gap_ms / frame_ms, frames))))
    cutoff = rate_pct / 100.0 * frame_ms
    records: list[tuple[int, int, float]] = []
    for column, name in enumerate(names):
        if name not in enabled:
            continue
        spec = PARAMS[name]
        x = values[:, INDEX[name]]
        if spec.log or spec.lo > 0:
            positive = x > 0
            coordinate = np.zeros(frames, dtype=np.float64)
            coordinate[positive] = np.log(x[positive])
            valid = positive[:-1] & positive[1:]
        else:
            coordinate = x / (spec.hi - spec.lo)
            valid = np.ones(frames - 1, dtype=bool)
        with np.errstate(over="ignore", invalid="ignore"):
            change = np.diff(coordinate)
        change[~valid] = 0.0
        if not np.isfinite(change).all():
            raise ValueError(f"observation changes for {name} must be finite")

        def add_run(first: int, last: int) -> None:
            weights = np.abs(change[first - 1:last])
            with np.errstate(over="ignore", invalid="ignore"):
                cumulative = np.cumsum(weights)
            total = float(cumulative[-1])
            if not math.isfinite(total):
                raise ValueError(f"observation excursion for {name} must be finite")
            if total < min_jump:
                return
            with np.errstate(over="ignore", invalid="ignore"):
                jump = float(abs(raw[last, column] - raw[first - 1, column]))
            if not math.isfinite(jump):
                raise ValueError(f"observed_raw excursion for {name} must be finite")
            if jump == 0.0:
                return
            center = first + int(np.searchsorted(cumulative, 0.5 * total))
            records.append((center, column, jump))

        first = last = 0
        direction = 0
        for frame, delta in enumerate(change, start=1):
            sign = int(np.sign(delta))
            if first and (not valid[frame - 1] or (sign and sign != direction)):
                add_run(first, last)
                first = 0
            if not valid[frame - 1] or abs(delta) <= cutoff:
                continue
            if first and frame - last > gap:
                add_run(first, last)
                first = 0
            if not first:
                first = frame
                direction = sign
            last = frame
        if first:
            add_run(first, last)

    if not records:
        return empty()
    records.sort()
    indices = np.array([r[0] for r in records], dtype=np.int64)
    columns = np.array([r[1] for r in records], dtype=np.int64)
    jumps = np.array([r[2] for r in records], dtype=np.float64)
    scale = max(fade_ms / frame_ms / (2.0 * math.log(9.0)), 0.25)
    z = np.clip((np.arange(frames, dtype=np.float64)[:, None] - indices) / scale,
                -60.0, 60.0)
    basis = 1.0 / (1.0 + np.exp(-z))
    return ControlEvents(basis, columns, indices, jumps)
