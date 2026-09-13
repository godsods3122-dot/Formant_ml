"""Sparse articulation support, independent of rendering and fitting."""
import numpy as np
import pytest

from formant_ml.engine.control import PARAMS, ControlTrack, default_vector
from formant_ml.engine.events import ControlEvents, build_control_events


EVENT_NAMES = ("f1", "f2", "f3", "f4", "a_c", "tract_gain", "front_len", "obstacle")


def _track(frames=60, frame_ms=1.0):
    track = ControlTrack(np.tile(default_vector(), (frames, 1)), frame_ms)
    for name, value in (("f1", 500), ("f2", 1500), ("f3", 2500),
                        ("f4", 3500), ("front_len", 3)):
        track[name] = value
    return track


def _raw(track, names):
    raw = np.empty((track.n_frames, len(names)), dtype=np.float64)
    for column, name in enumerate(names):
        spec = PARAMS[name]
        if spec.log:
            x = ((np.log(np.maximum(track[name], 1e-9)) - np.log(spec.lo))
                 / (np.log(spec.hi) - np.log(spec.lo)))
        else:
            x = (track[name] - spec.lo) / (spec.hi - spec.lo)
        x = np.clip(x, 1e-4, 1 - 1e-4)
        raw[:, column] = np.log(x / (1 - x))
    return raw


def _build(track, names=EVENT_NAMES, observed_raw=None, **kwargs):
    options = dict(params=EVENT_NAMES, rate_pct=0.0, min_gap_ms=12.0,
                   min_jump=0.08, fade_ms=1.0)
    options.update(kwargs)
    if observed_raw is None:
        observed_raw = _raw(track, names)
    return build_control_events(track, names, observed_raw, **options)


def _assert_empty(events, frames):
    assert isinstance(events, ControlEvents)
    assert events.basis.shape == (frames, 0)
    assert events.columns.shape == events.indices.shape == events.jumps.shape == (0,)
    assert events.basis.dtype == events.jumps.dtype == np.float64
    assert events.columns.dtype == events.indices.dtype == np.int64


def test_front_len_step_has_one_scalar_and_no_unchanged_control_columns():
    track = _track()
    track["front_len"][20:] = 1.5
    raw = _raw(track, EVENT_NAMES)
    original_values, original_raw = track.values.copy(), raw.copy()

    events = _build(track, observed_raw=raw)

    assert events.basis.shape == (60, 1)
    np.testing.assert_array_equal(events.columns, [EVENT_NAMES.index("front_len")])
    np.testing.assert_array_equal(events.indices, [20])
    expected = abs(raw[20, 6] - raw[19, 6])
    np.testing.assert_allclose(events.jumps, [expected])
    assert expected != pytest.approx(abs(np.log(1.5 / 3)))
    assert events.basis.dtype == events.jumps.dtype == np.float64
    assert events.columns.dtype == events.indices.dtype == np.int64
    np.testing.assert_array_equal(track.values, original_values)
    np.testing.assert_array_equal(raw, original_raw)


def test_formant_motion_does_not_allocate_front_len_coefficient():
    track = _track()
    track["f1"][10:] = 900
    events = _build(track, names=("front_len", "f1"))
    np.testing.assert_array_equal(events.columns, [1])
    np.testing.assert_array_equal(events.indices, [10])
    assert events.basis.shape == (60, 1)


def test_obstacle_transition_from_zero_is_supported():
    track = _track()
    track["obstacle"][15:] = 0.75
    events = _build(track, names=("obstacle",))
    np.testing.assert_array_equal(events.columns, [0])
    np.testing.assert_array_equal(events.indices, [15])
    raw = _raw(track, ("obstacle",))
    np.testing.assert_allclose(events.jumps, [raw[15, 0] - raw[14, 0]])
    assert events.jumps[0] > 0


def test_bounded_nonlog_changes_use_physical_span_even_above_zero():
    track = _track()
    track["obstacle"] = 0.01
    track["obstacle"][20:] = 0.02
    _assert_empty(_build(track, names=("obstacle",)), track.n_frames)


def test_separately_timed_controls_keep_their_own_indices():
    track = _track()
    track["f1"][5:] = 900
    track["front_len"][9:] = 1.5
    events = _build(track, names=("front_len", "f1"))
    np.testing.assert_array_equal(events.indices, [5, 9])
    np.testing.assert_array_equal(events.columns, [1, 0])
    assert events.basis.shape == (60, 2)
    np.testing.assert_array_equal(events.basis[events.indices, np.arange(2)], 0.5)


def test_opening_then_closing_splits_events_despite_minimum_gap():
    track = _track()
    track["front_len"] = 2.0
    track["front_len"][8:12] = 4.0
    events = _build(track, names=("front_len",))
    np.testing.assert_array_equal(events.indices, [8, 12])
    np.testing.assert_array_equal(events.columns, [0, 0])
    assert events.basis.shape == (60, 2)
    assert (events.jumps > 0).all()
    assert events.jumps[0] == pytest.approx(events.jumps[1])
    correction = events.basis @ (events.jumps * [1, -1])
    assert correction[10] > 0
    assert correction[-1] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("motion", ["constant", "small_ramp", "jitter"])
def test_tiny_or_constant_motion_is_filtered(motion):
    track = _track()
    if motion == "small_ramp":
        track["f1"] *= np.exp(np.linspace(0, 0.04, track.n_frames))
    elif motion == "jitter":
        track["f1"] *= np.exp(0.005 * np.sin(np.arange(track.n_frames)))
    _assert_empty(_build(track), track.n_frames)


@pytest.mark.parametrize(
    "names,params",
    [(("f1",), EVENT_NAMES), (EVENT_NAMES, ("f1",)), ((), EVENT_NAMES),
     (EVENT_NAMES, ())],
)
def test_only_fitted_and_enabled_names_have_support(names, params):
    track = _track()
    track["front_len"][20:] = 1.5
    _assert_empty(_build(track, names=names, params=params), track.n_frames)


def test_same_direction_run_uses_cumulative_change_center_and_raw_endpoints():
    track = _track()
    track["f1"][4:6] *= np.exp(0.04)
    track["f1"][6:8] *= np.exp(0.14)
    track["f1"][8:] *= np.exp(0.20)
    raw = _raw(track, ("f1",)) * 7.5 + 2.0
    events = _build(track, names=("f1",), observed_raw=raw)
    np.testing.assert_array_equal(events.indices, [6])
    np.testing.assert_allclose(events.jumps, [abs(raw[8, 0] - raw[3, 0])])


@pytest.mark.parametrize("second_frame,count", [(16, 1), (17, 0)])
def test_gap_merges_nearby_changes_but_not_distant_tiny_steps(second_frame, count):
    track = _track()
    track["f1"][4:second_frame] *= np.exp(0.05)
    track["f1"][second_frame:] *= np.exp(0.10)
    events = _build(track, names=("f1",))
    assert events.basis.shape == (60, count)


@pytest.mark.parametrize("frame_ms,count", [(1.0, 1), (2.0, 2)])
def test_merge_gap_is_measured_in_ms(frame_ms, count):
    track = _track(frame_ms=frame_ms)
    track["f1"][4:12] *= np.exp(0.1)
    track["f1"][12:] *= np.exp(0.2)
    events = _build(track, names=("f1",), min_gap_ms=12)
    assert len(events.indices) == count


def test_log_default_markers_split_runs_without_discarding_valid_motion():
    track = _track()
    track["front_len"] = 0.0
    track["front_len"][4:8] = 2.0
    track["front_len"][8:10] = 3.0
    track["front_len"][12:16] = 1.0
    track["front_len"][16:] = 2.0
    events = _build(track, names=("front_len",))
    np.testing.assert_array_equal(events.indices, [8, 16])
    raw = _raw(track, ("front_len",))
    np.testing.assert_allclose(events.jumps, [raw[8, 0] - raw[7, 0],
                                            raw[16, 0] - raw[15, 0]])


@pytest.mark.parametrize("before,after", [(0.0, 2.0), (2.0, 0.0), (-1.0, 2.0)])
def test_invalid_adjacent_log_pairs_do_not_create_events(before, after):
    track = _track()
    track["front_len"] = before
    track["front_len"][20:] = after
    _assert_empty(_build(track, names=("front_len",)), track.n_frames)


@pytest.mark.parametrize("rate_pct,count", [(0.0, 1), (9.0, 1), (11.0, 0)])
def test_rate_cutoff_is_per_control_percent_per_ms(rate_pct, count):
    track = _track(frame_ms=2)
    track["front_len"][20:] *= np.exp(0.2)
    events = _build(track, names=("front_len",), rate_pct=rate_pct)
    assert len(events.indices) == count


def test_subthreshold_reversal_still_splits_runs():
    track = _track()
    track["obstacle"][5:7] = 0.2
    track["obstacle"][7:9] = 0.19
    track["obstacle"][9:] = 0.5
    events = _build(track, names=("obstacle",), rate_pct=5)
    np.testing.assert_array_equal(events.indices, [5, 9])


def test_zero_raw_excursion_does_not_allocate_an_unused_scalar():
    track = _track()
    track["front_len"][20:] = 1.5
    raw = np.zeros((track.n_frames, 1))
    _assert_empty(_build(track, names=("front_len",), observed_raw=raw), track.n_frames)


@pytest.mark.parametrize("frames", [0, 1, 2, 3])
def test_empty_and_short_tracks_return_shaped_empty_arrays(frames):
    track = _track(frames)
    track["front_len"][1:] = 1.5
    _assert_empty(_build(track), frames)


@pytest.mark.parametrize("fade_ms", [0.0, 1.0, 8.0])
def test_logistic_basis_is_finite_smooth_and_centered(fade_ms):
    track = _track(3000)
    track["front_len"][1500:] = 1.5
    with np.errstate(over="raise", invalid="raise"):
        events = _build(track, names=("front_len",), fade_ms=fade_ms)
    basis = events.basis[:, 0]
    assert np.isfinite(basis).all()
    assert ((basis >= 0) & (basis <= 1)).all()
    assert (np.diff(basis) >= 0).all()
    assert basis[1500] == 0.5
    assert 0 < basis[1499] < 0.5 < basis[1501] < 1
    scale = max(fade_ms / (2 * np.log(9)), 0.25)
    np.testing.assert_allclose(basis[1499:1502],
                               1 / (1 + np.exp(-np.arange(-1, 2) / scale)))


@pytest.mark.parametrize("field", ["rate_pct", "min_gap_ms", "min_jump", "fade_ms"])
@pytest.mark.parametrize("value", [-1, np.nan, np.inf, "bad"])
def test_invalid_configuration_raises_value_error(field, value):
    with pytest.raises(ValueError, match=field):
        _build(_track(), **{field: value})


@pytest.mark.parametrize("frame_ms", [0.0, -1.0, np.nan, np.inf, None])
def test_invalid_sample_grid_raises_value_error(frame_ms):
    with pytest.raises(ValueError, match="frame_ms"):
        _build(_track(frame_ms=frame_ms))


@pytest.mark.parametrize("field", ["names", "params"])
@pytest.mark.parametrize("bad_names", ["front_len", ("unknown",), ("f1", "f1"),
                                      (None,), None])
def test_invalid_names_raise_value_error(field, bad_names):
    track = _track()
    kwargs = dict(names=("f1",), params=EVENT_NAMES)
    kwargs[field] = bad_names
    with pytest.raises(ValueError, match=field):
        _build(track, observed_raw=np.zeros((60, 1)), **kwargs)


@pytest.mark.parametrize("raw", [np.zeros(60), np.zeros((59, 8)),
                                np.zeros((60, 9)), np.zeros((60, 8, 1))])
def test_raw_shape_must_match_frames_and_names(raw):
    with pytest.raises(ValueError, match="observed_raw"):
        _build(_track(), observed_raw=raw)


@pytest.mark.parametrize("values", [np.zeros(60), np.zeros((60, len(PARAMS) - 1)),
                                   np.zeros((60, len(PARAMS), 1))])
def test_track_shape_must_match_control_schema(values):
    with pytest.raises(ValueError, match="track.values"):
        _build(ControlTrack(values), observed_raw=np.zeros((60, 8)))


@pytest.mark.parametrize("field", ["track.values", "observed_raw"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, 1j, "bad"])
def test_nonfinite_or_nonreal_observations_are_rejected(field, bad_value):
    track = _track()
    raw = _raw(track, EVENT_NAMES)
    dtype = object if isinstance(bad_value, str) else np.result_type(bad_value)
    if field == "track.values":
        track.values = track.values.astype(dtype)
        track.values[0, 0] = bad_value
    else:
        raw = raw.astype(dtype)
        raw[0, 0] = bad_value
    with pytest.raises(ValueError, match=field):
        _build(track, observed_raw=raw)
