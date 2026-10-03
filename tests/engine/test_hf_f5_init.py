"""F5 출발 자리 (`tract.hf_log_df_for`, `copyfit --hf-f5-hz`) — MEASUREMENTS §52.254."""
import math

from formant_ml.engine import tract as T


def _f(x, k, L):
    return (2 * k - 1) * T.C_SOUND / (4.0 * L) * math.exp(T.HF_DF_LIM * math.tanh(x / T.HF_DF_LIM))


def test_it_places_f5_where_asked_inside_the_range():
    L = 14.6
    for hz in (5000.0, 6600.0, 8000.0, 8500.0):
        x, got = T.hf_log_df_for(hz, 5, L)
        assert abs(got - hz) < 1.0, (hz, got)
        assert abs(_f(x, 5, L) - got) < 1e-6


def test_out_of_range_is_clipped_to_the_edge():
    L = 14.6
    ab = 9 * T.C_SOUND / (4.0 * L)
    x, got = T.hf_log_df_for(20000.0, 5, L)
    assert got < ab * math.exp(T.HF_DF_LIM) and got > 0.99 * ab * math.exp(T.HF_DF_LIM)
    x, got = T.hf_log_df_for(100.0, 5, L)
    assert got > ab * math.exp(-T.HF_DF_LIM)


def test_zero_is_the_uniform_tube_default():
    L = 14.6
    ab = 9 * T.C_SOUND / (4.0 * L)
    x, got = T.hf_log_df_for(ab, 5, L)
    assert abs(x) < 1e-9 and abs(got - ab) < 1e-6


