"""Independent tract states and nasal decay across streaming boundaries."""
import pytest
import torch

import formant_ml.engine.tract as tract_module
from formant_ml.engine.control import INDEX, PARAM_NAMES, default_vector
from formant_ml.engine.tract import VocalTract

FS, HOP = 48000, 48
PATHS = ("audio", "glottal_path", "front_path", "nasal_path")


def _ctrl(frames, **values):
    controls = torch.as_tensor(default_vector(), dtype=torch.float64).repeat(frames + 1, 1)
    for name, value in values.items():
        controls[:, INDEX[name]] = value
    return {name: controls[None, :, INDEX[name]] for name in PARAM_NAMES}


def _stream(tract, sources, controls, ends):
    state, outputs = {}, []
    start = 0
    for end in ends:
        # Emit only this chunk; the extra control frame supplies interpolation lookahead.
        output = tract(
            *(source[:, start * HOP:end * HOP] for source in sources),
            {name: value[:, start:end + 1] for name, value in controls.items()},
            state=state,
        )
        state = output["state"]
        outputs.append(output)
        start = end
    return {name: torch.cat([output[name] for output in outputs], -1) for name in PATHS}, state


def _assert_stream_matches(full, streamed, state):
    for name in PATHS:
        torch.testing.assert_close(streamed[name], full[name], atol=1e-10, rtol=1e-8)
    assert state.keys() == full["state"].keys()
    for name in state:
        torch.testing.assert_close(state[name], full["state"][name], atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("hf_fixed", [False, True])
@torch.no_grad()
def test_noise_copy_and_nasal_states_are_independent(monkeypatch, hf_fixed):
    monkeypatch.setattr(tract_module, "NOISE_V2", True)
    monkeypatch.setattr(tract_module, "HF_FIXED", hf_fixed)
    tract = VocalTract(FS, HOP)
    frames = 40
    generator = torch.Generator().manual_seed(102)
    sources = tuple(0.1 * torch.randn(1, frames * HOP, generator=generator, dtype=torch.float64)
                    for _ in range(4))
    controls = _ctrl(
        frames,
        f1=torch.linspace(450.0, 750.0, frames + 1),
        f2=torch.linspace(1600.0, 1200.0, frames + 1),
        velum=torch.linspace(0.25, 0.7, frames + 1),
        front_len=torch.linspace(1.0, 2.0, frames + 1),
        back_leak=0.3,
    )
    full = tract(*sources, controls)
    streamed, state = _stream(tract, sources, controls, (5, 13, 19, 27, frames))
    _assert_stream_matches(full, streamed, state)
    assert all(f"noise_x{i}" in state for i in range(tract.n_extra))
    assert all(f"nx{i}" in state for i in range(len(tract.nasal_extra)))


@pytest.mark.parametrize("ends", [(8, 9, 30, 87, 88, 104), (3, 14, 45, 84, 104)])
@torch.no_grad()
def test_nasal_closing_and_reopening_matches_offline(ends):
    tract = VocalTract(FS, HOP)
    frames = 104
    velum = torch.zeros(frames + 1, dtype=torch.float64)
    velum[:8], velum[88:] = 0.7, 0.7
    controls = _ctrl(frames, velum=velum, nasal_f=torch.linspace(280.0, 350.0, frames + 1))
    generator = torch.Generator().manual_seed(103)
    du = torch.zeros(1, frames * HOP, dtype=torch.float64)
    du[:, :8 * HOP] = torch.randn(1, 8 * HOP, generator=generator, dtype=torch.float64)
    zero = torch.zeros_like(du)
    sources = (du, zero, zero, zero)
    full = tract(*sources, controls)
    streamed, state = _stream(tract, sources, controls, ends)
    _assert_stream_matches(full, streamed, state)
    assert full["nasal_path"][:, 8 * HOP:9 * HOP].abs().max() > 1e-6
    assert streamed["nasal_path"][:, 87 * HOP:].abs().max() < 1e-9


@torch.no_grad()
def test_closed_nasal_states_decay_without_reopening_energy():
    tract = VocalTract(FS, HOP)
    generator = torch.Generator().manual_seed(104)
    x = torch.randn(1, 4 * HOP, generator=generator, dtype=torch.float64)
    state = {}
    tract._n_emit = x.shape[-1]
    excited = tract._nasal_branch(x, _ctrl(4, velum=1.0), state)
    before = {name: value.clone() for name, value in state.items()}

    tract._n_emit = HOP
    tail = tract._nasal_branch(torch.zeros(1, HOP, dtype=torch.float64), _ctrl(1), state)
    assert tail.abs().max() > 1e-6
    assert any(not torch.equal(state[name], before[name]) for name in state)
    assert all(torch.count_nonzero(value) > 0 for value in state.values())

    tract._n_emit = 80 * HOP
    tract._nasal_branch(torch.zeros(1, 80 * HOP, dtype=torch.float64), _ctrl(80), state)
    assert max(value.abs().max() for value in state.values()) < (
        max(value.abs().max() for value in before.values()) * 1e-8
    )
    tract._n_emit = 4 * HOP
    reopened = tract._nasal_branch(torch.zeros_like(x), _ctrl(4, velum=1.0), state)
    assert reopened.abs().max() < excited.abs().max() * 1e-9


@pytest.mark.parametrize("noise_v2", [False, True])
@torch.no_grad()
def test_fresh_closed_nasal_branch_is_silent_and_preserves_oral_output(monkeypatch, noise_v2):
    monkeypatch.setattr(tract_module, "NOISE_V2", noise_v2)
    tract = VocalTract(FS, HOP)
    frames = 4
    generator = torch.Generator().manual_seed(105)
    sources = tuple(torch.randn(1, frames * HOP, generator=generator, dtype=torch.float64)
                    for _ in range(4))
    output = tract(*sources, _ctrl(frames))
    assert torch.count_nonzero(output["nasal_path"]) == 0
    torch.testing.assert_close(output["audio"], output["glottal_path"] + output["front_path"],
                               atol=0.0, rtol=0.0)
    assert not any(name.startswith(("nb", "nx")) for name in output["state"])


def test_fresh_closed_nasal_branch_skips_unused_filters(monkeypatch):
    tract = VocalTract(FS, HOP)
    tract._n_emit = HOP
    state = {"noise_x0": torch.ones(1, 2, dtype=torch.float64)}

    def unexpected_filter(*args, **kwargs):
        pytest.fail("A fresh, closed nasal branch should not run filters")

    monkeypatch.setattr(tract_module, "tv_biquad", unexpected_filter)
    output = tract._nasal_branch(torch.zeros(1, HOP, dtype=torch.float64), _ctrl(1), state)
    assert torch.count_nonzero(output) == 0
    assert set(state) == {"noise_x0"}


@pytest.mark.parametrize("velum", [0.0, 1e-5])
def test_small_velum_keeps_the_opening_gradient(velum):
    tract = VocalTract(FS, HOP)
    frames = 4
    du = torch.ones(1, frames * HOP, dtype=torch.float64)
    zero = torch.zeros_like(du)
    controls = _ctrl(frames, velum=velum)
    controls["velum"].requires_grad_()
    output = tract(du, zero, zero, zero, controls)
    gradient, = torch.autograd.grad(output["nasal_path"].sum(), controls["velum"])
    assert torch.isfinite(gradient).all()
    if velum == 0.0:
        # The clamp's boundary derivative may be zero, but the graph must remain connected.
        assert torch.count_nonzero(output["nasal_path"]) == 0
        return
    assert gradient.abs().max() > 0.0

    epsilon = 1e-6
    with torch.no_grad():
        opened = tract(du, zero, zero, zero, _ctrl(frames, velum=velum + epsilon))
        difference = (opened["nasal_path"].sum() - output["nasal_path"].sum()) / epsilon
    torch.testing.assert_close(gradient.sum(), difference,
                               atol=1e-8, rtol=3e-6)
