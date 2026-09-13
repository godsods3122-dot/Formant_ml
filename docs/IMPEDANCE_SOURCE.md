# Optional impedance-loaded source

**Default remains `EngineConfig(glottal_source="lf")`.** The opt-in source is a
Level-1, prescribed-pulse acoustic flow/load coupling, not self-oscillating
vocal-fold mechanics. Its load is a reduced positive-real input-impedance
proxy, not measured anatomy or a full tract acoustic two-port. Nothing here
establishes improved listening quality, 91/96% envelope agreement, or 1 dB
individual-harmonic error. Evaluation windows/objectives are unchanged.

## Use and persistence

```python
from formant_ml.engine import EngineConfig, VoiceEngine

engine = VoiceEngine(EngineConfig(
    residual=False, glottal_source="loaded", load_coupling=0.7))
audio, parts = engine.render(track, return_parts=True)
```

```powershell
python scripts\copyfit.py input.wav --glottal-source loaded --load-coupling 0.7 --out out\loaded
# Re-render the saved source and constants without another fitting pass.
python scripts\copyfit.py input.wav --init out\loaded --global-iters 0 --stage-iters 0 --phase-iters 0 --out out\replay
# Explicit source-only A/B: retain saved controls/constants, choose LF.
python scripts\copyfit.py input.wav --init out\loaded --glottal-source lf --global-iters 0 --stage-iters 0 --phase-iters 0 --out out\lf
```

`parametrize_corpus.py` accepts the same two flags. Their defaults are omitted:
restore saved source settings from initialization/calibration when present,
otherwise use LF with a dormant coupling value of 1. An explicit option
overrides that value; conflicting saved settings otherwise raise an error.
Changing coupling alone does not select the loaded model.

The existing `acoustic_constants.model` signature records `glottal_source`,
`load_coupling`, and `loaded_source_version`. Existing speaker/recording/
utterance fields and archive formats are unchanged. Direct API users construct
the engine with `calibration.source_config((saved_constants,))` before
`apply_calibration`. Deliberate API A/B uses `allow_source_override=True`;
this relaxes only source-model fields, not sample-rate, tract-size, parameter
shape, or other model checks. Source settings are instance configuration, not
mutable module-global switches or fitted parameters.

The active backend requires the already-declared `realtime` extra (numba),
supports CPU and first-order gradients, and fails explicitly without it.
There is no giant per-sample Torch-graph fallback, GPU implementation, or
second-order adjoint. `OPEN_DAMP` cannot be combined with active loading:
that would introduce another model of glottal loading in the output filter.
The already non-streaming `HJIT_CYCLE` phase-noise option is also rejected.

## Source variables and units

The existing LF output `du` is a **normalized flow derivative**, not an area
track or a pressure measurement. Active synthesis forms its zero-mean
periodic Fourier primitive using the same harmonic coefficients, tilt,
masking, phase, amplitude, and optional source EQ. The Rd table's uncentered
integrated pulse maximum normalizes that primitive. A fixed nominal pulse
excursion of **100 cm^3/s** supplies a dimensional reference scale. This is an
empirical operating-regime calibration, not identified anatomical flow and
not an added output gain; the same scale is divided out of the load-induced
flow correction when returning to engine units.
The integrated LF waveform is never used as glottal area.

The actual state `q` is AC flow [cm^3/s] about the existing quasi-static
orifice mean `Udc`. Area remains `ag_dc` [cm^2] from existing adduction
physiology. With `rho=1.14e-3 g/cm^3` and pressure in dyn/cm^2:

```text
Udc = Ag * sqrt(2 * Ps / rho)          Ps = p_sub * 980.665 dyn/cm^2
M   = rho * 0.3 cm / Ag               glottal inertance
K   = rho / (2 * Ag^2)
D(q) = Rg*q + K*((Udc+q)*abs(Udc+q) - Udc*abs(Udc))
M*dq/dt + D(q) + coupling*p_load = p_drive
```

`Rg=1 dyn*s/cm^5` is a small fixed positive viscous resistance. The incremental
Bernoulli law is monotone, permits signed reverse flow, and satisfies
`q*D(q) >= 0`. No clipping, post-hoc gain compensation, or harmonic gains
stabilize the solve. The reference pulse determines equivalent driving
pressure by the **discrete inverse** of the same inertance/resistance law.
It is not a measured subglottal pressure waveform; no fold displacement,
contact mechanics, or pressure-driven oscillation timing is inferred.
The mean is quasi-static: its derivative is not another inertial state.

Diagnostics returned in `parts`:

| Name | Meaning |
|---|---|
| `flow` | Solved AC flow q, cm^3/s |
| `glottal_flow` | Udc + q, cm^3/s |
| `reference_flow` | Prescribed zero-mean LF reference flow, cm^3/s |
| `load_pressure` | Coupling-scaled midpoint pressure actually fed back, dyn/cm^2 |
| `source_rate` | Actual pulse-phase rate used for correction normalization, cycles/s |
| `du` | Analytic legacy LF derivative plus differentiated load-flow deviation |

## Load impedance and feedback

Use the first three **existing prepared, smoothed** formant/BW tracks. Tract
preparation runs once per emitted chunk; the load and audio path consume
those same tensors. No new Q range or BW estimator is introduced.

```text
omega_j = 2*pi*F_j
gamma_j = 2*pi*BW_j
H = 2*rho*c^2 / (Atract*Ltract)        Atract = 3 cm^2
R0 = 0.02*rho*c/Atract
Z(s) = R0 + sum_j H*s / (s^2 + gamma_j*s + omega_j^2)

dp_j/dt = -gamma_j*p_j - omega_j*z_j + H*q
dz_j/dt = omega_j*p_j
p_load = R0*q + sum_j p_j
```

The positive residues follow the low-mode uniform open-tube impedance
expansion; frequencies and damping use empirical formant tracks. Each modal
term is positive real for positive BW, with reactive sign reversal across
its resonance. The fixed area, residual resistance, truncation to three
modes, and use of transfer-formant frequencies as impedance-pole proxies
are modeling approximations.

These states are **not** the audio biquad states: the normalized output
formant cascade does not expose a physical input-pressure port. The loaded
source feeds the existing forward tract once. Modal pressure is returned to the
source solve, not cascaded or added onto output audio. There are no duplicate
audio poles, arbitrary feedback delay, or restored constriction multiplier.
Aspiration/frication and their mean-flow physiology remain unchanged. Nasal,
lateral, radiation, and high-mode input impedances are not fully modeled;
the forward output tract still renders its existing branches.

## Implicit integration, energy, and gradients

All dynamic equations use implicit midpoint at the audio sample rate.
`v=(q_new+q_old)/2` and `h=dt/2`. Eliminating each mode gives:

```text
a_j = 1 / (1 + h*gamma_j + (h*omega_j)^2)
b_j = -h*omega_j*a_j
zeta_j = h*H*a_j
p_mid_j = a_j*p_old_j + b_j*z_old_j + zeta_j*v

p_drive = M*(qref_new-qref_old)/dt + D((qref_new+qref_old)/2)
```

Substitution leaves one monotone scalar quadratic in `Udc+v`. A
cancellation-resistant closed-form root solves it; there is no iteration,
step-size-dependent explicit feedback, or arbitrary solver tolerance.
All modal and flow states advance together. The discrete modal impedance
is the bilinear image of `Z(s)`, with its usual frequency warping.

For fixed geometry and nonnegative coupling, stored energy is:

```text
E = M*q^2/2 + coupling*sum_j (p_j^2+z_j^2)/(2*H)

delta_E/dt =
    v*p_drive - v*D(v) - coupling*R0*v^2
    - coupling*sum_j gamma_j*p_mid_j^2/H
```

The dissipative terms are nonnegative; unforced static energy cannot grow.
Changing area changes inertance, and prescribed controls/source can do
work, so this is not a claim of passivity under arbitrary moving geometry.
Closed/nonvoiced controls do not freeze or reset accumulated load energy:
zero reference flow leaves the states to decay through the same equations.
The current model does not simulate an exactly sealed moving glottal valve.

The compiled custom-autograd operation saves eight state scalars per sample
and runs one reverse recurrence. Its implicit derivative denominator is the
positive sum of inertance, incremental resistance and modal input terms.
It propagates gradients to reference flow, mean flow, area/inertance,
frequencies/BWs, and initial/final streaming states. Coefficient preparation
remains ordinary Torch; source/load coupling is fixed configuration.

## Bypass, timing, and cost

Default LF and **literal zero coupling** bypass new source synthesis and
load state entirely, including when the backend is absent. They reproduce
the legacy renderer exactly. Active output preserves the existing analytic
LF component and differentiates **only the load-induced flow deviation**:

```text
delta[n] = q_actual[n] - q_reference[n]
du_out[n] = du_legacy[n]
            + fs*(delta[n]-delta[n-1]) / (flow_scale[n]*source_rate[n])
```

This is a residual/defect correction atop the baseline's prescribed,
time-varying LF approximation, **not the exact full waveform derivative of
physical glottal flow**. It keeps control-modulation and numerical
differentiation artifacts of the reference out of the no-load limit.
The internal actual-flow/nonlinear-pressure equations and their power balance
remain as stated. There is no fitted EQ, compensating gain, or phase shift.
Tiny positive coupling converges to legacy output to numerical precision;
even with changing reference controls the no-load recurrence reproduces the
reference, so its output correction is zero.

For internally accumulated phase, `source_rate` is the actual F0 including
jitter. With external `pulse_phase` (including fitter `--pulse-lock`), it is
the sample difference of the **supplied unwrapped phase**, in cycles/s, not
`f0_target`. The full absolute phase trajectory is required on every forward
call, matching the engine's existing slicing contract. At chunk boundaries,
the preceding sample of that trajectory supplies the left endpoint. At the
very first sample only, where no prior phase exists, the first available
interval supplies the rate. This is consistent between streaming and
monolithic output, introduces no modulo-induced rate jumps, and requires no
lookahead beyond the existing interpolation frame. Nonfinite,
non-increasing, wrapped, or undersized trajectories fail explicitly.

Recurrent state per channel: actual q, previous reference q, and six modal
pressure/quadrature values (**eight scalars**). Existing tract smoothing
retains its own state. Only emitted samples advance these states; the current
one-frame interpolation lookahead remains, and reset clears them. No extra
lookahead or independently restarted chunk dynamics are introduced.

**Parameter delta: 0 new frame columns; 0 new trainable scalar parameters.**
The two instance settings are source selection and bounded [0,1] coupling.
The fixed three-mode count, reference flow, glottal length, neutral tract
area and resistance assumptions above are not optimization knobs.

The recurrence and first-order adjoint are O(3N), with O(8N) state history
(about 3.1 MB per mono second at 48 kHz/float64, excluding input/gradient
arrays). Computing the optional Fourier flow primitive adds arithmetic to
the existing harmonic synthesis; it does not create a per-sample Torch
recurrence graph. First use includes numba compilation.

A short warmed CPU probe (200 ms synthetic vowel, 48 kHz, two Torch threads,
three repetitions, default forward tract, no residual) measured:

| Source (default harmonic path) | Render | Forward + backward |
|---|---|---|
| LF | 61.8 ms | 234.5 ms |
| loaded | 82.4 ms | 285.5 ms |

This is not a corpus fit, quality result, or small-buffer realtime guarantee.
Parent A/B must choose adoption; the default remains off.

Focused tests cover analytic/finite-difference gradients including terminal
state, positive-real discrete frequency response, unforced static energy
decay, exact bypass and tiny-coupling convergence, dynamic-reference no-load
identity, externally pulse-locked phase-rate normalization/gradients/
streaming, and saved-mode replay/intentional source override with unchanged
initial controls.
