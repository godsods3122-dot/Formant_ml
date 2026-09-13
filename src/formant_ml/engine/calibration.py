"""Reusable acoustic calibration, separate from utterance control tracks.

These are empirical model constants, not independently measured anatomy.
Source aperiodicity and level normalization must not become speaker constants.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

SCOPES = ("speaker", "recording", "utterance")
# Public storage name: (module, parameter, scope, legacy archive field).
FIELDS = {
    "log_extra_bw": ("tract", "log_extra_bw", "speaker", "engine_params"),
    "log_front_bw": ("tract", "log_front_bw", "speaker", "engine_params"),
    "hf_log_df": ("tract", "hf_log_df", "speaker", "hf_params"),
    "hf_log_bw": ("tract", "hf_log_bw", "speaker", "hf_params"),
    "pir_log_f": ("tract", "pir_log_f", "speaker", "hf_params"),
    "pir_depth": ("tract", "pir_depth", "speaker", "hf_params"),
    "fric_log_lp_ratio": ("frication", "log_lp_ratio", "speaker", "hf_params"),
    "open_damp": ("tract", "open_damp", "speaker", "hf_params"),
    "open_f1": ("tract", "open_f1", "speaker", "hf_params"),
    "hf_eq_db": ("tract", "hf_eq_db", "recording", "hf_params"),
    "log_mvf": ("aspiration", "log_mvf", "utterance", "asp_params"),
    "glottis_hjit_log": ("glottis", "hjit_log", "utterance", "hf_params"),
    "glottis_src_eq_db": ("glottis", "src_eq_db", "utterance", "hf_params"),
}
EXTRA_FIELDS = {"room_ir": "recording", "hf_eq_enabled": "recording", "gain_db": "utterance",
                "pulse_phi0": "utterance"}
SOURCE_MODEL_FIELDS = ("glottal_source", "load_coupling", "loaded_source_version")


def source_config(calibrations=(), *, glottal_source=None, load_coupling=None) -> dict:
    """Resolve saved source settings before engine creation, or explicit A/B choices.

    Missing legacy fields mean the legacy source. Explicit options override only
    their own values; conflicting saved values otherwise require a decision.
    """
    from .voice import EngineConfig, LOADED_SOURCE_VERSION

    models = [normalize_calibration(item)["model"] for item in calibrations if item is not None]
    result = {}
    for name, explicit, default in (("glottal_source", glottal_source, "lf"),
                                    ("load_coupling", load_coupling, 1.0)):
        saved = [m[name] for m in models if name in m]
        if explicit is not None:
            result[name] = explicit
        elif saved:
            if any(value != saved[0] for value in saved):
                raise ValueError(f"Conflicting saved {name}; select it explicitly for A/B")
            result[name] = saved[0]
        else:
            result[name] = default
    if glottal_source is None:
        for model in models:
            if model.get("loaded_source_version", LOADED_SOURCE_VERSION) != LOADED_SOURCE_VERSION:
                raise ValueError("Unsupported loaded source version; select a source explicitly")
    EngineConfig(**result)  # Validate even when a caller is only inspecting settings.
    return result


def parameter_scope(name: str) -> str:
    return FIELDS[name][2] if name in FIELDS else EXTRA_FIELDS[name]


def engine_parameters(engine) -> dict[str, torch.nn.Parameter]:
    return {name: getattr(getattr(engine, module), attr)
            for name, (module, attr, _, _) in FIELDS.items()}


def model_signature(engine) -> dict:
    from .voice import LOADED_SOURCE_VERSION
    profile = engine.profile
    return dict(sample_rate=engine.cfg.sample_rate,
                tract_length_cm=float(engine.tract.length_cm),
                n_formants=engine.tract.K, n_extra_formants=engine.tract.n_extra,
                speaker=engine.cfg.speaker,
                f0_range=None if profile is None else
                [profile.f0_lo, profile.f0_hi, profile.f0_nominal],
                noise_modulation=engine.cfg.noise_modulation,
                glottal_source=engine.cfg.glottal_source,
                load_coupling=engine.cfg.load_coupling,
                loaded_source_version=LOADED_SOURCE_VERSION)


def normalize_calibration(data: Mapping) -> dict:
    """Read versioned calibration or a legacy flat speaker-lock dictionary."""
    if not isinstance(data, Mapping):
        raise ValueError("Calibration must be a JSON object")
    out = dict(schema_version=1, model={}, **{scope: {} for scope in SCOPES})
    if "schema_version" in data:
        if data["schema_version"] != 1:
            raise ValueError(f"Unsupported calibration schema: {data['schema_version']}")
        allowed = {"schema_version", "model", "sources", "formant_band", "q_band", *SCOPES}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown calibration fields: {sorted(unknown)}")
        if not isinstance(data.get("model", {}), Mapping):
            raise ValueError("Calibration model must be an object")
        out["model"] = dict(data.get("model", {}))
        for scope in SCOPES:
            values = data.get(scope, {})
            if not isinstance(values, Mapping):
                raise ValueError(f"Calibration {scope} must be an object")
            out[scope] = dict(values)
    else:
        for name, value in data.items():
            if name in ("formant_band", "q_band"):
                continue
            if name not in FIELDS and name not in EXTRA_FIELDS:
                raise ValueError(f"Unknown legacy calibration parameter: {name}")
            out[parameter_scope(name)][name] = value
    for scope in SCOPES:
        for name, value in out[scope].items():
            if name not in FIELDS and name not in EXTRA_FIELDS:
                raise ValueError(f"Unknown calibration parameter: {name}")
            if parameter_scope(name) != scope:
                raise ValueError(f"{name} belongs to {parameter_scope(name)}, not {scope}")
            if name == "hf_eq_enabled":
                if not isinstance(value, (bool, np.bool_)):
                    raise ValueError("hf_eq_enabled must be boolean")
                out[scope][name] = bool(value)
                continue
            a = np.asarray(value, dtype=np.float64)
            if a.ndim > 1 or a.size == 0 or not np.isfinite(a).all():
                raise ValueError(f"{name} must contain finite scalar/vector values")
            if name in ("gain_db", "pulse_phi0") and a.size != 1:
                raise ValueError(f"{name} must be scalar")
            if name == "room_ir" and a.ndim != 1:
                raise ValueError("room_ir must be a nonempty one-dimensional impulse response")
            scalar = a.ndim == 0 or name in ("gain_db", "pulse_phi0")
            out[scope][name] = float(a.reshape(-1)[0]) if scalar else a.tolist()
    # Preserve legacy constraints as metadata; speaker locking does not enable Q bands.
    for field in ("formant_band", "q_band"):
        if field not in data:
            continue
        bands = data[field]
        if not isinstance(bands, Mapping):
            raise ValueError(f"{field} must be an object")
        for name, limits in bands.items():
            a = np.asarray(limits, dtype=np.float64)
            if (name not in {f"f{k}" for k in range(1, 9)} or a.shape != (2,)
                    or not np.isfinite(a).all() or not 0 < a[0] < a[1]):
                raise ValueError(f"Invalid {field} for {name}")
        out[field] = dict(bands)
    if "sources" in data:
        if not isinstance(data["sources"], list) or not all(isinstance(s, str) for s in data["sources"]):
            raise ValueError("Calibration sources must be a list of paths")
        out["sources"] = list(data["sources"])
    return out


def capture_calibration(engine, *, gain_db: float | None = None,
                        pulse_phi0: float | None = None,
                        room_ir: np.ndarray | None = None) -> dict:
    out = dict(schema_version=1, model=model_signature(engine),
               **{scope: {} for scope in SCOPES})
    for name, p in engine_parameters(engine).items():
        out[parameter_scope(name)][name] = p.detach().cpu().numpy().tolist()
    out["recording"]["hf_eq_enabled"] = engine.tract.output_eq_enabled
    for name, value in (("gain_db", gain_db), ("pulse_phi0", pulse_phi0),
                        ("room_ir", room_ir)):
        if value is not None:
            out[parameter_scope(name)][name] = value
    return normalize_calibration(out)


def apply_calibration(engine, data: Mapping,
                      scopes: Sequence[str] = SCOPES, *,
                      allow_source_override: bool = False) -> tuple[str, ...]:
    """Validate all selected shapes before copying anything; never truncate."""
    data = normalize_calibration(data)
    if set(scopes) - set(SCOPES):
        raise ValueError(f"Unknown calibration scopes: {scopes}")
    actual = model_signature(engine)
    for name, expected in data["model"].items():
        if allow_source_override and name in SOURCE_MODEL_FIELDS:
            continue
        if name not in actual or actual[name] != expected:
            raise ValueError(f"Calibration model mismatch for {name}: "
                             f"saved {expected!r}, engine {actual.get(name)!r}")
    params = engine_parameters(engine)
    pending = {}
    for scope in scopes:
        for name, value in data[scope].items():
            if name in EXTRA_FIELDS:
                continue
            p = params[name]
            a = torch.as_tensor(value, dtype=p.dtype, device=p.device)
            if not torch.isfinite(a).all():
                raise ValueError(f"{name} exceeds the engine parameter precision")
            if a.numel() != p.numel() or (p.ndim > 0 and a.shape != p.shape):
                raise ValueError(f"Calibration shape mismatch for {name}: "
                                 f"saved {tuple(a.shape)}, engine {tuple(p.shape)}")
            pending[name] = a.reshape(p.shape)
    with torch.no_grad():
        for name, value in pending.items():
            params[name].copy_(value)
    if "recording" in scopes and "hf_eq_enabled" in data["recording"]:
        engine.tract.recording_eq_enabled = data["recording"]["hf_eq_enabled"]
    elif "hf_eq_db" in pending:
        engine.tract.recording_eq_enabled = bool(torch.count_nonzero(params["hf_eq_db"]))
    return tuple(pending)


def load_calibration(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            if "acoustic_constants" in z:
                return normalize_calibration(json.loads(str(z["acoustic_constants"])))
            flat = {}
            for blob in ("engine_params", "hf_params", "asp_params"):
                if blob in z:
                    values = json.loads(str(z[blob]))
                    values.pop("mvf_hz", None)  # Derived display value in old archives.
                    flat.update(values)
            if "gain_db" in z:
                flat["gain_db"] = float(z["gain_db"])
            if not flat:
                raise ValueError(f"No acoustic constants in {path}")
            out = normalize_calibration(flat)
        room_path = path.with_name(path.name.removesuffix("_track.npz") + "_room.npy")
        if path.name.endswith("_track.npz") and room_path.exists():
            out["recording"]["room_ir"] = np.load(room_path, allow_pickle=False)
            out["model"]["sample_rate"] = 48000
        return normalize_calibration(out)
    with path.open(encoding="utf-8") as f:
        return normalize_calibration(json.load(f))


def legacy_fields(data: Mapping) -> dict:
    """Keep old NPZ consumers working while the scoped snapshot is authoritative."""
    data = normalize_calibration(data)
    out = {blob: {} for blob in ("engine_params", "hf_params", "asp_params")}
    for name, (_, _, scope, blob) in FIELDS.items():
        if name in data[scope]:
            out[blob][name] = data[scope][name]
    return out


def combine_calibrations(items: Sequence[Mapping], scope: str = "speaker") -> dict:
    """Pool only one shared scope, with identical model and parameter shapes."""
    if scope not in ("speaker", "recording"):
        raise ValueError("Only speaker or recording calibration can be shared")
    if not items:
        raise ValueError("At least one calibration is required")
    items = [normalize_calibration(item) for item in items]
    first = items[0]
    if not first[scope]:
        raise ValueError(f"No {scope} constants to combine")
    for item in items[1:]:
        if item["model"] != first["model"]:
            raise ValueError("Cannot combine different or missing model signatures")
        if item[scope].keys() != first[scope].keys():
            raise ValueError(f"Cannot combine incomplete {scope} parameter sets")
    out = dict(schema_version=1, model=first["model"],
               **{group: {} for group in SCOPES})
    for name in first[scope]:
        values = [np.asarray(item[scope][name], dtype=np.float64) for item in items]
        if any(value.shape != values[0].shape for value in values):
            raise ValueError(f"Cannot combine different shapes for {name}")
        if name == "room_ir" and any(not np.array_equal(value, values[0]) for value in values[1:]):
            raise ValueError("Do not average different room impulse responses; "
                             "select one recording-chain calibration")
        if name == "hf_eq_enabled":
            if any(value != values[0] for value in values[1:]):
                raise ValueError("Cannot combine enabled and disabled recording EQ")
            out[scope][name] = bool(values[0])
            continue
        out[scope][name] = np.median(np.stack(values), axis=0).tolist()
    return normalize_calibration(out)
