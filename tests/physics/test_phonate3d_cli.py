"""Reference CLI: checkpoint resume must replay the straight run exactly."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "examples" / "phonation3d_reference.json"


def _cli():
    spec = importlib.util.spec_from_file_location("phonate3d_cli", ROOT / "scripts" / "phonate3d.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_resume_is_bitwise_identical_to_straight_run(tmp_path, capsys):
    cli = _cli()
    cli.main(["--config", str(CONFIG), "--steps", "2", "--out", str(tmp_path / "straight")])
    cli.main(["--config", str(CONFIG), "--steps", "1", "--out", str(tmp_path / "first")])
    cli.main(["--config", str(CONFIG), "--steps", "1", "--out", str(tmp_path / "resumed"),
              "--resume", str(tmp_path / "first" / "checkpoint.npz")])
    capsys.readouterr()
    straight = np.load(tmp_path / "straight" / "forward.npz")
    first = np.load(tmp_path / "first" / "forward.npz")
    resumed = np.load(tmp_path / "resumed" / "forward.npz")
    assert set(straight.files) == set(first.files) == set(resumed.files)
    for key in straight.files:
        np.testing.assert_array_equal(
            np.concatenate((first[key], resumed[key])), straight[key], err_msg=key,
        )
    metadata = json.loads((tmp_path / "resumed" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["start_time_s"] == pytest.approx(float(first["time_s"][-1]), rel=0, abs=0)

    # A checkpoint never resumes under a different configuration.
    changed = json.loads(CONFIG.read_text(encoding="utf-8"))
    changed["controls"][-1]["muscle_pressure_pa"] = 300.0
    other = tmp_path / "changed.json"
    other.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint configuration"):
        cli.main(["--config", str(other), "--steps", "1", "--out", str(tmp_path / "bad"),
                  "--resume", str(tmp_path / "first" / "checkpoint.npz")])
