"""The CLI must not promote a clipped diagnostic to a confidence bound."""
import importlib.util
from pathlib import Path


def _script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "copyfit.py"
    spec = importlib.util.spec_from_file_location("copyfit_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unresolved_fidelity_is_not_a_lower_bound():
    fid = dict(fine_corr=100.0, resolved=False, trust=0.0, noise_ratio=1.2,
               floor=34.5, spectrum_match=60.0)
    text = _script().fidelity_summary(fid)
    assert "분해 불가" in text
    assert "≥" not in text and ">=" not in text
    assert "상한" not in text and "신뢰도" not in text
    assert "진단값 100.00" in text


def test_resolved_fidelity_keeps_numeric_diagnostic():
    fid = dict(fine_corr=92.0, resolved=True, trust=0.8, noise_ratio=1.0,
               floor=90.0, spectrum_match=80.0)
    text = _script().fidelity_summary(fid)
    assert "92.00 %" in text
    assert "분해 불가" not in text
