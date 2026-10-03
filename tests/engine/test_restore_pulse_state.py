"""펄스 잠금 상태 복원 (`fit.restore_pulse_state`) — 재렌더가 적합 당시 소리를 재현하게 (MEASUREMENTS §52.249)."""
import inspect

import numpy as np
import torch

from formant_ml.engine import fit as F


class _Stub:
    def __init__(self, n=1000):
        self.pulse_phi0 = torch.zeros(1, dtype=torch.float64)
        self._pulse_phase = torch.zeros(1, n, dtype=torch.float64)


def test_restores_phase_and_offset():
    s = _Stub()
    pp = np.linspace(0.0, 50.0, 1000)
    msg = F.restore_pulse_state(s, {"pulse_phi0": np.array(-2.62), "pulse_phase": pp})
    assert abs(float(s.pulse_phi0.item()) + 2.62) < 1e-12
    assert torch.allclose(s._pulse_phase[0], torch.as_tensor(pp))
    assert "보정된 펄스 위상" in msg


def test_length_is_matched():
    s = _Stub(n=1200)
    F.restore_pulse_state(s, {"pulse_phase": np.arange(1000, dtype=float)})
    assert s._pulse_phase.shape == (1, 1200)
    assert float(s._pulse_phase[0, -1]) == 999.0            # 모자라면 끝값으로 채운다
    s2 = _Stub(n=800)
    F.restore_pulse_state(s2, {"pulse_phase": np.arange(1000, dtype=float)})
    assert s2._pulse_phase.shape == (1, 800)


def test_old_archive_warns_and_keeps_the_built_phase():
    s = _Stub()
    before = s._pulse_phase.clone()
    msg = F.restore_pulse_state(s, {"pulse_phi0": np.array(0.5)})
    assert torch.equal(s._pulse_phase, before)
    assert "저장돼 있지 않다" in msg


def test_copyfit_saves_and_restores():
    src = open("scripts/copyfit.py", encoding="utf-8").read()
    assert "pulse_phase=(fit._pulse_phase" in src
    # 되살리기는 `engine.pipeline.restore_state` 로 옮겼다 (§52.470) — copyfit 이 그것을 부르고, 그 안에서 펄스 위상이 보정보다 먼저다 (§52.295)
    assert "_pl.restore_state(fit, a.init" in src
    pl = open("src/formant_ml/engine/pipeline.py", encoding="utf-8").read()
    assert pl.index("restore_pulse_state(fit, z)") < pl.index("restore_hcorr(fit, z)")
