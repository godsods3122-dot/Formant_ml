"""공기음향 난류 소스 + 성대-난류 결합 검증 (문헌 기반).

  PYTHONPATH=src python3 tests/test_aeroacoustic.py

근거: Stevens(1971) 임계 레이놀즈수·압력강하 소스, Story&Titze(1995) body-cover,
Titze(1988) mucosal wave, Jackson&Shadle(2000) 성문동기 마찰음 변조.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from formant_ml import aeroacoustic as ac
from formant_ml.config import Config
from formant_ml.dsp.vocalfold import (FoldParams, simulate_body_cover,
                                      simulate_stack, glottal_flow_to_modulation,
                                      cycle_rate, mucosal_wave_delay)
from formant_ml.score import render
from formant_ml.voice import VoiceProfile

CFG = Config()
PROF = VoiceProfile()
FS = 24000


def test_turbulence_turns_on_only_above_critical_reynolds():
    """좁은 협착(빠른 기류)은 난류(마찰음), 열린 협착(모음)은 층류(무음).

    유량은 직렬 구동에서 나온 실제 값을 쓴다(모음은 구강이 열려 저항이 성문으로
    옮겨가 유량·속도가 함께 떨어진다).
    """
    Ps = torch.tensor(8000.0)
    u_tight = ac.series_flow(Ps, torch.tensor(0.12), torch.tensor(0.10))
    u_open = ac.series_flow(Ps, torch.tensor(0.05), torch.tensor(3.0))
    tight = ac.turbulence_gate(u_tight, torch.tensor(0.10))   # /s/ 협착
    openv = ac.turbulence_gate(u_open, torch.tensor(3.0))     # 모음 개방
    assert float(tight) > 0.9, f"협착인데 난류가 안 켜진다: {float(tight)}"
    assert float(openv) < 0.2, f"열렸는데 난류가 안 꺼진다: {float(openv)}"
    assert float(ac.reynolds(u_tight, torch.tensor(0.10))) > ac.RE_C


def test_source_amplitude_follows_pressure_drop():
    """소스 세기 ∝ ½ρ(U/A)² (Stevens 1971): 속도 2배면 진폭 ~4배."""
    a = torch.tensor(0.10)
    lo = ac.frication_source_amp(torch.tensor(150.0), a)
    hi = ac.frication_source_amp(torch.tensor(300.0), a)
    assert 3.5 < float(hi / lo.clamp_min(1e-9)) < 4.5, float(hi / lo)


def test_series_driver_couples_glottis_and_oral_constriction():
    """폐압 하나가 성문·구강을 직렬로 지나 한 유량을 만든다. 구강이 좁으면 마찰음,
    열리면 성문(기식) 쪽이 지배 — 발성·기식·마찰음이 같은 구동에서 결합."""
    Ps = torch.tensor(8000.0)
    # /s/: 성문 열림·구강 좁음 -> 구강 마찰 큼
    fr_s = ac.frication_source_amp(ac.series_flow(Ps, torch.tensor(0.12),
                                                  torch.tensor(0.10)), torch.tensor(0.10))
    # 모음: 구강 열림 -> 구강 마찰 거의 0
    fr_v = ac.frication_source_amp(ac.series_flow(Ps, torch.tensor(0.05),
                                                  torch.tensor(3.0)), torch.tensor(3.0))
    assert float(fr_s) > 100.0 * float(fr_v.clamp_min(1e-6))


def test_centroid_descends_from_constriction_area_alone():
    """치찰음 무게중심이 협착 면적 궤적만으로 내려간다(손 곡선 없이).

    협착이 열리며 입자속도가 떨어져 무게중심이 내려간다(Stevens 1971).
    """
    prof = PROF
    sa = {"type": "syllable", "onset": "s", "vowel": "a", "dur": 0.5,
          "aero": True, "onset_s": 0.14,
          "constriction_area": [[0, 0.13], [0.2, 0.22], [0.26, 1.2],
                                [0.34, 3.0], [1, 3.0]]}
    y = render({"timeline": [sa], "seed": 5}, prof, CFG).reshape(-1)
    f = torch.linspace(0, FS / 2, 361)

    def cen(a, b, win=720):
        cs = []
        for i in range(int(a * FS), int(b * FS) - win, win // 2):
            S = torch.fft.rfft(y[i:i + win] * torch.hann_window(win)).abs()
            m = f > 1500
            cs.append(float((f[m] * S[m] ** 2).sum() / (S[m] ** 2).sum().clamp_min(1e-9)))
        return cs
    c = cen(0.0, 0.16)
    assert c[0] - c[-1] > 800, f"무게중심이 안 내려간다: {c[0]:.0f}->{c[-1]:.0f}"


def test_body_cover_three_mass_self_oscillates():
    """Story & Titze(1995) body-cover 3질량이 자가진동한다."""
    flow, _ = simulate_body_cover(FoldParams(ps=8000.0, a01=0.02, a02=0.02),
                                  n_samples=6000, sample_rate=FS)
    assert float(flow.std()) > 1e-3, "진동하지 않는다"
    assert 60 < cycle_rate(flow, FS) < 400, cycle_rate(flow, FS)


def test_vertical_multimass_has_mucosal_wave():
    """수직 다질량은 하연이 상연을 앞선다(점막파, Titze 1988). 쓸모없지 않다."""
    _, traj = simulate_stack(FoldParams(ps=8000.0, a01=0.02, a02=0.02),
                             n_masses=5, n_samples=6000, sample_rate=FS)
    delay = mucosal_wave_delay(traj, FS)
    assert 0.1 < delay < 3.0, f"점막파 지연이 비생리적: {delay:.2f} ms"


def test_fold_flow_drives_noise_modulation():
    """성대 유량이 난류 변조 신호로 이어진다(소스-치찰음 결합)."""
    flow, _ = simulate_body_cover(FoldParams(ps=8000.0, a01=0.02, a02=0.02),
                                  n_samples=6000, sample_rate=FS)
    open_env, mod = glottal_flow_to_modulation(flow, FS, 240)
    assert open_env.shape[-1] == 1 and open_env.dim() == 3
    assert float(mod.mean()) > 0.05, "성문 맥동이 변조로 안 넘어간다"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1; print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} 통과")
    sys.exit(1 if failed else 0)
