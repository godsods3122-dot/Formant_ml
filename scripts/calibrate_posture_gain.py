"""`presets.POSTURE_GAIN_20` 을 다시 잰다.

    PYTHONPATH=src python3 scripts/calibrate_posture_gain.py

이 표는 손으로 고른 값이 아니라 **측정값**이다. 전극 도파관에는 방사
임피던스가 없어 자세가 좁을수록 출력이 커진다(실제 물리는 반대다). 주파수축
정규화를 두 번 시도했지만(균등 RMS, 1/f 가중) 둘 다 못 잡았다 — 소스 하모닉과
낮은 F1 의 정렬 효과라 정적 정규화로는 안 된다. 그래서 각 자세를 **표준
소스로 실제 렌더해 세기를 재고 그 역수**를 보정으로 둔다.

렌더 경로가 바뀌면 이 표도 다시 재야 한다. 안 그러면 자음/모음 세기 관계가
어긋난다 — 측정: 소스 기울기를 7.0 에서 0 으로 되돌린 뒤 표를 그대로 뒀더니
"라" 의 폐쇄 구간이 모음보다 1.4 dB **커졌다**(실측은 10 dB 작다).
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from formant_ml.config import Config, sections_for
from formant_ml.dsp.filters import antiresonator_response
from formant_ml.liquid import posture_area
from formant_ml.models.synth import Controls, PhysicalVoiceSynth
from formant_ml.presets import LIQUID_POSTURE_20, POSTURE_GAIN_20

FS = 24000
SECONDS = 0.4


def level_db(name: str, cfg: Config, syn: PhysicalVoiceSynth) -> float:
    n_sec = cfg.filt.n_tract_sections
    t = int(SECONDS * FS / cfg.audio.hop_size)
    area = posture_area(name, n_sec).reshape(1, 1, -1).expand(1, t, n_sec)
    K, nb = cfg.filt.n_formants, cfg.noise.n_bands
    c = Controls(
        f0=torch.full((1, t, 1), 200.0), harmonic_amp=torch.ones(1, t, 1),
        rd=torch.full((1, t, 1), 1.1),
        formant_freq=torch.linspace(500, 6000, K).reshape(1, 1, -1)
                          .expand(1, t, K).contiguous(),
        formant_bw=torch.full((1, t, K), 90.0), formant_gain=torch.ones(1, t, K),
        noise_bands=torch.zeros(1, t, nb), noise_entry=torch.zeros(1, t, 1),
        noise_am=torch.zeros(1, t, 1), tilt=torch.zeros(1, t, 1),
        area=area.contiguous())
    # 유음 자세는 영점이 세기의 일부다 — 렌더 경로와 같게 걸고 잰다.
    p = LIQUID_POSTURE_20.get(name)
    if p is not None:
        c.antiformant_freq = torch.tensor(p["zero_hz"]).reshape(1, 1, -1) \
            .expand(1, t, 2).contiguous()
        c.antiformant_bw = torch.tensor(p["zero_bw"]).reshape(1, 1, -1) \
            .expand(1, t, 2).contiguous()
    with torch.no_grad():
        y = syn(c)["audio"][0, FS // 10:]
    return 20.0 * math.log10(float(y.pow(2).mean().sqrt()) + 1e-12)


def main() -> None:
    cfg = Config()
    cfg.filt.n_tract_sections = sections_for(FS, 14.6)
    syn = PhysicalVoiceSynth(cfg, tract_mode="waveguide")
    ref = level_db("a", cfg, syn)
    print("POSTURE_GAIN_20 = {  # /a/ 를 0 dB 기준으로 잰 값의 역수")
    for name in POSTURE_GAIN_20:
        d = level_db(name, cfg, syn) - ref
        print(f'    "{name}": {10 ** (-d / 20.0):.3f},'
              f"   # 렌더 세기 {d:+.1f} dB")
    print("}")


if __name__ == "__main__":
    main()
