"""학습 단으로 넘긴 것들 — 노출된 제어, 교사 지도학습, 성도 앵커.

    PYTHONPATH=src python3 tests/test_training.py
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from formant_ml.config import Config
from formant_ml.models.encoder import ControlEncoder
from formant_ml.models.losses import (VoiceLoss, band_energy_loss,
                                      centroid_trajectory_loss,
                                      control_supervision_loss,
                                      envelope_moment_loss, tract_anchor_loss)
from formant_ml.models.synth import PhysicalVoiceSynth

CFG = Config()
SR = CFG.audio.sample_rate


def _enc_out(seed: int = 0, t: int = 40):
    torch.manual_seed(seed)
    enc = ControlEncoder(CFG)
    mel = torch.randn(1, t, CFG.audio.n_mels) * 0.1 - 4.0
    return enc, enc(mel, torch.full((1, t), 120.0), torch.ones(1, t))


def test_encoder_emits_aspiration_and_teeth():
    """이번 세션에 만든 손잡이가 인코더 출력에 실제로 있는가.

    `aspiration_bands` 는 `Controls` 23 필드 중 유일하게 인코더가 안 내던
    필드였고, `teeth_f/bw/gain` 은 `SibilantParams` 11 필드 중 인코더 밖에
    있던 셋이다. 둘 다 이번 세션의 결론을 만드는 손잡이다.
    """
    _, c = _enc_out()
    assert c.aspiration_bands is not None, "기식 대역이 여전히 인코더 밖이다"
    assert c.aspiration_bands.shape[-1] == CFG.noise.n_bands
    for k in ("teeth_f", "teeth_bw", "teeth_gain"):
        assert getattr(c.sib, k) is not None, f"sib.{k} 가 인코더 밖이다"


def test_teeth_resonance_stays_above_the_front_cavity_pole():
    """앞니 공명은 구조적으로 앞공동 극보다 위여야 한다.

    포먼트 순서를 누적합으로 강제한 것과 같은 이유다 — 학습 중에 뒤집히면
    회복이 안 된다. 제트는 협착을 빠져나온 **뒤** 더 짧은 틈에서 울린다.
    """
    for seed in range(4):
        _, c = _enc_out(seed)
        assert float((c.sib.teeth_f - c.sib.pole_f).min()) > 0.0, \
            f"seed {seed}: 앞니 공명이 앞공동 극 아래로 내려갔다"


def test_sibilant_fields_are_not_shifted_by_position():
    """`SibilantParams` 를 위치 인자로 넘기던 버그의 회귀 시험.

    필드 순서는 (... mix, slope_lo, slope_hi, teeth_f ...) 인데 인코더가
    7번째로 `roughness` 를 넘기고 있었다. 그래서 roughness(0~1)가 스커트
    기울기 자리에, slope_hi(**음수**)가 teeth_f 자리에 들어갔다.
    """
    _, c = _enc_out()
    assert float(c.sib.slope_lo.min()) >= 0.0, "봉우리 아래 스커트가 음수다"
    assert float(c.sib.slope_hi.max()) <= 0.0, "봉우리 위 스커트가 양수다"
    assert float(c.sib.teeth_f.min()) > 1000.0, "teeth_f 에 음수/엉뚱한 값이 들어갔다"
    assert c.sib.roughness is not None and 0.0 <= float(c.sib.roughness.mean()) <= 1.0


def test_gradients_reach_the_new_heads():
    """새 손잡이가 **소리로부터** 기울기를 받는가. 못 받으면 노출한 의미가 없다."""
    enc, c = _enc_out(t=32)
    y = PhysicalVoiceSynth(CFG)(c)["audio"]
    x = torch.randn_like(y) * 0.05
    n = min(y.shape[-1], x.shape[-1])
    VoiceLoss(sample_rate=SR, hop=CFG.audio.hop_size)(
        y[..., :n], x[..., :n], c)["total"].backward()
    for name in ("head_asp", "head_sib"):
        g = getattr(enc, name).weight.grad
        assert g is not None and float(g.norm()) > 0.0, f"{name} 에 기울기가 안 온다"


def test_control_supervision_is_zero_only_for_the_same_controls():
    """교사 지도학습 항: 같은 제어면 0, 한 필드만 틀려도 0 이 아니다."""
    import dataclasses
    _, c = _enc_out()
    assert float(control_supervision_loss(c, c)) == 0.0
    off = dataclasses.replace(c, formant_freq=c.formant_freq + 300.0)
    assert float(control_supervision_loss(off, c)) > 0.0, "포먼트가 300 Hz 틀려도 0 이다"
    off = dataclasses.replace(c, aspiration_bands=c.aspiration_bands + 0.2)
    assert float(control_supervision_loss(off, c)) > 0.0, "기식이 틀려도 0 이다"


def test_script_renders_paired_audio_and_controls():
    """`score.render_with_controls` 가 (오디오, 제어) 쌍을 준다 — 교사 데이터."""
    from formant_ml.score import render_with_controls
    from formant_ml.voice import VoiceProfile
    audio, ctrl = render_with_controls(
        {"timeline": [{"type": "fricative", "name": "s", "dur": 0.25}], "seed": 3},
        VoiceProfile(), CFG)
    assert audio.shape[-1] > 0 and ctrl.formant_freq.shape[1] > 0
    assert float(control_supervision_loss(ctrl, ctrl)) == 0.0


def test_tract_anchor_has_a_dead_band():
    """상위 포먼트 앵커: 실측 ±10 % 안에서는 벌점 0, 밖에서만 벌한다.

    실측 자체가 흔들리므로 데드밴드 없이 묶으면 모델이 잡음을 따라간다.
    F1~F3 은 모음이 정하므로 `start=3` 부터만 묶는다.
    """
    import dataclasses
    _, c = _enc_out()
    anchor = [float(v) for v in c.formant_freq[0, 0]]
    assert float(tract_anchor_loss(c, anchor, start=3)) == 0.0, "정답인데 벌점이 있다"
    near = dataclasses.replace(c, formant_freq=c.formant_freq * 1.05)
    assert float(tract_anchor_loss(near, anchor, start=3)) == 0.0, "5 % 인데 벌한다"
    far = dataclasses.replace(c, formant_freq=c.formant_freq * 1.40)
    assert float(tract_anchor_loss(far, anchor, start=3)) > 0.0, "40 % 인데 안 벌한다"
    # F1~F2 만 크게 흔들면(모음이 바뀐 것) 벌점이 없어야 한다
    f = c.formant_freq.clone()
    f[..., :2] = f[..., :2] * 1.5
    vowel = dataclasses.replace(c, formant_freq=f)
    assert float(tract_anchor_loss(vowel, anchor, start=3)) == 0.0, "모음 변화를 벌한다"


def test_existing_losses_already_see_envelope_timing():
    """**반증 기록**: 포락선/무게중심 손실은 필요 없다.

    "기존 손실은 시간축 모양을 못 본다"는 가설로 `envelope_moment_loss` 와
    `centroid_trajectory_loss` 를 만들었는데, 재 보니 틀렸다. 고역이 중역보다
    40 dB 조용해도 `band_energy_loss` 는 고역 온셋에 대해 부호가 맞고 세기가
    10~35배 큰 기울기를 준다(로그대역 등가중이라 그렇다).

    이 시험은 그 반증을 고정한다 — 누가 다시 넣으려 하면 여기서 걸린다.
    """
    n_samp = SR
    t = torch.arange(n_samp) / SR
    g = torch.Generator().manual_seed(0)

    def band(lo, hi):
        X = torch.fft.rfft(torch.randn(n_samp, generator=g))
        f = torch.linspace(0, SR / 2, X.shape[-1])
        X[(f < lo) | (f >= hi)] = 0
        y = torch.fft.irfft(X, n_samp)
        return y / y.pow(2).mean().sqrt()

    mid, hi = band(2000.0, 4000.0), band(8000.0, 11000.0)
    amp = 10 ** (-40.0 / 20)                      # 고역은 40 dB 아래

    def arch(on, dur=0.4):
        u = ((t - on) / dur).clamp(0, 1)
        return (torch.sin(math.pi * u) ** 2
                * torch.sigmoid((t - on) / 2e-3)
                * torch.sigmoid((on + dur - t) / 2e-3))

    tgt = (mid * arch(torch.tensor(0.30)) + amp * hi * arch(torch.tensor(0.36)))[None]
    grads = {}
    for name, fn in (("band", lambda a, b: band_energy_loss(a, b, SR)),
                     ("env", lambda a, b: envelope_moment_loss(a, b, SR))):
        on = torch.tensor(0.30, requires_grad=True)
        pred = (mid * arch(torch.tensor(0.30)) + amp * hi * arch(on))[None]
        grads[name], = torch.autograd.grad(fn(pred, tgt), on)
    assert float(grads["band"]) < 0.0, "band 가 고역 온셋 방향을 못 가리킨다"
    assert abs(float(grads["band"])) > 5 * abs(float(grads["env"])), (
        f"band {float(grads['band']):.3f} vs env {float(grads['env']):.3f} — "
        "포락선 항이 이길 만하면 반증을 다시 검토해라")
    # 무게중심 쪽도 살아 있는지만 확인한다(측정 도구로 남긴 함수)
    assert float(centroid_trajectory_loss(tgt, tgt, SR)) == 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} 통과")
    sys.exit(1 if failed else 0)
