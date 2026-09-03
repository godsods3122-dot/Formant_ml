"""손실 함수.

크기 스펙트럼만 맞추면 위상은 아무렇게나 되어도 되므로(= 흔한 '메탈릭/버즈' 잡음의
원인), 여기서는 위상의 *미분량* — 순시주파수(IF)와 군지연(GD) — 도 함께 맞춘다.
위상은 순환량이므로 언랩 대신 anti-wrapping 함수를 쓴다.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..data.features import log_mel, stft


def _anti_wrap(x: torch.Tensor) -> torch.Tensor:
    """위상 차이를 (-pi, pi] 로 접어 넣는다."""
    return x - 2 * math.pi * torch.round(x / (2 * math.pi))


def spectral_convergence(x_mag, y_mag, eps=1e-7):
    return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro").clamp_min(eps)


def multi_resolution_stft_loss(x, y, ffts=(2048, 1024, 512, 256),
                               hops=(480, 240, 120, 60), eps=1e-7):
    """크기 스펙트럼: log-L1 + spectral convergence."""
    total = x.new_zeros(())
    for n_fft, hop in zip(ffts, hops):
        X = stft(x, n_fft, hop).abs().clamp_min(eps)
        Y = stft(y, n_fft, hop).abs().clamp_min(eps)
        total = total + F.l1_loss(torch.log(X), torch.log(Y)) + spectral_convergence(X, Y)
    return total / len(ffts)


def phase_derivative_loss(x, y, n_fft=1024, hop=240, weight_by_mag=True, eps=1e-7):
    """순시주파수(시간축 위상차) + 군지연(주파수축 위상차) 손실.

    크기가 작은 빈의 위상은 지각적으로 무의미하므로 크기로 가중한다.
    """
    X, Y = stft(x, n_fft, hop), stft(y, n_fft, hop)
    px, py = torch.angle(X), torch.angle(Y)
    w = (Y.abs() / Y.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(eps)) \
        if weight_by_mag else torch.ones_like(py)

    ifr = _anti_wrap((px[..., 1:] - px[..., :-1]) - (py[..., 1:] - py[..., :-1]))
    gd = _anti_wrap((px[..., 1:, :] - px[..., :-1, :])
                    - (py[..., 1:, :] - py[..., :-1, :]))
    return (ifr.abs() * w[..., 1:]).mean() + (gd.abs() * w[..., 1:, :]).mean()


def mel_loss(x, y, **kw):
    return F.l1_loss(log_mel(x, **kw), log_mel(y, **kw))


def smoothness(param: torch.Tensor, order: int = 1) -> torch.Tensor:
    """제어 파라미터의 시간 미분 페널티(떨림/삐걱임 억제)."""
    d = param
    for _ in range(order):
        d = d[:, 1:] - d[:, :-1]
    return d.pow(2).mean()


def area_smoothness(area: torch.Tensor) -> torch.Tensor:
    """성도 단면적의 공간 급변 페널티(해부학적으로 매끄러운 관)."""
    d = area[..., 1:] - area[..., :-1]
    return d.pow(2).mean()


def formant_ordering_penalty(freq: torch.Tensor, margin: float = 100.0):
    """F1 < F2 < ... 위반량. (인코더에서 구조적으로 보장하면 0 이 된다.)"""
    gap = freq[..., 1:] - freq[..., :-1]
    return F.relu(margin - gap).mean()


class VoiceLoss(torch.nn.Module):
    """전체 손실 = 크기 + 위상 + 멜 + 정규화."""

    def __init__(self, w_stft=1.0, w_phase=0.2, w_mel=1.0,
                 w_smooth=1e-3, w_area=1e-3, w_noise=2e-2):
        super().__init__()
        self.w = dict(stft=w_stft, phase=w_phase, mel=w_mel,
                      smooth=w_smooth, area=w_area, noise=w_noise)

    def forward(self, pred_audio, target_audio, controls=None,
                voicing: torch.Tensor | None = None) -> dict:
        out = {
            "stft": multi_resolution_stft_loss(pred_audio, target_audio),
            "phase": phase_derivative_loss(pred_audio, target_audio),
            "mel": mel_loss(pred_audio, target_audio),
        }
        # 유성 구간에서 노이즈 경로가 에너지를 가져가는 것을 억제한다.
        # (초기 학습에서 인코더가 '노이즈로 다 채우기'로 손실을 줄이는 붕괴 방지.
        #  실제로 600스텝 복사합성 실험에서 HNR 이 12 dB -> -5 dB 로 무너졌다.)
        noise_pen = pred_audio.new_zeros(())
        if controls is not None:
            v = voicing[..., None] if voicing is not None else 1.0
            noise_pen = (controls.noise_bands * v).mean()
        out["noise"] = noise_pen

        reg = pred_audio.new_zeros(())
        if controls is not None:
            reg = reg + smoothness(controls.formant_freq / 1000.0)
            reg = reg + smoothness(controls.noise_bands)
            if controls.area is not None:
                reg = reg + area_smoothness(controls.area)
        out["reg"] = reg
        out["total"] = (self.w["stft"] * out["stft"] + self.w["phase"] * out["phase"]
                        + self.w["mel"] * out["mel"] + self.w["smooth"] * out["reg"]
                        + self.w["noise"] * out["noise"])
        return out
