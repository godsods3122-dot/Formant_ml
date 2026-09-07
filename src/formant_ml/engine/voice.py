"""VoiceEngine — 스크립트(ms 단위 물리 factor) -> 오디오. 오프라인과 스트리밍이 같은 코드다.

    eng = VoiceEngine()                                  # 여성 화자 기본값, 44.1 kHz
    track = track_from_keyframes([...])                  # 또는 phones.syllables("아라")
    y = eng.render(track)                                # (N,) numpy
    for chunk in eng.stream(track, chunk_ms=20): ...     # 상태를 이어 가며 청크 출력

파이프라인 (사용자 명세 순서):

    GlottalSource ─ du ─────────────────────────────┐
    FricationNoise ─ 마찰 소스 (협착 레이놀즈) ─────┤
    AspirationNoise ─ 기식 (성문 포락선 × 색) ──────┼─> VocalTract (포먼트+올패스 -> 비강 -> 측지) ─> ResidualCorrector -> y
    TransientTemplateBank ─ 이벤트 과도음 ──────────┘

상태(스트리밍): 성문 위상, 모든 biquad 의 zi, 프레임 카운터. 청크 경계는 존재하지 않는 것과
같다 — 필터가 샘플마다 이어지므로 오프라인 결과와 부동소수 오차 안에서 같다(테스트).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .control import INDEX, PARAM_NAMES, ControlTrack, frames_to_samples
from .glottis import GlottalSource
from .noise import AspirationNoise, FricationNoise, TransientTemplateBank
from .profile import SpeakerProfile
from .residual import ResidualCorrector
from .rng import NoiseBank
from .tract import VocalTract


def _dilate(x: torch.Tensor, k: int) -> torch.Tensor:
    """1 차원 최대 팽창. 마찰 게이트의 경계를 k 샘플만큼 넓힌다."""
    if k < 2:
        return x
    return torch.nn.functional.max_pool1d(x.unsqueeze(1), 2 * k + 1, 1, k).squeeze(1)


@dataclass
class EngineConfig:
    sample_rate: int = 48000          # 1 ms 프레임 = 정확히 48 샘플 (44.1 kHz 는 44.1 이라 시간축이 어긋난다)
    frame_ms: float = 1.0
    speaker: str = "female"
    tract_length_cm: float = 14.6
    n_extra_formants: int | None = None      # None = 나이퀴스트까지 전부 (고차 극 보정)
    residual: bool = True
    seed: int = 0

    @property
    def hop(self) -> int:
        return int(round(self.sample_rate * self.frame_ms / 1000.0))


class VoiceEngine(nn.Module):
    def __init__(self, cfg: EngineConfig | None = None, profile: SpeakerProfile | None = None):
        super().__init__()
        self.cfg = cfg or EngineConfig()
        self.profile = profile
        fs, hop = self.cfg.sample_rate, self.cfg.hop
        f0r = (profile.f0_lo, profile.f0_hi, profile.f0_nominal) if profile else None
        if profile:
            self.cfg.tract_length_cm = profile.tract_length_cm
        self.glottis = GlottalSource(fs, hop, speaker=self.cfg.speaker, f0_range=f0r)
        self.frication = FricationNoise(fs, hop)
        self.aspiration = AspirationNoise(fs, hop)
        self.transients = TransientTemplateBank(fs)
        fbw = float(profile.sibilant.get("front_bw_slope", 0.20)) if profile else 0.20
        self.tract = VocalTract(fs, hop, length_cm=self.cfg.tract_length_cm,
                                n_extra=self.cfg.n_extra_formants,
                                front_bw_slope=fbw)
        self.residual = ResidualCorrector(fs, hop) if self.cfg.residual else None
        self.reset()

    # ------------------------------------------------------------ 상태
    def reset(self) -> None:
        self.state = dict(phase=None, amp=None, tract={}, residual={}, glottis={},
                          fric={}, asp={}, frame=0)
        self.noise = NoiseBank(self.cfg.seed)

    # ------------------------------------------------------------ 핵심
    def forward(self, ctrl: torch.Tensor, events: list[dict] | None = None,
                t_offset_s: float = 0.0, state: dict | None = None,
                emit: int | None = None) -> dict:
        """ctrl: (B, T, P) 프레임률 제어 텐서. events: 과도음 이벤트(절대 시각).

        emit: 내보낼 프레임 수. 스트리밍은 T+1 프레임을 주고 emit=T 로 불러 마지막
        프레임의 샘플 보간이 다음 프레임을 보게 한다(선행 1 프레임 = 1 ms 지연).
        """
        st = state if state is not None else self.state
        hop = self.cfg.hop
        b, t_all, _ = ctrl.shape
        t = t_all if emit is None else int(emit)
        n = t * hop
        c = {name: ctrl[..., INDEX[name]] for name in PARAM_NAMES}
        f0i, s0 = st["frame"], st["frame"] * hop
        g = self.glottis(c, phase0=st["phase"], noise=self.noise, frame0=f0i,
                         amp0=st["amp"], state=st["glottis"], emit=t)
        fr = self.frication(c, g["ag_dc_frames"], g["phase"], g["voiced"], noise=self.noise,
                            frame0=f0i, state=st["fric"], emit=t)
        asp = self.aspiration(g["asp_env"], noise=self.noise, sample0=s0, state=st["asp"])
        ev = [dict(e, t=e["t"] - t_offset_s) for e in (events or [])
              if -0.1 <= e["t"] - t_offset_s < n / self.cfg.sample_rate]
        tr = self.transients.render(ev, n, b, noise=self.noise, device=ctrl.device, sample0=s0) \
            if ev else torch.zeros(b, n, device=ctrl.device)
        out = self.tract(g["du"], fr["source"], asp["source"], tr, c, state=st["tract"])
        y = out["audio"]
        if self.residual is not None:
            heads = self.residual(ctrl, state=st["residual"], emit=t)
            # **마찰 구간에서는 잔차를 끈다.** 사용자 규칙: 치찰음은 학습에 맡기지 말고
            # 직접 구현할 것. 잔차 EQ 는 전 대역·전 구간에 걸리므로, 막지 않으면 치찰음
            # 물리가 틀렸을 때 신경망이 ±6 dB EQ 로 덮어 버린다. 그러면 (1) 물리는 계속
            # 틀린 채로 남고 (2) 학습 데이터에 없는 새 발화에서 치찰음이 무너진다.
            # 기준은 레이놀즈 게이트를 지난 **실제 마찰 유량**이다 — 제어값이 아니라
            # 물리량이라 "마찰이 실제로 일어나는 곳" 과 정확히 일치한다.
            fr_on = (fr["source"].detach().abs() > 0).to(y.dtype)
            fr_on = _dilate(fr_on, hop)                     # 경계에서 새지 않게 넓힌다
            mix = frames_to_samples(c["residual_mix"].unsqueeze(-1), hop)[..., 0][:, :n]
            mix = mix * (1.0 - fr_on[:, :n])
            r = self.residual.apply(y, heads, mix, state=st["residual"])
            y, st["residual"] = r["audio"], r["state"]
        st["phase"] = g["phase_last"]
        st["amp"] = g["amp_last"]
        st["glottis"], st["fric"], st["asp"] = g["state"], fr["state"], asp["state"]
        st["tract"] = out["state"]
        st["frame"] += t
        return dict(audio=y, du=g["du"], fric=fr["source"], asp=asp["source"], transient=tr,
                    glottal_path=out["glottal_path"], front_path=out["front_path"],
                    f0=g["f0"], amp=g["amp"], phase=g["phase"], reynolds=fr["reynolds"],
                    state=st)

    # ------------------------------------------------------------ 편의 API
    @torch.no_grad()
    def render(self, track: ControlTrack, return_parts: bool = False):
        self.reset()
        ctrl = track.to_tensor()
        out = self.forward(ctrl, track.events, 0.0)
        y = out["audio"][0].numpy()
        return (y, {k: v[0].numpy() for k, v in out.items()
                    if torch.is_tensor(v) and v.dim() == 2}) if return_parts else y

    @torch.no_grad()
    def stream(self, track: ControlTrack, chunk_ms: float = 20.0):
        """청크 단위 생성기. 상태는 self.state 에 이어진다."""
        self.reset()
        ctrl = track.to_tensor()
        step = max(1, int(round(chunk_ms / self.cfg.frame_ms)))
        T = ctrl.shape[1]
        for i in range(0, T, step):
            j = min(T, i + step)
            t0 = i * self.cfg.hop / self.cfg.sample_rate      # 프레임 시각은 샘플 기준으로
            out = self.forward(ctrl[:, i:min(T, j + 1)], track.events, t0, emit=j - i)
            yield out["audio"][0].numpy()

    def latency_ms(self) -> float:
        return self.cfg.frame_ms      # 1 프레임: 필터는 인과이고 선행 프레임이 필요 없다
