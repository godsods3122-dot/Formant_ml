"""잔차(residual) 보정: 물리모델이 설명하지 못하는 부분만 학습한다.

무엇을 맡기는가
---------------
우리가 이론으로 통제하는 것 — 성문 소스(F0/Rd/tilt/위상차), 성도 포먼트,
치찰음 극-영점, 난류 — 은 물리모델이 만든다. 그 밖의 것들은 방정식이 없거나
(있어도) 역추정이 안 된다.

* 혀가 입천장·치아에 닿았다 떨어질 때 나는 접촉 노이즈
* 비강 공명과 비인두 결합 (반공명 2개로는 모자란다)
* 성도-성문 상호작용 (성도 입력 임피던스가 성대 진동에 되먹임된다)
* 좌우 비대칭 기류, 침, 입술 소리 같은 잡다한 간섭

이것들을 **원음에서 물리모델 출력을 뺀 나머지**로 정의하고 신경망에 맡긴다.

붕괴를 막는 두 겹의 제약
------------------------
그냥 "잔차를 만들어라" 라고 하면 신경망이 결국 전부 만들어 버리고, 우리가
없애려던 위상 아티팩트가 그대로 돌아온다. 그래서 구조로 막는다.

1. **파형을 만들지 못한다.** 잔차는 두 가지 형태로만 나올 수 있다.
   (a) 물리 출력에 곱하는 **최소위상 LTV 필터** — 크기 스펙트럼의 세부만 고친다.
       최소위상이라 프리링잉이 없고, 위상은 크기에서 유일하게 결정된다.
   (b) **추가 난류 소스** — 대역 게인으로만 색칠된다.
   둘 다 위상을 자유롭게 예측할 수 없다.
2. **보정량에 상한이 있다.** 필터 이득은 ±`max_db`(기본 6 dB)로 묶여 있고,
   추가 노이즈 이득도 묶여 있다. 여기에 잔차 에너지 페널티가 더해진다
   (`losses.residual_energy_db`, 목표 −20 dB 이하).

즉 잔차망이 아무리 열심히 해도 물리모델을 대체할 수 없다. 못 하는 게 아니라
**구조적으로 불가능하다**.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config, DEFAULT
from ..dsp.core import ltv_filter
from ..dsp.filters import bands_to_response
from .encoder import ConvGRUBackbone


class ResidualCorrector(nn.Module):
    """(원음 멜, 물리출력 멜, F0, 유성도) -> 보정 필터 + 추가 노이즈."""

    def __init__(self, cfg: Config = DEFAULT, hidden: int = 192, n_bands: int = 32,
                 max_db: float = 6.0, max_noise: float = 0.15):
        super().__init__()
        self.cfg = cfg
        self.n_bands = n_bands
        self.max_db = max_db
        self.max_noise = max_noise
        self.n_freq = cfg.audio.n_fft // 2 + 1
        # 입력: 목표 멜 + 물리 출력 멜 + log f0 + voicing
        self.backbone = ConvGRUBackbone(cfg.audio.n_mels * 2 + 2, hidden, n_conv=2)
        d = self.backbone.out_dim
        self.head_filter = nn.Linear(d, n_bands)
        self.head_noise = nn.Linear(d, n_bands + 1)
        # 0 에서 시작 = 처음에는 아무것도 안 고친다 (물리모델을 먼저 믿는다)
        nn.init.zeros_(self.head_filter.weight)
        nn.init.zeros_(self.head_filter.bias)
        nn.init.zeros_(self.head_noise.weight)
        nn.init.constant_(self.head_noise.bias, -4.0)

    def forward(self, mel_target: torch.Tensor, mel_phys: torch.Tensor,
                f0: torch.Tensor, voicing: torch.Tensor) -> dict:
        lf0 = torch.log(f0.clamp_min(self.cfg.source.f0_min))[..., None] / 10.0
        h = self.backbone(torch.cat([mel_target, mel_phys, lf0,
                                     voicing[..., None]], dim=-1))
        fb = torch.tanh(self.head_filter(h)) * self.max_db          # ±max_db
        nz = self.head_noise(h)
        return {
            "filter_db": fb,
            "noise_bands": torch.sigmoid(nz[..., :-1]),
            "noise_gain": torch.sigmoid(nz[..., -1:]) * self.max_noise,
        }

    def apply(self, audio: torch.Tensor, r: dict,
              generator: torch.Generator | None = None) -> torch.Tensor:
        """물리 출력에 보정을 적용한다. 반환 (B, N)."""
        hop, ir = self.cfg.audio.hop_size, self.cfg.filt.ir_size
        b, n = audio.shape
        t = r["filter_db"].shape[1]
        H = bands_to_response(10.0 ** (r["filter_db"] / 20.0), self.n_freq,
                              min_phase=True)
        out = ltv_filter(audio, H, hop, ir)
        w = torch.randn(b, t * hop, device=audio.device, dtype=audio.dtype,
                        generator=generator)
        Hn = bands_to_response(r["noise_bands"] * r["noise_gain"], self.n_freq,
                               min_phase=True)
        extra = ltv_filter(w, Hn, hop, 256)
        m = min(out.shape[-1], extra.shape[-1], n)
        # 추가 노이즈는 물리 출력의 RMS 에 비례하도록 스케일 (레벨 불변)
        scale = audio[:, :m].pow(2).mean(-1, keepdim=True).clamp_min(1e-9).sqrt()
        return out[:, :m] + extra[:, :m] * scale
