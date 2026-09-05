"""실시간(청크 단위) 합성.

왜 이게 되는가
--------------
이 합성기는 파형을 한 번에 만들지 않는다. 매 순간의 상태는
(a) 성문의 순시위상 하나와 (b) 필터의 OLA 꼬리뿐이다. 그 둘만 이어 주면
100 ms 씩 잘라 만들어도 한 번에 만든 것과 실질적으로 같은 결과가 나온다
(측정: 청크 크기 20~250 ms 에서 차이가 최대 −60 dB, 들리지 않는다).
즉 파라미터를 프레임(10 ms)마다 바꿔 넣으면 그게 곧 실시간 제어다 —
사람이 후두와 혀를 움직이는 것과 같은 방식으로.

지연
----
* LTV 필터의 고정 지연  `dsp.core.ltv_delay` = `ir_size/2 + hop` = 496 샘플
  = 20.7 ms (한 번만 생긴다). 교차창이 한 프레임 뒤를 보기 때문에 `hop` 이
  붙는다 — 직사각 블록의 10 ms 격자 타일을 없애는 값이다(core.ltv_filter 주석).
* 제어 보간용 선행 1 프레임 = 10 ms
합쳐서 약 31 ms. 대화형 응답에는 충분하다.

    st = StreamingSynth(cfg)
    for ctrl_chunk in controller:          # 매 청크의 물리 파라미터
        audio = st.step(ctrl_chunk)        # 바로 스피커로
"""
from __future__ import annotations

import torch

from .config import Config, DEFAULT
from .dsp.core import ltv_delay
from .models.synth import Controls, PhysicalVoiceSynth, cat_controls


class StreamingSynth:
    def __init__(self, cfg: Config = DEFAULT, tract_mode: str = "formant",
                 synth: PhysicalVoiceSynth | None = None,
                 generator: torch.Generator | None = None):
        self.cfg = cfg
        self.synth = synth or PhysicalVoiceSynth(cfg, tract_mode=tract_mode)
        self.generator = generator
        self.reset()

    def reset(self) -> None:
        self.state: dict = {}
        self.pending: Controls | None = None

    @property
    def latency_samples(self) -> int:
        return (ltv_delay(self.cfg.filt.ir_size, self.cfg.audio.hop_size)
                + self.cfg.audio.hop_size)

    @property
    def latency_ms(self) -> float:
        return 1000.0 * self.latency_samples / self.cfg.audio.sample_rate

    @torch.no_grad()
    def step(self, ctrl: Controls) -> torch.Tensor:
        """제어 청크 -> 오디오 청크 (B, N). 첫 호출은 1 프레임 짧게 나온다."""
        buf = ctrl if self.pending is None else cat_controls([self.pending, ctrl])
        n = buf.n_frames
        if n < 2:
            self.pending = buf
            return torch.zeros(buf.f0.shape[0], 0, device=buf.f0.device,
                               dtype=buf.f0.dtype)
        out = self.synth(buf, generator=self.generator, state=self.state,
                         emit_frames=n - 1)
        self.state = out["state"]
        self.pending = buf.slice(n - 1, n)      # 마지막 프레임 = 다음 보간 기준점
        return out["audio"]

    @torch.no_grad()
    def flush(self) -> torch.Tensor:
        """마지막에 남은 1 프레임과 필터 꼬리를 내보낸다."""
        if self.pending is None:
            return torch.zeros(1, 0)
        tail = self.step(self.pending.slice(0, 1))
        self.reset()
        return tail
