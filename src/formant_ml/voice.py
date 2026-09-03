"""VoiceProfile — 한 화자의 목소리를 이루는 고정 파라미터 묶음.

`analysis/extract.py` 가 실제 녹음에서 이걸 채우고, `score.py` 가 이걸 읽어
합성한다. 전부 물리적으로 의미가 있는 숫자라서 손으로 고쳐도 되고, 두 화자의
값을 섞어도 된다("이 사람의 치찰음 + 저 사람의 성구").

JSON 한 장이 곧 목소리다.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import torch


def extend_formants(values, n: int, step: float = 1000.0,
                    min_gap: float = 300.0) -> list[float]:
    """포먼트 목록을 n 개로 늘린다. 마지막 값을 *반복하지 않는다*.

    반복하면 같은 주파수에 극이 겹쳐 캐스케이드 이득이 Q^m 으로 폭발하고,
    그 상태에서 노이즈 경로가 그 단을 지나가면 출력이 10^5 배까지 튄다
    (실제로 겪은 버그: 무음->마찰음 전이에서 클릭).
    """
    out = [float(v) for v in values[:n]]
    while len(out) < n:
        out.append((out[-1] if out else 700.0) + step)
    for i in range(1, len(out)):                 # 단조 + 최소 간격 보장
        out[i] = max(out[i], out[i - 1] + min_gap)
    return out


@dataclass
class VoiceProfile:
    name: str = "default"
    sample_rate: int = 24000

    # --- 성대 ---------------------------------------------------------------
    f0_median: float = 120.0
    f0_low: float = 90.0                 # 5 백분위
    f0_high: float = 200.0               # 95 백분위
    rd_median: float = 1.1               # 기본 성구(pressed 0.3 ~ breathy 2.7)
    rd_low: float = 0.7
    rd_high: float = 1.8
    jitter: float = 0.004                # 주기 요동 비율
    shimmer: float = 0.04
    passaggio: list = field(default_factory=list)   # [(f0_hz, 세기, 아래, 위), ...]
    register_stats: dict = field(default_factory=dict)

    # --- 소스 스펙트럼 ------------------------------------------------------
    tilt: float = 0.0                    # dB/oct, 1 kHz 기준. 양수면 고역이 산다.

    # --- 성도 ---------------------------------------------------------------
    # 나이퀴스트(12 kHz)까지 대략 1 kHz 당 1 개. 개수가 모자라면 아래
    # `extend_formants` 가 채운다 — 마지막 값을 반복하면 극이 겹쳐서 캐스케이드
    # 이득이 폭발한다(같은 주파수에 4 중극이면 Q^4).
    formants: list = field(default_factory=lambda: [730., 1090., 2440., 3400.,
                                                    4500., 5400., 6300., 7200.,
                                                    8200., 9200., 10200., 11200.])
    bandwidths: list = field(default_factory=lambda: [60., 90., 120., 160.,
                                                      200., 240., 280., 320.,
                                                      380., 450., 530., 620.])

    # --- 치찰음 지문 --------------------------------------------------------
    sibilant: dict = field(default_factory=lambda: {
        "pole_f": 6600.0, "pole_bw": 2200.0, "zero_f": 2900.0, "zero_bw": 2600.0,
        "tilt": 1.0})
    sibilant_moments: dict = field(default_factory=dict)

    # --- 위상차 -------------------------------------------------------------
    dispersion: dict = field(default_factory=lambda: {"freq": [], "radius": []})

    # --- 난류 ---------------------------------------------------------------
    # 난류의 느린 세기 변동. 크게 두면 진폭 분포의 꼬리가 두꺼워져 지글거린다
    # (핑크 노이즈의 첨도 2.97 이 기준선; docs/VOICE.md §3 참고).
    roughness: float = 0.12
    breathiness: float = 0.15            # 유성 구간 기식 노이즈 세기

    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ 입출력
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "VoiceProfile":
        with open(path, encoding="utf-8") as f:
            return VoiceProfile(**json.load(f))

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "VoiceProfile":
        known = {k: v for k, v in d.items() if k in VoiceProfile.__annotations__}
        return VoiceProfile(**known)

    # -------------------------------------------------------------- 합성 보조
    def sibilant_params(self, shape, mix: float = 1.0, roughness: float | None = None,
                        device=None, dtype=torch.float32):
        from .dsp.sibilant import SibilantParams
        s = self.sibilant
        return SibilantParams.constant(
            shape, s["pole_f"], s["pole_bw"], s["zero_f"], s["zero_bw"], s["tilt"],
            mix, self.roughness if roughness is None else roughness,
            device=device, dtype=dtype)

    def dispersion_tensors(self, batch: int, n_frames: int, device=None,
                           dtype=torch.float32):
        """(disp_freq, disp_radius) 또는 (None, None)."""
        d = self.dispersion
        if not d.get("freq"):
            return None, None
        f = torch.tensor(d["freq"], device=device, dtype=dtype)
        r = torch.tensor(d["radius"], device=device, dtype=dtype)
        shape = (batch, n_frames, len(f))
        return f.expand(shape).contiguous(), r.expand(shape).contiguous()

    def formant_tensor(self, batch: int, n_frames: int, n_formants: int,
                       device=None, dtype=torch.float32):
        f = extend_formants(self.formants, n_formants)
        return torch.tensor(f[:n_formants], device=device, dtype=dtype
                            ).view(1, 1, -1).expand(batch, n_frames, n_formants
                                                    ).contiguous()

    def bandwidth_tensor(self, batch: int, n_frames: int, n_formants: int,
                         device=None, dtype=torch.float32):
        b = list(self.bandwidths)
        while len(b) < n_formants:
            b.append(b[-1] * 1.2)
        return torch.tensor(b[:n_formants], device=device, dtype=dtype
                            ).view(1, 1, -1).expand(batch, n_frames, n_formants
                                                    ).contiguous()

    def register_for(self, f0_hz: float) -> str:
        """이 F0 가 파사지오 위인지 아래인지 (해석용)."""
        if not self.passaggio:
            return "modal"
        p = sorted(float(r[0]) for r in self.passaggio)
        n = sum(1 for v in p if f0_hz > v)
        return ["chest", "mixed", "head", "falsetto"][min(n, 3)]
