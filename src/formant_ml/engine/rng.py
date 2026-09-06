"""위치 기반 난수 — 스트리밍과 오프라인이 **같은 잡음**을 쓰게 한다.

청크 단위로 `torch.randn(generator)` 를 부르면 청크 경계에 따라 난수열이 달라져
스트리밍 = 오프라인 불변량이 깨진다. 여기서는 잡음을 (채널, 블록 번호) 로 시드된
고정 블록으로 만들고 필요한 구간을 잘라 준다. 어떤 청크 경계로 잘라도 같은 샘플이
같은 난수를 받는다.
"""
from __future__ import annotations

import torch

BLOCK = 8192


class NoiseBank:
    CHANNELS = ("fric", "asp", "jitter", "shimmer", "mod", "transient")

    def __init__(self, seed: int = 0, cache_blocks: int = 64):
        self.seed = int(seed)
        self._cache: dict[tuple[str, int], torch.Tensor] = {}
        self.cache_blocks = cache_blocks

    def _block(self, channel: str, idx: int, dtype, device) -> torch.Tensor:
        key = (channel, idx)
        b = self._cache.get(key)
        if b is None:
            ch = self.CHANNELS.index(channel) if channel in self.CHANNELS else hash(channel) % 997
            g = torch.Generator().manual_seed((self.seed * 1000003 + ch * 7919 + idx) % (2 ** 31))
            b = torch.randn(BLOCK, generator=g, dtype=torch.float32)
            if len(self._cache) > self.cache_blocks:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = b
        return b.to(dtype=dtype, device=device)

    def white(self, channel: str, start: int, n: int, b: int = 1, dtype=torch.float32,
              device=None) -> torch.Tensor:
        """샘플(또는 프레임) 위치 start 부터 n 개. 반환 (b, n) — 배치는 같은 잡음의 복제."""
        parts = []
        i0, i1 = start // BLOCK, (start + n - 1) // BLOCK
        for idx in range(i0, i1 + 1):
            blk = self._block(channel, idx, dtype, device)
            lo = max(start - idx * BLOCK, 0)
            hi = min(start + n - idx * BLOCK, BLOCK)
            parts.append(blk[lo:hi])
        w = torch.cat(parts)
        return w.unsqueeze(0).expand(b, n)

    def scalar(self, channel: str, index: int) -> float:
        return float(self.white(channel, index, 1)[0, 0])
