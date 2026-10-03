"""위치 기반 난수 — 스트리밍과 오프라인이 **같은 잡음**을 쓰게 한다.

청크 단위로 `torch.randn(generator)` 를 부르면 청크 경계에 따라 난수열이 달라져
스트리밍 = 오프라인 불변량이 깨진다. 여기서는 잡음을 (채널, 블록 번호) 로 시드된
고정 블록으로 만들고 필요한 구간을 잘라 준다. 어떤 청크 경계로 잘라도 같은 샘플이
같은 난수를 받는다.
"""
from __future__ import annotations

import zlib

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
            # **이름에서 번호를 뽑을 때 `hash()` 를 쓰면 안 된다.** 문자열 해시는 프로세스마다
            # 무작위화되어(PYTHONHASHSEED) 같은 씨앗으로도 프로세스마다 다른 난수가 나온다.
            # 실측: 배음 위상 채널("hphase")이 그래서, 같은 제어열을 다시 렌더한 치찰음 고역
            # 변조가 프로세스마다 +9.1 / +6.6 dB 로 달랐다(MEASUREMENTS §50). crc32 는 고정이다.
            ch = (self.CHANNELS.index(channel) if channel in self.CHANNELS
                  else 100 + zlib.crc32(channel.encode("utf-8")) % 997)
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

class LockedNoiseBank(NoiseBank):
    """난류 채널을 **목표에서 뽑은 여기**로 갈아 끼운 잡음 은행 (MEASUREMENTS §52.114).

    난수로 난류를 만드는 한, 같은 제어열을 씨앗만 바꿔 렌더한 둘이 서로 96.19 % 다 — 그것이
    목표 대비 점수의 상한이다. 복제가 목표인 단계에서는 여기를 목표의 **비조화 잔차**(표백)
    에서 받아 오는 것이 정당하다 (`waveform.whiten_aperiodic`).

    지터·시머·배음 위상 같은 **제어 축**은 그대로 난수다 — 그것들은 난류가 아니라 몸의 떨림이고,
    펄스 잠금이 이미 목표의 주기 경계를 준다. 여기서 바꾸는 것은 `LOCKED` 의 난류 채널뿐이다.
    """

    #: 이 이름(또는 접두어)으로 들어오는 요청만 목표 여기로 답한다.
    LOCKED = ("asp", "fric", "aspu")

    def __init__(self, seed: int = 0, excitation=None, bands=None, cache_blocks: int = 64):
        super().__init__(seed, cache_blocks)
        self.exc = None if excitation is None else torch.as_tensor(
            excitation, dtype=torch.float32).reshape(-1)
        #: `aspu{i}` 는 대역마다 다른 여기를 받는다 (없으면 통짜를 쓴다).
        self.bands = None if bands is None else [
            torch.as_tensor(b, dtype=torch.float32).reshape(-1) for b in bands]

    def _locked_src(self, channel: str):
        if self.exc is None:
            return None
        if channel.startswith("aspu") and self.bands:
            try:
                i = int(channel[4:])
            except ValueError:
                return self.exc
            return self.bands[i % len(self.bands)]
        return self.exc if channel in self.LOCKED else None

    def white(self, channel: str, start: int, n: int, b: int = 1, dtype=torch.float32,
              device=None) -> torch.Tensor:
        src = self._locked_src(channel)
        if src is None:
            return super().white(channel, start, n, b, dtype, device)
        # 구간을 벗어나면 되풀이한다 (스트리밍에서 목표보다 길어질 수 있다).
        idx = torch.arange(start, start + n) % max(len(src), 1)
        w = src[idx].to(dtype=dtype, device=device)
        return w.unsqueeze(0).expand(b, n)

