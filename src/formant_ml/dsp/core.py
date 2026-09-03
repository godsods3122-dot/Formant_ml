"""공통 DSP 유틸: 보간, FFT 컨볼루션, 시변(LTV) 필터.

핵심 아이디어
-------------
모든 필터는 "프레임별 복소 주파수응답 -> 임펄스응답 -> 프레임 단위 FFT 컨볼루션 ->
overlap-add" 경로로 적용한다. 재귀(IIR) 루프를 돌지 않으므로 시퀀스 길이에 대해
병렬이고, 응답이 복소수이므로 크기뿐 아니라 위상(군지연)까지 그대로 살아 있다.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

TWO_PI = 2.0 * math.pi


def next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def upsample(x: torch.Tensor, hop_size: int, method: str = "linear") -> torch.Tensor:
    """프레임률 제어신호 (B, T, C) -> 샘플률 (B, T*hop, C).

    method="linear"  : 선형보간(연속적으로 변하는 파라미터: F0, 포먼트, 게인)
    method="nearest" : 계단형(이산 스위칭)
    """
    b, t, c = x.shape
    if method == "nearest":
        return x.repeat_interleave(hop_size, dim=1)
    # 끝단 보정을 위해 마지막 프레임을 한 번 더 붙였다가 잘라낸다.
    xx = torch.cat([x, x[:, -1:]], dim=1).transpose(1, 2)  # (B, C, T+1)
    out = F.interpolate(xx, size=t * hop_size + 1, mode="linear", align_corners=True)
    return out[..., : t * hop_size].transpose(1, 2)


def fft_convolve(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """마지막 축에 대한 선형 컨볼루션. (..., n) * (..., m) -> (..., n+m-1)"""
    n = signal.shape[-1] + kernel.shape[-1] - 1
    nfft = next_power_of_two(n)
    out = torch.fft.irfft(
        torch.fft.rfft(signal, nfft) * torch.fft.rfft(kernel, nfft), nfft
    )
    return out[..., :n]


def response_to_ir(H: torch.Tensor, ir_size: int, window: bool = True) -> torch.Tensor:
    """복소 주파수응답 (..., n_freq) -> 유한 길이 임펄스응답 (..., ir_size).

    ir_size//2 샘플의 순수 지연이 생기며, 이는 호출부에서 보정한다.
    """
    ir = torch.fft.irfft(H)                       # (..., n_fft), 0 중심(원형)
    ir = torch.roll(ir, ir_size // 2, dims=-1)[..., :ir_size]
    if window:
        w = torch.hann_window(ir_size, dtype=ir.dtype, device=ir.device)
        ir = ir * w
    return ir


def ltv_filter(
    x: torch.Tensor,
    H: torch.Tensor,
    hop_size: int,
    ir_size: int = 512,
    tail: torch.Tensor | None = None,
    return_tail: bool = False,
):
    """시변 필터링. x: (B, N), H: (B, T, n_freq) 복소응답. 반환 (B, N).

    x를 hop_size 길이의 비중첩 프레임으로 자르고, 각 프레임을 해당 프레임의
    임펄스응답과 컨볼루션한 뒤 overlap-add 한다.

    스트리밍(`return_tail=True`): 지연 보정을 하지 않고 OLA 꼬리를 그대로
    돌려준다. 다음 청크의 앞에 그 꼬리를 더하면 결과가 오프라인 합성과
    **정확히 같다**. 대신 스트림 전체에 `ir_size//2` 샘플의 고정 지연이 남는다
    (한 번만 생기는 상수 지연이라 실시간 제어에는 문제가 없다).
    """
    b, n = x.shape
    t = H.shape[1]
    n_pad = t * hop_size
    if n < n_pad:
        x = F.pad(x, (0, n_pad - n))
    frames = x[:, :n_pad].reshape(b, t, hop_size)

    ir = response_to_ir(H, ir_size)                     # (B, T, ir_size)
    wet = fft_convolve(frames, ir)                      # (B, T, hop+ir-1)

    out_len = n_pad + ir_size - 1
    out = torch.zeros(b, out_len, dtype=x.dtype, device=x.device)
    out = out.index_put_(
        (
            torch.arange(b, device=x.device)[:, None, None],
            (torch.arange(t, device=x.device)[:, None] * hop_size
             + torch.arange(wet.shape[-1], device=x.device)[None, :])[None],
        ),
        wet,
        accumulate=True,
    )
    if tail is not None:
        m = tail.shape[-1]
        out = torch.cat([out[:, :m] + tail, out[:, m:]], dim=-1)
    if return_tail:
        return out[:, :n_pad], out[:, n_pad:]
    delay = ir_size // 2
    return out[:, delay : delay + n]


def amp_to_db(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return 20.0 * torch.log10(x.abs() + eps)


def scale_sigmoid(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """신경망 출력을 [lo, hi] 물리 범위로 안전하게 사상."""
    return lo + (hi - lo) * torch.sigmoid(x)


def exp_sigmoid(x: torch.Tensor, max_value: float = 2.0, eps: float = 1e-7,
                threshold: float = 1e-7) -> torch.Tensor:
    """DDSP의 진폭 활성화: 양수 + 지수적 다이내믹 레인지."""
    return max_value * torch.sigmoid(x) ** math.log(10.0) + threshold + eps


def hz_to_omega(f_hz: torch.Tensor, sample_rate: float) -> torch.Tensor:
    """Hz -> 정규화 각주파수(rad/sample)."""
    return TWO_PI * f_hz / sample_rate


def freq_grid(n_freq: int, sample_rate: float, device=None, dtype=torch.float32
              ) -> torch.Tensor:
    """rfft 주파수 격자(Hz). n_freq = n_fft//2 + 1."""
    return torch.linspace(0.0, sample_rate / 2, n_freq, device=device, dtype=dtype)


def group_delay(H: torch.Tensor, sample_rate: float) -> torch.Tensor:
    """복소응답의 군지연(초). 위상 언랩 대신 인접 빈 위상차를 사용(랩 안전)."""
    phase = torch.angle(H)
    d = phase[..., 1:] - phase[..., :-1]
    d = d - TWO_PI * torch.round(d / TWO_PI)     # anti-wrapping
    n_freq = H.shape[-1]
    df = (sample_rate / 2) / (n_freq - 1)
    return -d / (TWO_PI * df)
