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


def ltv_delay(ir_size: int, hop_size: int) -> int:
    """`ltv_filter` 가 만드는 고정 지연 [샘플].

    IR 중심을 맞추는 `ir_size//2` 에, 교차창이 한 프레임 뒤를 보기 위한
    `hop_size` 가 더해진다. 스트리밍 지연 계산과 오프라인-스트리밍 정렬이
    같은 값을 쓰도록 여기 한 곳에 둔다.
    """
    return ir_size // 2 + hop_size


def ltv_filter(
    x: torch.Tensor,
    H: torch.Tensor,
    hop_size: int,
    ir_size: int = 512,
    tail: tuple[torch.Tensor, torch.Tensor] | None = None,
    return_tail: bool = False,
):
    """시변 필터링. x: (B, N), H: (B, T, n_freq) 복소응답. 반환 (B, N).

    **왜 창을 씌우는가 (직사각 블록으로 하면 안 되는 이유).**
    예전 구현은 x 를 `hop_size` 길이의 **직사각** 블록으로 잘라 각각 그 프레임의
    IR 로 컨볼루션한 뒤 겹쳐 더했다. 응답이 고정이면 블록의 합이 곧 x 라
    완전복원이지만, **응답이 프레임마다 다르면 직사각 절단이 만든 스펙트럼
    번짐(주파수축 sinc)이 상쇄되지 않고 그대로 남는다.** 포먼트가 움직이는
    구간에서 10 ms 격자의 타일이 스펙트로그램에 찍혔고, 3.5~9 kHz 포락선의
    프레임간 |Δ| 가 원본 1.9 dB 대비 4.2 dB 로 튀었다.
    (측정과 재현: `scripts/diag_hifreq.py --ablate`, docs/HANDOFF_LIQUID.md §2)

    그래서 `2*hop_size` 길이의 **주기적 Hann 창**을 50 % 중첩으로 쓴다. 이 창은
    50 % 중첩에서 합이 정확히 1(COLA)이라 **응답이 고정이면 여전히 완전복원**
    이면서, 응답이 변할 때는 인접 프레임의 결과가 부드럽게 교차한다.
    같은 여기신호·같은 H 로 재면 프레임간 |Δ| 4.25 -> 1.77 dB (원본 1.92).
    제어율을 4 배로 올려도 직사각을 유지하면 2.76 까지밖에 안 내려간다 —
    **제어율이 아니라 창이 문제다.**

    한 프레임 뒤를 보게 되므로 고정 지연이 `ir_size//2` 에서
    `ltv_delay(ir_size, hop_size)` 로 `hop_size` 만큼 늘어난다.

    스트리밍(`return_tail=True`): 지연 보정을 하지 않고 상태를 그대로 돌려준다.
    상태는 **(OLA 꼬리, 입력 마지막 hop 샘플)** 두 개다 — 교차창은 프레임보다
    `hop_size` 앞의 입력을 보므로 꼬리만으로는 청크를 이을 수 없다. 그대로
    다음 호출의 `tail` 에 넣으면 오프라인 합성과 정확히 같은 결과가 나온다.
    """
    b, n = x.shape
    t = H.shape[1]
    n_pad = t * hop_size
    if n < n_pad:
        x = F.pad(x, (0, n_pad - n))
    x = x[:, :n_pad]

    # 직전 청크의 마지막 hop 샘플(스트리밍) 또는 0(오프라인)을 앞에 붙인다.
    hist = tail[1] if tail is not None else \
        torch.zeros(b, hop_size, dtype=x.dtype, device=x.device)
    win_len = 2 * hop_size
    # 오프라인에서는 프레임을 하나 더 돌린다 — 마지막 hop 샘플은 다음 프레임의
    # 앞창이 있어야 창 합이 1 이 되기 때문이다(없으면 끝이 Hann 으로 페이드아웃).
    # 스트리밍에서는 그 프레임이 다음 청크의 프레임 0 이므로 더하면 이중계산이다.
    n_frames = t if return_tail else t + 1
    xp = torch.cat([hist, x, x.new_zeros(b, win_len)], dim=1)
    frames = xp.unfold(1, win_len, hop_size)[:, :n_frames]      # (B, T', win)
    w = torch.hann_window(win_len, periodic=True, dtype=x.dtype, device=x.device)

    ir = response_to_ir(H, ir_size)                             # (B, T, ir_size)
    if n_frames > t:                        # 여분 프레임은 마지막 응답을 그대로 쓴다
        ir = torch.cat([ir, ir[:, -1:]], dim=1)
    wet = fft_convolve(frames * w, ir)                          # (B, T', win+ir-1)

    out_len = (n_frames - 1) * hop_size + wet.shape[-1]
    out = torch.zeros(b, out_len, dtype=x.dtype, device=x.device)
    out = out.index_put_(
        (
            torch.arange(b, device=x.device)[:, None, None],
            (torch.arange(n_frames, device=x.device)[:, None] * hop_size
             + torch.arange(wet.shape[-1], device=x.device)[None, :])[None],
        ),
        wet,
        accumulate=True,
    )
    if tail is not None:
        m = tail[0].shape[-1]
        out = torch.cat([out[:, :m] + tail[0], out[:, m:]], dim=-1)
    if return_tail:
        return out[:, :n_pad], (out[:, n_pad:], x[:, n_pad - hop_size:])
    d = ltv_delay(ir_size, hop_size)
    return out[:, d: d + n]


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
