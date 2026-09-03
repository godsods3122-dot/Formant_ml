"""특징 추출: STFT, 멜, YIN F0.

외부 의존(torchaudio/librosa) 없이 torch 만으로 구현한다. 학습 손실이 이 함수들을
직접 쓰기 때문에 STFT 는 **복소수 그대로** 반환한다 (위상 손실에 필요).

배열 규약
---------
stft   : (B, N) -> (B, n_freq, T)      마지막 축이 시간 (위상 시간차 = 순시주파수)
log_mel: (B, N) -> (B, T, n_mels)      인코더 입력 규약과 동일
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_WINDOW_CACHE: dict = {}
_MEL_CACHE: dict = {}


# ------------------------------------------------------------------------ STFT
def _window(n: int, device, dtype):
    key = (n, str(device), dtype)
    if key not in _WINDOW_CACHE:
        _WINDOW_CACHE[key] = torch.hann_window(n, device=device, dtype=dtype)
    return _WINDOW_CACHE[key]


def stft(x: torch.Tensor, n_fft: int = 1024, hop: int = 240,
         win: int | None = None, center: bool = True) -> torch.Tensor:
    """(B, N) -> 복소 (B, n_fft//2+1, T)."""
    if x.dim() == 1:
        x = x[None]
    win = win or n_fft
    return torch.stft(x, n_fft, hop, win, _window(win, x.device, x.dtype),
                      center=center, pad_mode="reflect", return_complex=True)


def magnitude(x: torch.Tensor, n_fft: int = 1024, hop: int = 240,
              eps: float = 1e-7) -> torch.Tensor:
    return stft(x, n_fft, hop).abs().clamp_min(eps)


# ------------------------------------------------------------------------- 멜
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int = 24000, n_fft: int = 1024, n_mels: int = 80,
                   fmin: float = 40.0, fmax: float = 12000.0,
                   device=None, dtype=torch.float32) -> torch.Tensor:
    """삼각 멜 필터뱅크 (n_mels, n_fft//2+1). 면적 정규화(slaney) 없이 진폭 기준."""
    key = (sample_rate, n_fft, n_mels, fmin, fmax, str(device), dtype)
    if key in _MEL_CACHE:
        return _MEL_CACHE[key]
    n_freq = n_fft // 2 + 1
    f = torch.linspace(0, sample_rate / 2, n_freq, device=device, dtype=dtype)
    m = torch.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2,
                       device=device, dtype=dtype)
    edges = 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    lo, ctr, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    up = (f[None] - lo) / (ctr - lo).clamp_min(1e-6)
    dn = (hi - f[None]) / (hi - ctr).clamp_min(1e-6)
    fb = torch.minimum(up, dn).clamp_min(0.0)
    _MEL_CACHE[key] = fb
    return fb


def log_mel(x: torch.Tensor, sample_rate: int = 24000, n_fft: int = 1024,
            hop: int = 240, n_mels: int = 80, fmin: float = 40.0,
            fmax: float = 12000.0, eps: float = 1e-5) -> torch.Tensor:
    """(B, N) -> (B, T, n_mels) 로그 멜."""
    mag = stft(x, n_fft, hop).abs()
    fb = mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax, mag.device, mag.dtype)
    mel = torch.einsum("mf,bft->btm", fb, mag)
    return torch.log(mel.clamp_min(eps))


# ------------------------------------------------------------------- 대역 에너지
def log_band_energy(x: torch.Tensor, sample_rate: int = 24000, n_fft: int = 1024,
                    hop: int = 240, n_bands: int = 24, fmin: float = 50.0,
                    fmax: float | None = None, eps: float = 1e-6) -> torch.Tensor:
    """로그 주파수축으로 등분한 대역별 로그 에너지 (B, T, n_bands).

    멜/STFT 손실은 절대 에너지가 큰 저역이 지배하기 때문에 고역이 학습되지 않는다.
    각 대역을 *따로* 맞추게 하면 6~12 kHz 처럼 에너지가 40 dB 낮은 구간도
    동등한 무게를 갖는다. (고역 부족 문제의 직접 대응)
    """
    fmax = fmax or sample_rate / 2 * 0.99
    mag = stft(x, n_fft, hop).abs()
    n_freq = mag.shape[-2]
    f = torch.linspace(0, sample_rate / 2, n_freq, device=mag.device, dtype=mag.dtype)
    edges = torch.logspace(math.log10(fmin), math.log10(fmax), n_bands + 1,
                           device=mag.device, dtype=mag.dtype)
    masks = ((f[None] >= edges[:-1, None]) & (f[None] < edges[1:, None])).to(mag.dtype)
    masks = masks / masks.sum(-1, keepdim=True).clamp_min(1.0)
    band = torch.einsum("bf,bft->bt" if False else "kf,bft->btk", masks, mag)
    return torch.log(band.clamp_min(eps))


# ------------------------------------------------------------------------- F0
def frame_signal(x: torch.Tensor, frame_len: int, hop: int) -> torch.Tensor:
    """(B, N) -> (B, T, frame_len). 중앙정렬(center) 패딩, T = ceil(N/hop)."""
    n = x.shape[-1]
    n_frames = n // hop + 1               # torch.stft(center=True) 와 동일한 프레임 수
    pad_l = frame_len // 2
    pad_r = frame_len // 2 + (n_frames - 1) * hop + 1 - n
    xp = F.pad(x, (pad_l, max(pad_r, 0)), mode="reflect" if n > frame_len else "constant")
    return xp.unfold(1, frame_len, hop)[:, :n_frames]


def yin_f0(x: torch.Tensor, sample_rate: int = 24000, hop: int = 240,
           fmin: float = 55.0, fmax: float = 880.0, frame_len: int = 1024,
           threshold: float = 0.15):
    """YIN 기본주파수 추정. 반환 (f0 (B,T) Hz, voicing (B,T) 0..1).

    누적평균정규화 차이함수(CMND)의 최솟값이 임계 이하이면 유성으로 본다.
    voicing 은 0/1 이 아니라 `1 - d'` 의 연속값이라 손실 가중에 바로 쓸 수 있다.
    """
    if x.dim() == 1:
        x = x[None]
    w = frame_signal(x, frame_len, hop)                      # (B, T, W)
    W = frame_len
    L = W // 2
    w = w - w.mean(-1, keepdim=True)

    # r(tau) = sum_{j<L} w[j] w[j+tau]  (FFT 상호상관)
    nfft = 1
    while nfft < 2 * W:
        nfft <<= 1
    A = torch.fft.rfft(w[..., :L], nfft)
    B_ = torch.fft.rfft(w, nfft)
    r = torch.fft.irfft(A.conj() * B_, nfft)[..., :L]         # (B, T, L)

    p = w.pow(2)
    csum = torch.cat([torch.zeros_like(p[..., :1]), p.cumsum(-1)], dim=-1)
    e0 = csum[..., L:L + 1] - csum[..., :1]                   # sum_{j<L}
    taus = torch.arange(L, device=x.device)
    e_tau = csum[..., taus + L] - csum[..., taus]             # (B, T, L)

    d = (e0 + e_tau - 2.0 * r).clamp_min(0.0)
    cum = d[..., 1:].cumsum(-1) / taus[1:].to(d.dtype)
    dn = torch.ones_like(d)
    dn[..., 1:] = d[..., 1:] / cum.clamp_min(1e-9)

    tau_min = max(int(sample_rate / fmax), 2)
    tau_max = min(int(sample_rate / fmin) + 1, L - 1)
    seg = dn[..., tau_min:tau_max]                            # (B, T, S)

    # 임계 이하이면서 *국소 최소* 인 첫 지점. (국소최소 조건이 없으면 골짜기의
    # 왼쪽 사면을 잡아 주기를 짧게 추정하고, F0 가 6% 정도 높게 나온다.)
    pad = torch.nn.functional.pad(seg, (1, 1), value=float("inf"))
    is_min = (seg <= pad[..., :-2]) & (seg <= pad[..., 2:])
    cand = (seg < threshold) & is_min
    first = torch.where(cand.any(-1), cand.float().argmax(-1), seg.argmin(-1))
    idx = first + tau_min

    # 포물선 보간 (샘플 격자보다 정밀한 주기)
    i0 = idx.clamp(1, L - 2)
    g = torch.gather
    y0 = g(dn, -1, (i0 - 1).unsqueeze(-1)).squeeze(-1)
    y1 = g(dn, -1, i0.unsqueeze(-1)).squeeze(-1)
    y2 = g(dn, -1, (i0 + 1).unsqueeze(-1)).squeeze(-1)
    denom = (y0 - 2 * y1 + y2)
    shift = torch.where(denom.abs() > 1e-9, 0.5 * (y0 - y2) / denom.clamp_min(1e-9),
                        torch.zeros_like(denom)).clamp(-0.5, 0.5)
    period = i0.to(x.dtype) + shift
    f0 = sample_rate / period.clamp_min(1.0)
    f0 = f0.clamp(fmin, fmax)

    conf = (1.0 - y1).clamp(0.0, 1.0)
    voiced = (y1 < threshold).to(x.dtype)
    voicing = conf * voiced
    f0 = torch.where(voiced > 0, f0, torch.full_like(f0, 0.0))
    return f0, voicing


def fill_unvoiced(f0: torch.Tensor, default: float = 120.0) -> torch.Tensor:
    """무성 구간의 F0(0)을 직전 유성 값으로 채운다(합성 시 위상 연속성 유지)."""
    out = f0.clone()
    b, t = out.shape
    last = torch.full((b,), default, dtype=out.dtype, device=out.device)
    for i in range(t):
        cur = out[:, i]
        use = torch.where(cur > 0, cur, last)
        out[:, i] = use
        last = use
    return out
