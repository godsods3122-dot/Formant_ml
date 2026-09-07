"""합성 결과 검증: 켑스트럼 포락선에서 포먼트 추정 + 스펙트럼 무게중심."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch

from formant_ml.utils import load_wav


def envelope(y, n_fft=4096, quefrency=48):
    """켑스트럼 리프터링으로 하모닉 미세구조를 지우고 성도 포락선만 남긴다."""
    w = torch.hann_window(min(len(y), n_fft))
    seg = y[: len(w)] * w
    logmag = torch.log(torch.fft.rfft(seg, n_fft).abs().clamp_min(1e-9))
    cep = torch.fft.irfft(logmag.to(torch.complex64), n_fft)
    cep[quefrency:-quefrency] = 0
    return torch.exp(torch.fft.rfft(cep, n_fft).real)


def formants(y, sr, n=4, fmax=6000, n_fft=4096):
    env = envelope(y, n_fft)
    f = torch.linspace(0, sr / 2, len(env))
    idx = [i for i in range(1, len(env) - 1)
           if env[i] > env[i - 1] and env[i] > env[i + 1] and f[i] < fmax]
    idx.sort(key=lambda i: -env[i])
    return sorted(round(float(f[i])) for i in idx[:n])


def centroid(y, sr):
    Y = torch.fft.rfft(y * torch.hann_window(len(y))).abs()
    f = torch.linspace(0, sr / 2, len(Y))
    return float((f * Y).sum() / Y.sum().clamp_min(1e-9))


def hnr(y, sr, f0_hint=None):
    """대략적인 harmonics-to-noise ratio (dB): 자기상관 최대치 기반."""
    y = y - y.mean()
    ac = torch.fft.irfft(torch.fft.rfft(y).abs() ** 2)[: len(y) // 2]
    ac = ac / ac[0].clamp_min(1e-12)
    lo = int(sr / 500)
    r = float(ac[lo:int(sr / 60)].max().clamp(0.0, 0.999))
    return 10 * torch.log10(torch.tensor(r / (1 - r))).item()


if __name__ == "__main__":
    for p in sorted(sys.argv[1:]):
        y = load_wav(p)
        mid = y[len(y) // 3: len(y) // 3 + 16384]
        print(f"{os.path.basename(p):40s} F={formants(mid, 24000)}  "
              f"centroid={centroid(mid, 24000):5.0f}Hz  HNR={hnr(mid, 24000):5.1f}dB")
