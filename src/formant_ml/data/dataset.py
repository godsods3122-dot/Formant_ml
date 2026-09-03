"""wav 폴더 데이터셋 + 학습용 특징 계산."""
from __future__ import annotations

import glob
import os
import random

import torch
from torch.utils.data import Dataset

from ..config import Config, DEFAULT
from ..utils import load_wav
from .features import fill_unvoiced, log_mel, yin_f0


class AudioFolder(Dataset):
    """폴더 안의 wav 를 무작위 구간으로 잘라 내보낸다.

    데이터가 작을 때는 `repeat` 로 에폭 길이를 늘린다(DataLoader 재시작 비용 절감).
    """

    def __init__(self, root: str, cfg: Config = DEFAULT, seconds: float = 1.5,
                 repeat: int = 1, extensions=(".wav", ".flac")):
        self.cfg = cfg
        self.seconds = seconds
        self.repeat = max(1, repeat)
        self.files = sorted(
            p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
            if p.lower().endswith(extensions))
        if not self.files:
            raise FileNotFoundError(f"{root} 안에 wav/flac 이 없습니다")
        self.n_samples = int(seconds * cfg.audio.sample_rate)
        # hop 배수로 맞춰야 프레임/샘플 길이가 정확히 대응한다.
        self.n_samples -= self.n_samples % cfg.audio.hop_size

    def __len__(self) -> int:
        return len(self.files) * self.repeat

    def __getitem__(self, i: int) -> dict:
        path = self.files[i % len(self.files)]
        y = load_wav(path, self.cfg.audio.sample_rate)
        n = self.n_samples
        if len(y) < n:
            y = torch.nn.functional.pad(y, (0, n - len(y)))
        else:
            s = random.randint(0, len(y) - n)
            y = y[s:s + n]
        peak = y.abs().max().clamp_min(1e-5)
        return {"audio": y / peak * 0.95, "path": path}


def compute_features(x: torch.Tensor, cfg: Config = DEFAULT) -> dict:
    """(B, N) 파형 -> {mel (B,T,n_mels), f0 (B,T), voicing (B,T)}.

    프레임 수는 N//hop 로 맞춘다(합성기가 T*hop 샘플을 만들기 때문).
    """
    a = cfg.audio
    t = x.shape[-1] // a.hop_size
    mel = log_mel(x, a.sample_rate, a.n_fft, a.hop_size, a.n_mels, a.fmin, a.fmax)
    f0, voicing = yin_f0(x, a.sample_rate, a.hop_size,
                         cfg.source.f0_min, cfg.source.f0_max)
    return {
        "mel": mel[:, :t],
        "f0": fill_unvoiced(f0[:, :t]),
        "voicing": voicing[:, :t],
        "f0_raw": f0[:, :t],
    }
