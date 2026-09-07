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
                 repeat: int = 1, extensions=(".wav", ".flac"),
                 target_rms: float = 0.05):
        self.cfg = cfg
        self.target_rms = target_rms
        self._gains: dict[str, float] = {}
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
        return {"audio": y * self._gain(path), "path": path}

    def _gain(self, path: str) -> float:
        """**파일 단위** RMS 정규화 (크롭 단위 피크 정규화가 아니다).

        크롭마다 피크를 0.95 로 맞추면 조용한 구간이 큰 소리로 부풀려지고, 모델은
        그 들쭉날쭉한 레벨을 설명하려고 성문 세기를 엉뚱하게 흔든다. 1~10 초짜리
        조각을 짜깁기해 쓸 때 특히 문제가 된다 — 조각마다 다른 이득이 붙으면
        화자의 다이내믹이 통째로 사라진다. 파일 하나당 하나의 이득을 캐시한다.
        """
        if path not in self._gains:
            y = load_wav(path, self.cfg.audio.sample_rate)
            rms = y.pow(2).mean().clamp_min(1e-10).sqrt()
            g = float(self.target_rms / rms)
            peak = float(y.abs().max()) * g
            self._gains[path] = g if peak <= 0.99 else g * 0.99 / peak
        return self._gains[path]


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
