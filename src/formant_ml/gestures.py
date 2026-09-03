"""비언어 발성(제스처): 웃음, 숨, 한숨, 프라이, 속삭임, 흐느낌, 헛기침.

이것들은 특수 케이스가 아니다. 전부 같은 물리 손잡이를 다르게 흔든 것뿐이고,
그래서 전부 스크립트에서 연속적으로 섞을 수 있다("반쯤 웃으면서 말하기").

예: 웃음 '하하하' 는
    * 호기 압력의 펄스열 (5~7 Hz)  -> harmonic_amp / noise 세기의 주기적 포락선
    * 펄스마다 F0 가 튀어올랐다 떨어짐
    * 성문 마찰(/ㅎ/)이 앞에 붙음    -> noise_entry 를 성문쪽으로
    * 펄스가 짧아 Rd 가 pressed 쪽
    * 후반부로 갈수록 공기가 빠져 기식성(rd, noise_am) 증가
이 다섯 줄이 곧 아래 `laugh()` 다. 파라미터를 바꾸면 킥킥/껄껄/헛웃음이 된다.

모든 함수는 (n_frames, VoiceProfile, **옵션) -> 제어 dict 를 돌려준다.
값은 (1, T, ·) 텐서이며 그대로 `Controls` 에 넣을 수 있다.
"""
from __future__ import annotations

import math

import torch

from .presets import FRICATIVES, VOWELS
from .utils import band_bump, ramp
from .voice import VoiceProfile, extend_formants


def _c(t: int, v: float) -> torch.Tensor:
    return torch.full((1, t, 1), float(v))


def _phase_ramp(t: int, rate_hz: float, frame_rate: float, phase0: float = 0.0):
    n = torch.arange(t, dtype=torch.float32) / frame_rate
    return 2 * math.pi * rate_hz * n + phase0


def base(t: int, prof: VoiceProfile, n_formants: int = 8, n_bands: int = 40,
         vowel: str = "a") -> dict:
    """프로파일의 기본 상태(중립 모음, 말하는 F0)."""
    # 화자 포먼트로 스케일: 모음의 상대 형태는 유지하고 전체 규모만 화자에 맞춘다
    scale = (prof.formants[0] / VOWELS["a"][0]) if prof.formants else 1.0
    f = [v * scale for v in VOWELS.get(vowel, VOWELS["a"])]
    ff = extend_formants(f + list(prof.formants[len(f):]), n_formants)
    return {
        "f0": _c(t, prof.f0_median),
        "harmonic_amp": _c(t, 1.0),
        "rd": _c(t, prof.rd_median),
        "tilt": _c(t, prof.tilt),
        "jitter": _c(t, prof.jitter),
        "shimmer": _c(t, prof.shimmer),
        "formant_freq": torch.tensor(ff).reshape(1, 1, -1).expand(1, t, n_formants
                                                                 ).contiguous(),
        "formant_bw": prof.bandwidth_tensor(1, t, n_formants),
        "formant_gain": torch.ones(1, t, n_formants),
        "noise_bands": torch.full((1, t, n_bands), 1e-4),
        "noise_entry": _c(t, 0.0),
        "noise_am": _c(t, prof.breathiness),
        "noise_rough": _c(t, prof.roughness),
        # 난류가 성도를 울릴 때는 감쇠가 크다 (Controls.noise_bw_scale 주석 참고)
        "noise_bw_scale": _c(t, 3.0),
        "noise_back_leak": _c(t, 0.25),

    }


# --------------------------------------------------------------------- 웃음
def laugh(t: int, prof: VoiceProfile, frame_rate: float = 100.0, rate_hz: float = 5.5,
          voiced: float = 0.85, breathiness: float = 0.45, pitch_lift: float = 1.45,
          decay: float = 0.55, vowel: str = "a", n_formants: int = 8,
          n_bands: int = 40, sample_rate: int = 24000) -> dict:
    """웃음 '하하하'. rate_hz 가 펄스 속도, voiced 가 유성/기식 비율.

    voiced=0.2, rate=8 이면 '킥킥', voiced=0.95, rate=4 이면 '껄껄',
    voiced=0.0 이면 소리 없이 숨만 터지는 웃음이 된다.
    """
    c = base(t, prof, n_formants, n_bands, vowel)
    ph = _phase_ramp(t, rate_hz, frame_rate)
    # 각 호기 펄스: 빠르게 열리고 천천히 닫히는 비대칭 포락선
    frac = torch.frac(torch.as_tensor(ph) / (2 * math.pi))
    pulse = torch.exp(-frac / 0.35) * (frac < 0.85).float()
    pulse = (pulse / pulse.max().clamp_min(1e-6)).reshape(1, t, 1)
    # 전체적으로 공기가 빠지며 잦아든다
    fade = ramp(t, [(0.0, 1.0), (1.0, decay)])

    c["harmonic_amp"] = pulse * fade * voiced
    # 펄스마다 F0 가 위로 튄다 (웃음의 특징적인 억양)
    c["f0"] = _c(t, prof.f0_median) * (1.0 + (pitch_lift - 1.0) * pulse) \
        * ramp(t, [(0.0, 1.06), (1.0, 0.88)])
    # 짧고 눌린 성문 펄스 -> 뒤로 갈수록 기식적으로
    c["rd"] = ramp(t, [(0.0, max(prof.rd_low, 0.45)), (1.0, prof.rd_high)])
    c["noise_am"] = _c(t, 0.8)
    c["noise_rough"] = _c(t, min(1.0, prof.roughness + 0.10))
    # 성문 마찰(/ㅎ/): 노이즈가 성도 전체를 통과한다
    c["noise_entry"] = _c(t, 0.0)
    nb = band_bump(n_bands, 1400.0, 3500.0, breathiness, sample_rate)
    c["noise_bands"] = (nb.reshape(1, 1, -1) * (0.35 + pulse)).contiguous()
    c["jitter"] = _c(t, min(0.03, prof.jitter * 4))
    c["shimmer"] = _c(t, min(0.3, prof.shimmer * 3))
    return c


def breath(t: int, prof: VoiceProfile, inhale: bool = False, strength: float = 0.5,
           n_formants: int = 8, n_bands: int = 40, sample_rate: int = 24000) -> dict:
    """들숨/날숨. 성대 진동 없이 성문 마찰만."""
    c = base(t, prof, n_formants, n_bands)
    c["harmonic_amp"] = torch.zeros(1, t, 1)
    cf, bw, pos, g = FRICATIVES["h"]
    env = ramp(t, [(0.0, 0.05), (0.35, 1.0), (1.0, 0.1)]) if inhale else \
        ramp(t, [(0.0, 1.0), (1.0, 0.05)])
    c["noise_bands"] = (band_bump(n_bands, cf * (1.4 if inhale else 1.0), bw,
                                  strength, sample_rate).reshape(1, 1, -1)
                        * env).contiguous()
    c["noise_entry"] = _c(t, 0.0)
    c["noise_rough"] = _c(t, 0.20)
    return c


def sigh(t: int, prof: VoiceProfile, n_formants: int = 8, n_bands: int = 40,
         sample_rate: int = 24000) -> dict:
    """한숨: 유성으로 시작해 기식으로 풀리며 F0 가 내려간다."""
    c = base(t, prof, n_formants, n_bands)
    c["harmonic_amp"] = ramp(t, [(0.0, 0.9), (0.6, 0.55), (1.0, 0.0)])
    c["f0"] = _c(t, prof.f0_median) * ramp(t, [(0.0, 1.12), (1.0, 0.8)])
    c["rd"] = ramp(t, [(0.0, prof.rd_median), (1.0, min(2.7, prof.rd_high + 0.5))])
    c["noise_am"] = ramp(t, [(0.0, 0.3), (1.0, 1.0)])
    c["noise_bands"] = (band_bump(n_bands, 1800.0, 4000.0, 0.35, sample_rate
                                  ).reshape(1, 1, -1)
                        * ramp(t, [(0.0, 0.3), (1.0, 1.0)])).contiguous()
    return c


def creak(t: int, prof: VoiceProfile, rate_hz: float = 45.0, n_formants: int = 8,
          n_bands: int = 40) -> dict:
    """성대 프라이(보컬 프라이). 아주 낮은 F0 + 큰 지터 + 압착된 성문파.

    (2질량 ODE 로 하면 주기 배가 분기로 '진짜' 프라이가 나온다.
     LF 경로에서는 낮은 F0 + jitter + pressed Rd 로 근사한다.)
    """
    c = base(t, prof, n_formants, n_bands)
    c["f0"] = _c(t, max(rate_hz, 35.0))
    c["rd"] = _c(t, max(0.3, prof.rd_low - 0.2))
    c["jitter"] = _c(t, 0.03)
    c["shimmer"] = _c(t, 0.25)
    c["harmonic_amp"] = _c(t, 0.6)
    c["tilt"] = _c(t, prof.tilt - 2.0)
    return c


def whisper(t: int, prof: VoiceProfile, vowel: str = "a", strength: float = 0.6,
            n_formants: int = 8, n_bands: int = 40, sample_rate: int = 24000) -> dict:
    """속삭임: 성대 진동 0, 성문 협착 난류가 성도 전체를 통과."""
    c = base(t, prof, n_formants, n_bands, vowel)
    c["harmonic_amp"] = torch.zeros(1, t, 1)
    c["noise_bands"] = band_bump(n_bands, 1600.0, 5000.0, strength, sample_rate
                                 ).reshape(1, 1, -1).expand(1, t, -1).contiguous()
    c["noise_entry"] = _c(t, 0.0)
    c["noise_rough"] = _c(t, 0.15)
    return c


def sob(t: int, prof: VoiceProfile, frame_rate: float = 100.0, rate_hz: float = 2.6,
        n_formants: int = 8, n_bands: int = 40) -> dict:
    """흐느낌: 웃음과 같은 펄스 구조지만 느리고, F0 가 내려꽂히며 기식적."""
    c = laugh(t, prof, frame_rate, rate_hz, voiced=0.7, breathiness=0.6,
              pitch_lift=1.25, decay=0.4, n_formants=n_formants, n_bands=n_bands)
    c["f0"] = c["f0"] * ramp(t, [(0.0, 1.15), (1.0, 0.75)])
    c["rd"] = ramp(t, [(0.0, prof.rd_high), (1.0, min(2.7, prof.rd_high + 0.6))])
    return c


def throat_clear(t: int, prof: VoiceProfile, n_formants: int = 8, n_bands: int = 40,
                 sample_rate: int = 24000) -> dict:
    """헛기침: 짧은 압착 유성 + 넓은 대역 난류."""
    c = base(t, prof, n_formants, n_bands)
    c["harmonic_amp"] = ramp(t, [(0.0, 0.0), (0.15, 1.0), (0.5, 0.3), (1.0, 0.0)])
    c["f0"] = _c(t, prof.f0_low * 0.85)
    c["rd"] = _c(t, 0.4)
    c["jitter"] = _c(t, 0.02)
    c["noise_bands"] = (band_bump(n_bands, 1200.0, 3000.0, 0.6, sample_rate
                                  ).reshape(1, 1, -1)
                        * ramp(t, [(0.0, 0.2), (0.2, 1.0), (1.0, 0.15)])).contiguous()
    c["noise_entry"] = _c(t, 0.0)
    c["noise_rough"] = _c(t, 0.25)
    return c


GESTURES = {
    "laugh": laugh, "breath": breath, "sigh": sigh, "creak": creak,
    "whisper": whisper, "sob": sob, "throat_clear": throat_clear,
}
