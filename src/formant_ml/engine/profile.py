"""화자 프로파일 — 한 화자를 이루는 숫자들 (JSON 한 장).

엔진의 물리 상수(관 길이, F0 범위)와 음소 제스처의 목표(모음 포먼트, 유음 자세, 치찰음
지문, 비음, 음절 길이)를 한 곳에 둔다. `phones.Builder(profile)` 와
`VoiceEngine(cfg, profile)` 가 읽는다. 값은 전부 **녹음 계측**에서 온다
(docs/MEASUREMENTS.md). 손으로 다듬은 값은 `notes` 에 이유를 적는다.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class SpeakerProfile:
    name: str = "female_default"
    sex: str = "female"
    tract_length_cm: float = 14.6
    f0_nominal: float = 220.0
    f0_lo: float = 110.0                 # tension=0
    f0_hi: float = 440.0                 # tension=1
    p_sub: float = 7.5                   # 기본 성문하압 [cmH2O]
    # 모음 (F1, F2, F3) Hz
    vowels: dict = field(default_factory=lambda: {
        "a": [945, 1590, 2850], "eo": [668, 1187, 2882], "o": [486, 889, 2894],
        "u": [401, 1052, 2926], "eu": [464, 1608, 2906], "i": [344, 2814, 3336],
        "e": [494, 2453, 3138], "ae": [630, 2280, 3070]})
    # 유음
    lateral: dict = field(default_factory=lambda: dict(
        f=[459, 1914, 2907], zeros=[3300, 4400], zero_bw=350, gain=1.05,
        hold_ms=120, release_ms=90, level_db=-4.0))
    tap: dict = field(default_factory=lambda: dict(
        f=[486, 1782, 2736], approach_ms=30, closure_ms=25, release_ms=35,
        gain=0.7, dip_db=-5.0, transient_amp=0.0))
    # 치찰음 지문 (앞공동 정점 등)
    sibilant: dict = field(default_factory=lambda: dict(
        peak_hz=8000.0, level_db=-11.0, dur_ms=130, tense_dur_ms=70, a_min=0.10,
        obstacle=0.15, back_leak=0.10, locus=[470.0, 1900.0, 2900.0],
        lax_level_db=-11.0, tense_level_db=-6.0, affricate_level_db=-7.0,
        aspirated_level_db=-4.5,
        abduct_lead_ms=35, abduct_lag_ms=55))
    # 비음
    # 비음: 병렬 비강 분기의 극 3 개 + 폐쇄 위치가 정하는 측지 영점. 시간 상수는 실측.
    nasal: dict = field(default_factory=lambda: dict(
        poles=[350.0, 1200.0, 2000.0], damp=1.2, gain=1.0,
        zero_hz={"m": 1100.0, "n": 1500.0, "ng": 2400.0},
        murmur_db=-10.0, dur_ms=70,
        velum_lead_ms=60, velum_lag_ms=90, closure_ms=35, release_ms=18,
        place_gain={"m": 2.0, "n": 1.0, "ng": 1.2},
        f_murmur=[300.0, 1100.0, 2400.0]))
    # 운율/시간
    timing: dict = field(default_factory=lambda: dict(
        vowel_ms=150, final_vowel_ms=220, silence_ms=50))
    notes: dict = field(default_factory=dict)

    # ------------------------------------------------------------- IO
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, s: str) -> "SpeakerProfile":
        d = json.loads(s)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "SpeakerProfile":
        with open(path, encoding="utf-8") as f:
            return cls.from_json(f.read())

    @property
    def sib_front_len_cm(self) -> float:
        """앞공동 1/4 파장 공진 = 치찰음 정점 → 길이."""
        return 35000.0 / (4.0 * float(self.sibilant["peak_hz"]))


DEFAULT_PROFILE = SpeakerProfile()
