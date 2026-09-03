"""VoiceProfile — 한 화자의 목소리를 이루는 고정 파라미터 묶음.

`analysis/extract.py` 가 실제 녹음에서 이걸 채우고, `score.py` 가 이걸 읽어
합성한다. 전부 물리적으로 의미가 있는 숫자라서 손으로 고쳐도 되고, 두 화자의
값을 섞어도 된다("이 사람의 치찰음 + 저 사람의 성구").

JSON 한 장이 곧 목소리다.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import torch


def extend_formants(values, n: int, step: float = 1000.0,
                    min_gap: float = 300.0) -> list[float]:
    """포먼트 목록을 n 개로 늘린다. 마지막 값을 *반복하지 않는다*.

    반복하면 같은 주파수에 극이 겹쳐 캐스케이드 이득이 Q^m 으로 폭발하고,
    그 상태에서 노이즈 경로가 그 단을 지나가면 출력이 10^5 배까지 튄다
    (실제로 겪은 버그: 무음->마찰음 전이에서 클릭).
    """
    out = [float(v) for v in values[:n]]
    while len(out) < n:
        out.append((out[-1] if out else 700.0) + step)
    for i in range(1, len(out)):                 # 단조 + 최소 간격 보장
        out[i] = max(out[i], out[i - 1] + min_gap)
    return out


@dataclass
class VoiceProfile:
    name: str = "default"
    sample_rate: int = 24000

    # --- 성대 ---------------------------------------------------------------
    f0_median: float = 120.0
    f0_low: float = 90.0                 # 5 백분위
    f0_high: float = 200.0               # 95 백분위
    rd_median: float = 1.1               # 기본 성구(pressed 0.3 ~ breathy 2.7)
    rd_low: float = 0.7
    rd_high: float = 1.8
    jitter: float = 0.004                # 주기 요동 비율
    shimmer: float = 0.04
    passaggio: list = field(default_factory=list)   # [(f0_hz, 세기, 아래, 위), ...]
    register_stats: dict = field(default_factory=dict)

    # --- 소스 스펙트럼 ------------------------------------------------------
    tilt: float = 0.0                    # dB/oct, 1 kHz 기준. 양수면 고역이 산다.

    # --- 성도 규모 ----------------------------------------------------------
    # 성도 길이 [cm]. 남성 16.9~18, 여성 14.1~14.5 (문헌 평균).
    # 도파관 모드의 단면 수와 상위 포먼트 간격을 여기서 정한다.
    tract_length_cm: float = 17.5
    vowel_set: str = "male"              # "male" | "female" — 모음 포먼트 표 선택

    # --- 성도 ---------------------------------------------------------------
    # 나이퀴스트(12 kHz)까지 대략 1 kHz 당 1 개. 개수가 모자라면 아래
    # `extend_formants` 가 채운다 — 마지막 값을 반복하면 극이 겹쳐서 캐스케이드
    # 이득이 폭발한다(같은 주파수에 4 중극이면 Q^4).
    formants: list = field(default_factory=lambda: [730., 1090., 2440., 3400.,
                                                    4500., 5400., 6300., 7200.,
                                                    8200., 9200., 10200., 11200.])
    bandwidths: list = field(default_factory=lambda: [60., 90., 120., 160.,
                                                      200., 240., 280., 320.,
                                                      380., 450., 530., 620.])

    # --- 치찰음 지문 --------------------------------------------------------
    # 앞공동은 짧고 슬릿·치열의 손실이 커서 Q 가 낮다(대역폭 1000~1500 Hz).
    # 800 Hz 로 좁게 두면 -10 dB 폭이 3.0 kHz 밖에 안 되어 실제 /s/(4 kHz 이상)보다
    # 좁은 단봉이 되고, 넓은 '쉬~' 가 아니라 좁은 '삐~' 에 가깝게 들린다.
    sibilant: dict = field(default_factory=lambda: {
        "pole_f": 6600.0, "pole_bw": 1200.0, "zero_f": 2900.0, "zero_bw": 900.0,
        "tilt": 0.5})
    sibilant_moments: dict = field(default_factory=dict)

    # --- 위상차 -------------------------------------------------------------
    dispersion: dict = field(default_factory=lambda: {"freq": [], "radius": []})

    # --- 난류 ---------------------------------------------------------------
    # 난류 시간변조 깊이. 0.35 는 너무 커서 /s/ 가 기름에 튀기는 듯한 '지글거림'이
    # 된다(측정: 3~30 Hz 변조지수 0.043 -> 0.088, 즉 잡음 자체의 요동 대비 2배).
    # 0.12 면 '살아 있는 난류' 느낌은 남으면서 지글거림이 안 들린다.
    # 마찰음은 원래 꽤 정상적(steady)이고, 거친 느낌이 필요한 건 기식/성대프라이 쪽이다.
    roughness: float = 0.12              # 난류 시간변조 깊이
    breathiness: float = 0.15            # 유성 구간 기식 노이즈 세기

    # 이상와(piriform fossa) 반공진 [Hz]. 성도의 곁가지라 화자 고정값이다.
    # 4~5 kHz 에 스펙트럼 골을 만든다 (Dang & Honda 1997).
    piriform: dict = field(default_factory=lambda: {"freq": 4500.0, "bw": 600.0})

    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ 프리셋
    @staticmethod
    def female(name: str = "female") -> "VoiceProfile":
        """여성 화자 기본 프로파일.

        * 성도 14.1 cm (남성 17.5 의 0.81 배) -> 포먼트가 전체적으로 올라간다.
          상위 포먼트 간격 = c/2L = 35000/28.2 = 1241 Hz (남성은 1000 Hz).
        * F0 중앙값 200 Hz. 하모닉이 성겨서 F1 을 제대로 못 짚는 문제가 생기므로
          F1 대역폭을 남성보다 넓게 잡는다(아래 §소스-성도 상호작용 참고).
        * 소스: Rd 1.25 (여성이 조금 더 기식적), tilt +3 dB/oct 로 고역을 되살린다.
          tilt 없이 Rd 만 올리면 H1 만 커져 '저역에서 F0 가 포먼트보다 튀는' 소리가 된다.
        * /s/ 앞공동이 짧아 치찰음 극이 높다 (6.6 kHz -> 8.0 kHz).
        """
        spacing = 1241.0                       # c / 2L = 35000 / 28.2
        f = [850., 1220., 2810.]
        while f[-1] + spacing < 11500.0:       # 나이퀴스트 위에는 극을 두지 않는다
            f.append(f[-1] + spacing)
        return VoiceProfile(
            name=name, f0_median=200.0, f0_low=150.0, f0_high=320.0,
            rd_median=1.25, rd_low=0.8, rd_high=2.0, tilt=3.0,
            jitter=0.005, shimmer=0.045,
            tract_length_cm=14.1, vowel_set="female",
            formants=f,
            bandwidths=[80., 110., 140., 180., 220., 270., 310., 350.,
                        410., 480.][:len(f)],
            # 앞공동이 짧아 극이 남성보다 높지만, 8 kHz 로 두면 에너지의 70% 가
            # 8~12 kHz 로 몰려 '얇고 바람 새는' 소리가 된다(실측). 7.4 kHz + 넓은 Q.
            sibilant={"pole_f": 7400.0, "pole_bw": 1400.0, "zero_f": 3300.0,
                      "zero_bw": 900.0, "tilt": 0.5},
            piriform={"freq": 4700.0, "bw": 650.0},
            roughness=0.10, breathiness=0.20)

    def n_formants(self, sample_rate: int | None = None) -> int:
        """이 성도 길이에서 나이퀴스트 아래에 실제로 존재하는 극의 개수.

        포먼트 간격은 c/2L 이다 — 남성 17.5 cm 면 1000 Hz(12 kHz 안에 12 개),
        여성 14.1 cm 면 1241 Hz(9~10 개). 여성 성도에 12 개를 억지로 끼우면
        나이퀴스트 위에 극이 생겨 응답이 접히고 고역이 폭발한다(실측: 스펙트럼
        무게중심 1.3 kHz -> 10.5 kHz).
        """
        sr = sample_rate or self.sample_rate
        spacing = 35000.0 / (2.0 * self.tract_length_cm)
        return max(4, int((sr / 2) // spacing))

    def n_tract_sections(self, sample_rate: int | None = None) -> int:
        """도파관 모드의 단면 수 (단면당 왕복지연 = 1 샘플)."""
        from .config import sections_for
        return sections_for(sample_rate or self.sample_rate, self.tract_length_cm)

    # ------------------------------------------------------------------ 입출력
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "VoiceProfile":
        with open(path, encoding="utf-8") as f:
            return VoiceProfile(**json.load(f))

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "VoiceProfile":
        known = {k: v for k, v in d.items() if k in VoiceProfile.__annotations__}
        return VoiceProfile(**known)

    # -------------------------------------------------------------- 합성 보조
    def sibilant_params(self, shape, mix: float = 1.0, roughness: float | None = None,
                        device=None, dtype=torch.float32):
        from .dsp.sibilant import SibilantParams
        s = self.sibilant
        return SibilantParams.constant(
            shape, s["pole_f"], s["pole_bw"], s["zero_f"], s["zero_bw"], s["tilt"],
            mix, self.roughness if roughness is None else roughness,
            device=device, dtype=dtype)

    def dispersion_tensors(self, batch: int, n_frames: int, device=None,
                           dtype=torch.float32):
        """(disp_freq, disp_radius) 또는 (None, None)."""
        d = self.dispersion
        if not d.get("freq"):
            return None, None
        f = torch.tensor(d["freq"], device=device, dtype=dtype)
        r = torch.tensor(d["radius"], device=device, dtype=dtype)
        shape = (batch, n_frames, len(f))
        return f.expand(shape).contiguous(), r.expand(shape).contiguous()

    def formant_tensor(self, batch: int, n_frames: int, n_formants: int,
                       device=None, dtype=torch.float32):
        f = extend_formants(self.formants, n_formants)
        return torch.tensor(f[:n_formants], device=device, dtype=dtype
                            ).view(1, 1, -1).expand(batch, n_frames, n_formants
                                                    ).contiguous()

    def bandwidth_tensor(self, batch: int, n_frames: int, n_formants: int,
                         device=None, dtype=torch.float32):
        b = list(self.bandwidths)
        while len(b) < n_formants:
            b.append(b[-1] * 1.2)
        return torch.tensor(b[:n_formants], device=device, dtype=dtype
                            ).view(1, 1, -1).expand(batch, n_frames, n_formants
                                                    ).contiguous()

    def register_for(self, f0_hz: float) -> str:
        """이 F0 가 파사지오 위인지 아래인지 (해석용)."""
        if not self.passaggio:
            return "modal"
        p = sorted(float(r[0]) for r in self.passaggio)
        n = sum(1 for v in p if f0_hz > v)
        return ["chest", "mixed", "head", "falsetto"][min(n, 3)]
