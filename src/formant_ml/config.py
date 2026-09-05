"""전역 설정: 샘플레이트/프레임률/파라미터 범위."""
from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    sample_rate: int = 24000
    hop_size: int = 240          # 10 ms 프레임
    win_size: int = 1024
    n_fft: int = 1024
    n_mels: int = 80
    fmin: float = 40.0
    fmax: float = 12000.0

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_size


@dataclass
class SourceConfig:
    """성문(글로탈) 소스 설정."""
    f0_min: float = 55.0
    f0_max: float = 880.0
    # 가산합성 상한(나이퀴스트 위에서는 자동 마스킹).
    # 24 kHz(나이퀴스트 12 kHz)에서 저음 화자 F0=50 Hz 까지 12 kHz 를 채우려면
    # 240 개가 필요하다. 180 이면 F0<67 Hz 에서 고역이 통째로 비어 버린다.
    n_harmonics: int = 240
    n_rd_tables: int = 16         # LF 파형 사전(Rd 그리드) 크기
    rd_min: float = 0.3           # pressed(긴장)
    rd_max: float = 2.7           # breathy(기식)
    table_size: int = 2048


@dataclass
class FilterConfig:
    """성도(공명) 필터 설정."""
    # 경험칙: 성도 17.5 cm 는 대략 1 kHz 당 포먼트 1 개. 나이퀴스트가 12 kHz 이므로
    # 12 개가 맞다. 6 개로 덮으면 최상단 극 위에서 캐스케이드가 극당 -12 dB/oct 씩
    # 겹쳐 떨어져(6 극이면 -72 dB/oct) 5 kHz 위가 통째로 죽는다 — 이전 샘플이
    # 먹먹했던 구조적 원인이다.
    n_formants: int = 12
    n_antiformants: int = 2       # 비음/마찰음용 반공명(zero)
    f_min: float = 150.0
    f_max: float = 11500.0
    bw_min: float = 30.0
    bw_max: float = 800.0
    # **빈 포먼트 슬롯을 중립으로 만드는 대역폭.**
    # 극 하나의 응답 Ddc/D 는 대역폭이 커질수록 1 에 가까워진다
    # (r = exp(-pi*BW/fs) -> 0). 즉 "아주 넓은 극" = "극이 없음" 이다.
    # bw_max(800) 로는 어림도 없다: 24 kHz 에서 r=0.90, 10.8 kHz 에 놓으면
    # **Q=13.5 짜리 진짜 공명**이다. 실제로 `track._fill` 이 못 찾은 슬롯에
    # 놓던 '무해한' 극 2 개가 7~11 kHz 를 +25.6 dB 올리고 캐스케이드의 최대점을
    # 1195 Hz(F2 근처) 에서 8227 Hz 로 옮겨 놨다 — 무음 구간에서 8 kHz
    # 휘파람으로 들렸고, 전체 발화에서 저역이 15 dB 묻혔다.
    # 20 kHz 면 r=0.073 이라 응답이 1 에서 ±0.6 dB 안에 든다.
    bw_neutral: float = 80000.0
    #: 나이퀴스트 위 고차극 보정에 쓸 극 개수 (dsp.filters.higher_pole_correction).
    higher_poles: int = 2
    ir_size: int = 512            # LTV 필터 임펄스응답 길이
    n_allpass: int = 4            # 성도 위상(군지연) 정형용 올패스 단수
    n_dispersion: int = 3         # 성문 소스 하모닉 위상차(올패스) 단수
    # Kelly-Lochbaum 단면 수 N = 2 * L_tract * fs / c.
    # 24 kHz, c=350 m/s, L=17.5 cm -> N=24 (균일관 공진이 정확히 500/1500/2500 Hz).
    n_tract_sections: int = 24
    tract_length_cm: float = 17.5
    sound_speed_cm_s: float = 35000.0


@dataclass
class NoiseConfig:
    n_bands: int = 40             # 노이즈 대역 게인 개수
    # 난류 **소스**의 고역 컷오프(TurbulenceSource 의 학습 사전 초기값).
    # 협착부 제트의 에디에는 특징적인 크기가 있어 그 위로 스펙트럼이 떨어진다.
    #
    # 실측 긴 /s/ 는 12->18 kHz 에서 26.6 dB(-46 dB/oct) 떨어지는데 여기 값은
    # -6 dB/oct 라 3.6 dB 밖에 안 떨어진다. 그래서 14 kHz / 25 dB/oct 로 바꿔
    # 봤는데 **무게중심 아치가 무너졌다**(2823 Hz -> 279 Hz): 상단을 고정 컷오프로
    # 자르면 제트 속도에 따른 소스 기울기 변화가 갈 곳이 없어진다. 실측의 상단
    # 절벽은 소스의 정적 컷오프가 아니라 다른 데서 온다 — 아직 못 찾았다.
    # HANDOFF §6.10. 그래서 원래 값으로 되돌렸고, 설정으로만 빼 두었다.
    # **노이즈 경로의 입술 방사** H(z)=1-alpha·z^-1 (0 이면 끔).
    #
    # 하모닉 경로는 LF 성문 모델이 유량의 **미분**을 직접 내므로 방사가 이미
    # 소스에 들어 있다(그래서 `radiation` 버퍼가 alpha=0 이다). 난류는 다르다 —
    # 협착부의 유량/압력 요동이지 그 미분이 아니라서, 입에서 방사될 때 +6 dB/oct
    # 를 한 번 더 받아야 한다. 그게 빠져 있으면 **성도가 저역에서 공진하는 만큼
    # 그대로 저역이 부각된다**. 실제 숨소리는 성도가 저역에서 공진해도 저역이
    # 두드러지지 않는데, 그건 소스에도 방사에도 저역이 없기 때문이다.
    noise_radiation_alpha: float = 0.0
    source_corner_hz: float = 5000.0
    source_slope_db_oct: float = 6.0
    # 성문 개방기에 동기화된 진폭변조(기식음) 세기 범위
    am_depth_max: float = 1.0
    roughness_max: float = 1.0    # 난류의 시간 변조(정상 히스가 되는 것을 막는다)


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    filt: FilterConfig = field(default_factory=FilterConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)


DEFAULT = Config()


def sections_for(sample_rate: int, tract_length_cm: float = 17.5,
                 sound_speed_cm_s: float = 35000.0) -> int:
    """샘플레이트에 맞는 도파관 단면 수(단면당 왕복지연 = 1샘플)."""
    return max(2, round(2.0 * tract_length_cm * sample_rate / sound_speed_cm_s))
