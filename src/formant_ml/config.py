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
    ir_size: int = 512            # LTV 필터 임펄스응답 길이
    # 포먼트가 움직이는 부분공간의 차원. 0 이면 K 개가 서로 독립(예전 동작).
    # 포먼트는 (턱, 혀몸통, 혀끝, 입술) 정도의 소수 좌표로 결정되므로 독립일 수 없다.
    # Story 의 면적함수 실증 직교모드에서는 2 개 모드가 분산의 97% 이상을 설명한다.
    # 여기서는 여유를 둬 4 로 잡는다 (자음 협착까지 포함하므로 모음보다 크다).
    formant_basis_dim: int = 4
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
