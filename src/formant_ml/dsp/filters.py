"""공명/반공명/올패스/방사 필터의 미분가능 주파수응답.

모든 함수는 프레임별 파라미터 (B, T, K)를 받아 복소응답 (B, T, n_freq)을 돌려준다.
극점 반지름을 r = exp(-pi*BW/fs) < 1 로 두므로 설계상 항상 안정하다.
"""
from __future__ import annotations

import math

import torch

from .core import TWO_PI, freq_grid


def _z_powers(n_freq: int, sample_rate: float, device, dtype):
    """z^-1, z^-2 를 rfft 격자 위에서 평가."""
    f = freq_grid(n_freq, sample_rate, device=device, dtype=dtype)
    w = TWO_PI * f / sample_rate                       # (n_freq,)
    z1 = torch.exp(-1j * w.to(torch.float32))          # e^{-jw}
    return z1, z1 * z1


def _pole_pair(freq, bw, sample_rate, n_freq):
    """공명 극점쌍의 분모 D(w)와 DC 정규화 상수.

    Klatt(1980) 관례대로 DC 이득이 1이 되도록 정규화한다(A = 1 - 2r cos(t) + r^2).
    캐스케이드로 쌓아도 전체 이득이 발산/소멸하지 않으며, 공진점에서는 Q 만큼
    솟아오르는 '진짜 포먼트'가 된다. (피크 정규화로 하면 대역통과 필터들의 곱이
    되어 캐스케이드 전체가 0 으로 죽는다.)
    """
    r = torch.exp(-math.pi * bw / sample_rate)         # (B, T, K)
    theta = TWO_PI * freq / sample_rate
    z1, z2 = _z_powers(n_freq, sample_rate, freq.device, freq.dtype)
    b1 = (-2.0 * r * torch.cos(theta)).unsqueeze(-1)   # (B, T, K, 1)
    b2 = (r * r).unsqueeze(-1)
    D = 1.0 + b1 * z1 + b2 * z2                        # (B, T, K, n_freq) complex
    Ddc = 1.0 + b1 + b2                                # z = 1 (DC)
    return D, Ddc


def resonator_response(freq, bw, gain, sample_rate: float, n_freq: int) -> torch.Tensor:
    """포먼트 공명기 캐스케이드. freq/bw/gain: (B, T, K) -> (B, T, n_freq)."""
    D, Ddc = _pole_pair(freq, bw, sample_rate, n_freq)
    return (Ddc / D) * gain.unsqueeze(-1).to(D.dtype)


def resonator_stage_responses(freq, bw, gain, sample_rate: float, n_freq: int):
    """단(stage)별 응답을 그대로 반환. (B, T, K, n_freq)

    노이즈를 성도 중간(협착 지점)에서 주입하기 위해 필요하다.
    """
    D, Ddc = _pole_pair(freq, bw, sample_rate, n_freq)
    return (Ddc / D) * gain.unsqueeze(-1).to(D.dtype)


def gated_cascade_response(freq, bw, gain, weight, sample_rate: float,
                           n_freq: int, normalize: bool = True) -> torch.Tensor:
    """단별 가중치 w_i 로 부분 적용한 캐스케이드: H = exp(Σ w_i · log H_i).

    노이즈를 성도 중간(협착 지점)에서 주입하려면 '앞쪽 단은 통과, 뒤쪽 단은
    우회' 를 부드럽게 보간해야 한다. 소박한 방법인
    ``Π (1 − w_i + w_i·H_i)`` 는 **쓰면 안 된다**: w 가 중간값일 때 감쇠하는 단이
    1 로 밀려 올라가 캐스케이드의 정규화가 깨진다. 12 단이 각각 2~3 배씩만 되어도
    곱이 10^5 이 되고, 실제로 무음->마찰음 전이에서 출력이 폭발했다.

    로그 영역에서 보간하면 w=0 에서 정확히 1, w=1 에서 정확히 H_i 이고
    그 사이에서도 이득이 단조롭게 이어진다. 극이 단위원 안에 있으므로
    (r<1, D(1)>0, D(−1)>0) ∠D 의 주치는 연속이라 복소 로그의 분지 문제도 없다.

    `normalize=True` 는 로그크기의 **최댓값**을 0 으로 맞춘다(peak 이득 1, 즉 이
    필터는 절대 증폭하지 않는다).
    이게 필요한 이유: Klatt 관례의 DC 정규화에서 **고역 극 하나는 나이퀴스트에서
    이득이 1 을 훨씬 넘는다**(F=11.2 kHz, BW=620 이면 27 배). 캐스케이드 전체에서는
    앞쪽 저역 극들이 만드는 저역통과가 그걸 눌러 주는데, 노이즈 경로처럼 앞쪽 단을
    우회하면 그 억제가 사라져 나이퀴스트 부근 이득이 10^5 까지 뛴다.
    부분 캐스케이드에는 DC 정규화가 맞는 규약이 아니다.
    노이즈 경로의 절대 레벨은 어차피 `noise_bands` 가 정하므로, 여기서는 모양만
    남기고 전체 이득을 빼는 것이 맞다. (포먼트 캐스케이드가 근사이기 때문에 생기는
    문제다. 도파관 모드에서는 협착 하류 면적함수로 정확히 계산한다.)
    """
    r = torch.exp(-math.pi * bw / sample_rate)
    theta = TWO_PI * freq / sample_rate
    z1, z2 = _z_powers(n_freq, sample_rate, freq.device, freq.dtype)
    b1 = (-2.0 * r * torch.cos(theta)).unsqueeze(-1)
    b2 = (r * r).unsqueeze(-1)
    D = 1.0 + b1 * z1 + b2 * z2                        # (B, T, K, n_freq)
    Ddc = (1.0 + b1 + b2).abs().clamp_min(1e-9)
    w = weight.unsqueeze(-1)                           # (B, T, K, 1)
    log_mag = torch.log(Ddc) - 0.5 * torch.log(
        (D.real ** 2 + D.imag ** 2).clamp_min(1e-20))
    log_mag = log_mag + torch.log(gain.unsqueeze(-1).clamp_min(1e-9))
    phase = -torch.angle(D)
    acc_mag = (w * log_mag).sum(dim=2)
    acc_ph = (w * phase).sum(dim=2)
    if normalize:
        # 단 하나를 peak 정규화하면 캐스케이드가 죽지만(모듈 상단 주석 참고),
        # *곱 전체*를 한 번 정규화하는 것은 안전하다: 모양은 그대로이고 프레임마다
        # 상수 배만 빠진다. 노이즈 경로의 절대 레벨은 noise_bands 가 정한다.
        acc_mag = acc_mag - acc_mag.amax(dim=-1, keepdim=True)
    return torch.polar(torch.exp(acc_mag.clamp(-60.0, 60.0)), acc_ph)


def higher_pole_correction(sample_rate: float, n_freq: int,
                           n_poles: int = 2, spacing: float = 1000.0,
                           device=None, dtype=None) -> torch.Tensor:
    """Fant 의 **고차극 보정** — 나이퀴스트 위의 극들이 남기는 상승 셸프.

    캐스케이드는 K 개의 극만 담는다. 실제 성도에는 그 위로도 c/2L 간격으로 극이
    계속 서 있고, 각각이 모델링 대역 안쪽까지 완만한 상승 이득을 남긴다. 그걸
    빼면 최상단 극 위가 통째로 무너진다 — DC 정규화 극은 자기 중심 위로
    -12 dB/oct 씩 떨어지므로, 12 개를 지나면 나이퀴스트 근처가 절벽이 된다.
    측정(캐스케이드만, '라' 전체 발화): 9.5~12 kHz 가 -15.6 dB.

    이 레포는 그동안 이 자리를 **가짜 극**으로 때우고 있었다. 추적기가 극을
    7~8.5 kHz 에 몰아 넣고 빈 슬롯을 좁은 대역폭으로 되살린 결과가 우연히
    비슷한 레벨을 냈을 뿐이고, 그 대가가 사용자가 지적한 "지글거리는 고주파"
    였다. 몰림을 고치면(`analysis.track._assign`) 이 구멍이 드러난다.

    보정은 아날로그 극쌍의 크기응답을 그대로 쓴다(위상은 건드리지 않는다 —
    나이퀴스트 위의 극이라 대역 안에서는 거의 선형이다):

        |H_n(f)| = Fn^2 / sqrt((Fn^2 - f^2)^2 + (Bn f)^2)

    Fn 은 나이퀴스트 위로 `spacing` 간격, Bn 은 Fant 경험식이다. n_poles=2 를
    쓴다 — 24 kHz 에서 11 kHz +17.6 / 8 kHz +7.7 / 5 kHz +2.6 / 1 kHz +0.1 dB 로,
    측정된 구멍(+16 / +7 / ~0)과 모양이 맞는다. 3 개면 11 kHz 에서 +24 dB 로
    넘친다.

    **`noise_bw_scale` 로 이 극들까지 넓혀 보았고, 안 된다.** 난류 경로의
    추가 감쇠를 고차극에도 걸면(bn x6) 셸프가 11 kHz 에서 -3 dB 짜리
    감쇠기로 뒤집힌다 — Q 가 원래 낮아서 조금만 넓혀도 이득이 1 아래로
    떨어진다. 속삭임 울림 지표가 0.495 -> 0.558 로 더 나빠졌다.
    """
    nyq = sample_rate / 2.0
    f = torch.linspace(0.0, nyq, n_freq, device=device, dtype=dtype)
    g = torch.ones_like(f)
    for k in range(1, n_poles + 1):
        fn = nyq + (k - 0.5) * spacing
        bn = 50.0 + 20.0 * (fn / 1000.0) ** 2 + 10.0 * (fn / 1000.0)
        g = g * (fn ** 2) / torch.sqrt((fn ** 2 - f ** 2) ** 2 + (bn * f) ** 2)
    return g


def antiresonator_response(freq, bw, sample_rate: float, n_freq: int) -> torch.Tensor:
    """반공명(영점) 캐스케이드: 비음의 안티포먼트, 마찰음의 스펙트럼 골."""
    D, Ddc = _pole_pair(freq, bw, sample_rate, n_freq)
    return (D / Ddc).prod(dim=2)


def allpass_response(freq, radius, sample_rate: float, n_freq: int) -> torch.Tensor:
    """2차 올패스 캐스케이드: 크기응답은 1, 군지연(위상차)만 바꾼다.

    H(z) = (r^2 - 2r cos(t) z^-1 + z^-2) / (1 - 2r cos(t) z^-1 + r^2 z^-2)
    """
    r = radius.clamp(0.0, 0.995)
    theta = TWO_PI * freq / sample_rate
    z1, z2 = _z_powers(n_freq, sample_rate, freq.device, freq.dtype)
    a1 = (-2.0 * r * torch.cos(theta)).unsqueeze(-1)
    a2 = (r * r).unsqueeze(-1)
    num = a2 + a1 * z1 + z2
    den = 1.0 + a1 * z1 + a2 * z2
    return (num / den).prod(dim=2)


def one_pole_tilt(tilt_db: torch.Tensor, sample_rate: float, n_freq: int) -> torch.Tensor:
    """스펙트럼 기울기(성문 소스의 spectral tilt). tilt_db: (B, T) 나이퀴스트 감쇠량."""
    a = torch.exp(-torch.nn.functional.softplus(tilt_db) * 0.05).unsqueeze(-1)
    z1, _ = _z_powers(n_freq, sample_rate, tilt_db.device, tilt_db.dtype)
    return ((1.0 - a) / (1.0 - a * z1))


def lip_radiation_response(sample_rate: float, n_freq: int, alpha: float = 0.98,
                           device=None) -> torch.Tensor:
    """입술 방사 = 미분기 근사 H(z) = 1 - alpha z^-1. (1, 1, n_freq)

    주의: LF 모델은 '유량의 미분'을 직접 생성하므로 기본 체인에서는 꺼 둔다.
    """
    z1, _ = _z_powers(n_freq, sample_rate, device, torch.float32)
    return (1.0 - alpha * z1).reshape(1, 1, -1)


def min_phase_from_log_mag(log_mag: torch.Tensor) -> torch.Tensor:
    """log|H| (..., n_freq) -> **최소위상** 복소응답 (..., n_freq).

    실수 캡스트럼의 인과 부분만 남기는 표준 힐베르트 재구성이다.

    **왜 크기만 주면 안 되는가.** 크기만 실수로 돌려주면 그건 영위상 필터이고,
    영위상 필터의 임펄스응답은 t=0 대칭이라 **비인과적**이다. `response_to_ir`
    이 ir_size//2 만큼 롤하고 `ltv_filter` 가 그 지연을 보정하므로, 대칭인 절반이
    그대로 **프리에코**(여기가 되기 전에 소리가 나는 것)로 남는다.

    측정(치찰음 경로, 정점보다 앞에 오는 임펄스응답 에너지 비율):
        극·영점  0.0 %      앞니 공진기 0.0 %     (둘 다 최소위상)
        스커트  21.3 %      tilt       46.7 %    (둘 다 영위상이었다)
        노이즈 경로 전체 26.0 %, -20 dB 기준 0.16 ms 선행

    **정상 잡음에서는 위상이 관측 불가능하다**(시간 이동에 대해 통계가 불변이다 —
    실제로 군지연 추정기가 이론값 1.06 ms 짜리 공진기에서도 0 을 돌려준다).
    그래서 이 결함은 마찰음의 **정상부가 아니라 개시·해제·변조** 에서만 드러난다.

    .. warning::
       `tilt_response` / `skirt_response` 는 **아직 이걸 안 쓴다**. 위상은 곱셈
       에서는 크기에 영향이 없지만 **병렬 덧셈**(치찰음 필터의 `floor_db`,
       `_oral_leak`, 앞니 병렬 혼합)에서는 영향이 있고, 프로파일의 `floor_db` 와
       포먼트 추출기가 지금의 혼합위상 모델에 맞춰 적합돼 있다. 바꿨더니
       분석/합성 왕복이 깨졌다(pole_f 3500 -> 4266). 실측 대비 개선은 무시할
       수준이었다(무게중심 아치 7862->8549->8010 vs 7513->8476->7468).
       바꾸려면 `floor_db` 재적합과 추출기 수정이 같이 가야 한다. HANDOFF §6.12.
    """
    n_freq = log_mag.shape[-1]
    n_fft = 2 * (n_freq - 1)
    cep = torch.fft.irfft(log_mag.to(torch.complex64), n_fft)
    w = torch.zeros(n_fft, device=log_mag.device, dtype=log_mag.dtype)
    w[0] = 1.0
    w[1: n_fft // 2] = 2.0
    w[n_fft // 2] = 1.0
    return torch.exp(torch.fft.rfft(cep * w, n_fft))


def bands_to_response(band_gains: torch.Tensor, n_freq: int,
                      min_phase: bool = True) -> torch.Tensor:
    """대역 게인 (B, T, n_bands) -> 매끄러운 복소응답 (B, T, n_freq).

    min_phase=True 면 힐베르트 변환으로 최소위상화하여 프리링잉(pre-echo)을 없앤다.
    """
    b, t, n_bands = band_gains.shape
    mag = torch.nn.functional.interpolate(
        band_gains.reshape(b * t, 1, n_bands), size=n_freq,
        mode="linear", align_corners=True
    ).reshape(b, t, n_freq).clamp_min(1e-7)
    if not min_phase:
        return mag.to(torch.complex64)
    return min_phase_from_log_mag(torch.log(mag))


def tilt_response(tilt_db_per_oct: torch.Tensor, sample_rate: float, n_freq: int,
                  pivot_hz: float = 1000.0, lo_db: float = -90.0,
                  hi_db: float = 40.0) -> torch.Tensor:
    """스펙트럼 기울기 (실수 이득, 영위상). tilt: (B, T, 1) dB/oct, 1 kHz 기준.

    `one_pole_tilt` 은 한 극점의 고정된 -6 dB/oct 모양만 낼 수 있어서 6~12 kHz 를
    따로 올리거나 내릴 수 없다. 여기서는 log 주파수에 대한 직선(=옥타브당 dB)이라
    고역만 정확히 원하는 만큼 조절할 수 있다. (고역 부족 문제의 직접 손잡이)

    **감쇠 쪽 클램프는 넉넉해야 한다.** -40 dB 로 두면 44.1 kHz 에서 문제가 된다:
    1 kHz 피벗에서 22 kHz 는 4.46 옥타브라, 기울기가 -9 dB/oct 보다 가파르면
    나이퀴스트에 닿기 전에 클램프에 걸려 **고역이 -40 dB 짜리 선반**이 된다.
    치찰음 기울기는 -5 ~ -12 dB/oct(+ 소스 기울기)라 거의 항상 걸렸고, 그래서
    18~22 kHz 가 실측 대비 12~23 dB 과다했다. 더 나쁜 건 기울기가 가팔라질수록
    선반이 넓어져서 **어두워야 할 구간이 오히려 상대적으로 밝아진다**는 것이다
    (측정: 긴 /s/ 의 11~18 kHz 가 가운데보다 양 끝에서 8 dB 높았다 — 실측은
    반대로 가운데가 4~7 dB 높다).

    부스트 쪽(hi_db)은 그대로 둔다. 그쪽은 이득이 폭발하는 위험한 방향이다.
    """
    f = freq_grid(n_freq, sample_rate, device=tilt_db_per_oct.device,
                  dtype=tilt_db_per_oct.dtype).clamp_min(20.0)
    oct_ = torch.log2(f / pivot_hz)                        # (n_freq,)
    db = tilt_db_per_oct * oct_.view(1, 1, -1)             # (B, T, n_freq)
    # **영위상(실수)이다.** 최소위상이 물리적으로 맞지만 지금은 못 바꾼다 —
    # 이유와 측정은 `min_phase_from_log_mag` 주석과 HANDOFF §6.12 에 있다.
    return (10.0 ** (db.clamp(lo_db, hi_db) / 20.0)).to(torch.complex64)


def pole_zero_response(pole_f, pole_bw, zero_f, zero_bw, sample_rate: float,
                       n_freq: int) -> torch.Tensor:
    """극-영점 한 쌍의 응답 (B, T, n_freq). 마찰음 스펙트럼의 최소 모형.

    마찰음의 스펙트럼은 '앞공동 공진(극) + 뒤공동에 의한 골(영점)' 로 요약된다.
    이 두 쌍이 /s/ 와 /ʃ/ 를 가르고, 사람마다 미세하게 다른 그 위치가 곧
    화자의 치찰음 지문이다.
    """
    Dp, Dpdc = _pole_pair(pole_f, pole_bw, sample_rate, n_freq)
    Dz, Dzdc = _pole_pair(zero_f, zero_bw, sample_rate, n_freq)
    return ((Dpdc / Dp) * (Dz / Dzdc)).prod(dim=2)


def skirt_response(peak_f, slope_lo, slope_hi, sample_rate: float, n_freq: int,
                   floor_db: float = -60.0) -> torch.Tensor:
    """봉우리를 중심으로 로그주파수에 대해 **직선 두 개**인 스커트 (B, T, n_freq).

    극 하나로는 삼각형이 안 나온다. 2차 공명의 크기응답은 로렌치안이라 봉우리가
    둥글고, 대역폭을 넓혀 음조를 없애면 그냥 **둥근 돔**이 된다(측정: 사람 /s/ 의
    저역 스커트가 +20~40 dB/oct 인데 우리는 +7.7).

    반면 실제 마찰음 스펙트럼은 봉우리 아래로 가파르게 떨어지는 삼각형에 가깝다.
    그 가파름은 공진의 Q 가 아니라 **협착-앞공동 계의 고역통과 성질 + 소스
    스펙트럼 + 방사**가 겹쳐 만든 것이라, 공진을 좁히지 않고도 만들 수 있다.
    여기서는 로그주파수의 조각별 직선으로 직접 준다 — 공진이 없으므로 울리지
    않고, 기울기를 원하는 대로 세울 수 있다.

        dB(f) = slope_lo · log2(f/f_peak)   (f < f_peak, slope_lo > 0 이면 상승)
        dB(f) = slope_hi · log2(f/f_peak)   (f > f_peak, slope_hi < 0 이면 하강)
    """
    f = freq_grid(n_freq, sample_rate, device=peak_f.device,
                  dtype=peak_f.dtype).clamp_min(20.0).view(1, 1, -1)
    o = torch.log2(f / peak_f.clamp_min(100.0))
    db = torch.where(o < 0, slope_lo * o, slope_hi * o).clamp_min(floor_db)
    # tilt_response 와 같은 이유로 영위상이다(HANDOFF §6.12).
    return (10.0 ** (db / 20.0)).to(torch.complex64)


def rms_normalize(H: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """주파수축 RMS 를 1 로 맞춘다(피크 정규화보다 그래디언트가 안정적)."""
    rms = H.abs().pow(2).mean(dim=-1, keepdim=True).clamp_min(eps).sqrt()
    return H / rms.to(H.dtype)


def resonator_magnitude_db(freq_hz: torch.Tensor, f_form: torch.Tensor,
                           bw_form: torch.Tensor, sample_rate: float) -> torch.Tensor:
    """포먼트 캐스케이드의 크기응답(dB)을 임의 주파수에서 평가 (…, K).

    H = D(1)/D(z) (DC 이득 1). 하모닉 크기에서 성도의 기여를 빼내는 데 쓴다
    (H1-H2 를 모음 종류에 무관하게 만드는 보정).
    freq_hz: (…, K), f_form/bw_form: (…, S) -> 반환 (…, K)
    """
    w = (TWO_PI * freq_hz / sample_rate).unsqueeze(-1)
    r = torch.exp(-math.pi * bw_form / sample_rate).unsqueeze(-2)
    th = (TWO_PI * f_form / sample_rate).unsqueeze(-2)
    b1 = -2.0 * r * torch.cos(th)
    b2 = r * r
    c1, s1 = torch.cos(w), -torch.sin(w)
    c2, s2 = torch.cos(2 * w), -torch.sin(2 * w)
    dr = 1.0 + b1 * c1 + b2 * c2
    di = b1 * s1 + b2 * s2
    mag2 = (dr * dr + di * di).clamp_min(1e-12)
    dc = (1.0 + b1 + b2).abs().clamp_min(1e-9)
    return (20.0 * torch.log10(dc) - 10.0 * torch.log10(mag2)).sum(-1)
