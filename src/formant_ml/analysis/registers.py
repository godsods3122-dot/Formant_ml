"""성대 진동 모드 분석: 개방지수, 성구(register), 파사지오, 서브하모닉.

무엇을 재는가
-------------
성대는 하나의 진동 방식만 갖지 않는다. 갑상피열근/윤상갑상근의 균형이 바뀌면
진동하는 조직의 깊이(body-cover 비율)가 바뀌고, 그 결과가

* **개방지수(open quotient)** — 성문이 한 주기 중 열려 있는 비율. 이것이 크면
  성문파가 완만해지고 스펙트럼이 급히 떨어진다. 측정 대용물이 **H1-H2**
  (첫 두 하모닉의 크기 차, dB) 이고, 우리 모델의 **Rd** 와 단조 대응한다.
* **스펙트럼 기울기** — 폐쇄가 날카로울수록 고역이 산다. 우리 모델의 `tilt`.
* **서브하모닉(SHR)** — 좌우 성대가 1:1 로 잠기지 않으면 f0/2, f0/3 성분이
  생긴다. 이중음(diplophonia)·성대 프라이·거친 소리가 여기 걸린다.

**파사지오**는 이 지표들이 특정 F0 부근에서 *급격히* 꺾이는 지점이다.
연속적으로 변하면 그냥 음역이고, 계단처럼 꺾이면 성구 전환이다. 그래서
"F0 에 대한 지표의 변화율이 국소적으로 최대인 곳"을 찾는 문제가 된다.

주의: 이 지표들은 상관량(correlate)이지 성문 직접 측정(EGG)이 아니다. 절대값보다
**같은 화자 안에서의 변화**가 훨씬 믿을 만하다.
"""
from __future__ import annotations

import math

import torch

from ..data.features import frame_signal, stft, yin_f0
from ..dsp.glottal import LFTableBank


# ------------------------------------------------------------------ 하모닉 추출
def harmonic_spectrum(x: torch.Tensor, f0: torch.Tensor, sample_rate: int = 24000,
                      hop: int = 240, n_fft: int = 2048, n_harmonics: int = 24,
                      multiple: float = 1.0, block: int = 64) -> torch.Tensor:
    """하모닉(또는 그 배수) 위치의 복소 스펙트럼 (T, K). **정확한 주파수에서** 평가.

    STFT 격자의 가장 가까운 빈을 쓰면 안 된다. 창 누설 계수가 복소수이고 그 위상이
    (빈 − 실제 주파수)에 비례하기 때문에, 반올림 오차가 하모닉마다 다른 위상 오차로
    들어온다(측정해 보면 상대위상에 1 rad 이 넘는 편향이 생긴다). 그래서 각 프레임의
    창 신호에 대해 f_k = k·f0·multiple 에서 DFT 를 직접 계산한다.

    multiple=0.5 로 부르면 반정수 하모닉(서브하모닉) 위치를 본다.
    """
    if x.dim() == 1:
        x = x[None]
    w = frame_signal(x, n_fft, hop)[0]                        # (T, n_fft)
    t = min(w.shape[0], f0.shape[-1])
    w, f0 = w[:t], f0.reshape(-1)[:t]
    win = torch.hann_window(n_fft, device=w.device, dtype=w.dtype)
    w = w * win
    k = torch.arange(1, n_harmonics + 1, dtype=w.dtype, device=w.device)
    n = torch.arange(n_fft, dtype=w.dtype, device=w.device)
    out = []
    for s0 in range(0, t, block):
        s1 = min(s0 + block, t)
        fk = f0[s0:s1, None] * k * multiple                   # (tb, K)
        ang = -2.0 * math.pi * fk[:, :, None] * n[None, None, :] / sample_rate
        basis = torch.polar(torch.ones_like(ang), ang)        # (tb, K, n_fft)
        out.append(torch.einsum("tn,tkn->tk", w[s0:s1].to(basis.dtype), basis))
    return torch.cat(out, dim=0)


def h1_h2_db(H: torch.Tensor, f0: torch.Tensor | None = None,
             formant_f=None, formant_bw=None, sample_rate: int = 24000,
             eps: float = 1e-9) -> torch.Tensor:
    """첫 두 하모닉의 크기 차 (T,) dB. 개방지수(기식성)의 표준 대용물.

    포먼트를 주면 **H1*-H2* 보정**을 한다 (Iseli & Alwan 과 같은 발상이되, 우리
    공명기 식을 그대로 쓴다). 성도 전달함수의 기여를 하모닉 위치에서 계산해 뺀다.

    **기본값은 보정 없음이다.** 보정에는 프레임별 F1/F2 가 정확해야 하는데,
    /이/ /우/ 처럼 F1 이 H2 근처인 모음에서는 F1 추정이 수십 Hz만 틀려도 보정량이
    10 dB 넘게 흔들려서 보정 안 한 것보다 나빠진다. 여기 있는 `track_f1f2` 는
    그 정확도가 안 나온다(측정: /아/ 는 좋아지지만 /이/ /우/ 는 크게 나빠졌다).
    제대로 하려면 LPC 기반 포먼트 추적기가 필요하다.

    실용적 지침: **Rd 는 지속된 열린 모음(/아/)에서 재라.** 음성 품질 연구의
    표준 관행이기도 하고, 그 조건에서는 보정 없이도 Rd 왕복 오차가 0.05 이내다
    (tests/test_voice.py::test_rd_round_trips_through_h1h2).
    /이/ /우/ 에서는 H1-H2 가 낮게 나와 pressed 쪽으로 치우친다 — 알려진 한계다.
    """
    a = 20.0 * torch.log10(H.abs().clamp_min(eps))
    h1, h2 = a[:, 0], a[:, 1]
    if formant_f is not None and f0 is not None:
        from ..dsp.filters import resonator_magnitude_db
        ff = torch.as_tensor(formant_f, dtype=torch.float32)
        bw = torch.as_tensor(formant_bw, dtype=torch.float32)
        if ff.dim() == 1:                     # 화자 평균 포먼트 (모든 프레임 공통)
            ff = ff.view(1, -1).expand(f0.shape[0], -1)
            bw = bw.view(1, -1).expand(f0.shape[0], -1)
        fk = torch.stack([f0, 2.0 * f0], dim=-1)               # (T, 2)
        corr = resonator_magnitude_db(fk, ff, bw, sample_rate)
        h1, h2 = h1 - corr[:, 0], h2 - corr[:, 1]
    return h1 - h2


def subharmonic_ratio_db(x, f0, sample_rate=24000, hop=240, eps=1e-9):
    """SHR (T,) dB: 반정수 하모닉 에너지 / 정수 하모닉 에너지.

    0 dB 에 가까우면 주기 배가(이중음/프라이), 크게 음수면 정상 1:1 진동.
    """
    Hi = harmonic_spectrum(x, f0, sample_rate, hop, n_harmonics=12, multiple=1.0)
    Hs = harmonic_spectrum(x, f0, sample_rate, hop, n_harmonics=12, multiple=0.5)
    # 반정수만 남긴다 (0.5, 1.5, 2.5 ... = multiple=0.5 의 홀수 인덱스)
    sub = Hs[:, ::2]
    ei = Hi.abs().pow(2).sum(-1).clamp_min(eps)
    es = sub.abs().pow(2).sum(-1).clamp_min(eps)
    return 10.0 * torch.log10(es / ei)


def spectral_tilt_db_per_oct(x, sample_rate=24000, hop=240, n_fft=1024,
                             fmin=300.0, fmax=8000.0, eps=1e-9) -> torch.Tensor:
    """로그주파수에 대한 로그크기의 회귀 기울기 (T,) dB/oct.

    합성기의 `tilt` 제어와 같은 단위이므로 그대로 초기값으로 쓸 수 있다.
    """
    X = stft(x if x.dim() > 1 else x[None], n_fft, hop)[0].abs().transpose(0, 1)
    f = torch.linspace(0, sample_rate / 2, X.shape[-1], device=X.device)
    m = (f >= fmin) & (f <= fmax)
    lf = torch.log2(f[m] / 1000.0)
    ly = 20.0 * torch.log10(X[:, m].clamp_min(eps))
    lf = lf - lf.mean()
    return (ly * lf).mean(-1) * lf.numel() / (lf.pow(2).sum().clamp_min(eps))


def cpp(x, sample_rate=24000, hop=240, n_fft=1024, f0_min=55.0, f0_max=500.0,
        eps=1e-9) -> torch.Tensor:
    """켑스트럼 피크 현저도 (T,) dB. 낮으면 기식/거친 소리, 높으면 또렷한 유성음."""
    X = stft(x if x.dim() > 1 else x[None], n_fft, hop)[0].abs().transpose(0, 1)
    logmag = 20.0 * torch.log10(X.clamp_min(eps))
    cep = torch.fft.irfft(logmag.to(torch.complex64), n_fft).real.abs()
    lo, hi = int(sample_rate / f0_max), min(int(sample_rate / f0_min), n_fft // 2)
    seg = cep[:, lo:hi]
    q = torch.arange(seg.shape[-1], dtype=seg.dtype, device=seg.device)
    qc = q - q.mean()
    slope = (seg * qc).mean(-1) * q.numel() / qc.pow(2).sum().clamp_min(eps)
    base = seg.mean(-1, keepdim=True) + slope[:, None] * qc
    return (seg - base).amax(-1)


def track_f1f2(x: torch.Tensor, sample_rate: int = 24000, hop: int = 240,
               n_fft: int = 1024, quefrency: int = 30, eps: float = 1e-9):
    """프레임별 F1/F2 의 거친 추정 (T,), (T,). 켑스트럼 포락선의 대역별 최댓값.

    H1*-H2* 보정에는 **프레임별** 포먼트가 필요하다. 화자 평균 포먼트를 쓰면
    /아/ 구간을 /이/ 의 평균으로 보정하게 되어 오히려 더 틀린다.
    정밀한 포먼트 추적기는 아니지만, 보정에 필요한 건 F1 이 H2 를 얼마나
    끌어올리는지 정도라 이 해상도면 충분하다.
    """
    if x.dim() == 1:
        x = x[None]
    X = stft(x, n_fft, hop)[0].abs().transpose(0, 1)          # (T, F)
    logmag = torch.log(X.clamp_min(eps))
    cep = torch.fft.irfft(logmag.to(torch.complex64), n_fft)
    cep[:, quefrency:-quefrency] = 0
    env = torch.fft.rfft(cep, n_fft).real                     # (T, F) 로그 포락선
    f = torch.linspace(0, sample_rate / 2, env.shape[-1], device=env.device)
    b1 = (f >= 180.0) & (f <= 1200.0)
    b2 = (f >= 700.0) & (f <= 3000.0)
    f1 = f[b1][env[:, b1].argmax(-1)]
    f2 = f[b2][env[:, b2].argmax(-1)]
    return f1, torch.maximum(f2, f1 + 150.0)


# ------------------------------------------------------------ H1-H2 <-> Rd 대응
_RD_TABLE: dict = {}


def rd_h1h2_table(n: int = 129, rd_min: float = 0.3, rd_max: float = 2.7):
    """LF 사전 자체에서 (Rd, H1-H2) 대응표를 만든다.

    문헌의 회귀식을 베끼는 대신 **우리 합성기가 실제로 내는** 값을 쓰기 때문에
    분석-합성이 서로 어긋나지 않는다.
    """
    key = (n, rd_min, rd_max)
    if key in _RD_TABLE:
        return _RD_TABLE[key]
    bank = LFTableBank(n_tables=n, table_size=4096, rd_min=rd_min, rd_max=rd_max,
                       n_harmonics=4)
    a = bank.spectra.abs()                                    # (n, 4)
    h1h2 = 20.0 * torch.log10(a[:, 0].clamp_min(1e-9) / a[:, 1].clamp_min(1e-9))
    rd = bank.rd_grid
    order = torch.argsort(h1h2)
    _RD_TABLE[key] = (h1h2[order].contiguous(), rd[order].contiguous())
    return _RD_TABLE[key]


def rd_from_h1h2(h1h2: torch.Tensor) -> torch.Tensor:
    """H1-H2 (dB) -> Rd. 표에 대한 단조 보간(표 밖은 양끝으로 포화)."""
    xs, ys = rd_h1h2_table()
    xs, ys = xs.to(h1h2.device), ys.to(h1h2.device)
    i = torch.searchsorted(xs, h1h2.clamp(float(xs[0]), float(xs[-1])).contiguous())
    i = i.clamp(1, len(xs) - 1)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    w = ((h1h2 - x0) / (x1 - x0).clamp_min(1e-6)).clamp(0, 1)
    return y0 + (y1 - y0) * w


# ------------------------------------------------------------------- 프레임 분석
def register_features(x: torch.Tensor, sample_rate: int = 24000, hop: int = 240,
                      formant_f=None, formant_bw=None) -> dict:
    """한 파일 -> 프레임별 성대 지표 dict (모두 (T,) 텐서).

    `formant_f/bw` 를 주면 H1-H2 에 성도 보정(H1*-H2*)이 걸린다. **프레임별**
    포먼트일 때만 켤 것 — 아래 `h1_h2_db` 의 주의사항을 읽어라.
    """
    x = x.detach()
    if x.dim() == 1:
        x = x[None]
    f0, voicing = yin_f0(x, sample_rate, hop)
    f0f = torch.where(f0[0] > 0, f0[0], torch.full_like(f0[0], 120.0))
    H = harmonic_spectrum(x, f0f, sample_rate, hop)
    t = H.shape[0]
    cut = lambda v: v[:t]                                     # noqa: E731
    h1h2 = h1_h2_db(H, cut(f0f), formant_f, formant_bw, sample_rate)
    return {
        "f0": cut(f0[0]),
        "voicing": cut(voicing[0]),
        "h1h2": h1h2,
        "rd": rd_from_h1h2(h1h2),
        "tilt": cut(spectral_tilt_db_per_oct(x, sample_rate, hop)),
        "shr": cut(subharmonic_ratio_db(x, f0f, sample_rate, hop)),
        "cpp": cut(cpp(x, sample_rate, hop)),
        "periodicity": cut(_periodicity(x, sample_rate, hop)),
    }


def _periodicity(x, sample_rate: int, hop: int) -> torch.Tensor:
    from ..models.losses import periodicity
    return periodicity(x, sample_rate, hop)[0]


def steady_voiced_mask(feat: dict, min_voicing: float = 0.6,
                       max_f0_slope: float = 0.02) -> torch.Tensor:
    """지속 유성 구간(성구 지표를 재기 좋은 프레임)의 마스크.

    글리산도/전이/마찰음이 섞인 녹음 전체의 중앙값을 그대로 쓰면 '이 화자의 Rd'
    가 아니라 '녹음 내용의 중앙값'이 나온다.
    """
    f0 = feat["f0"]
    d = torch.zeros_like(f0)
    d[1:] = (f0[1:] - f0[:-1]).abs() / f0[:-1].clamp_min(1e-3)
    return (feat["voicing"] > min_voicing) & (d < max_f0_slope) & (f0 > 0)


# --------------------------------------------------------------------- 성구/파사지오
def register_label(feat: dict, i: int) -> str:
    """한 프레임의 대략적인 성구 이름(해석용 라벨, 진단이 아니다)."""
    if float(feat["voicing"][i]) < 0.2:
        return "unvoiced"
    if float(feat["shr"][i]) > -6.0:
        return "subharmonic"          # 이중음/프라이/거친 소리
    h = float(feat["h1h2"][i])
    if h < 1.0:
        return "pressed"              # 압착 — 개방지수 작음
    if h > 8.0:
        return "breathy"              # 기식 — 개방지수 큼
    return "modal"


def passaggio_candidates(feat: dict, n_bins: int = 28, min_count: int = 6,
                         indicator: str = "h1h2", smooth: int = 3):
    """F0 축에서 성구 지표가 급변하는 지점(파사지오 후보)을 찾는다.

    반환: [(f0_hz, jump_db_per_semitone, 아래쪽 평균, 위쪽 평균), ...] 강한 순.

    방법: 유성 프레임을 반음 단위로 묶고 F0 빈마다 지표의 평균을 낸 뒤,
    이웃 빈과의 차분이 국소 최대인 곳을 고른다. 지표가 F0 를 따라 매끄럽게
    변하면 아무것도 안 나오고, 계단처럼 꺾일 때만 후보가 나온다.
    """
    v = feat["voicing"] > 0.3
    f0 = feat["f0"][v]
    y = feat[indicator][v]
    if f0.numel() < min_count * 3:
        return []
    st = 12.0 * torch.log2(f0.clamp_min(1e-3) / 55.0)          # 반음 축
    lo, hi = float(st.min()), float(st.max())
    if hi - lo < 3.0:
        return []
    edges = torch.linspace(lo, hi, n_bins + 1)
    means, centers, counts = [], [], []
    for i in range(n_bins):
        m = (st >= edges[i]) & (st < edges[i + 1] if i < n_bins - 1 else st <= edges[i + 1])
        if int(m.sum()) < min_count:
            continue
        means.append(float(y[m].median()))
        centers.append(float(55.0 * 2 ** ((edges[i] + edges[i + 1]) / 24.0)))
        counts.append(int(m.sum()))
    if len(means) < 5:
        return []
    mt = torch.tensor(means)
    if smooth > 1 and len(mt) >= smooth:
        k = torch.ones(1, 1, smooth) / smooth
        mt = torch.nn.functional.conv1d(mt.view(1, 1, -1), k,
                                        padding=smooth // 2).view(-1)[:len(means)]
    d = (mt[1:] - mt[:-1]).abs()
    out = []
    for i in range(1, len(d) - 1):
        if d[i] >= d[i - 1] and d[i] >= d[i + 1] and float(d[i]) > 0.8:
            out.append((round(0.5 * (centers[i] + centers[i + 1]), 1),
                        round(float(d[i]), 2), round(float(mt[i]), 2),
                        round(float(mt[i + 1]), 2)))
    return sorted(out, key=lambda r: -r[1])


def tension_from_f0(f0_hz: float, f0_ref: float = 170.0, exponent: float = 0.53
                    ) -> float:
    """F0 -> 2질량 모델의 긴장도 q (근사).

    `dsp/vocalfold.py` 는 k <- q·k, m <- m/q 라 고전적으로 F0 ∝ q 지만, 충돌과
    베르누이 되먹임 때문에 실제로는 지수가 0.5 근처다(q=1 -> 170 Hz,
    q=2 -> 247 Hz 로 측정됨). 정밀한 값이 필요하면 `calibrate_tension` 으로
    직접 곡선을 다시 재라.
    """
    return float(max(f0_hz, 1.0) / f0_ref) ** (1.0 / exponent)


def calibrate_tension(q_values=(0.7, 1.0, 1.5, 2.0, 3.0), n_samples: int = 7200,
                      sample_rate: int = 24000):
    """2질량 ODE 를 실제로 돌려 q -> F0 곡선을 재고 멱지수를 적합한다.

    LF 경로(학습 본선)와 ODE 경로(해석)를 잇는 다리. 느리므로 필요할 때만.
    """
    from ..dsp.vocalfold import FoldParams, cycle_rate, simulate
    qs, fs = [], []
    for q in q_values:
        flow, _, _ = simulate(FoldParams(q=q, a01=0.02, a02=0.02), n_samples,
                              sample_rate, oversample=4)
        r = cycle_rate(flow[n_samples // 3:], sample_rate)
        if r > 30:
            qs.append(math.log(q))
            fs.append(math.log(r))
    if len(qs) < 2:
        return {"f0_ref": 170.0, "exponent": 0.53, "points": []}
    qt, ft = torch.tensor(qs), torch.tensor(fs)
    qc = qt - qt.mean()
    slope = float((ft * qc).sum() / qc.pow(2).sum().clamp_min(1e-9))
    inter = float(ft.mean() - slope * qt.mean())
    return {"f0_ref": round(math.exp(inter), 2), "exponent": round(slope, 4),
            "points": [(q, round(math.exp(f), 1)) for q, f in zip(q_values, fs)]}
