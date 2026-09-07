"""녹음 -> VoiceProfile(JSON). 실제 음성에서 '그 목소리'를 뽑아내는 진입점.

    PYTHONPATH=src python -m formant_ml.analysis.extract \\
        --wav data/me/*.wav --out profiles/me.json --name me

권장 녹음
---------
* 지속 모음 /아/ /이/ /우/ 를 편한 높이로 각 3초  -> 포먼트, Rd, 위상차
* 같은 모음을 낮은 음 -> 높은 음으로 미끄러뜨리며(글리산도) 2회 -> **파사지오**
* /스ㅡ/ /슈/ 를 각 2초                          -> 치찰음 지문
* 보통 말하기 30초 이상                          -> F0 분포, 지터/시머
조용한 방에서 녹음할 것. 배경잡음은 그대로 '난류 노이즈'로 흡수된다.
"""
from __future__ import annotations

import argparse
import glob
import math
import os

import torch

from ..data.features import frame_signal, stft
from ..utils import load_wav
from ..voice import VoiceProfile
from . import phase as ph
from . import registers as rg
from . import sibilant as sb


def lpc_formants(x: torch.Tensor, voicing: torch.Tensor, sample_rate: int = 24000,
                 hop: int = 240, frame_len: int = 1024, order: int | None = None,
                 n: int = 12, max_bw: float = 900.0, fmin: float = 90.0,
                 fmax: float = 11500.0, preemph: float = 0.97):
    """유성 프레임 평균 자기상관의 LPC 근에서 포먼트 n 개 (화자 평균).

    켑스트럼 포락선은 /아/ 처럼 F1(730) 과 F2(1090) 이 가까우면 둘을 하나의
    봉우리(≈880 Hz)로 뭉개 버린다. 그 값으로 프로브를 합성해 소스를 역추정하면
    성도 오차가 전부 소스 파라미터로 흘러 들어간다(측정: tilt 가 +2 대신 −10).
    LPC 근은 켤레쌍 하나가 포먼트 하나라서 가까운 포먼트도 분리한다.

    차수는 관례대로 2 + fs/1000 (24 kHz -> 26).
    """
    order = order or int(2 + sample_rate / 1000)
    if x.dim() == 1:
        x = x[None]
    xp = torch.cat([x[:, :1], x[:, 1:] - preemph * x[:, :-1]], dim=1)  # 프리엠퍼시스
    w = frame_signal(xp, frame_len, hop)[0]
    m = voicing[:w.shape[0]] > 0.5
    if int(m.sum()) < 5:
        m = torch.ones(w.shape[0], dtype=torch.bool)
    w = w[m] * torch.hann_window(frame_len)
    nfft = 1
    while nfft < 2 * frame_len:
        nfft <<= 1
    S = torch.fft.rfft(w.double(), nfft)
    r = torch.fft.irfft(S.real ** 2 + S.imag ** 2, nfft)[:, :order + 1].mean(0)
    r = r / r[0].clamp_min(1e-12)
    r[0] = r[0] + 1e-6                                   # 수치 안정용 백색잡음 바닥

    a = torch.zeros(order + 1, dtype=torch.float64)      # Levinson-Durbin
    a[0] = 1.0
    e = float(r[0])
    for i in range(1, order + 1):
        acc = float(r[i]) + float((a[1:i] * r[1:i].flip(0)).sum())
        k = -acc / max(e, 1e-12)
        a_new = a.clone()
        a_new[1:i + 1] = a[1:i + 1] + k * a[1:i + 1].flip(0) if i > 1 else a[1:i + 1]
        a_new[i] = k if i == 1 else a[i] + k * a[0]
        for j in range(1, i):
            a_new[j] = a[j] + k * a[i - j]
        a_new[i] = k
        a = a_new
        e *= (1 - k * k)
        if e <= 0:
            break

    p = order
    comp = torch.zeros(p, p, dtype=torch.float64)
    comp[0] = -a[1:p + 1]
    comp[1:, :-1] = torch.eye(p - 1, dtype=torch.float64)
    roots = torch.linalg.eigvals(comp)
    out = []
    for z in roots.tolist():
        if z.imag <= 0:
            continue
        mag = abs(z)
        if not (0.0 < mag < 1.0):
            continue
        f = float(torch.atan2(torch.tensor(z.imag), torch.tensor(z.real))
                  ) * sample_rate / (2 * math.pi)
        bw = -math.log(max(mag, 1e-9)) * sample_rate / math.pi
        if fmin < f < fmax and bw < max_bw:
            out.append((f, bw))
    out.sort()
    freqs = [round(f, 1) for f, _ in out[:n]]
    bws = [round(max(40.0, b), 1) for _, b in out[:n]]
    return freqs, bws


def mean_formants(x: torch.Tensor, voicing: torch.Tensor, sample_rate: int = 24000,
                  hop: int = 240, n_fft: int = 2048, n: int = 8,
                  quefrency: int = 40, fmax: float = 11000.0):
    """켑스트럼 포락선 기반 포먼트 (LPC 가 실패했을 때의 대비책)."""
    X = stft(x if x.dim() > 1 else x[None], n_fft, hop)[0].abs().transpose(0, 1)
    m = voicing[:X.shape[0]] > 0.5
    if int(m.sum()) < 5:
        m = torch.ones(X.shape[0], dtype=torch.bool)
    logmag = torch.log(X[m].clamp_min(1e-9)).mean(0)
    cep = torch.fft.irfft(logmag.to(torch.complex64), n_fft)
    cep[quefrency:-quefrency] = 0
    env = torch.exp(torch.fft.rfft(cep, n_fft).real)
    f = torch.linspace(0, sample_rate / 2, len(env))
    peaks = [(float(env[i]), float(f[i])) for i in range(1, len(env) - 1)
             if env[i] > env[i - 1] and env[i] > env[i + 1] and f[i] < fmax]
    peaks.sort(key=lambda p: -p[0])
    got = sorted(round(p[1], 1) for p in peaks[:n])
    while len(got) < n:
        got.append((got[-1] if got else 700.0) + 900.0)
    return got


def _probe(f0: float, rd: float, tilt: float, formants, bandwidths,
           sample_rate: int = 24000, n_frames: int = 120):
    """주어진 (f0, Rd, tilt, 포먼트) 로 지속모음을 합성한다 (분석 기준선용)."""
    from ..config import Config
    from ..models.synth import Controls, PhysicalVoiceSynth
    cfg = Config()
    K = cfg.filt.n_formants
    T = n_frames
    ones = torch.ones(1, T, 1)
    ff = torch.as_tensor(formants[:K], dtype=torch.float32
                         ).view(1, 1, -1).expand(1, T, -1)
    bw = torch.as_tensor(bandwidths[:K], dtype=torch.float32
                         ).view(1, 1, -1).expand(1, T, -1)
    c = Controls(f0=ones * f0, harmonic_amp=ones, rd=ones * rd, tilt=ones * tilt,
                 formant_freq=ff.contiguous(), formant_bw=bw.contiguous(),
                 formant_gain=torch.ones(1, T, ff.shape[-1]),
                 noise_bands=torch.full((1, T, cfg.noise.n_bands), 1e-6),
                 noise_entry=torch.zeros(1, T, 1), noise_am=torch.zeros(1, T, 1))
    with torch.no_grad():
        return PhysicalVoiceSynth(cfg)(c)["audio"]


def mean_voiced_log_spectrum(x, mask, hop: int = 240, n_fft: int = 1024,
                             smooth_bins: int = 15, eps: float = 1e-9):
    """선택 프레임의 평균 로그 크기 스펙트럼 (F,) dB, 하모닉 잔물결은 평활."""
    X = stft(x.detach() if x.dim() > 1 else x.detach()[None], n_fft, hop
             )[0].abs().transpose(0, 1)
    m = mask[:X.shape[0]]
    db = 20.0 * torch.log10(X[m].clamp_min(eps)).mean(0)
    if smooth_bins > 1:
        k = torch.ones(1, 1, smooth_bins) / smooth_bins
        db = torch.nn.functional.conv1d(
            torch.nn.functional.pad(db.view(1, 1, -1),
                                    (smooth_bins // 2, smooth_bins // 2),
                                    mode="replicate"), k).view(-1)
    return db


def fit_source_rd_tilt(target_db: torch.Tensor, f0: float, formants, bandwidths,
                       sample_rate: int = 24000, hop: int = 240, n_grid: int = 9,
                       fmin: float = 300.0, fmax: float = 9000.0,
                       rd_prior: float | None = None, rd_span: float = 0.45):
    """Rd 와 소스 tilt 를 **스펙트럼 포락선 전체에** 맞춰 함께 추정. (rd, tilt, rmse)

    왜 함께 풀어야 하나
    -------------------
    두 파라미터가 같은 관측량을 움직인다. H1 과 H2 는 정확히 한 옥타브 차이라
    tilt 를 t dB/oct 주면 H1-H2 가 t dB 줄어든다. 그래서 H1-H2 만 보고 Rd 를
    읽으면 tilt 가 있는 화자는 pressed 쪽으로 치우친다(참값 1.0 -> 0.71 로 측정됨).
    반대로 입에서 잰 기울기(-9.7 dB/oct)를 그대로 소스 tilt 로 쓰면 성도 롤오프까지
    두 번 세어 재앙적으로 어두워진다.

    H1-H2 와 기울기라는 **스칼라 두 개**로 푸는 것도 안 된다 — 관측 잡음에
    너무 민감해서 격자 끝으로 튄다(실제로 그랬다). 스펙트럼 포락선 전체를 쓴다.

    푸는 법: Rd 를 격자로 훑으며 tilt=0 프로브를 합성하고, 목표와의 잔차를
    ``a + t·log2(f/1kHz)`` 로 회귀한다. 로그 영역에서 tilt 는 정확히 직선이므로
    각 Rd 에 대한 최적 tilt 는 **닫힌 형태로** 나온다. 남은 잔차가 가장 작은
    Rd 를 고른다. 프로브에 같은 포먼트를 쓰기 때문에 성도 항은 대부분 상쇄된다.
    """
    n_freq = target_db.numel()
    f = torch.linspace(0, sample_rate / 2, n_freq)
    band = (f >= fmin) & (f <= fmax)
    oct_ = torch.log2(f[band].clamp_min(20.0) / 1000.0)
    oc = oct_ - oct_.mean()
    tgt = target_db[band]

    # Rd 는 **H1-H2 측정값을 중심으로 좁게만** 탐색한다.
    # 포락선 전체를 자유롭게 맞추면 Rd 와 tilt 가 축퇴해서 격자 끝으로 튄다
    # (실측: H1-H2 가 6.2 dB(=Rd 1.2)인데 자유 적합은 0.35 를 골랐다).
    # H1-H2 는 소스를 직접 재는 양이고 열린 모음에서 왕복 오차가 0.05 다.
    lo, hi = (0.35, 2.6) if rd_prior is None else (
        max(0.35, rd_prior - rd_span), min(2.6, rd_prior + rd_span))
    best = (float("inf"), rd_prior or 1.0, 0.0)
    for rd in torch.linspace(lo, hi, n_grid).tolist():
        y = _probe(f0, rd, 0.0, formants, bandwidths, sample_rate)
        pdb = mean_voiced_log_spectrum(y, torch.ones(y.shape[-1] // hop + 1,
                                                     dtype=torch.bool), hop)[band]
        r = tgt - pdb
        t = float(((r - r.mean()) * oc).sum() / oc.pow(2).sum().clamp_min(1e-9))
        t = max(-12.0, min(12.0, t))
        resid = r - r.mean() - t * oc
        rmse = float(resid.pow(2).mean().sqrt())
        if rmse < best[0]:
            best = (rmse, rd, t)
    return round(best[1], 3), round(best[2], 3), round(best[0], 3)


def fant_bandwidth(f: float) -> float:
    """포먼트 주파수 -> 대역폭 [Hz] (Fant 의 경험식 근사).

    F1 ~ 60, F3 ~ 120, F5 ~ 250 정도가 되도록 잡았다. 벽 손실·점성 손실·방사
    손실이 모두 주파수에 따라 커지는 것을 한 줄로 요약한 것이다.
    """
    return 50.0 + 20.0 * (f / 1000.0) ** 2 + 10.0 * (f / 1000.0)


def cycle_jitter_shimmer(feat: dict) -> tuple[float, float]:
    """프레임 간 F0/진폭 변동에서 지터·시머를 근사한다(프레임률 근사값)."""
    v = feat["voicing"] > 0.5
    f0 = feat["f0"][v]
    if f0.numel() < 5:
        return 0.004, 0.04
    d = (f0[1:] - f0[:-1]).abs() / f0[:-1].clamp_min(1e-3)
    jit = float(d.median())
    return round(min(max(jit, 0.0005), 0.05), 5), round(min(max(jit * 8, 0.005), 0.3), 5)


def _load(paths, sample_rate: int):
    x = torch.cat([load_wav(p, sample_rate) for p in paths])[None]
    return x / x.abs().max().clamp_min(1e-6) * 0.95


def extract_profile(paths: list[str], name: str = "voice", sample_rate: int = 24000,
                    hop: int = 240, n_stages: int = 3, verbose: bool = True,
                    vowel_paths: list[str] | None = None,
                    sibilant_paths: list[str] | None = None,
                    glissando_paths: list[str] | None = None,
                    vowel_name: str | None = None) -> VoiceProfile:
    """녹음 -> VoiceProfile.

    측정마다 필요한 녹음이 다르다. 역할별 파일을 따로 주면 훨씬 정확해진다.

    * `vowel_paths`     지속 모음 -> 포먼트, Rd, tilt, 위상차
    * `sibilant_paths`  /스ㅡ/ /슈/ -> 치찰음 지문
    * `glissando_paths` 낮은음->높은음 미끄러뜨리기 -> **파사지오**

    안 주면 전부 `paths` 를 쓴다. 그러면 모음·마찰음·글리산도가 한 통에 섞여서
    중앙값이 '이 화자의 값'이 아니라 '녹음 내용의 값'이 된다 — 특히 파사지오는
    F0 축 통계라서 다른 내용이 섞이면 엉뚱한 곳을 짚는다.
    """
    x = _load(paths, sample_rate)
    x_vowel = _load(vowel_paths, sample_rate) if vowel_paths else x
    x_sib = _load(sibilant_paths, sample_rate) if sibilant_paths else x
    x_gliss = _load(glissando_paths, sample_rate) if glissando_paths else x

    # 1) 먼저 화자 평균 포먼트를 구한다 (H1-H2 성도 보정과 위상 기준선에 필요).
    from ..data.features import yin_f0
    _, voi_pre = yin_f0(x_vowel, sample_rate, hop)
    formants, lpc_bw = lpc_formants(x_vowel, voi_pre[0], sample_rate, hop)
    if len(formants) < 4:                                # LPC 가 실패하면 켑스트럼
        formants = mean_formants(x_vowel, voi_pre[0], sample_rate, hop)
        lpc_bw = []
    # 대역폭은 LPC 근에서 읽지 않고 **Fant 근사식**을 쓴다. 짧고 잡음 있는 녹음에서
    # LPC 는 대역폭을 크게 과대추정한다(실측 /아/: F1 이 631 Hz 로 나왔는데
    # 사람은 60~90 이다). 과대추정된 대역폭으로 합성하면 포먼트가 뭉개져
    # 1~4 kHz 가 10 dB 넘게 주저앉는다.
    bws = [fant_bandwidth(f) for f in formants]
    prof_lpc_bw = [round(b, 1) for b in lpc_bw[:len(formants)]]

    # 2) 그 포먼트로 보정한 성대 지표
    # H1-H2 는 보정 없이 잰다(위 h1_h2_db 주석 참고: 평균 포먼트로 보정하면 더 나빠진다).
    feat = rg.register_features(x_vowel, sample_rate, hop)
    feat_all = (feat if x_vowel is x
                else rg.register_features(x, sample_rate, hop))
    v = feat["voicing"] > 0.5
    if int(v.sum()) < 20:
        raise ValueError("유성 프레임이 너무 적습니다. 녹음을 확인하세요.")
    # Rd/tilt 는 '지속 유성' 프레임에서만 잰다 (전이·글리산도가 중앙값을 흐린다)
    steady = rg.steady_voiced_mask(feat)
    if int(steady.sum()) < 20:
        steady = v

    v_all = feat_all["voicing"] > 0.5
    f0v = feat_all["f0"][v_all]
    rdv = feat["rd"][steady]
    q = lambda t, p: float(torch.quantile(t, p))               # noqa: E731
    jit, shim = cycle_jitter_shimmer(feat)
    feat_gliss = (feat if x_gliss is x_vowel
                  else rg.register_features(x_gliss, sample_rate, hop))

    prof = VoiceProfile(
        name=name, sample_rate=sample_rate,
        f0_median=round(float(f0v.median()), 2),
        f0_low=round(q(f0v, 0.05), 2), f0_high=round(q(f0v, 0.95), 2),
        rd_median=round(float(rdv.median()), 3),
        rd_low=round(q(rdv, 0.10), 3), rd_high=round(q(rdv, 0.90), 3),
        jitter=jit, shimmer=shim,
        tilt=0.0,   # 아래에서 모델 기준선을 빼고 채운다
        formants=formants, bandwidths=[round(b, 1) for b in bws],
        formant_gain=[1.0] * len(formants),
        passaggio=rg.passaggio_candidates(feat_gliss),
        register_stats={
            "shr_median_db": round(float(feat["shr"][v].median()), 2),
            "cpp_median_db": round(float(feat["cpp"][v].median()), 2),
            "h1h2_median_db": round(float(feat["h1h2"][steady].median()), 2),
            "steady_frames": int(steady.sum()),
            "voiced_ratio": round(float(v_all.float().mean()), 3),
        },
    )

    # 치찰음 지문
    try:
        mask = sb.find_sibilant_frames(x_sib, sample_rate, hop)
        if int(mask.sum()) >= 5:
            fit = sb.fit_sibilant(x_sib, sample_rate, hop, mask=mask, steps=500)
            prof.sibilant = {k: fit[k] for k in
                             ("pole_f", "pole_bw", "zero_f", "zero_bw", "tilt",
                              "slope_lo", "slope_hi", "teeth_f", "teeth_bw",
                              "floor_db")}
            prof.sibilant_moments = sb.measure(x_sib, sample_rate, hop, mask)
            prof.sibilant_moments["fit_rmse_db"] = fit["rmse_db"]
        elif verbose:
            print("  ! 마찰음 프레임을 못 찾아 치찰음 지문은 기본값을 씁니다"
                  " (/스ㅡ/ 를 녹음에 넣어 주세요)")
    except Exception as e:                                     # noqa: BLE001
        if verbose:
            print(f"  ! 치찰음 적합 실패: {e}")

    # 위상차 파라미터
    try:
        prof.dispersion = {k: v_ for k, v_ in
                           ph.extract(x_vowel, rd=prof.rd_median,
                                      sample_rate=sample_rate,
                                      hop=hop, n_stages=n_stages,
                                      formant_f=formants, formant_bw=bws).items()
                           if k in ("freq", "radius", "residual_rad")}
    except Exception as e:                                     # noqa: BLE001
        if verbose:
            print(f"  ! 위상차 적합 실패: {e}")

    # Rd 와 소스 tilt 를 스펙트럼 포락선에 함께 맞춘다.
    measured_tilt = float(feat["tilt"][steady].median())
    ff_pad = list(formants) + [formants[-1] + 1000.0 * (i + 1) for i in range(12)]
    bw_pad = list(bws) + [max(50.0, 0.06 * f + 40.0) for f in ff_pad[len(bws):]]
    try:
        target_db = mean_voiced_log_spectrum(x_vowel, steady, hop)
        rd_fit, tilt_fit, rmse = fit_source_rd_tilt(
            target_db, prof.f0_median, ff_pad, bw_pad, sample_rate, hop,
            rd_prior=prof.rd_median)
        prof.register_stats["source_fit_rmse_db"] = rmse
        shift = rd_fit - prof.rd_median
        prof.rd_median = rd_fit
        prof.rd_low = round(max(0.3, prof.rd_low + shift), 3)
        prof.rd_high = round(min(2.7, prof.rd_high + shift), 3)
        prof.tilt = tilt_fit
    except Exception as e:                                     # noqa: BLE001
        if verbose:
            print(f"  ! Rd/tilt 동시추정 실패: {e}")
    prof.register_stats["measured_tilt_db_per_oct"] = round(measured_tilt, 2)
    prof.register_stats["lpc_bandwidths"] = prof_lpc_bw

    if vowel_name:
        prof.vowel_formants[vowel_name] = [round(x, 1) for x in formants]
    prof.meta = {"files": [os.path.basename(p) for p in paths],
                 "seconds": round(x.shape[-1] / sample_rate, 2)}
    return prof


def main() -> None:
    ap = argparse.ArgumentParser(description="녹음에서 VoiceProfile 추출")
    ap.add_argument("--wav", nargs="+", required=True, help="wav 파일 또는 글롭")
    ap.add_argument("--out", default="profiles/voice.json")
    ap.add_argument("--name", default="voice")
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--stages", type=int, default=3, help="위상차 올패스 단수")
    ap.add_argument("--vowel-wav", nargs="*", help="지속 모음 (포먼트/Rd/위상차)")
    ap.add_argument("--sibilant-wav", nargs="*", help="/스ㅡ/ /슈/ (치찰음 지문)")
    ap.add_argument("--vowel-name", help="--vowel-wav 가 어떤 모음인지 (예: a). "
                    "주면 그 모음의 포먼트를 측정값으로 저장한다")
    ap.add_argument("--glissando-wav", nargs="*",
                    help="낮은음->높은음 글리산도 (파사지오). 이걸 주면 검출이 크게 좋아진다")
    args = ap.parse_args()

    def expand(pats):
        if not pats:
            return None
        out = []
        for p in pats:
            out.extend(sorted(glob.glob(p)) or [p])
        return out

    paths = expand(args.wav)
    print(f"{len(paths)} 개 파일 분석 중 …")
    prof = extract_profile(paths, args.name, args.sample_rate,
                           n_stages=args.stages,
                           vowel_paths=expand(args.vowel_wav),
                           sibilant_paths=expand(args.sibilant_wav),
                           glissando_paths=expand(args.glissando_wav),
                           vowel_name=args.vowel_name)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    prof.save(args.out)

    print(f"\n저장: {args.out}")
    print(f"  F0        {prof.f0_low:.0f} – {prof.f0_median:.0f} – {prof.f0_high:.0f} Hz")
    print(f"  Rd        {prof.rd_low:.2f} – {prof.rd_median:.2f} – {prof.rd_high:.2f}"
          f"  (H1-H2 {prof.register_stats.get('h1h2_median_db')} dB)")
    print(f"  tilt      {prof.tilt:+.2f} dB/oct (소스에 더할 양; 입에서 잰 값은 "
          f"{prof.register_stats.get('measured_tilt_db_per_oct')})")
    print(f"  포먼트     {[round(f) for f in prof.formants[:5]]}")
    print(f"  치찰음     극 {prof.sibilant['pole_f']:.0f} Hz / 영점 "
          f"{prof.sibilant['zero_f']:.0f} Hz")
    print(f"  위상차     {prof.dispersion.get('freq')} r={prof.dispersion.get('radius')}")
    if prof.passaggio:
        for f0, jump, lo, hi in prof.passaggio[:3]:
            print(f"  파사지오   {f0:.0f} Hz 부근에서 H1-H2 가 {lo} -> {hi} dB "
                  f"(변화량 {jump})")
    else:
        print("  파사지오   후보 없음 (글리산도 녹음을 넣으면 잘 잡힙니다)")


if __name__ == "__main__":
    main()
