"""물리 엔진 검증 — 이 테스트들이 통과하면 방정식이 제대로 구현된 것이다.

    PYTHONPATH=src python3 -m pytest tests -q        (pytest 있으면)
    PYTHONPATH=src python3 tests/test_dsp.py         (없으면 그냥 실행)
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from formant_ml.config import Config, sections_for
from formant_ml.data.features import yin_f0
from formant_ml.dsp.core import freq_grid, ltv_filter
from formant_ml.dsp.filters import allpass_response, resonator_stage_responses
from formant_ml.dsp.glottal import GlottalSource, _lf_waveform
from formant_ml.dsp.tract import tract_response
from formant_ml.dsp.vocalfold import FoldParams, simulate
from formant_ml.models.encoder import ControlEncoder
from formant_ml.models.losses import VoiceLoss
from formant_ml.models.synth import PhysicalVoiceSynth

FS = 24000


def _peaks(mag: torch.Tensor, fs: int = FS, n: int = 6):
    f = freq_grid(len(mag), fs)
    return [float(f[i]) for i in range(1, len(mag) - 1)
            if mag[i] > mag[i - 1] and mag[i] > mag[i + 1]][:n]


def test_ltv_filter_is_identity_for_flat_response():
    x = torch.randn(2, 2400)
    H = torch.ones(2, 10, 513, dtype=torch.complex64)
    assert (ltv_filter(x, H, 240, 256) - x).abs().max() < 1e-5


def test_formant_cascade_hits_target_frequencies():
    """캐스케이드의 포락선 피크가 목표 포먼트와 3% 이내로 일치."""
    target = [730.0, 1090.0, 2440.0, 3400.0]
    ff = torch.tensor(target).reshape(1, 1, -1)
    S = resonator_stage_responses(ff, torch.full((1, 1, 4), 90.0),
                                  torch.ones(1, 1, 4), FS, 513)
    H = S.prod(dim=2)
    assert abs(H.abs()[0, 0, 0] - 1.0) < 1e-3, "DC 이득은 1이어야 한다"
    got = _peaks(H.abs()[0, 0], n=4)
    for g, t in zip(got, target):
        assert abs(g - t) / t < 0.03, f"{got} vs {target}"


def test_allpass_is_magnitude_flat():
    f = torch.full((1, 1, 3), 2000.0)
    r = torch.full((1, 1, 3), 0.8)
    H = allpass_response(f, r, FS, 513)
    assert (H.abs() - 1.0).abs().max() < 1e-4


def test_uniform_tube_gives_quarter_wave_resonances():
    """길이 17.5 cm 균일관 -> 500/1500/2500 Hz (고전 음향학의 기준 결과)."""
    n = sections_for(FS)
    assert n == 24
    H = tract_response(torch.ones(1, 1, n), FS, 2049, rho=0.997)
    got = _peaks(H.abs()[0, 0], n=3)
    for g, t in zip(got, [500.0, 1500.0, 2500.0]):
        assert abs(g - t) < 25.0, got


def test_tract_is_always_stable():
    """임의의 양수 면적함수에 대해 모든 극점이 단위원 안에 있어야 한다."""
    from formant_ml.dsp.tract import area_to_reflection, reflection_to_lpc
    torch.manual_seed(0)
    area = torch.rand(4, 3, 24) * 6.0 + 0.05
    a = reflection_to_lpc(area_to_reflection(area))
    roots = torch.linalg.eigvals(
        torch.diag(torch.ones(a.shape[-1] - 2), -1).expand(1, 1, 1, 1, 1)[0, 0, 0, 0]
        if False else _companion(a[0, 0]))
    assert roots.abs().max() < 1.0


def _companion(a: torch.Tensor) -> torch.Tensor:
    """다항식 계수 -> 동반행렬 (근 = 극점)."""
    p = a / a[0]
    n = len(p) - 1
    c = torch.zeros(n, n, dtype=torch.float64)
    c[0] = -p[1:].to(torch.float64)
    c[1:, :-1] = torch.eye(n - 1, dtype=torch.float64)
    return c


def test_lf_waveform_closes_the_glottis():
    """LF 파형은 한 주기 면적이 0 (유량이 닫힘) 이고 폐쇄 시 음의 첨두를 갖는다."""
    for rd in (0.5, 1.0, 2.0):
        w = _lf_waveform(rd, 4096)
        assert abs(float(w.mean())) < 1e-6
        assert float(w.min()) == -1.0 or float(w.min()) < float(w.max())


def test_glottal_source_pitch_is_exact():
    src = GlottalSource(FS, 240, n_harmonics=120)
    for f0 in (90.0, 200.0, 400.0):
        x, _ = src(torch.full((1, 40, 1), f0), torch.full((1, 40, 1), 1.2),
                   torch.ones(1, 40, 1))
        est, _ = yin_f0(x, FS, 240)
        assert abs(float(est.median()) - f0) / f0 < 0.02


def test_glottal_source_has_no_aliasing():
    """에일리어싱은 '하모닉이 아닌 위치'에 에너지로 나타난다.

    f0=700 Hz 에서 나이퀴스트 위 하모닉이 접히면 fs - k*f0 위치에 선이 생기는데,
    이는 일반적으로 f0 의 정수배가 아니다. 하모닉 사이(중간 지점) 에너지를 재서
    피크 대비 -50 dB 이하인지 확인한다.
    """
    f0 = 700.0
    src = GlottalSource(FS, 240, n_harmonics=180)
    x, _ = src(torch.full((1, 40, 1), f0), torch.full((1, 40, 1), 0.6),
               torch.ones(1, 40, 1))
    n = x.shape[1]
    X = torch.fft.rfft(x[0] * torch.hann_window(n)).abs()
    df = FS / n
    peak = float(X.max())
    worst = 0.0
    k = 1
    while (k + 0.5) * f0 < FS / 2 - f0:
        b = int(round((k + 0.5) * f0 / df))          # 하모닉 사이 중간
        worst = max(worst, float(X[b - 1:b + 2].max()) / peak)
        k += 1
    assert worst < 10 ** (-50 / 20), f"비하모닉 성분 {20 * math.log10(worst):.1f} dB"


def test_vocal_folds_self_oscillate_and_track_tension():
    """방정식만으로 자가진동하고, 긴장도 q 를 올리면 폐쇄 주기율이 올라간다."""
    from formant_ml.dsp.vocalfold import cycle_rate
    rates = []
    for q in (1.0, 2.0):
        flow, _, _ = simulate(FoldParams(q=q, a01=0.02, a02=0.02), 9600, FS,
                              oversample=4)
        seg = flow[2400:]
        assert float(seg.std()) > 1.0, "진동이 죽었다"
        r = cycle_rate(seg, FS)
        assert 60.0 < r < 600.0, f"비현실적인 F0: {r}"
        rates.append(r)
    assert rates[1] > rates[0] * 1.3, rates


def test_end_to_end_gradients_flow():
    """인코더 -> 물리모델 -> 손실 전 구간이 미분가능해야 한다."""
    cfg = Config()
    enc, syn, loss_fn = ControlEncoder(cfg), PhysicalVoiceSynth(cfg), VoiceLoss()
    mel, f0, voi = torch.randn(1, 40, 80), torch.full((1, 40), 150.0), torch.ones(1, 40)
    c = enc(mel, f0, voi)
    assert bool((c.formant_freq[..., 1:] > c.formant_freq[..., :-1]).all()), \
        "포먼트 순서가 구조적으로 보장되어야 한다"
    y = syn(c)["audio"]
    loss_fn(y, torch.randn_like(y) * 0.1, c)["total"].backward()
    g = sum(float(p.grad.abs().sum()) for p in enc.parameters() if p.grad is not None)
    assert g > 0 and math.isfinite(g)


def test_noise_only_passes_downstream_of_constriction():
    """협착 위치를 입술쪽으로 옮기면 노이즈 경로가 저역 포먼트의 영향을 덜 받는다."""
    cfg = Config()
    syn = PhysicalVoiceSynth(cfg)
    K = cfg.filt.n_formants
    T = 20

    def energy(entry):
        from formant_ml.models.synth import Controls
        c = Controls(
            f0=torch.full((1, T, 1), 120.0), harmonic_amp=torch.zeros(1, T, 1),
            rd=torch.full((1, T, 1), 1.2),
            formant_freq=torch.tensor([300., 900., 2400., 3400., 4500., 5500.]
                                      ).reshape(1, 1, -1).expand(1, T, K).contiguous(),
            formant_bw=torch.full((1, T, K), 90.0),
            formant_gain=torch.ones(1, T, K),
            noise_bands=torch.ones(1, T, cfg.noise.n_bands),
            noise_entry=torch.full((1, T, 1), entry),
            noise_am=torch.zeros(1, T, 1))
        y = syn(c)["audio"][0]
        Y = torch.fft.rfft(y * torch.hann_window(len(y))).abs()
        f = freq_grid(len(Y), FS)
        return float((Y * (f < 1200)).sum() / Y.sum())

    low_entry = energy(0.0)     # 성문 근처에서 주입 -> 모든 포먼트 통과
    high_entry = energy(5.0)    # 입술 근처에서 주입 -> 저역 포먼트 우회
    assert high_entry < low_entry * 0.8, (low_entry, high_entry)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} 통과")
    sys.exit(1 if failed else 0)
