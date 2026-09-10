"""VoiceEngine 통합 — 스크립트 -> 오디오, 스트리밍 = 오프라인, 유음 지표."""
import numpy as np
import pytest
import torch

from formant_ml.engine import VoiceEngine, EngineConfig, phones
from formant_ml.engine.control import track_from_keyframes


@pytest.fixture(scope="module")
def eng():
    return VoiceEngine(EngineConfig(n_extra_formants=3))


def _env_db(y, fs=48000, win=480):
    return np.array([20 * np.log10(np.sqrt((y[i:i + win] ** 2).mean()) + 1e-9)
                     for i in range(0, len(y) - win, win)])


def test_render_is_finite_and_bounded(eng):
    y = eng.render(phones.ara())
    assert np.isfinite(y).all() and np.abs(y).max() < 20


def test_streaming_equals_offline(eng):
    tr = phones.ra()
    off = eng.render(tr)
    chunks = np.concatenate(list(eng.stream(tr, chunk_ms=17)))
    assert chunks.shape == off.shape
    assert np.abs(chunks - off).max() < 1e-4 * np.abs(off).max()


def test_tap_makes_a_short_dip_with_continuous_voicing(eng):
    y = eng.render(phones.ara())
    e = _env_db(y)
    v = e[10:20].mean()                       # 첫 모음 정상부
    dip = e[23:31].min()
    assert 2.0 < v - dip < 12.0               # 실측 3.5~5.5 dB 골 (여유 있게)
    assert (e[23:31] > v - 25).all()          # 발성이 끊기지 않는다


def test_lateral_onset_is_quieter_than_vowel_but_voiced(eng):
    """설측 유지부가 뒤 모음보다 프로파일의 `lateral.level_db` 만큼 조용하다.

    창을 프레임 번호로 박아 두면 안 된다. 지속시간이 화자마다 다르고, 그 창이
    엇나가면 테스트가 **엉뚱한 이유로** 통과한다(실제로 겪었다: `f0_target` 위에
    폐압 결합이 곱해져 F0 가 18 % 높던 시절, 하모닉이 F1 을 비껴가 유지부가 우연히
    조용했고 그걸로 통과하고 있었다). 창은 제어열에서 읽는다.
    """
    tr = phones.ra()
    y = eng.render(tr)
    hop = eng.cfg.hop
    mix, ps = tr["lat_mix"], tr["p_sub"]
    after = np.flatnonzero(mix > 0.1)
    # **압력이 다 선 프레임만** 쓴다. `phones.BREATH_ONSET_S` 가 45 ms 램프라
    # `ps > 0` 은 소리가 거의 없는 개시 램프까지 끌어들여 rms 를 −45 dB 로 끌어내린다
    # (계단으로 세우면 개시에 파열음 버스트가 난다 — 실측 1.57 → 0.27).
    on = ps >= 0.999 * ps.max()      # 고원만 — 램프의 꼬리도 뺀다
    hold = (mix > 0.9) & on
    vowel = np.zeros(tr.n_frames, dtype=bool)
    vowel[after[-1] + 40:] = True
    vowel &= on

    def rms_db(m):
        i = np.flatnonzero(m)
        seg = y[i[0] * hop:min((i[-1] + 1) * hop, len(y))]
        return 20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-12)

    got = rms_db(hold) - rms_db(vowel)
    want = eng.profile.lateral["level_db"] if eng.profile else -2.0
    assert abs(got - want) < 2.5, (got, want)
    assert rms_db(hold) > rms_db(vowel) - 25.0          # 발성이 끊기지 않는다


def test_keyframe_track_defaults_before_first_key_and_zero_hold():
    tr = track_from_keyframes([dict(t=0.1, lat_z1=3000.0, f1=500.0),
                               dict(t=0.2, lat_z1=0.0, f1=800.0)], seconds=0.3)
    assert tr["lat_z1"][50] == 0.0 and tr["lat_z1"][150] == 3000.0 and tr["lat_z1"][250] == 0.0
    assert 500 < tr["f1"][150] < 800 and tr["f1"][50] == 0.0
    assert abs(tr["f1"][250] - 800.0) < 1e-6


def test_residual_is_disabled_where_frication_is_active():
    """치찰음은 학습에 맡기지 않는다 — 잔차 EQ 가 마찰 구간에 닿으면 안 된다.

    잔차 EQ 는 전 대역·전 구간에 걸리므로 막지 않으면 치찰음 물리가 틀렸을 때
    신경망이 ±6 dB 로 덮어 버린다. 그러면 물리는 틀린 채로 남고, 학습 데이터에 없는
    새 발화에서 치찰음이 무너진다. 규칙을 코드가 강제하게 둔다.
    """
    import numpy as np
    from formant_ml.engine.control import ControlTrack, default_vector
    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, residual=True))
    # 잔차 헤드를 항등이 아니게 만든다 (학습된 상태를 흉내)
    with torch.no_grad():
        for p in eng.residual.body.parameters():
            if p.dim() > 1:
                p.add_(torch.randn_like(p) * 0.3)
        eng.residual.head_bias_nonzero = True
    n = 300
    v = np.tile(default_vector(), (n, 1))
    tr = ControlTrack(v, 1.0)
    tr["p_sub"] = 8.0; tr["f0_target"] = 220.0; tr["residual_mix"] = 1.0
    tr["f1"] = 700; tr["f2"] = 1400; tr["f3"] = 2600; tr["f4"] = 3600
    tr["a_c"] = 3.0                                   # 모음 자세: 마찰 없음
    tr["a_c"][120:200] = 0.08                         # 가운데만 협착 (마찰)
    tr["adduction"] = 0.6
    tr["adduction"][120:200] = 0.05
    tr["fric_gain"] = 6.0
    tr = tr.clamp()
    ctrl = tr.to_tensor()
    eng.reset()
    out = eng(ctrl, [], 0.0)
    fr = out["fric"][0]
    hop = eng.cfg.hop
    on = (fr.abs() > 0)
    assert on.any(), "마찰이 실제로 켜져야 검사가 성립한다"
    # 마찰이 켜진 구간에서는 잔차가 적용되지 않아야 한다:
    # residual_mix 를 0 으로 둔 렌더와 그 구간의 파형이 같아야 한다.
    tr2 = ControlTrack(tr.values.copy(), tr.frame_ms)
    tr2["residual_mix"] = 0.0
    eng.reset()
    out2 = eng(tr2.clamp().to_tensor(), [], 0.0)
    a, b = out["audio"][0], out2["audio"][0]
    i = torch.nonzero(on).flatten()
    core = i[(i > i.min() + hop) & (i < i.max() - hop)]     # 경계 팽창 제외한 안쪽
    assert len(core) > 100
    d = (a[core] - b[core]).abs().max()
    assert float(d) < 1e-5, float(d)
