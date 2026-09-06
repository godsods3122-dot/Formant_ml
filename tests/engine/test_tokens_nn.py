"""감정 토큰 레지스트리 + shifted softplus 망."""
import json

import numpy as np
import pytest
import torch

from formant_ml.engine.control import INDEX, PARAM_NAMES, track_from_keyframes
from formant_ml.engine.nn import SSPMLP, SSPTCN, ShiftedSoftplus, shifted_softplus
from formant_ml.engine.tokens import TokenDiscovery, TokenRegistry, TokenTable


def _delta(**kw):
    d = np.zeros(len(PARAM_NAMES))
    for k, v in kw.items():
        d[INDEX[k]] = v
    return d


def test_spawn_auto_name_describe_and_budget():
    reg = TokenRegistry(budget=2)
    t = reg.spawn(_delta(p_sub=1.5, adduction=-0.2, tilt=-3), "angry", reason="loss cluster")
    assert t.name == "angry_sigh_00" and "p_sub+1.5" in t.describe()
    reg.spawn(_delta(f0_scale=0.2), "happy")
    with pytest.raises(RuntimeError):
        reg.spawn(_delta(tilt=1), "sad")


def test_apply_marks_and_strip_restores_exactly():
    reg = TokenRegistry()
    reg.spawn(_delta(p_sub=1.0, adduction=-0.2, f0_scale=0.1), "angry", name="angry_sigh")
    kf = [dict(t=0.0, p_sub=7.0, adduction=0.6, f0_scale=1.0), dict(t=0.5, p_sub=7.0)]
    tr = track_from_keyframes(kf, seconds=0.5)
    before = tr.values.copy()
    reg.apply(tr, "angry_sigh", 0.1, 0.3, strength=0.8)
    assert tr["p_sub"][200] > 7.5 and tr["adduction"][200] < 0.5
    assert abs(tr["f0_scale"][200] / np.exp(0.08) - 1.0) < 1e-6     # 로그 파라미터는 비율
    assert tr["p_sub"][50] == 7.0 and tr["p_sub"][450] == 7.0
    m = reg.markers(tr)
    assert len(m) == 1 and m[0]["name"] == "angry_sigh" and m[0]["t0"] == 0.1
    reg.strip(tr)
    assert np.allclose(tr.values, before, atol=1e-9) and reg.markers(tr) == []


def test_exclude_freeze_prune_and_roundtrip():
    reg = TokenRegistry(allowed_params=["p_sub", "adduction", "tilt", "f0_scale"])
    t = reg.spawn(_delta(p_sub=1.0, f1=200.0), "angry", support=10)   # f1 은 화이트리스트 밖
    assert t.delta[INDEX["f1"]] == 0.0
    reg.exclude(t.name)
    tr = track_from_keyframes([dict(t=0.0, p_sub=7.0), dict(t=0.2, p_sub=7.0)], 0.2)
    reg.apply(tr, t.name, 0.0, 0.2)
    assert reg.markers(tr) == [] and tr["p_sub"][100] == 7.0
    reg.exclude(t.name, False); reg.freeze(t.name)
    reg.spawn(_delta(tilt=1.0), "sad", support=1)
    gone = reg.prune(min_support=5)
    assert gone == ["sad_misc_00"] and t.name in reg.tokens             # frozen 은 남는다
    reg2 = TokenRegistry.from_json(reg.to_json())
    assert reg2.tokens[t.name].frozen and json.loads(reg2.to_json())["budget"] == 64


def test_token_table_is_differentiable_and_grows():
    reg = TokenRegistry()
    reg.spawn(_delta(p_sub=1.0), "angry", name="a")
    tab = TokenTable(reg)
    w = torch.zeros(1, 5, 1); w[0, 2, 0] = 1.0
    out = tab(w)
    assert out.shape == (1, 5, len(PARAM_NAMES)) and float(out[0, 2, INDEX["p_sub"]]) == 1.0
    out.sum().backward(); assert tab.delta.grad is not None
    i = tab.add_token("b", _delta(tilt=-2.0), emotion="sad")
    assert i == 1 and tab.n_tokens == 2 and "b" in reg.tokens
    reg.freeze("a")
    w2 = torch.ones(1, 3, 2)
    tab.delta.grad = None
    tab(w2).sum().backward()
    assert float(tab.delta.grad[0].abs().sum()) == 0.0 and float(tab.delta.grad[1].abs().sum()) > 0


def test_discovery_finds_clusters_per_emotion():
    disc = TokenDiscovery(loss_quantile=0.5, min_support=20, k=2, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(100):
        disc.observe(_delta(p_sub=1.0) + rng.normal(0, 0.01, len(PARAM_NAMES)), "angry", 1.0)
        disc.observe(_delta(tilt=-2.0) + rng.normal(0, 0.01, len(PARAM_NAMES)), "angry", 1.0)
        disc.observe(_delta(f0_scale=0.1), "happy", 0.1)          # 낮은 손실: 후보 아님
    props = disc.propose()
    emos = {p["emotion"] for p in props}
    assert emos == {"angry"} and len(props) == 2
    reg = TokenRegistry()
    for p in props:
        reg.spawn(p["delta"], p["emotion"], p["reason"], support=p["support"])
    names = sorted(reg.tokens)
    assert any("sigh" in n or "misc" in n for n in names)


def test_shifted_softplus_properties():
    x = torch.linspace(-5, 5, 101)
    y = shifted_softplus(x)
    assert abs(float(y[50])) < 1e-6 and (y[1:] > y[:-1]).all()
    mlp = SSPMLP(4, 16, 3)
    assert float(mlp(torch.randn(7, 4)).abs().sum()) == 0.0       # 0 초기화 = 항등
    tcn = SSPTCN(6, 8, 2, kernel=3, n_blocks=2)
    x = torch.randn(1, 20, 6, requires_grad=True)
    tcn.out.weight.data.normal_()
    tcn(x)[:, 5].sum().backward()
    assert float(x.grad[0, 6:].abs().sum()) == 0.0                # 인과: 미래가 영향 없음
    assert isinstance(ShiftedSoftplus()(x), torch.Tensor)
