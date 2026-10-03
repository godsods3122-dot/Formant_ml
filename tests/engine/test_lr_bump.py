"""정체하면 학습률을 올린다 (`fit.LR_BUMP_MAX`, docs/MEASUREMENTS §52.97).

사용자: *"항상 하는 말인데, lr을 고정하라는 게 아니라, 정 확률이 낮게 수렴하려고 하면 lr을 동적으로
올리라는 뜻이야."* 그러므로 **고정 lr 을 올리는 것이 아니라 루프 안에서 올라가야** 한다.
"""
import numpy as np
import pytest
import torch

from formant_ml.engine import fit as F
from formant_ml.engine.control import ControlTrack, default_vector
from formant_ml.engine.fit import CopySynthFitter
from formant_ml.engine.profile import DEFAULT_PROFILE
from formant_ml.engine.voice import EngineConfig, VoiceEngine


@pytest.fixture(scope="module")
def _base():
    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, residual=False),
                      DEFAULT_PROFILE)
    v = np.tile(default_vector(), (120, 1))
    tr = ControlTrack(v, 1.0)
    tr["p_sub"] = 7.0; tr["adduction"] = 0.6; tr["f0_target"] = 200.0
    tr["residual_mix"] = 0.0
    tr = tr.clamp()
    return eng, tr, np.asarray(eng.render(tr), dtype=np.float64)


def _plateau(f, env_pct):
    """손실이 꿈쩍도 안 하는 가짜 loss — 포락만 원하는 값으로 준다."""
    def loss(*_a, **_k):
        base = (f.w ** 2).sum() * 0.0                 # requires_grad 를 살리되 값은 상수
        one = torch.ones((), dtype=base.dtype)
        f._last_db = 3.0                              # 최선 스냅샷이 읽는 값
        return base + 1.0, one * 0.2, one * (1.0 - env_pct / 100.0), {}
    return loss


def test_bumps_once_then_stops_when_it_does_not_help(_base, monkeypatch):
    """낮은 데서 정체하면 올리되, **올려도 안 오르면 멈춘다** — 구조적 한계일 수 있다.

    실측(M57 1 단계): 상향이 두 번 걸렸는데 포락이 66.02 → 66.09 로 꿈쩍 안 했다. 1 단계는 전역
    스칼라뿐이라 그 값이 한계였고, 360 회를 낭비할 뻔했다.
    """
    eng, tr, y = _base
    monkeypatch.setattr(F, "LR_BUMP_PATIENCE", 3)
    monkeypatch.setattr(F, "LR_BUMP_MAX", 3)
    f = CopySynthFitter(eng, y, 48000, tr)
    monkeypatch.setattr(f, "loss", _plateau(f, 70.0))     # 포락이 전혀 안 오른다
    f.fit(40, 0.04, 10 ** 9, False)
    assert f._lr_bumps == 1, f._lr_bumps


def test_bumps_again_when_it_helps(_base, monkeypatch):
    """상향 뒤 포락이 실제로 오르면 다음 정체에서 또 올린다."""
    eng, tr, y = _base
    monkeypatch.setattr(F, "LR_BUMP_PATIENCE", 3)
    monkeypatch.setattr(F, "LR_BUMP_MAX", 3)
    monkeypatch.setattr(F, "LR_BUMP_MIN_GAIN", 0.2)
    f = CopySynthFitter(eng, y, 48000, tr)
    st = {"n": 0}

    def loss(*_a, **_k):
        st["n"] += 1
        env = 70.0 + 5.0 * f._lr_bumps        # 상향마다 포락이 5 점 오른다
        base = (f.w ** 2).sum() * 0.0
        one = torch.ones((), dtype=base.dtype)
        f._last_db = 3.0
        return base + 1.0, one * 0.2, one * (1.0 - env / 100.0), {}

    monkeypatch.setattr(f, "loss", loss)
    f.fit(60, 0.04, 10 ** 9, False)
    assert f._lr_bumps >= 2, f._lr_bumps


def test_does_not_bump_when_already_good(_base, monkeypatch):
    eng, tr, y = _base
    monkeypatch.setattr(F, "LR_BUMP_PATIENCE", 3)
    monkeypatch.setattr(F, "LR_BUMP_MAX", 2)
    f = CopySynthFitter(eng, y, 48000, tr)
    monkeypatch.setattr(f, "loss", _plateau(f, 95.0))   # 이미 LR_BUMP_ENV 위
    f.fit(20, 0.04, 10 ** 9, False)
    assert f._lr_bumps == 0, f._lr_bumps


def test_off_when_max_is_zero(_base, monkeypatch):
    eng, tr, y = _base
    monkeypatch.setattr(F, "LR_BUMP_PATIENCE", 3)
    monkeypatch.setattr(F, "LR_BUMP_MAX", 0)
    f = CopySynthFitter(eng, y, 48000, tr)
    monkeypatch.setattr(f, "loss", _plateau(f, 70.0))
    f.fit(20, 0.04, 10 ** 9, False)
    assert f._lr_bumps == 0


def test_bump_extends_the_iteration_budget(_base, monkeypatch):
    """올렸으면 **쓸 시간**을 줘야 한다 — 상향마다 예산이 `LR_BUMP_EXTRA` 만큼 늘어난다.

    실측(M55): 2.1 단계는 반복 200 인데 정체가 150 회쯤 와서, 연장이 없으면 상향 뒤 10 회만 남아
    아무 일도 못 한다.
    """
    eng, tr, y = _base
    monkeypatch.setattr(F, "LR_BUMP_PATIENCE", 3)
    monkeypatch.setattr(F, "LR_BUMP_MAX", 2)
    monkeypatch.setattr(F, "LR_BUMP_EXTRA", 7)
    f = CopySynthFitter(eng, y, 48000, tr)
    monkeypatch.setattr(f, "loss", _plateau(f, 70.0))
    rep = f.fit(10, 0.04, 10 ** 9, False)
    assert f._lr_bumps == 1, f._lr_bumps                    # 안 오르면 한 번에 멈춘다
    assert len(rep.history) > 10, len(rep.history)          # 예산이 실제로 늘었다
    assert len(rep.history) <= 10 + 2 * 7, len(rep.history)  # 상한도 지킨다


def test_bump_restores_the_best_state_first(monkeypatch):
    """상향은 **최선 지점에서** 출발해야 한다 (§52.136).

    발산한 자리에서 lr 을 올리면 더 멀리 간다 — 실측으로 연장 예산 260 회를 버렸다.
    """
    import re
    from pathlib import Path

    src = Path("src/formant_ml/engine/fit.py").read_text(encoding="utf-8")
    block = src[src.index("                    env_at_bump = env"):]
    block = block[:block.index("bumps += 1")]
    # 복원이 lr 상향 **앞**에 있어야 한다.
    i_restore = block.index("self._restore(best[1])")
    i_lr = block.index('g["lr"] = float(g["lr"]) * LR_BUMP_FACTOR')
    assert i_restore < i_lr, "복원이 lr 상향보다 뒤에 있으면 뜻이 없다"
    assert re.search(r"if best\[1\] is not None and score < best\[0\]", block)


def test_does_not_bump_on_a_transient_dip_below_the_threshold(_base, monkeypatch):
    """단계 최선값이 이미 문턱 위면, 격자 교체로 **잠깐** 내려간 포락을 보고 올리지 않는다 (§52.185).

    실측 `hnr35` 2.3 단계: [0] 91.74 % → [25] 89.47 % 에서 현재값만 보고 올렸고, 포락을 −5.10 점 잃었다.
    """
    eng, tr, y = _base
    monkeypatch.setattr(F, "LR_BUMP_PATIENCE", 3)
    monkeypatch.setattr(F, "LR_BUMP_MAX", 3)
    f = CopySynthFitter(eng, y, 48000, tr)
    calls = {"n": 0}

    def loss(*_a, **_k):
        calls["n"] += 1
        env = 92.0 if calls["n"] <= 1 else 89.0          # 첫 회차가 최선(92 %), 이후 과도로 89 %
        # `w` 를 **저장하지 않는** 그래프여야 한다 — 최선 상태 복귀가 `w` 를 제자리에서 바꾸므로 `w**2` 는
        # backward 에서 버전 오류를 낸다. 실제 손실은 `w` 를 기저와 선형으로만 섞어 이 문제가 없다.
        base = (f.w * 0.0).sum()
        one = torch.ones((), dtype=base.dtype)
        f._last_db = 3.0
        # 손실은 처음이 가장 낮다 — 최선 스냅샷이 92 % 상태로 잡힌다
        return base + (1.0 if calls["n"] <= 1 else 1.01), one * 0.2, one * (1.0 - env / 100.0), {}
    monkeypatch.setattr(f, "loss", loss)
    f.fit(40, 0.04, 10 ** 9, False)
    assert getattr(f, "_lr_bumps", 0) == 0, f._lr_bumps


def test_still_bumps_when_the_best_is_also_low(_base, monkeypatch):
    """최선값도 문턱 아래면 예전처럼 올린다 — 사용자의 "낮게 수렴하면 동적으로 올려라" 는 그대로다."""
    eng, tr, y = _base
    monkeypatch.setattr(F, "LR_BUMP_PATIENCE", 3)
    monkeypatch.setattr(F, "LR_BUMP_MAX", 3)
    f = CopySynthFitter(eng, y, 48000, tr)
    calls = {"n": 0}

    def loss(*_a, **_k):
        calls["n"] += 1
        env = 85.0 if calls["n"] <= 1 else 84.0
        # `w` 를 **저장하지 않는** 그래프여야 한다 — 최선 상태 복귀가 `w` 를 제자리에서 바꾸므로 `w**2` 는
        # backward 에서 버전 오류를 낸다. 실제 손실은 `w` 를 기저와 선형으로만 섞어 이 문제가 없다.
        base = (f.w * 0.0).sum()
        one = torch.ones((), dtype=base.dtype)
        f._last_db = 3.0
        return base + (1.0 if calls["n"] <= 1 else 1.01), one * 0.2, one * (1.0 - env / 100.0), {}
    monkeypatch.setattr(f, "loss", loss)
    f.fit(40, 0.04, 10 ** 9, False)
    assert getattr(f, "_lr_bumps", 0) >= 1


def test_bumped_schedule_anneals_over_the_extension_and_never_climbs_back(_base, monkeypatch):
    """상향 뒤 lr 은 **그 순간에만** 오르고, 연장 끝까지 바닥으로 식어야 한다 (§52.256).

    예전에는 기준값만 두 배로 올리고 코사인 주기(`iters`)를 그대로 돌렸다. 코사인은 `T_max` 를 넘으면
    되올라가므로 연장 구간에서 lr 이 매 회차 커져, 600 회째에 처음의 8 배가 됐고 올린 단계가 모두 무너졌다.
    """
    eng, tr, y = _base
    monkeypatch.setattr(F, "LR_BUMP_PATIENCE", 3)
    monkeypatch.setattr(F, "LR_BUMP_MAX", 3)
    monkeypatch.setattr(F, "LR_BUMP_EXTRA", 12)
    monkeypatch.setattr(F, "LR_BUMP_MIN_GAIN", -1e9)   # 효과가 없어도 끝까지 올리게 한다
    monkeypatch.setattr(F, "SCHED", "cos")
    lrs: list[float] = []

    class _Rec(torch.optim.Adam):
        def step(self, *a, **k):
            lrs.append(float(self.param_groups[0]["lr"]))
            return super().step(*a, **k)

    monkeypatch.setattr(F.torch.optim, "Adam", _Rec)
    f = CopySynthFitter(eng, y, 48000, tr)

    def loss(*_a, **_k):
        base = (f.w * 0.0).sum()
        one = torch.ones((), dtype=base.dtype)
        f._last_db = 3.0
        return base + 1.0, one * 0.2, one * 0.3, {}      # 포락 70 %, 손실 불변 -> 계속 정체

    monkeypatch.setattr(f, "loss", loss)
    lr0, iters = 0.04, 10
    f.fit(iters, lr0, 10 ** 9, False)
    assert f._lr_bumps == 3, f._lr_bumps
    # 되돌린 회차는 걸음을 건너뛴다 (`test_a_restoring_bump_does_not_step_on_a_stale_graph`)
    assert len(lrs) == iters + 3 * 12 - f._lr_bumps, (len(lrs), f._lr_bumps)
    ups = sum(1 for a, b in zip(lrs, lrs[1:]) if b > a * (1 + 1e-9))
    assert ups == f._lr_bumps, (ups, lrs)                 # 오르는 회차는 상향한 회차뿐이다
    # 연장 끝에 바닥 언저리까지 식는다. 되돌린 회차는 걸음도 스케줄러도 건너뛰므로 상향 횟수만큼
    # 덜 밟아 정확히 바닥에 닿지는 않는다 — 중요한 성질은 '되올라가지 않는다' 쪽이다.
    assert lrs[-1] <= lr0 * 0.05 * 1.5, lrs[-1]
    assert max(lrs) <= lr0 * 2.0 ** 3 + 1e-12, max(lrs)


def test_legacy_switch_restores_the_old_climbing_schedule(_base, monkeypatch):
    """`BUMP_RESCHED=False` 는 예전 판을 재현하려는 진단 스위치다 — 정말 예전처럼 연장 구간에서 되올라야 한다."""
    eng, tr, y = _base
    monkeypatch.setattr(F, "LR_BUMP_PATIENCE", 3)
    monkeypatch.setattr(F, "LR_BUMP_MAX", 3)
    monkeypatch.setattr(F, "LR_BUMP_EXTRA", 12)
    monkeypatch.setattr(F, "LR_BUMP_MIN_GAIN", -1e9)
    monkeypatch.setattr(F, "SCHED", "cos")
    monkeypatch.setattr(F, "BUMP_RESCHED", False)
    lrs: list[float] = []

    class _Rec(torch.optim.Adam):
        def step(self, *a, **k):
            lrs.append(float(self.param_groups[0]["lr"]))
            return super().step(*a, **k)

    monkeypatch.setattr(F.torch.optim, "Adam", _Rec)
    f = CopySynthFitter(eng, y, 48000, tr)

    def loss(*_a, **_k):
        base = (f.w * 0.0).sum()
        one = torch.ones((), dtype=base.dtype)
        f._last_db = 3.0
        return base + 1.0, one * 0.2, one * 0.3, {}

    monkeypatch.setattr(f, "loss", loss)
    f.fit(10, 0.04, 10 ** 9, False)
    ups = sum(1 for a, b in zip(lrs, lrs[1:]) if b > a * (1 + 1e-9))
    assert ups > f._lr_bumps, (ups, f._lr_bumps)


def test_warmup_blocks_the_grid_change_transient_from_counting_as_a_plateau(_base, monkeypatch):
    """단계 초반 `LR_BUMP_WARMUP` 회까지는 상향하지 않는다 (§52.264).

    실측: 격자를 갈아 끼운 직후 손실이 잠깐 오르는 과도를 정체로 읽어 **정확히 `LR_BUMP_PATIENCE` 회째**에
    상향이 걸렸고(18 건 중 7 건), 포락을 3~6 점 잃었다.
    """
    eng, tr, y = _base
    monkeypatch.setattr(F, "LR_BUMP_PATIENCE", 3)
    monkeypatch.setattr(F, "LR_BUMP_MAX", 3)
    monkeypatch.setattr(F, "LR_BUMP_WARMUP", 30)
    f = CopySynthFitter(eng, y, 48000, tr)
    monkeypatch.setattr(f, "loss", _plateau(f, 70.0))
    f.fit(20, 0.04, 10 ** 9, False)            # 예산이 warmup 보다 짧다 — 한 번도 못 올린다
    assert f._lr_bumps == 0, f._lr_bumps


def test_warmup_still_lets_a_real_plateau_bump_later(_base, monkeypatch):
    """warmup 을 지난 뒤의 진짜 정체는 예전처럼 올린다 — 실측의 진짜 정체는 112~296 회에 걸린다."""
    eng, tr, y = _base
    monkeypatch.setattr(F, "LR_BUMP_PATIENCE", 3)
    monkeypatch.setattr(F, "LR_BUMP_MAX", 3)
    monkeypatch.setattr(F, "LR_BUMP_WARMUP", 10)
    f = CopySynthFitter(eng, y, 48000, tr)
    monkeypatch.setattr(f, "loss", _plateau(f, 70.0))
    f.fit(40, 0.04, 10 ** 9, False)
    assert f._lr_bumps == 1, f._lr_bumps


def test_warmup_off_by_default_keeps_the_old_timing(_base, monkeypatch):
    eng, tr, y = _base
    assert F.LR_BUMP_WARMUP == 0
    monkeypatch.setattr(F, "LR_BUMP_PATIENCE", 3)
    monkeypatch.setattr(F, "LR_BUMP_MAX", 3)
    f = CopySynthFitter(eng, y, 48000, tr)
    monkeypatch.setattr(f, "loss", _plateau(f, 70.0))
    f.fit(20, 0.04, 10 ** 9, False)
    assert f._lr_bumps == 1, f._lr_bumps


def test_a_restoring_bump_does_not_step_on_a_stale_graph(_base, monkeypatch):
    """되돌린 회차는 **걸음을 건너뛴다** — `_restore` 가 파라미터를 제자리에서 바꾸므로 그래프가 낡았다.

    주기별 배음 보정(`HCORR_K`)을 켜자 autograd 가 "inplace 로 바뀌었다" 며 죽었다. 그전에도 조용히
    낡은 기울기를 새 상태에 적용하고 있었다.
    """
    eng, tr, y = _base
    monkeypatch.setattr(F, "LR_BUMP_PATIENCE", 3)
    monkeypatch.setattr(F, "LR_BUMP_MAX", 1)
    monkeypatch.setattr(F, "LR_BUMP_EXTRA", 10)
    monkeypatch.setattr(F, "LR_BUMP_MIN_GAIN", -1e9)
    f = CopySynthFitter(eng, y, 48000, tr)
    extra = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    monkeypatch.setattr(f, "opt_params", lambda: [f.w, extra])
    calls = {"n": 0}

    def loss(*_a, **_k):
        calls["n"] += 1
        # `extra` 를 **그래프가 붙들게** 한다 (곱셈은 입력을 저장한다) — 제자리 변경이 잡히는 조건
        base = (extra * extra).sum() * 0.0 + (f.w * 0.0).sum()
        one = torch.ones((), dtype=base.dtype)
        f._last_db = 3.0
        return base + (1.0 if calls["n"] <= 1 else 1.02), one * 0.2, one * 0.3, {}

    monkeypatch.setattr(f, "loss", loss)
    f.fit(25, 0.04, 10 ** 9, False)          # 죽지 않아야 한다
    assert f._lr_bumps == 1, f._lr_bumps


def test_a_failed_bump_returns_to_the_best_state_and_the_pre_bump_lr(_base, monkeypatch):
    """상향이 헛돌아 멈출 때는 **그 자리를 버린다** — 최선 상태, 상향 전 lr, 빈 Adam 모멘트 (§52.473).

    예전에는 "상향 중단" 뒤 올린 lr(×2, ×4) 그대로 망가진 자리에서 이어 갔다. W18b(보정만 적합)는 그 자리에서
    포락 78 → 9 % 로 발산했다가 회복하며 4~9 kHz 에 +20~38 dB 를 남긴 채 끝났다.
    """
    eng, tr, y = _base
    monkeypatch.setattr(F, "LR_BUMP_PATIENCE", 3)
    monkeypatch.setattr(F, "LR_BUMP_MAX", 3)
    monkeypatch.setattr(F, "SCHED", "cos")
    lrs: list[float] = []

    class _Rec(torch.optim.Adam):
        def step(self, *a, **k):
            lrs.append(float(self.param_groups[0]["lr"]))
            return super().step(*a, **k)

    monkeypatch.setattr(F.torch.optim, "Adam", _Rec)
    f = CopySynthFitter(eng, y, 48000, tr)

    def loss(*_a, **_k):
        base = (f.w * 0.0).sum()
        one = torch.ones((), dtype=base.dtype)
        f._last_db = 3.0
        return base + 1.0, one * 0.2, one * 0.3, {}      # 포락 70 % 에서 꿈쩍 안 함 → 한 번 올리고 헛돌아 멈춘다

    monkeypatch.setattr(f, "loss", loss)
    f.fit(40, 0.04, 10 ** 9, False)
    assert f._lr_bumps == 1, f._lr_bumps
    up = next(i for i, (a, b) in enumerate(zip(lrs, lrs[1:])) if b > a * (1 + 1e-9)) + 1
    pre = lrs[up - 1]
    peak = max(lrs[up:])
    after = [v for i, v in enumerate(lrs) if i > up and v < peak * 0.99]    # 멈춘 뒤 (올린 값 아래로 내려온 뒤)
    assert after, lrs
    assert max(after) <= pre * (1 + 1e-9), (pre, max(after))


def test_no_bump_when_only_the_correction_is_fitted(_base, monkeypatch):
    """`FREEZE_CONTROLS` (보정만 적합) 에서는 lr 을 올리지 않는다."""
    eng, tr, y = _base
    monkeypatch.setattr(F, "LR_BUMP_PATIENCE", 3)
    monkeypatch.setattr(F, "LR_BUMP_MAX", 3)
    monkeypatch.setattr(F, "FREEZE_CONTROLS", True)
    f = CopySynthFitter(eng, y, 48000, tr)
    extra = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    monkeypatch.setattr(f, "_hc_leaves", lambda: [extra])

    def loss(*_a, **_k):
        base = (extra * 0.0).sum() + (f.w * 0.0).sum()
        one = torch.ones((), dtype=base.dtype)
        f._last_db = 3.0
        return base + 1.0, one * 0.2, one * 0.3, {}

    monkeypatch.setattr(f, "loss", loss)
    f.fit(20, 0.04, 10 ** 9, False)
    assert f._lr_bumps == 0, f._lr_bumps
