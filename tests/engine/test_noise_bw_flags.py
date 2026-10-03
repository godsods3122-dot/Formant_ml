"""잡음 가지 공진 폭 손잡이 (`tract.NOISE_BW_REL`, `NOISE_EXTRA_BW`) — 기본값을 지키고 깃발로만 바꾼다 (§52.246)."""
import subprocess
import sys

from formant_ml.engine import tract as T


def test_defaults_are_unchanged():
    assert T.NOISE_BW_REL == 0.15
    assert T.NOISE_EXTRA_BW == 2.5


