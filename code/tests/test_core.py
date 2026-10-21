import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from faithaal import interventions as iv  # noqa: E402
from faithaal.data import Event  # noqa: E402
from faithaal.ladder import authority  # noqa: E402
from faithaal.prompts import render_log  # noqa: E402

EV = [Event(0.0, "M01", "ON"), Event(1.0, "I01", "ABSENT"), Event(2.0, "AD1-B", "0.5"),
      Event(3.0, "E01", "STOP_INSTRUCT"), Event(4.0, "M01", "OFF")]


def test_remove_and_keep_are_complementary():
    assert [e.sensor for e in iv.remove(EV, {1, 2})] == ["M01", "E01", "M01"]
    assert [e.sensor for e in iv.keep_only(EV, {1, 2})] == ["I01", "AD1-B"]


def test_value_change():
    out = iv.change_value(EV, {0, 1, 2, 3})
    assert [e.msg for e in out] == ["OFF", "PRESENT", "0.0", "STOP_INSTRUCT"] + ["OFF"][:0] + [out[4].msg]
    assert out[4].msg == "OFF"  # untouched
    assert iv.effective_value_changes(EV, {0, 3}) == 1


def test_shift_reorders_and_is_stable():
    out = iv.shift_time(EV, {0}, delta=2.5)
    assert [e.sensor for e in out] == ["I01", "AD1-B", "M01", "E01", "M01"]


def test_uncited_sample_is_disjoint_and_matched():
    s = iv.sample_uncited(20, {1, 2, 3}, np.random.default_rng(0))
    assert len(s) == 3 and not (s & {1, 2, 3})


def test_render_is_one_based_and_relative():
    first = render_log(EV).splitlines()[0]
    assert first.startswith("[1] 0.0s | M01 | ON")


def test_authority_monotone_and_gated():
    ref = 0.1
    assert authority(0.99, 0.2, 0.0, 1.0, ref) == 3
    assert authority(0.99, 0.0, 0.0, 1.0, ref) == 1      # accurate but explanations do not track evidence
    assert authority(0.85, 0.2, 0.0, 1.0, ref) == 1
    assert authority(0.99, 0.2, 0.3, 1.0, ref) == 0      # unstable under a null change
    assert authority(0.70, 0.2, 0.0, 1.0, ref) == 0
