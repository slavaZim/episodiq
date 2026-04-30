"""Tests for PathStateCalculator."""

from dataclasses import dataclass, field

import pytest

from episodiq.analytics.path_state import PathStateCalculator
from episodiq.utils import transition_profile_to_vector


@dataclass
class FakePath:
    """Stand-in for TrajectoryPath."""
    trace: list[str] = field(default_factory=list)
    transition_profile: dict[str, float] | None = None
    from_obs_label: str = ""
    action_label: str | None = None
    to_obs_label: str | None = None


class TestPathStateCalculator:
    def test_first_observation(self):
        """No prev_path → empty profile, trace = [obs_label]."""
        calc = PathStateCalculator()
        profile, embed, trace = calc.step(None, "o:text:greeting")

        assert profile is None
        assert embed is None
        assert trace == ["o:text:greeting"]

    def test_second_observation(self):
        """prev_path with completed triplet → profile has one transition."""
        calc = PathStateCalculator()
        prev = FakePath(
            trace=["o:text:greeting"],
            from_obs_label="o:text:greeting",
            action_label="a:text:response",
            to_obs_label="o:text:followup",
        )
        profile, embed, trace = calc.step(prev, "o:text:followup")

        expected_key = "o:text:greeting.a:text:response.o:text:followup"
        assert profile == {expected_key: 1.0}
        assert trace == ["o:text:greeting", "a:text:response", "o:text:followup"]
        assert embed == transition_profile_to_vector(profile)

    def test_decay_on_third_observation(self):
        """Third observation — previous profile gets decayed."""
        calc = PathStateCalculator()
        first_key = "o:A.a:X.o:B"
        prev = FakePath(
            trace=["o:A", "a:X", "o:B"],
            transition_profile={first_key: 1.0},
            from_obs_label="o:B",
            action_label="a:Y",
            to_obs_label="o:C",
        )
        profile, embed, trace = calc.step(prev, "o:C")

        assert trace == ["o:A", "a:X", "o:B", "a:Y", "o:C"]
        assert profile[first_key] == pytest.approx(0.8)
        assert profile["o:B.a:Y.o:C"] == 1.0

    def test_same_transition_accumulates(self):
        """Repeating the same transition accumulates count."""
        calc = PathStateCalculator()
        key = "o:A.a:X.o:A"
        prev = FakePath(
            trace=["o:A", "a:X", "o:A"],
            transition_profile={key: 1.0},
            from_obs_label="o:A",
            action_label="a:X",
            to_obs_label="o:A",
        )
        profile, _, _ = calc.step(prev, "o:A")

        # decayed old (0.8) + new (1.0) = 1.8
        assert profile[key] == pytest.approx(1.8)

    def test_prev_path_without_action_treated_as_first(self):
        """prev_path with no action (incomplete triplet) → same as first obs."""
        calc = PathStateCalculator()
        prev = FakePath(
            trace=["o:A"],
            from_obs_label="o:A",
            action_label=None,
            to_obs_label=None,
        )
        profile, embed, trace = calc.step(prev, "o:B")

        assert profile is None
        assert embed is None
        assert trace == ["o:B"]

    def test_custom_decay_lambda(self):
        calc = PathStateCalculator(decay_lambda=0.5)
        key = "o:A.a:X.o:B"
        prev = FakePath(
            trace=["o:A"],
            transition_profile={key: 2.0},
            from_obs_label="o:A",
            action_label="a:X",
            to_obs_label="o:B",
        )
        profile, _, _ = calc.step(prev, "o:B")

        assert profile[key] == pytest.approx(0.5 * 2.0 + 1.0)
