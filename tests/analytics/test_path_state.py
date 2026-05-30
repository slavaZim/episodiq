"""Tests for PathStateCalculator."""

from dataclasses import dataclass, field

from episodiq.analytics.path_state import PathStateCalculator
from episodiq.retrieval.minhash import MinHasher


@dataclass
class FakePath:
    """Stand-in for TrajectoryPath."""
    trace: list[str] = field(default_factory=list)
    trace_tokens: list[int] | None = None
    from_obs_label: str = ""
    action_label: str | None = None
    to_obs_label: str | None = None


def _calc(n: int = 3) -> PathStateCalculator:
    # Small K + seed for deterministic signatures.
    return PathStateCalculator(minhasher=MinHasher(k=16, seed=1), ngram_n=n)


class TestGranularStep:
    def test_first_observation(self):
        """No prev_path → trace = [obs_label]."""
        trace = _calc().granular_step(None, "o:text:greeting")
        assert trace == ["o:text:greeting"]

    def test_completed_triplet_extends_trace(self):
        """prev_path with an action label → trace extends with [action, obs]."""
        prev = FakePath(
            trace=["o:text:greeting"],
            action_label="a:text:response",
        )
        trace = _calc().granular_step(prev, "o:text:followup")
        assert trace == ["o:text:greeting", "a:text:response", "o:text:followup"]

    def test_no_action_label_yields_single_obs(self):
        """prev_path without an action label → trace is just [obs_label]."""
        prev = FakePath(trace=["o:text:foo"], action_label=None)
        trace = _calc().granular_step(prev, "o:text:bar")
        assert trace == ["o:text:bar"]


class TestTokenStep:
    def test_first_token_no_prev(self):
        """No prev_path → tokens = [token], signature is None until n-gram window fills."""
        tokens, sig = _calc(n=3).token_step(None, 5)
        assert tokens == [5]
        assert sig is None

    def test_signature_appears_once_window_fills(self):
        """token_step yields a signature once the trace reaches n tokens."""
        calc = _calc(n=3)
        prev = FakePath(trace_tokens=[1, 2])
        tokens, sig = calc.token_step(prev, 3)
        assert tokens == [1, 2, 3]
        assert sig is not None
        assert len(sig) == 16  # MinHasher k=16

    def test_noise_kept_in_tokens(self):
        """Noise ordinal (-1) is preserved in the token trace like any other."""
        prev = FakePath(trace_tokens=[1, 2])
        tokens, _ = _calc().token_step(prev, -1)
        assert tokens == [1, 2, -1]

    def test_signature_is_deterministic(self):
        """Same calc + same prev + same token → same signature."""
        calc = _calc(n=3)
        prev = FakePath(trace_tokens=[1, 2])
        _, sig_a = calc.token_step(prev, 3)
        _, sig_b = calc.token_step(prev, 3)
        assert sig_a == sig_b

    def test_distinct_traces_yield_distinct_signatures(self):
        """Different trace content → different signatures (with high probability)."""
        calc = _calc(n=3)
        _, sig_a = calc.token_step(FakePath(trace_tokens=[1, 2]), 3)
        _, sig_b = calc.token_step(FakePath(trace_tokens=[9, 8]), 7)
        assert sig_a != sig_b
