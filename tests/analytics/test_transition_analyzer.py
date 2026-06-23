"""Unit tests for the post-refactor TransitionAnalyzer (loop + path-frequency
+ fail_similarity signals from a pre-retrieved candidate list).
"""

from dataclasses import dataclass, field
from uuid import UUID, uuid4

from episodiq.analytics.transition_analyzer import TransitionAnalyzer
from episodiq.config.config import AnalyticsConfig
from episodiq.retrieval.candidate import RetrievalCandidate


@dataclass
class FakePath:
    """Stand-in for TrajectoryPath in analyzer.analyze()."""

    trace: list[str] = field(default_factory=list)
    data: dict | None = None


def _candidate(
    action_cluster_id: UUID | None,
    status: str = "success",
) -> RetrievalCandidate:
    return RetrievalCandidate(
        trajectory_id=uuid4(),
        score=1.0,
        best_path_action_cluster_id=action_cluster_id,
        trajectory_status=status,
    )


def _analyzer(loop_threshold: int = 3) -> TransitionAnalyzer:
    cfg = AnalyticsConfig(
        loop_threshold=loop_threshold,
        low_entropy=0.0,
        high_entropy=10.0,
    )
    return TransitionAnalyzer(config=cfg)


class TestLoopSignal:
    def test_no_candidates_short_trace_no_loop_signal(self):
        analytics = _analyzer().analyze(FakePath(trace=["o:foo"]), [])
        assert analytics.loop_signal is None

    def test_loop_below_threshold_not_detected(self):
        # trace has only one duplet → streak=1 < threshold(3)
        trace = ["o:a", "a:b"]
        analytics = _analyzer().analyze(FakePath(trace=trace), [])
        assert analytics.loop_signal is not None
        assert analytics.loop_signal.streak == 1
        assert analytics.loop_signal.is_detected is False

    def test_loop_meets_threshold_detected(self):
        # trailing duplet "a:b.o:c" repeats 3 times -> streak=3
        trace = ["a:b", "o:c", "a:b", "o:c", "a:b", "o:c"]
        analytics = _analyzer(loop_threshold=3).analyze(FakePath(trace=trace), [])
        assert analytics.loop_signal.streak == 3
        assert analytics.loop_signal.is_detected is True
        assert analytics.loop_signal.duplet == "a:b.o:c"


class TestPathFrequency:
    def test_empty_candidates_no_signal(self):
        analytics = _analyzer().analyze(FakePath(trace=["o:foo"]), [])
        assert analytics.path_frequency_signal is None

    def test_all_candidates_share_action_zero_entropy(self):
        action_cid = uuid4()
        cands = [_candidate(action_cid) for _ in range(4)]
        analytics = _analyzer().analyze(FakePath(trace=["o:foo"]), cands)
        sig = analytics.path_frequency_signal
        assert sig is not None
        assert sig.entropy == 0.0
        assert sig.n_matches == 4

    def test_evenly_split_actions_maximum_entropy(self):
        # 2 + 2 split → 1 bit of entropy
        a, b = uuid4(), uuid4()
        cands = [_candidate(a), _candidate(a), _candidate(b), _candidate(b)]
        sig = _analyzer().analyze(FakePath(trace=["o:foo"]), cands).path_frequency_signal
        assert sig.entropy == 1.0
        assert sig.n_matches == 4

    def test_candidates_without_action_cluster_id_are_ignored(self):
        a = uuid4()
        cands = [_candidate(a), _candidate(None), _candidate(None)]
        sig = _analyzer().analyze(FakePath(trace=["o:foo"]), cands).path_frequency_signal
        # only the one with a cluster id is counted → single-class → zero entropy
        assert sig.entropy == 0.0
        assert sig.n_matches == 1


class TestFailSimilarity:
    def test_no_candidates_no_prev_returns_none(self):
        analytics = _analyzer().analyze(FakePath(trace=["o:foo"]), [])
        assert analytics.fail_similarity is None

    def test_no_candidates_carries_prev_forward(self):
        """A path with empty retrieval doesn't contribute — the running
        aggregate from the previous path is preserved verbatim.
        """
        prev = FakePath(data={
            "fail_similarity": {
                "current": 0.3, "cummax": 0.5, "cummean": 0.4,
                "cummeanmax": 0.4, "_count": 2,
            },
        })
        analytics = _analyzer().analyze(FakePath(trace=["o:foo"]), [], prev)
        assert analytics.fail_similarity == prev.data["fail_similarity"]

    def test_first_contributor_initializes_dict(self):
        cands = [
            _candidate(uuid4(), status="failure"),
            _candidate(uuid4(), status="success"),
        ]
        analytics = _analyzer().analyze(FakePath(trace=["o:foo"]), cands)
        assert analytics.fail_similarity == {
            "current": 0.5, "cummax": 0.5, "cummean": 0.5,
            "cummeanmax": 0.5, "_count": 1,
        }

    def test_rolls_aggregates_with_prev(self):
        """prev count=2 mean=0.4 max=0.5 cummeanmax=0.45; current=0.7 →
        new mean = (0.4*2 + 0.7)/3 = 0.5; cummax = 0.7;
        cummeanmax = max(0.45, 0.5) = 0.5; count = 3.
        """
        prev = FakePath(data={
            "fail_similarity": {
                "current": 0.3, "cummax": 0.5, "cummean": 0.4,
                "cummeanmax": 0.45, "_count": 2,
            },
        })
        cands = [_candidate(uuid4(), status="failure")] * 7 + [
            _candidate(uuid4(), status="success") for _ in range(3)
        ]
        analytics = _analyzer().analyze(
            FakePath(trace=["o:foo"]), cands, prev,
        )
        sim = analytics.fail_similarity
        assert sim["current"] == 0.7
        assert sim["cummax"] == 0.7
        assert sim["cummean"] == 0.5
        assert sim["cummeanmax"] == 0.5
        assert sim["_count"] == 3
