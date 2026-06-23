"""Unit tests for the deterministic plumbing inside
``episodiq.retrieval.tune.sweep``: the dotted-path shuffle-key
resolver, the Optuna parameter sampler, and the ``SweepReport``
selector logic. The async ``RetrievalSweep.run()`` itself needs a
Postgres-backed corpus and is covered separately by
``test_sweep_db.py``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from episodiq.retrieval.tune.sweep import (
    RetrievalSweepConfig, SweepReport, TrialResult,
    _resolve_shuffle_key, _sample_params,
)


# ----------------------------------------------------------------------
# _resolve_shuffle_key — dotted path resolver
# ----------------------------------------------------------------------


class TestResolveShuffleKey:
    """The CLI flag ``--shuffle-field`` is a dotted path; the resolver
    chains ``getattr`` / ``dict.get`` per segment and stringifies the
    final value so heterogeneous trajectories sort deterministically."""

    def test_top_level_attribute(self):
        obj = SimpleNamespace(id="42", meta={})
        assert _resolve_shuffle_key(obj, "id") == "42"

    def test_dotted_into_nested_dict(self):
        obj = SimpleNamespace(meta={"instance_id": "swe-7"})
        assert _resolve_shuffle_key(obj, "meta.instance_id") == "swe-7"

    def test_missing_segment_returns_empty(self):
        obj = SimpleNamespace(meta={"other": "x"})
        assert _resolve_shuffle_key(obj, "meta.instance_id") == ""

    def test_none_meta_returns_empty(self):
        obj = SimpleNamespace(meta=None)
        assert _resolve_shuffle_key(obj, "meta.instance_id") == ""

    def test_non_string_value_is_stringified(self):
        obj = SimpleNamespace(id=7)
        assert _resolve_shuffle_key(obj, "id") == "7"


# ----------------------------------------------------------------------
# SweepReport selectors — argmax / per-(W) / per-(W, agg, metric)
# ----------------------------------------------------------------------


def _trial(window: int, agg: str, metric: str, auc: float) -> TrialResult:
    """Minimal ``TrialResult`` — params/auc_per_metric stubbed empty,
    the selector logic only depends on (window, aggregation,
    target_metric, target_auc).
    """
    return TrialResult(
        window=window, aggregation=agg, params={},
        auc_per_metric={}, target_metric=metric, target_auc=auc,
    )


class TestSweepReport:

    def test_best_picks_global_max_target_auc(self):
        r = SweepReport(trials=[
            _trial(10, "mean", "cummax", 0.6),
            _trial(14, "min_distance", "cummeanmax", 0.7),
            _trial(10, "mean", "cummean", 0.5),
        ])
        assert r.best.target_auc == pytest.approx(0.7)
        assert r.best.window == 14

    def test_best_is_none_when_no_trials(self):
        assert SweepReport().best is None

    def test_by_window_filters_then_argmax(self):
        r = SweepReport(trials=[
            _trial(10, "mean", "cummax", 0.6),
            _trial(10, "min_distance", "cummax", 0.65),
            _trial(14, "mean", "cummax", 0.9),
        ])
        # The W=14 leader (0.9) must not leak into the W=10 selector.
        assert r.by_window(10).target_auc == pytest.approx(0.65)
        assert r.by_window(14).target_auc == pytest.approx(0.9)
        assert r.by_window(99) is None

    def test_by_window_per_metric_groups_by_metric(self):
        r = SweepReport(trials=[
            _trial(10, "mean", "cummax", 0.6),
            _trial(10, "min_distance", "cummax", 0.65),
            _trial(10, "mean", "cummean", 0.4),
            _trial(10, "min_distance", "cummean", 0.5),
            _trial(10, "mean", "cummeanmax", 0.7),
        ])
        per_m = r.by_window_per_metric(10)
        assert per_m["cummax"].target_auc == pytest.approx(0.65)
        assert per_m["cummean"].target_auc == pytest.approx(0.5)
        assert per_m["cummeanmax"].target_auc == pytest.approx(0.7)

    def test_by_window_agg_per_metric_restricts_to_slot(self):
        r = SweepReport(trials=[
            _trial(10, "mean", "cummax", 0.6),          # in slot
            _trial(10, "min_distance", "cummax", 0.8),  # wrong agg
            _trial(10, "mean", "cummax", 0.5),          # in slot, lower
            _trial(14, "mean", "cummax", 0.9),          # wrong window
        ])
        slot = r.by_window_agg_per_metric(10, "mean")
        # 0.8 (min_distance) and 0.9 (W=14) must be excluded.
        assert slot["cummax"].target_auc == pytest.approx(0.6)

    def test_by_window_agg_per_metric_empty_slot(self):
        r = SweepReport(trials=[_trial(10, "mean", "cummax", 0.6)])
        assert r.by_window_agg_per_metric(10, "min_distance") == {}


# ----------------------------------------------------------------------
# _sample_params — Optuna param flow (mocked trial)
# ----------------------------------------------------------------------


def _mock_trial(return_value_by_name: dict[str, object]):
    """Build a mock Optuna trial that returns canned values for each
    sampling call AND records every ``suggest_*`` invocation so the
    test can assert ranges/choices/steps verbatim.
    """
    trial = MagicMock()
    calls: list[tuple[str, str, tuple, dict]] = []

    def _make(kind: str):
        def _suggest(name, *args, **kwargs):
            calls.append((kind, name, args, kwargs))
            return return_value_by_name[name]
        return _suggest

    trial.suggest_int = MagicMock(side_effect=_make("int"))
    trial.suggest_float = MagicMock(side_effect=_make("float"))
    trial.suggest_categorical = MagicMock(side_effect=_make("categorical"))
    trial._calls = calls  # so tests can introspect
    return trial


class TestSampleParams:
    """``_sample_params`` is the only place Optuna sees the search
    space — these tests pin the exact ranges/choices/steps for each
    knob and verify the resulting cascade configs carry the sampled
    values into ``RetrievalConfig`` / ``AggShiftConfig``."""

    def _canned(self):
        # One value per sampled name; chosen on the grid to avoid
        # round-off surprises (lam = 1.7 → round(1.7, 1) = 1.7).
        return {
            "prefetch_n_uniq": 120,
            "jaccard_n_uniq": 80,
            "top_k": 17,
            "penalty_shape": "lin",
            "lam": 1.7,
            "gap_open": 1.3,
            "gap_extend": 0.6,
            "sigma": 2.4,
            "metric": "cummeanmax",
        }

    def test_returns_configs_carrying_sampled_values(self):
        trial = _mock_trial(self._canned())
        cas_cfg, ms_cfg, metric, params = _sample_params(
            trial, W=10, agg="min_distance", cfg=RetrievalSweepConfig(),
        )
        # Outer-slot agg is FIXED, never sampled — sweep loop owns it.
        assert cas_cfg.aggregation == "min_distance"
        # Integer knobs flow into RetrievalConfig.
        assert cas_cfg.prefetch_n_uniq == 120
        assert cas_cfg.jaccard_n_uniq == 80
        assert cas_cfg.top_k == 17
        # Min-shift floats flow into AggShiftConfig, rounded to 0.1
        # so the AggShiftCache key is stable across adjacent trials.
        assert ms_cfg.window == 10
        assert ms_cfg.lam == pytest.approx(1.7)
        assert ms_cfg.gap_open == pytest.approx(1.3)
        assert ms_cfg.gap_extend == pytest.approx(0.6)
        assert ms_cfg.sigma == pytest.approx(2.4)
        assert ms_cfg.penalty_shape == "lin"
        # ``metric`` is returned separately — it's the Optuna objective,
        # NOT part of RetrievalConfig / AggShiftConfig.
        assert metric == "cummeanmax"
        # ``params`` round-trips the raw samples for CSV writing.
        assert params == self._canned()

    def test_ranges_passed_to_trial_match_cfg(self):
        """Pin the exact ``(low, high)`` ranges the sweep sees so a
        regression in constants/CLI plumbing fails this unit test
        instead of polluting trials.csv with off-grid samples.
        """
        cfg = RetrievalSweepConfig()
        trial = _mock_trial(self._canned())
        _sample_params(trial, W=10, agg="mean", cfg=cfg)

        by_name = {c[1]: c for c in trial._calls}
        # Integer knobs — step=10 for the count-style params, default
        # step for top_k.
        assert by_name["prefetch_n_uniq"][2] == cfg.prefetch_n_uniq_range
        assert by_name["prefetch_n_uniq"][3] == {"step": 10}
        assert by_name["jaccard_n_uniq"][2] == cfg.jaccard_n_uniq_range
        assert by_name["jaccard_n_uniq"][3] == {"step": 10}
        assert by_name["top_k"][2] == cfg.top_k_range
        # Float knobs — uniform step=0.1 (caller rounds to dodge FP drift).
        for fname, rng in (
            ("lam", cfg.lam_range),
            ("gap_open", cfg.gap_open_range),
            ("gap_extend", cfg.gap_extend_range),
            ("sigma", cfg.sigma_range),
        ):
            assert by_name[fname][2] == rng
            assert by_name[fname][3] == {"step": 0.1}
        # Categorical knobs — exact choice lists.
        assert by_name["penalty_shape"][2][0] == list(
            cfg.penalty_shape_choices,
        )
        assert by_name["metric"][2][0] == [
            "cummax", "cummean", "cummeanmax",
        ]

    def test_float_rounding_collapses_optuna_drift(self):
        """Optuna's ``step=0.1`` can drift (``0.1·7 == 0.7000000000…1``);
        the sampler rounds to 0.1 so AggShiftConfig hashes stay
        cache-friendly across adjacent trials.
        """
        drifted = self._canned()
        drifted["lam"] = 0.1 * 7  # = 0.7000000000000001
        trial = _mock_trial(drifted)
        _cas, ms_cfg, _m, params = _sample_params(
            trial, W=10, agg="mean", cfg=RetrievalSweepConfig(),
        )
        # Both the round-tripped param AND the AggShiftConfig field
        # land exactly on the 0.1 grid.
        assert params["lam"] == pytest.approx(0.7)
        assert ms_cfg.lam == pytest.approx(0.7)


# ----------------------------------------------------------------------
# _run_study — TPESampler + create_study construction (mocked Optuna)
# ----------------------------------------------------------------------


class TestRunStudySamplerWiring:
    """``_run_study`` is the only place sweep configures Optuna: it
    builds a ``TPESampler(seed, multivariate)`` and a maximisation
    study. Patch ``optuna.create_study`` / ``optuna.samplers.TPESampler``
    and assert the kwargs sweep passes — guards against silent regressions
    where ``cfg.multivariate`` / ``cfg.optuna_seed`` drop out of the call.
    """

    @pytest.mark.asyncio
    async def test_tpesampler_gets_seed_offset_and_multivariate(
        self, monkeypatch,
    ):
        import episodiq.retrieval.tune.sweep as sweep_mod

        sampler_calls = []
        study_calls = []

        # No-op TPESampler that records its kwargs.
        def fake_sampler(**kwargs):
            sampler_calls.append(kwargs)
            return MagicMock(name="fake_sampler")

        # Stub Study with .trials = [] (used by early-stop logger) and
        # .ask() / .tell() that no-ops; we exit immediately via 0 trials.
        def fake_create_study(**kwargs):
            study_calls.append(kwargs)
            study = MagicMock()
            study.trials = []
            return study

        monkeypatch.setattr(sweep_mod.optuna.samplers, "TPESampler", fake_sampler)
        monkeypatch.setattr(sweep_mod.optuna, "create_study", fake_create_study)

        cfg = RetrievalSweepConfig(
            optuna_seed=1337, multivariate=True, n_trials_per_window=0,
        )
        results = await sweep_mod._run_study(
            ai=2,  # offset bumps seed by 2
            W=10,
            agg="min_distance",
            mh_cfg=MagicMock(),
            cfg=cfg,
            snapshots=[],
            status={},
            session_factory=MagicMock(),
            table_name="tbl",
            cache=MagicMock(),
        )

        # Zero trials → empty results, but the sampler + study STILL
        # had to be built before _one_trial fanned out.
        assert results == []
        assert sampler_calls == [{
            "seed": 1337 + 2,
            "multivariate": True,
        }]
        # ``direction="maximize"`` is non-negotiable — flipping it would
        # silently make the sweep pick the WORST config.
        assert len(study_calls) == 1
        assert study_calls[0]["direction"] == "maximize"
        # The study must receive the sampler instance we built — no
        # silent default-sampler fallback.
        assert "sampler" in study_calls[0]

    @pytest.mark.asyncio
    async def test_multivariate_false_propagates(self, monkeypatch):
        import episodiq.retrieval.tune.sweep as sweep_mod

        sampler_calls = []

        def fake_sampler(**kwargs):
            sampler_calls.append(kwargs)
            return MagicMock()

        def fake_create_study(**kwargs):
            s = MagicMock()
            s.trials = []
            return s

        monkeypatch.setattr(sweep_mod.optuna.samplers, "TPESampler", fake_sampler)
        monkeypatch.setattr(sweep_mod.optuna, "create_study", fake_create_study)

        cfg = RetrievalSweepConfig(
            optuna_seed=7, multivariate=False, n_trials_per_window=0,
        )
        await sweep_mod._run_study(
            ai=0, W=14, agg="mean",
            mh_cfg=MagicMock(), cfg=cfg, snapshots=[], status={},
            session_factory=MagicMock(), table_name="tbl",
            cache=MagicMock(),
        )
        assert sampler_calls == [{"seed": 7, "multivariate": False}]
