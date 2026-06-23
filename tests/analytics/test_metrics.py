"""Unit tests for the shared metric utilities."""

from uuid import uuid4

import numpy as np
import pytest

from episodiq.analytics.metrics import (
    SIMILARITY_METRICS, AucCI, MetricCurve, StepAUC,
    aggregate_ffs, bootstrap_aucs, compute_metric_curves, weighted_aucs,
)


# ----------------------------------------------------------------------
# aggregate_ffs — pure fold over fail_sim sequence
# ----------------------------------------------------------------------


class TestAggregateFfs:

    def test_known_sequence(self):
        # ffs = [0.2, 0.8, 0.4]
        # cummax = 0.8
        # cummean = (0.2 + 0.8 + 0.4) / 3 = 0.4666...
        # cumulative mean curve = [0.2, 0.5, 0.4666] → max = 0.5
        ffs = np.array([0.2, 0.8, 0.4], dtype=np.float64)
        cummax, cummean, cummeanmax = aggregate_ffs(ffs)
        assert cummax == pytest.approx(0.8)
        assert cummean == pytest.approx(1.4 / 3.0)
        assert cummeanmax == pytest.approx(0.5)

    def test_single_element_collapses(self):
        ffs = np.array([0.7], dtype=np.float64)
        cm, cmean, cmmax = aggregate_ffs(ffs)
        assert cm == cmean == cmmax == pytest.approx(0.7)

    def test_monotone_increasing(self):
        ffs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        cm, cmean, cmmax = aggregate_ffs(ffs)
        assert cm == pytest.approx(0.4)
        assert cmean == pytest.approx(0.25)
        # cummeanmax tracks max of running mean → equals final mean
        # when the sequence is non-decreasing.
        assert cmmax == pytest.approx(0.25)


# ----------------------------------------------------------------------
# weighted_aucs — scalar per metric (what the sweep stores per trial)
# ----------------------------------------------------------------------


def _two_traj_fixture():
    """Two trajectories with cleanly separable fail_sims so every per-
    step AUC = 1.0 — used by the AUC tests below."""
    fail_tid = uuid4()
    succ_tid = uuid4()
    fail_sims = {
        fail_tid: [(50, 0.9), (51, 0.95), (52, 0.92)],
        succ_tid: [(50, 0.1), (51, 0.05)],
    }
    status = {fail_tid: "failure", succ_tid: "success"}
    return fail_sims, status


class TestWeightedAucs:

    def test_returns_one_for_perfect_separation(self):
        fail_sims, status = _two_traj_fixture()
        out = weighted_aucs(fail_sims, status, eval_min_step=50)
        assert set(out) == set(SIMILARITY_METRICS)
        for m in SIMILARITY_METRICS:
            assert out[m] == pytest.approx(1.0)

    def test_weighting_by_n_active_per_step(self):
        """Weighted AUC = Σ(auc_s · n_active_s) / Σ n_active_s, NOT
        a plain mean across steps. Construct a fixture where:

          step 50 — 3 active trajs, AUC = 1.0 (perfect rank)
          step 51 — 2 active trajs, AUC = 0.5 (tied scores)

        Plain mean would yield 0.75. Correct weighting yields
        ``(1.0·3 + 0.5·2) / 5 = 0.8``. This test fails loud if anyone
        switches the reduction to ``np.mean(per_step_aucs)``.
        """
        succ_tid = uuid4()
        fail_a = uuid4()  # alive at both steps
        fail_b = uuid4()  # drops out after step 50

        # At step 50 cummax of the prefix [(50, X)] = X.
        # Choose X so positives (fail_a=0.9, fail_b=0.7) outrank
        # the negative (succ=0.1) → AUC = 1.0 across 3 trajs.
        # At step 51 only succ_tid and fail_a remain (fail_b is gone).
        # Their cummax over [(50, X), (51, Y)] is max(X, Y):
        #   succ: max(0.1, 0.9) = 0.9
        #   fail_a: max(0.9, 0.5) = 0.9
        # → tied scores → AUC = 0.5 across 2 trajs.
        fail_sims = {
            succ_tid: [(50, 0.1), (51, 0.9)],
            fail_a:   [(50, 0.9), (51, 0.5)],
            fail_b:   [(50, 0.7)],
        }
        status = {
            succ_tid: "success", fail_a: "failure", fail_b: "failure",
        }
        out = weighted_aucs(fail_sims, status, eval_min_step=50)
        # 1.0·3 + 0.5·2 = 4.0; total weight = 5 → 0.8.
        assert out["cummax"] == pytest.approx(0.8)

    def test_empty_fail_sims_returns_empty(self):
        assert weighted_aucs({}, {}, eval_min_step=50) == {}

    def test_eval_min_step_above_max_returns_empty(self):
        fail_sims, status = _two_traj_fixture()
        # Max step = 52; ask for ≥100 → no curves contribute.
        assert weighted_aucs(fail_sims, status, eval_min_step=100) == {}

    def test_single_class_step_skipped(self):
        # No step has both a failure and a success → AUC undefined,
        # weighted output is empty.
        only_fail = uuid4()
        only_succ_late = uuid4()
        fail_sims = {
            only_fail: [(50, 0.9)],
            only_succ_late: [(51, 0.1)],
        }
        status = {only_fail: "failure", only_succ_late: "success"}
        out = weighted_aucs(fail_sims, status, eval_min_step=50)
        assert out == {}


# ----------------------------------------------------------------------
# compute_metric_curves — same iteration + retains per-step curves
# ----------------------------------------------------------------------


class TestComputeMetricCurves:

    def test_returns_curves_for_all_metrics_when_signal(self):
        fail_sims, status = _two_traj_fixture()
        curves = compute_metric_curves(fail_sims, status, eval_min_step=50)
        assert set(curves) == set(SIMILARITY_METRICS)
        for m in SIMILARITY_METRICS:
            assert isinstance(curves[m], MetricCurve)
            assert curves[m].weighted_auc == pytest.approx(1.0)
            # Steps 50 and 51 have both classes; step 52 has only
            # the failure → skipped by the ``len(set(y)) < 2`` guard.
            assert len(curves[m].per_step) == 2

    def test_per_step_record_shape(self):
        fail_sims, status = _two_traj_fixture()
        curves = compute_metric_curves(fail_sims, status, eval_min_step=50)
        sa = curves["cummax"].per_step[0]
        assert isinstance(sa, StepAUC)
        assert sa.step == 50
        assert sa.auc == pytest.approx(1.0)
        assert sa.n_active == 2

    def test_scalar_matches_compute_metric_curves_weighted(self):
        """``weighted_aucs`` and ``compute_metric_curves`` share one
        internal iterator — their scalars must match exactly,
        otherwise the wrappers have split out of sync."""
        fail_sims, status = _two_traj_fixture()
        scalar = weighted_aucs(fail_sims, status, eval_min_step=50)
        full = compute_metric_curves(fail_sims, status, eval_min_step=50)
        for m in SIMILARITY_METRICS:
            assert scalar[m] == pytest.approx(full[m].weighted_auc)


# ----------------------------------------------------------------------
# bootstrap_aucs — per-trajectory resample CI
# ----------------------------------------------------------------------


def _separable_fixture(n_fail: int = 30, n_succ: int = 30):
    """Build a fixture with cleanly separable fail_sims: failures
    cluster around 0.9, successes around 0.1. Bootstrap should converge
    to AUC ≈ 1.0 with a narrow CI for n=60.
    """
    fail_sims: dict = {}
    status: dict = {}
    for i in range(n_fail):
        tid = uuid4()
        fail_sims[tid] = [(50, 0.85 + 0.001 * i), (51, 0.9 + 0.001 * i)]
        status[tid] = "failure"
    for i in range(n_succ):
        tid = uuid4()
        fail_sims[tid] = [(50, 0.1 + 0.001 * i), (51, 0.15 + 0.001 * i)]
        status[tid] = "success"
    return fail_sims, status


class TestBootstrapAucs:

    def test_returns_per_metric_ci(self):
        fail_sims, status = _separable_fixture()
        cis = bootstrap_aucs(
            fail_sims, status, eval_min_step=50,
            n_boot=100, seed=0,
        )
        # All three metrics should produce a CI for separable signal.
        assert set(cis) == set(SIMILARITY_METRICS)
        for m in SIMILARITY_METRICS:
            ci = cis[m]
            assert isinstance(ci, AucCI)
            assert 0.0 <= ci.lo <= ci.hi <= 1.0
            # Bounds straddle the bootstrap mean.
            assert ci.lo <= ci.mean <= ci.hi

    def test_perfectly_separable_signal_has_tight_band(self):
        """With cleanly separable fail/success populations, every
        bootstrap draw should hit AUC ≈ 1.0 → narrow CI."""
        fail_sims, status = _separable_fixture(n_fail=40, n_succ=40)
        cis = bootstrap_aucs(
            fail_sims, status, eval_min_step=50,
            n_boot=200, seed=0,
        )
        for m in SIMILARITY_METRICS:
            width = cis[m].hi - cis[m].lo
            # Separable signal + n=80 trajs → CI width ≲ 0.05.
            assert width < 0.05, (
                f"{m}: CI [{cis[m].lo:.3f}, {cis[m].hi:.3f}] too wide"
            )
            assert cis[m].mean > 0.95

    def test_mean_close_to_point_estimate(self):
        """Bootstrap mean per metric should land near the point
        estimate from ``weighted_aucs`` — confirms the resampling
        loop computes the same AUC, just over draws."""
        fail_sims, status = _separable_fixture(n_fail=20, n_succ=20)
        point = weighted_aucs(fail_sims, status, eval_min_step=50)
        cis = bootstrap_aucs(
            fail_sims, status, eval_min_step=50,
            n_boot=200, seed=0,
        )
        for m in SIMILARITY_METRICS:
            # Within ±0.05 of the point estimate (bootstrap variance
            # on n=40 trajs is sub-0.05 for separable signal).
            assert cis[m].mean == pytest.approx(point[m], abs=0.05)

    def test_empty_fail_sims_returns_empty(self):
        assert bootstrap_aucs({}, {}, eval_min_step=50) == {}

    def test_reproducible_under_fixed_seed(self):
        fail_sims, status = _separable_fixture(n_fail=10, n_succ=10)
        a = bootstrap_aucs(
            fail_sims, status, eval_min_step=50, n_boot=50, seed=7,
        )
        b = bootstrap_aucs(
            fail_sims, status, eval_min_step=50, n_boot=50, seed=7,
        )
        for m in SIMILARITY_METRICS:
            assert a[m] == b[m]
