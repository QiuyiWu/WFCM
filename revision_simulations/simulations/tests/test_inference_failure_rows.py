from types import SimpleNamespace

import numpy as np

import simulations.inference_experiment as experiment
from simulations.inference_experiment import BootstrapIntervalFailure, Profile
from simulations.wfcm_core import WFCMParams


def _profile() -> Profile:
    return Profile(
        coverage_reps=1,
        bootstrap_reps=5,
        lrt_reps=1,
        lrt_bootstrap_reps=5,
        selection_reps=1,
        coverage_ns=(20,),
        lrt_ns=(20,),
        selection_ns=(20,),
        fit_n_init=2,
        maxiter=20,
        fisher_draws=100,
    )


def _fit(k: int = 2, *, success: bool = True):
    centers = np.linspace(-2.0, 2.0, k)[:, None]
    params = WFCMParams(
        sigma=1.0,
        centers=centers,
        weights=np.full(k, 1.0 / k),
        m=2.0,
    )
    return SimpleNamespace(
        params=params,
        success=success,
        optimizer_success=success,
        message="converged" if success else "deliberate nonconvergence",
        gradient_inf_norm=1e-7,
        average_nll=1.0,
        diagnostics={"ess": 80.0, "ess_fraction": 0.8},
    )


def _fisher(fit, _n, **_kwargs):
    centers = fit.params.centers[:, 0]
    diagnostics = {
        "fisher_min_eigenvalue": 0.5,
        "fisher_max_eigenvalue": 2.0,
        "fisher_condition_clipped": 4.0,
        "fisher_eigen_floor": 1e-7,
    }
    return centers - 0.25, centers + 0.25, diagnostics


def test_coverage_fit_exception_emits_all_four_attempt_rows(monkeypatch):
    monkeypatch.setattr(
        experiment,
        "sample_wfcm_1d",
        lambda *_args, **_kwargs: np.zeros((20, 1)),
    )

    def fail_fit(*_args, **_kwargs):
        raise RuntimeError("deliberate observed-fit exception")

    monkeypatch.setattr(experiment, "fit_wfcm", fail_fit)
    rows = experiment._coverage_one(_profile(), 20, 0, 9)

    assert len(rows) == 4
    assert {(row["method"], row["center"]) for row in rows} == {
        ("fisher", 0), ("fisher", 1), ("bootstrap", 0), ("bootstrap", 1)
    }
    assert all(not row["method_success"] for row in rows)
    assert all(not row["fit_success"] for row in rows)
    assert all(row["exception_type"] == "RuntimeError" for row in rows)
    assert all("observed-fit exception" in row["failure_message"] for row in rows)
    assert all(row["bootstrap_requested"] == 5 for row in rows)
    assert all(row["bootstrap_valid"] == 0 for row in rows)
    assert all(row["base_fit_runtime_seconds"] >= 0.0 for row in rows)


def test_coverage_methods_fail_independently_and_retain_bootstrap_counts(monkeypatch):
    monkeypatch.setattr(
        experiment,
        "sample_wfcm_1d",
        lambda *_args, **_kwargs: np.linspace(-2.0, 2.0, 20)[:, None],
    )
    monkeypatch.setattr(experiment, "fit_wfcm", lambda *_args, **_kwargs: _fit())

    def fail_fisher(*_args, **_kwargs):
        raise np.linalg.LinAlgError("deliberate Fisher failure")

    monkeypatch.setattr(experiment, "fisher_center_intervals", fail_fisher)
    bootstrap_diagnostics = {
        "bootstrap_requested": 5.0,
        "bootstrap_finite": 5.0,
        "bootstrap_valid": 5.0,
        "bootstrap_optimizer_success": 5.0,
        "bootstrap_runtime_seconds": 0.01,
    }
    monkeypatch.setattr(
        experiment,
        "bootstrap_center_intervals",
        lambda *_args, **_kwargs: (
            np.array([-2.5, 1.5]),
            np.array([-1.5, 2.5]),
            bootstrap_diagnostics,
        ),
    )

    rows = experiment._coverage_one(_profile(), 20, 0, 10)
    fisher = [row for row in rows if row["method"] == "fisher"]
    bootstrap = [row for row in rows if row["method"] == "bootstrap"]
    assert all(not row["method_success"] for row in fisher)
    assert all(row["exception_type"] == "LinAlgError" for row in fisher)
    assert all(row["method_success"] for row in bootstrap)
    assert all(row["failure_message"] == "" for row in bootstrap)
    assert all(row["bootstrap_valid"] == 5 for row in rows)


def test_coverage_bootstrap_failure_keeps_partial_refit_denominator(monkeypatch):
    monkeypatch.setattr(
        experiment,
        "sample_wfcm_1d",
        lambda *_args, **_kwargs: np.linspace(-2.0, 2.0, 20)[:, None],
    )
    monkeypatch.setattr(experiment, "fit_wfcm", lambda *_args, **_kwargs: _fit())
    monkeypatch.setattr(experiment, "fisher_center_intervals", _fisher)
    diagnostics = {
        "bootstrap_requested": 5.0,
        "bootstrap_finite": 4.0,
        "bootstrap_valid": 3.0,
        "bootstrap_optimizer_success": 3.0,
        "bootstrap_runtime_seconds": 0.02,
    }

    def fail_bootstrap(*_args, **_kwargs):
        raise BootstrapIntervalFailure("insufficient refits", diagnostics)

    monkeypatch.setattr(experiment, "bootstrap_center_intervals", fail_bootstrap)
    rows = experiment._coverage_one(_profile(), 20, 0, 11)
    fisher = [row for row in rows if row["method"] == "fisher"]
    bootstrap = [row for row in rows if row["method"] == "bootstrap"]
    assert all(row["method_success"] for row in fisher)
    assert all(not row["method_success"] for row in bootstrap)
    assert all(row["bootstrap_finite"] == 4 for row in rows)
    assert all(row["bootstrap_valid"] == 3 for row in rows)
    assert all("insufficient refits" in row["failure_message"] for row in bootstrap)


def test_lrt_observed_exception_emits_one_failed_outer_row(monkeypatch):
    def fail_observed(*_args, **_kwargs):
        raise FloatingPointError("deliberate observed LRT failure")

    monkeypatch.setattr(experiment, "fit_alternative_for_lrt", fail_observed)
    row = experiment._lrt_one(_profile(), 20, 0.0, 0, 12)

    assert not row["analysis_success"]
    assert row["exception_type"] == "FloatingPointError"
    assert "observed LRT failure" in row["failure_message"]
    assert row["bootstrap_requested"] == 5
    assert row["bootstrap_finite"] == 0
    assert row["bootstrap_valid"] == 0
    assert row["observed_runtime_seconds"] >= 0.0
    assert row["bootstrap_runtime_seconds"] == 0.0


def test_lrt_bootstrap_exceptions_do_not_abort_and_keep_counts(monkeypatch):
    calls = 0

    def observed_then_fail(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("deliberate bootstrap LRT failure")
        alternative = _fit()
        null = _fit(k=1)
        null.exact_loglik = -25.0
        return alternative, null, 0.2, 0.2

    monkeypatch.setattr(
        experiment, "fit_alternative_for_lrt", observed_then_fail
    )
    row = experiment._lrt_one(_profile(), 20, 0.0, 0, 13)

    assert calls == 1 + _profile().lrt_bootstrap_reps
    assert not row["analysis_success"]
    assert row["bootstrap_finite"] == 0
    assert row["bootstrap_valid"] == 0
    assert row["exception_type"] == "RuntimeError"
    assert "0/5" in row["failure_message"]
    assert row["bootstrap_runtime_seconds"] >= 0.0


def test_selection_incomplete_grid_never_uses_best_survivor(monkeypatch):
    monkeypatch.setattr(
        experiment,
        "sample_wfcm_1d",
        lambda *_args, **_kwargs: np.linspace(-2.0, 2.0, 20)[:, None],
    )

    def partly_failing_fit(*_args, **kwargs):
        if kwargs["k"] == 3:
            raise RuntimeError("deliberate k=3 failure")
        return _fit(kwargs["k"])

    monkeypatch.setattr(experiment, "fit_wfcm", partly_failing_fit)
    monkeypatch.setattr(
        experiment,
        "weighted_xie_beni",
        lambda _x, params: float(params.k),
    )
    monkeypatch.setattr(experiment, "fisher_center_intervals", _fisher)

    rows = experiment._selection_one(_profile(), 20, 0, 14)
    assert len(rows) == 4
    assert all(not row["selection_defined"] for row in rows)
    assert all(row["candidate_k2_valid"] for row in rows)
    assert all(not row["candidate_k3_valid"] for row in rows)
    assert all(row["candidate_k4_valid"] for row in rows)
    assert all(np.isnan(row["xbi_k3"]) for row in rows)
    fixed = [row for row in rows if row["analysis"] == "fixed_k"]
    plug_in = [row for row in rows if row["analysis"] == "selected_k_plug_in"]
    assert all(row["selected_k"] == 2 for row in fixed)
    assert all(row["inference_valid"] for row in fixed)
    assert all(np.isnan(row["selected_k"]) for row in plug_in)
    assert all(not row["true_k_selected"] for row in plug_in)
    assert all(not row["inference_valid"] for row in plug_in)
    assert all(row["conditional_cover"] is None for row in plug_in)
    assert all(not row["select_and_cover"] for row in plug_in)
    assert all("k=3" in row["failure_message"] for row in rows)


def test_selection_true_k_reuses_one_fisher_calculation_and_exact_rows(
    monkeypatch,
):
    monkeypatch.setattr(
        experiment,
        "sample_wfcm_1d",
        lambda *_args, **_kwargs: np.linspace(-2.0, 2.0, 20)[:, None],
    )
    monkeypatch.setattr(
        experiment,
        "fit_wfcm",
        lambda *_args, **kwargs: _fit(kwargs["k"]),
    )
    monkeypatch.setattr(
        experiment,
        "weighted_xie_beni",
        lambda _x, params: float(params.k),
    )
    fisher_calls = 0

    def counting_fisher(*args, **kwargs):
        nonlocal fisher_calls
        fisher_calls += 1
        return _fisher(*args, **kwargs)

    monkeypatch.setattr(experiment, "fisher_center_intervals", counting_fisher)
    rows = experiment._selection_one(_profile(), 20, 0, 16)

    assert fisher_calls == 1
    assert all(row["selection_defined"] for row in rows)
    fixed = sorted(
        (row for row in rows if row["analysis"] == "fixed_k"),
        key=lambda row: row["center"],
    )
    plug_in = sorted(
        (row for row in rows if row["analysis"] == "selected_k_plug_in"),
        key=lambda row: row["center"],
    )
    assert len(fixed) == len(plug_in) == 2
    for fixed_row, plug_in_row in zip(fixed, plug_in, strict=True):
        assert fixed_row["selected_k"] == plug_in_row["selected_k"] == 2
        assert fixed_row["true_k_selected"]
        assert plug_in_row["true_k_selected"]
        assert {
            key: value for key, value in fixed_row.items() if key != "analysis"
        } == {
            key: value for key, value in plug_in_row.items() if key != "analysis"
        }


def test_selection_sampling_exception_still_emits_four_rows(monkeypatch):
    def fail_sample(*_args, **_kwargs):
        raise FloatingPointError("deliberate selection sampler failure")

    monkeypatch.setattr(experiment, "sample_wfcm_1d", fail_sample)
    rows = experiment._selection_one(_profile(), 20, 0, 15)

    assert len(rows) == 4
    assert all(not row["selection_defined"] for row in rows)
    assert all(not row["candidate_k2_valid"] for row in rows)
    assert all(not row["candidate_k3_valid"] for row in rows)
    assert all(not row["candidate_k4_valid"] for row in rows)
    assert all(row["exception_type"] == "FloatingPointError" for row in rows)
    assert all("sampler failure" in row["failure_message"] for row in rows)
