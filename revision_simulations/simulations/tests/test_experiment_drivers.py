import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import simulations.benchmark_experiment as benchmark_experiment
import simulations.legacy_corrected_experiment as legacy_experiment
from simulations.benchmark_experiment import _center_metrics, _label_metrics, summarize
from simulations.computation_experiment import _json_clean as computation_json_clean
from simulations.computation_experiment import _summary as computation_summary
from simulations.inference_experiment import (
    _json_clean as inference_json_clean,
    exact_loglik_1d,
    fisher_center_intervals,
    fit_alternative_for_lrt,
)
from simulations.legacy_corrected_experiment import _json_clean as legacy_json_clean
from simulations.task_checkpoint import (
    run_checkpointed_tasks,
    write_or_validate_config,
)
from simulations.wfcm_core import (
    WFCMParams,
    average_nll_and_gradient,
    fit_wfcm,
    sample_wfcm_1d,
)


def _mock_legacy_fit(candidate: float):
    spec = legacy_experiment.WFCMParameterization(k=2, d=1, m_fixed=candidate)
    params = WFCMParams(
        sigma=0.9,
        centers=np.array([[-1.0], [1.0]]),
        weights=np.array([0.35, 0.65]),
        m=candidate,
    )
    return SimpleNamespace(
        success=True,
        optimizer_success=True,
        average_nll=(candidate - 2.0) ** 2,
        gradient_inf_norm=1e-6,
        runtime_seconds=0.01,
        diagnostics={
            "ess": 1600.0,
            "ess_fraction": 0.8,
            "max_normalized_weight": 0.002,
        },
        message="converged",
        params=params,
        parameterization=spec,
        free_params=spec.pack(params),
    )


def test_benchmark_label_mapping_penalizes_noise_and_handles_permutation():
    truth = np.array([0, 0, 1, 1])
    predicted = np.array([7, 7, 3, -1])
    accuracy, macro_recall, minority_recall, mapping = _label_metrics(
        truth, predicted, k=2
    )
    assert mapping == {7: 0, 3: 1}
    assert accuracy == 0.75
    assert macro_recall == 0.75
    assert minority_recall == 0.75


def test_benchmark_center_metric_requires_exact_center_count_for_primary_rmse():
    truth = np.array([[-2.0], [2.0]])
    exact, matched, count, exact_count = _center_metrics(
        np.array([[2.0], [-2.0]]), truth
    )
    assert exact == 0.0 and matched == 0.0 and count == 2 and exact_count

    exact, matched, count, exact_count = _center_metrics(np.array([[-2.0]]), truth)
    assert np.isnan(exact)
    assert matched == 0.0 and count == 1 and not exact_count


def test_benchmark_center_rmse_uses_euclidean_prototype_error_not_coordinate_rmse():
    truth = np.array([[0.0, 0.0], [4.0, 4.0]])
    estimated = np.array([[1.0, 1.0], [5.0, 5.0]])
    exact, matched, count, exact_count = _center_metrics(estimated, truth)
    # Each prototype is displaced by Euclidean distance sqrt(2), so the
    # prototype-level RMSE is sqrt(2), not the coordinatewise RMSE 1.
    np.testing.assert_allclose(exact, np.sqrt(2.0))
    np.testing.assert_allclose(matched, np.sqrt(2.0))
    assert count == 2 and exact_count


def test_benchmark_convenience_summary_excludes_synthetic_failure_endpoints():
    common = {
        "scenario": "spherical",
        "imbalance": "balanced",
        "overlap": "low",
        "method": "wfcm",
        "macro_recall": 0.8,
        "minority_recall": 0.8,
        "matched_center_rmse": 0.2,
        "center_rmse_exact_k": 0.2,
        "exact_center_count": True,
        "n_clusters_found": 2,
        "noise_fraction": 0.0,
    }
    frame = pd.DataFrame(
        [
            {
                **common,
                "replicate": 0,
                "success": True,
                "ari": 0.8,
                "optimal_label_accuracy": 0.9,
                "runtime_seconds": 1.0,
            },
            {
                **common,
                "replicate": 1,
                "success": False,
                # Exception rows use an all-noise placeholder.  Its finite
                # zero scores must not contaminate the primary endpoint means.
                "ari": 0.0,
                "optimal_label_accuracy": 0.0,
                "macro_recall": 0.0,
                "minority_recall": 0.0,
                "matched_center_rmse": np.nan,
                "center_rmse_exact_k": np.nan,
                "exact_center_count": False,
                "n_clusters_found": 0,
                "noise_fraction": 1.0,
                "runtime_seconds": 2.0,
            },
        ]
    )
    row = summarize(frame).iloc[0]
    assert row["attempted_replicates"] == 2
    assert row["successful_replicates"] == 1
    assert row["failed_replicates"] == 1
    assert row["success_rate"] == 0.5
    assert row["mean_ari_successful"] == 0.8
    assert row["ari_successful_defined_replicates"] == 1
    assert row["mean_accuracy_successful"] == 0.9
    assert row["mean_runtime_seconds_all_attempts"] == 1.5
    assert "mean_ari" not in row.index


def test_benchmark_rows_record_deterministic_method_specific_seeds(monkeypatch):
    scenario = SimpleNamespace(
        scenario="spherical",
        x=np.array([[-1.0, 0.0], [1.0, 0.0]]),
        labels=np.array([0, 1]),
        true_centers=np.array([[-1.0, 0.0], [1.0, 0.0]]),
        n=2,
        d=2,
        k=2,
        counts=np.array([1, 1]),
        imbalance_ratio=1.0,
        metadata={"realized_majority_to_minority_ratio": 1.0, "separation": 4.0},
    )
    monkeypatch.setattr(benchmark_experiment, "generate_scenario", lambda *args, **kwargs: scenario)
    params = WFCMParams(
        sigma=1.0,
        centers=scenario.true_centers,
        weights=np.array([0.5, 0.5]),
        m=2.0,
    )
    wfcm = SimpleNamespace(
        params=params,
        success=True,
        message="converged",
        average_nll=1.0,
        gradient_inf_norm=1e-6,
        optimizer_success=True,
        diagnostics={
            "ess": 100.0,
            "ess_fraction": 0.5,
            "max_normalized_weight": 0.02,
        },
        proposal_name="mock",
    )
    monkeypatch.setattr(benchmark_experiment, "fit_wfcm", lambda *args, **kwargs: wfcm)
    monkeypatch.setattr(
        benchmark_experiment,
        "membership",
        lambda *args, **kwargs: np.eye(2),
    )

    def mock_competitors(x, k, *, methods, seed, method_options):
        children = np.random.SeedSequence(seed).spawn(len(methods))
        return {
            method: SimpleNamespace(
                labels=scenario.labels,
                centers=scenario.true_centers,
                runtime_seconds=0.01,
                success=True,
                message="fit completed",
                noise_fraction=0.0,
                diagnostics={},
                seed=int(child.generate_state(1, dtype=np.uint32)[0]),
            )
            for method, child in zip(methods, children, strict=True)
        }

    monkeypatch.setattr(benchmark_experiment, "fit_competitors", mock_competitors)
    profile = benchmark_experiment.Profile(
        1,
        2,
        1,
        1,
        20,
        10,
        ("spherical",),
        ("balanced",),
        ("low",),
    )
    rows = benchmark_experiment._one_dataset(
        profile, "spherical", "balanced", "low", 0, 123
    )
    assert len(rows) == 1 + len(benchmark_experiment.DEFAULT_METHODS)
    assert all(isinstance(row["method_seed"], int) for row in rows)
    assert rows[0]["method_seed"] == benchmark_experiment._seed(123, 302, 0, 0, 0, 0)
    assert len({row["method_seed"] for row in rows}) == len(rows)


def test_driver_json_cleaners_emit_strict_json():
    payload = {
        "nan": np.nan,
        "positive_infinity": np.inf,
        "integer": np.int64(3),
        "boolean": np.bool_(True),
        "array": np.array([1.0, np.nan]),
    }
    for cleaner in (
        inference_json_clean,
        computation_json_clean,
        legacy_json_clean,
    ):
        encoded = json.dumps(cleaner(payload), allow_nan=False)
        decoded = json.loads(encoded)
        assert decoded == {
            "nan": None,
            "positive_infinity": None,
            "integer": 3,
            "boolean": True,
            "array": [1.0, None],
        }


def test_computation_summary_excludes_failed_endpoints_from_success_metrics():
    frame = pd.DataFrame(
        [
            {
                "d": 2,
                "proposal_kind": "q",
                "total_seconds": 1.0,
                "proposal_seconds": 0.1,
                "importance_draw_seconds": 0.2,
                "optimizer_seconds": 0.6,
                "fit_wall_seconds": 0.7,
                "ess_fraction": 0.5,
                "max_normalized_weight": 0.1,
                "fit_success": True,
                "center_rmse": 1.0,
                "ari": 0.8,
                "replicate": 0,
            },
            {
                "d": 2,
                "proposal_kind": "q",
                "total_seconds": 2.0,
                "proposal_seconds": 0.1,
                "importance_draw_seconds": 0.2,
                "optimizer_seconds": 1.6,
                "fit_wall_seconds": 1.7,
                "ess_fraction": 0.01,
                "max_normalized_weight": 0.9,
                "fit_success": False,
                "center_rmse": 99.0,
                "ari": -0.2,
                "replicate": 1,
            },
        ]
    )
    result = computation_summary(frame, ["d", "proposal_kind"]).iloc[0]
    assert result["success_rate"] == 0.5
    assert result["mean_center_rmse_all_endpoints"] == 50.0
    assert result["mean_center_rmse_successful"] == 1.0
    assert result["mean_ari_successful"] == 0.8


def test_fisher_center_interval_has_inverse_square_root_n_scaling():
    truth = WFCMParams(
        sigma=0.9,
        centers=np.array([[-2.0], [2.0]]),
        weights=np.array([0.35, 0.65]),
        m=2.0,
    )
    x = sample_wfcm_1d(truth, 100, seed=117)
    fit = fit_wfcm(x, 2, m=2.0, importance_size=2_000, n_init=2, seed=118)
    assert fit.success
    lo_100, hi_100, _ = fisher_center_intervals(
        fit, 100, draws=10_001, seed=119
    )
    lo_400, hi_400, _ = fisher_center_intervals(
        fit, 400, draws=10_001, seed=119
    )
    np.testing.assert_allclose(hi_100 - lo_100, 2.0 * (hi_400 - lo_400))


def test_lrt_statistic_explicitly_retains_the_embedded_null_candidate():
    rng = np.random.default_rng(221)
    x = rng.normal(size=(80, 1))
    fit, null_fit, statistic, raw = fit_alternative_for_lrt(
        x,
        m=2.0,
        seed=222,
        m_is=2_000,
        n_init=2,
        maxiter=180,
    )
    assert fit.success and null_fit.success
    assert null_fit.estimator_name == "analytic_gaussian_mle"
    assert not null_fit.optimizer_success
    assert null_fit.gradient_inf_norm == 0.0
    assert np.isfinite(raw)
    assert statistic == max(0.0, raw)
    np.testing.assert_allclose(
        exact_loglik_1d(x, null_fit.params),
        null_fit.exact_loglik,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        exact_loglik_1d(x, null_fit.embedded_params),
        null_fit.exact_loglik,
        rtol=1e-12,
        atol=1e-12,
    )
    expected_raw = 2.0 * (
        exact_loglik_1d(x, fit.params) - null_fit.exact_loglik
    )
    np.testing.assert_allclose(raw, expected_raw, rtol=1e-12, atol=1e-12)
    embedded_value, _ = average_nll_and_gradient(
        fit.parameterization.pack(null_fit.embedded_params),
        x,
        fit.importance_context,
        fit.parameterization,
    )
    np.testing.assert_allclose(
        null_fit.average_nll, embedded_value, rtol=1e-12, atol=1e-12
    )


def test_legacy_estimation_exception_retains_attempted_coordinate_block(monkeypatch):
    profile = legacy_experiment.Profile(1, 1, (20,), (20,), 1, 10, 100)
    monkeypatch.setattr(
        legacy_experiment,
        "sample_wfcm_1d",
        lambda params, size, seed: np.linspace(-2.0, 2.0, size)[:, None],
    )

    def fail_fit(*args, **kwargs):
        raise RuntimeError("deliberate fit failure")

    monkeypatch.setattr(legacy_experiment, "fit_wfcm", fail_fit)
    rows = legacy_experiment._estimation_one(
        profile, n=20, replicate=0, base_seed=9, inverse_information=np.eye(4)
    )
    assert len(rows) == 4
    assert {row["coordinate"] for row in rows} == {
        "log_sigma",
        "center_1",
        "center_2",
        "weight_floor_adjusted_log_ratio",
    }
    assert all(not row["fit_success"] for row in rows)
    assert all(row["exception_type"] == "RuntimeError" for row in rows)
    assert all("deliberate fit failure" in row["failure_message"] for row in rows)
    assert all(np.isnan(row["estimate_free"]) for row in rows)
    assert all(np.isfinite(row["truth_free"]) for row in rows)
    assert all(row["proposal_kind"] == "defensive" for row in rows)
    assert all(row["proposal_name"] == "" for row in rows)
    assert len({row["proposal_seed"] for row in rows}) == 1
    assert len({row["importance_seed"] for row in rows}) == 1
    assert all(row["proposal_seed"] >= 0 for row in rows)
    assert all(row["importance_seed"] >= 0 for row in rows)
    assert all(row["importance_cloud_sha256"] == "" for row in rows)


def test_legacy_sampling_exception_also_retains_attempted_rows(monkeypatch):
    profile = legacy_experiment.Profile(1, 1, (20,), (20,), 1, 10, 100)

    def fail_sample(*args, **kwargs):
        raise FloatingPointError("deliberate sampler failure")

    monkeypatch.setattr(legacy_experiment, "sample_wfcm_1d", fail_sample)
    estimation = legacy_experiment._estimation_one(
        profile, n=20, replicate=0, base_seed=12, inverse_information=np.eye(4)
    )
    assert len(estimation) == 4
    assert all(not row["fit_success"] for row in estimation)
    assert all(row["exception_type"] == "FloatingPointError" for row in estimation)
    assert all("sampler failure" in row["failure_message"] for row in estimation)

    selection = legacy_experiment._m_selection_one(
        profile, n=20, replicate=0, base_seed=13
    )
    assert len(selection) == len(legacy_experiment.M_GRID)
    assert all(not row["candidate_fit_valid"] for row in selection)
    assert all(not row["selection_defined"] for row in selection)
    assert all(np.isnan(row["selected_m"]) for row in selection)
    assert all("common importance-cloud setup failed" in row["failure_message"] for row in selection)
    assert all(row["proposal_name"] == "" for row in selection)
    assert all(row["importance_cloud_sha256"] == "" for row in selection)
    assert all(np.isnan(row["max_normalized_weight"]) for row in selection)
    assert len({row["proposal_seed"] for row in selection}) == 1
    assert len({row["importance_seed"] for row in selection}) == 1


def test_legacy_estimation_archives_importance_diagnostics_and_fingerprint(monkeypatch):
    profile = legacy_experiment.Profile(1, 1, (20,), (20,), 1, 10, 100)
    monkeypatch.setattr(
        legacy_experiment,
        "sample_wfcm_1d",
        lambda params, size, seed: np.linspace(-2.0, 2.0, size)[:, None],
    )

    def successful_fit(x, *, m, seed, **kwargs):
        spec = legacy_experiment.WFCMParameterization(k=2, d=1, m_fixed=m)
        params = WFCMParams(
            sigma=0.9,
            centers=np.array([[-2.0], [2.0]]),
            weights=np.array([0.35, 0.65]),
            m=m,
        )
        rng = np.random.default_rng(seed)
        rng.integers(0, 2**31 - 1)
        importance_seed = int(rng.integers(0, 2**31 - 1))
        context = SimpleNamespace(
            samples=np.array([[-1.0], [1.0]]),
            log_q=np.array([-1.0, -1.0]),
            proposal_name="mock_proposal",
            seed=importance_seed,
        )
        return SimpleNamespace(
            success=True,
            optimizer_success=True,
            average_nll=1.0,
            gradient_inf_norm=1e-6,
            runtime_seconds=0.01,
            diagnostics={
                "ess": 1600.0,
                "ess_fraction": 0.8,
                "max_normalized_weight": 0.002,
            },
            message="converged",
            params=params,
            parameterization=spec,
            free_params=spec.pack(params),
            proposal_name="mock_proposal",
            importance_seed=importance_seed,
            importance_context=context,
        )

    monkeypatch.setattr(legacy_experiment, "fit_wfcm", successful_fit)
    rows = legacy_experiment._estimation_one(
        profile, n=20, replicate=0, base_seed=14,
        inverse_information=np.eye(4),
    )
    assert all(row["fit_success"] for row in rows)
    assert all(row["failure_message"] == "" for row in rows)
    assert all(row["ess"] == 1600.0 for row in rows)
    assert all(row["max_normalized_weight"] == 0.002 for row in rows)
    assert all(row["proposal_kind"] == "defensive" for row in rows)
    assert all(row["proposal_name"] == "mock_proposal" for row in rows)
    assert len({row["proposal_seed"] for row in rows}) == 1
    assert len({row["importance_seed"] for row in rows}) == 1
    assert len({row["importance_cloud_sha256"] for row in rows}) == 1
    assert len(rows[0]["importance_cloud_sha256"]) == 64


def test_legacy_m_selection_is_undefined_if_any_grid_fit_fails(monkeypatch):
    profile = legacy_experiment.Profile(1, 1, (20,), (20,), 1, 10, 100)
    monkeypatch.setattr(
        legacy_experiment,
        "sample_wfcm_1d",
        lambda params, size, seed: np.zeros((size, 1)),
    )
    monkeypatch.setattr(legacy_experiment, "build_proposal", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        legacy_experiment,
        "ISContext",
        SimpleNamespace(
            draw=lambda proposal, size, seed: SimpleNamespace(
                samples=np.array([[-1.0], [1.0]]),
                log_q=np.array([-1.0, -1.0]),
                proposal_name="mock_proposal",
                seed=seed,
            )
        ),
    )
    attempted = []

    def partly_failing_fit(*args, **kwargs):
        candidate = float(kwargs["m"])
        attempted.append(candidate)
        if np.isclose(candidate, 1.9):
            raise FloatingPointError("deliberate candidate failure")
        return _mock_legacy_fit(candidate)

    monkeypatch.setattr(legacy_experiment, "fit_wfcm", partly_failing_fit)
    rows = legacy_experiment._m_selection_one(
        profile, n=20, replicate=0, base_seed=10
    )
    assert attempted == list(legacy_experiment.M_GRID)
    assert len(rows) == len(legacy_experiment.M_GRID)
    assert all(not row["selection_defined"] for row in rows)
    assert all(not row["complete_grid_valid"] for row in rows)
    assert all(
        row["candidate_fit_valid"] == row["eligible_for_selection"] for row in rows
    )
    assert all(np.isnan(row["selected_m"]) for row in rows)
    assert all(np.isnan(row["nll_above_best"]) for row in rows)
    failed = [row for row in rows if np.isclose(row["candidate_m"], 1.9)]
    assert len(failed) == 1
    assert not failed[0]["eligible_for_selection"]
    assert failed[0]["exception_type"] == "FloatingPointError"
    successful = [row for row in rows if row["eligible_for_selection"]]
    assert all(row["minimum_center_separation"] == 2.0 for row in successful)
    assert all(row["weight_floor_margin"] > 0.0 for row in successful)
    assert all(
        not row["theoretical_center_separation_constraint_enforced"]
        for row in rows
    )
    assert len({row["importance_seed"] for row in rows}) == 1
    assert len({row["importance_cloud_sha256"] for row in rows}) == 1
    assert len(rows[0]["importance_cloud_sha256"]) == 64
    assert all(row["proposal_kind"] == "defensive" for row in rows)
    assert all(row["proposal_name"] == "mock_proposal" for row in rows)


def test_legacy_m_selection_uses_full_grid_when_every_fit_is_valid(monkeypatch):
    profile = legacy_experiment.Profile(1, 1, (20,), (20,), 1, 10, 100)
    monkeypatch.setattr(
        legacy_experiment,
        "sample_wfcm_1d",
        lambda params, size, seed: np.linspace(-2.0, 2.0, size)[:, None],
    )
    monkeypatch.setattr(legacy_experiment, "build_proposal", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        legacy_experiment,
        "ISContext",
        SimpleNamespace(
            draw=lambda proposal, size, seed: SimpleNamespace(
                samples=np.array([[-1.0], [1.0]]),
                log_q=np.array([-1.0, -1.0]),
                proposal_name="mock_proposal",
                seed=seed,
            )
        ),
    )

    def successful_fit(*args, **kwargs):
        candidate = float(kwargs["m"])
        return _mock_legacy_fit(candidate)

    monkeypatch.setattr(legacy_experiment, "fit_wfcm", successful_fit)
    rows = legacy_experiment._m_selection_one(
        profile, n=20, replicate=0, base_seed=11
    )
    assert all(row["selection_defined"] for row in rows)
    assert all(row["complete_grid_valid"] for row in rows)
    assert all(row["candidate_fit_valid"] for row in rows)
    assert all(row["selected_m"] == 2.0 for row in rows)
    assert all(row["selected_true_m"] for row in rows)
    assert all(row["minimum_center_separation"] == 2.0 for row in rows)
    assert all(row["minimum_relative_free_coordinate_bound_margin"] > 0.0 for row in rows)
    expected_gaps = {
        candidate: (candidate - 2.0) ** 2 for candidate in legacy_experiment.M_GRID
    }
    for row in rows:
        assert row["nll_above_best"] == expected_gaps[row["candidate_m"]]
        assert row["ess"] == 1600.0
        assert row["max_normalized_weight"] == 0.002
        assert row["proposal_name"] == "mock_proposal"
        assert len(row["importance_cloud_sha256"]) == 64


def test_legacy_returned_fit_failure_names_failed_numerical_gates():
    fit = _mock_legacy_fit(2.0)
    fit.success = False
    fit.optimizer_success = True
    fit.gradient_inf_norm = 0.02
    fit.message = "optimizer reported convergence"
    message = legacy_experiment._fit_failure_message(fit)
    assert "projected_gradient_inf_norm" in message
    assert "exceeds" in message
    assert "optimizer reported convergence" in message


def test_legacy_cloud_fingerprint_covers_samples_and_log_q():
    context = SimpleNamespace(
        samples=np.array([[-1.0], [1.0]]),
        log_q=np.array([-1.5, -1.5]),
    )
    baseline = legacy_experiment._importance_cloud_sha256(context)
    assert len(baseline) == 64
    changed_samples = SimpleNamespace(
        samples=np.array([[-1.0], [1.1]]),
        log_q=context.log_q,
    )
    changed_log_q = SimpleNamespace(
        samples=context.samples,
        log_q=np.array([-1.5, -1.4]),
    )
    assert legacy_experiment._importance_cloud_sha256(changed_samples) != baseline
    assert legacy_experiment._importance_cloud_sha256(changed_log_q) != baseline


def test_atomic_task_checkpoints_resume_without_rerunning_completed_tasks(tmp_path):
    tasks = [("cell-a", 0), ("cell-b", 1)]
    calls = []

    def worker(task):
        calls.append(task)
        return [{"cell": task[0], "replicate": task[1], "nonfinite": np.nan}]

    def validate(task, rows):
        assert len(rows) == 1
        assert (rows[0]["cell"], rows[0]["replicate"]) == task

    first = run_checkpointed_tasks(tasks, tmp_path / "checkpoints", worker, validate)
    assert calls == tasks
    # The durable strict-JSON representation converts nonfinite diagnostics to
    # null on both fresh and resumed paths.
    assert all(chunk[0]["nonfinite"] is None for chunk in first)

    calls.clear()
    second = run_checkpointed_tasks(tasks, tmp_path / "checkpoints", worker, validate)
    assert calls == []
    assert second == first
    assert len(list((tmp_path / "checkpoints").glob("task_*.json"))) == 2


def test_atomic_task_checkpoint_rejects_corrupted_payload(tmp_path):
    directory = tmp_path / "checkpoints"
    tasks = [("cell", 0)]
    run_checkpointed_tasks(tasks, directory, lambda task: [{"task": list(task)}], lambda *_: None)
    path = directory / "task_000000.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["rows"][0]["task"] = ["tampered", 0]
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RuntimeError, match="integrity check"):
        run_checkpointed_tasks(
            tasks,
            directory,
            lambda task: [{"task": list(task)}],
            lambda *_: None,
        )


def test_resume_config_refuses_mixed_scientific_settings(tmp_path):
    path = tmp_path / "run_config.json"
    write_or_validate_config(path, {"profile": "full", "seed": 17})
    write_or_validate_config(path, {"profile": "full", "seed": 17})
    with pytest.raises(RuntimeError, match="different run configuration"):
        write_or_validate_config(path, {"profile": "full", "seed": 18})
