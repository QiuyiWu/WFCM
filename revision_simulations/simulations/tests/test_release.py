"""Release checks: archived numbers, runnable entry points, and safe outputs."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def invoke(script, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / script), *map(str, args)],
        cwd=ROOT, capture_output=True, text=True,
    )


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    output = tmp_path_factory.mktemp("summary")
    result = invoke("reporting/summarize.py", "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    return output


def test_interval_table_uses_dataset_level_averages(tables):
    frame = pd.read_csv(tables / "coverage.csv")
    np.testing.assert_allclose(frame.coverage, [0.94, 0.9225, 0.96, 0.9575])
    np.testing.assert_allclose(frame.length, [0.5428451553, 0.5129397202,
                                            0.3402173091, 0.3280478825])
    np.testing.assert_allclose(frame.seconds, [8.5427175210, 0.0743652073,
                                             15.3936843675, 0.1104747858])
    assert frame.datasets.tolist() == [200] * 4


def test_lrt_keeps_primary_and_confirmation_separate(tables):
    frame = pd.read_csv(tables / "lrt.csv")
    assert frame.rejections.tolist() == [7, 59, 149, 150, 14, 115, 150, 150, 23]
    assert frame.datasets.tolist() == [150] * 8 + [600]
    assert frame.run.tolist() == ["primary"] * 8 + ["confirmation"]


def test_paper_benchmark_has_two_geometries_and_all_methods(tables):
    frame = pd.read_csv(tables / "benchmark.csv")
    assert set(frame.scenario) == {"spherical", "anisotropic"}
    assert frame.method.nunique() == 9
    assert len(frame) == 162
    assert (frame.attempted == 100).all()


def test_selection_reproduces_counts(tables):
    frame = pd.read_csv(tables / "selection.csv")
    assert frame.true_k_selected.tolist() == [149, 201]
    np.testing.assert_allclose(frame.fixed_coverage, [0.938, 0.956])
    np.testing.assert_allclose(frame.select_and_cover, [0.566, 0.772])


def test_report_has_figures_and_additional_results(tables):
    for name in ["benchmark.png", "computation.png", "additional/estimation.csv",
                 "additional/normality.csv", "additional/m_grid.csv",
                 "additional/m_selection.csv", "additional/diagnostics.png"]:
        assert (tables / name).stat().st_size > 100


def test_computation_matches_paired_scoring_reference(tables):
    frame = pd.read_csv(tables / "mc_size_sensitivity.csv")
    np.testing.assert_allclose(frame.nll_difference,
        [0.8791188787, 0.1121188315, 0.0132304630, 0,
         0.0556424918, 0.0068313024, 0.0011202417, 0], atol=1e-9)
    assert frame.datasets.tolist() == [20] * 8


def test_confirmation_accepts_an_empty_output_directory(tmp_path):
    result = invoke("run.py", "confirmation", "--output", tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    frame = pd.read_csv(tmp_path / "lrt_null_confirmation_raw.csv")
    assert len(frame) == 1 and frame.n.iloc[0] == 500


def test_runner_refuses_to_write_into_archived_results():
    result = invoke("run.py", "coverage", "--output", ROOT / "results")
    assert result.returncode != 0
    assert "archived results" in result.stderr


def test_runner_refuses_nonempty_output(tmp_path):
    (tmp_path / "keep.txt").write_text("keep")
    result = invoke("run.py", "coverage", "--output", tmp_path)
    assert result.returncode != 0
    assert "not empty" in result.stderr
    assert (tmp_path / "keep.txt").read_text() == "keep"


def test_archived_engine_identity():
    from simulations.lrt_null_confirmation import validate_locked_engine
    assert validate_locked_engine(require_pinned_hash=True)
