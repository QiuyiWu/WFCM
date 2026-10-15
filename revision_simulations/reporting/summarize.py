"""Rebuild compact tables and figures from the archived simulation records."""

import argparse
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs" / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd
from scipy.stats import norm

METHODS = ["wfcm", "fcm", "polynomial_fcm", "kmeans", "ekmeans", "hdbscan",
           "agglomerative", "spectral", "gmm_full"]
LABELS = ["WFCM", "FCM", "Polynomial FCM", "k-means", "Equilibrium k-means",
          "HDBSCAN", "Ward", "Spectral", "Gaussian mixture"]


def load(group, stem, keys):
    frame = pd.read_csv(ROOT / "results" / group / f"{stem}_raw.csv.gz")
    if frame.duplicated(keys).any():
        raise ValueError(f"duplicate records in {group}/{stem}")
    return frame


def stats(values):
    x = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return (float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))), len(x)) if len(x) > 1 else (
        float(x[0]) if len(x) else np.nan, np.nan, len(x))


def add_stats(row, name, values):
    row[name], row[name + "_mcse"], row[name + "_defined"] = stats(values)


def write(frame, output, name):
    frame.to_csv(output / f"{name}.csv", index=False)


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def benchmark(output):
    frame = load("benchmark_full", "benchmark",
                 ["scenario", "imbalance", "overlap", "replicate", "method"])
    paper = frame[frame.scenario.isin(["spherical", "anisotropic"])]
    rows = []
    keys = ["scenario", "imbalance", "overlap", "method"]
    for values, group in paper.groupby(keys, sort=True):
        row = dict(zip(keys, values))
        valid = group[group.success]
        row.update(attempted=len(group), successful=len(valid))
        for label, column in [("ari", "ari"), ("accuracy", "optimal_label_accuracy"),
                              ("minority_recall", "minority_recall"),
                              ("prototype_rmse", "center_rmse_exact_k")]:
            add_stats(row, label, valid[column])
        add_stats(row, "seconds", group.runtime_seconds)
        rows.append(row)
    summary = pd.DataFrame(rows)
    write(summary, output, "benchmark")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
    cells = [(i, o) for i in ["balanced", "moderate", "severe"]
             for o in ["low", "moderate", "high"]]
    for ax, geometry in zip(axes, ["spherical", "anisotropic"]):
        part = summary[summary.scenario == geometry].set_index(["method", "imbalance", "overlap"])
        matrix = np.array([[part.loc[(m, i, o), "ari"] for i, o in cells] for m in METHODS])
        im = ax.imshow(matrix, vmin=-0.1, vmax=1, cmap="viridis", aspect="auto")
        ax.set(title=geometry.title(), yticks=range(9), yticklabels=LABELS,
               xticks=range(9), xticklabels=[f"{dict(balanced='1:1', moderate='4:1', severe='10:1')[i]}\n{o}" for i, o in cells])
        ax.tick_params(axis="x", labelsize=7)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if value > 0.55 else "white")
    fig.colorbar(im, ax=axes, label="Mean ARI", shrink=0.85)
    save(fig, output, "benchmark")


def coverage(output):
    frame = load("inference_coverage_full", "coverage", ["n", "replicate", "method", "center"])
    rows = []
    for (n, method), group in frame.groupby(["n", "method"]):
        valid = group.groupby("replicate").filter(lambda x: len(x) == 2 and x.method_success.all())
        paired = valid.groupby("replicate")[["covered", "length"]].mean()
        times = group.drop_duplicates("replicate")
        row = dict(n=n, method=method, datasets=group.replicate.nunique(), valid_datasets=len(paired))
        add_stats(row, "coverage", paired.covered)
        add_stats(row, "length", paired.length)
        add_stats(row, "seconds", times.base_fit_runtime_seconds + times[method + "_runtime_seconds"])
        rows.append(row)
    write(pd.DataFrame(rows), output, "coverage")


def lrt(output):
    rows = []
    for label, directory, stem, success in [
        ("primary", "inference_lrt_full", "lrt", "analysis_success"),
        ("confirmation", "inference_lrt_null_confirmation", "lrt_null_confirmation", "confirmation_analysis_valid_95")]:
        frame = load(directory, stem, ["n", "separation", "replicate"])
        for (n, delta), group in frame.groupby(["n", "separation"]):
            valid = group[group[success]]
            count = int(valid.bootstrap_reject_05.sum())
            rate = count / len(valid) if len(valid) else np.nan
            rows.append(dict(run=label, n=n, delta=delta, datasets=len(group), valid_datasets=len(valid),
                             rejections=count, rejection=rate,
                             rejection_mcse=np.sqrt(rate * (1 - rate) / len(valid)) if len(valid) else np.nan))
    write(pd.DataFrame(rows), output, "lrt")


def selection(output):
    frame = load("inference_selection_full", "selection", ["n", "replicate", "analysis", "center"])
    rows = []
    for n, group in frame.groupby("n"):
        fixed = group[group.analysis == "fixed_k"]
        selected = group[group.analysis == "selected_k_plug_in"]
        datasets = selected.drop_duplicates("replicate")
        row = dict(n=n, datasets=len(datasets), defined_selections=int(datasets.selection_defined.sum()),
                   true_k_selected=int(datasets.true_k_selected.sum()))
        for k in [2, 3, 4]:
            row[f"k{k}_count"] = int((datasets.selected_k == k).sum())
        add_stats(row, "fixed_coverage", fixed.groupby("replicate").conditional_cover.mean())
        add_stats(row, "conditional_coverage", selected.groupby("replicate").conditional_cover.mean())
        add_stats(row, "select_and_cover", selected.groupby("replicate").select_and_cover.mean())
        rows.append(row)
    write(pd.DataFrame(rows), output, "selection")


def computation(output):
    summaries = {}
    for stem, keys in [("dimension_scaling", ["d", "proposal_kind"]),
                       ("sample_size_scaling", ["n"]),
                       ("mc_size_sensitivity", ["n", "m_over_n"])]:
        frame = load("computation_full", stem, keys + ["replicate"])
        if stem == "mc_size_sensitivity":
            ref = frame[frame.m_over_n == 80][["n", "replicate", "average_nll_independent_scoring_context", "fit_success"]]
            frame = frame.merge(ref, on=["n", "replicate"], suffixes=("", "_reference"), validate="many_to_one")
            frame["nll_difference"] = (frame.average_nll_independent_scoring_context - frame.average_nll_independent_scoring_context_reference).where(frame.fit_success & frame.fit_success_reference)
        rows = []
        for values, group in frame.groupby(keys):
            values = values if isinstance(values, tuple) else (values,)
            row = dict(zip(keys, values))
            row.update(datasets=len(group), successful=int(group.fit_success.sum()),
                       seconds_median=group.total_seconds.median(),
                       seconds_q25=group.total_seconds.quantile(.25), seconds_q75=group.total_seconds.quantile(.75))
            for name, column in [("ess_fraction", "ess_fraction"), ("max_weight", "max_normalized_weight"),
                                 ("center_coordinate_rmse", "center_rmse")]:
                add_stats(row, name, group[column])
            if stem == "mc_size_sensitivity":
                add_stats(row, "nll_difference", group.nll_difference)
                add_stats(row, "center_displacement", group.center_rmse_vs_largest_valid_m)
            rows.append(row)
        summaries[stem] = pd.DataFrame(rows)
        write(summaries[stem], output, stem)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), layout="constrained")
    for proposal, part in summaries["dimension_scaling"].groupby("proposal_kind"):
        part = part.sort_values("d")
        for ax, metric, label in zip(axes, ["ess_fraction", "center_coordinate_rmse", "seconds_median"],
                                     ["ESS / M", "Center-coordinate RMSE", "Median seconds"]):
            ax.plot(part.d, part[metric], marker="o", label=proposal)
            ax.set(xlabel="Dimension", ylabel=label)
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    save(fig, output, "computation")


def additional(output):
    output = output / "additional"
    output.mkdir(exist_ok=True)
    estimation = load("legacy_full", "estimation", ["n", "replicate", "coordinate"])
    valid = estimation[estimation.fit_success]
    reps = valid.drop_duplicates(["n", "replicate"])
    rows = []
    for n, group in reps.groupby("n"):
        row = dict(n=n, datasets=estimation[estimation.n == n].replicate.nunique(), valid_datasets=len(group),
                   sigma_rmse=np.sqrt(np.mean(group.sigma_abs_error ** 2)))
        for name in ["center_rmse", "weight_l1_error", "free_parameter_l2_error"]:
            add_stats(row, name, group[name])
        rows.append(row)
    summary = pd.DataFrame(rows)
    write(summary, output, "estimation")
    normality = valid.groupby(["n", "coordinate"]).marginal_z.agg(
        mean="mean", sd="std", datasets="size", q025=lambda x: x.quantile(.025),
        q975=lambda x: x.quantile(.975)).reset_index()
    write(normality, output, "normality")
    grid = load("legacy_full", "m_selection", ["n", "replicate", "candidate_m"])
    rows = []
    for (n, m), group in grid.groupby(["n", "candidate_m"]):
        row = dict(n=n, m=m, attempted=len(group))
        add_stats(row, "excess_nll", group.loc[group.eligible_for_selection, "nll_above_best"])
        rows.append(row)
    curves = pd.DataFrame(rows)
    write(curves, output, "m_grid")
    selected = grid.drop_duplicates(["n", "replicate"])
    counts = selected[selected.selection_defined].groupby(["n", "selected_m"]).size().rename("count").reset_index()
    counts["frequency"] = counts["count"] / counts.groupby("n")["count"].transform("sum")
    write(counts, output, "m_selection")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
    axes[0].loglog(summary.n, summary.center_rmse, "o-", label="Mean center RMSE")
    axes[0].loglog(summary.n, summary.center_rmse.iloc[0] * np.sqrt(summary.n.iloc[0] / summary.n), "--", label=r"$n^{-1/2}$")
    axes[0].set(xlabel="Sample size", ylabel="Estimation error")
    axes[0].set_xticks(summary.n, summary.n.astype(str))
    axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].legend(fontsize=8)
    for coordinate, part in valid[valid.n == valid.n.max()].groupby("coordinate"):
        z = np.sort(part.marginal_z.to_numpy())
        axes[1].plot(norm.ppf((np.arange(len(z)) + .5) / len(z)), z, ".", ms=3, label=coordinate)
    axes[1].plot([-3, 3], [-3, 3], "k--", lw=.7)
    axes[1].set(xlabel="Normal quantile", ylabel="Standardized estimate", title=f"n = {valid.n.max()}")
    axes[1].legend(fontsize=7)
    for n, part in curves.groupby("n"):
        axes[2].errorbar(part.m, part.excess_nll, yerr=part.excess_nll_mcse, marker="o", label=f"n = {n}")
    axes[2].set(xlabel="m", ylabel="Mean excess average NLL")
    axes[2].legend(fontsize=8)
    save(fig, output, "diagnostics")


def verify_archive():
    manifest = ROOT / "results" / "manifest.sha256"
    for line in manifest.read_text().splitlines():
        digest, name = line.split("  ", 1)
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"checksum mismatch: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "summary")
    args = parser.parse_args()
    output = args.output.resolve()
    archive = (ROOT / "results").resolve()
    if output == archive or archive in output.parents or output in archive.parents:
        parser.error("choose an output outside the archived results")
    verify_archive()
    output.mkdir(parents=True, exist_ok=True)
    for build in [benchmark, coverage, lrt, selection, computation, additional]:
        build(output)
    print(f"Tables and figures written to {output}")


if __name__ == "__main__":
    main()
