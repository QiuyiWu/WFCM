"""Reconstruct separate consistency, QQ, m-selection figures and Table 6.

No fitting or resampling occurs here. All displayed quantities come from the
saved Algorithm 1 records. Pilot renders are explicitly watermarked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from simulations.algorithm1_experiments import (
    atomic_json, coverage_summary, planned_tasks, read_record, validate_record,
)


def _statistics(values):
    a = np.asarray(values, dtype=float)
    return (float(a.mean()) if len(a) else np.nan,
            float(a.std(ddof=1)) if len(a) > 1 else np.nan,
            float(a.std(ddof=1)/np.sqrt(len(a))) if len(a) > 1 else np.nan)


def reconstruct_normality(records, inverse, truth):
    """Use actual fitted weights, without the generic packer's 1e-12 clip.

    The frozen archive retains its original diagnostics. Published QQ points
    are derived from the saved physical estimates, with no refitting or
    mutation of those records. This matters for very small positive weights.
    """
    truth_free = np.r_[np.log(truth['sigma']), np.asarray(truth['centers']).ravel(),
                      np.log(truth['weights'][0])-np.log(truth['weights'][1])]
    result = []
    for record in records:
        if record['experiment'] != 'estimation' or 'fit' not in record:
            result.append(record)
            continue
        fit = record['fit']
        order = np.argsort(np.asarray(fit['centers'])[:, 0])
        weights = np.asarray(fit['w'])[order]
        free = np.r_[np.log(fit['sigma']), np.asarray(fit['centers'])[order, 0],
                     np.log(weights[0])-np.log(weights[1])]
        result.append(dict(record, estimate_free=free, truth_free=truth_free,
                           marginal_z=(free-truth_free)/np.sqrt(np.diag(inverse)/record['n'])))
    return result


def summarize_additional(records):
    estimation, normality, curves, selection = [], [], [], []
    estimates = [r for r in records if r['experiment'] == 'estimation']
    grids = [r for r in records if r['experiment'] == 'grid']
    for n in sorted({r['n'] for r in estimates}):
        group = [r for r in estimates if r['n'] == n]
        valid = [r for r in group if r['success']]
        row = dict(n=n, attempted=len(group), valid=len(valid))
        for metric in ('center_rmse', 'sigma_abs_error', 'weight_l1_error'):
            mean, sd, mcse = _statistics([r[metric] for r in valid])
            row.update({metric: mean, metric+'_sd': sd, metric+'_mcse': mcse})
        estimation.append(row)
        if valid:
            for j, coordinate in enumerate(valid[0]['coordinates']):
                z = np.sort([r['marginal_z'][j] for r in valid])
                quantiles = norm.ppf((np.arange(len(z))+.5)/len(z))
                normality.append(dict(n=n, coordinate=coordinate, attempted=len(group), valid=len(z),
                    mean=float(z.mean()), sd=float(z.std(ddof=1)), q025=float(np.quantile(z, .025)),
                    q975=float(np.quantile(z, .975)), qq_rmse=float(np.sqrt(np.mean((z-quantiles)**2)))))
    for n in sorted({r['n'] for r in grids}):
        group = [r for r in grids if r['n'] == n]
        valid = [r for r in group if r['success']]
        grid = sorted({c['candidate_m'] for r in group for c in r['candidates']})
        for m in grid:
            values = [next(c['average_nll'] for c in r['candidates'] if c['candidate_m'] == m)
                      for r in valid]
            average, sd, mcse = _statistics(values)
            curves.append(dict(n=n, m=m, attempted=len(group), valid=len(valid),
                mean_total_nll=n*average, sd_total_nll=n*sd, mcse_total_nll=n*mcse,
                mean_average_nll=average, sd_average_nll=sd, mcse_average_nll=mcse))
            count = sum(r['selected_m'] == m for r in valid)
            selection.append(dict(n=n, m=m, count=count, attempted=len(group), valid=len(valid),
                frequency_valid=count/len(valid) if valid else np.nan,
                frequency_attempted=count/len(group)))
    return {key: pd.DataFrame(value) for key, value in (
        ('estimation', estimation), ('normality', normality), ('m_grid', curves), ('m_selection', selection))}


def _save(fig, output, name, pilot):
    if pilot:
        fig.text(.5, 1.02, 'PILOT — NOT MANUSCRIPT RESULTS', ha='center', color='firebrick', fontsize=10)
    fig.savefig(output/f'{name}.pdf', bbox_inches='tight')
    fig.savefig(output/f'{name}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def figures(records, tables, output, pilot):
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42})
    colors = ['#24547a', '#b65d32', '#49846d', '#805c99']
    estimation = tables['estimation']
    if not estimation.empty:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), layout='constrained')
        for ax, metric, title, color in zip(axes,
                ('center_rmse', 'sigma_abs_error', 'weight_l1_error'),
                ('Center error', 'Scale error', 'Weight error'), colors):
            ax.errorbar(estimation.n, estimation[metric],
                yerr=1.96*estimation[metric+'_mcse'], marker='o', capsize=3,
                color=color, label='Mean ± 1.96 MCSE')
            first = estimation[metric].iloc[0]
            ax.plot(estimation.n, first*np.sqrt(estimation.n.iloc[0]/estimation.n),
                    '--', color='.45', label=r'$n^{-1/2}$ reference')
            ax.set(xscale='log', yscale='log', xlabel='Sample size n', ylabel='Estimation error', title=title)
            ax.set_xticks(estimation.n, estimation.n.astype(str))
            ax.minorticks_off()
            ax.legend(fontsize=8, loc='upper right')
        _save(fig, output, 'figure10_consistency', pilot)
        valid = [r for r in records if r['experiment'] == 'estimation' and r['success']]
        fig, axes = plt.subplots(2, 2, figsize=(8, 7), layout='constrained')
        for j, ax in enumerate(axes.ravel()):
            extreme = max(abs(r['marginal_z'][j]) for r in valid) > 5
            zoom = (ax.inset_axes([.08, .26, .43, .40] if j == 1 else [.45, .36, .48, .41])
                    if extreme else None)
            for color, n in zip(colors, sorted({r['n'] for r in valid})):
                z = np.sort([r['marginal_z'][j] for r in valid if r['n'] == n])
                quantiles = norm.ppf((np.arange(len(z))+.5)/len(z))
                ax.plot(quantiles, z, '.-', color=color,
                        ms=2.4, lw=1.15, alpha=.9, label=f'n = {n}')
                if zoom is not None:
                    # Identical points and ranks; only the inset's view is zoomed.
                    zoom.plot(quantiles, z, '.-', color=color, ms=2, lw=1.2, alpha=.9)
            ax.plot([-3.5, 3.5], [-3.5, 3.5], '--', color='.35', lw=1)
            ax.set(xlabel='Standard normal quantile', ylabel='Standardized estimation error',
                   title=(r'$\log\sigma$', r'$v_1$', r'$v_2$', r'$\log(w_1/w_2)$')[j])
            ax.legend(fontsize=8, loc='upper left' if j == 1 else 'lower right')
            if zoom is not None:
                zoom.plot([-4, 4], [-4, 4], '--', color='.35', lw=.7)
                zoom.set(xlim=(-3.2, 3.2), ylim=(-4, 4), xticks=[-2, 0, 2], yticks=[-3, 0, 3])
                zoom.tick_params(labelsize=6, length=2)
                zoom.set_title('Central range (zoom)', fontsize=7)
                zoom.patch.set_alpha(.95)
        _save(fig, output, 'figure11_qq', pilot)
    curves = tables['m_grid']
    if not curves.empty:
        groups = list(curves.groupby('n'))
        fig, axes = plt.subplots(2, len(groups), figsize=(3.4*len(groups), 5.4),
            layout='constrained', squeeze=False, sharex='col',
            gridspec_kw={'height_ratios': [1.05, 1.]})
        for column, ((n, group), color) in enumerate(zip(groups, colors)):
            ax, zoom = axes[:, column]
            ax.errorbar(group.m, group.mean_total_nll,
                yerr=1.96*group.mcse_total_nll if group.mcse_total_nll.notna().any() else None,
                fmt='o-', color=color, ms=4, lw=1.8, ecolor='.55',
                elinewidth=.8, capsize=2.5, capthick=.8, label='Mean ± 1.96 MCSE')
            zoom.plot(group.m, group.mean_total_nll, 'o-', color=color, ms=4, lw=1.8)
            for panel in (ax, zoom):
                panel.axvline(2., color='.5', ls='--', lw=.8, label=r'True $m_0=2$')
                panel.set_xticks(group.m)
                panel.tick_params(axis='x', labelsize=9, labelrotation=45)
                panel.tick_params(axis='y', labelsize=9)
                panel.ticklabel_format(axis='y', style='plain', useOffset=False)
            if group.mean_total_nll.notna().any():
                best = group.loc[group.mean_total_nll.idxmin()]
                for panel in (ax, zoom):
                    panel.plot(best.m, best.mean_total_nll, '*', ms=12, color=color,
                               label='Minimum mean NLL')
            ax.set(ylabel='Total training NLL', title=f'n = {n}')
            zoom.set(xlabel='Fuzziness parameter m', ylabel='Mean total NLL (zoom)',
                     title='Mean curve only - no error bars')
            zoom.set_title('Mean curve only - no error bars', fontsize=9)
            zoom.margins(y=.18)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, -.02),
                   ncol=3, frameon=False, fontsize=9)
        _save(fig, output, 'figure12_m_selection', pilot)


def table6_latex(summary, path):
    names = {'normal': 'Normal bootstrap', 'fisher': 'Model Fisher', 'percentile': 'Percentile bootstrap'}
    lines = [r'\begin{tabular}{rlccc}', r'\toprule',
             r'$n$ & Method & Coverage (\%) & Length & CPU time (s)\\', r'\midrule']
    for bootstrap, label in [('normal', 'Parametric normal bootstrap comparison'),
                              ('percentile', 'Supplementary nonparametric percentile comparison')]:
        lines.append(r'\multicolumn{5}{l}{\textit{'+label+r'}}\\')
        for n in sorted(summary.n.unique()):
            for method in (bootstrap, 'fisher'):
                row = summary[(summary.n == n) & (summary.method == method)].iloc[0]
                lines.append(f'{n} & {names[method]} & '
                    f'{100*row.coverage:.1f} ({100*row.coverage_mcse:.1f}) & '
                    f'{row.length:.3f} ({row.length_mcse:.3f}) & '
                    f'{row.cpu_seconds:.3f} ({row.cpu_seconds_mcse:.3f})'+r'\\')
        lines.append(r'\midrule' if bootstrap == 'normal' else r'\bottomrule')
    lines.append(r'\end{tabular}')
    path.write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = read_record(args.run_dir/'run_config.json')
    records = [read_record(p) for p in sorted((args.run_dir/'datasets').glob('*.json.gz'))]
    identities = [(r['experiment'], r['n'], r['replicate']) for r in records]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate saved task identities')
    for record, identity in zip(records, identities):
        validate_record(record, identity, config)
    expected = set(planned_tasks(config))
    if not set(identities) <= expected:
        raise ValueError('Unexpected records in the result archive')
    if set(identities) != expected and config['profile'] == 'full':
        raise ValueError('Full-run reporting requires every planned task; partial results cannot replace the paper')
    with np.load(args.run_dir/'truth_fisher.npz') as saved:
        reported_records = reconstruct_normality(records, saved['inverse'], config['truth'])
    corrected_coordinates = sum(
        r['experiment'] == 'estimation' and 'fit' in r and
        not np.allclose(r['estimate_free'], s['estimate_free'], atol=1e-10, rtol=0)
        for r, s in zip(records, reported_records))
    records = reported_records
    args.output.mkdir(parents=True, exist_ok=True)
    tables = summarize_additional(records)
    for name, table in tables.items():
        table.to_csv(args.output/f'{name}.csv', index=False)
    coverage = coverage_summary(records)
    coverage.to_csv(args.output/'table6_summary.csv', index=False)
    if not coverage.empty:
        table6_latex(coverage, args.output/'table6_rows.tex')
    figures(records, tables, args.output, config['profile'] != 'full')
    atomic_json(args.output/'reporting_definitions.json', dict(
        profile=config['profile'], complete=set(identities) == expected,
        figure10='Mean aligned center RMSE, absolute scale error, L1 weight error; ±1.96 dataset MCSE',
        figure11='Four zero-floor free-coordinate errors divided by independent truth-Fisher marginal SE; overlaid n; same-color points connected in quantile order; full range retains all points, central-range insets preserve the same ranks',
        figure11_coordinates='Reconstructed from saved physical estimates without the generic parameter packer 1e-12 clip; frozen raw records unchanged',
        figure11_raw_coordinate_corrections=corrected_coordinates,
        figure12='Unchanged mean total training NLL; top panels use ±1.96 dataset-level MCSE (SD/sqrt(R)); lower panels enlarge the same mean curves without error bars; one column per n',
        figure12_error_bar_target='Monte Carlo precision of the estimated mean total NLL, pointwise in m; not the spread of individual datasets',
        figure12_change='The original paper varied numerical seeds for one dataset; the additional setting varies datasets',
        table6='Dataset is the independent MCSE unit; both panels use CPU seconds including observed fit',
        invalid_results='Excluded from conditional error/coverage summaries; attempted and valid counts preserved',
        numerical_success_counts={kind: dict(attempted=sum(r['experiment'] == kind for r in records),
            valid=sum(r['experiment'] == kind and r['success'] for r in records))
            for kind in ('coverage', 'estimation', 'grid')}))
    print(f'Wrote three separate figures and Table 6 to {args.output}', flush=True)


if __name__ == '__main__':
    main()
