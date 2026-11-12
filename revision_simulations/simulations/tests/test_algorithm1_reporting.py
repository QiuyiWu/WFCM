import numpy as np


def test_qq_reporting_uses_unclipped_aligned_weights_and_preserves_raw_records():
    from reporting.algorithm1_results import reconstruct_normality
    truth = dict(sigma=.9, centers=[[-2.], [2.]], weights=[.35, .65])
    record = dict(experiment='estimation', n=500, success=True,
        marginal_z=[0., 0., 0., 0.], estimate_free=[0., 0., 0., 0.],
        fit=dict(sigma=.9, centers=[[2.], [-2.]], w=[1-1e-14, 1e-14]))
    corrected = reconstruct_normality([record], np.eye(4), truth)[0]
    expected = (np.log(1e-14)-np.log(1-1e-14)-np.log(.35/.65))*np.sqrt(500)
    np.testing.assert_allclose(corrected['marginal_z'][3], expected, atol=1e-12)
    np.testing.assert_allclose(corrected['marginal_z'][:3], 0., atol=1e-12)
    assert record['marginal_z'] == [0., 0., 0., 0.]


def test_reporting_keeps_failed_datasets_and_uses_dataset_mcse():
    from reporting.algorithm1_results import summarize_additional
    records = [dict(experiment='estimation', n=200, replicate=i, success=ok,
        center_rmse=error, sigma_abs_error=error, weight_l1_error=error,
        coordinates=['center_1'], marginal_z=[error])
        for i, (ok, error) in enumerate([(True, 1.), (True, 3.), (False, 100.)])]
    tables = summarize_additional(records)
    row = tables['estimation'].iloc[0]
    assert row.attempted == 3 and row.valid == 2
    assert row.center_rmse == 2
    assert row.center_rmse_mcse == 1


def test_grid_reporting_uses_total_nll_sd_and_records_selection_denominators():
    from reporting.algorithm1_results import summarize_additional
    records = [dict(experiment='grid', n=200, replicate=i, success=ok,
                    selected_m=m, candidates=[
        dict(candidate_m=1.5, average_nll=a, success=ok),
        dict(candidate_m=2., average_nll=b, success=ok)])
        for i, (ok,m,a,b) in enumerate([(True,1.5,1.,2.), (True,2.,3.,2.), (False,None,100.,100.)])]
    tables = summarize_additional(records)
    row = tables['m_grid'].query('m == 1.5').iloc[0]
    assert row.mean_total_nll == 400
    np.testing.assert_allclose(row.sd_total_nll, np.sqrt(80000))
    assert row.attempted == 3 and row.valid == 2
    selected = tables['m_selection'].query('m == 2.0').iloc[0]
    assert selected.frequency_valid == .5
    assert selected.frequency_attempted == 1/3
