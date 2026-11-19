"""Gaussian regression and survival at the limits of floating-point weights."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from orthodr import FitResult, orthoDr_reg, orthoDr_surv


def stored_fit(x, y, bw, *, basis=None, event=None):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if basis is None:
        basis = np.ones((1, 1))
    fit = FitResult(B=basis, x=x, y=np.asarray(y, dtype=float), bw=bw,
                    kind="reg" if event is None else "surv", **{"keep.data": True})
    if event is not None:
        fit["censor"] = np.asarray(event)
    return fit


def test_supported_curve_retains_terminal_failure_at_remote_training_point():
    n = 256
    x = np.r_[np.linspace(-1, 1, n - 1), 100.][:, None]
    times = np.arange(1, n + 1, dtype=float)
    failures = [60, 120, 180, 256]
    fit = orthoDr_surv(x, times, np.isin(times, failures), ndr=1,
                      method="forward", keep_data=True)
    result = fit.predict([[0.]])
    assert_array_equal(result["timepoints"], failures)
    assert_allclose(result["surv"][:3, 0],
                    [.9949181125960931, .9873658069139568, .9738847865250174],
                    rtol=1e-13, atol=0)
    assert result["surv"][-1, 0] == 0

    with pytest.warns(RuntimeWarning, match="underflow") as caught:
        batch = fit.predict([[0.], [1e6]])["surv"]
    assert len(caught) == 1
    assert_array_equal(batch[:, 0], result["surv"][:, 0])
    assert np.isnan(batch[:, 1]).all()


def test_weighted_km_preserves_tiny_survivor_fractions_and_cumulative_losses():
    fit = stored_fit([0, 1], [1, 2], 1 / np.sqrt(20 * np.log(10)), event=[1, 0])
    assert_allclose(fit.predict([[0.]])["surv"][0, 0], 1e-20, rtol=1e-13, atol=0)

    n = 200
    x = np.r_[np.ones(n), 0]
    bw = 1 / (x.std(ddof=1) * np.sqrt(2 * 38))
    fit = stored_fit(x, np.arange(n + 1), bw, event=np.r_[np.ones(n), 0])
    result = fit.predict([[0.]])["surv"][-1, 0]
    assert result < 1
    assert_allclose(result, np.exp(-np.log1p(n * np.exp(-38))),
                    atol=np.finfo(float).eps, rtol=0)


def test_unresolvable_mixed_risk_set_reports_overflowed_log_distances():
    fit = stored_fit([0, 100, 100], [1, 2, 3], 1e-200, event=[0, 1, 0])
    with pytest.raises(ValueError, match="overflowed in a mixed survival risk set"):
        fit.predict([[0.]])


def test_gaussian_mean_recovers_products_that_absolute_weights_erase():
    x = np.r_[np.linspace(-1, -.9, 20), np.linspace(.9, 1, 20)][:, None]
    y = .4 + .09 * (x[:, 0] + 1) / 2
    fit = orthoDr_reg(x, y, ndr=1, bw=.02423, keep_data=True, ncore=1)
    assert_allclose(fit.predict([[0.]]), [.445], rtol=1e-13, atol=0)
    assert fit.predict(np.empty((0, 1))).shape == (0,)

    fit = orthoDr_reg(np.array([[0.], [1.]]), [0., 1e150], ndr=1, bw=1 / np.sqrt(750),
                      keep_data=True, ncore=1)
    assert_allclose(fit.predict([[0.]]), [1.9016849634750065e-176], rtol=1e-12, atol=0)


@pytest.mark.parametrize("response", [np.finfo(float).smallest_subnormal,
                                      np.finfo(float).max, -np.finfo(float).max])
def test_gaussian_mean_retains_subnormal_responses_and_avoids_overflow(response):
    fit = stored_fit([0, 0, 0, 1], [response, response, response, 0], .01)
    assert_array_equal(fit.predict([[0.]]), [response])


def test_fixed_basis_regression_keeps_units_and_common_origin():
    x = np.array([[0., 0.], [.25, 0.], [-.25, .25], [.5, .25]])
    queries = np.array([[.25, .25], [-.25, 0.]])
    y = np.array([-3., 2., 4., -1.])
    basis = np.full((2, 1), np.sqrt(.5))
    fit = stored_fit(x, y, .4, basis=basis)
    expected = fit.predict(queries)
    assert np.all((expected >= y.min()) & (expected <= y.max()))
    for scale in (1e-200, 1e200):
        fit["x"] = x * scale
        assert_allclose(fit.predict(queries * scale), expected, rtol=1e-13, atol=0)
    fit["x"] = x + 2**50
    assert_array_equal(fit.predict(queries + 2**50), expected)


def test_extreme_training_anchor_keeps_neighbor_distances():
    x = np.array([2**54, 0., 1., 2.])
    bw = .22 / x.std(ddof=1)
    epsilon = np.exp(-.5 / .22**2)
    fit = stored_fit(x, [0, 0, 1, 0], bw)
    assert_allclose(fit.predict([[1.]]), [1 / (1 + 2 * epsilon)], rtol=1e-13, atol=0)

    fit = stored_fit(x, [1, 2, 3, 4], bw, event=[0, 0, 1, 0])
    curve = fit.predict([[1.]])["surv"]
    assert_allclose(curve, [[epsilon / (1 + epsilon)]], rtol=1e-13, atol=0)
    order = [1, 2, 3, 0]
    for key in ("x", "y", "censor"):
        fit[key] = fit[key][order]
    assert_allclose(fit.predict([[1.]])["surv"], curve, rtol=1e-13, atol=0)


def test_finite_standardized_distances_survive_overflow_in_raw_differences():
    fit = stored_fit([-3e307, 0, 3e307], [0, 0, 1], 10)
    assert_allclose(fit.predict([[1.7e308]]), [.3518031938714887], rtol=1e-13, atol=0)


def test_cancelling_predictors_preserve_the_training_projection_scale():
    x = np.array([[2.**54, 2.**54], [0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    y = np.array([0., 0., 1., -1., 0.])
    basis = np.array([[1.], [-1.]]) / np.sqrt(2)
    reduced = np.array([0., 0., 1., -1., 0.]) / np.sqrt(2)
    weight = np.exp(-.5 * ((reduced - reduced[2]) / (.6 * reduced.std(ddof=1))) ** 2)
    expected = weight @ y / weight.sum()
    for order in ([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]):
        fit = stored_fit(x[order], y[order], .6, basis=basis)
        assert_allclose(fit.predict([[1., 0.]]), [expected], rtol=1e-13, atol=0)
