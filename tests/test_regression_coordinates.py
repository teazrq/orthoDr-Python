"""Independent moment and prediction checks for regression fitting."""
from math import ldexp

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from orthodr import _core, orthoDr_reg


def _weights(z):
    distances = z[:, None, :] - z[None, :, :]
    kernel = np.exp(-np.sum(distances**2, axis=2))
    return kernel / kernel.sum(axis=1, keepdims=True)


def _objective(x, Y, B, bw, method):
    centered = x - x.mean(axis=0)
    projected = centered @ B
    wx = _weights(projected / (projected.std(axis=0, ddof=1) * bw * np.sqrt(2)))
    wy = _weights(Y)
    if method == "sir":
        ey = wy @ centered
        moment = (ey - wx @ ey).T @ (centered - wx @ centered) / len(x)
    elif method == "phd":
        xx = (centered[:, :, None] * centered[:, None, :]).reshape(len(x), -1)
        moment = (xx - wx @ xx).T @ (Y - wx @ Y) / len(x)
    else:
        def conditional(w):
            mean = w @ centered
            deviations = centered[None, :, :] - mean[:, None, :]
            cov = np.einsum("ij,ijk,ijl->ikl", w, deviations, deviations)
            return centered - mean, cov

        rx, cx = conditional(wx)
        ry, cy = conditional(wy)
        moment = ((np.eye(x.shape[1]) - cy) @ (ry[:, :, None] * rx[:, None, :] - cx)).mean(axis=0)
    return np.sum(moment**2)


@pytest.mark.parametrize("method,factor,shift", [("sir", .01, 0), ("save", 1, 1e5)])
def test_fixed_basis_and_tangent_derivative(method, factor, shift):
    rng = np.random.default_rng(139)
    x = rng.normal(size=(32, 4))
    y = x[:, 0] + 0.4 * x[:, 1]**2 + rng.normal(scale=.2, size=len(x))
    Y = ((y - y.mean()) / y.std(ddof=1) * len(x)**(1 / 6) / np.sqrt(2))[:, None]
    bw = .7
    base, tangent = np.eye(4)[:, :1], np.eye(4)[:, 1:2]
    objective = getattr(_core, method + "_init")
    h = 1e-5
    plus, minus = [np.cos(h) * base + sign * np.sin(h) * tangent for sign in (1, -1)]
    oracle = [_objective(x, Y, B, bw, method) for B in (base, plus, minus)]
    units = factor**4 if method == "sir" else 1
    transformed = x * factor + shift
    native = [objective(B, transformed, Y, bw, 1) / units for B in (base, plus, minus)]
    assert_allclose(native, oracle, rtol=2e-10, atol=1e-13)
    assert_allclose((native[1] - native[2]) / (2 * h),
                    (oracle[1] - oracle[2]) / (2 * h), rtol=2e-7, atol=1e-10)
    fit = orthoDr_reg(transformed, y, method=method, ndr=1, B_initial=base,
                      bw=bw, control={"gtol": 1e5})
    assert fit.itr == 0
    assert_allclose(fit.fn / units, oracle[0], rtol=2e-10, atol=1e-13)


def test_save_local_covariances_with_separated_means():
    # The two exact Gaussian partitions intersect differently. In each, some
    # local means lie between adjacent representable predictor values.
    groups_x = np.array([0, 1, 1, 0, 1, 0, 1, 1, 3, 3, 2, 2, 2, 2, 3, 2])
    groups_y = np.array([1, 0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 3, 2, 3, 3, 2])
    offsets = np.repeat([-2.**50, 2.**50], 8)
    local_x = np.column_stack((groups_x, np.tile([0]*7 + [1], 2)))
    x = local_x + np.column_stack((np.zeros(16), offsets))
    Y = 100. * groups_y[:, None]
    B, bw = np.array([[1.], [0.]]), .005

    def moments(groups):
        # The kernel has weight 1 for matching labels and 0 otherwise.
        # Each group belongs to one offset, so the unshifted coordinates
        # produce an independent exact-within-group covariance reference.
        means = np.array([local_x[groups == g].mean(axis=0) for g in groups])
        covs = np.array([np.cov(local_x[groups == g].T, ddof=0) for g in groups])
        return local_x - means, covs

    rx, cx = moments(groups_x)
    ry, cy = moments(groups_y)
    reference = np.sum(((np.eye(2) - cy) @ (ry[:, :, None] * rx[:, None, :] - cx)).mean(axis=0)**2)
    assert_allclose(_core.save_init(B, x, Y, bw, 1), reference, rtol=1e-14)
    fit = _core.save_solver(B, x, Y, bw, 1e-4, .2, .85, 1e-3,
                            1e-6, 1e-6, 1e-6, 1e100, 5, 0, 1)
    assert fit["itr"] == 0
    assert_allclose(fit["fn"], reference, rtol=1e-14)


def _unit_change_data():
    rng = np.random.default_rng(117)
    x = np.round(32 * rng.normal(size=(56, 4))) / 32
    y = x[:, 0]**2 + .5 * x[:, 1] + rng.normal(scale=.15, size=len(x))
    return x, y


@pytest.mark.parametrize("start", [None, np.array([[.3], [1.], [.5], [-.25]])])
@pytest.mark.parametrize("method,power", [("phd", 4), ("local", 2), ("seff", 2)])
def test_small_predictor_units_preserve_directions_objective_and_prediction(method, power, start):
    x, y = _unit_change_data()
    factor = 2.**-10
    options = dict(method=method, ndr=1, B_initial=start, bw=.65,
                   keep_data=True, maxitr=12)
    base = orthoDr_reg(x, y, **options)
    fit = orthoDr_reg(factor * x, y, **options)
    assert fit.itr == base.itr > 0
    assert_allclose(fit.B @ fit.B.T, base.B @ base.B.T, rtol=1e-10, atol=1e-12)
    assert_allclose(fit.fn / factor**power, base.fn, rtol=1e-10, atol=0)
    query = x[1:6] + .125
    assert_allclose(fit.predict(factor * query), base.predict(query), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("start", [None, np.array([[.3], [1.], [.5], [-.25]])])
def test_phd_uses_centered_outer_products(start):
    x, y = _unit_change_data()
    shifted = x + [10, 3, -2, 5]
    options = dict(method="phd", ndr=1, B_initial=start, bw=.65,
                   keep_data=True, maxitr=12)
    base = orthoDr_reg(x, y, **options)
    fit = orthoDr_reg(shifted, y, **options)
    assert_allclose(fit.B @ fit.B.T, base.B @ base.B.T, rtol=1e-10, atol=1e-12)
    Y = ((y - y.mean()) / y.std(ddof=1) * len(x)**(1 / 6) / np.sqrt(2))[:, None]
    assert_allclose(fit.fn, _objective(x, Y, fit.B, .65, "phd"), rtol=1e-10, atol=0)
    assert_allclose(fit.predict(shifted[1:6] + .125), base.predict(x[1:6] + .125),
                    rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("start", [None, np.array([[.3], [1.], [.5], [-.25]])])
@pytest.mark.parametrize("factor", [.001, 16.])
def test_save_descends_and_reports_the_original_estimating_equation(factor, start):
    rng = np.random.default_rng(139)
    x = rng.normal(size=(32, 4))
    y = x[:, 0] + .4 * x[:, 1]**2 + rng.normal(scale=.2, size=len(x))
    Y = ((y - y.mean()) / y.std(ddof=1) * len(x)**(1 / 6) / np.sqrt(2))[:, None]
    x = factor * x
    options = dict(method="save", ndr=1, B_initial=start, bw=.7, maxitr=12)
    initial = orthoDr_reg(x, y, **options, control={"gtol": 1e100})
    fit = orthoDr_reg(x, y, **options)
    assert initial.itr == 0
    assert fit.itr > 0 and fit.fn < initial.fn
    assert_allclose(fit.fn, _objective(x, Y, fit.B, .7, "save"), rtol=1e-10, atol=0)


def test_sir_reports_range_safe_objective_and_stable_predictions():
    x = np.array([[-1.], [1.]])
    y = np.array([0., 1.])
    query = np.array([[0.], [.5]])
    base = orthoDr_reg(x, y, keep_data=True)
    for power in (256, -260):
        factor = ldexp(1., power)
        fit = orthoDr_reg(factor * x, y, keep_data=True)
        expected = ldexp(base.fn, 4 * power)
        assert np.isfinite(fit.fn) and fit.fn > 0
        assert_allclose(fit.fn, expected, rtol=1e-11, atol=0)
        assert_array_equal(fit.B, base.B)
        assert_allclose(fit.predict(factor * query), base.predict(query), rtol=1e-14)
        assert_array_equal(fit.transform(factor * query), factor * query)
    extreme = np.finfo(float).max
    fit = orthoDr_reg(extreme * x, y, keep_data=True)
    assert fit.converge and np.isposinf(fit.fn)
    assert_allclose(fit.predict(extreme * query), base.predict(query), rtol=1e-14)


@pytest.mark.parametrize("method,outlier_y,reference", [
    ("sir", 10., 1.89117288877e-10),
    ("save", 1e6, 9.2609525559169e-6),
])
def test_narrow_kernels_retain_resolution_with_a_distant_first_observation(method, outlier_y, reference):
    t = np.arange(64.)
    x = np.column_stack((np.r_[2.**54, t % 4], np.r_[0., np.round(16 * np.sin(1.21 * t)) / 16]))
    y = np.r_[outlier_y, np.sin(2.1 * t)]
    bw = .35 / x[:, 0].std(ddof=1)
    for order in (np.arange(len(x)), np.r_[1:len(x), 0]):
        fit = orthoDr_reg(x[order], y[order], method=method, ndr=1,
                          B_initial=np.array([[1.], [0.]]), bw=bw, control={"gtol": 1e100})
        assert fit.itr == 0
        assert_allclose(fit.fn, reference, rtol=1e-8)


def test_sir_narrow_kernel_preserves_both_separated_clusters():
    t = np.arange(64.)
    local = np.tile(np.column_stack((.25 * (t % 4), np.round(16 * np.sin(1.21 * t)) / 16)), (2, 1))
    block = np.repeat([-1., 1.], len(t))
    offset = 2.**50
    x = local.copy()
    x[:, 0] += offset * block
    y = 2 * block + np.tile(np.sin(2.1 * t), 2)
    bw = .35 * .25 / x[:, 0].std(ddof=1)
    Y = ((y - y.mean()) / y.std(ddof=1) * len(x)**(1 / 6) / np.sqrt(2))[:, None]
    wx = np.exp(-((local[:, :1] - local[:, 0]) / (.35 * .25 * np.sqrt(2)))**2)
    wx *= block[:, None] == block
    wx /= wx.sum(axis=1, keepdims=True)
    wy = _weights(Y)
    # Keep each cluster's offset separate from its local coordinates so
    # neither the Gaussian distances nor the conditional means erase gaps.
    ey = wy @ local
    r_ey = ey - wx @ ey
    ey_block = wy @ block
    r_ey[:, 0] += offset * (ey_block - wx @ ey_block)
    residual = local - wx @ local
    reference = np.sum((r_ey.T @ residual / len(x))**2)
    for order in (np.arange(len(x)), np.r_[64:128, 0:64]):
        fit = orthoDr_reg(x[order], y[order], ndr=1, B_initial=np.array([[1.], [0.]]),
                          bw=bw, control={"gtol": 1e100})
        assert fit.itr == 0
        assert_allclose(fit.fn, reference, rtol=1e-10)


@pytest.mark.parametrize("method,reference", [("sir", 2.4256406167671955e61), ("save", 2.131848331387999e126)])
def test_fitting_preserves_a_contrast_between_large_nearly_equal_columns(method, reference):
    x = np.array([[2.**54, 2.**54]] + [[0., 0.], [1., 0.], [0., 1.], [1., 1.]] * 3)
    y = np.sin(1.7 * np.arange(len(x)))
    B = np.array([[1.], [-1.]]) / np.sqrt(2)
    for order in (np.arange(len(x)), np.r_[1:len(x), 0]):
        fit = orthoDr_reg(x[order], y[order], method=method, ndr=1,
                          B_initial=B, bw=.6, control={"gtol": 1e150})
        assert fit.itr == 0
        assert_allclose(fit.fn, reference, rtol=1e-10)


@pytest.mark.parametrize("method", ["sir", "save"])
def test_supplied_start_with_more_predictors_than_observations(method):
    x = np.arange(20).reshape(4, 5)
    y = [0., 1., 3., 2.]
    fit = orthoDr_reg(x, y, method=method, ndr=1, B_initial=np.eye(5)[:, :1],
                      maxitr=5, keep_data=True)
    assert np.isfinite(fit.fn)
    assert np.isfinite(fit.predict(x)).all()
