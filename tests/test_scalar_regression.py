"""Scalar regression still evaluates its usual objective and kernel predictions."""
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import orthodr.models as models


METHODS = ("sir", "save", "phd", "local", "seff")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n", [2, 12, 40])
def test_scalar_defaults_and_supplied_starts_use_the_normal_solver(monkeypatch, method, n):
    x = (np.arange(n, dtype=float) + 4)[:, None]
    y = x[:, 0] + .17 * np.sin(2 * x[:, 0])
    query = np.array([[3.8], [4 + n / 2 + .3], [4 + n + .2]])
    original_x, original_y = x.copy(), y.copy()
    native_solver = getattr(models._native(), method + "_solver")
    calls = []

    def solver(B, X, *args, **kwargs):
        calls.append((B.copy(), X.copy()))
        return native_solver(B, X, *args, **kwargs)

    def sliced(*args):
        pytest.fail("Scalar fitting must not use sliced initialization")

    monkeypatch.setattr(models, "regression_initializers", sliced)
    monkeypatch.setattr(models, "_native", lambda: SimpleNamespace(**{method + "_solver": solver}))
    options = dict(method=method, keep_data=True, ncore=1)
    supplied = models.orthoDr_reg(x, y, B_initial=np.ones((1, 1)), **options)
    # Omitted ndr defaults to two and is capped at the single predictor.
    for dimension in ({}, {"ndr": 1}, {"ndr": 7}):
        fit = models.orthoDr_reg(x, y, **dimension, **options)
        assert_array_equal(fit.B, [[1.]])
        assert fit.fn == supplied.fn
        assert fit.itr == supplied.itr == 0
        assert fit.converge and supplied.converge
        assert_array_equal(fit.predict(query), supplied.predict(query))
    negative_start = np.array([[-3.]])
    negative = models.orthoDr_reg(x, y, B_initial=negative_start, **options)
    assert_array_equal(negative.B, [[-1.]])
    assert_array_equal(negative_start, [[-3.]])
    assert_allclose(negative.predict(query), supplied.predict(query), atol=1e-12, rtol=1e-12)
    assert len(calls) == 5
    working = (x - x.mean(axis=0)) / x.std(ddof=1)
    for B, X in calls:
        assert B.shape == (1, 1)
        if method == "save":
            X = (X - X.mean(axis=0)) / X.std(ddof=1)
        assert_allclose(X, working, rtol=1e-14, atol=1e-14)
    assert_array_equal(x, original_x)
    assert_array_equal(y, original_y)
    assert_array_equal(fit.x, x)
    assert np.isfinite(fit.fn)

    # This prediction oracle uses raw scalar distances, without the fitted
    # basis or the package's prediction/preprocessing helpers.
    scale = np.std(x[:, 0], ddof=1) * fit.bw * np.sqrt(2)
    kx = np.exp(-((x[:, 0, None] - x[None, :, 0]) / scale) ** 2)
    weights = np.exp(-((query - x[:, 0]) / scale) ** 2)
    assert_allclose(fit.predict(query), weights @ y / weights.sum(axis=1),
                    atol=1e-12, rtol=1e-12)
    if method == "sir":
        # Independent semiparametric SIR estimating equation at the scalar
        # basis, including the response kernel used by the native solver.
        v = (y - y.mean()) / np.std(y, ddof=1) * n ** (1 / 6) / np.sqrt(2)
        ky = np.exp(-((v[:, None] - v[None, :]) ** 2))
        ex_y = ky @ x[:, 0] / ky.sum(axis=1)
        residual_x = x[:, 0] - kx @ x[:, 0] / kx.sum(axis=1)
        residual_ex_y = ex_y - kx @ ex_y / kx.sum(axis=1)
        assert_allclose(fit.fn, np.mean(residual_x * residual_ex_y) ** 2,
                        atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("x,y,match", [
    (np.array([[1.]]), np.array([2.]), "at least 2 rows"),
    (np.ones((2, 1)), np.array([1., 2.]), "varying predictor"),
    (np.array([[0.], [1.]]), np.ones(2), "positive sample variance"),
    (np.arange(12.)[:, None], np.full(12, .1), "positive sample variance"),
    (np.array([[0.], [np.nan]]), np.array([1., 2.]), "finite"),
    (np.array([[0.], [np.inf]]), np.array([1., 2.]), "finite"),
    (np.array([[0.], [1.]]), np.array([1., np.nan]), "finite"),
    (np.array([[0.], [1.]]), np.array([1., np.inf]), "finite"),
])
def test_invalid_scalar_data_fail_before_native_work(monkeypatch, x, y, match):
    monkeypatch.setattr(models, "_native", lambda: pytest.fail("Invalid data must fail before native work"))
    for method in METHODS:
        for start in (None, np.ones((1, 1))):
            with pytest.raises(ValueError, match=match):
                models.orthoDr_reg(x, y, method=method, B_initial=start)


@pytest.mark.parametrize("method", METHODS)
def test_zero_supplied_scalar_basis_is_rejected(method):
    with pytest.raises(ValueError, match="full column rank"):
        models.orthoDr_reg(np.array([[0.], [1.]]), np.array([1., 2.]),
                          method=method, B_initial=np.zeros((1, 1)))
