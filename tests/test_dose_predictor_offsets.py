import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose

from orthodr import _core, orthoDr_pdose


def test_direct_dose_fitting_and_basis_gradient_use_predictor_differences():
    i = np.arange(32)
    x = np.column_stack(((11 * i + 4) % 33 - 16, (7 * i**2 + 13 * i + 2) % 33 - 16))
    query = np.column_stack(((17 * i + 3) % 33 - 16, (19 * i + 7) % 33 - 16))
    a = ((23 * i + 5) % 37) / 36
    r = -(a - .5 - .3 * np.sin(x.sum(axis=1) / 8))**2

    def fit(z):
        return orthoDr_pdose(z, a, r, ndr=1, B_initial=[[1.], [0.]], intercept=True,
                             keep_data=True, ncore=1)

    base = fit(x)

    def evaluate(z):
        return _core.pdose_direct_evaluate(base.B, z, a, r, base.bw, base.W, .1,
                                           0., True, np.array([a.min(), a.max()]), base.intercept)

    state = evaluate(x)
    assert state["valid"]
    for offset in (1e9, 2.**52):
        shifted = fit(x + offset)
        moved_state = evaluate(x + offset)
        assert moved_state["valid"]
        assert_allclose(moved_state["gB"], state["gB"], atol=1e-12, rtol=0)
        assert_allclose(shifted.B, base.B, atol=1e-12, rtol=0)
        assert_allclose(shifted.W, base.W, atol=1e-12, rtol=0)
        assert shifted.stop_reason == base.stop_reason
        assert_allclose(shifted.predict(query + offset), base.predict(query), atol=1e-12, rtol=0)


@pytest.mark.parametrize("method", ["direct", "pseudo_direct"])
@pytest.mark.parametrize("intercept", [False, True])
def test_dose_prediction_resolves_small_index_differences_after_exact_shift(method, intercept):
    u, z, a = np.array(list(itertools.product((-8., 0., 8.), (-1., 0., 1.), np.linspace(0., 1., 9)))).T
    x = np.column_stack((u, z - u))
    r = -(a - (.5 + .4 * z))**2
    query = np.column_stack((np.zeros(3), np.arange(-1., 2.)))
    offset = np.array([2.**55, 0.])

    def fit(z):
        return orthoDr_pdose(z, a, r, ndr=1, B_initial=[[1.], [1.]], bw=.6, K=9,
                             method=method, intercept=intercept, keep_data=True, ncore=1)

    base, shifted = fit(x), fit(x + offset)
    train_index, query_index = (x @ shifted.B).ravel(), (query @ shifted.B).ravel()
    delta = (query_index[:, None] - train_index) / (train_index.std(ddof=1) * shifted.bw)
    expected = np.exp(-.5 * delta**2) @ shifted.W.ravel()
    if intercept:
        expected += shifted.intercept
        if method == "direct":
            expected -= np.dot(shifted["kernel.mean"].ravel(), shifted.W.ravel())
    actual = shifted.predict(query + offset)
    assert np.ptp(expected) > .1
    assert_allclose(actual, expected, atol=1e-12, rtol=0)
    assert_allclose(actual, base.predict(query), atol=1e-12, rtol=0)


def test_large_observation_with_zero_contrast_preserves_dose_kernel():
    s = np.tile([-1., 0., 1.], 12)
    x = np.vstack(([2.**55, 2.**55], np.column_stack((s, np.zeros_like(s)))))
    b = np.array([[1.], [-1.]]) / np.sqrt(2)
    q = np.column_stack(([-1., 0., 1.], np.zeros(3)))
    w, z = np.r_[0., -s / 12], np.r_[0., s] / np.sqrt(2)
    expected = np.exp(-.5 * ((q @ b - z) / (.5 * z.std(ddof=1)))**2) @ w
    for order in (np.arange(len(x)), np.roll(np.arange(len(x)), -1)):
        actual = _core.dosepred(b, x[order], q, .5, w[order]).ravel()
        assert_allclose(actual, expected, atol=1e-12, rtol=0)
