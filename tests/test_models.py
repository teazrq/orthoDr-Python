"""Public behavior checked against simple truths and array/persistence contracts."""
import pickle

import numpy as np
from numpy.testing import assert_allclose
import pytest

from orthodr import FitResult, orthoDr_pdose, orthoDr_reg, orthoDr_surv, predict


def test_regression_prediction_matches_hand_weighted_mean():
    # At zero, these two observations have equal Gaussian weight.
    fit = FitResult(B=np.ones((1, 1)), x=np.array([[-1.], [1.]]),
                    y=np.array([2., 6.]), bw=.8, kind="reg",
                    **{"keep.data": True})
    assert_allclose(fit.predict(np.array([[0.]])), [4.])
    assert predict(fit, np.array([[0.]]))["pred"].shape == (1,)
    assert fit.predict(np.empty((0, 1))).shape == (0,)
    with pytest.warns(RuntimeWarning, match="underflow"):
        assert np.isnan(fit.predict(np.array([[1e8]]))[0])


def test_tied_survival_risk_set_matches_hand_kaplan_meier():
    # Projection values +/-1 give equal weights at zero. The censoring at
    # t=1 remains in that time's risk set: two of four fail, then one of one.
    fit = FitResult(B=np.ones((1, 1)), x=np.array([[-1.], [1.], [-1.], [1.]]),
                    y=np.array([1., 1., 1., 2.]), censor=np.array([1, 1, 0, 1]),
                    bw=.8, kind="surv", **{"keep.data": True})
    result = fit.predict(np.array([[0.]]))
    assert_allclose(result["timepoints"], [1, 1, 2])
    assert_allclose(result["surv"], [[.5], [.5], [0.]])
    empty = fit.predict(np.empty((0, 1)))
    assert empty["surv"].shape == (3, 0)


def _dose_data():
    rng = np.random.default_rng(509)
    x = rng.normal(size=(28, 3))
    a = rng.uniform(0, 4, size=len(x))
    r = np.exp(-(a - 2 - .3 * x[:, 0]) ** 2)
    return x, a, r


@pytest.mark.parametrize("intercept", [False, True])
def test_direct_fit_preserves_inputs_and_numpy_layouts(intercept):
    x, a, r = _dose_data()
    basis = np.array([[1.], [.2], [-.3]])
    originals = [v.copy() for v in (x, a, r, basis)]
    opts = dict(ndr=1, B_initial=basis, maxitr=3, keep_data=True, intercept=intercept)
    fit = orthoDr_pdose(x, a, r, **opts)
    # Fortran ordering and negative strides exercise explicit NumPy/Armadillo
    # conversions while retaining exactly the same observations and order.
    a_reversed_storage = a[::-1].copy()[::-1]
    alternate = orthoDr_pdose(np.asfortranarray(x), a_reversed_storage, r, **opts)
    assert_allclose(alternate.B, fit.B, atol=0, rtol=0)
    assert_allclose(alternate.predict(x), fit.predict(x), atol=0, rtol=0)
    for before, after in zip(originals, (x, a, r, basis)):
        assert_allclose(after, before, atol=0, rtol=0)
    assert_allclose(fit.B.T @ fit.B, np.eye(1), atol=1e-12)
    assert_allclose(fit.fn, -fit.value + fit.penalty, atol=1e-12)
    assert_allclose(fit.penalty, fit.lambda_reward * np.sum(fit.W ** 2), atol=1e-12)
    assert fit.predict(np.empty((0, 3))).shape == (0,)
    if intercept:
        assert_allclose(fit.predict(x).mean(), fit.intercept, atol=1e-12)
    restored = pickle.loads(pickle.dumps(fit))
    assert_allclose(restored.predict(x[:3]), fit.predict(x[:3]), atol=0, rtol=0)
    # Retained training data must not alias the caller's mutable array.
    x[:] = 999
    assert_allclose(fit.x, originals[0], atol=0, rtol=0)


def test_dose_initialization_can_return_without_optimization():
    x, a, r = _dose_data()
    fit = orthoDr_pdose(x, a, r, ndr=1, B_initial=np.eye(3)[:, :1],
                       maxitr=0, keep_data=True, intercept=True)
    assert fit.itr == 0
    assert not fit.converge
    assert fit.stop_reason == "iteration_limit"
    assert np.isfinite(fit.predict(x)).all()


def test_prediction_requires_retained_training_data():
    x, a, r = _dose_data()
    fit = orthoDr_pdose(x, a, r, ndr=1, B_initial=np.eye(3)[:, :1], maxitr=0)
    with pytest.raises(ValueError, match="keep_data"):
        fit.predict(x)
    assert_allclose(fit.transform(x), x @ fit.B)


@pytest.mark.parametrize("function", [orthoDr_reg, orthoDr_surv, orthoDr_pdose])
def test_invalid_training_retention_flag_fails_before_native_fitting(function):
    args = (np.ones((3, 2)), np.arange(3.))
    if function is not orthoDr_reg:
        args += (np.ones(3),)
    with pytest.raises(ValueError, match="keep_data"):
        function(*args, keep_data="yes")


def test_direct_argument_contract_and_stored_intercept_validation():
    x, a, r = _dose_data()
    basis = np.eye(3)[:, :1]
    for kwargs in (dict(intercept=True, lambda_reward=0),
                   dict(intercept_bounds=[0, 1]),
                   dict(method="pseudo_direct", lambda_reward=.1),
                   dict(lambda_init=0), dict(ndr=4), dict(bw=0),
                   dict(control={"eta": 1}), dict(maxitr=-1)):
        with pytest.raises(ValueError):
            orthoDr_pdose(x, a, r, B_initial=basis, **kwargs)
    fit = orthoDr_pdose(x, a, r, ndr=1, B_initial=basis, maxitr=0,
                       intercept=True, keep_data=True)
    del fit["kernel.mean"]
    with pytest.raises(ValueError, match="stored"):
        fit.predict(x[:1])
