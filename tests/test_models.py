"""Public behavior checked against simple truths and array/persistence contracts."""
import pickle

import numpy as np
from numpy.testing import assert_allclose
import pytest

from orthodr import (CP_SIR, FitResult, orthoDr_pdose, orthoDr_reg, orthoDr_surv,
                     predict, view_dr_surv)


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


@pytest.mark.parametrize("ndr", [1, 2])
@pytest.mark.parametrize("custom_start", [False, True])
def test_seff_fit_and_prediction_are_invariant_to_predictor_translation(ndr, custom_start):
    rng = np.random.default_rng(541)
    # Dyadic observations and offsets make the centered training matrices
    # bitwise equal. This checks the translation property without exciting
    # unrelated finite-difference sensitivity to last-bit input differences.
    x = rng.integers(-16, 17, size=(64, 4)) / 8
    y = x[:, 0] + .6 * x[:, 1] ** 2 + rng.normal(scale=.3, size=len(x))
    query = rng.integers(-16, 17, size=(7, 4)) / 8
    shift = np.array([8., -4., 16., 2.])
    basis = np.array([[1., .2], [.3, 1.], [-.4, .5], [.2, -.3]])[:, :ndr]
    options = dict(method="seff", ndr=ndr, maxitr=7, bw=.7, keep_data=True,
                   B_initial=basis if custom_start else None)
    original_x, original_basis = x.copy(), basis.copy()
    fit = orthoDr_reg(x, y, **options)
    moved = orthoDr_reg(x + shift, y, **options)
    assert_allclose(moved.B, fit.B, atol=0, rtol=0)
    assert_allclose(moved.fn, fit.fn, atol=0, rtol=0)
    assert moved.itr == fit.itr
    assert moved.converge == fit.converge
    assert np.isfinite(fit.fn)
    assert_allclose(fit.B.T @ fit.B, np.eye(ndr), atol=2e-12, rtol=0)
    predictions = fit.predict(query)
    assert np.isfinite(predictions).all()
    assert_allclose(moved.predict(query + shift), predictions, atol=2e-13, rtol=0)
    # The public basis and retained predictors stay in the supplied coordinates.
    assert_allclose(fit.x, x, atol=0, rtol=0)
    assert_allclose(moved.x, x + shift, atol=0, rtol=0)
    assert_allclose(moved.transform(query + shift), (query + shift) @ moved.B,
                    atol=0, rtol=0)
    restored = pickle.loads(pickle.dumps(moved))
    assert_allclose(restored.predict(query + shift), moved.predict(query + shift),
                    atol=0, rtol=0)
    assert_allclose(x, original_x, atol=0, rtol=0)
    assert_allclose(basis, original_basis, atol=0, rtol=0)


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


@pytest.mark.parametrize("function", [orthoDr_surv, CP_SIR, view_dr_surv])
def test_invalid_survival_status_reports_original_rows(function):
    x = np.arange(7.).reshape(-1, 1)
    y = np.array([6., 7., 2., 4., 1., 5., 3.])
    censor = [1., 2., 0., .5, np.nan, np.inf, -1.]
    with pytest.raises(ValueError, match="0/1 indicators") as error:
        function(x, y, censor)
    message = str(error.value)
    assert "0 for right censoring and 1 for an observed event" in message
    assert "0-based row index" in message
    assert "1=2.0, 3=0.5, 4=nan, 5=inf, 6=-1.0" in message


def test_survival_status_bounds_examples_and_validates_vector_inputs():
    x = np.arange(10.).reshape(-1, 1)
    y = np.arange(10.)
    with pytest.raises(ValueError, match="0=2, 1=2, 2=2, 3=2, 4=2 and 5 more"):
        CP_SIR(x, y, np.full(10, 2))
    with pytest.raises(ValueError, match="one value per observation"):
        CP_SIR(x, y, np.ones(9))
    for censor in (np.ones((10, 1)), ["1"] * 10):
        with pytest.raises(ValueError, match="one-dimensional numeric or Boolean vector"):
            CP_SIR(x, y, censor)


@pytest.mark.parametrize("function", [orthoDr_surv, CP_SIR])
def test_survival_estimators_require_an_event(function):
    x = np.arange(6.).reshape(-1, 1)
    with pytest.raises(ValueError, match="at least one .*event"):
        function(x, np.arange(6.), np.zeros(6, dtype=bool))


def test_retained_survival_predictions_validate_stored_status():
    x = np.arange(1., 7.).reshape(-1, 1)
    censor = np.array([True, False, True, True, False, True])
    fit = orthoDr_surv(x, x[:, 0][::-1], censor, keep_data=True, maxitr=0)
    restored = pickle.loads(pickle.dumps(fit))
    assert_allclose(restored.predict(x[:1])["timepoints"], [1., 3., 4., 6.])
    restored.censor[1] = 2
    restored.censor[4] = np.nan
    with pytest.raises(ValueError, match="stored survival status") as error:
        restored.predict(x[:1])
    assert "1=2.0, 4=nan" in str(error.value)
    assert "Recode the original status data and refit" in str(error.value)


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
