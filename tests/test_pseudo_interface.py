"""Pseudo-direct public behavior, independent of optimizer path matching."""
import pickle
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
import pytest

from orthodr import orthoDr_pdose


@pytest.mark.parametrize("method", ["pseudo_direct", "direct", "direct_intercept"])
def test_genuine_old_saved_predictions(method):
    with (Path(__file__).parent / "fixtures" / ("python-" + method + ".pickle")).open("rb") as stream:
        saved = pickle.load(stream)
    assert_allclose(saved["fit"].predict(saved["testx"]), saved["pred"], atol=2e-12, rtol=0)


def data():
    rng = np.random.default_rng(1181)
    x = rng.normal(size=(48, 4)) * [1, 3, .7, 2]
    a = rng.uniform(0, 4, len(x))
    r = 3 * np.exp(-((a - 2 - .6 * np.tanh(x[:, 0] + x[:, 1])) / .7)**2)
    return x, a, r


def fit(x, a, r, **kw):
    args = dict(method="pseudo_direct", ndr=2,
                B_initial=np.array([[1., 0], [.3, .2], [0, 1], [.2, .4]]),
                maxitr=0, keep_data=True, lambda_init=[.1, .3])
    args.update(kw)
    return orthoDr_pdose(x, a, r, **args)


@pytest.mark.parametrize("intercept", [False, True])
def test_metadata_prediction_and_serialization(intercept):
    x, a, r = data()
    f = fit(x, a, r, intercept=intercept)
    assert f.itr == 0
    assert f.converge == (f.gradient_norm <= 1e-6)
    assert_allclose(f.fn, f.reward_scale * np.sqrt(f.objective_scaled), atol=1e-14)
    assert_allclose(f.dose_bandwidth_scale, np.std(a, ddof=1) * f.bw / np.sqrt(2), atol=1e-15)
    assert_allclose(f.B.T @ f.B, np.eye(2), atol=1e-14)
    assert f.predict(x[:1]).shape == (1,)
    assert f.predict(np.empty((0, 4))).shape == (0,)
    assert_allclose(f.predict(x), np.concatenate([f.predict(x[:10]), f.predict(x[10:])]), atol=2e-12)
    restored = pickle.loads(pickle.dumps(f))
    assert_allclose(restored.predict(x), f.predict(x), atol=0, rtol=0)
    if intercept:
        del restored["intercept"]
        with pytest.raises(ValueError, match="intercept"):
            restored.predict(x[:1])
    small = fit(x, a, r, intercept=intercept, keep_data=False)
    assert not {"x", "a", "r", "kernel", "grid.targets", "grid.scores"}.intersection(small)
    assert np.isfinite(small.gradient_norm)


def test_intercept_follows_dose_origin_and_units_at_fixed_basis():
    x, a, r = data()
    f = fit(x, a, r, intercept=True)
    shifted = fit(x, a + 1000, r, intercept=True)
    scaled = fit(x, 7 * a, r, intercept=True)
    assert shifted.lambda_init == f.lambda_init == scaled.lambda_init
    assert_allclose(shifted.predict(x), 1000 + f.predict(x), atol=3e-11, rtol=0)
    assert_allclose(scaled.predict(x), 7 * f.predict(x), atol=3e-11, rtol=0)


def test_constant_grid_target_has_exact_zero_coefficients():
    x, a, r = data()
    a = a - a.min() + .1
    f = fit(x, a, r, intercept=True, K=1, lambda_init=[.3, .1])
    assert_allclose(f.W, 0, atol=0, rtol=0)
    assert f.intercept == .1 and f.lambda_init == .3
    assert_allclose(f.predict(x), .1, atol=0, rtol=0)


def test_pseudo_validation_precedes_native_fitting():
    x, a, r = data()
    for opts in (dict(lambda_reward=.1), dict(intercept_bounds=[0, 4]),
                 dict(control={"gamma": -1}), dict(control={"gamma": 1.1}),
                 dict(control={"eta": 1}), dict(control={"gtol": 0}),
                 dict(maxitr=-1), dict(intercept=1)):
        with pytest.raises(ValueError):
            fit(x, a, r, **opts)
    for gamma in (0, 1):
        assert np.isfinite(fit(x, a, r, control={"gamma": gamma}).fn)
    with pytest.raises(ValueError, match="Constant rewards"):
        fit(x, a, np.full(len(a), .1), B_initial=None)


def test_positive_reward_affine_change_preserves_fixed_state():
    x, a, r = data()
    f = fit(x, a, r)
    for scale in (1e-8, 1e8):
        changed = fit(x, a, scale * (r + 2))
        assert_allclose(changed.objective_scaled, f.objective_scaled, atol=1e-14)
        assert_allclose(changed.gradient_norm, f.gradient_norm, atol=1e-13)
        assert_allclose(changed.predict(x), f.predict(x), atol=2e-12)
