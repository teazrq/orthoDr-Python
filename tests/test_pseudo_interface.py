"""Pseudo-direct public behavior, independent of optimizer path matching."""
import pickle
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from orthodr import _core, orthoDr_pdose


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


@pytest.mark.parametrize("optimum, winner", [(.5, 2), (.5 - 1e-12, 2), (.5 + 1e-12, 3)])
def test_near_tie_dose_targets_and_predictions_across_patient_orders(optimum, winner):
    z = np.tile([-1., 0., 1.], 12)
    x = np.vstack(([2.**55, 2.**55], np.column_stack((z, np.zeros_like(z)))))
    a = np.r_[.5, np.repeat(np.linspace(0, 1, 12), 3)]
    r = -(a - np.r_[optimum, optimum + .4 * z])**2
    start = np.array([[1.], [-1.]]) / np.sqrt(2)
    grid = np.linspace(0, 1, 6)
    query = np.array([[-1., 0.], [0., 0.], [1., 0.]])
    rows = np.arange(len(x))
    orders = (rows, np.roll(rows, -1), rows[::-1], np.random.default_rng(20261118).permutation(rows))
    predictions, targets = [], []
    for order in orders:
        xx, aa, rr = x[order], a[order], r[order]
        f = fit(xx, aa, rr, B_initial=start, ndr=1, K=6, bw=.5, maxitr=10, lambda_init=[.1])
        state = _core.pdose_semi_evaluate(f.B, xx, rr, aa, .5, grid, np.array([.1]))
        assert_allclose(f.W, state["W"], atol=2e-12, rtol=0)
        assert f.lambda_init == state["lambda.init"]
        assert_allclose(f.B, start, atol=2e-15, rtol=0)
        assert_allclose(f.objective_scaled, 0.05039171448280783, atol=2e-15, rtol=0)
        predictions.append(f.predict(query))
        targets.append(state["grid.targets"].ravel()[np.argsort(order)])
    assert_array_equal(np.asarray(targets), np.broadcast_to(targets[0], (len(orders), len(x))))
    assert targets[0][0] == grid[winner]
    assert_allclose(predictions, np.broadcast_to(predictions[0], (len(orders), len(query))),
                    atol=2e-12, rtol=0)
    expected_center = 0.3973103528891346 if winner == 2 else 0.5957761499978212
    assert_allclose(predictions[0][1], expected_center, atol=2e-12, rtol=0)

    # A binary scaling of the units preserves the numerical tie or distinct winner.
    scaled = fit(x, 2 * a, 8 * r, B_initial=start, ndr=1, K=6, bw=.5,
                 maxitr=10, lambda_init=[.1])
    scaled_state = _core.pdose_semi_evaluate(scaled.B, x, 8 * r, 2 * a, .5,
                                           2 * grid, np.array([.1]))
    assert_array_equal(scaled_state["grid.targets"].ravel(), 2 * targets[0])
    assert_allclose(scaled.predict(query), 2 * predictions[0], atol=2e-12, rtol=0)
