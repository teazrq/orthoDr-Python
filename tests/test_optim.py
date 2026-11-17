import numpy as np
import pytest
from numpy.testing import assert_allclose

from orthodr._utils import distance, gram_schmidt
from orthodr.optim import ortho_optim


@pytest.mark.parametrize("shape", [(8, 2), (5, 2), (4, 3)])
@pytest.mark.parametrize("maximize", [False, True])
def test_quadratic_optimizer_reaches_analytic_pca_subspace(shape, maximize):
    p, d = shape
    eigenvalues = np.arange(1., p + 1) ** 2
    a = np.diag(eigenvalues)
    b = np.random.default_rng(121).normal(size=shape)
    fit = ortho_optim(b, lambda b, a: np.trace(b.T @ a @ b),
                     lambda b, a: 2 * a @ b, a, maximize=maximize)
    target = np.eye(p)[:, -d:] if maximize else np.eye(p)[:, :d]
    assert fit["converge"]
    assert_allclose(fit["B"].T @ fit["B"], np.eye(d), atol=2e-12)
    assert_allclose(distance(fit["B"], target), 0, atol=2e-5)
    assert_allclose(fit["fn"], np.sum(eigenvalues[-d:] if maximize else eigenvalues[:d]), atol=2e-8)
    assert len(fit["fn_Seq"]) == 500
    assert_allclose(fit["fn_Seq"][fit["itr"]:], 0)


def test_numerical_and_analytic_gradients_agree_on_simple_problem():
    b = np.array([[1.], [2.], [-.5]])
    a = np.diag([1., 4., 9.])
    fn = lambda b: float(-.5 * np.sum(b * (a @ b)))
    exact = ortho_optim(b, fn, lambda b: -a @ b)
    approx = ortho_optim(b, fn, control={"epsilon": 1e-7})
    assert exact["method"] == "true gradient"
    assert approx["method"] == "approx. gradient"
    assert_allclose(exact["fn"], approx["fn"], atol=1e-8)
    assert_allclose(distance(exact["B"], approx["B"]), 0, atol=3e-6)


def test_stationary_initial_point_and_constant_objective_complete_zero_updates():
    b = np.eye(4)[:, :2]
    fit = ortho_optim(b, lambda b: 5., lambda b: np.zeros_like(b), maxitr=12)
    assert fit["itr"] == 0 and fit["converge"] and fit["fn"] == 5
    assert_allclose(fit["B"], b)
    assert_allclose(fit["fn_Seq"], np.zeros(12))
    numeric = ortho_optim(b, lambda b: 5., maxitr=12)
    assert numeric["itr"] == 0 and numeric["converge"]


def test_iteration_exhaustion_and_argument_forwarding():
    b = np.array([[1.], [1.], [1.]])
    fit = ortho_optim(b, lambda b, *, scale: scale * np.sum(np.arange(3.)[:, None] * b),
                     lambda b, *, scale: scale * np.arange(3.)[:, None],
                     scale=2., maxitr=1, control={"gtol": 1e-20})
    assert fit["itr"] == 1 and not fit["converge"]
    assert fit["fn_Seq"].shape == (1,)


@pytest.mark.parametrize("value", [np.nan, np.inf, [1., 2.], True, 1 + 2j])
def test_objective_validation(value):
    with pytest.raises(ValueError, match="fn must return"):
        ortho_optim(np.eye(3)[:, :1], lambda b: value)


def test_rechecks_callbacks_after_initial_validation():
    count = 0

    def objective(b):
        nonlocal count
        count += 1
        return np.sum(b) if count < 3 else np.inf

    with pytest.raises(ValueError, match="finite number"):
        ortho_optim(np.array([[1.], [2.], [3.]]), objective, lambda b: np.ones_like(b))
    count = 0

    def gradient(b):
        nonlocal count
        count += 1
        return np.ones_like(b) if count < 3 else np.full_like(b, np.nan)

    with pytest.raises(ValueError, match="finite matrix"):
        ortho_optim(np.array([[1.], [2.], [3.]]), lambda b: np.sum(b), gradient)


def test_starting_basis_and_gradient_shapes_are_validated():
    with pytest.raises(ValueError, match="rank"):
        ortho_optim(np.ones((4, 2)), lambda b: 1.)
    with pytest.raises(ValueError, match="same dimension"):
        ortho_optim(np.eye(3)[:, :1], lambda b: 1., lambda b: np.zeros(3))
    with pytest.raises(ValueError, match="positive"):
        ortho_optim(np.eye(3)[:, :1], lambda b: 1., maxitr=0)
    b = np.arange(1., 7.).reshape(3, 2)
    expected = gram_schmidt(b)
    fit = ortho_optim(b, lambda b: 1., lambda b: np.zeros_like(b))
    assert_allclose(fit["B"], expected)
