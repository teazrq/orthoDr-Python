import numpy as np
import pytest
from numpy.testing import assert_allclose

from orthodr._utils import (control_check, dist_cross, distance, gram_schmidt,
                            kernel_weight, numeric_matrix, numeric_vector, silverman)
from orthodr.initializers import (CP_SIR, _grouped_save_moment, dr_fit,
                                 dr_slices, hMave, pSAVE)


def test_gaussian_weights_and_silverman():
    a = np.array([[0., 0.], [1., 2.]])
    b = np.array([[0., 1.], [1., 0.], [0., 0.]])
    expected = np.exp(-np.array([[1., 1., 0.], [2., 4., 5.]]))
    assert_allclose(dist_cross(a, b), expected)
    assert_allclose(kernel_weight(a, b), expected)
    assert_allclose(silverman(1, 100), (4 / 3 / 100) ** .2)


def test_subspace_metrics_have_the_documented_scale():
    a = np.array([[1.], [0.], [0.]])
    theta = .4
    b = np.array([[np.cos(theta)], [np.sin(theta)], [0.]])
    assert_allclose(distance(a, b), np.sqrt(2) * np.sin(theta))
    assert_allclose(distance(a, b, "trace"), np.cos(theta) ** 2)
    assert_allclose(distance(a, b, "sine"), np.sin(theta) ** 2)
    assert_allclose(distance(a, a * -3), 0, atol=1e-15)
    assert_allclose(distance(a, b, "canonical", np.eye(3)), np.cos(theta))


def test_gram_schmidt_column_order_rank_and_input_preservation():
    a = np.array([[1., 1.], [2., 2. + 1e-5], [3., 3. - 1e-5]])
    original = a.copy()
    q = gram_schmidt(a)
    assert_allclose(q.T @ q, np.eye(2), atol=1e-14)
    assert_allclose(q[:, 0], a[:, 0] / np.linalg.norm(a[:, 0]))
    assert_allclose(a, original)
    with pytest.raises(ValueError, match="rank"):
        gram_schmidt(np.column_stack((a[:, 0], 2 * a[:, 0])))
    with pytest.raises(ValueError):
        gram_schmidt(np.zeros((2, 3)))


def test_validation_and_control_defaults():
    with pytest.raises(ValueError, match="matrix"):
        numeric_matrix([1., 2.])
    with pytest.raises(ValueError, match="finite"):
        numeric_vector([1., np.nan])
    with pytest.raises(ValueError, match="vector"):
        numeric_vector([True, False])
    assert_allclose(numeric_vector([[1.], [2.]], allow_column=True), [1, 2])
    c = control_check({"rho": -2, "eta": np.inf, "tau": .02})
    assert c["rho"] == 1e-4 and c["eta"] == .2 and c["tau"] == .02


def test_slice_boundaries_preserve_ties_and_odd_size_truncation():
    y = np.array([8., 1., 3., 3., 2., 5., 6., 4., 7.])
    slices = dr_slices(y, 2)
    assert_allclose(slices["sizes"], [4, 5])
    assert_allclose(slices["indicator"], [1, 0, 0, 0, 0, 1, 1, 1, 1])
    # The first proposed breakpoint covers all but two observations. The
    # committed pSAVE repair returns one slice instead of an invalid partition.
    repaired = dr_slices(np.array([0.] * 8 + [1., 2.]), 2, repair=True)
    assert repaired["nslices"] == 1
    assert_allclose(repaired["sizes"], [10])


def test_regression_moments_use_population_qr_scaling_and_residual_phd():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(80, 3)) @ np.diag([1., 2., 3.])
    y = x[:, 0] ** 2 + .3 * x[:, 1] + rng.normal(size=80)
    fit = dr_fit(x, y, "phd")
    q, _ = np.linalg.qr(x - x.mean(axis=0))
    residual = y - y.mean() - q @ (q.T @ (y - y.mean()))
    assert_allclose(fit["M"], (q.T * residual) @ q, atol=1e-13)
    vals = np.linalg.eigvalsh(fit["M"])
    assert_allclose(fit["evalues"], vals[np.argsort(abs(vals))[::-1]])
    # Residual PHD is unchanged by adding a linear function of the predictors.
    second = dr_fit(x, y + 4 * x[:, 0] - 3 * x[:, 1], "phd")
    assert_allclose(second["M"], fit["M"], atol=1e-13)


def test_cpsir_tied_events_are_permutation_equivariant_and_normalized():
    rng = np.random.default_rng(32)
    x = rng.normal(size=(30, 3))
    y = np.repeat(np.arange(1., 11.), 3)
    censor = np.tile([1, 1, 0], 10)
    fit = CP_SIR(x, y, censor, bw=.4)
    permutation = rng.permutation(len(y))
    other = CP_SIR(x[permutation], y[permutation], censor[permutation], bw=.4)
    assert_allclose(fit["values"], other["values"], rtol=1e-12, atol=1e-12)
    assert_allclose(abs(np.sum(fit["vectors"] * other["vectors"], axis=0)), 1, atol=1e-12)
    assert_allclose(np.linalg.norm(fit["vectors"], axis=0), 1)
    with pytest.raises(ValueError, match="event"):
        CP_SIR(x, y, np.zeros(30))
    with pytest.raises(ValueError, match="positive definite"):
        CP_SIR(np.column_stack((x[:, 0], x[:, 0])), y, censor)


def test_cpsir_recovers_failure_direction_with_covariate_dependent_censoring():
    # The staggered product grid makes the censoring coordinate's risk-set
    # mean equal its instantaneous failure mean at every event time.
    grid = np.arange(16.) / 16
    failure = np.repeat(grid, grid.size)
    censoring = np.tile(grid + .5 / grid.size, grid.size)
    mixing = np.array([[1., .6], [-.3, 2.]])
    x = np.column_stack((failure, censoring)) @ mixing
    expected = np.linalg.solve(mixing, [1., 0.])
    expected /= np.linalg.norm(expected)

    # With a single window containing all failures, the smoothed mean on the
    # right mixes the failure contrast with drift in the censoring coordinate.
    fit = CP_SIR(x, np.minimum(failure, censoring), failure < censoring, bw=2.)
    assert_allclose(abs(fit["vectors"][:, 0] @ expected), 1., atol=1e-12, rtol=0)
    assert_allclose(fit["values"][1], 0., atol=1e-12)


def test_psave_small_slices_retain_identity_and_p1_is_exact():
    x = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    moment = _grouped_save_moment(x, np.arange(4.), np.ones(4), 4)
    assert_allclose(moment["raw_moment"], np.zeros((2, 2)))
    assert moment["fallback_mass"] == 1.0
    assert_allclose(pSAVE(np.arange(8.)[:, None], np.zeros(8), np.ones(8), 1), [[1.]])
    with pytest.raises(ValueError, match="rank"):
        pSAVE(np.ones((8, 1)), np.zeros(8), np.ones(8), 1)


def test_psave_information_admissibility_and_rotations():
    rng = np.random.default_rng(311)
    x = rng.normal(size=(45, 3))
    a, r = rng.normal(size=45), rng.normal(size=45)
    b = pSAVE(x, a, r, 1)
    rotation = gram_schmidt(rng.normal(size=(3, 3)))
    rotated = pSAVE(x @ rotation + 5, a, r, 1)
    assert_allclose(distance(b, rotation @ rotated), 0, atol=2e-12)
    with pytest.raises(ValueError, match="outcome-slice contrasts"):
        pSAVE(x, a, np.ones(45), 1)
    with pytest.raises(ValueError, match="directional slice information"):
        pSAVE(x, a, r, 1, nslices0=45)
    with pytest.raises(ValueError, match="admissible"):
        pSAVE(rng.normal(size=(130, 110)), rng.normal(size=130), rng.normal(size=130), 1)


def test_hmave_vector_column_b0_and_single_predictor_inputs():
    rng = np.random.default_rng(61)
    x = rng.normal(size=(13, 2))
    y = np.exp(.4 * x[:, 0] + rng.normal(size=13))
    censor = np.ones(13, dtype=bool)
    fit = hMave(x, y, censor, 1)
    col = hMave(x, y[:, None], censor[:, None], 1)
    assert fit["B"].shape == (2, 1) and np.isfinite(fit["cv"])
    assert_allclose(fit["B"], col["B"], atol=1e-13)
    assert_allclose(fit["cv"], col["cv"], atol=1e-12)
    initialized = hMave(x, y, censor, 1, B0=np.array([[1.], [0.]]))
    assert initialized["B"].shape == (2, 1) and np.isfinite(initialized["cv"])
    single = hMave(x[:, :1], y, censor, 1)
    assert single["B"].shape == (1, 1)
    assert_allclose(abs(single["B"]), [[1.]])
    with pytest.raises(ValueError, match="censor"):
        hMave(x, y, np.full(13, 2), 1)
    with pytest.raises(ValueError, match="one value"):
        hMave(x, y[:-1], censor, 1)
    with pytest.raises(ValueError, match="rank"):
        hMave(np.column_stack((x[:, 0], x[:, 0])), y, censor, 1)
