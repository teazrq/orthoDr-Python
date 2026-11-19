"""pSAVE identities checked using Cholesky local coordinates and a raw eigenproblem.

The oracle does not call pSAVE's moment or symmetric-root helpers. Its slice
partition uses the separately tested dependency adapter so ties are preserved.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import eig

import orthodr.initializers as initializers
from orthodr import orthoDr_pdose
from orthodr.initializers import dr_slices, pSAVE


def projector(b):
    q, _ = np.linalg.qr(b)
    return q @ q.T


def coordinate_oracle(x, a, r, d=1, h=2, *, transport_fallback=False):
    """Independent congruence formula and nonsymmetric raw eigenproblem."""
    n, p = x.shape
    first = max(1, int(np.floor(n / 2 - 50)))
    last = min(n, int(np.floor(n / 2 + 50)))
    covariance, raw, fallback = np.zeros((p, p)), np.zeros((p, p)), 0.
    retained = []
    for cut in np.sort(a)[first - 1:last]:
        groups = [np.flatnonzero((a <= cut) == g) for g in np.unique(a <= cut)]
        if not all(len(i) >= p + 1 and np.linalg.matrix_rank(x[i] - x[i].mean(0)) == p
                   for i in groups):
            continue
        retained.append(cut)
        for indices in groups:
            xx, yy = x[indices], r[indices]
            centered = xx - xx.mean(0)
            sigma = centered.T @ centered / len(indices)
            cholesky = np.linalg.cholesky(sigma)
            inverse = np.linalg.solve(cholesky, np.eye(p))
            slices = dr_slices(yy, h, repair=True)
            covariance += len(indices) / n * sigma
            for j, size in enumerate(slices["sizes"]):
                if size < 3:
                    fallback += size / n
                    if transport_fallback:
                        raw += size / n * inverse.T @ inverse
                    continue
                subset = xx[slices["indicator"] == j]
                subset = subset - subset.mean(0)
                within = subset.T @ subset / size
                contrast = np.eye(p) - inverse @ within @ inverse.T
                raw += size / n * inverse.T @ contrast @ contrast @ inverse
    covariance /= len(retained)
    raw /= len(retained)
    values, vectors = eig(raw @ covariance)
    assert_allclose(values.imag, 0., atol=1e-12)
    order = np.argsort(values.real)[::-1]
    vectors = vectors.real[:, order[:d]]
    vectors /= np.linalg.norm(vectors, axis=0)
    pivots = np.argmax(np.abs(vectors), axis=0)
    vectors *= np.where(vectors[pivots, np.arange(d)] < 0, -1., 1.)
    return dict(B=vectors, values=values.real[order], covariance=covariance,
                raw_moment=raw, fallback_mass=fallback / len(retained),
                thresholds=np.asarray(retained))


def heterogeneous_data(seed=119, n=181, p=4):
    rng = np.random.default_rng(seed)
    a = np.round(rng.uniform(0, 4, n), 1)
    x = rng.normal(size=(n, p))
    transform = np.diag(np.array([1., 3., .7, 2.])[:p])
    transform[0, 1] = .8
    high = a > 2
    x[high] = x[high] @ transform + np.arange(p)
    r = (x[:, 0] + .4 * x[:, 1]) ** 2 + .3 * x[:, 2] + rng.normal(size=n)
    return x, a, r


@pytest.mark.parametrize("scales", [[2.**27, 1., 1., 1.], 2.**np.array([200, -200, 0, -27])])
def test_psave_whitening_respects_extreme_predictor_units(scales):
    x, a, r = heterogeneous_data()
    scales = np.asarray(scales)
    expected = coordinate_oracle(x, a, r, d=2)
    scaled = pSAVE(x * scales, a, r, 2)
    assert_allclose(np.linalg.norm(scaled, axis=0), 1.)
    mapped = scaled * scales[:, None]
    mapped /= np.linalg.norm(mapped, axis=0)
    assert_allclose(projector(mapped), projector(expected["B"]), atol=3e-11, rtol=0)


@pytest.mark.parametrize("dose_kind", ["continuous", "tied", "constant"])
def test_corrected_construction_matches_cholesky_raw_oracle(dose_kind):
    x, a, r = heterogeneous_data()
    if dose_kind == "continuous":
        a = a + np.arange(len(a)) / 10000
    elif dose_kind == "constant":
        a = np.ones(len(a))
    expected = coordinate_oracle(x, a, r, 2)
    assert expected["values"][1] - expected["values"][2] > 1e-4
    actual = pSAVE(x, a, r, 2)
    assert_allclose(actual, expected["B"], atol=3e-11, rtol=3e-11)
    assert_allclose(np.linalg.norm(actual, axis=0), 1., atol=3e-15)
    assert np.all(actual[np.argmax(abs(actual), axis=0), np.arange(2)] > 0)


def test_raw_subspace_is_affine_equivariant_with_unequal_covariances():
    x, a, r = heterogeneous_data()
    change = np.array([[2., .3, 0., -.2], [0., .7, .4, 0.],
                       [.2, 0., 1.4, .1], [0., -.3, 0., 1.1]])
    actual = pSAVE(x @ change + np.array([5., -2., 1., 8.]), a, r, 2)
    expected = np.linalg.solve(change, pSAVE(x, a, r, 2))
    assert_allclose(projector(actual), projector(expected), atol=2e-11, rtol=0)


def test_equal_covariance_reduction_recovers_anisotropic_mixed_direction():
    sigma = np.array([[2., 1.2, .3], [1.2, 16., 1.], [.3, 1., .7]])
    values, vectors = np.linalg.eigh(sigma)
    root = (vectors * np.sqrt(values)) @ vectors.T
    beta = np.array([1., 1., 0.]) / np.sqrt(2)
    whitened_direction = root @ beta
    q, _ = np.linalg.qr(np.column_stack((whitened_direction, [0., 0., 1.], [1., 0., 0.])))
    blocks = []
    for variance in [.25, 1.75]:
        z = np.concatenate((np.diag(np.sqrt(3 * np.array([variance, 1., 1.]))),
                            -np.diag(np.sqrt(3 * np.array([variance, 1., 1.])))))
        blocks.append(z @ q.T @ root)
    x, r = np.vstack(blocks), np.repeat([0., 1.], 6)
    actual = pSAVE(x, np.ones(12), r, 1)
    assert_allclose(actual[:, 0], beta, atol=2e-14)
    local = initializers._psave_group_moment(x, r, 2)
    _, v = np.linalg.eigh(local["moment"])
    reduced = local["inverse_sqrt"] @ v[:, -1:]
    assert_allclose(projector(actual), projector(reduced), atol=2e-14)


def test_gaussian_anisotropic_correlated_direction():
    rng = np.random.default_rng(716)
    scales = np.array([1., 5., 2., .5])
    sigma = scales[:, None] * .6 ** np.abs(np.subtract.outer(np.arange(4), np.arange(4))) * scales
    x = rng.multivariate_normal(np.zeros(4), sigma, size=4000)
    beta = np.array([1., 1., 0., 0.]) / np.sqrt(2)
    r = (x @ beta / np.sqrt(beta @ sigma @ beta)) ** 2 + rng.normal(0, .02, len(x))
    actual = pSAVE(x, np.ones(len(x)), r, 1)
    assert np.linalg.norm(projector(actual) - np.outer(beta, beta)) < .12


def test_a_complete_tied_eigenspace_is_compared_as_a_subspace():
    sigma = np.array([[2., 1.2, .3], [1.2, 16., 1.], [.3, 1., .7]])
    values, vectors = np.linalg.eigh(sigma)
    root = (vectors * np.sqrt(values)) @ vectors.T
    blocks = []
    for variance in [.25, 1.75]:
        z = np.diag(np.sqrt(3 * np.array([variance, variance, 1.])))
        blocks.append(np.concatenate((z, -z)) @ root)
    x, r = np.vstack(blocks), np.repeat([0., 1.], 6)
    expected = np.linalg.solve(root, np.eye(3)[:, :2])
    actual = pSAVE(x, np.ones(12), r, 2)
    assert_allclose(projector(actual), projector(expected), atol=3e-14)
    # The individual columns in this tied pair are not identified.
    assert_allclose(projector(pSAVE(x[::-1], np.ones(12), r[::-1], 2)),
                    projector(expected), atol=3e-14)


def test_mixed_tiny_slices_supply_only_scalar_common_identity(monkeypatch):
    rng = np.random.default_rng(88)
    x = rng.normal(size=(300, 3))
    x[:250] = x[:250] @ np.array([[1., 3., 0.], [0., 6., 0.], [0., 0., .4]])
    x[250:] += [2., -1., 4.]
    a = np.repeat([0., 1.], [250, 50])
    r = np.concatenate((np.arange(250.), np.repeat([0., 1.], [20, 30])))
    expected = coordinate_oracle(x, a, r, h=300)
    assert_allclose(expected["fallback_mass"], 250 / 300)
    actual = pSAVE(x, a, r, 1, nslices0=300)
    assert_allclose(projector(actual), projector(expected["B"]), atol=2e-11, rtol=0)
    polluted = coordinate_oracle(x, a, r, h=300, transport_fallback=True)
    assert np.linalg.norm(projector(actual) - projector(polluted["B"])) > .2
    # Altering only this scalar cannot change any selected direction.
    original = initializers._grouped_save_moment

    def changed(*args):
        result = original(*args)
        result["fallback_mass"] += 100.
        return result

    monkeypatch.setattr(initializers, "_grouped_save_moment", changed)
    assert_allclose(pSAVE(x, a, r, 1, nslices0=300), actual, atol=0, rtol=0)


def test_threshold_multiset_odd_truncation_and_group_rank_gate(monkeypatch):
    x, a, r = heterogeneous_data(n=151)
    original = initializers._grouped_save_moment
    observed = []

    def record(xx, yy, group, h):
        observed.append(group.copy())
        return original(xx, yy, group, h)

    monkeypatch.setattr(initializers, "_grouped_save_moment", record)
    pSAVE(x, a, r, 1)
    cuts = np.sort(a)[24:125]
    assert len(observed) == 101
    for actual, cut in zip(observed, cuts):
        assert np.array_equal(actual, a <= cut)
    assert len(np.unique(cuts)) < len(cuts)
    observed.clear()
    x = np.column_stack((np.arange(1, 13), np.r_[2 * np.arange(1, 7), np.sin(np.arange(7, 13))]))
    pSAVE(x, np.arange(1, 13), np.arange(1, 13), 1)
    assert [np.sum(group) for group in observed] == [7, 8, 9, 12]


def test_invalid_inputs_preserve_psave_failures():
    x, a, r = heterogeneous_data(n=45)
    for d in [0, 5, 1.5, np.inf, True]:
        with pytest.raises(ValueError, match="ndr"):
            pSAVE(x, a, r, d)
    for h in [0, -1, 1.5, np.inf, True]:
        with pytest.raises(ValueError, match="nslices0"):
            pSAVE(x, a, r, 1, h)
    with pytest.raises(ValueError, match="rank"):
        pSAVE(np.column_stack((x[:, :3], x[:, 0] + x[:, 1])), a, r, 1)
    with pytest.raises(ValueError, match="finite"):
        pSAVE(x, np.full(len(a), np.nan), r, 1)


def test_psave_recognizes_global_constant_predictors_and_scalar_variation():
    t = np.arange(40)
    with pytest.raises(ValueError, match="full centred predictor rank"):
        pSAVE(np.column_stack((t, np.full(len(t), .1))), np.cos(t), np.sin(t) + t / 40, 1)
    with pytest.raises(ValueError, match="full centred predictor rank"):
        pSAVE(np.full((40, 1), .1), t, t, 1)
    tiny = np.nextafter(0., 1.)
    assert_allclose(pSAVE(np.array([[0.], [tiny], [0.]]), t[:3], t[:3], 1), [[1.]])


def test_psave_group_rank_and_covariance_are_translation_invariant(monkeypatch):
    a = np.arange(1, 13)
    x = np.column_stack((a, [0.] * 7 + [1., 0., 0., 1., 3.]))
    expected = coordinate_oracle(x, a, a)
    kept = []
    original = initializers._grouped_save_moment

    def record(xx, yy, group, h):
        kept.append(np.count_nonzero(group))
        return original(xx, yy, group, h)

    monkeypatch.setattr(initializers, "_grouped_save_moment", record)
    for shift in [0., .1, 2.**50]:
        moved = x + [0., shift]
        kept.clear()
        actual = pSAVE(moved, a, a, 1)
        assert kept == [8, 9, 12]
        assert_allclose(projector(actual), projector(expected["B"]), atol=2e-13, rtol=0)
        permuted = pSAVE(moved[::-1], a[::-1], a[::-1], 1)
        assert_allclose(projector(permuted), projector(expected["B"]), atol=2e-13, rtol=0)
        group_moment = initializers._psave_group_moment(moved[:8], a[:8], 2)
        assert_allclose(group_moment["covariance"][1, 1], 7 / 64, atol=1e-15, rtol=0)


@pytest.mark.parametrize("method", ["direct", "pseudo_direct"])
def test_psave_dose_policies_match_shared_start_after_predictor_translation(method):
    t = np.arange(32)
    u = np.sin(t * 1.234)
    v = np.where(t < 16, 0., np.cos(t * 3.339))
    x, a = np.column_stack((u, v)), t / 32
    reward = .2 * u - 3 * (a - .45 - .3 * np.sin(2 * u))**2
    query = np.column_stack((np.sin(t * .91), np.cos(t * .51)))
    optimal = .45 + .3 * np.sin(2 * query[:, 0])
    fixed = pSAVE(x, a, reward, 1)
    options = dict(ndr=1, method=method, intercept=True, keep_data=True)
    control = orthoDr_pdose(x, a, reward, B_initial=fixed, **options).predict(query)
    assert np.sqrt(np.mean((control - optimal)**2)) < .09
    for shift in [0., .1]:
        moved = np.array([0., shift])
        fit = orthoDr_pdose(x + moved, a, reward, **options)
        assert_allclose(fit.predict(query + moved), control, atol=5e-7, rtol=0)
