import itertools

import numpy as np
import pytest

from orthodr import CP_SIR, distance, ortho_optim, orthoDr_pdose, orthoDr_reg, orthoDr_surv, pSAVE


def test_survival_rejects_redundant_predictors_in_any_order():
    i = np.arange(32)
    a, b = i % 7 - 3, (5 * i) % 11 - 5
    x = np.column_stack((a, b, a + b))
    t, e = (11 * i) % 37 + 1, i % 3 != 1
    for perm in itertools.permutations(range(3)):
        for fit in (CP_SIR, orthoDr_surv):
            with pytest.raises(ValueError, match="rank"):
                fit(x[:, perm], t, e)
    duplicate = np.column_stack((a, a))
    with pytest.raises(ValueError, match="projected"):
        orthoDr_surv(duplicate, t, e, ndr=1, B_initial=np.array([[1.], [-1.]]))
    assert np.isfinite(orthoDr_surv(duplicate, t, e, ndr=1, B_initial=np.eye(2, 1), maxitr=0).B).all()
    for fit in (CP_SIR, orthoDr_surv):
        with pytest.raises(ValueError, match="Constant predictor"):
            fit(np.column_stack((a, np.full(len(a), .1))), t, e)
    # An identifiable direction with a different predictor unit remains admissible.
    ref = CP_SIR(x[:, :2], t, e)
    scales = np.array([1., 1e-12])
    other = CP_SIR(x[:, :2] * scales, t, e)
    assert distance(ref["vectors"][:, 0], scales * other["vectors"][:, 0]) < 1e-11


def test_missing_observation_masks_are_rejected_before_conversion():
    x = np.arange(12.)[:, None]
    y = x[:, 0] + 1
    mask = np.arange(len(y)) == 3
    valid = orthoDr_reg(np.ma.array(x, mask=False), y, ndr=1, keep_data=True)
    for hidden in (0., 1.):
        missing = np.ma.array(y.copy(), mask=mask)
        missing.data[3] = hidden
        event = np.ma.array(y > 2, mask=mask)
        event.data[3] = bool(hidden)
        for fit in (orthoDr_surv, CP_SIR):
            with pytest.raises(ValueError, match=r"censor.*masked.*\(3\)"):
                fit(x, y, event)
        with pytest.raises(ValueError, match=r"y.*masked.*\(3\)"):
            orthoDr_reg(x, missing)
        with pytest.raises(ValueError, match=r"r.*masked.*\(3\)"):
            orthoDr_pdose(x, y, missing)
        with pytest.raises(ValueError, match=r"x.*masked.*\(3, 0\)"):
            orthoDr_reg(missing[:, None], y)
        with pytest.raises(ValueError, match=r"testx.*masked.*\(3, 0\)"):
            valid.predict(missing[:, None])


def test_rescaling_basis_columns_preserves_distances():
    q = np.eye(3)[:, :2]
    scaled = q * [1., 1e-25]
    assert distance(q, scaled) == 0
    assert distance(q, scaled, "trace") == 1


def test_optimizer_and_penalty_reject_masked_values():
    x = np.arange(12.)[:, None]
    missing = np.ma.array([.05, .1], mask=[False, True])
    with pytest.raises(ValueError, match=r"lambda_init.*masked.*\(1\)"):
        orthoDr_pdose(x, x.ravel(), np.sin(x.ravel()), ndr=1, lambda_init=missing)
    with pytest.raises(ValueError, match="fn result.*masked"):
        ortho_optim(np.eye(2, 1), lambda b: np.ma.masked)
    with pytest.raises(ValueError, match="grad result.*masked"):
        ortho_optim(np.eye(2, 1), lambda b: 0., lambda b: np.ma.masked_array(b, mask=True))


def test_direction_conversion_distinguishes_tiny_units_from_lost_coefficients():
    rng = np.random.default_rng(46)
    x = rng.normal(size=(64, 2))
    time, dose, reward = rng.random((3, len(x)))
    event = time < .7
    for fit in (lambda z: CP_SIR(z, time, event)["vectors"],
                lambda z: orthoDr_surv(z, time, event, ndr=2, maxitr=0,
                                       B_initial=[[1., 1.], [1., -1.]]).B,
                lambda z: pSAVE(z, dose, reward, ndr=2)):
        with pytest.raises(ValueError, match="Rescale x"):
            fit(x * [1e-180, 1e170])
        a, b = fit(x), fit(x * 1e-312)
        assert distance(a[:, 0], b[:, 0]) < 1e-8
