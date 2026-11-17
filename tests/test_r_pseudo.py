"""Fixed-state R parity plus independent equations for revised pseudo-direct fits."""
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from orthodr import _core


REFERENCE = json.loads((Path(__file__).parent / "fixtures/r_pseudo.json").read_text())


def _arrays(value):
    if isinstance(value, dict):
        return {key: (np.atleast_1d(_arrays(item)) if key in {"R", "A", "a_seq", "lambda"}
                      else _arrays(item)) for key, item in value.items()}
    if isinstance(value, list):
        if any(isinstance(item, (dict, str)) for item in value):
            return [_arrays(item) for item in value]
        return np.asarray(value, dtype=float)
    return value


def _state(actual, expected, *, atol=3e-11, rtol=3e-10):
    if isinstance(expected, dict):
        for key, value in expected.items():
            assert key in actual, key
            _state(actual[key], value, atol=atol, rtol=rtol)
    elif isinstance(expected, list) and expected and isinstance(expected[0], dict):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _state(a, e, atol=atol, rtol=rtol)
    elif isinstance(expected, (str, bool)) or expected is None:
        assert actual == expected
    else:
        a, e = np.asarray(actual), np.asarray(expected, dtype=float)
        if a.shape != e.shape and a.size == e.size and (a.ndim <= 1 or e.ndim <= 1):
            a, e = a.reshape(-1), e.reshape(-1)
        assert_allclose(a, e, atol=atol, rtol=rtol)


def _center(values):
    # Independent two-pass centering around an anchor, before any squaring.
    extended = np.asarray(values, dtype=np.longdouble)
    location = extended[0] + np.mean(extended - extended[0])
    centered = extended - location
    scale = np.sqrt(np.sum(centered * centered) / (len(centered) - 1))
    return np.asarray(centered, dtype=float), float(location), float(scale)


def _equations(data, B=None, selected_lambda=None):
    """Pairwise ambient derivatives and explicit smoother matrices, not native code."""
    B = data["B"] if B is None else B
    x, a, reward, bw = data["X"], data["A"], data["R"], data["bw"]
    n, p = x.shape
    centered, _, scale = _center(reward)
    q = centered / scale
    dose_scale = _center(a)[2] * bw
    covariance = np.cov(x, rowvar=False, ddof=1)
    dx = x[:, None, :] - x[None, :, :]
    projected = x @ B
    projected_variance = np.var(projected, axis=0, ddof=1)
    delta = projected[:, None, :] - projected[None, :, :]
    logk = -np.sum(delta * delta / (2 * bw * bw * projected_variance), axis=2)
    loga = -(a[:, None] - a[None, :])**2 / (2 * dose_scale**2)
    logp = logk + loga
    probability = np.exp(logp - logp.max(axis=0))
    probability /= probability.sum(axis=0)
    fitted = q @ probability
    error = fitted - q
    objective = np.mean(error * error)
    sensitivity = probability * (q[:, None] - fitted[None, :]) * (2 * error[None, :] / n)
    gradient = np.zeros((p, B.shape[1]))
    for k in range(B.shape[1]):
        variance = projected_variance[k]
        pair_derivative = (-delta[:, :, k, None] * dx / (bw**2 * variance)
            + delta[:, :, k, None]**2 * (covariance @ B[:, k])[None, None, :]
              / (bw**2 * variance**2))
        gradient[:, k] = np.sum(sensitivity[:, :, None] * pair_derivative, axis=(0, 1))
    residual = gradient - B @ (gradient.T @ B)
    grid = np.atleast_1d(data["a_seq"])
    final_scale = dose_scale * data["dose_factor"]
    log_dose = -(a[:, None] - grid[None, :])**2 / (2 * final_scale**2)
    scores = np.empty((n, len(grid)))
    for i in range(n):
        joint = logk[:, i, None] + log_dose
        weights = np.exp(joint - joint.max(axis=0))
        scores[i] = q @ weights / weights.sum(axis=0)
    targets = grid[np.argmax(scores, axis=1)]
    kernel = np.exp(logk)
    ridge = []
    for penalty in np.atleast_1d(data["lambda"]):
        C = kernel + penalty * np.eye(n)
        try:
            np.linalg.cholesky(C)
            inverse = np.linalg.solve(C, np.eye(n))
        except np.linalg.LinAlgError:
            ridge.append(None)
            continue
        if data["intercept"]:
            v = inverse @ np.ones(n)
            t = np.sum(v)
            # Order the products so representable extreme penalties do not
            # underflow v'v before its scaling is applied.
            H = np.eye(n) - penalty * inverse + np.outer(v, (penalty * v) / t)
            if np.all(targets == targets[0]):
                alpha, W, gcv = float(targets[0]), np.zeros(n), 0.0
            else:
                centered, mean, _ = _center(targets)
                u = np.linalg.solve(C, centered)
                alpha0 = np.sum(u) / t
                alpha, W = mean + alpha0, u - v * alpha0
                # The full smoother includes the unpenalized intercept.
                gcv = n * np.sum((targets - H @ targets)**2) / (n - np.trace(H))**2
        else:
            alpha = 0.0
            W = np.linalg.solve(C, targets)
            H = np.eye(n) - penalty * inverse
            gcv = n * np.sum((targets - H @ targets)**2) / (n - np.trace(H))**2
        ridge.append({"intercept": alpha, "W": W, "GCV": gcv, "lambda": penalty})
    gcv = np.array([np.nan if item is None else item["GCV"] for item in ridge])
    selected_index = (int(np.flatnonzero(np.atleast_1d(data["lambda"]) == selected_lambda)[0])
                      if selected_lambda is not None else int(np.nanargmin(gcv)))
    selected = ridge[selected_index]
    return {"objective.scaled": objective, "gradient": gradient, "residual": residual,
            "gradient.norm": np.linalg.norm(residual), "reward.scaled": q,
            "fitted.scaled": fitted, "kernel": kernel, "log.kernel": logk,
            "grid.scores": scores, "grid.targets": targets, "GCV": gcv,
            "lambda.init": selected["lambda"], "W": selected["W"],
            "intercept": selected["intercept"]}


@pytest.mark.parametrize("case", REFERENCE["cases"])
def test_pseudo_fixed_r_state_complete_evaluation(case):
    item = _arrays(REFERENCE["cases"][case])
    if len(item["warnings"]):
        with pytest.warns(RuntimeWarning, match="Skipped unusable"):
            actual = _core.pdose_semi_evaluate(**item["input"])
    else:
        actual = _core.pdose_semi_evaluate(**item["input"])
    _state(actual, item["expected"])


@pytest.mark.parametrize("case", REFERENCE["cases"])
def test_pseudo_independent_derivative_scores_ridge_and_cayley_equations(case):
    item = _arrays(REFERENCE["cases"][case])
    data, actual = item["input"], item["expected"]
    expected = _equations(data, selected_lambda=actual["lambda.init"])
    # Explicit smoother subtraction after a large dose shift loses digits;
    # compare that GCV separately, keeping coefficients and scores strict.
    gcv = expected.pop("GCV")
    _state(actual, expected, atol=2e-10, rtol=2e-9)
    assert_allclose(actual["GCV"].reshape(-1), gcv, atol=2e-8 if case == "dose_affine" else 2e-10,
                    rtol=2e-8, equal_nan=True)
    # With an identity kernel all ridge candidates have the same mathematical
    # GCV; last-bit rounding need not rank those candidates the same way.
    # Enforce the reported score minimizer exactly and assess its score against
    # the independent smoother, while comparing coefficients at that ridge.
    penalties = np.atleast_1d(data["lambda"])
    selected = int(np.nanargmin(actual["GCV"].reshape(-1)))
    assert actual["lambda.init"] == penalties[selected]
    assert gcv[selected] <= np.nanmin(gcv) + 2e-10 * (1 + np.nanmax(gcv))
    B, G = data["B"], expected["gradient"]
    slope = np.sum(G * expected["residual"])
    H = (G @ B.T - B @ G.T) / 2
    step = data["cayley_step"]
    trial = np.linalg.solve(np.eye(B.shape[0]) + step * H, B - step * H @ B)
    for branch in ("cayley.dense", "cayley.lowrank"):
        assert actual[branch]["valid"]
        assert_allclose(actual[branch]["slope"], slope, atol=2e-12, rtol=2e-10)
        assert_allclose(actual[branch]["B"], trial, atol=2e-12, rtol=2e-10)


@pytest.mark.parametrize("case", ["p3_d1", "p3_d2", "p6_d1", "p6_d2"])
def test_pseudo_every_coordinate_and_tangent_derivative(case):
    item = _arrays(REFERENCE["cases"][case])
    data, actual = item["input"], item["expected"]
    B = data["B"]
    for index in np.ndindex(B.shape):
        plus, minus = B.copy(), B.copy()
        plus[index] += 2e-6
        minus[index] -= 2e-6
        derivative = (_equations(data, plus)["objective.scaled"]
                      - _equations(data, minus)["objective.scaled"]) / 4e-6
        assert_allclose(derivative, actual["gradient"][index], atol=2e-8, rtol=2e-6)
    tangent = np.cos(np.arange(B.size)).reshape(B.shape)
    tangent -= B @ ((B.T @ tangent + tangent.T @ B) / 2)
    derivative = (_equations(data, B + 1e-6 * tangent)["objective.scaled"]
                  - _equations(data, B - 1e-6 * tangent)["objective.scaled"]) / 2e-6
    assert_allclose(derivative, np.sum(actual["gradient"] * tangent), atol=2e-8, rtol=2e-6)
    # A central difference along the actual Cayley curve checks its sign and slope.
    G = actual["gradient"]
    H = (G @ B.T - B @ G.T) / 2
    def curve(step):
        return np.linalg.solve(np.eye(B.shape[0]) + step * H, B - step * H @ B)
    derivative = (_equations(data, curve(1e-6))["objective.scaled"]
                  - _equations(data, curve(-1e-6))["objective.scaled"]) / 2e-6
    assert_allclose(derivative, -actual["cayley.dense"]["slope"], atol=2e-8, rtol=2e-6)


@pytest.mark.parametrize("steps", REFERENCE["trajectories"])
def test_pseudo_complete_accepted_state_audit_matches_r(steps):
    actual = _core.pdose_semi_audit_solver(**_arrays(REFERENCE["audit_input"]),
        **REFERENCE["controls"], maxitr=int(steps), verbose=0, ncore=1)
    _state(actual, REFERENCE["trajectories"][steps], atol=3e-9, rtol=3e-8)
    assert actual["converge"] == (actual["stop.reason"] == "gradient_tolerance")
    for entry in actual["trace"]:
        if entry["event"] == "commit":
            assert np.linalg.norm(entry["B"].T @ entry["B"] - np.eye(entry["B"].shape[1])) < 1e-10


def test_pseudo_constants_and_narrow_bandwidth_have_exact_contracts():
    constant = REFERENCE["cases"]["constant_target"]["expected"]
    assert constant["intercept"] == .1
    assert np.count_nonzero(constant["W"]) == 0
    assert np.count_nonzero(constant["GCV"]) == 0
    assert constant["lambda.init"] == .2
    narrow = REFERENCE["cases"]["bw_0.001"]["expected"]
    assert narrow["objective.scaled"] == 0
    assert narrow["gradient.norm"] == 0
    for case in ("bw_0.05", "bw_0.01", "bw_0.001"):
        assert np.isfinite(REFERENCE["cases"][case]["expected"]["grid.scores"]).all()
