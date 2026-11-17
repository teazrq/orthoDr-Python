"""Fresh multi-direction R checks, with no loose long-run trajectory tolerances."""
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.stats import rankdata

import orthodr
from orthodr import _core
from orthodr._utils import silverman
from orthodr.models import FitResult

FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE = json.loads((FIXTURES / "r_extended.json").read_text())
DIRECT = tuple(REFERENCE["direct_evaluations"])


def _arrays(value):
    if isinstance(value, dict):
        return {key: _arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    return value


def _numeric(actual, expected, *, atol=2e-10, rtol=2e-9):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (actual.shape != expected.shape and actual.size == expected.size
            and (actual.ndim <= 1 or expected.ndim <= 1)):
        actual, expected = actual.reshape(-1), expected.reshape(-1)
    assert_allclose(actual, expected, atol=atol, rtol=rtol)


def _state(actual, expected, *, atol=2e-10, rtol=2e-9):
    for key, value in expected.items():
        assert key in actual, key
        if isinstance(value, dict):
            _state(actual[key], value, atol=atol, rtol=rtol)
        elif isinstance(value, (str, bool)) or value is None:
            assert actual[key] == value, key
        else:
            _numeric(actual[key], value, atol=atol, rtol=rtol)


def _fixed_fit(item):
    fit = FitResult(item["expected"])
    fit["kind"] = {"regression": "reg", "survival": "surv", "pdose": "pdose"}[item["family"]]
    for key in ("x", "y", "a", "r", "censor"):
        if key in item["input"]:
            fit[key] = item["input"][key]
    return fit


def _evaluate_direct(item, B=None, W=None, alpha=None):
    data, fit = item["input"], item["expected"]
    return _core.pdose_direct_evaluate(
        fit["B"] if B is None else B, data["x"], data["a"], data["r"], fit["bw"],
        np.asarray(fit["W"] if W is None else W).reshape(-1), 0.1,
        intercept=data.get("intercept", False),
        intercept_bounds=data.get("intercept_bounds"),
        alpha=fit.get("intercept", 0.0) if alpha is None else alpha)


def test_extended_source_fingerprints_match_base_reference():
    base = json.loads((FIXTURES / "r_reference.json").read_text())["provenance"]
    assert REFERENCE["provenance"]["source_sha"] == base["source_sha"]
    assert REFERENCE["provenance"]["source_files_md5"] == base["source_files_md5"]
    assert REFERENCE["provenance"]["seed"] == 271828


@pytest.mark.parametrize("case", REFERENCE["cases"])
def test_extended_fixed_r_state_prediction(case):
    item = _arrays(REFERENCE["cases"][case])
    result = orthodr.predict(_fixed_fit(item), np.asarray(REFERENCE["testx"]))
    _state(result, item["prediction"], atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("case", DIRECT)
def test_extended_two_direction_direct_complete_fit(case):
    item = _arrays(REFERENCE["cases"][case])
    fit = orthodr.orthoDr_pdose(**item["input"], keep_data=True, ncore=1)
    expected = dict(item["expected"])
    expected.pop("B")
    _state(fit, expected, atol=3e-8, rtol=3e-7)
    # Arbitrary eigenvector signs in pSAVE preserve the direct rule.
    q, _ = np.linalg.qr(fit.B)
    qr, _ = np.linalg.qr(item["expected"]["B"])
    _numeric(q @ q.T, qr @ qr.T, atol=3e-8, rtol=0)
    _state(orthodr.predict(fit, np.asarray(REFERENCE["testx"])), item["prediction"],
           atol=3e-9, rtol=3e-8)


@pytest.mark.parametrize("case", DIRECT)
def test_extended_two_direction_direct_evaluation(case):
    item = _arrays(REFERENCE["cases"][case])
    _state(_evaluate_direct(item), _arrays(REFERENCE["direct_evaluations"][case]))


@pytest.mark.parametrize("case", DIRECT)
def test_extended_direct_gradient_in_every_basis_coordinate(case):
    item = _arrays(REFERENCE["cases"][case])
    B = item["initial"]
    state = _evaluate_direct(item, B=B)
    for index in np.ndindex(B.shape):
        plus, minus = B.copy(), B.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numeric = (_evaluate_direct(item, B=plus)["fn"] -
                   _evaluate_direct(item, B=minus)["fn"]) / 2e-6
        _numeric(numeric, state["gB"][index], atol=2e-8, rtol=2e-6)


@pytest.mark.parametrize("case", REFERENCE["native"])
def test_extended_exact_r_preprocessing_early_native_trajectory(case):
    item = _arrays(REFERENCE["native"][case])
    step = min(map(int, item["trajectory"]))
    if item["family"] == "regression":
        solver = getattr(_core, item["method"] + "_solver")
    elif item["family"] == "pdose":
        solver = _core.pdose_semi_solver
    else:
        solver = getattr(_core, "surv_" + item["method"] + "_solver")
    fit = solver(**item["prepared"], **item["controls"], maxitr=step, verbose=0, ncore=1)
    expected = item["trajectory"][str(step)]
    _numeric(fit["B"], expected["B"], atol=1e-8, rtol=1e-7)
    _numeric(fit["fn"], expected["fn"], atol=2e-10, rtol=2e-9)
    assert fit["itr"] == expected["itr"]
    assert fit["converge"] == expected["converge"]


@pytest.mark.parametrize("step", [step for step in REFERENCE["native"]["dose_pseudo_direct_2_default"]["trajectory"]
                                 if int(step) <= 20])
def test_extended_analytic_pseudo_native_complete_trajectory(step):
    item = _arrays(REFERENCE["native"]["dose_pseudo_direct_2_default"])
    fit = _core.pdose_semi_solver(**item["prepared"], **item["controls"],
                                  maxitr=int(step), verbose=0, ncore=1)
    _state(fit, item["trajectory"][step], atol=3e-8, rtol=3e-7)


def test_extended_long_pseudo_states_report_their_own_diagnostics():
    # Load the independent oracle by path so pytest's importlib mode also works
    # when exercising the shared installation from outside the source tree.
    import importlib.util
    spec = importlib.util.spec_from_file_location("pseudo_equation_oracle",
        Path(__file__).with_name("test_r_pseudo.py"))
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    _equations = oracle._equations

    item = _arrays(REFERENCE["cases"]["dose_pseudo_direct_2_default"])
    fit = orthodr.orthoDr_pdose(**item["input"], keep_data=True, ncore=1)
    data = dict(_arrays(REFERENCE["native"]["dose_pseudo_direct_2_default"])["prepared"],
                intercept=False, dose_factor=1 / np.sqrt(2), cayley_step=0.0)
    # Long nonstationary spectral trajectories can amplify rounding even with
    # analytic gradients. Check each terminal state against the independent
    # equations rather than assert a loose cross-language endpoint tolerance.
    for state in (fit, item["expected"]):
        data["B"] = state["B"]
        equations = _equations(data, selected_lambda=state["lambda.init"])
        _numeric(state["objective.scaled"], equations["objective.scaled"], atol=2e-11, rtol=2e-10)
        _numeric(state["gradient.norm"], equations["gradient.norm"], atol=2e-11, rtol=2e-10)
        _numeric(state["fn"], np.std(data["R"], ddof=1) * np.sqrt(equations["objective.scaled"]),
                 atol=2e-11, rtol=2e-10)
        _numeric(state["W"], equations["W"], atol=2e-10, rtol=2e-9)
        evaluated = _core.pdose_semi_evaluate(**data)
        _numeric(evaluated["gradient"], equations["gradient"], atol=2e-11, rtol=2e-10)
        assert state["converge"] == (state["stop.reason"] == "gradient_tolerance")


def _final_objective(item):
    data, fit = item["input"], item["expected"]
    B, x, bw = fit["B"], data["x"], fit["bw"]
    n = len(x)
    if data["method"] == "direct":
        return _evaluate_direct(item)["fn"]
    if item["family"] == "regression":
        y = data["y"]
        Y = ((y-y.mean()) / y.std(ddof=1) / n**(-1/(data["ndr"]+5)) / np.sqrt(2))[:, None]
        fn = getattr(_core, "local_f" if data["method"] == "local" else data["method"] + "_init")
        return fn(B, x, Y, bw, 1)
    if item["family"] == "pdose":
        z = np.column_stack((x @ B, data["a"]))
        z /= z.std(axis=0, ddof=1) * bw * np.sqrt(2)
        K = np.exp(-np.sum((z[:, None, :] - z[None, :, :])**2, axis=2))
        residual = data["r"] - K @ data["r"] / K.sum(axis=1)
        return np.sqrt(np.mean(residual**2))
    sx = x.std(axis=0, ddof=1)
    X = (x - x.mean(axis=0)) / sx
    ranks = rankdata(data["y"]) / (n+1)
    Y = (ranks-ranks.mean()) / ranks.std(ddof=1) / np.sqrt(2) / silverman(1, n)
    order = np.argsort(Y, kind="stable")
    X, Y, C = X[order], Y[order], data["censor"][order]
    events = np.flatnonzero(C == 1)
    starts = np.searchsorted(Y, Y[events])
    B = B * sx[:, None]
    B /= np.linalg.norm(B, axis=0)
    fn = getattr(_core, "surv_" + data["method"] + "_f")
    if data["method"] == "forward":
        return fn(B, X, events+1, starts+1, bw, 1)
    ky = np.exp(-np.abs(Y[:, None]-Y[None, :]))
    ky[Y[:, None] < Y[None, :]] = 0
    phi = np.empty((x.shape[1], len(events)))
    for j, event in enumerate(events):
        weight = C * ky[:, event]
        phi[:, j] = weight @ X / weight.sum() - X[starts[j]:].mean(axis=0)
    return fn(B, X, phi, events+1, starts+1, bw, 1)


@pytest.mark.parametrize("case", [name for name, item in REFERENCE["cases"].items()
                                 if item["input"]["method"] != "seff"])
def test_extended_returned_objective_at_fixed_r_state(case):
    item = _arrays(REFERENCE["cases"][case])
    _numeric(_final_objective(item), item["expected"]["fn"], atol=2e-11, rtol=2e-10)
