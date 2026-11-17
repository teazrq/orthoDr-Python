"""Numerical checks against the committed R source, not Python-generated truth.

Regenerate fixtures with tools/generate_r_fixtures.R. Short solver runs check
implementation parity; they do not establish statistical accuracy or convergence.
"""
from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
from numpy.testing import assert_allclose
import pytest

import orthodr
from orthodr import _core


REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "r_reference.json").read_text())


def _arrays(value):
    if isinstance(value, dict):
        return {key: _arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    return value


def _numeric(actual, expected, *, atol=2e-8, rtol=2e-7, label=""):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    # Rcpp serializes column and row vectors as matrices. The Python API uses
    # vectors for vector-valued outputs, with the same coordinate ordering.
    if actual.shape != expected.shape and actual.size == expected.size:
        if actual.ndim <= 1 or expected.ndim <= 1:
            actual, expected = actual.reshape(-1), expected.reshape(-1)
    assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=label)


def _state(actual, expected, *, atol=2e-8, rtol=2e-7, label=""):
    for key, value in expected.items():
        assert key in actual, f"{label}: missing returned field {key!r}"
        result = actual[key]
        where = f"{label}.{key}"
        if isinstance(value, dict):
            _state(result, value, atol=atol, rtol=rtol, label=where)
        elif isinstance(value, (str, bool)) or value is None:
            assert result == value, where
        else:
            _numeric(result, value, atol=atol, rtol=rtol, label=where)


def _projector(B):
    Q, _ = np.linalg.qr(np.asarray(B, dtype=float))
    return Q @ Q.T


def _subspace(actual, expected, *, atol=2e-8):
    _numeric(_projector(actual), _projector(expected), atol=atol, rtol=0,
             label="basis projection matrices")


def _fit(section, case):
    item = REFERENCE[section][case]
    kwargs = _arrays(item["input"])
    if "lambda" in kwargs:
        kwargs["lambda_init"] = kwargs.pop("lambda")
    kwargs["keep_data"] = True
    function = {"regression": orthodr.orthoDr_reg,
                "survival": orthodr.orthoDr_surv,
                "pdose": orthodr.orthoDr_pdose}[section]
    return function(**kwargs), item


def _check_prediction_and_pickle(fit, item, label, *, atol=5e-6, rtol=3e-6):
    testx = np.asarray(item["testx"], dtype=float)
    pred = orthodr.predict(fit, testx)
    _state(pred, item["prediction"], atol=atol, rtol=rtol, label=label + ".prediction")
    restored = pickle.loads(pickle.dumps(fit))
    _state(orthodr.predict(restored, testx), pred, atol=0, rtol=0,
           label=label + ".pickle")
    _numeric(restored.transform(testx), testx @ fit.B, atol=0, rtol=0)


def test_fixture_provenance():
    assert REFERENCE["provenance"]["source_sha"] == "a9588d8dd73c39188d1c46f371763addb10bee9a"
    assert _core.source_revision == REFERENCE["provenance"]["source_sha"]
    assert REFERENCE["provenance"]["seed"] == 20261116


@pytest.mark.parametrize("case", REFERENCE["regression"])
def test_regression_last_state_and_predictions(case):
    fit, item = _fit("regression", case)
    # Shared numerical differentiation can accumulate cross-BLAS rounding;
    # this still constrains both coordinates and objective after seven steps.
    _state(fit, item["expected"], atol=2e-6, rtol=2e-5, label=case)
    _subspace(fit.B, item["expected"]["B"], atol=3e-6)
    _check_prediction_and_pickle(fit, item, case)


@pytest.mark.parametrize("case", REFERENCE["regression"])
def test_regression_initial_objective(case):
    from orthodr._utils import gram_schmidt

    item = REFERENCE["regression"][case]
    data = _arrays(item["input"])
    function = getattr(_core, "local_f" if data["method"] == "local" else data["method"] + "_init")
    value = function(gram_schmidt(data["B_initial"]), data["x"],
                     np.asarray(item["scaled_y"]), data["bw"], 1)
    _numeric(value, item["initial_objective"], atol=2e-11, rtol=2e-10, label=case)


@pytest.mark.parametrize("case", REFERENCE["survival"])
def test_survival_last_state_ties_single_event_and_p1(case):
    fit, item = _fit("survival", case)
    if case == "dm_ties":
        # The saved R perturbation study shows 1e-13 changes in B0 can move
        # this seven-step result by 1.46e-3 in its projector and 1.15e-3 in
        # prediction. Bound that outcome variation, while separately checking
        # native first-step agreement and fixed-state prediction to roundoff.
        expected = dict(item["expected"])
        expected.pop("B")
        _state(fit, expected, atol=2e-7, rtol=3e-5, label=case)
        _numeric(fit.B, item["expected"]["B"], atol=2e-3, rtol=0, label=case + ".B")
        _subspace(fit.B, item["expected"]["B"], atol=2e-3)
        _check_prediction_and_pickle(fit, item, case, atol=1.5e-3, rtol=0)
    else:
        _state(fit, item["expected"], atol=3e-6, rtol=3e-5, label=case)
        _subspace(fit.B, item["expected"]["B"], atol=4e-6)
        _check_prediction_and_pickle(fit, item, case)
    survival = orthodr.predict(fit, np.asarray(item["testx"]))["surv"]
    assert np.all((survival >= -1e-14) & (survival <= 1 + 1e-14))
    assert np.all(np.diff(survival, axis=0) <= 1e-14)


@pytest.mark.parametrize("case", REFERENCE["pdose"])
def test_dose_last_state_and_predictions(case):
    fit, item = _fit("pdose", case)
    # Both dose solvers use analytic derivatives and a deterministic serial search.
    pseudo = item["input"]["method"] == "pseudo_direct"
    _state(fit, item["expected"], atol=3e-8, rtol=3e-7, label=case)
    _check_prediction_and_pickle(fit, item, case,
                                 atol=3e-9, rtol=3e-8)
    if not pseudo:
        _numeric(fit.fn, -fit.value + fit.penalty, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("case", REFERENCE["direct_evaluations"])
def test_direct_objective_gradient_hessian_and_cayley(case):
    item = REFERENCE["direct_evaluations"][case]
    result = _core.pdose_direct_evaluate(**_arrays(item["input"]))
    _state(result, item["expected"], atol=2e-10, rtol=2e-9, label=case)


@pytest.mark.parametrize("case", REFERENCE["direct_evaluations"])
def test_direct_analytic_gradient_with_central_differences(case):
    data = _arrays(REFERENCE["direct_evaluations"][case]["input"])
    result = _core.pdose_direct_evaluate(**data)
    step = 1e-6
    for field, gradient, indices in (
        ("B", "gB", [(i, 0) for i in range(len(data["B"]))]),
        ("W", "gW", [0, len(data["W"]) // 2, len(data["W"]) - 1]),
    ):
        for index in indices:
            plus, minus = dict(data), dict(data)
            plus[field], minus[field] = data[field].copy(), data[field].copy()
            plus[field][index] += step
            minus[field][index] -= step
            derivative = (_core.pdose_direct_evaluate(**plus)["fn"]
                          - _core.pdose_direct_evaluate(**minus)["fn"]) / (2 * step)
            _numeric(derivative, np.asarray(result[gradient])[index],
                     atol=2e-8, rtol=2e-6, label=f"{case}.{gradient}[{index}]")
    if data["intercept"]:
        plus, minus = dict(data, alpha=data["alpha"] + step), dict(data, alpha=data["alpha"] - step)
        derivative = (_core.pdose_direct_evaluate(**plus)["fn"]
                      - _core.pdose_direct_evaluate(**minus)["fn"]) / (2 * step)
        _numeric(derivative, result["g.intercept"], atol=2e-8, rtol=2e-6)


@pytest.mark.parametrize("method", REFERENCE["initializers"]["dr"])
def test_dr_sir_save_and_residual_phd_initializers(method):
    from orthodr.initializers import dr_fit

    item = REFERENCE["initializers"]["dr"][method]
    fit = dr_fit(np.asarray(item["x"]), np.asarray(item["y"]), method=method)
    expected = item["expected"]
    _numeric(fit["M"], expected["M"], atol=2e-11, rtol=2e-10)
    _numeric(fit["evalues"], expected["values"], atol=2e-11, rtol=2e-10)
    for d in range(1, len(expected["values"])):
        _subspace(fit["evectors"][:, :d], np.asarray(expected["vectors"])[:, :d], atol=2e-9)


@pytest.mark.parametrize("case", REFERENCE["initializers"]["reg_selected"])
def test_default_regression_initializer_selection(case):
    from orthodr._utils import gram_schmidt
    from orthodr.initializers import regression_initializers

    data = _arrays(REFERENCE["initializers"]["reg_selected"][case])
    bases = regression_initializers(data["x"], data["y"].reshape(-1))
    d = data["ndr"]
    candidates = [gram_schmidt(bases[m][:, :d]) for m in ("sir", "save", "phd")]
    if d == 2:
        mixed = np.column_stack((bases["sir"][:, 0], bases["save"][:, 0]))
        if np.linalg.matrix_rank(mixed) == 2:
            candidates.append(gram_schmidt(mixed))
    objective = getattr(_core, "local_f" if data["method"] == "local" else data["method"] + "_init")
    values = [objective(B, data["x"], data["y"], data["bw"], 1) for B in candidates]
    selected = candidates[np.argmin(values)]
    _subspace(selected, data["expected"], atol=2e-8)


@pytest.mark.parametrize("case", REFERENCE["initializers"]["pSAVE"])
def test_psave_subspace(case):
    data = _arrays(REFERENCE["initializers"]["pSAVE"][case])
    expected = data.pop("expected")
    result = orthodr.pSAVE(**data)
    # Sign conventions of real eigensolvers are not part of the estimator.
    _subspace(result, expected, atol=2e-8)


@pytest.mark.parametrize("case", REFERENCE["initializers"]["CP_SIR"])
def test_counting_process_sir_values_and_subspaces(case):
    data = _arrays(REFERENCE["initializers"]["CP_SIR"][case])
    expected = data.pop("expected")
    result = orthodr.CP_SIR(**data)
    _numeric(result["values"], expected["values"], atol=2e-10, rtol=2e-9)
    # Comparing the whole p-dimensional span would be vacuous. Check each
    # identifiable leading subspace; a one-event matrix has rank at most one.
    rank = np.count_nonzero(expected["values"] > np.max(expected["values"]) * 1e-10)
    for d in range(1, rank + 1):
        _subspace(result["vectors"][:, :d], expected["vectors"][:, :d], atol=2e-8)


@pytest.mark.parametrize("case", REFERENCE["initializers"]["hMave"])
def test_hmave_subspace_cv_and_vector_column_equivalence(case):
    data = _arrays(REFERENCE["initializers"]["hMave"][case])
    expected = data.pop("expected")
    result = orthodr.hMave(**data)
    _subspace(result["B"], expected["B"], atol=3e-8)
    _numeric(result["cv"], expected["cv"], atol=2e-8, rtol=2e-8)
    data["y"], data["censor"] = data["y"][:, None], data["censor"][:, None]
    column_result = orthodr.hMave(**data)
    _subspace(column_result["B"], result["B"], atol=1e-14)
    _numeric(column_result["cv"], result["cv"], atol=0, rtol=0)


@pytest.mark.parametrize("case", REFERENCE["default_fits"])
def test_complete_fit_with_default_initialization(case):
    item = REFERENCE["default_fits"][case]
    kwargs = _arrays(item["input"])
    if "lambda" in kwargs:
        kwargs["lambda_init"] = kwargs.pop("lambda")
    function = {"regression": orthodr.orthoDr_reg,
                "survival": orthodr.orthoDr_surv,
                "pdose": orthodr.orthoDr_pdose}[item["family"]]
    fit = function(**kwargs, keep_data=True)
    expected = dict(item["expected"])
    expected.pop("B")
    _state(fit, expected, atol=3e-6, rtol=3e-5, label=case)
    _subspace(fit.B, item["expected"]["B"], atol=3e-6)
    _check_prediction_and_pickle(fit, item, case)


@pytest.mark.parametrize("case", REFERENCE["optim"])
def test_general_optimizer_analytic_numeric_max_min_and_both_branches(case):
    data = _arrays(REFERENCE["optim"][case]["input"])
    A = data.pop("A")
    analytic = data.pop("analytic")
    sign = 1 if data["maximize"] else -1

    def fn(B):
        return sign * 0.5 * np.sum(B * (A @ B))

    def grad(B):
        return sign * (A @ B)

    result = orthodr.ortho_optim(**data, fn=fn, grad=grad if analytic else None)
    _state(result, REFERENCE["optim"][case]["expected"], atol=2e-6,
           rtol=2e-6, label=case)
    _numeric(result["B"].T @ result["B"], np.eye(result["B"].shape[1]), atol=2e-12, rtol=0)


@pytest.mark.parametrize("case", ["dm_ties", "dn_one_event"])
def test_native_survival_first_step_from_exact_r_preprocessing(case):
    item = REFERENCE["survival"][case]
    kwargs = {k: v for k, v in _arrays(item["prepared"]).items() if k != "Y"}
    kwargs["Fail_Ind"] = np.atleast_1d(kwargs["Fail_Ind"])
    kwargs["Risk_Start"] = np.atleast_1d(kwargs["Risk_Start"])
    solver = getattr(_core, "surv_" + item["input"]["method"] + "_solver")
    fit = solver(**kwargs, **item["input"]["control"], maxitr=1, verbose=0, ncore=1)
    _state(fit, item["native_trajectory"][0], atol=2e-11, rtol=2e-10, label=case)


@pytest.mark.parametrize("section,case", [
    (section, case) for section in ("regression", "survival", "pdose")
    for case in REFERENCE[section]
])
def test_prediction_arithmetic_from_fixed_r_state(section, case):
    from orthodr.models import FitResult

    item = REFERENCE[section][case]
    fit = FitResult(_arrays(item["expected"]))
    fit["kind"] = {"regression": "reg", "survival": "surv", "pdose": "pdose"}[section]
    for key in ("x", "y", "a", "r", "censor"):
        if key in item["input"]:
            fit[key] = np.asarray(item["input"][key], dtype=float)
    result = orthodr.predict(fit, np.asarray(item["testx"]))
    _state(result, item["prediction"], atol=2e-12, rtol=2e-12, label=case)


@pytest.mark.parametrize("case", REFERENCE["stationary_fits"])
def test_stationary_full_fit_objective_subspace_prediction_and_residual(case):
    item = REFERENCE["stationary_fits"][case]
    kwargs = _arrays(item["input"])
    function = orthodr.orthoDr_reg if item["family"] == "regression" else orthodr.orthoDr_surv
    fit = function(**kwargs, keep_data=True)
    assert fit.converge and item["expected"]["converge"]
    _state(fit, item["expected"], atol=2e-8, rtol=2e-7, label=case)
    _subspace(fit.B, item["expected"]["B"], atol=2e-8)
    _check_prediction_and_pickle(fit, item, case, atol=2e-8, rtol=2e-8)

    if item["family"] == "regression":
        x, y = kwargs["x"], kwargs["y"]
        Y = ((y - y.mean()) / y.std(ddof=1) / len(y) ** (-1 / 6) / np.sqrt(2))[:, None]
        objective = lambda B: _core.sir_init(B, x, Y, kwargs["bw"], 1)
        bases = [fit.B, np.asarray(item["expected"]["B"])]
    else:
        prep = _arrays(REFERENCE["survival"]["forward_ties"]["prepared"])
        objective = lambda B: _core.surv_forward_f(B, prep["X"],
            np.atleast_1d(prep["Fail_Ind"]), np.atleast_1d(prep["Risk_Start"]), kwargs["bw"], 1)
        scale = kwargs["x"].std(axis=0, ddof=1)[:, None]
        bases = [np.asarray(B) * scale for B in (fit.B, item["expected"]["B"])]
        bases = [B / np.linalg.norm(B, axis=0) for B in bases]
    for B in bases:
        _numeric(objective(B), item["expected"]["fn"], atol=2e-10, rtol=2e-8)
        gradient = np.zeros_like(B)
        for i in range(B.shape[0]):
            plus, minus = B.copy(), B.copy()
            plus[i, 0] += 1e-6
            minus[i, 0] -= 1e-6
            gradient[i, 0] = (objective(plus) - objective(minus)) / 2e-6
        residual = np.linalg.norm(gradient - B @ (gradient.T @ B))
        assert residual <= 2 * kwargs["control"]["gtol"], (case, residual)
