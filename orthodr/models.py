"""Dimension-reduction models and prediction on the original feature scale."""
from __future__ import annotations

import warnings
from collections.abc import Mapping
from math import frexp, ldexp
import numpy as np
from scipy.stats import rankdata

from ._prediction import gaussian_mean, prediction_log_weights, weighted_km
from ._utils import (_reject_mask, control_check, gram_schmidt, numeric_matrix,
                     numeric_vector, positive_integer, silverman, survival_coordinates,
                     raw_predictor_directions, survival_status, survival_times)
from .initializers import CP_SIR, pSAVE, regression_initializers


def _native():
    try:
        from . import _core
    except ImportError as exc:
        raise ImportError("Build the orthoDr native extension before fitting; see the installation instructions.") from exc
    return _core


def _scalar(value, name, *, lower=0, strict=False):
    _reject_mask(value, name)
    a = np.asarray(value)
    if a.ndim or a.dtype.kind not in "iuf" or not np.isfinite(a):
        raise ValueError(f"{name} must be one finite real number")
    value = float(a)
    if value < lower or (strict and value == lower):
        raise ValueError(f"{name} must be {'greater than' if strict else 'at least'} {lower}")
    return value


def _integer(value, name):
    result = _scalar(value, name)
    if result != np.floor(result) or result > np.iinfo(np.int32).max:
        raise ValueError(f"{name} must be a nonnegative integer within the integer range")
    return int(result)


def _iterations(value, minimum=0):
    value = _scalar(value, "maxitr")
    if value > np.iinfo(np.int32).max:
        raise ValueError("maxitr exceeds the integer range")
    return max(minimum, int(np.floor(value)))


def _bandwidth(bw, d, n):
    return silverman(d, n) if bw is None else _scalar(bw, "bw", strict=True)


def _basis(B, p, d):
    B = numeric_matrix(B, "B_initial")
    if B.shape != (p, d):
        raise ValueError(f"B_initial must have shape ({p}, {d})")
    return gram_schmidt(B)


def _control_args(control):
    return [control[k] for k in ("rho", "eta", "gamma", "tau", "epsilon", "btol", "ftol", "gtol")]


def _pseudo_controls(control):
    if control is None:
        control = {}
    if not isinstance(control, Mapping):
        raise ValueError("control must be a mapping of tuning parameters")
    defaults = dict(rho=1e-4, eta=0.2, gamma=0.85, tau=1e-3,
                    epsilon=1e-6, btol=1e-6, ftol=1e-6, gtol=1e-6)
    out = dict(control)
    for name, default in defaults.items():
        out[name] = _scalar(control.get(name, default), "control['" + name + "']",
                            strict=name != "gamma")
        if (name in ("rho", "eta") and out[name] >= 1) or (name == "gamma" and out[name] > 1):
            raise ValueError("Pseudo-direct rho and eta must be below one; gamma must be between zero and one")
    return out


def _check_keep_data(keep_data):
    if not isinstance(keep_data, (bool, np.bool_)):
        raise ValueError("keep_data must be True or False")


class FitResult(dict):
    """Fitted directions and numerical diagnostics, with NumPy prediction methods.

    Dictionary keys retain the R names; attributes also accept underscores in
    place of dots, for example ``fit.gradient_norm`` and ``fit.stop_reason``.
    Prediction requires ``keep_data=True`` at fitting. Pickle retains the fitted
    rule and its training preprocessing without a running R session.
    """

    def __getattr__(self, name):
        key = name if name in self else name.replace("_", ".")
        if key in self:
            return self[key]
        raise AttributeError(name)

    def transform(self, x):
        x = _test_matrix(self, x)
        return x @ self["B"]

    def predict(self, x):
        result = predict(self, x)
        return result["pred"] if "pred" in result else result

    def __repr__(self):
        return (f"FitResult(kind={self.get('kind')!r}, method={self.get('method')!r}, "
                f"directions={self['B'].shape[1]}, iterations={self.get('itr')}, "
                f"converged={bool(self.get('converge'))})")


def _result(native, kind, method, bw, x, keep_data, **training):
    result = FitResult(native)
    result.update(kind=kind, method=method, bw=bw)
    result["keep.data"] = bool(keep_data)
    result["B"] = np.asarray(result["B"], dtype=float)
    if keep_data:
        result["x"] = x.copy()
        result.update({k: np.asarray(v).copy() for k, v in training.items()})
    return result


def _reg_working_predictors(x, method):
    with np.errstate(over="ignore"):
        differences = x - x[0]
    shrink = 1
    if not np.isfinite(differences).all():
        # Halving first allows differences between opposite finite extremes.
        shrink = 2
        differences = x / shrink - x[0] / shrink
    peak = np.max(np.abs(differences))
    if peak == 0:
        raise ValueError(f"{method.upper()} requires predictor variation")
    unit = differences / peak
    centered = unit - unit.mean(axis=0)
    rms = np.sqrt(np.sum(centered**2) / ((len(x) - 1) * x.shape[1]))
    return centered / rms, (shrink, peak, rms)


def _sir_local_predictors(x, scale):
    # Preserve represented gaps for narrow kernels by removing only offsets
    # that subtract exactly and using a power-of-two scale before differencing.
    lower, upper = x.min(axis=0), x.max(axis=0)
    origin = np.where((lower > 0) & (upper / 2 <= lower), lower,
                      np.where((upper < 0) & (lower / 2 >= upper), upper, 0.))
    exponent = sum(frexp(part)[1] - 1 for part in scale)
    binary_scale = ldexp(1., max(-1074, min(1023, exponent)))
    shrink, peak, rms = scale
    return (x - origin) / binary_scale, 1 / ((peak / binary_scale) * shrink * rms)


def _sir_fitting_coordinates(x, centered, scale, candidates, bw):
    eps = np.finfo(float).eps
    peak = np.max(np.abs(centered), axis=0)
    if bw < np.sqrt(eps) or any(
        np.any((eps * peak) @ np.abs(B) > np.sqrt(eps) * bw * (centered @ B).std(axis=0, ddof=1))
        for B in candidates
    ):
        local, pooled_ratio = _sir_local_predictors(x, scale)
        return local, {"pooled_ratio": pooled_ratio}
    return centered, {}


def _save_objective_scale(scale, pooled_ratio):
    # A SAVE moment is s**2 A - s**4 D after division by the pooled SD s.
    # Preserve the ratio between the terms, with both weights bounded by one.
    shrink, peak, rms = scale
    large = peak >= 1 / (shrink * rms)
    if large:
        identity, covariance, power = (1 / peak / shrink / rms)**2, 1., 8
    else:
        identity, covariance, power = 1., (peak * (shrink * rms))**2, 4
    return {"identity_weight": identity,
            "covariance_weight": covariance * pooled_ratio**2,
            "moment_scale": pooled_ratio**4}, power


def _reg_original_objective(fn, scale, power):
    # Keep the binary exponent separate until the original-unit objective
    # has been formed, even when raising the scale to a power overflows.
    if fn == 0 or not np.isfinite(fn):
        return fn
    fraction, exponent = frexp(fn)
    for part in scale:
        mantissa, part_exponent = frexp(part)
        fraction *= mantissa**power
        exponent += power * part_exponent
    try:
        return ldexp(fraction, exponent)
    except OverflowError:
        return float("inf")


def orthoDr_reg(x, y, method="sir", ndr=2, B_initial=None, bw=None,
                keep_data=False, control=None, maxitr=500, verbose=0, ncore=1):
    """Fit regression directions by SIR, SAVE, PHD, LOCAL or efficient estimation.

    ``x`` has observations in rows. SIR, PHD, LOCAL, and SEFF center the
    predictors and use one pooled predictor scale for optimization. PHD uses
    centered outer products. SAVE adjusts the objective scale for numerical
    stopping while retaining its supplied-unit estimating equation. All
    methods return ``fn`` in the supplied predictor units. Apply any desired
    standardization consistently to training and new data.
    Set ``keep_data=True`` to predict mean responses. As in R, at least five
    iterations are allowed. One varying predictor defaults to the scalar basis
    ``[[1]]``. With multiple predictors, automatic initialization uses
    SIR/SAVE/PHD candidates; one-direction LOCAL selects among three fitted starts
    by in-sample kernel R-squared.
    """
    if method not in ("sir", "save", "phd", "local", "seff"):
        raise ValueError("method must be 'sir', 'save', 'phd', 'local', or 'seff'")
    _check_keep_data(keep_data)
    x = numeric_matrix(x, "x", min_rows=2)
    y = numeric_vector(y, "y", n=x.shape[0])
    d = min(positive_integer(ndr, "ndr"), x.shape[1])
    if x.shape[1] == 1 and np.all(x[:, 0] == x[0, 0]):
        raise ValueError("A one-predictor fit requires a varying predictor")
    if d > 4:
        warnings.warn("More than four directions is not recommended for kernel estimation.", UserWarning)
    sy = np.std(y, ddof=1)
    if np.all(y == y[0]) or sy <= 0:
        raise ValueError("y must have positive sample variance")
    Y = ((y - y.mean()) / sy / x.shape[0] ** (-1 / (d + 5)) / np.sqrt(2))[:, None]
    bw = _bandwidth(bw, d, x.shape[0])
    controls = control_check(control)
    args = _control_args(controls) + [_iterations(maxitr, 5), _integer(verbose, "verbose"), _integer(ncore, "ncore")]
    core = _native()
    solver = getattr(core, method + "_solver")
    centered, scale = _reg_working_predictors(x, method)
    X = centered
    solver_args = {}
    objective_power = 2 if method in ("local", "seff") else 4
    if method == "save":
        X, pooled_ratio = _sir_local_predictors(x, scale)
        solver_args, objective_power = _save_objective_scale(scale, pooled_ratio)
    if args[-2] > 0:
        print(f"{method.upper()} optimizer diagnostics use the predictor-scale-adjusted objective.", flush=True)
    if B_initial is None and x.shape[1] == 1:
        B_initial = np.ones((1, 1))
    if B_initial is None:
        bases = regression_initializers(centered, Y[:, 0])
        candidates = [_basis(bases[m][:, :d], x.shape[1], d) for m in ("sir", "save", "phd")]
        if method == "local" and d == 1:
            fits = [solver(B, X, Y, bw, *args) for B in candidates]
            def r_squared(fit):
                u = X @ fit["B"]
                u = u / (np.std(u, ddof=1) * bw * np.sqrt(2))
                K = np.exp(-((u[:, None, :] - u[None, :, :]) ** 2).sum(axis=2))
                fitted = K @ Y[:, 0] / K.sum(axis=0)
                return 1 - np.sum((Y[:, 0] - fitted) ** 2) / np.sum((Y[:, 0] - Y.mean()) ** 2)
            native = fits[int(np.argmax([r_squared(f) for f in fits]))]
        else:
            if d == 2:
                mixed = np.column_stack((bases["sir"][:, 0], bases["save"][:, 0]))
                first = mixed[:, 0] / np.linalg.norm(mixed[:, 0])
                residual = mixed[:, 1] - first * (first @ mixed[:, 1])
                # R's non-LAPACK QR uses a relative tolerance of 1e-7.
                if np.linalg.norm(residual) > 1e-7 * np.linalg.norm(mixed[:, 1]):
                    candidates.append(gram_schmidt(mixed))
            objective = getattr(core, "local_f" if method == "local" else method + "_init")
            if method == "sir":
                X, solver_args = _sir_fitting_coordinates(x, X, scale, candidates, bw)
            values = [objective(B, X, Y, bw, args[-1], **solver_args) for B in candidates]
            native = solver(candidates[int(np.argmin(values))], X, Y, bw, *args, **solver_args)
    else:
        B = _basis(B_initial, x.shape[1], d)
        if method == "sir":
            X, solver_args = _sir_fitting_coordinates(x, X, scale, [B], bw)
        native = solver(B, X, Y, bw, *args, **solver_args)
    native["fn"] = _reg_original_objective(native["fn"], scale, objective_power)
    return _result(native, "reg", method, bw, x, keep_data, y=y)


def orthoDr_surv(x, y, censor, method="dm", ndr=None, B_initial=None, bw=None,
                 keep_data=False, control=None, maxitr=500, verbose=0, ncore=1):
    """Fit censored-outcome directions with 0 for censoring and 1 for an event.

    Boolean ``censor`` values are accepted. Invalid values raise ``ValueError``.
    Methods are ``'forward'`` (one direction), ``'dn'`` and ``'dm'``.
    Prediction returns weighted survival curves at sorted observed event times,
    retaining repeated event times to match R. Tied events share complete risk
    sets. The fitted basis is returned on the original predictor scale.
    """
    if method not in ("forward", "dn", "dm"):
        raise ValueError("method must be 'forward', 'dn', or 'dm'")
    _check_keep_data(keep_data)
    x = numeric_matrix(x, "x", min_rows=2)
    n, p = x.shape
    y = survival_times(y, n)
    censor = survival_status(censor, n)
    if not (censor == 1).any():
        raise ValueError("censor must contain 0/1 indicators with at least one observed event")
    if ndr is None:
        ndr = 1 if method == "forward" else 2
    d = min(positive_integer(ndr, "ndr"), p)
    if method == "forward" and d != 1:
        warnings.warn("The forward method uses one direction.", UserWarning)
        d = 1
    bw = _bandwidth(bw, d, n)
    X, sx = survival_coordinates(x)
    ranks = rankdata(y, method="average") / (n + 1)
    sy = np.std(ranks, ddof=1)
    Y = np.zeros(n) if sy == 0 else (ranks - ranks.mean()) / sy / np.sqrt(2) / silverman(1, n)
    if B_initial is None:
        B = CP_SIR(X, Y, censor)["vectors"][:, :d]
    else:
        B = B_initial
    B = _basis(B, p, d)
    def check_projection(b):
        singular = np.linalg.svd(X @ b, compute_uv=False)
        tol = max(X.shape) * np.finfo(float).eps * np.linalg.norm(X)
        if len(singular) < b.shape[1] or np.any(singular <= tol):
            raise ValueError("The survival directions must give linearly independent, varying projected predictors.")

    if B_initial is not None:
        check_projection(B)
    order = np.argsort(Y, kind="stable")
    X, Y, C = X[order], Y[order], censor[order]
    events = np.flatnonzero(C == 1)
    starts = np.searchsorted(Y, Y[events], side="left")
    args = _control_args(control_check(control)) + [_iterations(maxitr), _integer(verbose, "verbose"), _integer(ncore, "ncore")]
    core = _native()
    # The preserved native routines consume R's one-based event/risk indices.
    if method == "forward":
        native = core.surv_forward_solver(B, X, events + 1, starts + 1, bw, *args)
    else:
        ky = np.exp(-np.abs(Y[:, None] - Y[None, :]))
        ky[Y[:, None] < Y[None, :]] = 0
        phi = np.empty((p, len(events)))
        for j, event in enumerate(events):
            weight = C * ky[:, event]
            phi[:, j] = (weight @ X) / weight.sum() - X[starts[j]:].mean(axis=0)
        native = getattr(core, "surv_" + method + "_solver")(B, X, phi, events + 1, starts + 1, bw, *args)
    if B_initial is not None:
        check_projection(np.asarray(native["B"]))
    native["B"] = raw_predictor_directions(np.asarray(native["B"]), sx)
    return _result(native, "surv", method, bw, x, keep_data, y=y, censor=censor)


def orthoDr_pdose(x, a, r, ndr=2, B_initial=None, bw=None, lambda_init=0.1,
                  K=None, method="direct", keep_data=False, control=None,
                  maxitr=500, verbose=0, ncore=1, lambda_reward=None,
                  intercept=None, intercept_bounds=None):
    """Fit a continuous personalized dose rule from doses and rewards.

    Larger rewards are preferred. ``lambda_init`` is the R argument ``lambda``;
    for direct learning it controls ridge initialization, while ``lambda_reward``
    penalizes squared coefficients in the reward objective (default 0.1).
    For pseudo-direct learning, ``lambda_init`` controls the final kernel ridge
    dose rule. Its optional unpenalized intercept uses ``alpha + k(x) @ W``;
    it is unbounded and remains off by default. Direct learning instead centers
    kernel features and bounds its average fitted training dose. Individual
    predictions are not clipped under either method.

    If several grid doses tie for the highest estimated reward to numerical
    precision, pseudo-direct learning uses the lowest of them as the training
    target for the ridge fit.

    Inspect ``converge``, ``gradient_norm`` and ``stop_reason``. Pseudo-direct
    convergence requires a small constrained gradient of reward-standardized
    mean squared error; ``fn`` still reports raw reward RMSE. The final dose
    grid smoother uses dose bandwidth ``bw / sqrt(2)`` in dose SD units.
    Numerical convergence does not establish decision accuracy on new data.
    """
    if method not in ("direct", "pseudo_direct"):
        raise ValueError("method must be 'direct' or 'pseudo_direct'")
    _check_keep_data(keep_data)
    if method == "pseudo_direct" and any(v is not None for v in (lambda_reward, intercept_bounds)):
        raise ValueError("lambda_reward and intercept_bounds are only used by direct learning")
    if intercept is not None and not isinstance(intercept, (bool, np.bool_)):
        raise ValueError("intercept must be True or False")
    intercept = False if intercept is None else bool(intercept)
    if not intercept and intercept_bounds is not None:
        raise ValueError("intercept_bounds requires intercept=True")
    x = numeric_matrix(x, "x", min_rows=2)
    n, p = x.shape
    a = numeric_vector(a, "a", n=n)
    r = numeric_vector(r, "r", n=n)
    if method == "pseudo_direct" and np.all(r == r[0]):
        raise ValueError("Constant rewards do not identify directions or a preferred dose")
    d = positive_integer(ndr, "ndr", maximum=p)
    sa = np.std(a, ddof=1)
    if not np.isfinite(sa) or sa <= 0:
        raise ValueError("Observed doses must have positive finite sample variance")
    _reject_mask(lambda_init, "lambda_init")
    lam = np.atleast_1d(np.asarray(lambda_init))
    if lam.ndim != 1 or lam.dtype.kind not in "iuf" or not len(lam) or not np.isfinite(lam).all() or np.any(lam <= 0):
        raise ValueError("lambda_init must contain positive finite ridge parameters")
    lam = lam.astype(float)
    lr = 0.1 if lambda_reward is None else _scalar(lambda_reward, "lambda_reward")
    bounds = None
    if intercept and method == "direct":
        if lr <= 0:
            raise ValueError("lambda_reward must be positive when intercept=True")
        bounds = np.array([a.min(), a.max()]) if intercept_bounds is None else numeric_vector(intercept_bounds, "intercept_bounds", n=2)
        if bounds[0] >= bounds[1]:
            raise ValueError("intercept_bounds must be strictly increasing")
    bw = _bandwidth(bw, d, n)
    k = int(np.floor(np.sqrt(n) if K is None else _scalar(K, "K", lower=1)))
    if k > np.iinfo(np.int32).max:
        raise ValueError("K exceeds the integer range")
    B = pSAVE(x, a, r, ndr=d) if B_initial is None else B_initial
    B = _basis(B, p, d)
    controls = _pseudo_controls(control) if method == "pseudo_direct" else control_check(control)
    if method == "direct":
        if control is not None:
            for name in ("rho", "eta", "tau", "btol", "ftol", "gtol"):
                if name in control:
                    controls[name] = _scalar(control[name], "control['" + name + "']", strict=True)
        if controls["rho"] >= 1 or controls["eta"] >= 1:
            raise ValueError("Direct-learning rho and eta must be less than one")
        controls["gamma"], controls["epsilon"] = 0.85, 1e-6
    args = _control_args(controls) + [_iterations(maxitr), _integer(verbose, "verbose"), _integer(ncore, "ncore")]
    grid = np.linspace(a.min(), a.max(), k)
    core = _native()
    if method == "direct":
        a_dist = np.exp(-((a[:, None] / sa / bw - grid[None, :] / sa / bw) ** 2))
        native = core.pdose_direct_solver(B, x, a, a_dist, grid, r, lam, bw, *args, lr, intercept, bounds)
    else:
        native = core.pdose_semi_solver(B, x, r, a, grid, lam, bw, *args, intercept)
    return _result(native, "pdose", method, bw, x, keep_data, a=a, r=r)


def _test_matrix(fit, x):
    x = numeric_matrix(x, "testx", min_rows=0)
    if x.shape[1] != np.asarray(fit["B"]).shape[0]:
        raise ValueError("New observations must have the same number of predictors as the training data")
    return x


def predict(fit, testx):
    """Return R-compatible prediction fields: ``pred``, or ``surv`` and ``timepoints``."""
    if not isinstance(fit, FitResult):
        raise TypeError("fit must be a fitted orthoDr model")
    if not fit["keep.data"]:
        raise ValueError("Prediction requires keep_data=True when fitting")
    testx = _test_matrix(fit, testx)
    if fit["kind"] == "pdose":
        offset = 0.0
        if fit.get("objective.version") == "direct_joint_intercept_v2":
            alpha, means = fit.get("intercept"), fit.get("kernel.mean")
            if alpha is None or means is None:
                raise ValueError("Intercept prediction requires valid stored intercept and kernel means")
            means = numeric_vector(np.asarray(means).reshape(-1), "stored kernel means", n=len(fit["x"]))
            alpha_array = np.asarray(alpha)
            if alpha_array.ndim or alpha_array.dtype.kind not in "iuf" or not np.isfinite(alpha_array):
                raise ValueError("Intercept prediction requires a finite stored intercept")
            offset = alpha - np.dot(means.reshape(-1), np.asarray(fit["W"]).reshape(-1))
        elif fit.get("objective.version") == "pseudo_mse_intercept_v1":
            alpha = np.asarray(fit.get("intercept"))
            if alpha.ndim or alpha.dtype.kind not in "iuf" or not np.isfinite(alpha):
                raise ValueError("Pseudo-direct prediction requires a finite stored intercept")
            if "kernel.mean" in fit:
                raise ValueError("Pseudo-direct intercept prediction does not use kernel means")
            offset = float(alpha)
        elif "intercept" in fit or "kernel.mean" in fit:
            raise ValueError("Intercept state has an unsupported objective version")
        if len(testx) == 0:
            return {"pred": np.empty(0)}
        pred = np.asarray(_native().dosepred(fit["B"], fit["x"], testx, fit["bw"], fit["W"])).reshape(-1)
        pred += offset
        return {"pred": pred}
    if fit["kind"] == "surv":
        try:
            censor = survival_status(fit.get("censor"), len(fit["x"]))
        except ValueError as exc:
            raise ValueError(f"Invalid stored survival status. {exc} "
                             "Recode the original status data and refit the survival model.") from exc
        try:
            stored_times = survival_times(fit.get("y"), len(fit["x"]), "Stored y")
        except ValueError as exc:
            raise ValueError(f"{exc} Correct the original follow-up times and refit the survival model.") from exc
    bw = fit.get("bw")
    if bw is None:
        raise ValueError("This fitted object has no bandwidth. Supply the original bandwidth or refit the model.")
    bw = _scalar(bw, "The fitted bandwidth", lower=0, strict=True)
    log_weights = prediction_log_weights(fit["x"], testx, fit["B"], bw)
    # A positive subnormal absolute weight is sufficient to support a row.
    with np.errstate(under="ignore"):
        underflow = np.exp(log_weights.max(axis=1)) == 0
    if underflow.any():
        warnings.warn("Gaussian prediction weights underflowed; affected predictions are NaN.", RuntimeWarning)
    if fit["kind"] == "reg":
        out = np.full(len(testx), np.nan)
        y = np.asarray(fit["y"])
        for i in np.flatnonzero(~underflow):
            out[i] = gaussian_mean(log_weights[i], y)
        return {"pred": out}
    if fit["kind"] != "surv":
        raise ValueError("Unknown fitted model kind")
    order, blocks, survival = weighted_km(log_weights[~underflow], stored_times, censor == 1)
    observed = censor[order] == 1
    out = np.full((int(observed.sum()), len(testx)), np.nan)
    out[:, ~underflow] = survival[blocks[observed]]
    return {"surv": out, "timepoints": stored_times[order[observed]]}
