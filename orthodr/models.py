"""Dimension-reduction models and prediction on the original feature scale."""
from __future__ import annotations

import warnings
from collections.abc import Mapping
import numpy as np
from scipy.stats import rankdata

from ._utils import numeric_matrix, numeric_vector, positive_integer, gram_schmidt, control_check, silverman
from .initializers import CP_SIR, pSAVE, regression_initializers


def _native():
    try:
        from . import _core
    except ImportError as exc:
        raise ImportError("Build the orthoDr native extension before fitting; see the installation instructions.") from exc
    return _core


def _scalar(value, name, *, lower=0, strict=False):
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


def orthoDr_reg(x, y, method="sir", ndr=2, B_initial=None, bw=None,
                keep_data=False, control=None, maxitr=500, verbose=0, ncore=1):
    """Fit regression directions by SIR, SAVE, PHD, LOCAL or efficient estimation.

    ``x`` has observations in rows. Predictors are used on their supplied scale;
    apply any desired standardization consistently to training and new data.
    Set ``keep_data=True`` to predict mean responses. As in R, at least five
    iterations are allowed. Automatic initialization uses SIR/SAVE/PHD candidates;
    one-direction LOCAL uses three fits selected by in-sample kernel R-squared.
    """
    if method not in ("sir", "save", "phd", "local", "seff"):
        raise ValueError("method must be 'sir', 'save', 'phd', 'local', or 'seff'")
    _check_keep_data(keep_data)
    x = numeric_matrix(x, "x", min_rows=2)
    y = numeric_vector(y, "y", n=x.shape[0])
    d = min(positive_integer(ndr, "ndr"), x.shape[1])
    if d > 4:
        warnings.warn("More than four directions is not recommended for kernel estimation.", UserWarning)
    sy = np.std(y, ddof=1)
    if sy <= 0:
        raise ValueError("y must have positive sample variance")
    Y = ((y - y.mean()) / sy / x.shape[0] ** (-1 / (d + 5)) / np.sqrt(2))[:, None]
    bw = _bandwidth(bw, d, x.shape[0])
    controls = control_check(control)
    args = _control_args(controls) + [_iterations(maxitr, 5), _integer(verbose, "verbose"), _integer(ncore, "ncore")]
    core = _native()
    solver = getattr(core, method + "_solver")
    if B_initial is None:
        bases = regression_initializers(x, Y[:, 0])
        candidates = [_basis(bases[m][:, :d], x.shape[1], d) for m in ("sir", "save", "phd")]
        if method == "local" and d == 1:
            fits = [solver(B, x, Y, bw, *args) for B in candidates]
            def r_squared(fit):
                u = x @ fit["B"]
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
            values = [objective(B, x, Y, bw, args[-1]) for B in candidates]
            native = solver(candidates[int(np.argmin(values))], x, Y, bw, *args)
    else:
        native = solver(_basis(B_initial, x.shape[1], d), x, Y, bw, *args)
    return _result(native, "reg", method, bw, x, keep_data, y=y)


def orthoDr_surv(x, y, censor, method="dm", ndr=None, B_initial=None, bw=None,
                 keep_data=False, control=None, maxitr=500, verbose=0, ncore=1):
    """Fit censored-outcome directions; ``censor=1`` denotes an observed event.

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
    y = numeric_vector(y, "y", n=n)
    censor = numeric_vector(censor, "censor", n=n, allow_bool=True)
    if not np.isin(censor, [0, 1]).all() or not (censor == 1).any():
        raise ValueError("censor must contain 0/1 indicators with at least one observed event")
    if ndr is None:
        ndr = 1 if method == "forward" else 2
    d = min(positive_integer(ndr, "ndr"), p)
    if method == "forward" and d != 1:
        warnings.warn("The forward method uses one direction.", UserWarning)
        d = 1
    bw = _bandwidth(bw, d, n)
    sx = np.std(x, axis=0, ddof=1)
    if np.any(sx <= 0):
        raise ValueError("Each predictor must have positive sample variance")
    X = (x - x.mean(axis=0)) / sx
    ranks = rankdata(y, method="average") / (n + 1)
    sy = np.std(ranks, ddof=1)
    Y = np.zeros(n) if sy == 0 else (ranks - ranks.mean()) / sy / np.sqrt(2) / silverman(1, n)
    B = CP_SIR(X, Y, censor)["vectors"][:, :d] if B_initial is None else B_initial
    B = _basis(B, p, d)
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
    raw = np.asarray(native["B"]) / sx[:, None]
    native["B"] = raw / np.linalg.norm(raw, axis=0)
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


def _prediction_kernel(fit, x):
    u, v = fit["x"] @ fit["B"], x @ fit["B"]
    scale = np.std(u, axis=0, ddof=1) * fit["bw"] * np.sqrt(2)
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("Training projection scales must be positive and finite")
    mean = u.mean(axis=0)
    u, v = (u - mean) / scale, (v - mean) / scale
    return np.exp(-((v[:, None, :] - u[None, :, :]) ** 2).sum(axis=2))


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
    weights = _prediction_kernel(fit, testx)
    underflow = weights.sum(axis=1) == 0
    if underflow.any():
        warnings.warn("Gaussian prediction weights underflowed; affected predictions are NaN.", RuntimeWarning)
    if fit["kind"] == "reg":
        out = np.full(len(testx), np.nan)
        np.divide(weights @ fit["y"], weights.sum(axis=1), out=out, where=~underflow)
        return {"pred": out}
    if fit["kind"] != "surv":
        raise ValueError("Unknown fitted model kind")
    order = np.argsort(fit["y"], kind="stable")
    times, observed = fit["y"][order], fit["censor"][order] == 1
    weights = weights[:, order].T
    starts = np.r_[0, np.flatnonzero(np.diff(times) != 0) + 1]
    blocks = np.cumsum(np.r_[True, np.diff(times) != 0]) - 1
    masses = np.add.reduceat(weights, starts, axis=0)
    deaths = np.add.reduceat(weights * observed[:, None], starts, axis=0)
    risk = np.cumsum(masses[::-1], axis=0)[::-1]
    tolerance = 64 * np.finfo(float).eps * np.maximum(np.abs(deaths), np.abs(risk))
    if np.any(deaths - risk > tolerance):
        raise ValueError("Weighted survival death mass exceeds its risk mass")
    deaths = np.minimum(deaths, risk)
    factors = np.ones_like(risk)
    np.divide(deaths, risk, out=factors, where=deaths > 0)
    factors = np.where(deaths > 0, 1 - factors, 1)
    survival = np.cumprod(factors, axis=0)
    survival[:, underflow] = np.nan
    return {"surv": survival[blocks[observed]], "timepoints": times[observed].copy()}
