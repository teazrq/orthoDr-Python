"""Wen-Yin curvilinear optimization with orthonormal columns."""

from __future__ import annotations

import numpy as np

from ._utils import control_check, gram_schmidt, numeric_matrix, positive_integer


def _objective_value(value):
    a = np.asarray(value)
    if a.size != 1 or a.dtype.kind not in "iuf":
        raise ValueError("fn must return a single number at every evaluation")
    out = float(a.reshape(-1)[0])
    if not np.isfinite(out):
        raise ValueError("fn must return a finite number at every evaluation")
    return out


def _gradient_value(value, shape):
    a = np.asarray(value)
    if a.dtype.kind not in "iuf" or a.ndim != 2 or a.shape != shape:
        raise ValueError("grad must return a numeric matrix with the same dimension as B at every evaluation")
    out = np.array(a, dtype=float, copy=True)
    if not np.all(np.isfinite(out)):
        raise ValueError("grad must return a finite matrix at every evaluation")
    return out


def _bb_step(ss, sy, yy, even, previous, upper=1e20):
    numerator, denominator = (ss, sy) if even else (sy, yy)
    valid = (np.isfinite(numerator) and numerator > 0
             and np.isfinite(denominator) and denominator > 0)
    candidate = numerator / denominator if valid else previous
    valid = bool(valid and np.isfinite(candidate) and candidate > 0)
    if not valid:
        candidate = previous
    if not np.isfinite(candidate) or candidate <= 0:
        raise ValueError("Optimizer step must be positive and finite")
    return float(np.clip(candidate, 1e-20, upper)), valid


def _stiefel_solver(B, fn, grad, *, control=None, maxitr=500, verbose=0):
    """Internal solver operating on already orthonormal starting columns."""
    B = np.array(B, dtype=float, copy=True)
    c = control_check(control)
    maxitr = positive_integer(maxitr, "maxitr")
    p, d = B.shape
    tau = previous_tau = c["tau"]
    inv_h = not (d < p // 2)  # C++ integer division is intentional.
    crit = np.zeros((maxitr, 3))
    sequence = np.zeros(maxitr)

    def evaluate_f(b):
        return _objective_value(fn(b.copy()))

    def evaluate_g(b):
        if grad is not None:
            return _gradient_value(grad(b.copy()), b.shape)
        f0 = evaluate_f(b)
        out = np.empty_like(b)
        # Column-major evaluation order matches the committed native solver.
        for j in range(d):
            for i in range(p):
                new_b = b.copy()
                new_b[i, j] += c["epsilon"]
                out[i, j] = (evaluate_f(new_b) - f0) / c["epsilon"]
        if not np.all(np.isfinite(out)):
            raise ValueError("Gradient must be finite at every evaluation")
        return out

    def prepare(b, g):
        if inv_h:
            gxt = g @ b.T
            h = 0.5 * (gxt - gxt.T)
            return h, h @ b
        u = np.column_stack((g, b))
        v = np.column_stack((b, -g))
        return u, v.T @ u, v.T @ b

    F = evaluate_f(B)
    G = evaluate_g(B)
    prepared = prepare(B, G)
    dt_x = G - B @ (G.T @ B)
    norm_g = float(np.linalg.norm(dt_x))
    if not np.isfinite(norm_g):
        raise ValueError("Constrained gradient norm must be finite")
    converged = norm_g < c["gtol"]
    completed, clean_updates = 0, 0
    Q, cval = 1.0, F
    if verbose > 1:
        print(f"Initial value, F = {F}")

    for itr in range(1, maxitr + 1):
        if converged:
            break
        bp, fp, dt_xp = B.copy(), F, dt_x.copy()
        derivative = c["rho"] * norm_g * norm_g
        for nls in range(1, 6):
            if inv_h:
                h, rx = prepared
                B = np.linalg.solve(np.eye(p) + tau * h, bp - tau * rx)
            else:
                u, vu, vx = prepared
                aa = np.linalg.solve(np.eye(2 * d) + 0.5 * tau * vu, vx)
                B = bp - u @ (tau * aa)
            if not np.all(np.isfinite(B)):
                raise ValueError("Optimizer update must be finite")
            F, G = evaluate_f(B), evaluate_g(B)
            if F <= cval - tau * derivative or nls >= 5:
                break
            tau *= c["eta"]
        completed = itr
        sequence[itr - 1] = F
        prepared = prepare(B, G)
        dt_x = G - B @ (G.T @ B)
        norm_g = float(np.linalg.norm(dt_x))
        if not np.isfinite(norm_g):
            raise ValueError("Constrained gradient norm must be finite")
        if norm_g < c["gtol"]:
            converged = True
            break
        s = B - bp
        b_diff = np.linalg.norm(s) / np.sqrt(p)
        f_diff = abs(fp - F) / (abs(fp) + 1)
        y = dt_x - dt_xp
        ss, sy, yy = np.sum(s * s), abs(np.sum(s * y)), np.sum(y * y)
        tau, valid_bb = _bb_step(ss, sy, yy, itr % 2 == 0, previous_tau)
        previous_tau = tau
        clean_updates = clean_updates + 1 if valid_bb and np.isfinite(ss) and ss > 0 else 0
        crit[itr - 1] = norm_g, b_diff, f_diff
        if verbose > 1 and itr % 10 == 0:
            print(f"At iteration {itr}, F = {F}")
        if itr >= 5:
            recent = crit[itr - 5:itr]
            if ((clean_updates > 0 and b_diff < c["btol"] and f_diff < c["ftol"])
                    or (clean_updates >= 5 and np.mean(recent[:, 1]) < c["btol"]
                        and np.mean(recent[:, 2]) < c["ftol"])):
                converged = True
                break
        qp = Q
        Q = c["gamma"] * qp + 1
        cval = (c["gamma"] * qp * cval + F) / Q

    if verbose:
        print(f"number of iterations: {completed}")
        print(f"objective: {F}; gradient norm: {norm_g}")
        print(f"feasibility norm: {np.linalg.norm(B.T @ B - np.eye(d))}")
        if not converged:
            print("exceed max iteration before convergence")
    return dict(B=B, fn=float(F), fn_Seq=sequence, itr=completed, converge=bool(converged))


def ortho_optim(B, fn, grad=None, *args, maximize=False, control=None,
                maxitr=500, verbose=0, **kwargs):
    """Optimize ``fn(B, *args, **kwargs)`` subject to ``B.T @ B == I``.

    Analytical gradients have the same shape as ``B``. If omitted, forward
    finite differences use ``control['epsilon']``. The returned dictionary
    includes ``B``, ``fn``, zero-padded ``fn_Seq``, ``itr``, ``converge``, and
    ``method``. Every callback result is checked, including line-search trials.
    """
    B = numeric_matrix(B, "B")
    if B.shape[1] > B.shape[0]:
        raise ValueError("B must have at least as many rows as columns")
    if not callable(fn):
        raise ValueError("fn must be callable")
    if grad is not None and not callable(grad):
        raise ValueError("grad must be callable or None")
    if not isinstance(maximize, (bool, np.bool_)):
        raise ValueError("maximize must be boolean")
    if np.sum(np.abs(B.T @ B - np.eye(B.shape[1]))) > 1e-15:
        B = gram_schmidt(B)
    sign = -1 if maximize else 1

    def f(b):
        return sign * _objective_value(fn(b, *args, **kwargs))

    def g(b):
        return sign * _gradient_value(grad(b, *args, **kwargs), b.shape)

    # Match R's wrapper validation before the solver's initial evaluations.
    f(B.copy())
    if grad is not None:
        g(B.copy())
    result = _stiefel_solver(B, f, g if grad is not None else None,
                             control=control, maxitr=maxitr, verbose=verbose)
    if maximize:
        result["fn"] *= -1
        result["fn_Seq"] *= -1
    result["method"] = "true gradient" if grad is not None else "approx. gradient"
    return result
