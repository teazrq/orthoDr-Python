"""Wen-Yin curvilinear optimization with orthonormal columns."""

from __future__ import annotations

import numpy as np

from ._utils import _reject_mask, control_check, gram_schmidt, numeric_matrix, positive_integer


_FEASIBILITY_TOLERANCE = 1e-10


def _objective_value(value):
    _reject_mask(value, "fn result")
    a = np.asarray(value)
    if a.size != 1 or a.dtype.kind not in "iuf":
        raise ValueError("fn must return a single number at every evaluation")
    out = float(a.reshape(-1)[0])
    if not np.isfinite(out):
        raise ValueError("fn must return a finite number at every evaluation")
    return out


def _gradient_value(value, shape):
    _reject_mask(value, "grad result")
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
    tau = c["tau"]
    dense = d >= p // 2
    identity = np.eye(d)
    system_identity = np.eye(p if dense else 2 * d)
    changes = np.zeros((5, 2))
    sequence = np.zeros(maxitr)

    def evaluate_f(b):
        return _objective_value(fn(b.copy()))

    def projected_gradient(b):
        if grad is not None:
            g = _gradient_value(grad(b.copy()), b.shape)
        else:
            g = np.empty_like(b)
            # Column-major evaluation order matches the R solver.
            for j in range(d):
                for i in range(p):
                    new_b = b.copy()
                    new_b[i, j] += c["epsilon"]
                    above = evaluate_f(new_b)
                    new_b[i, j] = b[i, j] - c["epsilon"]
                    g[i, j] = (above - evaluate_f(new_b)) / (2 * c["epsilon"])
            if not np.all(np.isfinite(g)):
                raise ValueError("Gradient must be finite at every evaluation")
        btg = b.T @ g
        projected = g - b @ (0.5 * btg + 0.5 * btg.T)
        # Remove any residual normal component after subtracting a large term.
        residual = b.T @ projected
        return projected - b @ (0.5 * residual + 0.5 * residual.T)

    def gradient_norm(projected):
        norm = float(np.linalg.norm(projected))
        if not np.isfinite(norm):
            raise ValueError("Constrained gradient norm must be finite")
        return norm

    def feasibility(b):
        return np.linalg.norm(b.T @ b - identity)

    if feasibility(B) > _FEASIBILITY_TOLERANCE:
        raise ValueError("Initial B must have orthonormal columns")

    F = evaluate_f(B)
    D = projected_gradient(B)
    norm_g = gradient_norm(D)
    best_norm = norm_g
    completed = 0
    last_improvement = 0
    reason = "iteration limit reached"
    Q, cval = 1.0, F
    if verbose > 1:
        print(f"Initial value, F = {F}")

    while norm_g > c["gtol"] and completed < maxitr:
        bp, fp = B, F
        # The tangent gradient defines the same skew generator. Scaling avoids
        # squaring the ambient gradient's magnitude in the reduced system.
        unit = D / norm_g
        direction = unit - B @ (unit.T @ B)
        slope = c["rho"] * norm_g * float(np.sum(unit * direction))
        if dense:
            gbt = unit @ B.T
            h = 0.5 * (gbt - gbt.T)
            hb = h @ B
        else:
            u = np.column_stack((unit, B))
            v = np.column_stack((B, -unit))
            vu = v.T @ u
            vb = v.T @ B

        step = tau * norm_g
        accepted = False
        for _ in range(30):
            if not np.isfinite(step) or step <= 0:
                break
            threshold = cval - step * slope
            if np.isfinite(threshold):
                try:
                    with np.errstate(over="ignore", invalid="ignore"):
                        if dense:
                            candidate = np.linalg.solve(system_identity + step * h, B - step * hb)
                        else:
                            coefficients = np.linalg.solve(system_identity + 0.5 * step * vu, vb)
                            candidate = B - u @ (step * coefficients)
                        feasible = (np.all(np.isfinite(candidate))
                                    and feasibility(candidate) <= _FEASIBILITY_TOLERANCE)
                except np.linalg.LinAlgError:
                    feasible = False
                if feasible and not np.array_equal(candidate, B):
                    value = evaluate_f(candidate)
                    if value <= threshold:
                        B, F = candidate, value
                        accepted = True
                        break
            smaller = c["eta"] * step
            if smaller >= step:
                break
            step = smaller
        if not accepted:
            reason = "line search failed to find a feasible sufficient-decrease step"
            break

        next_d = projected_gradient(B)
        s, y = B - bp, next_d - D
        accepted_tau = step / norm_g
        D, norm_g = next_d, gradient_norm(next_d)
        sequence[completed] = F
        changes[completed % 5] = (np.linalg.norm(s) / np.sqrt(p),
                                  abs(fp - F) / (abs(fp) + 1))
        completed += 1
        if verbose > 1 and completed % 10 == 0:
            print(f"At iteration {completed}, F = {F}")
        if norm_g <= c["gtol"]:
            break
        if norm_g < best_norm:
            best_norm = norm_g
            last_improvement = completed
        if (completed - last_improvement >= 5 and np.mean(changes[:, 0]) < c["btol"]
                and np.mean(changes[:, 1]) < c["ftol"]):
            reason = "parameter and objective changes stagnated"
            break
        ss, sy, yy = np.sum(s * s), abs(np.sum(s * y)), np.sum(y * y)
        tau, _ = _bb_step(ss, sy, yy, completed % 2 == 0,
                          accepted_tau if accepted_tau > 0 else tau)
        Q = c["gamma"] * Q + 1
        cval += (F - cval) / Q

    converged = norm_g <= c["gtol"]
    if verbose:
        if not converged:
            print(reason)
        print(f"number of iterations: {completed}")
        print(f"objective: {F}, gradient norm: {norm_g}")
        print(f"feasibility norm: {feasibility(B)}")
    return dict(B=B, fn=float(F), fn_Seq=sequence, itr=completed, converge=bool(converged))


def ortho_optim(B, fn, grad=None, *args, maximize=False, control=None,
                maxitr=500, verbose=0, **kwargs):
    """Optimize ``fn(B, *args, **kwargs)`` subject to ``B.T @ B == I``.

    Analytical gradients have the same shape as ``B``. If omitted, central
    finite differences use ``control['epsilon']``. The returned dictionary
    includes ``B``, ``fn``, zero-padded ``fn_Seq``, ``itr``, ``converge``, and
    ``method``. Convergence requires the norm of the projected gradient to
    meet ``control['gtol']``. A failed line search returns the last accepted
    state with ``converge=False``. Every callback result is checked.
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
