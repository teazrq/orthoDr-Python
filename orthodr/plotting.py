"""Optional Matplotlib views of a one-dimensional survival reduction."""

from __future__ import annotations

import numpy as np

from ._utils import numeric_matrix, numeric_vector, silverman


def _weighted_km_surface(index, times, event, xgrid, timegrid, bw):
    """Right-continuous weighted KM on complete tied-time risk sets.

    Rows correspond to xgrid and columns to timegrid. Gaussian weights are
    rescaled by a common positive row factor to avoid complete underflow.
    This does not change the weighted Kaplan-Meier estimator.
    """
    order = np.argsort(times, kind="stable")
    times, event, index = times[order], event[order], index[order]
    starts = np.r_[0, np.flatnonzero(np.diff(times) != 0) + 1]
    block_times = times[starts]
    with np.errstate(over="ignore", invalid="ignore"):
        log_weights = -.5 * ((xgrid[:, None] - index[None, :]) / bw) ** 2
        log_weights -= np.max(log_weights, axis=1, keepdims=True)
        weights = np.exp(log_weights)
    if not np.all(np.isfinite(weights)):
        raise ValueError("Survival kernel weights must be finite")
    weights = weights.T
    masses = np.add.reduceat(weights, starts, axis=0)
    deaths = np.add.reduceat(weights * event[:, None], starts, axis=0)
    risks = np.cumsum(masses[::-1], axis=0)[::-1]
    ratios = np.zeros_like(risks)
    np.divide(deaths, risks, out=ratios, where=risks > 0)
    survival = np.cumprod(1 - np.minimum(ratios, 1), axis=0)
    positions = np.searchsorted(block_times, timegrid, side="right") - 1
    result = np.ones((len(xgrid), len(timegrid)))
    after_observation = positions >= 0
    result[:, after_observation] = survival[positions[after_observation]].T
    return result


def view_dr_surv(x, y, censor, B=None, bw=None, FUN="log", type="2D",
                 legend_add=True, xlab="Reduced Direction", ylab="Time",
                 zlab="Survival", *, ax=None):
    """Plot survival observations or a weighted Kaplan-Meier surface.

    ``B`` is a vector or one-column feature-direction matrix. ``FUN`` is a
    callable or one of ``'log'``, ``'log10'``, ``'log2'``, ``'sqrt'``, and
    ``'identity'``; None also means identity. ``type`` is ``'2D'`` or ``'3D'``.
    The surface uses the R function's 100-point direction grid, transformed
    time grid, and Gaussian bandwidth convention. Matplotlib is loaded only
    when called. Returns the axes and never calls ``show()``.
    """
    x = numeric_matrix(x, min_rows=2)
    n, p = x.shape
    y = numeric_vector(y, n=n)
    censor = numeric_vector(censor, "censor", n=n, allow_bool=True)
    if not np.all(np.isin(censor, [0, 1])):
        raise ValueError("censor can only have two levels 0/1")
    if B is None:
        raise ValueError("B must be given")
    direction = numeric_vector(B, "B", n=p, allow_column=True)
    if type not in {"2D", "3D"}:
        raise ValueError("type must be '2D' or '3D'")
    if not isinstance(legend_add, (bool, np.bool_)):
        raise ValueError("legend_add must be boolean")
    transforms = dict(log=np.log, log10=np.log10, log2=np.log2,
                      sqrt=np.sqrt, identity=lambda a: a)
    if FUN is None:
        transform = transforms["identity"]
    elif callable(FUN):
        transform = FUN
    elif isinstance(FUN, str) and FUN in transforms:
        transform = transforms[FUN]
    else:
        raise ValueError("FUN must be a callable or a supported time transformation")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        times = numeric_vector(transform(y.copy()), "FUN(y)", n=n)
        index = x @ direction
    if not np.all(np.isfinite(index)):
        raise ValueError("The reduced direction must be finite")
    if type == "3D":
        if np.std(index, ddof=1) <= 0:
            raise ValueError("A survival surface requires a varying reduced direction")
        if bw is None:
            bw = silverman(1, n) * np.std(index, ddof=1)
        bandwidth = np.asarray(bw)
        if (bandwidth.ndim != 0 or bandwidth.dtype.kind not in "iuf"
                or not np.isfinite(bandwidth) or bandwidth <= 0):
            raise ValueError("bw must be a positive finite scalar")
        xgrid = np.linspace(np.min(index), np.max(index), 100)
        timegrid = np.unique(np.r_[times, np.linspace(0, np.max(times), 100)])
        survival = _weighted_km_surface(index, times, censor, xgrid, timegrid, float(bandwidth))
    try:
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise ImportError("Install matplotlib in your shared Python installation to use view_dr_surv") from exc
    if ax is None:
        figure = Figure(figsize=(7, 5))
        ax = figure.add_subplot(projection="3d" if type == "3D" else None)
    if type == "2D":
        event = censor == 1
        ax.scatter(index[event], times[event], c="blue", marker="o", label="failure")
        ax.scatter(index[~event], times[~event], c="red", marker="+", label="censored")
        if legend_add:
            ax.legend(loc="upper right")
    else:
        if not hasattr(ax, "plot_surface"):
            raise ValueError("type='3D' requires three-dimensional axes")
        xx, tt = np.meshgrid(xgrid, timegrid, indexing="ij")
        ax.plot_surface(xx, tt, survival, cmap="viridis", alpha=.9,
                        rcount=len(xgrid), ccount=len(timegrid), linewidth=0)
        ax.set_zlabel(zlab)
        ax.set_zlim(0, 1)
        ax.view_init(elev=20, azim=50)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    return ax
