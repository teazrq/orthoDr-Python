"""Optional Matplotlib views of a one-dimensional survival reduction."""

from __future__ import annotations

import numpy as np

from ._prediction import weighted_km
from ._utils import numeric_matrix, numeric_vector, silverman, survival_status, survival_times


def _weighted_km_surface(index, times, event, xgrid, timegrid, bw):
    """Right-continuous weighted KM on complete tied-time risk sets.

    Rows correspond to xgrid and columns to timegrid. The log weights retain
    relative information when absolute Gaussian weights round to zero.
    """
    with np.errstate(over="ignore", under="ignore"):
        log_weights = -((xgrid[:, None] - index[None, :]) / bw / np.sqrt(2)) ** 2
    if not np.isfinite(log_weights).any(axis=1).all():
        raise ValueError("Survival kernel weights must be finite")
    order, blocks, survival = weighted_km(log_weights, times, event == 1)
    block_times = times[order[np.r_[True, blocks[1:] != blocks[:-1]]]]
    positions = np.searchsorted(block_times, timegrid, side="right") - 1
    result = np.ones((len(xgrid), len(timegrid)))
    after_observation = positions >= 0
    result[:, after_observation] = survival[positions[after_observation]].T
    return result


def view_dr_surv(x, y, censor, B=None, bw=None, FUN="log", type="2D",
                 legend_add=True, xlab="Reduced Direction", ylab="Time",
                 zlab="Survival", *, ax=None):
    """Plot survival observations or a weighted Kaplan-Meier surface.

    Supply 0 for censoring and 1 for an event, or equivalent Boolean indicators.
    ``B`` is a vector or one-column feature-direction matrix. ``FUN`` is a
    callable or one of ``'log'``, ``'log10'``, ``'log2'``, ``'sqrt'``, and
    ``'identity'``; None also means identity. ``type`` is ``'2D'`` or ``'3D'``.
    The surface uses a 100-point direction grid and Gaussian bandwidth.
    Its time knots are the pooled follow-up's 0th through 100th empirical
    percentiles (inverse empirical CDF), plus every observed event time.
    Risk sets use original follow-up times. When several knots share a display
    coordinate, it shows survival at the latest of those follow-up times.
    A single time knot is drawn as a line. Matplotlib is loaded only when
    called. Returns the axes and never calls ``show()``.
    """
    x = numeric_matrix(x, min_rows=2)
    n, p = x.shape
    y = survival_times(y, n)
    censor = survival_status(censor, n)
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
        times = survival_times(transform(y.copy()), n, "FUN(y)")
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
        # Select observed order statistics before transforming. Integer ranks
        # avoid floating-point changes at exact empirical-CDF boundaries.
        ranks = np.array([max(1, (k * n + 99) // 100) - 1 for k in range(101)])
        order = np.argsort(y, kind="stable")
        knots = np.unique(np.r_[order[ranks], np.flatnonzero(censor == 1)])
        knots = knots[np.lexsort((y[knots], times[knots]))]
        display = times[knots]
        knots = knots[np.r_[display[1:] != display[:-1], True]]
        timegrid = times[knots]
        survival = _weighted_km_surface(index, y, censor, xgrid, y[knots], float(bandwidth))
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
        if len(timegrid) == 1:
            ax.plot(xgrid, np.full(len(xgrid), timegrid[0]), survival[:, 0])
        else:
            xx, tt = np.meshgrid(xgrid, timegrid, indexing="ij")
            ax.plot_surface(xx, tt, survival, cmap="viridis", alpha=.9,
                            rcount=len(xgrid), ccount=len(timegrid), linewidth=0)
        ax.set_zlabel(zlab)
        ax.set_zlim(0, 1)
        ax.view_init(elev=20, azim=50)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    return ax
