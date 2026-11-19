"""Numerical utilities shared by the orthoDr estimators."""

from __future__ import annotations

from collections.abc import Mapping
import warnings

import numpy as np
from scipy.linalg import qr
from scipy.spatial.distance import cdist


def _reject_mask(x, name):
    mask = np.ma.getmask(x)
    if mask is np.ma.nomask or not np.any(mask):
        return
    indices = np.argwhere(mask)
    shown = ", ".join("(" + ", ".join(map(str, idx)) + ")" for idx in indices[:8])
    if len(indices) > 8:
        shown += f" and {len(indices) - 8} more"
    raise ValueError(f"{name} contains masked values at 0-based indices {shown}.")


def numeric_matrix(x, name="x", min_rows=1, min_cols=1):
    """Return a finite, real two-dimensional floating point array."""
    _reject_mask(x, name)
    a = np.asarray(x)
    if a.ndim != 2 or a.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric matrix")
    if a.shape[0] < min_rows or a.shape[1] < min_cols:
        raise ValueError(f"{name} must have at least {min_rows} rows and {min_cols} columns")
    a = np.asarray(a, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain only finite values")
    return a


def numeric_vector(x, name="y", n=None, allow_bool=False, allow_column=False):
    """Validate a numeric observation vector, optionally accepting one column."""
    _reject_mask(x, name)
    a = np.asarray(x)
    if allow_column and a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]
    kinds = "iufb" if allow_bool else "iuf"
    if a.ndim != 1 or a.dtype.kind not in kinds:
        raise ValueError(f"{name} must be a finite numeric vector")
    if n is not None and len(a) != n:
        raise ValueError(f"{name} must have one value per observation")
    a = np.asarray(a, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite numeric vector")
    return a


def survival_times(times, n, name="y"):
    """Validate survival times, including missing observations masked by NumPy."""
    mask = np.ma.getmask(times)
    values = np.asarray(times)
    if values.ndim != 1 or values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite numeric vector")
    if len(values) != n:
        raise ValueError(f"{name} must have one value per observation")
    # Ranking and risk sets use the original precision. Consecutive int64
    # timestamps can round to the same float even though both are finite.
    invalid = np.flatnonzero(mask | ~np.isfinite(values))
    if invalid.size:
        rows = ", ".join(str(row) for row in invalid[:8])
        if invalid.size > 8:
            rows += f" and {invalid.size - 8} more"
        raise ValueError(f"{name} must contain finite times. Invalid at 0-based observation rows {rows}.")
    return values


def survival_coordinates(x):
    """Center observed differences and equilibrate predictor units before testing rank."""
    with np.errstate(over="ignore"):
        delta = x - x[0]
    span = np.max(np.abs(delta), axis=0)
    if np.any(span == 0):
        cols = ", ".join(map(str, np.flatnonzero(span == 0)))
        raise ValueError(f"Constant predictor at 0-based columns {cols}. Each predictor must have positive sample variance.")
    if not np.isfinite(span).all():
        raise ValueError("Predictor differences exceed the numeric range. Rescale x.")
    relative = delta / span
    centered = relative - relative.mean(axis=0)
    sd_relative = np.sqrt(np.sum(centered**2, axis=0) / (len(x) - 1))
    scale = span * sd_relative
    if np.any(~np.isfinite(scale) | (scale == 0)):
        raise ValueError("Predictor scales exceed the numeric range. Rescale x.")
    return centered / sd_relative, scale


def check_survival_rank(singular, shape):
    tol = max(shape) * np.finfo(float).eps * singular[0]
    if len(singular) < shape[1] or np.any(singular <= tol):
        raise ValueError("Survival fitting requires full centered predictor rank (a positive definite feature covariance). Remove redundant predictors.")


def raw_predictor_directions(b, scale):
    message = "Cannot express the fitted direction in the supplied predictor units. Rescale x."
    peak = np.max(np.abs(b), axis=0)
    if np.any(~np.isfinite(peak) | (peak == 0)):
        raise ValueError(message)
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        raw = b / scale[:, None]
        largest = np.max(np.abs(raw), axis=0)
        bad = ~np.isfinite(largest) | (largest == 0)
        if np.any(bad):
            mag = np.log(np.abs(b[:, bad])) - np.log(scale[:, None])
            raw[:, bad] = np.sign(b[:, bad]) * np.exp(mag - mag.max(axis=0))
        raw /= np.max(np.abs(raw), axis=0)
        raw /= np.linalg.norm(raw, axis=0)
        # Reject a unit vector that has lost a resolved component through underflow.
        mag = np.log(np.abs(raw)) + np.log(scale[:, None])
        restored = np.sign(raw) * np.exp(mag - mag.max(axis=0))
    if np.max(np.abs(restored - b / peak)) > np.sqrt(np.finfo(float).eps):
        raise ValueError(message)
    return raw


def survival_status(censor, n):
    """Validate one binary event indicator per observation."""
    coding = "Use 0 for right censoring and 1 for an observed event."
    vector_error = f"censor must be a one-dimensional numeric or Boolean vector. {coding}"
    try:
        status = np.asarray(censor)
    except (TypeError, ValueError) as exc:
        raise ValueError(vector_error) from exc
    if status.ndim != 1 or status.dtype.kind not in "iufb":
        raise ValueError(vector_error)
    if len(status) != n:
        raise ValueError(f"censor must have one value per observation (expected {n}, got {len(status)}). {coding}")
    _reject_mask(censor, "censor (0/1 event indicator)")
    invalid = np.flatnonzero((status != 0) & (status != 1))
    if invalid.size:
        examples = ", ".join(f"{i}={status[i].item()!r}" for i in invalid[:5])
        if invalid.size > 5:
            examples += f" and {invalid.size - 5} more"
        raise ValueError(f"censor must contain 0/1 indicators. {coding} "
                         f"Invalid values by 0-based row index (index=value) are {examples}.")
    return status.astype(float, copy=False)


def positive_integer(value, name, maximum=None):
    a = np.asarray(value)
    if (a.ndim != 0 or a.dtype.kind not in "iuf" or not np.isfinite(a)
            or a < 1 or a != np.floor(a) or (maximum is not None and a > maximum)):
        requirement = (f"an integer between 1 and {maximum}" if maximum is not None
                       else "a positive integer")
        raise ValueError(f"{name} must be {requirement}")
    return int(a)


def gram_schmidt(B):
    """Orthonormalize columns without changing their order or orientation.

    Modified Gram-Schmidt with a second orthogonalization pass avoids loss of
    orthogonality for nearly dependent inputs. Rank-deficient inputs are errors.
    """
    a = numeric_matrix(B, "B")
    p, d = a.shape
    if d > p:
        raise ValueError("B must have at least as many rows as columns")
    q = np.zeros_like(a)
    # Retain pracma's absolute rank tolerance for supplied starting columns.
    tol = np.sqrt(np.finfo(float).eps)
    for j in range(d):
        v = a[:, j].copy()
        for _ in range(2):
            for k in range(j):
                v -= np.dot(q[:, k], v) * q[:, k]
        norm = np.linalg.norm(v)
        if not np.isfinite(norm) or norm <= tol:
            raise ValueError("B does not have full column rank")
        q[:, j] = v / norm
    return q


_CONTROL_DEFAULTS = dict(rho=1e-4, eta=0.2, gamma=0.85, tau=1e-3,
                         epsilon=1e-6, btol=1e-6, ftol=1e-6, gtol=1e-6)


def control_check(control=None):
    """Fill invalid or absent positive tuning values with the R defaults."""
    if control is None:
        control = {}
    if not isinstance(control, Mapping):
        raise ValueError("control must be a mapping of tuning parameters")
    out = dict(control)
    for name, default in _CONTROL_DEFAULTS.items():
        val = np.asarray(out.get(name, default))
        if (val.ndim != 0 or val.dtype.kind not in "iuf" or
                not np.isfinite(val) or val <= 0):
            out[name] = default
        else:
            out[name] = float(val)
    return out


def silverman(d, n):
    """Silverman's dimension-dependent rule of thumb bandwidth."""
    d = positive_integer(d, "d")
    n = positive_integer(n, "n")
    return float((4 / (d + 2)) ** (1 / (d + 4)) * n ** (-1 / (d + 4)))


def dist_cross(x1, x2):
    """Return exp(-squared Euclidean distance) between all pairs of rows."""
    x1 = numeric_matrix(x1, "x1", min_rows=0)
    x2 = numeric_matrix(x2, "x2", min_rows=0)
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("x1 and x2 must have the same number of columns")
    return np.exp(-cdist(x1, x2, metric="sqeuclidean"))


def kernel_weight(x1, x2, kernel="gaussian", dist="euclidean"):
    """Gaussian kernel weights, using the R package's exp(-distance**2) rule.

    The compatibility arguments ``kernel`` and ``dist`` are currently unused,
    just as in orthoDr's R interface.
    """
    return dist_cross(x1, x2)


def _basis(x, name):
    a = np.asanyarray(x)
    if a.ndim == 1:
        a = a[:, None]
    a = numeric_matrix(a, name)
    peak = np.max(np.abs(a), axis=0)
    if np.any(peak == 0):
        raise ValueError(f"{name} must have full column rank")
    a = a / peak
    if np.linalg.matrix_rank(a) != a.shape[1]:
        raise ValueError(f"{name} must have full column rank")
    return a


def distance(s1, s2, type="dist", x=None):
    """Compare subspaces as in the original R package.

    ``dist`` gives a metric, the Frobenius distance between projection matrices.
    ``trace`` and ``canonical`` are similarities. For equal dimensions,
    ``sine`` returns the square root of the sum of fourth powers of the
    principal-angle sines. For unequal dimensions it adds any excess of the
    first subspace's dimension over the second's before taking the square
    root. A line in a plane gives 0 with the line first and 1 with the plane
    first. The sine discrepancy can violate the triangle inequality even at
    equal dimensions.
    """
    s1, s2 = _basis(s1, "s1"), _basis(s2, "s2")
    if s1.shape[0] != s2.shape[0]:
        raise ValueError("Dimension P of two spaces do not match")
    if s1.shape[1] != s2.shape[1]:
        warnings.warn("Dimension d of two spaces do not match", UserWarning, stacklevel=2)
    if type not in {"dist", "trace", "canonical", "sine"}:
        raise ValueError("type must be 'dist', 'trace', 'canonical', or 'sine'")
    q1 = qr(s1, mode="economic")[0]
    q2 = qr(s2, mode="economic")[0]
    if type == "canonical":
        if x is None:
            raise ValueError("x must be specified if use type = 'canonical'")
        x = numeric_matrix(x)
        if x.shape[1] != s1.shape[0]:
            raise ValueError("Dimension of x is not correct")
        # cancor(..., xcenter=FALSE, ycenter=FALSE) uses the uncentered QR
        # bases; rank-deficient projected data retain only nonzero directions.
        u, v = x @ q1, x @ q2
        qu, _, _ = qr(u, mode="economic", pivoting=True)
        qv, _, _ = qr(v, mode="economic", pivoting=True)
        ru, rv = np.linalg.matrix_rank(u), np.linalg.matrix_rank(v)
        if not ru or not rv:
            raise ValueError("canonical correlation requires nonzero projected rank")
        return float(np.mean(np.clip(np.linalg.svd(qu[:, :ru].T @ qv[:, :rv], compute_uv=False), 0, 1)))
    p1, p2 = q1 @ q1.T, q2 @ q2.T
    if type == "dist":
        return float(np.linalg.norm(p1 - p2, ord="fro"))
    if type == "trace":
        return float(np.trace(p1 @ p2) / s1.shape[1])
    # The nonzero eigenvalues of P1(I-P2) are those of this smaller symmetric
    # matrix, so the R expression can be evaluated without complex roundoff.
    eigenvalues = np.linalg.eigvalsh(np.eye(q1.shape[1]) - (q1.T @ q2) @ (q2.T @ q1))
    return float(np.linalg.norm(eigenvalues))
