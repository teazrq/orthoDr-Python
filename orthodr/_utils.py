"""Numerical utilities shared by the orthoDr estimators."""

from __future__ import annotations

from collections.abc import Mapping
import warnings

import numpy as np
from scipy.linalg import qr
from scipy.spatial.distance import cdist


def numeric_matrix(x, name="x", min_rows=1, min_cols=1):
    """Return a finite, real two-dimensional floating point array."""
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
    a = np.asarray(x)
    if a.ndim == 1:
        a = a[:, None]
    a = numeric_matrix(a, name)
    if np.linalg.matrix_rank(a) != a.shape[1]:
        raise ValueError(f"{name} must have full column rank")
    return a


def distance(s1, s2, type="dist", x=None):
    """Compare subspaces using the four metrics in the original R package.

    ``trace`` and ``canonical`` are similarities. ``sine`` retains R's
    square-root sum of squared eigenvalues of P1(I-P2), which differs from
    the more usual square-root sum of squared principal-angle sines.
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
        u, v = x @ s1, x @ s2
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
