"""Sliced and survival initial estimators used by orthoDr.

The slice construction and local SIR, SAVE, PHD, and grouped SAVE moments
follow the R ``dr`` dependency. pSAVE pools its group moments in one common
coordinate system. CP-SIR and hMAVE follow the committed orthoDr source.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, qr, solve_triangular
from scipy.special import ndtr

from ._utils import (gram_schmidt, numeric_matrix, numeric_vector,
                     positive_integer, silverman)


def _centered_rank(x):
    z = x - np.mean(x, axis=0)
    # R's qr(..., LAPACK=FALSE) uses a 1e-7 relative column tolerance.
    q = np.zeros_like(z)
    rank = 0
    for j in range(z.shape[1]):
        v = z[:, j].copy()
        for _ in range(2):
            for k in range(rank):
                v -= np.dot(q[:, k], v) * q[:, k]
        norm = np.linalg.norm(v)
        if norm > 1e-7 * np.linalg.norm(z[:, j]) and norm > 0:
            q[:, rank] = v / norm
            rank += 1
    return rank


def dr_slices(y, nslices, *, repair=False):
    """One-dimensional ``dr::dr.slices`` partition, including tied values.

    ``repair`` enables pSAVE's committed repair for the dependency's
    one-breakpoint branch. Slice labels here start at zero.
    """
    y = numeric_vector(y)
    h = positive_integer(nslices, "nslices")
    values, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    if len(values) <= h:
        return dict(indicator=inverse, nslices=len(values), sizes=counts)
    cumulative = np.cumsum(counts)
    m = len(y) // h
    if repair and cumulative[np.searchsorted(cumulative, m)] >= len(y) - 2:
        return dict(indicator=np.zeros(len(y), dtype=int), nslices=1,
                    sizes=np.array([len(y)]))
    end, breaks = 0, []
    while end < len(y) - 2:
        pos = min(np.searchsorted(cumulative, end + m), len(cumulative) - 1)
        breaks.append(int(pos))
        end = cumulative[pos]
    if len(breaks) < 2:
        raise ValueError("dr.slices has no valid multi-slice partition for these observations")
    breaks[-1] = len(cumulative) - 1
    indicator = np.searchsorted(np.asarray(breaks), inverse, side="left")
    sizes = np.diff(np.concatenate(([0], cumulative[breaks])))
    return dict(indicator=indicator, nslices=len(breaks), sizes=sizes)


def _symmetric_inverse_sqrt(a, name="covariance"):
    vals, vecs = eigh(a)
    if np.any(~np.isfinite(vals)) or np.any(vals <= 0):
        raise ValueError(f"{name} must be positive definite for whitening")
    return (vecs / np.sqrt(vals)) @ vecs.T


def dr_fit(x, y, method="sir", nslices=None):
    """Unweighted continuous-predictor SIR, SAVE, or residual PHD fit.

    Returns the moment ``M``, ``evalues``, normalized ``evectors``, and
    ``raw_evectors``. Directions are ordered by absolute eigenvalue, matching
    ``dr``. ``phdy`` is also accepted for the centered-response PHD variant.
    """
    x = numeric_matrix(x, min_rows=2)
    n, p = x.shape
    y = numeric_vector(y, n=n)
    if n <= p or _centered_rank(x) < p:
        raise ValueError("dr initializers require full centred predictor rank and more observations than predictors")
    if method not in {"sir", "save", "phd", "phdres", "phdy"}:
        raise ValueError("method must be 'sir', 'save', 'phd', 'phdres', or 'phdy'")
    xc = x - np.mean(x, axis=0)
    q, r = qr(xc, mode="economic")
    z = np.sqrt(n) * q
    slices = None
    if method in {"phd", "phdres", "phdy"}:
        response = y - np.mean(y)
        if method != "phdy":
            response = response - q @ (q.T @ response)
        moment = (z.T * response) @ z / n
    else:
        h = max(8, p + 3) if nslices is None else positive_integer(nslices, "nslices")
        slices = dr_slices(y, h)
        moment = np.zeros((p, p))
        for j, size in enumerate(slices["sizes"]):
            zj = z[slices["indicator"] == j]
            if method == "sir":
                mu = np.mean(zj, axis=0)
                moment += size * np.outer(mu, mu) / n
            else:
                if size < 2:
                    # dr.M.save's ungrouped weighted covariance is undefined
                    # for singleton slices. Do not silently substitute pSAVE's
                    # different small-slice convention.
                    raise ValueError("SAVE requires at least two observations in every outcome slice")
                centered = zj - np.mean(zj, axis=0)
                difference = np.eye(p) - centered.T @ centered / size
                moment += size * difference @ difference / n
    vals, raw = eigh(moment)
    order = np.argsort(np.abs(vals), kind="stable")[::-1]
    vals, raw = vals[order], raw[:, order]
    directions = solve_triangular(np.sqrt(n) * r, raw)
    directions /= np.linalg.norm(directions, axis=0)
    return dict(M=moment, evalues=vals, evectors=directions, raw_evectors=raw,
                slice_info=slices)


def regression_initializers(x, y, nslices=None):
    """Return normalized SIR, SAVE, and PHD candidate direction matrices."""
    return {method: dr_fit(x, y, method, nslices)["evectors"]
            for method in ("sir", "save", "phd")}


def _psave_group_moment(x, y, nslices):
    """Population local SAVE moment, with tiny-slice mass kept separate."""
    n, p = x.shape
    centered = x - np.mean(x, axis=0)
    covariance = centered.T @ centered / n
    inverse = _symmetric_inverse_sqrt(covariance, "dose group covariance")
    z = centered @ inverse
    slices = dr_slices(y, nslices, repair=True)
    moment = np.zeros((p, p))
    fallback = 0.0
    for j, size in enumerate(slices["sizes"]):
        if size < 3:
            fallback += size / n
            continue
        zj = z[slices["indicator"] == j]
        zj = zj - np.mean(zj, axis=0)
        difference = np.eye(p) - zj.T @ zj / size
        moment += size / n * (difference @ difference)
    return dict(covariance=covariance, inverse_sqrt=inverse, moment=moment,
                fallback_mass=fallback, slices=slices)


def _grouped_save_moment(x, y, group, nslices):
    """One threshold's raw congruence moment and reference covariance."""
    n, p = x.shape
    covariance, raw_moment = np.zeros((p, p)), np.zeros((p, p))
    fallback = 0.0
    has_contrasts = has_information = False
    # Match R's first-observed group order, retaining all original weights.
    _, first = np.unique(group, return_index=True)
    for g in group[np.sort(first)]:
        sel = group == g
        local = _psave_group_moment(x[sel], y[sel], nslices)
        weight = np.sum(sel) / n
        inverse = local["inverse_sqrt"]
        covariance += weight * local["covariance"]
        raw_moment += weight * (inverse @ local["moment"] @ inverse)
        fallback += weight * local["fallback_mass"]
        slices = local["slices"]
        has_contrasts |= slices["nslices"] > 1
        has_information |= slices["nslices"] > 1 and np.any(slices["sizes"] >= 3)
    return dict(covariance=covariance, raw_moment=raw_moment,
                fallback_mass=fallback, has_contrasts=has_contrasts,
                has_information=has_information)


def pSAVE(x, a, r, ndr=2, nslices0=2):
    """Partial SAVE initializer for personalized dose estimation.

    Thresholds retain repetitions and the original central-rank truncation.
    Each nonempty dose group must have p+1 observations and full centered
    rank. The p=1 result is exactly ``[[1.0]]`` after input validation.

    Local symmetric-whitened SAVE moments use population covariances and
    are transported to raw coefficient coordinates group by group. Their
    eigenproblem uses the group-size-weighted within-group covariance as
    a common reference. Slices smaller than three contribute a scalar
    identity only in those common coordinates, preserving their lack of
    directional information without renormalizing informative slice weights.

    Returns unit columns in original predictor coordinates, in decreasing
    common eigenvalue order. The largest absolute raw entry of each column
    is positive (first entry in a tie); columns need not be orthogonal.
    The raw selected subspace transforms affinely when its eigenvalue
    boundary is separated and numerical admissibility is unchanged. This
    coordinate correction changes historical output; statistical recovery
    still requires suitable SAVE identification assumptions.
    """
    x = numeric_matrix(x, min_rows=2)
    n, p = x.shape
    a, r = numeric_vector(a, "a", n=n), numeric_vector(r, "r", n=n)
    ndr = positive_integer(ndr, "ndr", p)
    nslices0 = positive_integer(nslices0, "nslices0")
    if _centered_rank(x) < p:
        raise ValueError("pSAVE requires full centred predictor rank; supply B_initial to orthoDr_pdose")
    if p == 1:
        return np.ones((1, 1))
    start, stop = max(1, int(np.floor(n / 2 - 50))), min(n, int(np.floor(n / 2 + 50)))
    thresholds = np.sort(a)[start - 1:stop]
    threshold_moments = []
    has_contrasts = has_information = False
    for cut in thresholds:
        group = a <= cut
        selections = [group == g for g in np.unique(group)]
        if not all(np.sum(sel) >= p + 1 and _centered_rank(x[sel]) == p
                   for sel in selections):
            continue
        local = _grouped_save_moment(x, r, group, nslices0)
        threshold_moments.append(local)
        has_contrasts |= local["has_contrasts"]
        has_information |= local["has_information"]
    if not threshold_moments:
        raise ValueError("pSAVE has no admissible dose threshold; supply B_initial to orthoDr_pdose")
    if not has_contrasts:
        raise ValueError("pSAVE has no outcome-slice contrasts; supply B_initial to orthoDr_pdose")
    if not has_information:
        raise ValueError("pSAVE has no directional slice information: every retained group has one outcome slice or only slices smaller than three; supply B_initial to orthoDr_pdose")
    nt = len(threshold_moments)
    reference = sum((local["covariance"] / nt for local in threshold_moments),
                    start=np.zeros((p, p)))
    raw_moment = sum((local["raw_moment"] / nt for local in threshold_moments),
                     start=np.zeros((p, p)))
    values, vectors = eigh(reference)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("pSAVE reference covariance must be positive definite for whitening")
    root = (vectors * np.sqrt(values)) @ vectors.T
    inverse = (vectors / np.sqrt(values)) @ vectors.T
    common = root @ raw_moment @ root
    # The omitted total fallback mass multiplies I and changes no directions.
    _, vectors = eigh((common + common.T) / 2)
    directions = inverse @ vectors[:, ::-1][:, :ndr]
    directions /= np.linalg.norm(directions, axis=0)
    pivots = np.argmax(np.abs(directions), axis=0)
    directions *= np.where(directions[pivots, np.arange(ndr)] < 0, -1.0, 1.0)
    return directions


def CP_SIR(x, y, censor, bw=None):
    """Counting-process SIR with tie-aware event neighborhoods."""
    x = numeric_matrix(x, min_rows=2)
    n, p = x.shape
    y = numeric_vector(y, "y", n=n)
    censor = numeric_vector(censor, "censor", n=n, allow_bool=True)
    if not np.all(np.isin(censor, [0, 1])):
        raise ValueError("censor can only have two levels 0/1")
    event = censor == 1
    n_fail = int(np.sum(event))
    if not n_fail:
        raise ValueError("CP_SIR requires at least one event")
    if bw is None:
        bw = silverman(1, n)
    b = np.asarray(bw)
    if b.ndim != 0 or b.dtype.kind not in "iuf" or not np.isfinite(b) or b <= 0:
        raise ValueError("bw must be positive and finite")
    xc = x - np.mean(x, axis=0)
    covariance = xc.T @ xc / (n - 1)
    inverse_sqrt = _symmetric_inverse_sqrt(covariance, "CP_SIR feature covariance")
    standardized = xc @ inverse_sqrt
    fail_indices = np.flatnonzero(event)
    ordered = fail_indices[np.argsort(y[event], kind="stable")]
    timepoints = y[ordered]
    _, inv, counts = np.unique(timepoints, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    midpoint_ranks = (cumulative - (counts - 1) / 2)[inv]
    width = n_fail * float(b) / 2
    gu, gu_risk = np.empty((n_fail, p)), np.empty((n_fail, p))
    for i in range(n_fail):
        lower = int(max(1, midpoint_ranks[i] - width)) - 1
        upper = int(min(n_fail, midpoint_ranks[i] + width)) - 1
        failure = (y >= timepoints[lower]) & (y <= timepoints[upper]) & event
        risk = y >= timepoints[i]
        gu[i] = np.mean(standardized[failure], axis=0)
        gu_risk[i] = np.mean(standardized[risk], axis=0)
    matrix = (standardized[ordered] - gu_risk).T @ (gu - gu_risk)
    _, values, vt = np.linalg.svd(matrix, full_matrices=True)
    directions = inverse_sqrt @ vt.T
    directions /= np.linalg.norm(directions, axis=0)
    return dict(values=values, vectors=directions)


def hMave(x, y, censor, m0, B0=None):
    """Hazard MAVE using the algorithm in orthoDr's hMave.R.

    Returns a p-by-m0 direction matrix and the source hMAVE leave-one-out
    kernel score. The score sums squared errors in predicting Gaussian
    event-time-kernel values among observed events. Vector and one-column
    y/censor inputs are equivalent.
    """
    x = numeric_matrix(x, min_rows=3)
    n, p = x.shape
    y = numeric_vector(y, "y", n=n, allow_column=True)
    censor = numeric_vector(censor, "censor", n=n, allow_bool=True, allow_column=True)
    if not np.all(np.isin(censor, [0, 1])):
        raise ValueError("censor can only have two levels 0/1")
    m0 = positive_integer(m0, "m0", p)
    if np.std(y, ddof=1) == 0:
        raise ValueError("hMave requires varying observed survival times")
    if np.sum(censor == 1) < 2 or np.std(y[censor == 1], ddof=1) == 0:
        raise ValueError("hMave cross-validation requires at least two events with varying times")
    centered = x - np.mean(x, axis=0)
    values, vectors = eigh(centered.T @ centered / n)
    if np.any(values <= 0) or _centered_rank(x) < p:
        raise ValueError("hMave requires full centred predictor rank")
    ss = vectors / np.sqrt(values)
    x = centered @ ss
    if B0 is None:
        m, B = p, np.eye(p)
    else:
        B0 = numeric_matrix(B0, "B0")
        if B0.shape[0] != p or not m0 <= B0.shape[1] <= p:
            raise ValueError("Dimension of B0 is not correct")
        if np.linalg.matrix_rank(B0) < B0.shape[1]:
            raise ValueError("B0 must have full column rank")
        B = np.linalg.solve(ss, B0)
        m = B.shape[1]
    ba, bi = B.copy(), B.copy()
    ridge = np.std(y, ddof=1) * np.mean(np.std(x, axis=0, ddof=1))
    y = (y - np.mean(y)) / np.std(y, ddof=1)
    order = np.argsort(y, kind="stable")
    y, x, censor = y[order], x[order], censor[order]
    censor0 = censor.copy()
    ky2 = (y[:, None] - y[None, :]) ** 2
    for iteration in range(1, int(np.floor(p * 3 / 2)) + 1):
        adj = p ** (1 / iteration) * n ** (1 / (m0 + 4) - 1 / (m + 4))
        hy = np.std(y, ddof=1) / n ** (1 / 5) * p ** (1 / (iteration + 1))
        xb = x @ ba
        h = p ** (1 / (iteration + 1)) * np.mean(np.std(xb, axis=0, ddof=1)) / n ** (1 / (m + 4))
        h2 = 2 * h * h * adj
        hazard_denom = np.empty(n)
        for i in range(n):
            kernel = np.exp(-np.sum((xb - xb[i]) ** 2, axis=1) / h2)
            hazard_denom[i] = kernel @ ndtr((y[i] - y) / hy) / np.sum(kernel)
        ky = censor[:, None] * np.exp(-ky2 / (2 * hy * hy)) / hy / hazard_denom[None, :]
        abi_all = np.zeros((m, n * n))
        for _ in range(max(1, (m < p) * (m // 2))):
            dd, dc = np.zeros((m * p, m * p)), np.zeros(m * p)
            for j in range(n):
                xij = x - x[j]
                sxij = np.sum((xij @ ba) ** 2, axis=1) + np.sum(xij ** 2, axis=1) / 1.5 ** iteration
                kernel = np.exp(-sxij / h2)
                onexi = np.column_stack((xij @ B, np.ones(n)))
                xk = (onexi * kernel[:, None]).T
                abi = np.linalg.solve(xk @ onexi + np.eye(m + 1) / n, xk @ ky)
                kxij = (xij * kernel[:, None]).T
                kxijy = kxij @ (ky - abi[m][None, :])
                ddx = kxij @ xij
                for k1 in range(m):
                    dc[k1 * p:(k1 + 1) * p] += kxijy @ abi[k1]
                slopes = abi[:m]
                dd += np.kron(slopes @ slopes.T, ddx)
                abi_all[:, j * n:(j + 1) * n] = slopes
            coefficient = np.linalg.solve(dd + ridge * np.eye(m * p) / n, dc)
            B0_work = coefficient.reshape((p, m), order="F") @ abi_all
            _, eigenvectors = eigh(B0_work @ B0_work.T)
            B = eigenvectors[:, -m:]
            ba = B
            if np.linalg.norm(B @ B.T - bi @ bi.T, ord=2) < 0.001:
                break
            bi = B
        mb = m
        m = max(m - 1, m0)
        B = B[:, mb - m:mb]
        ba = B
        if np.linalg.norm(B @ B.T - bi @ bi.T, ord=2) < 0.001 and iteration > p + 3:
            break
        bi = ba
    h2 = 2 * n ** (-2 / (m0 + 3))
    x, y = x[censor0 == 1], y[censor0 == 1]
    n_event = len(y)
    scaled_y = (y - np.mean(y)) / np.std(y, ddof=1)
    ky = np.exp(-(scaled_y[:, None] - scaled_y[None, :]) ** 2 / h2)
    predicted = ky.copy()
    for j in range(n_event):
        kernel = np.exp(-np.sum(((x - x[j]) @ ba) ** 2, axis=1) / h2)
        kernel[j] = 0
        denominator = np.sum(kernel)
        if denominator <= 0 or not np.isfinite(denominator):
            raise ValueError("hMave leave-one-out kernel weights have zero total weight")
        predicted[j] = kernel @ ky / denominator
    directions = ss @ B
    directions /= np.linalg.norm(directions, axis=0)
    cv = float(np.sum((ky - predicted) ** 2))
    if not np.all(np.isfinite(directions)) or not np.isfinite(cv):
        raise ValueError("hMave produced a nonfinite result")
    return dict(B=directions, cv=cv)
