"""Gaussian prediction and product-limit calculations."""

import numpy as np


def _split_float(value):
    """Split an already bounded significand into high and low parts."""
    shifted = (2**27 + 1) * value
    high = shifted - (shifted - value)
    return high, value - high


def _compensated_projection(x, origin, basis):
    """Project differences with subtraction, product and sum residuals."""
    delta = x - origin
    back = x - delta
    residual = (x - (delta + back)) + (back - origin)

    # Scale only large terms before splitting their significands.
    bound = np.finfo(float).max / 2**28
    divisor = np.where(np.abs(delta) >= bound, 2**28, 1)
    small = delta / divisor
    upper, lower = _split_float(small)
    basis_upper, basis_lower = _split_float(basis)
    total = np.zeros((len(x), basis.shape[1]))
    error = np.zeros_like(total)
    for j in range(x.shape[1]):
        product = small[:, j, None] * basis[j]
        hi = upper[:, j, None]
        lo = lower[:, j, None]
        tail = ((hi * basis_upper[j] - product) + hi * basis_lower[j]
                + lo * basis_upper[j] + lo * basis_lower[j])
        factor = divisor[:, j, None]
        term = product * factor
        updated = total + term
        carried = updated - total
        error += (total - (updated - carried)) + (term - carried)
        error += tail * factor + residual[:, j, None] * basis[j]
        total = updated
    return total + error


def _scaled_distance(delta, peak, shape, bw, divisor=1):
    # Combine opposite scales to keep the standardized distance in range.
    denominator = peak * bw * shape / divisor
    if np.isfinite(denominator) and denominator >= np.finfo(float).tiny:
        return delta / denominator
    with np.errstate(divide="ignore"):
        log_scale = np.log(peak) + np.log(bw) + np.log(shape / divisor)
        return np.exp(np.log(np.abs(delta)) - log_scale)


def prediction_log_weights(x, testx, basis, bw):
    """Gaussian log weights in the raw-feature basis, using training scales."""
    basis = np.asarray(basis, dtype=float)
    if not np.isfinite(basis).all():
        raise ValueError("Training projection scales must be positive and finite")
    x = np.asarray(x, dtype=float)
    # Anchor before projection so within-feature differences survive translation.
    # Leave headroom for subtraction and the dot product at extreme scales.
    bound = 2 ** (4 * x.shape[1] - 1).bit_length()
    if np.max(np.abs(x)) > np.finfo(float).max / bound:
        x, testx = x / bound, testx / bound
    origin = x[0]
    centered = x - origin
    u = centered @ basis
    peak = np.max(np.abs(u), axis=0)
    magnitude = np.max(np.abs(centered), axis=0) @ np.abs(basis)
    eps = np.finfo(float).eps
    ill_conditioned = eps * magnitude > np.sqrt(eps) * peak
    if ill_conditioned.any():
        u[:, ill_conditioned] = _compensated_projection(x, origin, basis[:, ill_conditioned])
        peak = np.max(np.abs(u), axis=0)
    if np.any(~np.isfinite(peak) | (peak <= 0)):
        raise ValueError("Training projection scales must be positive and finite")
    shape = np.std(u / peak, axis=0, ddof=1) * np.sqrt(2)
    if np.any(~np.isfinite(shape) | (shape <= 0)):
        raise ValueError("Training projection scales must be positive and finite")

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        centered_queries = testx - origin
        v = centered_queries @ basis
        query_magnitude = magnitude + np.abs(centered_queries) @ np.abs(basis)
        direct = ill_conditioned.any() | ~np.isfinite(query_magnitude).all(axis=1)
        log_weights = np.zeros((len(testx), len(x)))
        for col in range(u.shape[1]):
            # If projected-coordinate rounding can affect distances at the
            # bandwidth, form each query-to-training difference first.
            rounding = _scaled_distance(eps * query_magnitude[:, col], peak[col], shape[col], bw)
            direct |= rounding > np.sqrt(eps)
            delta = v[:, col, None] - u[None, :, col]
            scaled = _scaled_distance(delta, peak[col], shape[col], bw)
            log_weights -= scaled ** 2
        for i in np.flatnonzero(direct):
            difference = _compensated_projection(x, testx[i], basis)
            divisor = 1
            if not np.isfinite(difference).all():
                divisor = bound
                difference = _compensated_projection(x / divisor, testx[i] / divisor, basis)
            log_weights[i] = 0
            for col in range(u.shape[1]):
                scaled = _scaled_distance(difference[:, col], peak[col], shape[col], bw, divisor)
                log_weights[i] -= scaled ** 2
    if np.isnan(log_weights).any():
        raise ValueError("Gaussian prediction log weights are invalid")
    return log_weights


def gaussian_mean(log_weights, y):
    """Evaluate the mean on a row with a positive absolute weight."""
    shifted = log_weights - np.max(log_weights)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        weights = np.exp(shifted)
        terms = weights * y
        # A relative weight can round to zero before multiplication by y.
        tiny = weights < np.finfo(float).tiny
        root = np.exp(shifted[tiny] / 2)
        terms[tiny] = (y[tiny] * root) * root
        total = weights.sum()
        numerator = terms.sum()
        if np.isfinite(numerator):
            return numerator / total
        scale = np.max(np.abs(y))
        mean = (terms / scale).sum() / total
        return np.clip(mean, -1, 1) * scale


def weighted_km(log_weights, times, event):
    """Weighted KM at distinct event and censoring times, in sorted order.

    Log weights are query by observation. The survival array is time by
    query, and ``blocks`` gives the time index of each sorted observation.
    """
    if np.any(np.isnan(log_weights) | (log_weights > 0)):
        raise ValueError("Invalid weighted survival log weights")
    order = np.argsort(times, kind="stable")
    times, event = times[order], event[order]
    new_time = np.r_[True, times[1:] != times[:-1]]
    starts = np.flatnonzero(new_time)
    blocks = np.cumsum(new_time) - 1
    if log_weights.shape[0] == 0:
        return order, blocks, np.empty((len(starts), 0))

    log_weights = log_weights[:, order].T
    peak = log_weights.max(axis=0)
    if np.isfinite(peak).all() and np.all(log_weights.min(axis=0) - peak >= -512):
        # With this log range, every positive mass and every finite ratio
        # remains representable on one common scale, even late in follow-up.
        weights = np.exp(log_weights - peak)
        deaths = np.add.reduceat(weights * event[:, None], starts, axis=0)
        censored = np.add.reduceat(weights * (~event[:, None]), starts, axis=0)
        later = np.cumsum((deaths + censored)[::-1], axis=0)[::-1]
        alive = censored.copy()
        alive[:-1] += later[1:]
        ratio = np.zeros_like(deaths)
        with np.errstate(divide="ignore"):
            np.divide(deaths, alive, out=ratio, where=deaths > 0)
        return order, blocks, np.exp(-np.cumsum(np.log1p(ratio), axis=0))

    def block_mass(mask):
        logs = np.where(mask[:, None], log_weights, -np.inf)
        peak = np.maximum.reduceat(logs, starts, axis=0)
        reference = np.where(np.isfinite(peak), peak, 0)
        with np.errstate(under="ignore"):
            mass = np.add.reduceat(np.exp(logs - reference[blocks]), starts, axis=0)
        return peak, mass

    event_peak, event_mass = block_mass(event)
    censor_peak, censor_mass = block_mass(~event)
    has_event = np.logical_or.reduceat(event, starts)
    has_censor = np.logical_or.reduceat(~event, starts)
    n_times, n_queries = event_mass.shape
    log_factors = np.zeros((n_times, n_queries))
    later_peak = np.full(n_queries, -np.inf)
    later_mass = np.zeros(n_queries)

    def merge(peak_a, mass_a, peak_b, mass_b):
        peak = np.maximum(peak_a, peak_b)
        reference = np.where(np.isfinite(peak), peak, 0)
        mass = (mass_a * np.exp(peak_a - reference)
                + mass_b * np.exp(peak_b - reference))
        return peak, mass

    with np.errstate(under="ignore", over="ignore", divide="ignore"):
        for i in range(n_times - 1, -1, -1):
            if has_censor[i]:
                later_peak, later_mass = merge(later_peak, later_mass, censor_peak[i], censor_mass[i])
            if not has_event[i]:
                continue
            if i == n_times - 1 and not has_censor[i]:
                log_factors[i] = -np.inf
            else:
                if np.any(~np.isfinite(np.maximum(event_peak[i], later_peak))):
                    raise ValueError("Gaussian log distances overflowed in a mixed survival risk set; increase bw")
                # Compute the ratio before adding log masses. Adding to a
                # very negative peak first can erase the count of ties.
                ratio = (event_peak[i] - later_peak
                         + np.log(event_mass[i]) - np.log(later_mass))
                log_factors[i] = -np.maximum(ratio, 0) - np.log1p(np.exp(-np.abs(ratio)))
            later_peak, later_mass = merge(later_peak, later_mass, event_peak[i], event_mass[i])
        survival = np.exp(np.cumsum(log_factors, axis=0))
    return order, blocks, survival
