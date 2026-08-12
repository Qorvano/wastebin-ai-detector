"""Color math: vectorized float RGB→HSV and circular hue statistics.

PIL's built-in ``convert("HSV")`` is 8-bit quantized (≈1.41°/step) and
assigns hue 0 (red) to grey pixels. Both properties poison learned
color statistics, so this module implements the standard float
conversion itself and marks undefined hue as NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import CalibrationError

# Guard against division by zero in the HSV formulas. Purely a
# floating-point epsilon: far below one 8-bit quantum (1/255 ≈ 3.9e-3),
# so it can never reclassify a real color - it only protects 0/0.
_EPS = 1e-12

# Structural constants of the input encoding, not tuning values:
# hue is defined as 60° per RGB sextant, and source images are 8-bit.
# Together they give the hue quantization of a pixel with chroma c:
# one 8-bit step in a channel shifts hue by 60° / (255 · c).
HUE_DEG_PER_SEXTANT = 60.0
RGB_8BIT_LEVELS = 255.0


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert an (..., 3) float array in [0, 1] to HSV.

    Returns ``(hue_deg, sat, val)`` where ``hue_deg`` is in [0, 360) and
    NaN wherever chroma is 0: hue is mathematically undefined for grey
    pixels, and they must not enter circular hue statistics.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim < 1 or rgb.shape[-1] != 3:
        raise ValueError(f"expected (..., 3) RGB array, got shape {rgb.shape}")
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    val = maxc
    chroma = maxc - minc
    sat = np.where(maxc > _EPS, chroma / np.where(maxc > _EPS, maxc, 1.0), 0.0)
    safe_chroma = np.where(chroma > _EPS, chroma, 1.0)
    h_r = ((g - b) / safe_chroma) % 6.0
    h_g = (b - r) / safe_chroma + 2.0
    h_b = (r - g) / safe_chroma + 4.0
    hue = np.select([maxc == r, maxc == g], [h_r, h_g], default=h_b) * 60.0
    hue = np.where(chroma > _EPS, hue % 360.0, np.nan)
    return hue, sat, val


def circular_mean_deg(
    hues_deg: np.ndarray, weights: np.ndarray | None = None
) -> tuple[float, float]:
    """Circular mean of hue angles in degrees, ignoring NaN entries.

    Returns ``(mean_deg, resultant_length)``. The resultant length R in
    [0, 1] measures concentration: R→1 means tightly clustered hues,
    R→0 means the angles cancel out (no meaningful mean direction).
    Optional ``weights`` (same shape as ``hues_deg``) give a weighted
    mean, used by the mixture learner's M-step.
    Raises :class:`CalibrationError` if no valid (non-NaN) hue exists.
    """
    hues = np.asarray(hues_deg, dtype=np.float64).ravel()
    mask = ~np.isnan(hues)
    valid = hues[mask]
    if valid.size == 0:
        raise CalibrationError(
            "no valid hue pixels in sample (all grey/unsaturated) - the "
            "sample rectangle must cover a colored surface. Grey/black "
            "lids cannot be color-calibrated; attach a small colored "
            "marker to the lid and sample that instead"
        )
    rad = np.deg2rad(valid)
    if weights is None:
        x = float(np.cos(rad).mean())
        y = float(np.sin(rad).mean())
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()[mask]
        total = float(w.sum())
        if total <= 0.0:
            raise CalibrationError("weighted circular mean of zero weights")
        x = float((w * np.cos(rad)).sum() / total)
        y = float((w * np.sin(rad)).sum() / total)
    resultant = float(np.hypot(x, y))
    mean = float(np.rad2deg(np.arctan2(y, x))) % 360.0
    # Float-modulo edge: a tiny negative angle (hues symmetric around
    # the 0°/360° red wraparound) yields 360 − tiny, which rounds UP to
    # exactly 360.0 because ulp(360) ≈ 5.7e-14 exceeds the remainder.
    # Fold it back onto the documented [0, 360) interval.
    if mean >= 360.0:
        mean = 0.0
    return mean, resultant


def circular_dist_deg(hues_deg: np.ndarray, center_deg: float) -> np.ndarray:
    """Absolute circular distance in degrees, in [0, 180]; NaN propagates.

    The ``((h − c + 180) mod 360) − 180`` form handles the 0°/360° red
    wraparound without special cases.
    """
    hues = np.asarray(hues_deg, dtype=np.float64)
    return np.abs(((hues - center_deg + 180.0) % 360.0) - 180.0)


def vonmises_kappa_from_resultant(resultant: float) -> float:
    """Concentration parameter kappa from a resultant length.

    Closed-form approximation by Banerjee et al. (2005),
    ``kappa = R(2 - R^2) / (1 - R^2)``: one expression over the whole
    range, no piecewise breakpoints. The R→1 pole is the caller's
    business (kappa is capped at the quantization resolution there).
    """
    r = float(resultant)
    if r <= 0.0:
        return 0.0
    if r >= 1.0:
        return float("inf")
    return r * (2.0 - r * r) / (1.0 - r * r)


# np.i0 overflows once its result exceeds the float64 range; the
# switch point derives from the asymptotic form i0(x) ≈ e^x/sqrt(2πx),
# so log(i0) stays representable up to log(float64 max) ≈ 709.
_LOG_I0_SWITCH = float(np.log(np.finfo(np.float64).max))
_LOG_2PI = float(np.log(2.0 * np.pi))


def _log_i0(x: np.ndarray | float) -> np.ndarray:
    """log of the modified Bessel function I0, overflow-safe."""
    x = np.asarray(x, dtype=np.float64)
    small = x < _LOG_I0_SWITCH
    out = np.empty_like(x)
    out[small] = np.log(np.i0(x[small]))
    # Asymptotic expansion for large arguments (standard first term).
    big = ~small
    out[big] = x[big] - 0.5 * np.log(2.0 * np.pi * x[big])
    return out


@dataclass
class MixtureFit:
    """Result of the von Mises + uniform mixture fit."""

    center_deg: float
    kappa: float
    weight: float  # mixture weight of the von Mises (coherent) part
    posterior: np.ndarray  # per-pixel probability of being coherent
    log_likelihood: float
    n_iter: int


def fit_vonmises_uniform_mixture(
    hues_deg: np.ndarray,
    *,
    init_center_deg: float,
    init_weight: float,
    init_kappa: float,
    kappa_max: float,
) -> MixtureFit:
    """EM fit of a von Mises + circular-uniform mixture.

    Separates a coherent color population from junk pixels (overexposed
    highlights, edge artifacts) whose hues scatter over the circle. The
    posterior classifies every pixel; the caller thresholds it at the
    Bayes boundary 0.5.

    Termination: the loop breaks once the likelihood gain falls to (or
    below) the floating-point resolution of the likelihood, which also
    catches decreases (the closed-form kappa step is approximate, so
    strict EM monotonicity does NOT hold). As a hard bound against slow
    oscillation within that tolerance, iterations are capped at n: an
    EM on n points that has not settled after n rounds is circling
    inside numerical noise, and the best-likelihood state seen so far
    is returned.
    """
    hues = np.asarray(hues_deg, dtype=np.float64).ravel()
    hues = hues[~np.isnan(hues)]
    n = hues.size
    if n == 0:
        raise CalibrationError("mixture fit without valid hue pixels")
    if n == 1:
        # A single pixel is trivially its own coherent population; the
        # generic path would invert the weight clip bounds (floor 1/n
        # above ceiling 1 - 1/n).
        center = float(hues[0]) % 360.0
        if center >= 360.0:
            center = 0.0
        return MixtureFit(
            center_deg=center,
            kappa=kappa_max,
            weight=1.0,
            posterior=np.ones(1),
            log_likelihood=0.0,
            n_iter=0,
        )
    rad = np.deg2rad(hues)
    # A component weight of zero can never recover in EM; one pixel of
    # mass is the smallest meaningful floor.
    w_floor = 1.0 / n
    weight = float(np.clip(init_weight, w_floor, 1.0 - w_floor))
    mu = float(np.deg2rad(init_center_deg))
    kappa = float(np.clip(init_kappa, 0.0, kappa_max))
    log_uniform = -_LOG_2PI
    prev_ll = None
    best: tuple[float, float, float, float, np.ndarray] | None = None
    n_iter = 0
    while n_iter < n:
        n_iter += 1
        log_vm = kappa * np.cos(rad - mu) - _LOG_2PI - _log_i0(kappa)
        log_num = np.log(weight) + log_vm
        log_den = np.logaddexp(log_num, np.log(1.0 - weight) + log_uniform)
        posterior = np.exp(log_num - log_den)
        ll = float(log_den.sum())
        if best is None or ll > best[0]:
            best = (ll, mu, kappa, weight, posterior)
        if prev_ll is not None:
            # FP-derived stop: gains at or below the representable
            # resolution of the accumulated likelihood are numerical
            # noise; this also breaks on a likelihood decrease.
            if ll - prev_ll <= max(abs(ll), 1.0) * np.finfo(np.float64).eps * n:
                break
        prev_ll = ll
        # M-step: weighted circular statistics of the coherent part.
        mean_deg, resultant_w = circular_mean_deg(hues, weights=posterior)
        mu = float(np.deg2rad(mean_deg))
        total = float(posterior.sum())
        weight = float(np.clip(total / n, w_floor, 1.0 - w_floor))
        kappa = float(
            np.clip(vonmises_kappa_from_resultant(resultant_w), 0.0, kappa_max)
        )
    ll, mu, kappa, weight, posterior = best  # type: ignore[misc]
    center = float(np.rad2deg(mu)) % 360.0
    if center >= 360.0:
        center = 0.0
    return MixtureFit(
        center_deg=center,
        kappa=kappa,
        weight=weight,
        posterior=posterior,
        log_likelihood=ll,
        n_iter=n_iter,
    )
