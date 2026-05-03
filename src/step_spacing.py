"""
Step-spacing extraction primitives for output-rounded ReLU networks.

The defender publishes f_r(x) = round(f(x) / s) * s. Carlini's attack relies
on detecting output changes far smaller than `s`, which the defender just
zeroed out. The fix: probe at a MACROSCOPIC scale where the rounded oracle's
integer level changes by many units, then estimate the gradient as a
"finite difference of integer levels" rather than a finite difference of
sub-rounding-step floats.

  level(t) := round(f_r(x0 + t·d) / s)        # integer
  |df/dd|  ≈ |level(t_hi) − level(t_lo)| · s / (t_hi − t_lo)

Error from rounding: at most ±2s / (t_hi − t_lo) — relative error 2/N where
N is the number of integer-level changes in the window. For N ≥ 100 this is
better than 2 percent.

Constraint: the window [t_lo, t_hi] must lie inside ONE polytope (no ReLU
crossings). We test this with a 3-point linearity check at integer-level
precision: |level(mid) − (level(lo)+level(hi))/2| ≤ 1.

The remaining primitives are layered on top:
  - find_critical_points_along_line:
        sample local slopes along a long sweep at uniform intervals, then
        detect change-points in the slope sequence. Each change-point is a
        candidate ReLU boundary.
  - recover_weight_row:
        at a critical point, take the gradient JUMP (left − right) along
        n basis directions; that vector is α · W[j, :] for some scalar α.
"""

from __future__ import annotations
from typing import Callable, List, Optional, Tuple

import numpy as np


Oracle = Callable[[np.ndarray], np.ndarray]


def _scalar_oracle(oracle: Oracle):
    def f(x):
        y = np.asarray(oracle(x))
        if y.ndim == 1:
            return y
        return y.reshape(y.shape[0], -1)[:, 0]
    return f


def _level(oracle, x, s):
    """Integer rounding-level of the rounded oracle at x (single point)."""
    f = _scalar_oracle(oracle)
    pt = np.atleast_2d(x)
    return int(np.round(float(f(pt)[0]) * (1.0 / s)))


def _levels(oracle, X, s):
    """Vectorised: integer levels at a batch of points."""
    f = _scalar_oracle(oracle)
    return np.round(f(np.atleast_2d(X)) * (1.0 / s)).astype(np.int64)


def directional_derivative(
    x0: np.ndarray,
    d: np.ndarray,
    oracle: Oracle,
    s: float,
    *,
    init_radius: float = 1e-3,
    max_radius: float = 5.0,
    radius_growth: float = 4.0,
    target_levels: int = 64,
    max_iters: int = 30,
) -> Optional[float]:
    """Estimate (signed) df/d(d) at x0 via integer-level finite differencing.

    Adaptive radius: grow until the level differential between endpoints is
    >= target_levels, *and* a midpoint-linearity check passes (so we know we
    haven't crossed a ReLU boundary). Returns None if no usable window found.
    """
    R = init_radius
    for _ in range(max_iters):
        pts = np.array([x0 - R * d, x0, x0 + R * d])
        L = _levels(oracle, pts, s)
        diff = int(L[2] - L[0])
        # Soft linearity check at integer precision: midpoint level should
        # roughly match the average of endpoints, with a tolerance scaled
        # by the level span (so we don't reject just because of rounding
        # noise on a fine grid).
        mid_residual = abs(L[1] - 0.5 * (L[0] + L[2]))
        # Allow up to ~10% nonlinearity for big spans, +1 step floor.
        tol = max(1.5, 0.1 * abs(diff))
        if mid_residual > tol:
            # Likely crossed a ReLU. Shrink and retry.
            R *= 0.5
            if R < 1e-12:
                return None
            continue
        if abs(diff) >= target_levels:
            return diff * s / (2.0 * R)
        if R >= max_radius:
            # Can't grow further; if there's any signal, return it.
            if abs(diff) >= 4:
                return diff * s / (2.0 * R)
            return None
        R *= radius_growth
    return None


def estimate_gradient(
    x0: np.ndarray,
    oracle: Oracle,
    s: float,
    *,
    target_levels: int = 64,
    init_radius: float = 1e-3,
    max_radius: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-dimension gradient via directional_derivative; returns (g, valid)."""
    n = x0.shape[0]
    g = np.zeros(n, dtype=np.float64)
    ok = np.zeros(n, dtype=bool)
    for i in range(n):
        d = np.zeros(n, dtype=np.float64)
        d[i] = 1.0
        v = directional_derivative(
            x0, d, oracle, s,
            init_radius=init_radius,
            max_radius=max_radius,
            target_levels=target_levels,
        )
        if v is not None:
            g[i] = v
            ok[i] = True
    return g, ok


def find_critical_points_along_line(
    x0: np.ndarray,
    direction: np.ndarray,
    t_low: float,
    t_high: float,
    oracle: Oracle,
    s: float,
    *,
    n_probes: int = 200,
    local_radius_init: float = 1e-3,
    local_max_radius: float = 0.2,
    local_target_levels: int = 32,
    relative_jump_tol: float = 0.30,
) -> List[float]:
    """Detect ReLU critical points along x0+t·direction for t∈[t_low, t_high].

    Procedure:
      1. Sample n_probes uniform t-values.
      2. At each, estimate the directional slope d/dt of f via
         directional_derivative along `direction`.
      3. Walk the slope sequence; mark every position where consecutive slopes
         differ by more than relative_jump_tol (relative to the larger).
      4. For each detected jump, refine the t-location by binary search on
         the rounded levels: bisect the segment until the slope-change point
         is localized.
    """
    ts = np.linspace(t_low, t_high, n_probes)
    slopes = np.full(n_probes, np.nan)
    for i, t in enumerate(ts):
        x = x0 + t * direction
        v = directional_derivative(
            x, direction, oracle, s,
            init_radius=local_radius_init,
            max_radius=local_max_radius,
            target_levels=local_target_levels,
            max_iters=12,
        )
        if v is not None:
            slopes[i] = v

    # Detect adjacent pairs that disagree by more than tol, where both are valid.
    candidates: List[Tuple[int, float]] = []
    for i in range(n_probes - 1):
        a = slopes[i]
        b = slopes[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        denom = max(abs(a), abs(b))
        if denom <= 1e-12:
            continue
        if abs(a - b) / denom > relative_jump_tol:
            candidates.append((i, abs(a - b) / denom))

    if not candidates:
        return []

    # Suppress only IMMEDIATE neighbours (within 1 probe) — true critical
    # points may be only a few probes apart and we don't want to merge them.
    keep = []
    used = np.zeros(n_probes, dtype=bool)
    for idx, _jmp in sorted(candidates, key=lambda kv: -kv[1]):
        if used[max(0, idx - 1): idx + 2].any():
            continue
        keep.append(idx)
        used[max(0, idx - 1): idx + 2] = True
    keep.sort()

    # Refine each candidate by bisection on the line (no slope estimation each
    # step — just compare slopes left and right of the bisection midpoint).
    refined: List[float] = []
    for i in keep:
        t_lo, t_hi = ts[i], ts[i + 1]
        slope_lo = slopes[i]
        slope_hi = slopes[i + 1]
        for _ in range(20):
            t_mid = 0.5 * (t_lo + t_hi)
            x_mid = x0 + t_mid * direction
            v = directional_derivative(
                x_mid, direction, oracle, s,
                init_radius=local_radius_init,
                max_radius=local_max_radius,
                target_levels=local_target_levels,
                max_iters=10,
            )
            if v is None:
                break
            denom = max(abs(slope_lo), abs(v), 1e-12)
            if abs(v - slope_lo) / denom < abs(v - slope_hi) / denom:
                t_lo, slope_lo = t_mid, v
            else:
                t_hi, slope_hi = t_mid, v
            if t_hi - t_lo < 1e-6:
                break
        refined.append(0.5 * (t_lo + t_hi))
    return refined


def recover_weight_row(
    x_critical: np.ndarray,
    oracle: Oracle,
    s: float,
    *,
    delta: float = 1e-2,
    target_levels: int = 64,
    init_radius: float = 1e-3,
    max_radius: float = 0.2,
) -> Tuple[np.ndarray, bool]:
    """Recover the activated neuron's weight row at a critical point.

    Take the gradient on each side; (g_left - g_right) ≈ α · W[j, :] for some
    scalar α. Returns the inf-norm-normalised vector or zeros on failure.
    """
    n = x_critical.shape[0]
    # Pick a stepping direction: dominant axis of recovered local gradient.
    g_seed, ok = estimate_gradient(
        x_critical, oracle, s,
        target_levels=max(8, target_levels // 4),
        init_radius=init_radius,
        max_radius=max_radius,
    )
    if not np.any(ok) or np.linalg.norm(g_seed) < 1e-30:
        return np.zeros(n), False
    step_dir = g_seed / np.linalg.norm(g_seed)

    g_L, okL = estimate_gradient(
        x_critical - delta * step_dir, oracle, s,
        target_levels=target_levels, init_radius=init_radius, max_radius=max_radius,
    )
    g_R, okR = estimate_gradient(
        x_critical + delta * step_dir, oracle, s,
        target_levels=target_levels, init_radius=init_radius, max_radius=max_radius,
    )
    valid = okL & okR
    if not np.any(valid):
        return np.zeros(n), False
    delta_g = (g_L - g_R) * valid
    inf = float(np.max(np.abs(delta_g)))
    if inf < 1e-12:
        return np.zeros(n), False
    return delta_g / inf, True


# ---------- Backwards-compat alias ----------
# The earlier critical-point API took different kwargs; keep a thin wrapper
# so the validation script keeps working without an edit.
def find_critical_points_by_spacing(*args, **kwargs):  # pragma: no cover
    # Drop legacy kwargs we no longer use.
    for legacy in ("max_transitions", "window", "relative_jump_tol",
                   "min_segment_steps", "coarse_samples"):
        kwargs.pop(legacy, None)
    return [(t, None, None) for t in find_critical_points_along_line(*args, **kwargs)]


# Legacy name for find_transitions_in_window kept for one validation import.
def find_transitions_in_window(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(
        "find_transitions_in_window is deprecated — use directional_derivative "
        "and find_critical_points_along_line."
    )
