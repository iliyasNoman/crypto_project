#!/usr/bin/env python3
"""
End-to-end extraction of a 2-layer ReLU network from a *rounded* black-box
oracle, using gradient-vector change-point detection instead of fragile
directional slope-change detection.

Pipeline:
  1. Pick a random 1D sweep direction d.
  2. Sample n_probes uniform t-values along x(t) = x0 + t*d.
  3. At each probe, estimate the FULL gradient ∇f via integer-level finite
     differencing (src/step_spacing.py::estimate_gradient).
  4. Consecutive gradient differences are candidates for the weight row of
     a flipped neuron: g(x_{i+1}) - g(x_i) ≈ ± W2[j] * W1[j, :] when only
     neuron j flips between probes i and i+1.
  5. Aggregate candidates from many sweeps; cluster by cosine similarity into
     n_hidden groups; one canonical row per cluster.
  6. Recover layer-1 biases from cluster locations.
  7. Sign recovery: brute-force or meet-in-the-middle.
  8. Solve final layer by least squares.

Restricted to architectures with a single hidden layer (input - hidden - 1).
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import Callable, List, Tuple

import numpy as np

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)

from src.step_spacing import estimate_gradient
from src.mitm_signs import recover_signs_brute, recover_signs_mitm


# ---------- Reference forward pass / oracle helpers ----------

def forward(x, A, B, sizes):
    h = x
    for i, (a, b) in enumerate(zip(A, B)):
        h = h @ a + b
        if i < len(sizes) - 2:
            h = h * (h > 0)
    return h


def make_rounded_oracle(A, B, sizes, bits):
    s = 2.0 ** (1 - bits) if bits < 64 else 0.0
    def oracle(x):
        x = np.atleast_2d(x)
        y = forward(x, A, B, sizes)
        if s > 0:
            y = np.round(y / s) * s
        return y
    return oracle, s


# ---------- Phase 1: sweep gradients and collect jump candidates ----------

def canonicalise_row(r):
    """Flip sign so the largest-abs entry is positive."""
    j = int(np.argmax(np.abs(r)))
    if r[j] < 0:
        return -r
    return r


def collect_jump_candidates(
    oracle: Callable,
    s: float,
    n_in: int,
    n_sweeps: int = 30,
    n_probes: int = 400,
    sweep_extent: float = 2.0,
    grad_target_levels: int = 64,
    grad_init_radius: float = 1e-3,
    grad_max_radius: float = 0.05,
    jump_threshold: float = 0.05,
    rng: np.random.Generator = None,
):
    """Sweep many random lines, estimate full gradient at each probe, collect
    non-trivial consecutive differences as candidate weight rows.

    Returns list of (row_canonical, t_mid, x_mid).
    """
    rng = rng or np.random.default_rng(0)
    candidates: List[Tuple[np.ndarray, float, np.ndarray]] = []

    for _ in range(n_sweeps):
        x0 = rng.standard_normal(size=n_in) * 0.5
        d = rng.standard_normal(size=n_in)
        d = d / np.linalg.norm(d)
        ts = np.linspace(-sweep_extent, sweep_extent, n_probes)

        prev_g = None
        prev_ok = None
        prev_t = None
        for t in ts:
            x = x0 + t * d
            g, ok = estimate_gradient(
                x, oracle, s,
                target_levels=grad_target_levels,
                init_radius=grad_init_radius,
                max_radius=grad_max_radius,
            )
            # Need at least half the dims valid to do anything useful.
            if ok.sum() < n_in // 2:
                prev_g = None
                prev_ok = None
                prev_t = None
                continue
            if prev_g is not None:
                shared = ok & prev_ok
                if shared.sum() < max(4, n_in // 4):
                    prev_g, prev_ok, prev_t = g, ok, t
                    continue
                jump = (g - prev_g) * shared
                jump_norm = np.linalg.norm(jump)
                ref = max(
                    np.linalg.norm(g[shared]),
                    np.linalg.norm(prev_g[shared]),
                    1e-12,
                )
                if jump_norm / ref > jump_threshold:
                    # Zero out dims that weren't shared (so cluster cosine
                    # ignores them by construction; then renormalise valid
                    # entries when comparing).
                    jump_full = np.zeros_like(jump)
                    jump_full[shared] = jump[shared]
                    t_mid = 0.5 * (t + prev_t)
                    x_mid = x0 + t_mid * d
                    candidates.append((canonicalise_row(jump_full), t_mid, x_mid))
            prev_g, prev_ok, prev_t = g, ok, t

    return candidates


# ---------- Phase 2: cluster candidate rows ----------

def cluster_rows(candidates, k: int, sim_threshold: float = 0.92):
    """Greedy cosine-similarity clustering."""
    if len(candidates) == 0:
        return None, None, None
    rows = np.array([r for (r, _, _) in candidates])
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms = np.where(norms < 1e-30, 1.0, norms)
    rows_n = rows / norms

    clusters: List[List[int]] = []
    for i in range(len(candidates)):
        placed = False
        for c in clusters:
            sim = float(rows_n[c[0]] @ rows_n[i])
            if abs(sim) > sim_threshold:
                c.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])

    clusters.sort(key=lambda c: -len(c))
    if len(clusters) < k:
        return None, None, None
    clusters = clusters[:k]

    medians = []
    locs_per_cluster = []
    for c in clusters:
        rs = np.array([candidates[i][0] for i in c])
        rep = rs[0]
        signs = np.sign(rs @ rep)
        signs = np.where(signs == 0, 1, signs)
        rs = rs * signs[:, None]
        medians.append(np.median(rs, axis=0))
        locs_per_cluster.append([candidates[i][2] for i in c])
    return np.array(medians), locs_per_cluster, clusters


def recover_biases(W1_rows, locs_per_cluster):
    """For each neuron j with row w_j: find b_j such that w_j · x_crit + b_j ≈ 0.
    Take median over the cluster's critical-point locations.
    """
    biases = []
    for row, locs in zip(W1_rows, locs_per_cluster):
        bs = [-row @ x for x in locs]
        biases.append(float(np.median(bs)))
    return np.array(biases)


# ---------- Phase 3: solve final layer by least squares ----------

def solve_final_layer(W1, b1, oracle, n_queries: int, n_in: int, rng):
    X = rng.standard_normal(size=(n_queries, n_in))
    Y = oracle(X).reshape(-1)
    pre = X @ W1 + b1
    H = pre * (pre > 0)
    H_aug = np.concatenate([H, np.ones((H.shape[0], 1))], axis=1)
    sol, *_ = np.linalg.lstsq(H_aug, Y, rcond=None)
    W2 = sol[:-1].reshape(-1, 1)
    b2 = sol[-1:]
    return W2, b2


# ---------- Quality evaluation ----------

def evaluate(A_real, B_real, A_hat, B_hat, sizes, n_eval=20000, rng=None):
    rng = rng or np.random.default_rng(0)
    X = rng.standard_normal(size=(n_eval, sizes[0]))
    y_real = forward(X, A_real, B_real, sizes).reshape(-1)
    y_hat = forward(X, A_hat, B_hat, sizes).reshape(-1)
    diff = np.abs(y_real - y_hat)
    return {
        "max_logit_loss": float(np.max(diff)),
        "mean_logit_loss": float(np.mean(diff)),
        "rel_loss": float(np.mean(diff) / max(np.mean(np.abs(y_real)), 1e-30)),
    }


# ---------- Main extraction ----------

def extract_two_layer(
    oracle, sizes, s, *,
    n_sweeps: int = 30,
    n_probes: int = 400,
    sweep_extent: float = 2.0,
    grad_target_levels: int = 64,
    grad_init_radius: float = 1e-3,
    grad_max_radius: float = 0.05,
    jump_threshold: float = 0.05,
    sim_threshold: float = 0.92,
    use_mitm: bool = True,
    seed: int = 0,
    verbose: bool = True,
):
    n_in, n_hid, n_out = sizes
    assert n_out == 1, "Only single-output supported"
    rng = np.random.default_rng(seed)
    t0 = time.time()

    if verbose:
        print(f"[extract] sizes={sizes} s={s:.3e} n_sweeps={n_sweeps} n_probes={n_probes}")

    cands = collect_jump_candidates(
        oracle, s, n_in,
        n_sweeps=n_sweeps, n_probes=n_probes, sweep_extent=sweep_extent,
        grad_target_levels=grad_target_levels,
        grad_init_radius=grad_init_radius, grad_max_radius=grad_max_radius,
        jump_threshold=jump_threshold, rng=rng,
    )
    if verbose:
        print(f"[extract] {len(cands)} jump candidates from {n_sweeps} sweeps "
              f"(t={time.time()-t0:.1f}s)")
    if len(cands) < n_hid:
        if verbose:
            print("[extract] FAIL: too few jump candidates")
        return None

    W1_can, locs_per, clusters = cluster_rows(cands, n_hid,
                                              sim_threshold=sim_threshold)
    if W1_can is None:
        if verbose:
            print("[extract] FAIL: clustering didn't yield n_hid groups")
        return None
    W1 = W1_can.T  # shape (n_in, n_hid)
    b1 = recover_biases(W1.T, locs_per)
    cluster_sizes = [len(c) for c in clusters]
    if verbose:
        print(f"[extract] clustered into {n_hid} neurons, sizes={cluster_sizes} "
              f"(t={time.time()-t0:.1f}s)")

    # Sign recovery
    if use_mitm:
        signs = recover_signs_mitm(oracle, W1, b1, s, n_probes=200, rng=rng)
    else:
        signs = recover_signs_brute(oracle, W1, b1, s, n_probes=200, rng=rng)
    if signs is None:
        if verbose:
            print("[extract] FAIL: sign recovery failed")
        return None
    W1 = W1 * signs[None, :]
    b1 = b1 * signs

    if verbose:
        print(f"[extract] signs recovered (t={time.time()-t0:.1f}s)")

    W2, b2 = solve_final_layer(W1, b1, oracle, n_queries=2000,
                               n_in=n_in, rng=rng)
    if verbose:
        print(f"[extract] DONE (t={time.time()-t0:.1f}s)")

    return [W1, W2], [b1, b2]


# ---------- CLI ----------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arch")
    p.add_argument("seed", type=int)
    p.add_argument("bits", type=int)
    p.add_argument("--n-sweeps", type=int, default=30)
    p.add_argument("--n-probes", type=int, default=400)
    p.add_argument("--no-mitm", action="store_true")
    args = p.parse_args()

    sizes = list(map(int, args.arch.split("-")))
    if len(sizes) != 3:
        print("Only 2-layer architectures supported", file=sys.stderr)
        sys.exit(2)

    path = os.path.join(REPO_DIR, "models", f"{args.seed}_{args.arch}.npy")
    params = np.load(path, allow_pickle=True)
    A = [np.array(a, dtype=np.float64) for a in params[0]]
    B = [np.array(b, dtype=np.float64) for b in params[1]]

    oracle, s = make_rounded_oracle(A, B, sizes, args.bits)
    if s == 0:
        # Pretend baseline = no-op rounding for the same code path.
        s = 1e-30

    out = extract_two_layer(
        oracle, sizes, s,
        n_sweeps=args.n_sweeps, n_probes=args.n_probes,
        use_mitm=not args.no_mitm, seed=args.seed,
    )
    if out is None:
        print("Extraction FAILED")
        sys.exit(1)
    A_hat, B_hat = out
    metrics = evaluate(A, B, A_hat, B_hat, sizes)
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
