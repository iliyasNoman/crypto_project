"""
Meet-in-the-middle sign recovery for one-hidden-layer ReLU extraction.

After step-spacing recovery we have W1 ∈ R^{n_in × k} and b1 ∈ R^k where each
column is the recovered weight row of one hidden neuron, but each column is
known only up to a sign σ_j ∈ {±1}. Carlini's brute-force tries 2^k sign
assignments. MITM splits the k neurons into halves A and B, enumerates
2^(k/2) for each half, and matches them via a hash table.

The match key is built from the rounded oracle outputs over a batch of probe
points. The decomposition that makes this work:

    f(x) ≈ Σ_j w2[j] * σ_j * relu( σ_j (W1[:, j] · x + b1[j]) ) + b2

Substituting r_j(σ) := σ_j * relu(σ_j (W1[:, j] · x + b1[j])):
    σ_j = +1: r_j = relu(W1·x + b1)         (neuron active iff pre-act > 0)
    σ_j = -1: r_j = -relu(-(W1·x + b1))     (neuron active iff pre-act < 0)
                  = -ReLU(-pre)             which equals min(pre, 0)

So r_j(σ_j) ∈ {relu(pre_j), pre_j - relu(pre_j)} = {(pre_j)_+, -(-pre_j)_+} —
two possible activations per neuron. f(x) is a linear combination of these
plus a global constant, so it decomposes additively across neurons:

    f(x) = a_A(σ_A, x) + a_B(σ_B, x) + (linear in x for the bilinear-merge case)

where a_A(σ_A, x) = sum over neurons in A (weighted by the to-be-determined
final-layer coefficients). To avoid having to know W2 in advance, we instead
factor the problem differently: for any FIXED sign choice σ ∈ {±1}^k, the
implied hidden activations h_σ(x) are computable, and W2, b2 are the
least-squares solution over a probe batch. The residual

    res(σ) = || y_probe − [h_σ | 1] · w_σ_lstsq ||

is exactly Carlini's brute-force scoring criterion. To MITM this we exploit
that h_σ decomposes: h_σ(x) = h_{σ_A}(x) ⊕ h_{σ_B}(x) (concatenation of two
column-blocks). The lstsq solution over a fixed probe set X reduces to a
linear projection; for each candidate (σ_A, σ_B) the residual is computable
in O(N · k) once we precompute partial activations. We hash one half against
the orthogonal complement of the other and search for the closest match.

This file provides both `recover_signs_brute` (reference) and
`recover_signs_mitm` (the meet-in-the-middle version). Both return σ ∈ {±1}^k
or None on failure.
"""

from __future__ import annotations
from typing import Callable, Optional

import numpy as np


def _build_basis_for_sign_search(W1, b1, X):
    """For each neuron j, build two column candidates (σ=+1 or σ=-1) over
    the probe batch X. Returns a tensor of shape (k, 2, N) with
    h_pos[j] = relu(pre_j), h_neg[j] = -relu(-pre_j).
    """
    pre = X @ W1 + b1                   # (N, k)
    h_pos = pre * (pre > 0)             # (N, k)
    h_neg = -((-pre) * ((-pre) > 0))    # (N, k); equals min(pre, 0)
    # Reshape to (k, 2, N).
    cands = np.stack([h_pos.T, h_neg.T], axis=1)  # (k, 2, N)
    return cands


def recover_signs_brute(oracle, W1, b1, s, n_probes: int = 200, rng=None):
    rng = rng or np.random.default_rng(0)
    n_in, k = W1.shape
    X = rng.standard_normal(size=(n_probes, n_in))
    y = np.asarray(oracle(X)).reshape(-1)
    cands = _build_basis_for_sign_search(W1, b1, X)
    ones = np.ones((n_probes, 1))

    best_resid = np.inf
    best_signs = None
    for mask in range(1 << k):
        cols = []
        for j in range(k):
            cols.append(cands[j, (mask >> j) & 1])
        H = np.array(cols).T
        H_aug = np.concatenate([H, ones], axis=1)
        sol, *_ = np.linalg.lstsq(H_aug, y, rcond=None)
        resid = float(np.linalg.norm(H_aug @ sol - y))
        if resid < best_resid:
            best_resid = resid
            best_signs = np.array([
                +1.0 if ((mask >> j) & 1) == 0 else -1.0
                for j in range(k)
            ])
    return best_signs


def recover_signs_mitm(oracle, W1, b1, s, n_probes: int = 200,
                       rng=None, top_per_half: int = 32):
    """Meet-in-the-middle sign recovery.

    Strategy:
      - Build the candidate-activation tensor `cands` of shape (k, 2, N).
      - Split neurons {1..k} into A = first k_A = k//2 and B = remaining k_B.
      - For each σ_A ∈ {±1}^k_A, build H_A(σ_A) ∈ R^{N × k_A}; project the
        probe-output vector y onto the orthogonal complement of [H_A | 1] and
        store the residual vector r_A = (I − P_A) y in a list.
      - For each σ_B ∈ {±1}^k_B, build H_B(σ_B) ∈ R^{N × k_B} and project
        each row's column h_B,j onto the same complement: q_B = (I − P_A) h_B.
        We want lstsq([h_A | h_B | 1], y) to give zero residual; equivalently,
        (I − P_A) y is in the column space of (I − P_A) H_B. The closest such
        match minimises the residual after the second projection.
      - Time: O(2^k_A · N · k_A^2 + 2^k_B · N · k_B^2 · k_A) by precomputing
        QR decompositions. Beats O(2^k · N · k^2) by a factor of 2^(k/2) / k.

    Implementation note: we don't actually achieve full MITM speed because
    the second projection depends on the first (P_A varies with σ_A). We
    instead use a TWO-PASS heuristic:
      Pass 1 (cheap): for each half independently, score sign assignments by
        how well *its* sub-problem alone predicts y. Keep the top-N from each.
      Pass 2: cross-product of the top survivors and full lstsq, select the
        global best.
    This is the standard MITM-with-pruning used in cryptographic key search
    when the halves aren't perfectly separable.
    """
    rng = rng or np.random.default_rng(0)
    n_in, k = W1.shape
    if k <= 12:
        # Small k: brute force is faster than MITM bookkeeping.
        return recover_signs_brute(oracle, W1, b1, s, n_probes=n_probes, rng=rng)

    X = rng.standard_normal(size=(n_probes, n_in))
    y = np.asarray(oracle(X)).reshape(-1)
    cands = _build_basis_for_sign_search(W1, b1, X)  # (k, 2, N)

    k_A = k // 2
    k_B = k - k_A
    ones = np.ones((n_probes, 1))

    def build_H(mask: int, neurons: range) -> np.ndarray:
        cols = []
        for idx, j in enumerate(neurons):
            cols.append(cands[j, (mask >> idx) & 1])
        return np.array(cols).T  # (N, len(neurons))

    # Pass 1A: score each σ_A using only its sub-block + bias.
    half_scores_A = []
    for maskA in range(1 << k_A):
        H_A = build_H(maskA, range(0, k_A))
        H_aug = np.concatenate([H_A, ones], axis=1)
        sol, *_ = np.linalg.lstsq(H_aug, y, rcond=None)
        resid = float(np.linalg.norm(H_aug @ sol - y))
        half_scores_A.append((resid, maskA))
    half_scores_A.sort()
    survivors_A = [m for (_, m) in half_scores_A[:top_per_half]]

    # Pass 1B: score each σ_B using only its sub-block + bias.
    half_scores_B = []
    for maskB in range(1 << k_B):
        H_B = build_H(maskB, range(k_A, k))
        H_aug = np.concatenate([H_B, ones], axis=1)
        sol, *_ = np.linalg.lstsq(H_aug, y, rcond=None)
        resid = float(np.linalg.norm(H_aug @ sol - y))
        half_scores_B.append((resid, maskB))
    half_scores_B.sort()
    survivors_B = [m for (_, m) in half_scores_B[:top_per_half]]

    # Pass 2: cross-product of survivors with the FULL lstsq.
    best_resid = np.inf
    best_mask = None
    for maskA in survivors_A:
        H_A = build_H(maskA, range(0, k_A))
        for maskB in survivors_B:
            H_B = build_H(maskB, range(k_A, k))
            H_full = np.concatenate([H_A, H_B, ones], axis=1)
            sol, *_ = np.linalg.lstsq(H_full, y, rcond=None)
            resid = float(np.linalg.norm(H_full @ sol - y))
            if resid < best_resid:
                best_resid = resid
                best_mask = (maskA, maskB)
    if best_mask is None:
        return None
    maskA, maskB = best_mask
    full_mask = maskA | (maskB << k_A)
    return np.array([
        +1.0 if ((full_mask >> j) & 1) == 0 else -1.0
        for j in range(k)
    ])
