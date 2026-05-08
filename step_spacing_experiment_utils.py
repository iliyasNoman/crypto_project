#!/usr/bin/env python3
"""Utilities for standalone step-spacing experiments.

These helpers wrap the existing extraction code without modifying it.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

from extract_rounded import extract_two_layer, evaluate, make_rounded_oracle


REPO_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class CountedOracle:
    oracle: Callable
    query_count: int = 0

    def __call__(self, x):
        arr = np.atleast_2d(x)
        self.query_count += arr.shape[0]
        return self.oracle(arr)


def load_model(arch: str, seed: int):
    sizes = list(map(int, arch.split("-")))
    path = os.path.join(REPO_DIR, "models", f"{seed}_{arch}.npy")
    params = np.load(path, allow_pickle=True)
    A = [np.array(a, dtype=np.float64) for a in params[0]]
    B = [np.array(b, dtype=np.float64) for b in params[1]]
    return sizes, A, B


def default_params_for_cell(sizes, bits: int) -> Dict:
    return {
        "n_sweeps": 80 if sizes[1] >= 20 else 40,
        "n_probes": 600 if bits <= 16 else 400,
        "grad_target_levels": 64,
        "grad_init_radius": 1e-3,
        "grad_max_radius": 0.05,
        "jump_threshold": 0.05,
        "sim_threshold": 0.85,
        "use_mitm": True,
    }


def run_counted_extraction(
    arch: str,
    seed: int,
    bits: int,
    *,
    override_params: Dict | None = None,
    verbose: bool = False,
):
    sizes, A, B = load_model(arch, seed)
    oracle, s = make_rounded_oracle(A, B, sizes, bits)
    counted = CountedOracle(oracle=oracle)
    if s == 0:
        s = 1e-30

    params = default_params_for_cell(sizes, bits)
    if override_params:
        params.update(override_params)

    t0 = time.time()
    out = extract_two_layer(
        counted,
        sizes,
        s,
        seed=seed,
        verbose=verbose,
        **params,
    )
    dt = time.time() - t0

    row = {
        "arch": arch,
        "seed": seed,
        "bits": bits,
        "query_count": counted.query_count,
        "wallclock_s": dt,
        **params,
    }

    if out is None:
        row.update(
            {
                "extraction_success": False,
                "max_logit_loss": None,
                "mean_logit_loss": None,
                "rel_loss": None,
            }
        )
        return row

    A_hat, B_hat = out
    metrics = evaluate(A, B, A_hat, B_hat, sizes)
    row.update({"extraction_success": True, **metrics})
    return row

