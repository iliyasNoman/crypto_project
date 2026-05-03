#!/usr/bin/env python3
"""Run the step-spacing attack across the full (arch, seed, bits) grid.

For comparison with the prior weight-quantization / output-rounding tables,
we use the 2-layer architectures from this repo and bits ∈ {64, 32, 16, 8, 4}.
Saves results to results/step_spacing_results.json.
"""

import json
import os
import sys
import time
import numpy as np

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)

from extract_rounded import (
    extract_two_layer, make_rounded_oracle, evaluate, forward,
)


ARCHS = ["20-10-1", "40-20-1"]
SEEDS = [42, 43, 44]
# We skip bits=64 here: the unrounded oracle is exactly Carlini's baseline,
# which the prior README already shows succeeds. The interesting comparison
# is the rounded regime where Carlini fails and step-spacing recovers.
BITS = [32, 16, 8, 4]


def run_one(arch, seed, bits):
    sizes = list(map(int, arch.split("-")))
    path = os.path.join(REPO_DIR, "models", f"{seed}_{arch}.npy")
    params = np.load(path, allow_pickle=True)
    A = [np.array(a, dtype=np.float64) for a in params[0]]
    B = [np.array(b, dtype=np.float64) for b in params[1]]

    oracle, s = make_rounded_oracle(A, B, sizes, bits)
    if s == 0:
        s = 1e-30  # avoid division by zero in level computation
    # For wider arch, use more sweeps so clustering can hit n_hidden bins.
    n_sweeps = 80 if sizes[1] >= 20 else 40
    n_probes = 600 if bits <= 16 else 400
    grad_target_levels = 64
    jump_threshold = 0.05
    sim_threshold = 0.85

    t0 = time.time()
    out = extract_two_layer(
        oracle, sizes, s,
        n_sweeps=n_sweeps, n_probes=n_probes,
        grad_target_levels=grad_target_levels,
        grad_init_radius=1e-3, grad_max_radius=0.05,
        jump_threshold=jump_threshold, sim_threshold=sim_threshold,
        use_mitm=True, seed=seed, verbose=False,
    )
    dt = time.time() - t0

    if out is None:
        return {
            "arch": arch, "seed": seed, "bits": bits,
            "extraction_success": False,
            "max_logit_loss": None, "mean_logit_loss": None, "rel_loss": None,
            "wallclock_s": dt,
        }
    A_hat, B_hat = out
    metrics = evaluate(A, B, A_hat, B_hat, sizes)
    return {
        "arch": arch, "seed": seed, "bits": bits,
        "extraction_success": True,
        **metrics,
        "wallclock_s": dt,
    }


def main():
    out_path = os.path.join(REPO_DIR, "results", "step_spacing_results.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    else:
        existing = []
    # Drop any bits=64 entries from old runs (they belong to a different
    # axis: that's the Carlini baseline experiment).
    existing = [r for r in existing if r.get("bits") != 64]
    done = {(r["arch"], r["seed"], r["bits"]) for r in existing}
    results = list(existing)

    grid = [(a, s, b) for a in ARCHS for s in SEEDS for b in BITS]
    print(f"Running {len(grid)} cells ({len(done)} already done)...")
    for i, (arch, seed, bits) in enumerate(grid, 1):
        if (arch, seed, bits) in done:
            print(f"\n[{i}/{len(grid)}] arch={arch} seed={seed} bits={bits}  (cached)",
                  flush=True)
            continue
        print(f"\n[{i}/{len(grid)}] arch={arch} seed={seed} bits={bits}", flush=True)
        try:
            r = run_one(arch, seed, bits)
        except Exception as e:
            r = {"arch": arch, "seed": seed, "bits": bits,
                 "extraction_success": False, "error": str(e)}
        results.append(r)
        ok = "Y" if r.get("extraction_success") else "N"
        ll = r.get("max_logit_loss")
        rel = r.get("rel_loss")
        ll_s = f"{ll:.3e}" if ll is not None else "FAIL"
        rel_s = f"{rel:.3e}" if rel is not None else "FAIL"
        dt = r.get("wallclock_s", float("nan"))
        print(f"  {ok}  max_logit_loss={ll_s}  rel_loss={rel_s}  ({dt:.1f}s)",
              flush=True)
        # Save incrementally.
        out_path = os.path.join(REPO_DIR, "results", "step_spacing_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Summary table
    print("\n" + "=" * 110)
    print("SUMMARY: STEP-SPACING ATTACK ON ROUNDED ORACLE")
    print("=" * 110)
    print(f"{'Arch':<10} {'Seed':>4} {'Bits':>4} {'OK':>3} {'max_logit_loss':>14} "
          f"{'rel_loss':>11} {'wall':>7}")
    print("-" * 110)
    for r in results:
        ll = r.get("max_logit_loss")
        rel = r.get("rel_loss")
        ll_s = f"{ll:.2e}" if ll is not None else "FAIL"
        rel_s = f"{rel:.2e}" if rel is not None else "FAIL"
        ok = "Y" if r.get("extraction_success") else "N"
        dt = r.get("wallclock_s", 0.0)
        print(f"{r['arch']:<10} {r['seed']:>4} {r['bits']:>4} {ok:>3} "
              f"{ll_s:>14} {rel_s:>11} {dt:>6.1f}s")


if __name__ == "__main__":
    main()
