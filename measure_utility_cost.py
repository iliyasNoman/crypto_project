#!/usr/bin/env python3
"""
Utility-cost measurement for the output-rounding defense.

For each trained model and each rounding precision in {32, 16, 8, 4}, we
sample N inputs from the training distribution (standard Gaussian on the
input dimension) and compare:

  y_unrounded = f(x)
  y_rounded   = round(y_unrounded * 2^(N-1)) / 2^(N-1)

We report mean |y_rounded - y_unrounded| (mean absolute error), the max
absolute error, and the mean relative error |delta| / mean(|y|).

This is the "how much does the defense hurt the model's actual outputs"
companion to the extraction experiment.
"""

import json
import os
import numpy as np

ARCHITECTURES = ["10-15-15-1", "20-10-1", "40-20-1"]
SEEDS = [42, 43, 44]
BIT_WIDTHS = [32, 16, 8, 4]
N_SAMPLES = 100_000

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(REPO_DIR, "models")
RESULTS_DIR = os.path.join(REPO_DIR, "results")


def forward(x, A, B, sizes):
    for i, (a, b) in enumerate(zip(A, B)):
        x = np.dot(x, a) + b
        if i < len(sizes) - 2:
            x = x * (x > 0)
    return x


def round_to_bits(y, bits):
    scale = 2.0 ** (bits - 1)
    return np.round(y * scale) / scale


def main():
    rng = np.random.default_rng(0)
    rows = []
    for arch in ARCHITECTURES:
        sizes = list(map(int, arch.split("-")))
        in_dim = sizes[0]
        for seed in SEEDS:
            path = os.path.join(MODELS_DIR, f"{seed}_{arch}.npy")
            params = np.load(path, allow_pickle=True)
            A = [np.array(a, dtype=np.float64) for a in params[0]]
            B = [np.array(b, dtype=np.float64) for b in params[1]]

            x = rng.standard_normal(size=(N_SAMPLES, in_dim))
            y = forward(x, A, B, sizes)
            mean_abs_y = float(np.mean(np.abs(y)))
            for bits in BIT_WIDTHS:
                yq = round_to_bits(y, bits)
                err = np.abs(yq - y)
                rows.append({
                    "arch": arch,
                    "seed": seed,
                    "bits": bits,
                    "mean_abs_err": float(np.mean(err)),
                    "max_abs_err": float(np.max(err)),
                    "mean_rel_err": float(np.mean(err) / max(mean_abs_y, 1e-30)),
                    "mean_abs_y": mean_abs_y,
                    "rounding_step": 2.0 ** (1 - bits),
                })

    out_path = os.path.join(RESULTS_DIR, "utility_cost.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved {out_path}\n")

    # Per-run table
    print("=" * 100)
    print("PER-RUN UTILITY COST")
    print("=" * 100)
    print(f"{'Arch':<12} {'Seed':>4} {'Bits':>4} {'step':>10} "
          f"{'mean |Δy|':>12} {'max |Δy|':>12} {'mean |y|':>10} {'rel err':>10}")
    print("-" * 100)
    for r in rows:
        print(f"{r['arch']:<12} {r['seed']:>4} {r['bits']:>4} "
              f"{r['rounding_step']:>10.2e} "
              f"{r['mean_abs_err']:>12.2e} {r['max_abs_err']:>12.2e} "
              f"{r['mean_abs_y']:>10.3f} {r['mean_rel_err']:>10.2e}")

    # Aggregated by (arch, bits)
    print("\n" + "=" * 100)
    print("AGGREGATE (mean over seeds)")
    print("=" * 100)
    print(f"{'Arch':<12} {'Bits':>4} {'step':>10} "
          f"{'mean |Δy|':>12} {'max |Δy|':>12} {'mean |y|':>10} {'rel err':>10}")
    print("-" * 100)
    for arch in ARCHITECTURES:
        for bits in BIT_WIDTHS:
            sub = [r for r in rows if r["arch"] == arch and r["bits"] == bits]
            mae = np.mean([r["mean_abs_err"] for r in sub])
            mxe = np.mean([r["max_abs_err"] for r in sub])
            may = np.mean([r["mean_abs_y"] for r in sub])
            rel = np.mean([r["mean_rel_err"] for r in sub])
            print(f"{arch:<12} {bits:>4} {2.0**(1-bits):>10.2e} "
                  f"{mae:>12.2e} {mxe:>12.2e} {may:>10.3f} {rel:>10.2e}")


if __name__ == "__main__":
    main()
