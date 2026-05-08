#!/usr/bin/env python3
"""Query-count measurement for existing coarse step-spacing settings."""

from __future__ import annotations

import json
import os

from step_spacing_experiment_utils import REPO_DIR, run_counted_extraction


ARCHS = ["20-10-1", "40-20-1"]
SEEDS = [42, 43, 44]
BITS = [32, 16, 8, 4]
OUT_PATH = os.path.join(REPO_DIR, "results", "step_spacing_query_counts.json")


def main():
    results = []
    grid = [(a, s, b) for a in ARCHS for s in SEEDS for b in BITS]
    for i, (arch, seed, bits) in enumerate(grid, 1):
        print(f"[{i}/{len(grid)}] arch={arch} seed={seed} bits={bits}", flush=True)
        row = run_counted_extraction(arch, seed, bits, verbose=False)
        results.append(row)
        print(
            f"  queries={row['query_count']} rel_loss={row.get('rel_loss')} "
            f"({row['wallclock_s']:.1f}s)",
            flush=True,
        )
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

