#!/usr/bin/env python3
"""Full finer-bit sweep for the step-spacing attack.

Writes results to results/step_spacing_finer_bits_results.json and resumes if
that file already exists.
"""

from __future__ import annotations

import json
import os

from step_spacing_experiment_utils import run_counted_extraction, REPO_DIR


ARCHS = ["20-10-1", "40-20-1"]
SEEDS = [42, 43, 44]
BITS = [32, 24, 20, 18, 16, 14, 12, 10, 8, 4]
OUT_PATH = os.path.join(REPO_DIR, "results", "step_spacing_finer_bits_results.json")


def main():
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            results = json.load(f)
    else:
        partial = os.path.join(REPO_DIR, "results", "step_spacing_finer_bits_partial.json")
        if os.path.exists(partial):
            with open(partial) as f:
                results = json.load(f)
        else:
            results = []

    done = {(r["arch"], r["seed"], r["bits"]) for r in results}
    grid = [(a, s, b) for a in ARCHS for s in SEEDS for b in BITS]
    print(f"Running {len(grid)} cells ({len(done)} already done)...", flush=True)
    for i, (arch, seed, bits) in enumerate(grid, 1):
        if (arch, seed, bits) in done:
            print(f"[{i}/{len(grid)}] arch={arch} seed={seed} bits={bits} (cached)", flush=True)
            continue
        print(f"[{i}/{len(grid)}] arch={arch} seed={seed} bits={bits}", flush=True)
        row = run_counted_extraction(arch, seed, bits, verbose=False)
        ok = "Y" if row["extraction_success"] else "N"
        ll = row.get("max_logit_loss")
        rel = row.get("rel_loss")
        ll_s = f"{ll:.3e}" if ll is not None else "FAIL"
        rel_s = f"{rel:.3e}" if rel is not None else "FAIL"
        print(
            f"  {ok}  max_logit_loss={ll_s}  rel_loss={rel_s}  "
            f"queries={row['query_count']}  ({row['wallclock_s']:.1f}s)",
            flush=True,
        )
        results.append(row)
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

