#!/usr/bin/env python3
"""Scaling check on one wider 2-layer network."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from step_spacing_experiment_utils import REPO_DIR, run_counted_extraction


ARCH = "80-40-1"
SEEDS = [42]
BITS = [32, 20, 18, 16, 8, 4]
OUT_PATH = os.path.join(REPO_DIR, "results", "step_spacing_scaling_80_40_1.json")


def ensure_model(seed: int):
    path = os.path.join(REPO_DIR, "models", f"{seed}_{ARCH}.npy")
    if os.path.exists(path):
        return
    cmd = [sys.executable, "train_models.py", ARCH, str(seed)]
    print(f"Training missing model: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_DIR, check=True)


def main():
    results = []
    for seed in SEEDS:
        ensure_model(seed)
        for i, bits in enumerate(BITS, 1):
            print(f"[{i}/{len(BITS)}] arch={ARCH} seed={seed} bits={bits}", flush=True)
            row = run_counted_extraction(ARCH, seed, bits, verbose=False)
            results.append(row)
            print(
                f"  rel_loss={row.get('rel_loss')} queries={row['query_count']} "
                f"({row['wallclock_s']:.1f}s)",
                flush=True,
            )
            with open(OUT_PATH, "w") as f:
                json.dump(results, f, indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

