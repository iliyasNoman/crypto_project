#!/usr/bin/env python3
"""Small one-factor-at-a-time robustness sweep around threshold cells."""

from __future__ import annotations

import json
import os

from step_spacing_experiment_utils import REPO_DIR, default_params_for_cell, load_model, run_counted_extraction


CASES = [
    ("20-10-1", 42, 18),
    ("20-10-1", 42, 16),
]

OUT_PATH = os.path.join(REPO_DIR, "results", "step_spacing_robustness.json")


def experiments_for_case(arch: str, bits: int):
    sizes, _, _ = load_model(arch, 42)
    base = default_params_for_cell(sizes, bits)
    return [
        ("baseline", {}),
        ("n_sweeps_low", {"n_sweeps": max(10, base["n_sweeps"] // 2)}),
        ("n_sweeps_high", {"n_sweeps": base["n_sweeps"] * 2}),
        ("n_probes_low", {"n_probes": max(200, base["n_probes"] - 200)}),
        ("n_probes_high", {"n_probes": base["n_probes"] + 200}),
        ("jump_threshold_low", {"jump_threshold": 0.03}),
        ("jump_threshold_high", {"jump_threshold": 0.08}),
        ("sim_threshold_low", {"sim_threshold": 0.80}),
        ("sim_threshold_high", {"sim_threshold": 0.90}),
    ]


def main():
    results = []
    tasks = []
    for arch, seed, bits in CASES:
        for label, overrides in experiments_for_case(arch, bits):
            tasks.append((arch, seed, bits, label, overrides))
    for i, (arch, seed, bits, label, overrides) in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {arch} seed={seed} bits={bits} case={label}", flush=True)
        row = run_counted_extraction(arch, seed, bits, override_params=overrides, verbose=False)
        row["case"] = label
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

