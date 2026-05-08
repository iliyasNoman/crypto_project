#!/usr/bin/env python3
"""Plots for the additional pre-writeup step-spacing experiments."""

from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(REPO_DIR, "results")


def load(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def useful(row):
    return bool(row.get("extraction_success") and row.get("rel_loss") is not None and row["rel_loss"] < 0.5)


def plot_finer_bits(data):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    grouped = defaultdict(list)
    for r in data:
        grouped[(r["arch"], r["seed"])].append(r)
    colors = {"20-10-1": "#0b6e4f", "40-20-1": "#b5651d"}
    markers = {42: "o", 43: "s", 44: "^"}
    for (arch, seed), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: r["bits"], reverse=True)
        xs = [r["bits"] for r in rows if r.get("rel_loss") is not None]
        ys = [r["rel_loss"] for r in rows if r.get("rel_loss") is not None]
        ax.plot(xs, ys, marker=markers.get(seed, "o"), color=colors.get(arch, "#333"), label=f"{arch} / {seed}")
    ax.axhline(0.5, color="#aa3333", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Output rounding precision (bits)")
    ax.set_ylabel("Relative loss")
    ax.set_title("Step-Spacing Finer Bit Sweep")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "step_spacing_finer_bits_results_rel_loss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_query_counts(data):
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    bits = [32, 16, 8, 4]
    grouped = defaultdict(list)
    for r in data:
        grouped[r["bits"]].append(r["query_count"])
    means = [np.mean(grouped[b]) for b in bits]
    stds = [np.std(grouped[b]) for b in bits]
    ax.bar([str(b) for b in bits], means, yerr=stds, color="#4c78a8", capsize=4)
    ax.set_xlabel("Bits")
    ax.set_ylabel("Oracle queries")
    ax.set_title("Step-Spacing Query Counts")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "step_spacing_query_counts.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_robustness(data):
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    labels = [f"{r['arch']}\n{r['bits']}b\n{r['case']}" for r in data]
    vals = [r["rel_loss"] if r.get("rel_loss") is not None else 1.1 for r in data]
    ax.bar(range(len(vals)), vals, color="#54a24b")
    ax.axhline(0.5, color="#aa3333", linestyle="--", linewidth=1.5)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Relative loss")
    ax.set_title("Step-Spacing Hyperparameter Robustness")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "step_spacing_robustness.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scaling(data):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    rows = sorted(data, key=lambda r: r["bits"], reverse=True)
    xs = [r["bits"] for r in rows]
    ys = [r["rel_loss"] if r.get("rel_loss") is not None else np.nan for r in rows]
    ax.plot(xs, ys, marker="o", color="#e45756", linewidth=2)
    ax.axhline(0.5, color="#aa3333", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Output rounding precision (bits)")
    ax.set_ylabel("Relative loss")
    ax.set_title("Scaling Check: 80-40-1")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "step_spacing_scaling_80_40_1.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    finer = load("step_spacing_finer_bits_results.json")
    if finer:
        plot_finer_bits(finer)
    q = load("step_spacing_query_counts.json")
    if q:
        plot_query_counts(q)
    rob = load("step_spacing_robustness.json")
    if rob:
        plot_robustness(rob)
    sc = load("step_spacing_scaling_80_40_1.json")
    if sc:
        plot_scaling(sc)


if __name__ == "__main__":
    main()
