#!/usr/bin/env python3
"""Tables and plots for the output-rounding experiment.

Reads results/output_rounding_results.json and emits:
  - results/output_rounding_vs_extraction.png  (main: log-eps vs bits)
  - results/output_rounding_success_rate.png   (success rate by bits)
  - results/output_rounding_vs_quantization.png (side-by-side w/ original
                                                 weight-quantization run, if
                                                 results/experiment_results.json
                                                 is also present)
"""

import json
import os
import numpy as np

os.environ['MPLCONFIGDIR'] = '/tmp/mpl_config'
os.makedirs('/tmp/mpl_config', exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

ROUND_PATH = os.path.join(RESULTS_DIR, "output_rounding_results.json")
QUANT_PATH = os.path.join(RESULTS_DIR, "experiment_results.json")

with open(ROUND_PATH) as f:
    round_results = json.load(f)

quant_results = None
if os.path.exists(QUANT_PATH):
    with open(QUANT_PATH) as f:
        quant_results = json.load(f)

ARCHITECTURES = sorted(set(r['arch'] for r in round_results))
SEEDS = sorted(set(r['seed'] for r in round_results))
BIT_LEVELS = sorted(set(r['bits'] for r in round_results))

COLORS = {'10-15-15-1': '#1f77b4', '20-10-1': '#ff7f0e', '40-20-1': '#2ca02c'}
MARKERS = {42: 'o', 43: 's', 44: '^'}


def print_per_run_table(results, label):
    print(f"\n{'=' * 100}")
    print(f"PER-RUN: {label}")
    print('=' * 100)
    print(f"{'Arch':<12} {'Seed':>4} {'Bits':>5} {'OK?':>4} {'Logit Loss':>14} {'Wt Bits':>9} {'SVD Bits':>10} {'Emp Bits':>10}")
    print('-' * 100)
    for r in results:
        ll = f"{r['logit_loss']:.2e}" if r.get('logit_loss') is not None else "FAIL"
        wb = f"{r['weight_bits']:.1f}" if r.get('weight_bits') is not None else "-"
        sb = f"{r['svd_bits']:.1f}" if r.get('svd_bits') is not None else "-"
        eb = f"{r['empirical_bits']:.1f}" if r.get('empirical_bits') is not None else "-"
        ok = "Y" if r['extraction_success'] else "N"
        print(f"{r['arch']:<12} {r['seed']:>4} {r['bits']:>5} {ok:>4} {ll:>14} {wb:>9} {sb:>10} {eb:>10}")


def print_aggregate_table(results, label):
    print(f"\n{'=' * 100}")
    print(f"AGGREGATE: {label}")
    print('=' * 100)
    print(f"{'Arch':<12} {'Bits':>5} {'Success':>10} {'Mean Logit Loss':>18} {'Mean Wt Bits':>14} {'Mean Emp Bits':>15}")
    print('-' * 100)
    for arch in ARCHITECTURES:
        for bits in sorted(set(r['bits'] for r in results)):
            subset = [r for r in results if r['arch'] == arch and r['bits'] == bits]
            n_success = sum(1 for r in subset if r['extraction_success'])
            n_total = len(subset)
            lls = [r['logit_loss'] for r in subset if r.get('logit_loss') is not None]
            wbs = [r['weight_bits'] for r in subset if r.get('weight_bits') is not None]
            ebs = [r['empirical_bits'] for r in subset if r.get('empirical_bits') is not None]
            ll_str = f"{np.mean(lls):.2e}" if lls else "-"
            wb_str = f"{np.mean(wbs):.1f}" if wbs else "-"
            eb_str = f"{np.mean(ebs):.1f}" if ebs else "-"
            print(f"{arch:<12} {bits:>5} {n_success}/{n_total:<8} {ll_str:>18} {wb_str:>14} {eb_str:>15}")


def main_plot(results, title, fname):
    fig, ax = plt.subplots(figsize=(9, 6))
    for arch in ARCHITECTURES:
        arch_r = [r for r in results if r['arch'] == arch]
        bits_vals = sorted(set(r['bits'] for r in arch_r))
        means, stds, valid_bits = [], [], []
        for b in bits_vals:
            vals = [r['logit_loss'] for r in arch_r
                    if r['bits'] == b and r.get('logit_loss') is not None]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals) if len(vals) > 1 else 0)
                valid_bits.append(b)
        if valid_bits:
            ax.errorbar(valid_bits, means, yerr=stds, label=arch,
                        color=COLORS.get(arch, 'gray'),
                        marker='o', linewidth=2, capsize=4, markersize=8)
        for seed in SEEDS:
            sr = [r for r in arch_r if r['seed'] == seed]
            xs = [r['bits'] for r in sr if r.get('logit_loss') is not None]
            ys = [r['logit_loss'] for r in sr if r.get('logit_loss') is not None]
            if xs:
                ax.scatter(xs, ys, color=COLORS.get(arch, 'gray'),
                           marker=MARKERS.get(seed, 'o'), alpha=0.4, s=30, zorder=5)

    ax.set_xlabel('Output rounding precision (bits)', fontsize=13)
    ax.set_ylabel('Extraction Fidelity (ε = max logit loss)', fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_yscale('log')
    ax.set_xticks(BIT_LEVELS)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, fname)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved {out}")
    plt.close(fig)


def success_rate_plot(results, title, fname):
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_w = 0.25
    bits_axis = sorted(set(r['bits'] for r in results))
    x_pos = np.arange(len(bits_axis))
    for i, arch in enumerate(ARCHITECTURES):
        rates = []
        for b in bits_axis:
            sub = [r for r in results if r['arch'] == arch and r['bits'] == b]
            n_ok = sum(1 for r in sub if r['extraction_success'])
            rates.append(100 * n_ok / len(sub) if sub else 0)
        ax.bar(x_pos + i * bar_w, rates, bar_w, label=arch,
               color=COLORS.get(arch, 'gray'), alpha=0.85)
    ax.set_xlabel('Output rounding precision (bits)', fontsize=13)
    ax.set_ylabel('Extraction Success Rate (%)', fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x_pos + bar_w)
    ax.set_xticklabels(bits_axis)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 110)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, fname)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved {out}")
    plt.close(fig)


def comparison_plot(round_r, quant_r, fname):
    """Side-by-side: weight quantization vs output rounding."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, results, title, xlabel in [
        (axes[0], quant_r, 'Defense: Weight Quantization (post-training)',
         'Weight bit-width'),
        (axes[1], round_r, 'Defense: Output Rounding (at inference)',
         'Output rounding precision (bits)'),
    ]:
        archs = sorted(set(r['arch'] for r in results))
        bits_axis = sorted(set(r['bits'] for r in results))
        for arch in archs:
            arch_r = [r for r in results if r['arch'] == arch]
            means, stds, vb = [], [], []
            for b in bits_axis:
                vals = [r['logit_loss'] for r in arch_r
                        if r['bits'] == b and r.get('logit_loss') is not None]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals) if len(vals) > 1 else 0)
                    vb.append(b)
            if vb:
                ax.errorbar(vb, means, yerr=stds, label=arch,
                            color=COLORS.get(arch, 'gray'),
                            marker='o', linewidth=2, capsize=4, markersize=8)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_title(title, fontsize=13)
        ax.set_yscale('log')
        ax.set_xticks(bits_axis)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
    axes[0].set_ylabel('Extraction Fidelity (ε = max logit loss)', fontsize=13)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, fname)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    print_per_run_table(round_results, "OUTPUT ROUNDING")
    print_aggregate_table(round_results, "OUTPUT ROUNDING")
    main_plot(round_results,
              'Output Rounding (at Inference) vs. Cryptanalytic Extraction',
              'output_rounding_vs_extraction.png')
    success_rate_plot(round_results,
                      'Attack Success Rate vs. Output Rounding Precision',
                      'output_rounding_success_rate.png')
    if quant_results:
        comparison_plot(round_results, quant_results,
                        'output_rounding_vs_quantization.png')
