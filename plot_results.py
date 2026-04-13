#!/usr/bin/env python3
"""Generate summary table and plots from experiment results."""

import json
import os
import numpy as np

os.environ['MPLCONFIGDIR'] = '/tmp/mpl_config'
os.makedirs('/tmp/mpl_config', exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

with open(os.path.join(RESULTS_DIR, "experiment_results.json")) as f:
    results = json.load(f)

ARCHITECTURES = sorted(set(r['arch'] for r in results))
SEEDS = sorted(set(r['seed'] for r in results))

# Print summary table
print("=" * 100)
print("SUMMARY TABLE")
print("=" * 100)
print(f"{'Arch':<16} {'Seed':>4} {'Bits':>4} {'OK?':>4} {'Logit Loss':>14} {'Weight Bits':>12} {'SVD Bits':>10} {'Emp Bits':>10}")
print("-" * 100)
for r in results:
    ll = f"{r['logit_loss']:.2e}" if r.get('logit_loss') is not None else "FAIL"
    wb = f"{r['weight_bits']:.1f}" if r.get('weight_bits') is not None else "-"
    sb = f"{r['svd_bits']:.1f}" if r.get('svd_bits') is not None else "-"
    eb = f"{r['empirical_bits']:.1f}" if r.get('empirical_bits') is not None else "-"
    ok = "Y" if r['extraction_success'] else "N"
    print(f"{r['arch']:<16} {r['seed']:>4} {r['bits']:>4} {ok:>4} {ll:>14} {wb:>12} {sb:>10} {eb:>10}")

# Print aggregated view
print("\n" + "=" * 100)
print("AGGREGATED: Success rate and mean metrics by (arch, bits)")
print("=" * 100)
print(f"{'Arch':<16} {'Bits':>4} {'Success':>8} {'Mean Logit Loss':>16} {'Mean Wt Bits':>13} {'Mean Emp Bits':>14}")
print("-" * 100)
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
        print(f"{arch:<16} {bits:>4} {n_success}/{n_total:>5} {ll_str:>16} {wb_str:>13} {eb_str:>14}")

# --- PLOTS ---

colors = {'10-15-15-1': '#1f77b4', '20-10-1': '#ff7f0e', '40-20-1': '#2ca02c'}
markers = {42: 'o', 43: 's', 44: '^'}

# Main plot: bit-width vs extraction fidelity (log-scale epsilon)
fig, ax = plt.subplots(figsize=(9, 6))
for arch in ARCHITECTURES:
    arch_results = [r for r in results if r['arch'] == arch]
    bits_vals = sorted(set(r['bits'] for r in arch_results))
    means, stds, valid_bits = [], [], []
    for b in bits_vals:
        vals = [r['logit_loss'] for r in arch_results
                if r['bits'] == b and r.get('logit_loss') is not None]
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals) if len(vals) > 1 else 0)
            valid_bits.append(b)
    if valid_bits:
        ax.errorbar(valid_bits, means, yerr=stds,
                    label=arch, color=colors.get(arch, 'gray'),
                    marker='o', linewidth=2, capsize=4, markersize=8)
    # Individual seeds
    for seed in SEEDS:
        seed_results = [r for r in arch_results if r['seed'] == seed]
        xs = [r['bits'] for r in seed_results if r.get('logit_loss') is not None]
        ys = [r['logit_loss'] for r in seed_results if r.get('logit_loss') is not None]
        if xs:
            ax.scatter(xs, ys, color=colors.get(arch, 'gray'),
                      marker=markers.get(seed, 'o'), alpha=0.4, s=30, zorder=5)

ax.set_xlabel('Bit-width of Quantized Oracle', fontsize=13)
ax.set_ylabel('Extraction Fidelity (ε = max logit loss)', fontsize=13)
ax.set_title('Post-Training Quantization vs. Cryptanalytic Extraction Fidelity', fontsize=14)
ax.set_yscale('log')
ax.set_xticks([4, 8, 16, 32, 64])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.invert_xaxis()

# Add annotation for failed extractions
ax.annotate('8-bit and 4-bit:\nExtraction fails\n(attack breaks down)',
           xy=(6, ax.get_ylim()[0] * 10), fontsize=10, ha='center',
           color='red', style='italic')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "quantization_vs_extraction.png"), dpi=150, bbox_inches='tight')
print(f"\nMain plot saved to results/quantization_vs_extraction.png")

# Multi-panel plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metric_configs = [
    ('logit_loss', 'Max Logit Loss (ε)', True),
    ('weight_bits', 'Bits of Precision\n(weight matrix)', False),
    ('empirical_bits', 'Bits of Precision\n(empirical, random samples)', False),
]

for ax, (metric, title, use_log) in zip(axes, metric_configs):
    for arch in ARCHITECTURES:
        arch_results = [r for r in results if r['arch'] == arch]
        bits_vals = sorted(set(r['bits'] for r in arch_results))
        means, stds, valid_bits = [], [], []
        for b in bits_vals:
            vals = [r[metric] for r in arch_results
                    if r['bits'] == b and r.get(metric) is not None]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals) if len(vals) > 1 else 0)
                valid_bits.append(b)
        if valid_bits:
            ax.errorbar(valid_bits, means, yerr=stds,
                       label=arch, color=colors.get(arch, 'gray'),
                       marker='o', linewidth=2, capsize=4)
        for seed in SEEDS:
            seed_results = [r for r in arch_results if r['seed'] == seed]
            xs = [r['bits'] for r in seed_results if r.get(metric) is not None]
            ys = [r[metric] for r in seed_results if r.get(metric) is not None]
            if xs:
                ax.scatter(xs, ys, color=colors.get(arch, 'gray'),
                          marker=markers.get(seed, 'o'), alpha=0.4, s=30, zorder=5)

    ax.set_xlabel('Bit-width', fontsize=12)
    ax.set_ylabel(title, fontsize=11)
    ax.set_title(title, fontsize=12)
    if use_log:
        ax.set_yscale('log')
    ax.set_xticks([4, 8, 16, 32, 64])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "quantization_vs_extraction_detailed.png"), dpi=150, bbox_inches='tight')
print(f"Detailed plot saved to results/quantization_vs_extraction_detailed.png")

# Success rate plot
fig, ax = plt.subplots(figsize=(8, 5))
bar_width = 0.25
x_positions = np.arange(len([4, 8, 16, 32, 64]))
bit_labels = [4, 8, 16, 32, 64]

for i, arch in enumerate(ARCHITECTURES):
    rates = []
    for b in bit_labels:
        subset = [r for r in results if r['arch'] == arch and r['bits'] == b]
        n_success = sum(1 for r in subset if r['extraction_success'])
        rates.append(n_success / len(subset) * 100 if subset else 0)
    ax.bar(x_positions + i * bar_width, rates, bar_width,
           label=arch, color=colors.get(arch, 'gray'), alpha=0.8)

ax.set_xlabel('Bit-width', fontsize=13)
ax.set_ylabel('Extraction Success Rate (%)', fontsize=13)
ax.set_title('Attack Success Rate vs. Quantization Level', fontsize=14)
ax.set_xticks(x_positions + bar_width)
ax.set_xticklabels(bit_labels)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 110)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "success_rate.png"), dpi=150, bbox_inches='tight')
print(f"Success rate plot saved to results/success_rate.png")
