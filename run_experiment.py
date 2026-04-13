#!/usr/bin/env python3
"""
Experiment: Does post-training quantization degrade cryptanalytic model extraction?

For each (architecture, seed, bit-width):
  1. Load the trained full-precision model
  2. Quantize weights to the target bit-width (or skip for baseline)
  3. Save the quantized model so extract.py uses it as the oracle
  4. Run extract.py
  5. Run check_solution_svd.py and parse metrics
  6. Restore original model
  7. Collect all results, plot, and save
"""

import subprocess
import sys
import os
import re
import json
import shutil
import numpy as np
import signal

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLCONFIGDIR'] = '/tmp/mpl_config'
os.makedirs('/tmp/mpl_config', exist_ok=True)

def log(msg):
    print(msg, flush=True)

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(REPO_DIR, "models")
RESULTS_DIR = os.path.join(REPO_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ARCHITECTURES = ["10-15-15-1", "20-10-1", "40-20-1"]
SEEDS = [42, 43, 44]
BIT_WIDTHS = [64, 32, 16, 8, 4]  # 64 = baseline (no quantization)

# Timeout per extraction in seconds
EXTRACTION_TIMEOUT = 300  # 5 minutes


def quantize_weights(weights, bits):
    """Uniform symmetric quantization to `bits` bit-width."""
    if bits >= 64:
        return weights  # no quantization
    scale = 2 ** (bits - 1)
    return np.round(weights * scale) / scale


def quantize_model(model_path, bits):
    """Load model, quantize all weight matrices and biases, save back."""
    params = np.load(model_path, allow_pickle=True)
    A_list, B_list = params[0], params[1]

    A_q = [quantize_weights(np.array(a, dtype=np.float64), bits) for a in A_list]
    B_q = [quantize_weights(np.array(b, dtype=np.float64), bits) for b in B_list]

    save_params = np.array([A_q, B_q], dtype=object)
    np.save(model_path, save_params)


def run_extraction(arch, seed):
    """Run extract.py and return (success, logit_loss)."""
    cmd = [sys.executable, "extract.py", arch, str(seed)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=EXTRACTION_TIMEOUT,
            cwd=REPO_DIR
        )
        output = result.stdout + result.stderr

        # Parse "Maximum logit loss on the unit sphere X"
        match = re.search(r'Maximum logit loss on the unit sphere\s+([\d.eE\+\-]+)', output)
        logit_loss = float(match.group(1)) if match else None

        success = result.returncode == 0 and logit_loss is not None
        return success, logit_loss, output
    except subprocess.TimeoutExpired:
        return False, None, "TIMEOUT"
    except Exception as e:
        return False, None, str(e)


def run_svd_check(arch):
    """Run check_solution_svd.py and parse metrics."""
    cmd = [sys.executable, "check_solution_svd.py", arch]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=REPO_DIR,
            env={**os.environ, 'MPLCONFIGDIR': '/tmp/mpl_config'}
        )
        output = result.stdout + result.stderr

        metrics = {}

        # "Number of bits of precision in the weight matrix"
        match = re.search(r'Number of bits of precision in the weight matrix\s+([\d.eE\+\-]+)', output)
        if match:
            metrics['weight_bits'] = float(match.group(1))

        # "Upper bound on number of bits of precision in the output through SVD"
        match = re.search(r'Upper bound on number of bits.*?SVD\s+([\d.eE\+\-]+)', output)
        if match:
            metrics['svd_bits'] = float(match.group(1))

        # "Fewest number of bits of precision over N random samples: X"
        match = re.search(r'Fewest number of bits of precision over.*?:\s+([\d.eE\+\-]+)', output)
        if match:
            metrics['empirical_bits'] = float(match.group(1))

        return metrics, output
    except subprocess.TimeoutExpired:
        return {}, "TIMEOUT"
    except Exception as e:
        return {}, str(e)


def main():
    results = []

    # Back up all original models
    backup_dir = os.path.join(REPO_DIR, "models_backup")
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.copytree(MODELS_DIR, backup_dir)

    total_runs = len(ARCHITECTURES) * len(SEEDS) * len(BIT_WIDTHS)
    run_idx = 0

    for arch in ARCHITECTURES:
        for seed in SEEDS:
            model_filename = f"{seed}_{arch}.npy"
            model_path = os.path.join(MODELS_DIR, model_filename)
            backup_path = os.path.join(backup_dir, model_filename)

            for bits in BIT_WIDTHS:
                run_idx += 1
                label = f"[{run_idx}/{total_runs}] arch={arch} seed={seed} bits={bits}"
                log(f"\n{'='*70}")
                log(f"  {label}")
                log(f"{'='*70}")

                # Restore original model from backup
                shutil.copy2(backup_path, model_path)

                # Quantize if needed
                if bits < 64:
                    log(f"  Quantizing to {bits}-bit...")
                    quantize_model(model_path, bits)

                # Run extraction
                log(f"  Running extraction...")
                success, logit_loss, extract_output = run_extraction(arch, seed)
                log(f"  Extraction {'SUCCESS' if success else 'FAILED'}, logit_loss={logit_loss}")

                if not success:
                    log(f"  Extraction failed. Recording NaN metrics.")
                    results.append({
                        'arch': arch,
                        'seed': seed,
                        'bits': bits,
                        'extraction_success': False,
                        'logit_loss': None,
                        'weight_bits': None,
                        'svd_bits': None,
                        'empirical_bits': None,
                    })
                    # Restore model for next iteration
                    shutil.copy2(backup_path, model_path)
                    continue

                # Run SVD check
                log(f"  Running SVD check...")
                metrics, svd_output = run_svd_check(arch)
                log(f"  SVD metrics: {metrics}")

                results.append({
                    'arch': arch,
                    'seed': seed,
                    'bits': bits,
                    'extraction_success': True,
                    'logit_loss': logit_loss,
                    **metrics,
                })

                # Restore original model
                shutil.copy2(backup_path, model_path)

    # Save raw results
    results_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nResults saved to {results_path}")

    # Print summary table
    print("\n" + "="*90)
    print("SUMMARY TABLE")
    print("="*90)
    print(f"{'Arch':<16} {'Seed':>4} {'Bits':>4} {'Success':>7} {'Logit Loss':>14} {'Weight Bits':>12} {'SVD Bits':>10} {'Emp Bits':>10}")
    print("-"*90)
    for r in results:
        print(f"{r['arch']:<16} {r['seed']:>4} {r['bits']:>4} "
              f"{'Y' if r['extraction_success'] else 'N':>7} "
              f"{r.get('logit_loss', 'N/A'):>14} "
              f"{r.get('weight_bits', 'N/A'):>12} "
              f"{r.get('svd_bits', 'N/A'):>10} "
              f"{r.get('empirical_bits', 'N/A'):>10}")

    # Generate plots
    generate_plots(results)


def generate_plots(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metric_names = [
        ('logit_loss', 'Max Logit Loss (oracle-vs-extracted)', True),
        ('weight_bits', 'Bits of Precision (weight matrix)', False),
        ('empirical_bits', 'Bits of Precision (empirical, random samples)', False),
    ]

    colors = {'10-20-20-1': '#1f77b4', '80-40-20-1': '#ff7f0e', '784-128-1': '#2ca02c'}
    markers = {42: 'o', 43: 's', 44: '^'}

    for ax, (metric, title, use_log) in zip(axes, metric_names):
        for arch in ARCHITECTURES:
            arch_results = [r for r in results if r['arch'] == arch]

            # Average across seeds
            bits_vals = sorted(set(r['bits'] for r in arch_results))
            means = []
            stds = []
            valid_bits = []
            for b in bits_vals:
                vals = [r[metric] for r in arch_results
                        if r['bits'] == b and r.get(metric) is not None]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                    valid_bits.append(b)

            if valid_bits:
                means = np.array(means)
                stds = np.array(stds)
                ax.errorbar(valid_bits, means, yerr=stds,
                           label=arch, color=colors[arch],
                           marker='o', linewidth=2, capsize=4)

            # Plot individual seeds
            for seed in SEEDS:
                seed_results = [r for r in arch_results if r['seed'] == seed]
                xs = [r['bits'] for r in seed_results if r.get(metric) is not None]
                ys = [r[metric] for r in seed_results if r.get(metric) is not None]
                if xs:
                    ax.scatter(xs, ys, color=colors[arch], marker=markers[seed],
                             alpha=0.4, s=30, zorder=5)

        ax.set_xlabel('Bit-width', fontsize=12)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12)
        if use_log:
            ax.set_yscale('log')
        ax.set_xticks([4, 8, 16, 32, 64])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()  # Higher bits on left

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "quantization_vs_extraction.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")

    # Also make the specific plot from CLAUDE.md: bit-width vs log-scale epsilon
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for arch in ARCHITECTURES:
        arch_results = [r for r in results if r['arch'] == arch]
        bits_vals = sorted(set(r['bits'] for r in arch_results))
        means = []
        stds = []
        valid_bits = []
        for b in bits_vals:
            vals = [r['logit_loss'] for r in arch_results
                    if r['bits'] == b and r.get('logit_loss') is not None]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                valid_bits.append(b)
        if valid_bits:
            means = np.array(means)
            stds = np.array(stds)
            ax2.errorbar(valid_bits, means, yerr=stds,
                        label=arch, color=colors[arch],
                        marker='o', linewidth=2, capsize=4, markersize=8)

    ax2.set_xlabel('Bit-width of Quantized Oracle', fontsize=13)
    ax2.set_ylabel('Extraction Fidelity (ε = max logit loss)', fontsize=13)
    ax2.set_title('Post-Training Quantization vs. Cryptanalytic Extraction Fidelity', fontsize=14)
    ax2.set_yscale('log')
    ax2.set_xticks([4, 8, 16, 32, 64])
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    plot_path2 = os.path.join(RESULTS_DIR, "quantization_vs_extraction_main.png")
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    print(f"Main plot saved to {plot_path2}")


if __name__ == "__main__":
    main()
