#!/usr/bin/env python3
"""
Experiment: does rounding the oracle's outputs at INFERENCE TIME degrade
cryptanalytic model extraction (Carlini et al., CRYPTO'20)?

Mirrors run_experiment.py exactly, but instead of quantizing the model's
weights at-rest, the model file is left untouched and the env var
INFERENCE_ROUND_BITS=N tells src/utils.py::run() to round each oracle
response via:
        y' = round(y * 2^(N-1)) / 2^(N-1)

Same axis as the weight-quantization experiment (64 = no rounding baseline).
For each (architecture, seed, bits):
  1. Set INFERENCE_ROUND_BITS=bits in the subprocess env
  2. Run extract.py (oracle now rounds before responding)
  3. Run check_solution_svd.py (compares extracted vs real, both unrounded)
  4. Parse and record metrics
"""

import subprocess
import sys
import os
import re
import json
import signal
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLCONFIGDIR'] = '/tmp/mpl_config'
os.makedirs('/tmp/mpl_config', exist_ok=True)


def log(msg):
    print(msg, flush=True)


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(REPO_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ARCHITECTURES = ["10-15-15-1", "20-10-1", "40-20-1"]
SEEDS = [42, 43, 44]
BIT_WIDTHS = [64, 32, 16, 8, 4]  # 64 = baseline, no rounding

EXTRACTION_TIMEOUT = 90  # seconds; successful extractions finish in <60s


def _run_with_pgroup_timeout(cmd, env, timeout, cwd):
    """Run cmd in a new session so we can SIGKILL the entire process tree
    on timeout. subprocess.run's timeout doesn't kill mp.Pool children that
    extract.py spawns, which causes the parent to hang on stdout pipe EOF."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=cwd, env=env, start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout, stderr, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", "TIMEOUT (group kill)"
        return -9, stdout, stderr, True


def run_extraction(arch, seed, bits):
    """Run extract.py with INFERENCE_ROUND_BITS set; return (success, logit_loss, output)."""
    env = {**os.environ}
    if bits < 64:
        env["INFERENCE_ROUND_BITS"] = str(bits)
    else:
        env.pop("INFERENCE_ROUND_BITS", None)

    cmd = [sys.executable, "extract.py", arch, str(seed)]
    rc, stdout, stderr, timed_out = _run_with_pgroup_timeout(
        cmd, env, EXTRACTION_TIMEOUT, REPO_DIR,
    )
    output = (stdout or "") + (stderr or "")
    if timed_out:
        return False, None, output
    match = re.search(r'Maximum logit loss on the unit sphere\s+([\d.eE\+\-]+)', output)
    logit_loss = float(match.group(1)) if match else None
    success = rc == 0 and logit_loss is not None
    return success, logit_loss, output


def run_svd_check(arch):
    """Run check_solution_svd.py; parse weight_bits / svd_bits / empirical_bits."""
    cmd = [sys.executable, "check_solution_svd.py", arch]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=REPO_DIR,
            env={**os.environ, 'MPLCONFIGDIR': '/tmp/mpl_config'},
        )
        output = result.stdout + result.stderr
        metrics = {}
        m = re.search(r'Number of bits of precision in the weight matrix\s+([\d.eE\+\-]+)', output)
        if m: metrics['weight_bits'] = float(m.group(1))
        m = re.search(r'Upper bound on number of bits.*?SVD\s+([\d.eE\+\-]+)', output)
        if m: metrics['svd_bits'] = float(m.group(1))
        m = re.search(r'Fewest number of bits of precision over.*?:\s+([\d.eE\+\-]+)', output)
        if m: metrics['empirical_bits'] = float(m.group(1))
        return metrics, output
    except subprocess.TimeoutExpired:
        return {}, "TIMEOUT"
    except Exception as e:
        return {}, str(e)


def main():
    # Resume mode: load any prior results so completed configs are skipped.
    results_path = os.path.join(RESULTS_DIR, "output_rounding_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        log(f"Resuming: {len(results)} prior entries loaded.")
    else:
        results = []
    done = {(r['arch'], r['seed'], r['bits']) for r in results}

    total_runs = len(ARCHITECTURES) * len(SEEDS) * len(BIT_WIDTHS)
    run_idx = 0

    for arch in ARCHITECTURES:
        for seed in SEEDS:
            for bits in BIT_WIDTHS:
                run_idx += 1
                label = f"[{run_idx}/{total_runs}] arch={arch} seed={seed} bits={bits}"
                log("\n" + "=" * 70)
                log(f"  {label}")
                log("=" * 70)

                if (arch, seed, bits) in done:
                    log("  Already recorded; skipping.")
                    continue

                if bits < 64:
                    log(f"  INFERENCE_ROUND_BITS={bits} (oracle rounds outputs)")
                else:
                    log("  No rounding (baseline)")

                t0 = time.time()
                log("  Running extraction...")
                success, logit_loss, _ = run_extraction(arch, seed, bits)
                log(f"  Extraction {'SUCCESS' if success else 'FAILED'} ({time.time()-t0:.1f}s), logit_loss={logit_loss}")

                if not success:
                    results.append({
                        'arch': arch, 'seed': seed, 'bits': bits,
                        'extraction_success': False,
                        'logit_loss': None, 'weight_bits': None,
                        'svd_bits': None, 'empirical_bits': None,
                    })
                    # Persist incrementally so partial results survive interrupts
                    _save(results)
                    continue

                log("  Running SVD check...")
                metrics, _ = run_svd_check(arch)
                log(f"  SVD metrics: {metrics}")

                results.append({
                    'arch': arch, 'seed': seed, 'bits': bits,
                    'extraction_success': True,
                    'logit_loss': logit_loss,
                    **metrics,
                })
                _save(results)

    _save(results)
    log(f"\nResults saved to {RESULTS_DIR}/output_rounding_results.json")

    # Summary table to stdout
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (output rounding at inference)")
    print("=" * 100)
    hdr = f"{'Arch':<12} {'Seed':>4} {'Bits':>5} {'OK?':>4} {'Logit Loss':>14} {'Wt Bits':>9} {'SVD Bits':>10} {'Emp Bits':>10}"
    print(hdr)
    print("-" * 100)
    for r in results:
        ll = f"{r['logit_loss']:.2e}" if r.get('logit_loss') is not None else "FAIL"
        wb = f"{r['weight_bits']:.1f}" if r.get('weight_bits') is not None else "-"
        sb = f"{r['svd_bits']:.1f}" if r.get('svd_bits') is not None else "-"
        eb = f"{r['empirical_bits']:.1f}" if r.get('empirical_bits') is not None else "-"
        ok = "Y" if r['extraction_success'] else "N"
        print(f"{r['arch']:<12} {r['seed']:>4} {r['bits']:>5} {ok:>4} {ll:>14} {wb:>9} {sb:>10} {eb:>10}")


def _save(results):
    path = os.path.join(RESULTS_DIR, "output_rounding_results.json")
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
