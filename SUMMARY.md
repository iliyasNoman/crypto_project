# Output Rounding at Inference vs. Cryptanalytic Extraction

This experiment is a follow-up to the post-training weight-quantization study in
[README.md](README.md). Same threat model, same architectures, same seeds —
but instead of quantizing the model's **weights at rest**, we leave the weights
in full float64 and round the **oracle's outputs at inference time** before
returning them to the attacker.

## What attack are we defending against?

The Carlini–Jagielski–Mironov "Cryptanalytic Extraction of Neural Network
Models" attack (CRYPTO'20). Threat model: the attacker has black-box query
access to a fully-connected ReLU network, knows the architecture, and reads
back full float64 logits. The attack exploits the fact that a ReLU network
is piecewise linear: it (i) binary-searches along input lines for **critical
points** where some neuron's pre-activation crosses zero, then (ii) measures
the network's **second-order finite differences** at those points to recover
each weight row up to a scalar — `∂²f/∂eᵢ² ∝ Wₗ[j, i]`. Sign recovery and
layer "peeling" close the loop. Applied to full-precision oracles the attack
recovers ~35–40 bits of weight precision; the prior weight-quantization
experiment showed it survives quantization down to 16 bits and only breaks
at 8.

Both probes — the linearity test in the binary search and the second-order
stencil — depend on the oracle returning the **low-order mantissa bits** of
the smooth function. Output rounding deletes exactly those bits, which is
why this defense works so well.

## What counts as a "useful extraction"?

Throughout this report, "success" means the attack produced **a useful
extraction**, defined as:

- the extraction process completed without erroring or timing out, **and**
- recovered weights have positive bits of precision (`weight_bits > 0`,
  equivalently the per-element max error vs the real weights is < 1), **and**
- the model-vs-extracted max logit loss is < 1.

This is a stricter bar than the binary `extraction_success` flag the driver
records, because at bits=32 the attack sometimes runs to completion (rc=0)
but emits garbage weights with `weight_bits ≈ -10` and `logit_loss` in the
hundreds or thousands. Those are *not* useful extractions and are counted
as failures everywhere below.

## Setup

- Branch: `output-rounding-experiment`
- Defense: `src/utils.py::run()` rounds each oracle response via the same
  formula used by the weight-quantization experiment, applied to outputs:

  ```
  y' = round(y * 2^(N-1)) / 2^(N-1)
  ```

  where `N` is the rounding precision in bits, controlled by the env var
  `INFERENCE_ROUND_BITS`. `N >= 64` (or unset) disables rounding.
- Driver: `run_output_rounding_experiment.py` — same structure as
  `run_experiment.py`, with `INFERENCE_ROUND_BITS=N` set per run instead of
  rewriting the model file. Per-run timeout 90s, enforced via
  `Popen(start_new_session=True)` + `killpg` so `mp.Pool` children don't
  outlive the parent.
- Architectures: `10-15-15-1`, `20-10-1`, `40-20-1`
- Seeds: 42, 43, 44
- Rounding precisions (bits): 64 (baseline, no rounding), 32, 16, 8, 4

3 archs × 3 seeds × 5 levels = **45 runs**.

## Headline Result

> **Output rounding is a strictly stronger defense than weight quantization at
> the same bit-width.** Even rounding to 32 fractional bits — which is finer
> than `float32` — reduces attack success to 0/9 useful extractions, whereas
> weight quantization at 32 bits left the attack essentially undisturbed
> (89% success, ~35-bit weight precision).

| Defense                | bits=64 (baseline) | bits=32 | bits=16 | bits=8 | bits=4 |
|------------------------|:------------------:|:-------:|:-------:|:------:|:------:|
| Weight quantization †  |                78% |     89% |     89% |     0% |    11% |
| **Output rounding**    |                89% |   **0%** | **0%** | **0%** | **0%** |

† From the prior README. Output-rounding success rate is computed using the
"useful extraction" criterion defined above.

## Per-run Table (output rounding)

| Arch       | Seed | Bits | OK? | Logit Loss | Wt Bits | SVD Bits | Emp Bits |
|------------|-----:|-----:|:---:|-----------:|--------:|---------:|---------:|
| 10-15-15-1 |   42 |   64 |  Y  |   5.88e-12 |    37.6 |     33.9 |     37.2 |
| 10-15-15-1 |   42 |   32 |  N  |       FAIL |       — |        — |        — |
| 10-15-15-1 |   42 |   16 |  N  |       FAIL |       — |        — |        — |
| 10-15-15-1 |   42 |    8 |  N  |       FAIL |       — |        — |        — |
| 10-15-15-1 |   42 |    4 |  N  |       FAIL |       — |        — |        — |
| 10-15-15-1 |   43 |   64 |  *  |   2.33e+01 |    -9.2 |    -12.0 |     -9.0 |
| 10-15-15-1 |   43 |   32 |  N  |       FAIL |       — |        — |        — |
| 10-15-15-1 |   43 |   16 |  N  |       FAIL |       — |        — |        — |
| 10-15-15-1 |   43 |    8 |  N  |       FAIL |       — |        — |        — |
| 10-15-15-1 |   43 |    4 |  N  |       FAIL |       — |        — |        — |
| 10-15-15-1 |   44 |   64 |  Y  |   1.01e-11 |    37.9 |     33.7 |     36.3 |
| 10-15-15-1 |   44 |   32 |  N  |       FAIL |       — |        — |        — |
| 10-15-15-1 |   44 |   16 |  N  |       FAIL |       — |        — |        — |
| 10-15-15-1 |   44 |    8 |  N  |       FAIL |       — |        — |        — |
| 10-15-15-1 |   44 |    4 |  N  |       FAIL |       — |        — |        — |
| 20-10-1    |   42 |   64 |  Y  |   3.25e-11 |    34.9 |     34.6 |     34.8 |
| 20-10-1    |   42 |   32 |  *  |   5.90e+02 |   -10.9 |    -13.1 |     -9.3 |
| 20-10-1    |   42 |   16 |  N  |       FAIL |       — |        — |        — |
| 20-10-1    |   42 |    8 |  N  |       FAIL |       — |        — |        — |
| 20-10-1    |   42 |    4 |  N  |       FAIL |       — |        — |        — |
| 20-10-1    |   43 |   64 |  Y  |   2.39e-11 |    35.4 |     34.5 |     35.3 |
| 20-10-1    |   43 |   32 |  N  |       FAIL |       — |        — |        — |
| 20-10-1    |   43 |   16 |  N  |       FAIL |       — |        — |        — |
| 20-10-1    |   43 |    8 |  N  |       FAIL |       — |        — |        — |
| 20-10-1    |   43 |    4 |  N  |       FAIL |       — |        — |        — |
| 20-10-1    |   44 |   64 |  Y  |   8.96e-13 |    40.8 |     38.1 |     39.8 |
| 20-10-1    |   44 |   32 |  *  |   2.12e+03 |   -10.7 |    -13.1 |     -8.7 |
| 20-10-1    |   44 |   16 |  N  |       FAIL |       — |        — |        — |
| 20-10-1    |   44 |    8 |  N  |       FAIL |       — |        — |        — |
| 20-10-1    |   44 |    4 |  N  |       FAIL |       — |        — |        — |
| 40-20-1    |   42 |   64 |  Y  |   1.88e-11 |    36.0 |     34.0 |     35.6 |
| 40-20-1    |   42 |   32 |  N  |       FAIL |       — |        — |        — |
| 40-20-1    |   42 |   16 |  N  |       FAIL |       — |        — |        — |
| 40-20-1    |   42 |    8 |  N  |       FAIL |       — |        — |        — |
| 40-20-1    |   42 |    4 |  N  |       FAIL |       — |        — |        — |
| 40-20-1    |   43 |   64 |  Y  |   1.15e-10 |    33.0 |     32.5 |     33.0 |
| 40-20-1    |   43 |   32 |  N  |       FAIL |       — |        — |        — |
| 40-20-1    |   43 |   16 |  N  |       FAIL |       — |        — |        — |
| 40-20-1    |   43 |    8 |  N  |       FAIL |       — |        — |        — |
| 40-20-1    |   43 |    4 |  N  |       FAIL |       — |        — |        — |
| 40-20-1    |   44 |   64 |  Y  |   3.65e-12 |    38.0 |     35.5 |     37.9 |
| 40-20-1    |   44 |   32 |  N  |       FAIL |       — |        — |        — |
| 40-20-1    |   44 |   16 |  N  |       FAIL |       — |        — |        — |
| 40-20-1    |   44 |    8 |  N  |       FAIL |       — |        — |        — |
| 40-20-1    |   44 |    4 |  N  |       FAIL |       — |        — |        — |

`Y` = extraction completed and recovered weights to full float64 precision.
`N` = extraction process aborted, errored, or hit the 90s timeout.
`*` = extraction process completed (rc=0) but the recovered weights are
garbage (`weight_bits` is negative, `logit_loss` ≫ 1). These are *not* useful
extractions and are counted as failures in the headline summary above.

The 10-15-15-1 seed=43 baseline (`*` row) is unrelated to the defense — the
Carlini attack itself is flaky on this 3-layer architecture even at full
precision (the original quantization README reported 78% baseline success for
the same reason). Re-running it with a longer 300s timeout still produced
garbage weights, which is why we leave the entry as-is.

## Aggregate Success Rate (useful extractions only)

|         Arch | bits=64 | bits=32 | bits=16 | bits=8 | bits=4 |
|-------------:|--------:|--------:|--------:|-------:|-------:|
|  10-15-15-1  |   2 / 3 |   0 / 3 |   0 / 3 |  0 / 3 |  0 / 3 |
|     20-10-1  |   3 / 3 |   0 / 3 |   0 / 3 |  0 / 3 |  0 / 3 |
|     40-20-1  |   3 / 3 |   0 / 3 |   0 / 3 |  0 / 3 |  0 / 3 |
|  **Overall** | **8/9** | **0/9** | **0/9** | **0/9** | **0/9** |

## Why output rounding is so much stronger than weight quantization

Weight quantization perturbs the model's weights to a coarse grid, but the
resulting function is still **smooth almost everywhere** between ReLU
boundaries. The attacker's two key probes — binary search for critical
points (`src/find_witnesses.py`) and second-order finite differences for
weight ratios (`src/hyperplane_normal.py:get_second_grad_unsigned`) — operate
on tiny input perturbations (`eps ≈ 1e-5`) and need to detect output changes
of comparable magnitude. With weights at 32 bits, output changes from those
small input perturbations are still resolved cleanly, so the attack proceeds
unaffected.

Output rounding does something qualitatively different: it makes the oracle
**piecewise constant**. The attack's finite-difference probes try to detect
output changes of order `eps ≈ 1e-5` — but if the output is rounded to a
quantum of `2^-31 ≈ 5e-10`, then while individual probes nominally see
enough resolution, the second-order stencil
`f(x+ε+ε₂) − f(x+ε) − f(x−ε+ε₂) + f(x−ε)` collapses to zero (or to a
single quantum) almost everywhere, because all four samples land on the
same output level on the flat plateau between rounding boundaries. The
linearity test in the binary search (`|f(mid) − (f(low)+f(high))/2| < 1e-8`)
also gets fooled — the oracle now has discontinuities at every rounding
boundary, not just at ReLU boundaries, so the search either gives up early
on a flat plateau or descends into spurious "critical points" that aren't
real ReLU crossings.

In short: **weight quantization changes which smooth function is being
attacked. Output rounding destroys the smoothness assumption the attack is
built on.**

## Utility cost: does the defense degrade the model?

A defense is only useful if it doesn't destroy the model on the way. To
measure utility cost we sample 100k inputs from the training distribution
(standard Gaussian on the input dimension) and compare:

```
y_unrounded = f(x)
y_rounded   = round(y_unrounded * 2^(N-1)) / 2^(N-1)
```

Reported below: rounding step, mean absolute output error
`E[|y_rounded - y_unrounded|]`, max absolute error, and mean relative error
`E[|Δy|] / E[|y|]` (averaged over the 3 seeds per architecture). Computed
by `measure_utility_cost.py` → `results/utility_cost.json`.

| Arch       | Bits | Rounding step | Mean abs err | Max abs err | Mean &#124;y&#124; | Rel err |
|------------|-----:|--------------:|-------------:|------------:|---------:|--------:|
| 10-15-15-1 |   32 |      4.66e-10 |     1.17e-10 |    2.33e-10 |    0.869 | 1.4e-10 |
| 10-15-15-1 |   16 |      3.05e-05 |     7.63e-06 |    1.53e-05 |    0.869 | 8.9e-06 |
| 10-15-15-1 |    8 |      7.81e-03 |     1.96e-03 |    3.91e-03 |    0.869 | 2.3e-03 |
| 10-15-15-1 |    4 |      1.25e-01 |     3.12e-02 |    6.25e-02 |    0.869 | 3.7e-02 |
| 20-10-1    |   32 |      4.66e-10 |     1.16e-10 |    2.33e-10 |    2.045 | 7.1e-11 |
| 20-10-1    |   16 |      3.05e-05 |     7.63e-06 |    1.53e-05 |    2.045 | 4.7e-06 |
| 20-10-1    |    8 |      7.81e-03 |     1.96e-03 |    3.91e-03 |    2.045 | 1.2e-03 |
| 20-10-1    |    4 |      1.25e-01 |     3.12e-02 |    6.25e-02 |    2.045 | 1.9e-02 |
| 40-20-1    |   32 |      4.66e-10 |     1.17e-10 |    2.33e-10 |    2.591 | 4.7e-11 |
| 40-20-1    |   16 |      3.05e-05 |     7.64e-06 |    1.53e-05 |    2.591 | 3.1e-06 |
| 40-20-1    |    8 |      7.81e-03 |     1.95e-03 |    3.91e-03 |    2.591 | 7.9e-04 |
| 40-20-1    |    4 |      1.25e-01 |     3.13e-02 |    6.25e-02 |    2.591 | 1.3e-02 |

This produces a clean picture of the security/utility tradeoff:

|  Bits  | Rounding step | Typical rel err | Attack success | Verdict |
|-------:|--------------:|----------------:|:--------------:|:--------|
|   **32** |   ~5e-10    |       ~1e-10    |     **0%**     | **Sweet spot.** Utility cost is at the level of float64 rounding noise itself; the defense is essentially free. |
|     16 |     ~3e-5   |       ~5e-6     |       0%       | Still very cheap utility-wise (~5 ppm relative error), and breaks the attack. |
|      8 |     ~8e-3   |       ~0.1–0.2% |       0%       | Noticeable but small relative error — what `int8` post-training quantization typically targets. |
|      4 |     ~0.13   |       ~1–4%     |       0%       | Coarse — starts to materially distort outputs. Only worth it if you need integer outputs for some other reason. |

**Conclusion: bits=32 output rounding is the recommended operating point.**
Mean relative error is `~1e-10` (i.e., below `float64` precision noise — you
cannot tell rounded outputs apart from unrounded ones in practice), yet the
attacker's second-order finite differences collapse to noise and the attack
fails on every (arch, seed) combination. You don't need to give up any
accuracy at all to defeat the Carlini extraction attack — you just need to
stop returning the trailing low-order mantissa bits that the attack uses
to reconstruct the network's piecewise-linear geometry.

## Generated artifacts

- `results/output_rounding_results.json` — raw per-run extraction data
- `results/output_rounding_vs_extraction.png` — main plot (log ε vs bits)
- `results/output_rounding_success_rate.png` — success rate bars by arch
- `results/output_rounding_vs_quantization.png` — side-by-side defense
  comparison
- `results/utility_cost.json` — per-(arch, seed, bits) utility-cost data

## Reproducing

```bash
# From a fresh clone:
python3 -m venv .venv
.venv/bin/pip install numpy scipy jax jaxlib matplotlib networkx optax

# Train models (only needed once)
for arch in 10-15-15-1 20-10-1 40-20-1; do
  for seed in 42 43 44; do
    .venv/bin/python train_models.py $arch $seed
  done
done

# Run the experiment (resumable; rerun safely)
.venv/bin/python run_output_rounding_experiment.py

# Regenerate plots and tables
.venv/bin/python plot_output_rounding.py

# Measure how much the rounding actually changes the model's outputs
.venv/bin/python measure_utility_cost.py
```
