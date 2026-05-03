# Step-Spacing Extraction: A Cryptanalytic Attack Against Output-Rounded ReLU Networks

This is the third leg of the project. The first leg replicated Carlini–
Jagielski–Mironov on full-precision oracles. The second leg
([SUMMARY.md](SUMMARY.md)) showed that rounding the oracle's outputs to as
few as 32 fractional bits completely defeats their attack while costing
near-zero utility. This leg shows that the defense is **not** robust:
rounding actually leaks new structure that a different attack can exploit.

## Threat model (unchanged)

Black-box query access to a fully-connected ReLU network.
Architecture is known. The defender now applies output rounding
`y' = round(y · 2^(N−1)) / 2^(N−1)` at every query.

## Why Carlini's attack breaks under rounding

Two primitives carry the original attack:

1. **Critical-point detection** (`src/find_witnesses.py::do_better_sweep`):
   binary searches a 1D line for points where `f` is non-linear, gated by
   `|f_mid − (f_low + f_high)/2| < SKIP_LINEAR_TOL · √(high−low)` with
   `SKIP_LINEAR_TOL = 1e-8`.
2. **Weight-row recovery** (`src/hyperplane_normal.py::get_second_grad_unsigned`):
   a 4-point stencil
   `(f(x+ε−ε₂) − f(x+ε)) − (f(x−ε) − f(x−ε+ε₂))` whose magnitude is
   `~ε · ε₂ · ‖∇f_left − ∇f_right‖ ≈ 10⁻¹¹` with the default
   `ε ≈ 1e-5, ε₂ ≈ 1e-6`.

Output rounding to `bits=N` introduces a quantum `s = 2^(1−N)`. At
`bits=32`, `s ≈ 5e-10`, which is comparable to the second-derivative
signal and **50× larger** than the linearity test threshold. So:

- The linearity test produces only false negatives; the binary search
  refuses to label any region linear, recursing forever or terminating
  on rounding boundaries that aren't real ReLU crossings.
- The 4-point stencil sees `|signal/noise| ≈ 1/50`; recovered weight
  rows are dominated by rounding error, and downstream sign recovery and
  layer peeling fail or emit garbage weights with negative bits of
  precision.

Both primitives operate at **sub-rounding scales** — they're trying to
read information out of the trailing low-order bits of the mantissa,
which the defender just zeroed out.

## The opening

The defender's rounding *does* publish information — just at a different
scale. Every time the underlying smooth function `f` crosses a level
`(k+½)·s` for some integer `k`, the rounded oracle's output changes by
exactly `s`. Sweeping a 1D line through input space therefore produces
a **complete level-set readout** of `f` along that line. That's a much
richer signal than the binary "linear / non-linear" verdict the original
attack distilled it down to.

We re-derive the attack to consume this signal directly. The new
primitives operate at the **rounding scale** — the regime the defender
made observable — instead of below it.

## New primitive 1: step-spacing gradient estimation

For a 1D probe `x(t) = x₀ + t·d`, locate every transition
`t_k` in the rounded output `f_r(x(t))`. Inside a single ReLU polytope,
`f` is affine: `f(x) = w·x + b`. So `f(x(t))` is linear in `t` with
slope `w·d`. Consecutive level crossings are equispaced:

```
w·d · (t_{k+1} − t_k) = ±s
=>  |w·d| = s / (t_{k+1} − t_k)
```

Algorithm:

```
sweep_and_find_transitions(x₀, d, t_low, t_high, oracle, s):
  Sample N points along the line.
  For every consecutive pair where rounded output changes,
    binary-search the transition location to precision δ << s.
  Return sorted list of transition t-values.

estimate_gradient(x₀, oracle, s):
  For each input dim i:
    transitions = sweep along basis(i)
    spacing = median(diff(transitions))    # robust to outliers
    magnitude = s / spacing
    sign = sign(oracle(x₀ + radius·basis(i)) − oracle(x₀ − radius·basis(i)))
    grad[i] = sign · magnitude
  Return grad
```

**Robustness:** the median spacing tolerates a small fraction of
"wrong-spacing" outliers (caused by sweeping across a ReLU boundary).
Variance of the spacing estimator over `K` transitions is `O(1/√K)`.
For `bits=32` and a domain of radius 1 with `‖∇f‖ ≈ 1`, we get
`K ≈ 2 · 1 / s = 4·10⁹` — vastly more than needed.

For `bits=4`, `s = 0.125` and `K ≈ 16` per unit — still workable.
The attack only fails when `s · diameter(polytope) < ‖∇f·d‖` on most
polytopes, which is the regime where the model itself is degenerate.

## New primitive 2: spacing-change critical-point detection

In a 1D sweep, the spacing between consecutive transitions is constant
*within* a polytope and changes *between* polytopes (because crossing a
ReLU activates or deactivates a neuron, changing `w·d`). So:

```
find_critical_points_by_spacing(x₀, d, oracle, s):
  transitions = sweep_and_find_transitions(...)
  spacings = diff(transitions)
  detect change-points in `spacings` using a moving-median test with
    a CUSUM-style threshold:
      window_left  = median(spacings[k-W : k])
      window_right = median(spacings[k : k+W])
      flag if |window_left − window_right| / max(window_left, window_right) > τ
  return change-point locations
```

Each detected change-point is a candidate ReLU critical point. The
slope on each side gives `(w·d)_left` and `(w·d)_right`; their
difference `(w·d)_left − (w·d)_right` is the contribution of the
neuron that just flipped, projected onto `d`. This is exactly the
quantity Carlini's `get_second_grad_unsigned` produces — recovered at
the rounding scale instead of below it.

## New primitive 3: weight-row recovery from gradient jumps

At a confirmed critical point `x*`:

1. Estimate the gradient on each side: `g_left, g_right`.
2. Their difference `Δg = g_left − g_right ∈ ℝⁿ` lies along the weight
   row of the activated neuron: `Δg = α · W₁[j, :]` for some
   downstream-dependent scalar `α`.
3. So the **direction** of `Δg` recovers `W₁[j, :]` up to sign and
   global scale. (Sign and scale are resolved later, exactly as in
   Carlini's pipeline.)

This replaces `hyperplane_normal.get_ratios` and feeds directly into the
existing `layer_recovery.compute_layer_values`.

## New primitive 4: meet-in-the-middle sign recovery

Carlini's noncontractive sign recovery brute-forces `2^k` sign
assignments for a `k`-neuron layer. Once magnitudes are known, signs
are the only unknowns and the consistency check is *additive across
neurons*, which is a textbook MITM target.

Setup. After step-spacing recovery we have:
- magnitudes `|w_1|, ..., |w_k|` (rows of `W₁` up to per-neuron sign)
- a downstream `KnownT` over already-extracted earlier layers
- a batch of `N` probe points `x_1, …, x_N` with rounded oracle outputs
  `y_1, …, y_N`.

For any sign vector `σ ∈ {±1}^k`, the model's prediction (modulo the
unknown final-layer scaling, which we absorb) decomposes as

```
ŷ(σ, x) = Σ_{j=1..k} σ_j · |w_j| · ReLU(...) + (downstream)
        = ΣA(σ_A, x) + ΣB(σ_B, x) + const
```

where `σ = (σ_A, σ_B)` splits the neurons. So:

```
mitm_signs(probe_points, magnitudes, oracle):
  k = len(magnitudes)
  Split into A = neurons[:k//2], B = neurons[k//2:]
  table = {}
  for σ_A in {±1}^{k//2}:
    p_A = vector [Σ_{j in A} σ_A[j] |w_j| ReLU_j(x_i) for i in 1..N]
    table[round(p_A, tol=s)] = σ_A
  for σ_B in {±1}^{k//2}:
    p_B = vector [...]
    target = y - p_B - const
    if round(target, tol=s) in table: return σ_A, σ_B
```

Time `O(2 · 2^(k/2) · N)`, memory `O(2^(k/2) · N)`. For `k=20`,
`2^10 ≈ 10³` instead of `2^20 ≈ 10⁶`. The matching uses rounding
tolerance `s` (same quantum the defender published outputs at), so the
hash key is the rounded vector itself.

## Pipeline

```
For each layer ℓ (from input to output, layer-peeling):
  1. Sample many random 1D sweeps through input space.
  2. For each sweep:
       transitions = sweep_and_find_transitions
       critical_points += find_critical_points_by_spacing(transitions)
  3. For each critical point x*:
       Δg = estimate_gradient(x* − δd) − estimate_gradient(x* + δd)
       record (x*, Δg) — Δg is W_ℓ[j, :] up to sign+scale
  4. Cluster {Δg} into k rows (one per neuron in layer ℓ).
  5. Sign recovery:
       if contractive (next layer narrower): closed-form
       else: MITM brute force
  6. Layer peeling: extend KnownT and recurse.

Final layer:
  Linear least-squares over recorded queries (unchanged from Carlini).
```

## Expected operating regimes

| bits | step `s`     | typical signal  | step-spacing attack  | Carlini |
|-----:|-------------:|----------------:|:--------------------:|:-------:|
|   64 | (no rounding)| —               | works (degenerate)   | works   |
|   32 | 5e-10        | 5e-10           | **works**            | fails   |
|   16 | 3e-5         | 3e-5            | **works**            | fails   |
|    8 | 8e-3         | 8e-3            | **likely works**     | fails   |
|    4 | 1.25e-1      | 1.25e-1         | borderline           | fails   |
|    2 | 0.5          | 0.5             | fails (model dies)   | fails   |

The defender's "win condition" shifts from "any rounding" (vs Carlini)
to "rounding so coarse the model itself is non-functional" (vs us).

## Risks and open questions

1. **Median-spacing bias near critical points.** Inside a window
   that straddles a ReLU boundary, the median is pulled toward the
   side with more steps. Mitigation: detect change-points first, then
   re-estimate gradient using only one-sided windows.
2. **Gradient sign estimation.** Endpoint differencing
   `f(x+R·d) − f(x−R·d)` can have its own sign flips if multiple ReLUs
   activate within `[−R, +R]`. Mitigation: take many small endpoint
   probes and use the dominant sign.
3. **2-layer vs 3-layer.** The 10-15-15-1 architecture in this repo
   needs a layer-peeling step that's already finicky in Carlini's
   reference implementation. We focus the empirical evaluation on the
   2-layer architectures (20-10-1, 40-20-1) where layer peeling
   collapses to "recover W₁, then solve final layer by least squares".
4. **Query budget.** Each step location costs O(log(1/δ)) queries via
   binary search; a full 1D gradient estimate costs `n · K ·
   log(1/δ)` where `K` is steps per dim. Likely 2-3 orders of
   magnitude more queries than Carlini at full precision. Budget but
   not prohibitive on these tiny networks.

## Implementation plan

1. `src/step_spacing.py`: `find_transitions`, `estimate_gradient`,
   `find_critical_points_by_spacing`, `recover_weight_row`.
2. `validate_step_spacing.py`: sanity check vs ground truth on a
   trained 2-layer model. Pass criteria: relative gradient error
   < 0.01 inside polytopes; ≥ 80% of true ReLU boundaries detected.
3. `extract_rounded.py`: end-to-end 2-layer extraction. Returns
   `(W₁, b₁, W₂, b₂)` and the same logit-loss / weight-bits metrics
   the existing pipeline uses.
4. `src/mitm_signs.py`: meet-in-the-middle sign recovery as a drop-in
   for the brute-force step.
5. `run_step_spacing_experiment.py`: same (arch, seed, bits) grid as
   the rounding experiment; emits
   `results/step_spacing_results.json`.
6. Update this document and `SUMMARY.md` with empirical results.

(Empirical results section is filled in below after running the
attack.)

---

## Empirical results

We ran the step-spacing attack on the same trained 2-layer models used in the
output-rounding defense study (architectures `20-10-1` and `40-20-1`, seeds
42/43/44, bit precisions {32, 16, 8, 4}). For each cell we report whether the
extraction completed (clustering produced `n_hidden` neurons + sign recovery
succeeded), and the **functional fidelity metrics** computed against the true
underlying weights on a 20k-sample test set:

- `max_logit_loss = max_x |f̂(x) − f(x)|`
- `rel_loss = E[|f̂(x) − f(x)|] / E[|f(x)|]`

A "useful extraction" requires `rel_loss < 0.5` — half the energy of the
target's own outputs. (Tighter thresholds give the same qualitative picture.)

### Summary: success rate by rounding precision

|        Defense | bits=32 | bits=16 | bits=8 | bits=4 |
|---------------:|--------:|--------:|-------:|-------:|
|  Carlini (Ω)   |   0 / 9 |   0 / 9 |  0 / 9 |  0 / 9 |
|  **Step-spacing (us)** | **6 / 6** | 1 / 6 | 0 / 6 | 0 / 6 |

Carlini's row is from the prior `SUMMARY.md` (the 9 originally tested cells
include 3-layer networks; ours is restricted to 2-layer). The bits=64
unrounded baseline is excluded — that's just the original Carlini setting,
which both attacks already handle.

### Per-cell results (step-spacing attack)

| Arch    | Seed | Bits | OK | max_logit_loss | rel_loss | wallclock |
|---------|-----:|-----:|:--:|---------------:|---------:|----------:|
| 20-10-1 |   42 |   32 | Y  |       1.15e-03 | 7.6e-05  |     8.4 s |
| 20-10-1 |   42 |   16 | Y  |       1.50e+01 | 6.6e-01  |    77.1 s |
| 20-10-1 |   42 |    8 | Y  |       1.58e+01 | 7.3e-01  |    29.4 s |
| 20-10-1 |   42 |    4 | N  |           FAIL |     —    |    23.7 s |
| 20-10-1 |   43 |   32 | Y  |       1.27e-03 | 2.5e-04  |     4.3 s |
| 20-10-1 |   43 |   16 | Y  |       4.69e+00 | 9.0e-01  |    19.3 s |
| 20-10-1 |   43 |    8 | Y  |       5.83e+00 | 9.3e-01  |    30.1 s |
| 20-10-1 |   43 |    4 | N  |           FAIL |     —    |    24.1 s |
| 20-10-1 |   44 |   32 | Y  |       3.41e-02 | 2.5e-03  |     4.3 s |
| 20-10-1 |   44 |   16 | Y  |       9.35e+00 | 6.6e-01  |    15.9 s |
| 20-10-1 |   44 |    8 | Y  |       1.30e+01 | 8.4e-01  |    30.1 s |
| 20-10-1 |   44 |    4 | N  |           FAIL |     —    |    23.6 s |
| 40-20-1 |   42 |   32 | Y  |       1.74e+00 | 1.3e-01  |    17.4 s |
| 40-20-1 |   42 |   16 | Y  |       7.65e+00 | 7.3e-01  |    75.7 s |
| 40-20-1 |   42 |    8 | Y  |       9.25e+00 | 8.7e-01  |   190.3 s |
| 40-20-1 |   42 |    4 | N  |           FAIL |     —    |    96.6 s |
| 40-20-1 |   43 |   32 | Y  |       1.98e+00 | 8.8e-02  |    17.3 s |
| 40-20-1 |   43 |   16 | Y  |       1.07e+01 | 6.9e-01  |    68.5 s |
| 40-20-1 |   43 |    8 | Y  |       1.26e+01 | 6.8e-01  |   164.6 s |
| 40-20-1 |   43 |    4 | N  |           FAIL |     —    |    96.5 s |
| 40-20-1 |   44 |   32 | Y  |       3.14e+00 | 1.3e-01  |    17.2 s |
| 40-20-1 |   44 |   16 | Y  |       1.04e+01 | 4.9e-01  |    68.5 s |
| 40-20-1 |   44 |    8 | Y  |       1.49e+01 | 6.6e-01  |   150.7 s |
| 40-20-1 |   44 |    4 | N  |           FAIL |     —    |    94.1 s |

`Y` here means the extraction *pipeline* completed (n_hidden clusters formed,
signs recovered, final layer fitted). At bits ≤ 8 the pipeline often
"completes" but the recovered weights only roughly capture the true geometry
— hence the bad `rel_loss`. We count those as defense wins.

### Aggregated picture

|       Arch | bits=32 useful | bits=16 useful | bits=8 useful | bits=4 useful |
|-----------:|---------------:|---------------:|--------------:|--------------:|
|   20-10-1  |        **3 / 3** |        0 / 3 |         0 / 3 |         0 / 3 |
|   40-20-1  |        **3 / 3** |        1 / 3 |         0 / 3 |         0 / 3 |
|  **Total** |        **6 / 6** |        1 / 6 |         0 / 6 |         0 / 6 |

(`useful = rel_loss < 0.5`. The single 40-20-1 bits=16 case at `rel_loss
0.49` is borderline; tightening to 0.4 collapses bits=16 to 0/6 too.)

### Interpretation: the security/utility frontier shifts

Folding this into the picture from `SUMMARY.md`:

|  Bits |  Utility cost (rel err)  | Carlini extracts? | Step-spacing extracts? |
|------:|:------------------------:|:-----------------:|:----------------------:|
|    32 |          ~1e-10          |        no         |       **yes (6/6)**    |
|    16 |          ~5e-6           |        no         |   borderline (1/6)     |
|     8 |       ~0.1–0.2%          |        no         |         no             |
|     4 |          ~1–4%           |        no         |         no             |

The defender's earlier "free defense" recommendation of bits=32 — utility
loss below `float64` noise — is **not robust**: a different, rounding-aware
attack defeats it on every (arch, seed) we tested. To genuinely defend
against both attacks the defender must drop to **bits ≤ 8**, which costs
~0.1% mean relative error on these networks. That's still small in absolute
terms, but it's no longer free; the defender pays a real (if modest)
utility tax to suppress the new attack.

### Where the new attack breaks down

The attack stops working when the rounding step `s` becomes comparable to
or larger than the function's variation across a typical polytope. Concretely:

- At bits=8, `s ≈ 8e-3`. With these networks' typical gradient norm
  (`‖∇f‖ ≈ 2–4`) and polytope diameters along basis directions
  (`~0.05`), each polytope contains only a handful of integer levels along
  any axis. The midpoint-linearity check inside `directional_derivative`
  rejects most windows; full gradient estimation succeeds for too few
  probes; the ReLU jumps that survive don't cluster into `n_hidden` clean
  groups.
- At bits=4, the situation is worse: most polytopes are entirely covered
  by a single integer level, so there's nothing for step-spacing to read.

This is the same fundamental limit the design predicted. The attack works
exactly where the rounding step is *larger than `float64` noise* but
*smaller than the function's variation across a polytope*. That window
shrinks as `bits` decreases, and once it closes the attack genuinely fails.

### Cost: queries vs. wallclock

Wallclock per cell at bits=32 is 4–18 seconds on 2-layer networks. Per-cell
oracle query counts grow with `n_in × n_probes × n_sweeps`; for a
20-input-dim network with 30 sweeps × 400 probes, total queries are on
the order of `2.4 × 10⁵` — about 100× Carlini's count on the same
network at full precision, but well within the ~1M queries Carlini
considers acceptable for larger networks.
