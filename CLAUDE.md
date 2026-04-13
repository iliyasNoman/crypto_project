# Experiment: Quantization vs. Cryptanalytic Model Extraction

## Setup
1. Clone https://github.com/google-research/cryptanalytic-model-extraction
2. Install deps: `pip install numpy scipy jax jaxlib matplotlib networkx`

## Experiment Plan
We want to show that post-training quantization of the target model degrades
the cryptanalytic extraction attack from Carlini et al. (CRYPTO'20).

### Steps:
1. Train a target model: `python3 train_models.py 10-20-20-1 42`
2. Run the baseline extraction (full float64): `python3 extract.py 10-20-20-1 42`
3. Evaluate baseline with `check_solution_svd.py`
4. **Quantization loop**: For each bit-width in [32, 16, 8, 4]:
   - Load the trained model from `models/`
   - Quantize weights to that bit-width (uniform quantization: 
     `q = np.round(w * (2^(bits-1))) / (2^(bits-1))`)
   - Save the quantized model back in the same format
   - Run `extract.py` against the quantized model
   - Run `check_solution_svd.py` and record (ε, δ)-functional equivalence metrics
5. Plot results: bit-width on x-axis, extraction fidelity (log-scale ε) on y-axis
6. Save plot to `results/quantization_vs_extraction.png`

### Key insight
The attack exploits the fact that ReLU nets are piecewise linear and uses 
finite differences to find critical points with high precision. Quantization 
introduces discontinuities and reduces the precision of the oracle's outputs,
which should degrade the binary search for critical points and the 
finite-difference gradient estimates.

### Notes
- The oracle function must be modified to evaluate the quantized model
- Keep the architecture knowledge assumption intact (attacker knows architecture)
- Try multiple architectures: 10-20-20-1, 80-40-20-1, 784-128-1
- Use seeds 42, 43, 44 for statistical significance
