# Regime-Coherence Toy Model (v0.33)

This repository contains the minimal Python pipeline used to generate toy-model Monte Carlo samples and compute the geometric diagnostics reported in the manuscript (PCA transverse compression and intrinsic-dimension suppression), including robustness checks: band-width (ε) sweep, proposal-size (N) sweep, HDBSCAN negative control, and a three-timescale (3τ) extension.

## Repository layout (branches)

This repository uses separate branches for clarity:

- **main**: project overview + `README.md`, `requirements.txt`, licensing, and general notes  
- **scripts**: runnable analysis scripts (baseline / sweeps / three-τ)  
  → https://github.com/erikczinege350-ux/Regime-Coherence-as-a-Relational-Alternative-to-Cosmological-Fine-Tuning/tree/scripts
- **outputs**: generated figures and diagnostic dumps produced by the scripts  
  → https://github.com/erikczinege350-ux/Regime-Coherence-as-a-Relational-Alternative-to-Cosmological-Fine-Tuning/tree/outputs

If you just want to run the code, switch to the `scripts` branch. If you want to inspect the generated figures, switch to the `outputs` branch.

## Contents

- `scripts/`
  - `run_baseline_v033_maintext_bw.py`  
    Baseline (2-timescale) pipeline: sampling + gates + PASS/FAIL + PASS-fitted PCA in global z-score space + figures + ID table + diagnostic dump.
  - `controls/`  
    Negative controls (e.g., HDBSCAN on FAIL in PASS-fitted PCA space).
  - `sweeps/`  
    Robustness sweeps (epsilon sweep, N sweep) used to populate Appendix D tables.
  - `three-tau/`  
    Three-timescale (3τ) variant used for Appendix D.5.

- `outputs/`  
  Generated figures and summary tables (not tracked by git; see `.gitignore`).

## Requirements

Install dependencies:
```bash

pip install -r requirements.txt
