# [Experiments](@id experiments)

This repository contains three main experiment scripts used in the simulation study presented in Section “Simulation” of the paper. The experiments evaluate the proposed framework on a glucose regulation task based on the Bergman minimal model under infrequent and noisy glucose measurements.

All experiment scripts are located in the `experiments/` directory.

## Sampler Tuning and Diagnostics

**Script:** `experiments/sampler_tuning.jl`

This experiment runs the MMH sampler for a large number of iterations without thinning and analyzes the resulting Markov chain to assess sampling performance. In particular, it computes diagnostics such as autocorrelation functions (ACFs), which are used to determine an appropriate thinning interval for subsequent experiments.

The outputs produced by this script are used to generate **Figure 1** in the paper.

For a detailed explanation of the tuning and diagnostic procedure, see [Inference and Sampler Tuning](@ref sampling).

## Scenario-Based Optimal Control (Single Run)

**Script:** `experiments/optimal_control.jl`

This experiment demonstrates the full pipeline on a single representative run:

1. Bayesian inference of model parameters and latent state trajectories from infrequent measurements (MMH sampler with ODE integration),
2. scenario-based optimal control using posterior samples,
3. evaluation by simulating the true system forward and comparing against baselines (e.g., nominal + EKF).

The outputs produced by this script are used to generate **Figure 2** in the paper.

For a conceptual explanation of the scenario OCP setup, see [Optimal Control](@ref optimal-control).

## Monte Carlo Study

**Folder:** `experiments/monte_carlo/`

This study repeats the single-run pipeline across **100 independent simulation runs**, sampling the true model parameters and initial conditions from the prior distributions. It is used to assess robustness and performance statistically.

The aggregated results of this study are summarized in **Table 2** of the paper.

Detailed instructions, including information on running the experiments using SLURM job scripts, are provided in `experiments/monte_carlo/README.md`.