# API Reference

This page documents the main functions and types provided by `OdeMMHPlanner`.

## Sampling

Functions and types related to Bayesian learning of system dynamics and latent state trajectories using the Marginal Metropolis–Hastings (MMH) sampler.

```@docs
MMH_sample
ODE_MMH
staged_ODE_MMH
```

## Analysis and Diagnostics

Utilities for analyzing and diagnosing the MCMC chains produced by the MMH sampler.

```@docs
compute_autocorrelation
compute_ess
compute_gelman_rubin
```

## Optimal Control

Functions for formulating and solving the scenario-based optimal control problem using posterior samples obtained from MMH.

```@docs
solve_MMH_OCP
```
