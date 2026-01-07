# OdeMMHPlanner.jl

Welcome to the documentation for **OdeMMHPlanner.jl**. This package provides a framework for **uncertainty-aware learning and planning** in dynamical systems with unknown dynamics and infrequent output measurements.

OdeMMHPlanner implements the method described in:
> **Learning Dynamics from Infrequent Output Measurements for Uncertainty-Aware Optimal Control**
> *Robert Lefringhausen, Theodor Springer, Sandra Hirche*
> arXiv:2512.08013 (2025)
> <https://arxiv.org/abs/2512.08013>

## Overview

The package targets control problems in which the system dynamics are unknown and the state is only partially observed through infrequent and noisy output measurements.
Rather than identifying a single nominal model, OdeMMHPlanner explicitly represents uncertainty over both the system dynamics and the latent state trajectory.

The approach follows a Bayesian workflow:
1. **Learning:** A Marginal Metropolis–Hastings (MMH) sampler, equipped with a numerical ODE solver, is used to sample from the posterior distribution over unknown dynamics and latent state trajectories, conditioned on infrequent input–output measurements.
2. **Planning:** The resulting posterior samples are propagated through the system dynamics and used to formulate a scenario-based optimal control problem, yielding control inputs that explicitly account for model uncertainty.

By propagating uncertainty from system identification into the control design, the framework enables principled uncertainty quantification and safer decision-making compared to point-estimate-based approaches.

## Installation

This package is not registered in the General registry.
Clone the repository and instantiate the environment locally:

```bash
git clone https://github.com/TUM-ITR/ode-mmh-planner.git
cd ode-mmh-planner
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Getting Started

The following sections provide a structured entry point into the package:

- **[Inference and Sampler Tuning](@ref sampling)**: Introduces the Bayesian learning problem underlying OdeMMHPlanner. This section explains how unknown dynamics and latent state trajectories are inferred from infrequent input–output data using the Marginal Metropolis–Hastings sampler, and how to tune and diagnose the sampler to obtain reliable posterior samples.
- **[Optimal Control](@ref optimal-control)**: Demonstrates how the inferred posterior models are used to formulate and solve a scenario-based optimal control problem that explicitly accounts for model uncertainty.

## Reproducing the Experiments

The numerical results reported in the paper are generated using the experiment scripts provided in the repository.  
An overview of these experiments, including sampler diagnostics, single-run optimal control, and a Monte Carlo study, is given on the [Experiments](@ref experiments) page.