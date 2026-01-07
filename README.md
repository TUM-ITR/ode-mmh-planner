<p align="center">
<img width="800" height="100" src="assets/logo_with_text.png">
</p>

[![Dev](https://img.shields.io/badge/docs-stable-blue?logo=Julia&logoColor=white)](https://TUM-ITR.github.io/ode-mmh-planner)

**OdeMMHPlanner** provides a framework for **uncertainty-aware learning and planning** in dynamical systems with unknown dynamics and infrequent output measurements.  
It implements the method described in:

> **Learning Dynamics from Infrequent Output Measurements for Uncertainty-Aware Optimal Control**
> *Robert Lefringhausen, Theodor Springer, Sandra Hirche*
> arXiv:2512.08013 (2025)
> <https://arxiv.org/abs/2512.08013>

## Overview

The package targets control problems in which the system dynamics are unknown and the state is only partially observed through infrequent output measurements.
Instead of relying on a single identified model, OdeMMHPlanner explicitly represents uncertainty over both the dynamics and the latent state trajectory.

The approach combines:

1. Bayesian learning via a Marginal Metropolis–Hastings (MMH) sampler equipped with a numerical ODE solver to sample from the posterior distribution over dynamics and latent states.
2. Scenario-based optimal control, where posterior samples are propagated to compute control inputs that are robust to model uncertainty.

This enables principled uncertainty quantification and safer decision-making compared to point-estimate-based methods.

For detailed instructions and examples, please refer to the **[Documentation](docs/src/index.md)**.

## Installation

This package is not registered. Clone and instantiate it locally:

```bash
git clone https://github.com/TUM-ITR/ode-mmh-planner.git
cd ode-mmh-planner
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Solvers

The results reported in the paper were obtained using proprietary **HSL linear solvers** (in particular `ma57`) for improved numerical performance.  
HSL solvers are available from <https://www.hsl.rl.ac.uk/> and can be obtained under a free academic license; installation instructions are provided on their website.

If HSL solvers are not available, the code automatically falls back to standard open-source solvers (e.g. MUMPS). While functional, this configuration may be slower and has not been exhaustively tested.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{lefringhausen2025learning,
  title={Learning Dynamics from Infrequent Output Measurements for Uncertainty-Aware Optimal Control},
  author={Lefringhausen, Robert and Springer, Theodor and Hirche, Sandra},
  journal={arXiv preprint arXiv:2512.08013},
  year={2025}
}
```
