# Spatial Branch-and-Bound for Multiplayer Nash Equilibria

This repository contains the official implementation accompanying the paper:

**Černý, J., Das Gupta, S., & Kroer, C.**
*Spatial Branch-and-Bound for Computing Multiplayer Nash Equilibrium*
AAAI 2026
[[Extended Version]](https://arxiv.org/abs/2508.10204)

## Overview

This code implements a sound and complete algorithm for computing Nash equilibria in multiplayer normal-form games.
The method reformulates equilibrium conditions as a **polynomial complementarity problem (PCP)** and solves it using a customized **two-stage spatial branch-and-bound (SBnB)** algorithm:

1. **Local solve:** A nonlinear interior-point method produces a high-quality local optimum of the penalized PCP formulation.
2. **Global search:** A spatial branch-and-bound solver is warm-started from this solution to compute a global Nash equilibrium or a certified ε-approximate equilibrium when terminated early.

Empirical results demonstrate substantial improvement over existing complete methods and competitive performance with state-of-the-art incomplete algorithms.

## Citation

```bibtex
@inproceedings{cerny2026sbnbe,
  author    = {Jakub Cerny and Shuvomoy Das Gupta and Christian Kroer},
  title     = {Spatial Branch-and-Bound for Computing Multiplayer Nash Equilibrium},
  booktitle = {AAAI},
  year      = {2026},
  date      = {2026},
  note      = {}
}
```

## License

GNU General Public License.
