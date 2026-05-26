## Author

Adam Kautzky

## Overview

This repository contains code developed during my master's thesis on numerical
methods for elastic wave propagation in anisotropic media. The focus is on the
Lebedev grid, a discretisation composed of four interlaced staggered grids,
and its comparison with the conventional staggered-grid formulation associated
with Virieux (1984,1986).

The principal advantage of the Lebedev scheme is its treatment of anisotropic
constitutive laws containing coupled normal–shear stress terms. In a
conventional staggered-grid discretisation, the strain-rate components required
for such updates are generally located on different grids and must therefore
be interpolated or averaged onto a common location. This additional operation
introduces an extra numerical approximation. In the Lebedev formulation, the
required components are assembled directly on common material grids, allowing
the full constitutive update to be applied without this additional averaging
step.

Both the Lebedev and conventional staggered-grid methods are implemented in
two forms:

1. A sparse-matrix implementation using SciPy, intended for operator
   inspection, verification, and smaller-scale numerical experiments.
2. A matrix-free implementation using NumPy arrays and Numba-accelerated
   kernels, intended for larger three-dimensional simulations and performance
   studies.

## Suggested Future Work

- Add absorbing boundary conditions, such as PML or sponge layers.
- Introduce command-line configuration for grid, material, source, and output
  parameters.
- Add automated regression tests for grid mappings and derivative operators.
- Add automated convergence tests for isotropic reference cases.
- Benchmark sparse and matrix-free implementations systematically.
- Extend comparisons between Lebedev and conventional staggered grids for
  rotated anisotropic media.
- Save simulation metadata alongside generated figures to improve
  reproducibility.
