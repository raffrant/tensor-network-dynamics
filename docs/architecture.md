# Architecture

## Design philosophy

The repository is intentionally split by workflow rather than by a formal package API.

- `notebooks/` contains exploratory Julia studies built around `ITensors.jl` for matrix-product-state dynamics.
- `python/` contains explicit tensor-network and measurement-based state-preparation prototypes in dense-state style code.
- `docs/` collects the research framing and future cleanup path.

## Why this split works

The Julia side is best for efficient MPS evolution and observable tracking in spin chains, while the Python side is a better sandbox for custom contractions, symbolic checks, fidelity objectives, and optimization routines.

## Refactor path

A natural next version would introduce:

1. `src/tn_core/` for gates, contractions, SVD utilities, and state builders.
2. `src/models/` for Heisenberg, driven Ising, and graph-resource constructions.
3. `src/optimization/` for parameter-search utilities.
4. `results/` for plots, cached sweeps, and paper figures.
