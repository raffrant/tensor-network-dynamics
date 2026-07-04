# Contributing

## Good first improvements

- Add docstrings to all public functions.
- Remove global state and pass parameters explicitly.
- Add seed control for reproducibility.
- Separate simulation, optimization, and plotting.
- Save sweep outputs as CSV, NPZ, or HDF5.

## Style guidance

- Keep scientific assumptions explicit.
- Prefer small reusable functions over monolithic scripts.
- Name observables and parameters consistently.
- Add short comments where the physics is non-obvious.

## Pull request ideas

- Refactor the Python scripts into importable modules.
- Add tests for contraction correctness and state normalization.
- Add a reproducible DTC benchmark plot.
- Add an exact-vs-MPS comparison notebook.
