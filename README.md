# Tensor Network Dynamics for Measurement-Driven State Engineering and Time-Crystal Response

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Julia](https://img.shields.io/badge/Julia-1.9%2B-9558B2?style=flat-square&logo=julia&logoColor=white)
![Status](https://img.shields.io/badge/status-research%20prototype-8A6D3B?style=flat-square)
![Tensor Networks](https://img.shields.io/badge/focus-tensor%20networks-01696f?style=flat-square)

A research-oriented code repository for tensor-network-based studies of **measurement-driven state preparation**, **graph-state style concentration protocols**, and **non-equilibrium spin-chain dynamics**, with a particular emphasis on **Floquet / discrete time-crystal observables** and long-time MPS evolution. 

## Scientific scope

This repository combines two closely related strands of many-body quantum simulation:

- **Measurement-based tensor-network workflows** in Python, focused on controlled-phase resources, local measurement maps, fidelity objectives, and constrained numerical optimization.
- **MPS-based driven and static spin-chain dynamics** in Julia using `ITensors.jl`, including long-time magnetization tracking in Heisenberg and driven Ising settings.

The current codebase is best understood as a compact research lab for prototyping ideas at the interface of tensor networks, quantum state engineering, and driven many-body dynamics. 
## Repository layout

```text
tensor-network-dynamics-public/
├── docs/
│   ├── architecture.md
│   ├── contributing.md
│   ├── project-notes.md
│   └── roadmap.md
├── notebooks/
│   ├── discrete-time-crystal-ising-chain.ipynb
│   └── mps-heisenberg-chain-dynamics.ipynb
├── python/
│   ├── measurement_state_optimization.py
│   └── tensor_network_concentration.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Contents

### Notebooks

#### `notebooks/discrete-time-crystal-ising-chain.ipynb`
A Julia / `ITensors.jl` notebook for a disordered driven Ising-chain setup with near-\(\pi\) spin rotations and local magnetization tracking, naturally aligned with discrete time-crystal style diagnostics.

#### `notebooks/mps-heisenberg-chain-dynamics.ipynb`
A Julia / `ITensors.jl` notebook for MPS evolution in a Heisenberg chain, measuring middle-spin magnetization over long times and averaging across disorder realizations.

### Python scripts

#### `python/tensor_network_concentration.py`
Prototype routines for explicit tensor contractions, measurement maps, controlled-phase resource construction, entanglement diagnostics, and optimization-driven concentration/state-preparation experiments.

#### `python/measurement_state_optimization.py`
A more focused optimization script for measurement-assisted target-state preparation with constrained parameters and fidelity-based objectives.

## Installation

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Julia packages

```julia
using Pkg
Pkg.add(["ITensors", "Distributions", "Plots", "LaTeXStrings"])
```

## Research directions

This repo is particularly suitable for exploring: 

- measurement-based state engineering on graph-like entangled resources, 
- optimization of local rotations and measurement angles, 
- Floquet spin dynamics and subharmonic response diagnostics, 
- long-time MPS evolution of disordered spin chains.


## Status

This is a **research prototype** repository: scientifically interesting, technically substantial, and ideal for further cleanup into a stronger public codebase or paper companion repo. [file:73][file:74][file:75][file:76]
