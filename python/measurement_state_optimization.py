"""
Graph-state extraction experiment via local rotations and post-measurement filtering.

Core idea
---------
We represent the entangling structure as a graph G = (V, E).
Each edge corresponds to a controlled-phase gate CP(theta).

The numerical task is:

    resource graph + measurement filter + local single-qubit rotations
        -> target graph state

The graph structure is therefore isolated in edge lists, making it easy to
change the resource graph, target graph, or scan over different graph phases.

Tensor-network viewpoint
------------------------
Graph states are naturally tensor-network states. The optional MPS extractor
below performs a sequential SVD decomposition of a state vector, exposing the
bond dimensions and entanglement structure across bipartitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import functools as ft
import itertools

import numpy as np
import scipy.linalg
from scipy.optimize import minimize
import matplotlib.pyplot as plt


Array = np.ndarray
Edge = tuple[int, int]


# ---------------------------------------------------------------------
# Basic tensor utilities
# ---------------------------------------------------------------------

def kron_all(tensors: Sequence[Array]) -> Array:
    """Kronecker product of a list of tensors, preserving qubit ordering."""
    return ft.reduce(np.kron, tensors)


def plus_state(n_qubits: int) -> Array:
    """
    |+>^{⊗ n} state.

    This is the natural input state for graph-state preparation:
        |G> = Π_(i,j in E) CP_ij(theta) |+>^{⊗ n}.
    """
    single_plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    return kron_all([single_plus for _ in range(n_qubits)])


def bell_pair_state() -> Array:
    """Two-qubit Bell state (|00> + |11>) / sqrt(2)."""
    return np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2)


def computational_basis_bits(index: int, n_qubits: int) -> list[int]:
    """
    Return the binary representation of a basis index using the convention

        q0 q1 ... q(n-1)

    where q0 is the leftmost / most significant qubit.
    """
    return [(index >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]


# ---------------------------------------------------------------------
# Graph-state construction
# ---------------------------------------------------------------------

def adjacency_matrix(n_qubits: int, edges: Iterable[Edge]) -> Array:
    """
    Extract the graph as an adjacency matrix.

    This is the explicit graph-extraction layer of the code:
    the entangling pattern is represented by an edge list, then converted
    into an adjacency matrix or phase operator as needed.
    """
    adj = np.zeros((n_qubits, n_qubits), dtype=int)

    for i, j in edges:
        if i == j:
            raise ValueError("Self-loops are not valid graph-state edges.")
        if not (0 <= i < n_qubits and 0 <= j < n_qubits):
            raise ValueError(f"Invalid edge ({i}, {j}) for n={n_qubits}.")

        adj[i, j] = 1
        adj[j, i] = 1

    return adj


def graph_phase_vector(
    n_qubits: int,
    edges: Iterable[Edge],
    theta: float,
) -> Array:
    """
    Diagonal phase vector for the graph controlled-phase operator.

    For each edge (i, j), CP_ij(theta) contributes exp(i theta)
    only when both qubits i and j are in state |1>.

    Instead of building large dense matrices, we exploit the fact that graph
    controlled-phase gates are diagonal and commute.
    """
    dim = 2**n_qubits
    phases = np.ones(dim, dtype=complex)

    for basis_index in range(dim):
        bits = computational_basis_bits(basis_index, n_qubits)

        occupied_edges = sum(bits[i] * bits[j] for i, j in edges)
        phases[basis_index] = np.exp(1j * theta * occupied_edges)

    return phases


def apply_graph_phase(
    state: Array,
    n_qubits: int,
    edges: Iterable[Edge],
    theta: float,
) -> Array:
    """
    Apply all controlled-phase gates associated with a graph.

    This is equivalent to multiplying by
        Π_(i,j in E) CP_ij(theta),
    but implemented as elementwise multiplication by a diagonal phase vector.
    """
    return graph_phase_vector(n_qubits, edges, theta) * state


def graph_state(
    n_qubits: int,
    edges: Iterable[Edge],
    theta: float = np.pi,
) -> Array:
    """
    Construct a graph state from an edge list.

    For theta = pi, this is the usual CZ graph state.
    """
    return apply_graph_phase(
        state=plus_state(n_qubits),
        n_qubits=n_qubits,
        edges=edges,
        theta=theta,
    )


# ---------------------------------------------------------------------
# Tensor-network / MPS extraction
# ---------------------------------------------------------------------

def extract_mps_tensors(state: Array, n_qubits: int, cutoff: float = 0.0) -> list[Array]:
    """
    Extract an MPS representation by sequential SVD.

    The output tensors have shapes

        A[site].shape = (left_bond_dimension, physical_dimension=2, right_bond_dimension)

    This makes the tensor-network structure of the graph state explicit.
    Large intermediate bond dimensions indicate larger entanglement across
    the corresponding bipartition.
    """
    state = np.asarray(state, dtype=complex).reshape(-1)

    if state.size != 2**n_qubits:
        raise ValueError("State size must be 2**n_qubits.")

    tensors = []
    left_bond_dim = 1
    remaining = state.reshape(1, 2**n_qubits)

    for site in range(n_qubits - 1):
        remaining = remaining.reshape(left_bond_dim * 2, -1)

        u, singular_values, vh = np.linalg.svd(remaining, full_matrices=False)

        if cutoff > 0.0:
            keep = singular_values > cutoff
            u = u[:, keep]
            singular_values = singular_values[keep]
            vh = vh[keep, :]

        right_bond_dim = singular_values.size

        # Tensor at current site.
        tensors.append(u.reshape(left_bond_dim, 2, right_bond_dim))

        # Push Schmidt values into the remaining tensor.
        remaining = np.diag(singular_values) @ vh
        left_bond_dim = right_bond_dim

    tensors.append(remaining.reshape(left_bond_dim, 2, 1))
    return tensors


def mps_bond_dimensions(mps_tensors: Sequence[Array]) -> list[int]:
    """Return the internal MPS bond dimensions."""
    return [tensor.shape[2] for tensor in mps_tensors[:-1]]


# ---------------------------------------------------------------------
# Local rotations and measurement-induced filter
# ---------------------------------------------------------------------

def single_qubit_rotation(axis_polar: float, axis_azimuth: float, angle: float) -> Array:
    """
    General SU(2) rotation:

        U = exp[-i angle/2 * n . sigma]

    where n is parameterized by spherical angles:
        n = (sin(axis_polar) cos(axis_azimuth),
             sin(axis_polar) sin(axis_azimuth),
             cos(axis_polar)).
    """
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    nx = np.sin(axis_polar) * np.cos(axis_azimuth)
    ny = np.sin(axis_polar) * np.sin(axis_azimuth)
    nz = np.cos(axis_polar)

    generator = nx * sx + ny * sy + nz * sz
    return scipy.linalg.expm(-0.5j * angle * generator)


def local_rotation_layer(params: Array, n_qubits: int) -> Array:
    """
    Tensor product of arbitrary single-qubit rotations.

    Parameter layout:
        params[0:n]       -> polar angles of rotation axes
        params[n:2n]      -> azimuthal angles of rotation axes
        params[2n:3n]     -> rotation angles
    """
    polar = params[0:n_qubits]
    azimuth = params[n_qubits:2 * n_qubits]
    angles = params[2 * n_qubits:3 * n_qubits]

    rotations = [
        single_qubit_rotation(polar[q], azimuth[q], angles[q])
        for q in range(n_qubits)
    ]

    return kron_all(rotations)


def global_z_rotation_phase_vector(n_qubits: int, theta: float) -> Array:
    """
    Diagonal vector for Rz(theta)^{⊗ n}.

    This replaces the original dense w(n, theta) construction.
    """
    dim = 2**n_qubits
    phases = np.ones(dim, dtype=complex)

    for basis_index in range(dim):
        bits = computational_basis_bits(basis_index, n_qubits)

        phase = 1.0 + 0.0j
        for bit in bits:
            phase *= np.exp(1j * ((-1) ** (bit + 1)) * theta / 2)

        phases[basis_index] = phase

    return phases


def post_measurement_filter(
    state: Array,
    n_qubits: int,
    graph_phase_theta: float,
    measurement_angle: float,
    measurement_phase: float,
) -> Array:
    """
    Apply the non-unitary post-measurement filter.

    Original expression:

        exp(-i n theta/2) cos(th) I
        + exp(i phi) sin(th) W(theta)

    where W(theta) = Rz(theta)^{⊗ n}.

    Because W is diagonal, this is implemented as an elementwise operation.
    """
    w_diag = global_z_rotation_phase_vector(n_qubits, graph_phase_theta)

    filter_diag = (
        np.exp(-0.5j * n_qubits * graph_phase_theta) * np.cos(measurement_angle)
        + np.exp(1j * measurement_phase) * np.sin(measurement_angle) * w_diag
    )

    return filter_diag * state


# ---------------------------------------------------------------------
# Fidelity objective
# ---------------------------------------------------------------------

def state_infidelity(
    candidate: Array,
    target: Array,
    normalize_candidate: bool = False,
) -> float:
    """
    Objective used in the optimization.

    By default, this matches the original code:

        1 - |<candidate|target>|^2

    Since the post-measurement filter is non-unitary, candidate may not be
    normalized. Setting normalize_candidate=True gives the conditional-state
    fidelity after successful post-selection.
    """
    candidate = np.asarray(candidate, dtype=complex).reshape(-1)
    target = np.asarray(target, dtype=complex).reshape(-1)

    target = target / np.linalg.norm(target)

    if normalize_candidate:
        norm = np.linalg.norm(candidate)
        if norm == 0:
            return 1.0
        candidate = candidate / norm

    overlap = np.vdot(candidate, target)
    return float(1.0 - np.abs(overlap) ** 2)


def average_gate_fidelity(u1: Array, u2: Array, dimension: int) -> complex:
    """
    Average gate fidelity between two operators.

    Kept as a utility from the original script.
    """
    m = u1 @ u2.conjugate().T
    return (
        np.trace(m @ m.conjugate().T) + abs(np.trace(m)) ** 2
    ) / (dimension * (dimension + 1))


# ---------------------------------------------------------------------
# Optimization experiment
# ---------------------------------------------------------------------

@dataclass
class OptimizationRecord:
    graph_phase_theta: float

    constrained_loss: float
    unconstrained_loss: float

    constrained_measurement_angle: float
    constrained_measurement_phase: float

    constrained_success: bool
    unconstrained_success: bool


def initial_resource_state(n_qubits: int) -> Array:
    """
    Initial state used in the original code:

        Bell pair on the first two qubits
        tensor |+> states on the remaining two qubits.

    For n=4:
        (|00> + |11>) / sqrt(2) ⊗ |+> ⊗ |+>
    """
    if n_qubits != 4:
        raise ValueError(
            "This initial resource state is currently defined for n_qubits=4, "
            "matching the original script."
        )

    return kron_all([bell_pair_state(), plus_state(2)])


def build_candidate_state(
    params: Array,
    n_qubits: int,
    resource_edges: Sequence[Edge],
    graph_phase_theta: float,
) -> Array:
    """
    Build the variational candidate state.

    Pipeline:
        1. Prepare the initial resource state.
        2. Apply the resource graph controlled-phase pattern.
        3. Apply the post-measurement filter.
        4. Apply local single-qubit rotations.

    This is the tensor-network-inspired variational circuit:
    the graph encodes entanglement, while local rotations and measurement
    angles extract the desired target graph state.
    """
    measurement_angle = params[3 * n_qubits]
    measurement_phase = params[3 * n_qubits + 1]

    state = initial_resource_state(n_qubits)

    state = apply_graph_phase(
        state=state,
        n_qubits=n_qubits,
        edges=resource_edges,
        theta=graph_phase_theta,
    )

    state = post_measurement_filter(
        state=state,
        n_qubits=n_qubits,
        graph_phase_theta=graph_phase_theta,
        measurement_angle=measurement_angle,
        measurement_phase=measurement_phase,
    )

    local_layer = local_rotation_layer(params, n_qubits)
    state = local_layer @ state

    return state


def make_objective(
    n_qubits: int,
    resource_edges: Sequence[Edge],
    target_edges: Sequence[Edge],
    graph_phase_theta: float,
    normalize_candidate: bool = False,
):
    """
    Return the scalar loss function optimized by scipy.

    The target is a graph state generated from the target edge list.
    """
    target = graph_state(
        n_qubits=n_qubits,
        edges=target_edges,
        theta=np.pi,
    )

    def objective(params: Array) -> float:
        candidate = build_candidate_state(
            params=params,
            n_qubits=n_qubits,
            resource_edges=resource_edges,
            graph_phase_theta=graph_phase_theta,
        )

        return state_infidelity(
            candidate=candidate,
            target=target,
            normalize_candidate=normalize_candidate,
        )

    return objective


def parameter_bounds(n_qubits: int):
    """
    Bounds equivalent to the original inequality constraints:

        measurement_angle in [-pi/4, pi/4]
        measurement_phase in [-pi, pi]

    All local-rotation parameters remain unconstrained.
    """
    n_params = 3 * n_qubits + 2
    bounds = [(None, None) for _ in range(n_params)]

    bounds[-2] = (-np.pi / 4, np.pi / 4)
    bounds[-1] = (-np.pi, np.pi)

    return bounds


def random_initial_parameters(
    n_qubits: int,
    rng: np.random.Generator,
    mean: float = 0.2 * np.pi,
    std: float = 0.05 * np.pi,
) -> Array:
    """Random initialization matching the original script."""
    n_params = 3 * n_qubits + 2
    return rng.normal(loc=mean, scale=std, size=n_params)


def run_phase_scan(
    n_qubits: int = 4,
    n_phase_points: int = 100,
    seed: int | None = None,
    normalize_candidate: bool = False,
) -> list[OptimizationRecord]:
    """
    Scan the resource graph phase theta and optimize local extraction parameters.

    Two optimizations are run for each phase:
        1. constrained measurement angles
        2. unconstrained measurement angles

    This allows comparison with the original code.
    """
    rng = np.random.default_rng(seed)

    # Resource graph from the original code:
    # CP(1,2), CP(2,3), CP(0,3)
    resource_edges: list[Edge] = [(1, 2), (2, 3), (0, 3)]

    # Target graph from the original code:
    # CZ chain 0--1--2--3
    target_edges: list[Edge] = [(0, 1), (1, 2), (2, 3)]

    print("Resource adjacency matrix:")
    print(adjacency_matrix(n_qubits, resource_edges))
    print()

    print("Target adjacency matrix:")
    print(adjacency_matrix(n_qubits, target_edges))
    print()

    theta_grid = np.linspace(0, np.pi, n_phase_points)
    records: list[OptimizationRecord] = []

    for index, graph_phase_theta in enumerate(theta_grid):
        objective = make_objective(
            n_qubits=n_qubits,
            resource_edges=resource_edges,
            target_edges=target_edges,
            graph_phase_theta=graph_phase_theta,
            normalize_candidate=normalize_candidate,
        )

        x0_constrained = random_initial_parameters(n_qubits, rng)
        x0_unconstrained = random_initial_parameters(n_qubits, rng)

        constrained_result = minimize(
            objective,
            x0=x0_constrained,
            method="SLSQP",
            bounds=parameter_bounds(n_qubits),
        )

        unconstrained_result = minimize(
            objective,
            x0=x0_unconstrained,
            method="BFGS",
        )

        record = OptimizationRecord(
            graph_phase_theta=graph_phase_theta,
            constrained_loss=float(constrained_result.fun),
            unconstrained_loss=float(unconstrained_result.fun),
            constrained_measurement_angle=float(constrained_result.x[-2]),
            constrained_measurement_phase=float(constrained_result.x[-1]),
            constrained_success=bool(constrained_result.success),
            unconstrained_success=bool(unconstrained_result.success),
        )

        records.append(record)

        print(
            f"[{index:03d}] "
            f"theta={graph_phase_theta:.6f} | "
            f"constrained loss={record.constrained_loss:.6e} | "
            f"phi={record.constrained_measurement_phase:.6f} | "
            f"meas_angle={record.constrained_measurement_angle:.6f}"
        )

    return records


def plot_results(records: Sequence[OptimizationRecord]) -> None:
    """Plot constrained and unconstrained graph-extraction losses."""
    theta_values = np.array([r.graph_phase_theta for r in records])
    constrained_losses = np.array([r.constrained_loss for r in records])
    unconstrained_losses = np.array([r.unconstrained_loss for r in records])

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(theta_values, constrained_losses, label="Constrained measurement angles")
    ax.plot(theta_values, unconstrained_losses, label="Unconstrained measurement angles")

    ax.set_xlabel(r"Resource graph phase $\theta$")
    ax.set_ylabel(r"Infidelity $1 - |\langle \psi_{\mathrm{cand}}|\psi_{\mathrm{target}}\rangle|^2$")
    ax.set_title("Graph-state extraction from resource tensor network")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    records = run_phase_scan(
        n_qubits=4,
        n_phase_points=100,
        seed=1234,
        normalize_candidate=False,
    )

    plot_results(records)

    # Optional tensor-network diagnostic:
    #
    # Extract the MPS bond dimensions of the final target graph state.
    # This shows how much entanglement the target graph carries across
    # sequential bipartitions.
    target_edges = [(0, 1), (1, 2), (2, 3)]
    target = graph_state(4, target_edges, theta=np.pi)

    target_mps = extract_mps_tensors(target, n_qubits=4)
    print("Target MPS bond dimensions:", mps_bond_dimensions(target_mps))
