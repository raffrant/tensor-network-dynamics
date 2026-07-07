"""
Tensor-network graph extraction benchmark.

This script studies whether a four-qubit target graph state can be extracted
from a six-qubit resource tensor network by:

    1. Preparing a six-leg tensor-network resource.
    2. Applying a parameterized graph of controlled-phase gates.
    3. Post-selecting / measuring two qubits.
    4. Applying local single-qubit rotations.
    5. Benchmarking the extracted state against a target graph state by fidelity.

The central benchmark is

    infidelity = 1 - |<psi_extracted | psi_target>|^2.

Small infidelity indicates successful graph extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import functools as ft

import numpy as np
import scipy.linalg
from scipy.linalg import svdvals
from scipy.optimize import minimize
import matplotlib.pyplot as plt


Array = np.ndarray
Edge = tuple[int, int]


# ---------------------------------------------------------------------
# Basic tensor and state utilities
# ---------------------------------------------------------------------

def kron_all(tensors: Sequence[Array]) -> Array:
    """Kronecker product of a sequence of tensors."""
    return ft.reduce(np.kron, tensors)


def normalize_state(state: Array) -> Array:
    """Return a normalized copy of a state vector or tensor."""
    norm = np.linalg.norm(state.reshape(-1))

    if norm == 0:
        raise ValueError("Cannot normalize a zero-norm state.")

    return state / norm


def plus_vector() -> Array:
    """Single-qubit |+> state."""
    return np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)


def plus_state(n_qubits: int) -> Array:
    """Vectorized |+>^{⊗ n} state."""
    return kron_all([plus_vector() for _ in range(n_qubits)])


def plus_tensor(rank: int) -> Array:
    """
    Rank-r tensor with all physical legs initialized in |+>.

    This is useful for constructing small tensor-network building blocks.
    """
    return plus_state(rank).reshape((2,) * rank)


def computational_basis_bits(index: int, n_qubits: int) -> list[int]:
    """
    Return the computational-basis bit string of a basis index.

    Convention:
        qubit 0 is the most significant bit.
    """
    return [(index >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]


# ---------------------------------------------------------------------
# Graph-state and graph-extraction utilities
# ---------------------------------------------------------------------

def adjacency_matrix(n_qubits: int, edges: Sequence[Edge]) -> Array:
    """
    Return the adjacency matrix of the graph.

    This makes the graph structure explicit and separates graph extraction
    from the low-level tensor contractions.
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


def controlled_phase_gate(theta: float) -> Array:
    """
    Two-qubit controlled-phase gate:

        CP(theta) = diag(1, 1, 1, exp(i theta)).

    For theta = pi, this is the usual CZ gate.
    """
    return np.diag([1.0, 1.0, 1.0, np.exp(1j * theta)]).astype(complex)


def graph_phase_vector(
    n_qubits: int,
    edges: Sequence[Edge],
    theta: float,
) -> Array:
    """
    Diagonal phase vector for a graph of controlled-phase gates.

    Each edge contributes exp(i theta) if and only if both qubits on that
    edge are in state |1>.
    """
    dim = 2**n_qubits
    phases = np.ones(dim, dtype=complex)

    for basis_index in range(dim):
        bits = computational_basis_bits(basis_index, n_qubits)

        occupied_edges = sum(bits[i] * bits[j] for i, j in edges)
        phases[basis_index] = np.exp(1j * theta * occupied_edges)

    return phases


def graph_state(
    n_qubits: int,
    edges: Sequence[Edge],
    theta: float = np.pi,
) -> Array:
    """
    Construct the graph state

        |G(theta)> = Π_(i,j in E) CP_ij(theta) |+>^{⊗ n}.

    The target state in the fidelity benchmark is produced by this function.
    """
    state = plus_state(n_qubits)
    return graph_phase_vector(n_qubits, edges, theta) * state


# ---------------------------------------------------------------------
# Tensor-network resource construction
# ---------------------------------------------------------------------

def build_seed_tensor_network() -> Array:
    """
    Construct the six-qubit seed tensor network.

    The original code builds four small |+>-type tensors and contracts
    internal virtual indices. The remaining six open legs form the physical
    resource tensor.

    Diagrammatically, this is a small tensor network with six open physical
    legs and several internal contracted bonds. The final open tensor is then
    acted on by a graph of controlled-phase gates.
    """
    tensor_a = plus_tensor(3)  # physical legs 0, 1, virtual bond x
    tensor_b = plus_tensor(4)  # virtual x, physical 2, 3, virtual y
    tensor_c = plus_tensor(3)  # virtual y, physical 4, virtual z
    tensor_d = plus_tensor(3)  # virtual z, physical 5, sink leg

    # Contract virtual bond x.
    network = np.einsum(
        "abx,xcdy->abcdy",
        tensor_a,
        tensor_b,
    )

    # Contract virtual bond y.
    network = np.einsum(
        "abcdy,yef->abcdef",
        network,
        tensor_c,
    )

    # Contract virtual bond z.
    network = np.einsum(
        "abcdef,fgh->abcdegh",
        network,
        tensor_d,
    )

    # Remove the final sink leg by contraction with the all-ones effect.
    # This leaves six open physical legs.
    six_qubit_tensor = np.sum(network, axis=-1)

    return normalize_state(six_qubit_tensor)


# ---------------------------------------------------------------------
# Local tensor operations
# ---------------------------------------------------------------------

def apply_single_qubit_operator(
    tensor: Array,
    operator: Array,
    qubit: int,
) -> Array:
    """
    Apply a one-qubit operator to a tensor leg.

    The tensor is kept in rank-n tensor form instead of being immediately
    flattened. This is closer to the tensor-network interpretation.
    """
    tensor = np.moveaxis(tensor, qubit, 0)
    updated = np.tensordot(operator, tensor, axes=(1, 0))
    return np.moveaxis(updated, 0, qubit)


def apply_two_qubit_operator(
    tensor: Array,
    operator: Array,
    qubits: Edge,
) -> Array:
    """
    Apply a two-qubit operator to two tensor legs.

    The operator is interpreted as a 4 x 4 matrix with input and output
    grouped as two-qubit Hilbert spaces.
    """
    q1, q2 = qubits

    if q1 == q2:
        raise ValueError("Cannot apply a two-qubit gate to the same qubit twice.")

    n_qubits = tensor.ndim

    if not (0 <= q1 < n_qubits and 0 <= q2 < n_qubits):
        raise ValueError(f"Invalid qubits {qubits} for tensor with rank {n_qubits}.")

    # Bring the two active legs to the front.
    moved = np.moveaxis(tensor, (q1, q2), (0, 1))
    rest_shape = moved.shape[2:]

    # Matrix-vector action on the two active physical legs.
    moved_matrix = moved.reshape(4, -1)
    updated = operator.reshape(4, 4) @ moved_matrix
    updated = updated.reshape(2, 2, *rest_shape)

    # Return active legs to their original positions.
    return np.moveaxis(updated, (0, 1), (q1, q2))


def apply_graph_to_tensor(
    tensor: Array,
    edges: Sequence[Edge],
    theta: float,
) -> Array:
    """
    Apply a controlled-phase graph to a tensor-network state.

    This is the tensor-network analogue of applying

        Π_(i,j in E) CP_ij(theta)

    to a vectorized state.
    """
    cp = controlled_phase_gate(theta)
    state = tensor.copy()

    for edge in edges:
        state = apply_two_qubit_operator(state, cp, edge)

    return normalize_state(state)


# ---------------------------------------------------------------------
# Measurements and local rotations
# ---------------------------------------------------------------------

def bloch_projector(polar: float, azimuth: float) -> Array:
    """
    Rank-one measurement projector on the Bloch sphere:

        P(n) = 1/2 * (I + n . sigma),

    where

        n = (sin(polar) cos(azimuth),
             sin(polar) sin(azimuth),
             cos(polar)).

    The measured qubit is later contracted out, giving a post-selected
    reduced tensor-network state.
    """
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    nx = np.sin(polar) * np.cos(azimuth)
    ny = np.sin(polar) * np.sin(azimuth)
    nz = np.cos(polar)

    return 0.5 * (np.eye(2, dtype=complex) + nx * sx + ny * sy + nz * sz)


def measure_and_remove_qubit(
    tensor: Array,
    projector: Array,
    qubit: int,
) -> Array:
    """
    Apply a measurement projector and contract out the measured leg.

    This implements the graph-extraction step:
    local measurements reduce the six-qubit resource tensor to a four-qubit
    candidate graph state.
    """
    updated = apply_single_qubit_operator(tensor, projector, qubit)

    # Post-select by contracting the measured output leg with the all-ones
    # effect, matching the structure of the original script.
    reduced = np.sum(updated, axis=qubit)

    return normalize_state(reduced)


def single_qubit_rotation(
    polar: float,
    azimuth: float,
    angle: float,
) -> Array:
    """
    General SU(2) single-qubit rotation:

        U = exp[-i angle/2 * n . sigma].
    """
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    nx = np.sin(polar) * np.cos(azimuth)
    ny = np.sin(polar) * np.sin(azimuth)
    nz = np.cos(polar)

    generator = nx * sx + ny * sy + nz * sz
    return scipy.linalg.expm(-0.5j * angle * generator)


def local_rotation_layer(params: Array, n_qubits: int) -> Array:
    """
    Tensor product of local rotations on the extracted four-qubit state.

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


# ---------------------------------------------------------------------
# Fidelity benchmark
# ---------------------------------------------------------------------

def state_fidelity(candidate: Array, target: Array) -> float:
    """
    Pure-state fidelity:

        F = |<candidate | target>|^2.

    Both states are normalized before comparison.
    """
    candidate = normalize_state(candidate.reshape(-1))
    target = normalize_state(target.reshape(-1))

    return float(abs(np.vdot(candidate, target)) ** 2)


def state_infidelity(candidate: Array, target: Array) -> float:
    """
    Infidelity benchmark used as the optimization loss.

    Successful graph extraction corresponds to infidelity close to zero.
    """
    return 1.0 - state_fidelity(candidate, target)


def bipartite_entropy(state: Array, n_left: int, n_qubits: int) -> tuple[Array, float]:
    """
    Singular spectrum and entanglement entropy across a bipartition.

    This diagnostic is useful for tensor-network analysis because the
    singular values determine the MPS bond dimension required across the cut.
    """
    matrix = state.reshape(2**n_left, 2 ** (n_qubits - n_left))

    singular_values = svdvals(matrix)
    singular_values = singular_values / np.linalg.norm(singular_values)

    probabilities = np.abs(singular_values) ** 2
    probabilities = probabilities[probabilities > 0]

    entropy = -float(np.sum(probabilities * np.log2(probabilities)))

    return singular_values, entropy


# ---------------------------------------------------------------------
# Variational graph extraction model
# ---------------------------------------------------------------------

@dataclass
class GraphExtractionConfig:
    """
    Full graph-extraction problem specification.

    resource_edges:
        Graph applied to the six-qubit tensor-network resource.

    target_edges:
        Four-qubit graph state used in the fidelity benchmark.

    first_measured_qubit:
        First qubit removed from the six-qubit resource.

    second_measured_qubit_after_first_removal:
        Qubit removed from the five-qubit intermediate tensor.

        In the original script this is index 4 after the first measurement,
        corresponding to the final remaining tensor leg.
    """
    resource_edges: list[Edge]
    target_edges: list[Edge]
    first_measured_qubit: int = 1
    second_measured_qubit_after_first_removal: int = 4


def extracted_four_qubit_state(
    params: Array,
    resource_tensor: Array,
    config: GraphExtractionConfig,
    graph_phase_theta: float,
) -> Array:
    """
    Extract a four-qubit state from the six-qubit resource tensor.

    Parameter layout:
        params[0]       -> first measurement polar angle
        params[1]       -> first measurement azimuth
        params[2]       -> second measurement polar angle
        params[3]       -> second measurement azimuth
        params[4:16]    -> local SU(2) rotations on four output qubits
    """
    first_polar = params[0]
    first_azimuth = params[1]
    second_polar = params[2]
    second_azimuth = params[3]

    local_rotation_params = params[4:]

    # 1. Apply the six-qubit resource graph.
    resource = apply_graph_to_tensor(
        tensor=resource_tensor,
        edges=config.resource_edges,
        theta=graph_phase_theta,
    )

    # 2. First local measurement and tensor-leg removal.
    first_projector = bloch_projector(first_polar, first_azimuth)

    reduced = measure_and_remove_qubit(
        tensor=resource,
        projector=first_projector,
        qubit=config.first_measured_qubit,
    )

    # 3. Second local measurement and tensor-leg removal.
    second_projector = bloch_projector(second_polar, second_azimuth)

    reduced = measure_and_remove_qubit(
        tensor=reduced,
        projector=second_projector,
        qubit=config.second_measured_qubit_after_first_removal,
    )

    # 4. Local correction / graph-extraction rotations.
    local_layer = local_rotation_layer(local_rotation_params, n_qubits=4)

    corrected_state = local_layer @ reduced.reshape(2**4)

    return normalize_state(corrected_state)


def make_objective(
    resource_tensor: Array,
    config: GraphExtractionConfig,
    graph_phase_theta: float,
):
    """
    Build the optimization objective for a fixed resource graph phase.

    The objective is the graph-extraction infidelity against the target graph
    state.
    """
    target = graph_state(
        n_qubits=4,
        edges=config.target_edges,
        theta=np.pi,
    )

    def objective(params: Array) -> float:
        candidate = extracted_four_qubit_state(
            params=params,
            resource_tensor=resource_tensor,
            config=config,
            graph_phase_theta=graph_phase_theta,
        )

        return state_infidelity(candidate, target)

    return objective


def optimization_bounds() -> list[tuple[float | None, float | None]]:
    """
    Bounds matching the original constrained measurement search.

    Measurement polar angles:
        [-pi/4, pi/4]

    Measurement azimuths:
        [0, 2pi]

    Local rotation parameters:
        unconstrained
    """
    bounds: list[tuple[float | None, float | None]] = [
        (-np.pi / 4, np.pi / 4),
        (0.0, 2.0 * np.pi),
        (-np.pi / 4, np.pi / 4),
        (0.0, 2.0 * np.pi),
    ]

    # Four qubits x three local-rotation parameters per qubit.
    bounds += [(None, None)] * 12

    return bounds


def random_initial_parameters(
    graph_phase_theta: float,
    rng: np.random.Generator,
) -> Array:
    """
    Random initialization following the spirit of the original code.

    The measurement angles are initialized near zero, while local rotations
    are initialized around the current graph phase.
    """
    measurement_guess = np.zeros(4)
    local_rotation_guess = rng.normal(
        loc=graph_phase_theta,
        scale=0.25 * np.pi,
        size=12,
    )

    return np.concatenate([measurement_guess, local_rotation_guess])


@dataclass
class ScanRecord:
    graph_phase_theta: float
    infidelity: float
    fidelity: float
    measurement_1_polar: float
    measurement_1_azimuth: float
    measurement_2_polar: float
    measurement_2_azimuth: float
    success: bool


def optimize_for_phase(
    graph_phase_theta: float,
    resource_tensor: Array,
    config: GraphExtractionConfig,
    rng: np.random.Generator,
    n_restarts: int = 1,
) -> ScanRecord:
    """
    Optimize the graph-extraction fidelity for one value of the resource phase.
    """
    objective = make_objective(
        resource_tensor=resource_tensor,
        config=config,
        graph_phase_theta=graph_phase_theta,
    )

    best_result = None

    for _ in range(n_restarts):
        x0 = random_initial_parameters(graph_phase_theta, rng)

        result = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            bounds=optimization_bounds(),
            options={
                "maxiter": 1000,
                "ftol": 1e-12,
            },
        )

        if best_result is None or result.fun < best_result.fun:
            best_result = result

    assert best_result is not None

    return ScanRecord(
        graph_phase_theta=graph_phase_theta,
        infidelity=float(best_result.fun),
        fidelity=float(1.0 - best_result.fun),
        measurement_1_polar=float(best_result.x[0]),
        measurement_1_azimuth=float(best_result.x[1]),
        measurement_2_polar=float(best_result.x[2]),
        measurement_2_azimuth=float(best_result.x[3]),
        success=bool(best_result.success),
    )


def run_phase_scan(
    n_phase_points: int = 100,
    n_restarts: int = 1,
    seed: int = 1234,
) -> list[ScanRecord]:
    """
    Scan the controlled-phase angle of the resource graph.

    For every theta, we optimize the measurement settings and local rotations,
    then report the graph-extraction fidelity benchmark.
    """
    rng = np.random.default_rng(seed)

    # Six-qubit resource graph.
    #
    # This is the graph applied to the six-qubit tensor-network resource before
    # measurement. It is the explicit graph structure being tested for
    # extractability.
    resource_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (0, 5),
        (2, 5),
        (3, 5),
        (4, 5),
    ]

    # Four-qubit target graph.
    #
    # This is the state used in the final fidelity benchmark.
    target_edges = [
        (0, 1),
        (1, 3),
        (2, 3),
    ]

    config = GraphExtractionConfig(
        resource_edges=resource_edges,
        target_edges=target_edges,
    )

    print("Resource graph adjacency matrix:")
    print(adjacency_matrix(6, resource_edges))
    print()

    print("Target graph adjacency matrix:")
    print(adjacency_matrix(4, target_edges))
    print()

    resource_tensor = build_seed_tensor_network()
    theta_grid = np.linspace(0.0, np.pi, n_phase_points)

    records: list[ScanRecord] = []

    for index, graph_phase_theta in enumerate(theta_grid):
        record = optimize_for_phase(
            graph_phase_theta=graph_phase_theta,
            resource_tensor=resource_tensor,
            config=config,
            rng=rng,
            n_restarts=n_restarts,
        )

        records.append(record)

        print(
            f"[{index:03d}] "
            f"theta/pi={graph_phase_theta / np.pi:.6f} | "
            f"infidelity={record.infidelity:.6e} | "
            f"fidelity={record.fidelity:.12f}"
        )

    return records


# ---------------------------------------------------------------------
# Plotting and diagnostics
# ---------------------------------------------------------------------

def plot_infidelity_scan(records: Sequence[ScanRecord]) -> None:
    """
    Plot graph-extraction infidelity as a function of resource graph phase.
    """
    theta_over_pi = np.array([r.graph_phase_theta / np.pi for r in records])
    infidelities = np.array([r.infidelity for r in records])

    # Avoid log(0) in case the optimizer reaches machine precision.
    infidelities = np.maximum(infidelities, 1e-16)

    plt.figure(figsize=(7, 4))
    plt.plot(theta_over_pi, infidelities)

    plt.yscale("log")
    plt.xlabel(r"Resource graph phase $\theta / \pi$")
    plt.ylabel(r"Graph-extraction infidelity $1 - F$")
    plt.title("Fidelity benchmark for tensor-network graph extraction")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_target_entanglement_diagnostics() -> None:
    """
    Print MPS-style bipartite entanglement diagnostics for the target graph.

    These singular spectra describe the tensor-network bond dimensions needed
    to represent the target graph state across each sequential cut.
    """
    target_edges = [(0, 1), (1, 3), (2, 3)]
    target = graph_state(4, target_edges, theta=np.pi)

    print("Target graph-state bipartite entanglement diagnostics:")

    for cut in [1, 2, 3]:
        spectrum, entropy = bipartite_entropy(
            state=target,
            n_left=cut,
            n_qubits=4,
        )

        print(f"  cut {cut}|{4 - cut}: entropy = {entropy:.6f}")
        print(f"    singular values = {np.round(spectrum, 8)}")


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------

if __name__ == "__main__":
    records = run_phase_scan(
        n_phase_points=100,
        n_restarts=1,
        seed=1234,
    )

    plot_infidelity_scan(records)
    print_target_entanglement_diagnostics()
