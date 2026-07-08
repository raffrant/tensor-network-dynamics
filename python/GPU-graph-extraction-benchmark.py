
"""
GPU Tensor-Network Graph Extraction Benchmark (PyTorch Autograd + Matplotlib)
===============================================================================

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

All state evolution, gradient computation, and parameter updates are performed
on GPU using PyTorch's autograd, making this a fully GPU-driven variational
tensor-network benchmark.


Requirements
------------
- Python 3.9+
- CUDA-enabled PyTorch (tested with torch >= 2.0, CUDA 11.8/12.1)
- NumPy
- Matplotlib
- Pandas

Install via:

    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install numpy matplotlib pandas


Usage
-----
Run the script from the command line:

    python gpu_tensor_graph_autograd_matplotlib.py

It will:
- Check for CUDA availability and abort if no GPU is detected.
- Build the six-qubit resource tensor network.
- Scan multiple target families (path4, cycle4, star4, complete4, optionally ghz4, w4).
- For each (family, theta) pair, optimize extraction parameters with Adam on GPU.
- Print resource and target adjacency matrices to the terminal.
- Display interactive Matplotlib figures showing:
    • Infidelity vs phase by family
    • Best fidelity bar chart by family
    • 2|2 bipartite entropy vs best fidelity
    • Resource adjacency matrix heatmap
    • Target adjacency matrix heatmaps
    • Optimization history for a representative run


Tensor-Network Structure
------------------------
Resource graph (6 qubits):
    Edges: (0,1), (1,2), (2,3), (3,4), (4,5), (0,5), (2,5), (3,5), (4,5)

Target families (4 qubits):
- path4:     (0,1), (1,2), (2,3)
- cycle4:    (0,1), (1,2), (2,3), (3,0)
- star4:     center=0, edges (0,1), (0,2), (0,3)
- complete4: all pairs (i,j) with i < j
- ghz4:      GHZ state (not a graph state)
- w4:        W state (not a graph state)

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

if not torch.cuda.is_available():
    raise RuntimeError('CUDA torch is required for this script. Install a CUDA-enabled PyTorch build.')

DEVICE = 'cuda'
CDTYPE = torch.complex128
RDTYPE = torch.float64


def kron_all(tensors):
    out = tensors[0]
    for t in tensors[1:]:
        out = torch.kron(out, t)
    return out


def normalize_state(state):
    norm = torch.linalg.norm(state.reshape(-1))
    return state / norm


def plus_vector():
    return torch.tensor([1.0, 1.0], dtype=CDTYPE, device=DEVICE) / math.sqrt(2.0)


def plus_state(n_qubits: int):
    return kron_all([plus_vector() for _ in range(n_qubits)])


def plus_tensor(rank: int):
    return plus_state(rank).reshape((2,) * rank)


def computational_basis_bits(index: int, n_qubits: int):
    return [(index >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]


def graph_phase_vector(n_qubits, edges, theta):
    dim = 2 ** n_qubits
    phases = torch.ones(dim, dtype=CDTYPE, device=DEVICE)
    for basis_index in range(dim):
        bits = computational_basis_bits(basis_index, n_qubits)
        occupied = sum(bits[i] * bits[j] for i, j in edges)
        phases[basis_index] = torch.exp(1j * theta * torch.tensor(float(occupied), dtype=RDTYPE, device=DEVICE))
    return phases


def graph_state(n_qubits, edges, theta):
    return graph_phase_vector(n_qubits, edges, theta) * plus_state(n_qubits)


def adjacency_matrix(n_qubits, edges):
    mat = np.zeros((n_qubits, n_qubits), dtype=int)
    for i, j in edges:
        mat[i, j] = 1
        mat[j, i] = 1
    return mat


def path_edges(n):
    return [(i, i + 1) for i in range(n - 1)]


def cycle_edges(n):
    return [(i, (i + 1) % n) for i in range(n)]


def star_edges(n, center=0):
    return [(center, i) for i in range(n) if i != center]


def complete_edges(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def family_edges(name: str):
    if name == 'path4':
        return path_edges(4)
    if name == 'cycle4':
        return cycle_edges(4)
    if name == 'star4':
        return star_edges(4)
    if name == 'complete4':
        return complete_edges(4)
    return None


def ghz_state(n):
    psi = torch.zeros(2 ** n, dtype=CDTYPE, device=DEVICE)
    psi[0] = 1 / math.sqrt(2.0)
    psi[-1] = 1 / math.sqrt(2.0)
    return psi


def w_state(n):
    psi = torch.zeros(2 ** n, dtype=CDTYPE, device=DEVICE)
    amp = 1 / math.sqrt(float(n))
    for i in range(n):
        idx = 1 << (n - 1 - i)
        psi[idx] = amp
    return psi


def build_seed_tensor_network():
    a = plus_tensor(3)
    b = plus_tensor(4)
    c = plus_tensor(3)
    d = plus_tensor(3)
    net = torch.einsum('abx,xcdy->abcdy', a, b)
    net = torch.einsum('abcdy,yef->abcdef', net, c)
    net = torch.einsum('abcdef,fgh->abcdegh', net, d)
    return normalize_state(torch.sum(net, dim=-1))


def apply_single_qubit_operator(tensor, operator, qubit):
    perm = [qubit] + [i for i in range(tensor.ndim) if i != qubit]
    inv = list(np.argsort(perm))
    moved = tensor.permute(*perm)
    updated = torch.tensordot(operator, moved, dims=([1], [0]))
    return updated.permute(*inv)


def apply_graph_diagonal_phase(tensor, edges, theta):
    n_qubits = tensor.ndim
    phase = graph_phase_vector(n_qubits, edges, theta).reshape((2,) * n_qubits)
    return normalize_state(tensor * phase)


def bloch_projector(polar, azimuth):
    sx = torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE, device=DEVICE)
    sy = torch.tensor([[0, -1j], [1j, 0]], dtype=CDTYPE, device=DEVICE)
    sz = torch.tensor([[1, 0], [0, -1]], dtype=CDTYPE, device=DEVICE)
    nx = torch.sin(polar) * torch.cos(azimuth)
    ny = torch.sin(polar) * torch.sin(azimuth)
    nz = torch.cos(polar)
    return 0.5 * (torch.eye(2, dtype=CDTYPE, device=DEVICE) + nx * sx + ny * sy + nz * sz)


def measure_and_remove_qubit(tensor, projector, qubit):
    updated = apply_single_qubit_operator(tensor, projector, qubit)
    return normalize_state(torch.sum(updated, dim=qubit))


def pauli_mats():
    sx = torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE, device=DEVICE)
    sy = torch.tensor([[0, -1j], [1j, 0]], dtype=CDTYPE, device=DEVICE)
    sz = torch.tensor([[1, 0], [0, -1]], dtype=CDTYPE, device=DEVICE)
    eye = torch.eye(2, dtype=CDTYPE, device=DEVICE)
    return sx, sy, sz, eye


def single_qubit_rotation(polar, azimuth, angle):
    sx, sy, sz, eye = pauli_mats()
    nx = torch.sin(polar) * torch.cos(azimuth)
    ny = torch.sin(polar) * torch.sin(azimuth)
    nz = torch.cos(polar)
    generator = nx * sx + ny * sy + nz * sz
    c = torch.cos(angle / 2)
    s = torch.sin(angle / 2)
    return c * eye - 1j * s * generator


def local_rotation_layer(params, n_qubits=4):
    polar = params[:n_qubits]
    azimuth = params[n_qubits:2 * n_qubits]
    angles = params[2 * n_qubits:3 * n_qubits]
    rots = [single_qubit_rotation(polar[q], azimuth[q], angles[q]) for q in range(n_qubits)]
    return kron_all(rots)


def state_fidelity(candidate, target):
    candidate = normalize_state(candidate.reshape(-1))
    target = normalize_state(target.reshape(-1))
    return torch.abs(torch.vdot(candidate, target)) ** 2


def bipartite_entropy(state, n_left, n_qubits):
    matrix = state.reshape(2 ** n_left, 2 ** (n_qubits - n_left))
    s = torch.linalg.svdvals(matrix)
    s = s / torch.linalg.norm(s)
    p = torch.abs(s) ** 2
    p = p[p > 1e-12]
    entropy = -torch.sum(p * torch.log2(p))
    return float(entropy.detach().cpu()), int((s > 1e-8).sum().item())


@dataclass
class GraphExtractionConfig:
    resource_edges: list[tuple[int, int]]
    first_measured_qubit: int = 1
    second_measured_qubit_after_first_removal: int = 4


def extracted_four_qubit_state(params, resource_tensor, config, theta):
    first_polar, first_azimuth, second_polar, second_azimuth = params[:4]
    local_params = params[4:]
    resource = apply_graph_diagonal_phase(resource_tensor, config.resource_edges, theta)
    reduced = measure_and_remove_qubit(resource, bloch_projector(first_polar, first_azimuth), config.first_measured_qubit)
    reduced = measure_and_remove_qubit(reduced, bloch_projector(second_polar, second_azimuth), config.second_measured_qubit_after_first_removal)
    local_layer = local_rotation_layer(local_params, 4)
    corrected = local_layer @ reduced.reshape(2 ** 4)
    return normalize_state(corrected)


def target_family_state(name):
    if name == 'path4':
        return graph_state(4, path_edges(4), math.pi)
    if name == 'cycle4':
        return graph_state(4, cycle_edges(4), math.pi)
    if name == 'star4':
        return graph_state(4, star_edges(4), math.pi)
    if name == 'complete4':
        return graph_state(4, complete_edges(4), math.pi)
    if name == 'ghz4':
        return ghz_state(4)
    if name == 'w4':
        return w_state(4)
    raise ValueError(name)


def optimize_family_phase(family, theta, resource_tensor, target, config, n_steps=220, lr=0.06):
    params = torch.nn.Parameter(torch.randn(16, dtype=RDTYPE, device=DEVICE) * 0.15)
    opt = torch.optim.Adam([params], lr=lr)
    best_inf = float('inf')
    best_fid = 0.0
    tic = time.perf_counter()
    history = []
    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)
        candidate = extracted_four_qubit_state(params, resource_tensor, config, theta)
        fidelity = state_fidelity(candidate, target)
        loss = 1.0 - fidelity
        loss.backward()
        opt.step()
        with torch.no_grad():
            cur_inf = float(loss.detach().cpu())
            cur_fid = float(fidelity.detach().cpu())
            history.append((step, cur_inf, cur_fid))
            if cur_inf < best_inf:
                best_inf = cur_inf
                best_fid = cur_fid
    return {
        'family': family,
        'theta_over_pi': float(theta.detach().cpu().item() / math.pi),
        'infidelity': best_inf,
        'fidelity': best_fid,
        'wall_seconds': time.perf_counter() - tic,
        'history': history,
    }


def run_scan(families, n_phase_points=18, n_steps=220):
    resource_edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(0,5),(2,5),(3,5),(4,5)]
    print('Resource adjacency matrix:')
    print(adjacency_matrix(6, resource_edges))
    config = GraphExtractionConfig(resource_edges=resource_edges)
    resource_tensor = build_seed_tensor_network()
    theta_grid = torch.linspace(0.0, math.pi, n_phase_points, dtype=RDTYPE, device=DEVICE)
    records = []
    diagnostics = []
    for family in families:
        edges = family_edges(family)
        if edges is not None:
            print(f'Target adjacency matrix for {family}:')
            print(adjacency_matrix(4, edges))
        target = target_family_state(family)
        for cut in [1, 2, 3]:
            entropy, rank_est = bipartite_entropy(target, cut, 4)
            diagnostics.append({'family': family, 'cut': cut, 'entropy': entropy, 'rank_est': rank_est})
        for theta in theta_grid:
            rec = optimize_family_phase(family, theta, resource_tensor, target, config, n_steps=n_steps)
            records.append(rec)
            print(f"{family} theta/pi={rec['theta_over_pi']:.3f} infid={rec['infidelity']:.3e} wall={rec['wall_seconds']:.3f}s")
    return records, diagnostics, resource_edges


def make_plots(records, diagnostics, resource_edges, families):
    plt.style.use('seaborn-v0_8-whitegrid')
    rdf = pd.DataFrame([{k: v for k, v in rec.items() if k != 'history'} for rec in records])
    ddf = pd.DataFrame(diagnostics)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.ravel()

    for family, grp in rdf.groupby('family'):
        grp = grp.sort_values('theta_over_pi')
        ax1.plot(grp['theta_over_pi'], grp['infidelity'], marker='o', linewidth=1.8, markersize=4, label=family)
    ax1.set_yscale('log')
    ax1.set_xlabel('theta / pi')
    ax1.set_ylabel('infidelity')
    ax1.set_title('GPU infidelity scan')
    ax1.legend(fontsize=8)

    best = rdf.groupby('family', as_index=False)['fidelity'].max().sort_values('fidelity', ascending=False)
    ax2.bar(best['family'], best['fidelity'])
    ax2.set_xlabel('family')
    ax2.set_ylabel('fidelity')
    ax2.set_title('Best fidelity by family')
    ax2.tick_params(axis='x', rotation=20)

    ent = ddf[ddf['cut'] == 2].copy().rename(columns={'entropy': 'entropy_2_2'})
    merged = best.merge(ent[['family', 'entropy_2_2']], on='family', how='left')
    ax3.scatter(merged['entropy_2_2'], merged['fidelity'])
    for _, row in merged.iterrows():
        ax3.annotate(row['family'], (row['entropy_2_2'], row['fidelity']), fontsize=8, xytext=(4, 4), textcoords='offset points')
    ax3.set_xlabel('entropy 2|2')
    ax3.set_ylabel('fidelity')
    ax3.set_title('Entropy vs best fidelity')

    mat = adjacency_matrix(6, resource_edges)
    im = ax4.imshow(mat, cmap='Blues', vmin=0, vmax=1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax4.text(j, i, str(mat[i, j]), ha='center', va='center', color='black', fontsize=9)
    ax4.set_title('Resource adjacency matrix')
    ax4.set_xlabel('qubit')
    ax4.set_ylabel('qubit')
    fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    fig.suptitle('GPU tensor-network extraction benchmark', fontsize=15)
    fig.tight_layout()

    n_graph_families = sum(1 for fam in families if family_edges(fam) is not None)
    cols = 2
    rows = (n_graph_families + cols - 1) // cols
    fig_adj, axes_adj = plt.subplots(rows, cols, figsize=(10, 4.5 * rows))
    axes_adj = np.atleast_1d(axes_adj).ravel()
    idx = 0
    for fam in families:
        edges = family_edges(fam)
        if edges is None:
            continue
        mat = adjacency_matrix(4, edges)
        ax = axes_adj[idx]
        im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=1)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, str(mat[i, j]), ha='center', va='center', color='black', fontsize=10)
        ax.set_title(f'Target adjacency: {fam}')
        ax.set_xlabel('qubit')
        ax.set_ylabel('qubit')
        idx += 1
    for k in range(idx, len(axes_adj)):
        axes_adj[k].axis('off')
    fig_adj.suptitle('Target graph adjacency matrices', fontsize=15)
    fig_adj.tight_layout()

    first_record = records[0]
    hist = np.array(first_record['history'])
    fig_hist, (axh1, axh2) = plt.subplots(1, 2, figsize=(12, 4.5))
    axh1.plot(hist[:, 0], hist[:, 1], linewidth=1.8)
    axh1.set_yscale('log')
    axh1.set_xlabel('step')
    axh1.set_ylabel('infidelity')
    axh1.set_title(f"Optimization history: {first_record['family']} @ theta/pi={first_record['theta_over_pi']:.3f}")
    axh2.plot(hist[:, 0], hist[:, 2], linewidth=1.8)
    axh2.set_xlabel('step')
    axh2.set_ylabel('fidelity')
    axh2.set_title('Fidelity trajectory')
    fig_hist.tight_layout()

    plt.show()


def main():
    families = ['path4', 'cycle4', 'star4', 'complete4']
    print('torch cuda available:', torch.cuda.is_available())
    print('device:', torch.cuda.get_device_name(0))
    records, diagnostics, resource_edges = run_scan(families, n_phase_points=16, n_steps=180)
    make_plots(records, diagnostics, resource_edges, families)


if __name__ == '__main__':
    main()
