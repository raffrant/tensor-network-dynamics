using ITensors
using Random
using Distributions
using Plots
using LaTeXStrings


# ---------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------

Base.@kwdef struct SpinChainParams
    N::Int = 10
    n_steps::Int = 5_000
    n_realizations::Int = 10

    dt::Float64 = 1.0
    cutoff::Float64 = 1e-9

    # Coupling parameters for the XXZ/Heisenberg-type gate
    Jxy::Float64 = 1.0
    Jz_mean::Float64 = 1.0
    Jz_disorder_width::Float64 = 0.1

    # Observable site. If 0, use N ÷ 2.
    center_site::Int = 0

    # Initial state: "all_up", "domain_wall", or "neel"
    initial_state::String = "all_up"
end


function observable_site(params::SpinChainParams)
    return params.center_site == 0 ? params.N ÷ 2 : params.center_site
end


# ---------------------------------------------------------------------
# Initial product states
# ---------------------------------------------------------------------

function product_state_labels(params::SpinChainParams)
    """
    Return the initial product-state labels.

    Tensor-network interpretation:
    productMPS creates a bond-dimension-1 MPS, i.e. an unentangled tensor
    network. Entanglement is then generated dynamically by the two-site gates.
    """
    N = params.N

    if params.initial_state == "all_up"
        return ["Up" for _ in 1:N]

    elseif params.initial_state == "domain_wall"
        return [j <= N ÷ 2 ? "Up" : "Dn" for j in 1:N]

    elseif params.initial_state == "neel"
        return [isodd(j) ? "Up" : "Dn" for j in 1:N]

    else
        error("Unknown initial_state = $(params.initial_state)")
    end
end


function initialize_mps(sites, params::SpinChainParams)
    """
    Construct the initial MPS.

    The MPS is the tensor-network representation of the many-body wavefunction.
    """
    state = product_state_labels(params)
    return productMPS(sites, state)
end


# ---------------------------------------------------------------------
# Hamiltonian gates
# ---------------------------------------------------------------------

function sample_Jz_couplings(params::SpinChainParams, rng::AbstractRNG)
    """
    Sample disordered nearest-neighbor ZZ couplings.

    The original script sampled random couplings but did not use them.
    Here they enter explicitly through the Sz_j Sz_{j+1} term.
    """
    width = params.Jz_disorder_width
    distribution = Uniform(params.Jz_mean - width, params.Jz_mean + width)

    return rand(rng, distribution, params.N - 1)
end


function two_site_xxz_hamiltonian(sites, j::Int, Jz::Float64, params::SpinChainParams)
    """
    Two-site XXZ / Heisenberg-type Hamiltonian:

        h_j =
            Jz Sz_j Sz_{j+1}
            + Jxy / 2 * (S+_j S-_{j+1} + S-_j S+_{j+1})

    For Jxy = Jz = 1, this is the usual isotropic Heisenberg exchange up to
    conventional normalization.
    """
    s1 = sites[j]
    s2 = sites[j + 1]

    return (
        Jz * op("Sz", s1) * op("Sz", s2)
        + params.Jxy / 2 * op("S+", s1) * op("S-", s2)
        + params.Jxy / 2 * op("S-", s1) * op("S+", s2)
    )
end


function make_time_step_gates(sites, Jz, params::SpinChainParams)
    """
    Construct one stroboscopic time-evolution step.

    Tensor-network approach:
    each two-site gate acts locally on neighboring MPS tensors, then ITensors
    recompresses the MPS using the chosen cutoff.

    This uses a simple nearest-neighbor gate sequence:

        U(dt) ≈ Π_j exp[-i dt h_j]

    For more accurate Hamiltonian time evolution, one could replace this by
    an even-odd second-order Trotter decomposition.
    """
    gates = ITensor[]

    for j in 1:(params.N - 1)
        hj = two_site_xxz_hamiltonian(sites, j, Jz[j], params)
        Gj = exp(-1im * params.dt * hj)
        push!(gates, Gj)
    end

    return gates
end


# ---------------------------------------------------------------------
# Observable and fidelity-style diagnostics
# ---------------------------------------------------------------------

function center_magnetization(psi, center::Int)
    """
    Local magnetization <Sz_center>.

    This is an efficient MPS observable because it only probes one tensor leg,
    not the full 2^N-dimensional wavefunction.
    """
    sz_values = expect(psi, "Sz")
    return real(sz_values[center])
end


function bond_dimensions(psi)
    """
    Return the MPS bond dimensions.

    This is a useful tensor-network diagnostic: growing bond dimension signals
    entanglement growth during dynamics.
    """
    return [dim(linkind(psi, b)) for b in 1:(length(psi) - 1)]
end


# ---------------------------------------------------------------------
# One realization
# ---------------------------------------------------------------------

function run_single_realization(
    sites,
    params::SpinChainParams,
    rng::AbstractRNG,
)
    """
    Run one disorder realization of the stroboscopic dynamics.

    Returns:
        magnetization[t] = <Sz_center(t)>.
    """
    psi = initialize_mps(sites, params)

    Jz = sample_Jz_couplings(params, rng)
    gates = make_time_step_gates(sites, Jz, params)

    center = observable_site(params)
    magnetization = zeros(Float64, params.n_steps)

    for step in 1:params.n_steps
        # Measure before applying the next time step.
        magnetization[step] = center_magnetization(psi, center)

        # MPS time evolution by local two-site gates.
        psi = apply(gates, psi; cutoff=params.cutoff)
        normalize!(psi)
    end

    return magnetization
end


# ---------------------------------------------------------------------
# Disorder average
# ---------------------------------------------------------------------

function disorder_averaged_dynamics(params::SpinChainParams; seed::Int = 1234)
    """
    Average the central magnetization over disorder realizations:

        M(t) = disorder average of <Sz_center(t)>.

    This is the benchmark plotted at the end.
    """
    rng = MersenneTwister(seed)

    # conserve_qns=false is general and safe.
    # If no Sx drive or transverse field is added, total Sz is conserved and
    # conserve_qns=true may be more efficient.
    sites = siteinds("S=1/2", params.N; conserve_qns=false)

    averaged_magnetization = zeros(Float64, params.n_steps)

    for realization in 1:params.n_realizations
        println("Running realization $realization / $(params.n_realizations)")

        magnetization = run_single_realization(sites, params, rng)
        averaged_magnetization .+= magnetization
    end

    averaged_magnetization ./= params.n_realizations

    return averaged_magnetization
end


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

function plot_magnetization(signal, params::SpinChainParams)
    times = 1:params.n_steps

    plot(
        times,
        signal;
        xscale = :log10,
        minorgrid = true,
        linewidth = 3,
        xlabel = L"t",
        ylabel = L"\langle S^z_{\mathrm{center}}(t) \rangle",
        label = params.initial_state,
        title = L"Stroboscopic\ MPS\ dynamics",
    )
end


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------

params = SpinChainParams(
    N = 10,
    n_steps = 5_000,
    n_realizations = 10,
    dt = 1.0,
    cutoff = 1e-9,

    Jxy = 1.0,
    Jz_mean = 1.0,
    Jz_disorder_width = 0.1,

    # Try "domain_wall" or "neel" for nontrivial dynamics.
    initial_state = "all_up",
)

signal = disorder_averaged_dynamics(params; seed = 1234)

println(signal)

ENV["GKS_ENCODING"] = "utf-8"
plot_magnetization(signal, params)
