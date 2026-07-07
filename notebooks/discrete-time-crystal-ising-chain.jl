using ITensors
using Distributions
using Random
using Plots
using LaTeXStrings


# ---------------------------------------------------------------------
# Floquet simulation parameters
# ---------------------------------------------------------------------

Base.@kwdef struct FloquetParams
    N::Int = 10                       # Number of spins
    n_steps::Int = 10_000             # Number of Floquet periods
    n_realizations::Int = 10          # Disorder realizations

    tau::Float64 = 1.0                # Duration of one interaction step
    epsilon::Float64 = 0.03           # Pulse imperfection: π -> π - ε
    disorder_width::Float64 = 0.1 * 2π

    cutoff::Float64 = 1e-9            # MPS truncation cutoff

    # If center_site = 0, the code uses N ÷ 2.
    center_site::Int = 0
end


function observable_site(params::FloquetParams)
    return params.center_site == 0 ? params.N ÷ 2 : params.center_site
end


# ---------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------

function domain_wall_state(N::Int)
    """
    Product-state domain wall:

        |↑↑...↑↓↓...↓>

    This is a useful non-equilibrium initial state for probing Floquet
    dynamics and subharmonic response.
    """
    return [j <= N ÷ 2 ? "Up" : "Dn" for j in 1:N]
end


function initialize_mps(sites, params::FloquetParams)
    """
    Construct the initial MPS.

    The MPS is the tensor-network representation of the many-body wavefunction.
    ITensors automatically stores and compresses the state using bond indices.
    """
    state = domain_wall_state(params.N)
    return productMPS(sites, state)
end


# ---------------------------------------------------------------------
# Floquet gate construction
# ---------------------------------------------------------------------

function sample_disordered_couplings(params::FloquetParams, rng::AbstractRNG)
    """
    Sample disordered nearest-neighbor couplings:

        J_j ∈ [2π - δJ, 2π + δJ].

    These couplings define the interaction part of the Floquet unitary.
    """
    distribution = Uniform(2π - params.disorder_width, 2π + params.disorder_width)
    return rand(rng, distribution, params.N - 1)
end


function make_interaction_gates(sites, J, params::FloquetParams)
    """
    Build the disordered two-site interaction layer:

        U_ZZ = Π_j exp[-i τ J_j Sz_j Sz_{j+1}].

    Each gate is a rank-4 ITensor acting on neighboring physical legs of
    the MPS tensor network.
    """
    gates = ITensor[]

    for j in 1:(params.N - 1)
        hj = J[j] * op("Sz", sites[j]) * op("Sz", sites[j + 1])
        Gj = exp(-1im * params.tau * hj)
        push!(gates, Gj)
    end

    return gates
end


function make_drive_gates(sites, params::FloquetParams)
    """
    Build the global imperfect π-pulse layer:

        U_X = Π_j exp[-i (π - ε) Sx_j].

    For ε = 0, this is an ideal global spin-flip pulse.
    """
    gates = ITensor[]

    drive_angle = π - params.epsilon

    for j in 1:params.N
        hj = op("Sx", sites[j])
        Gj = exp(-1im * drive_angle * hj)
        push!(gates, Gj)
    end

    return gates
end


function make_floquet_gates(sites, J, params::FloquetParams)
    """
    Construct one full Floquet period.

    The ordering here matches the original script:

        U_F = U_X U_ZZ,

    implemented as the sequential application of:
        1. nearest-neighbor ZZ interaction gates,
        2. single-site X-drive gates.

    This list of ITensors is repeatedly applied to the MPS.
    """
    interaction_gates = make_interaction_gates(sites, J, params)
    drive_gates = make_drive_gates(sites, params)

    return vcat(interaction_gates, drive_gates)
end


# ---------------------------------------------------------------------
# Observables and benchmarks
# ---------------------------------------------------------------------

function central_magnetization(psi, center::Int)
    """
    Compute <Sz_center> from the MPS.

    This is a local observable, so it is efficient in the MPS representation.
    """
    sz_values = expect(psi, "Sz")
    return real(sz_values[center])
end


function period_doubled_signal(psi, step::Int, center::Int)
    """
    Period-doubled Floquet diagnostic:

        (-1)^t <Sz_center(t)>.

    A long-lived nonzero signal indicates robust subharmonic response.
    """
    return (-1)^(step - 1) * central_magnetization(psi, center)
end


# ---------------------------------------------------------------------
# One disorder realization
# ---------------------------------------------------------------------

function run_single_realization(
    sites,
    params::FloquetParams,
    rng::AbstractRNG,
)
    """
    Evolve one disordered Floquet realization.

    Returns:
        signal[t] = (-1)^t <Sz_center(t)>

    The state is evolved stroboscopically:

        |ψ(t + 1)> = U_F |ψ(t)>.

    After each Floquet period, the MPS is compressed using the specified cutoff.
    """
    psi = initialize_mps(sites, params)

    J = sample_disordered_couplings(params, rng)
    gates = make_floquet_gates(sites, J, params)

    center = observable_site(params)
    signal = zeros(Float64, params.n_steps)

    for step in 1:params.n_steps
        # Measure before applying the next Floquet period.
        signal[step] = period_doubled_signal(psi, step, center)

        # Tensor-network time evolution:
        # apply the Floquet gate layer to the MPS and truncate small Schmidt values.
        psi = apply(gates, psi; cutoff=params.cutoff)
        normalize!(psi)
    end

    return signal
end


# ---------------------------------------------------------------------
# Disorder average
# ---------------------------------------------------------------------

function disorder_averaged_dynamics(params::FloquetParams; seed::Int = 1234)
    """
    Average the period-doubled magnetization over disorder realizations.

    This estimates:

        C(t) = disorder average of [(-1)^t <Sz_center(t)>].

    The result is the central Floquet benchmark plotted below.
    """
    rng = MersenneTwister(seed)

    # S = 1/2 local Hilbert spaces.
    #
    # conserve_qns=false is important because the global Sx pulse does not
    # conserve total Sz.
    sites = siteinds("S=1/2", params.N; conserve_qns=false)

    averaged_signal = zeros(Float64, params.n_steps)

    for realization in 1:params.n_realizations
        println("Running disorder realization $realization / $(params.n_realizations)")

        signal = run_single_realization(sites, params, rng)
        averaged_signal .+= signal
    end

    averaged_signal ./= params.n_realizations

    return averaged_signal
end


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

function plot_floquet_signal(signal, params::FloquetParams)
    """
    Plot the disorder-averaged period-doubled magnetization.

    A logarithmic time axis is useful for diagnosing slow decay and
    prethermal or many-body-localized Floquet behavior.
    """
    times = 1:params.n_steps

    plot(
        times,
        signal;
        xscale = :log10,
        minorgrid = true,
        linewidth = 3,
        xlabel = L"\mathrm{Floquet\ period}\ t",
        ylabel = L"(-1)^t \langle S^z_{\mathrm{center}}(t) \rangle",
        label = L"| \uparrow\uparrow\uparrow\uparrow\uparrow\downarrow\downarrow\downarrow\downarrow\downarrow \rangle",
        title = L"Disorder\ averaged\ Floquet\ response",
    )
end


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------

params = FloquetParams(
    N = 10,
    n_steps = 10_000,
    n_realizations = 10,
    tau = 1.0,
    epsilon = 0.03,
    disorder_width = 0.1 * 2π,
    cutoff = 1e-9,
)

signal = disorder_averaged_dynamics(params; seed = 1234)

println(signal)

ENV["GKS_ENCODING"] = "utf-8"
plot_floquet_signal(signal, params)
