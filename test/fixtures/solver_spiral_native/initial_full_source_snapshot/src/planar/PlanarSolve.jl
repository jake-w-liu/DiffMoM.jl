# PlanarSolve.jl — Port excitation, solve, and network-parameter extraction
#
# Port model (Rautio & Harrington 1987 / co-calibrated port theory): a port
# is an infinitesimal gap-voltage source placed between the metal edge and
# the adjacent sidewall (wall port) or inside the sheet (internal port).
# The Galerkin RHS for a gap voltage V on basis b of lateral width w_b is
#     rhs_b = -s_q * w_b * V        (induced E cancels applied E on metal)
# where s_q = +1 for :west/:south ports and -1 for :east/:north ports so
# that positive V drives current *into* the network on every wall.  The
# port current into the network is I_p = s_p * sum_b w_b * i_b, hence
#     Y[p,q] = -s_p s_q * sum_{b in p, b' in q} w_b Z^{-1}[b,b'] w_b'.
# S-parameters follow from S = (I - Z0*Y)(I + Z0*Y)^{-1}.

export PlanarProblem, PlanarResult
export build_planar_problem, solve_planar, planar_sparams
export planar_y_to_s, write_touchstone

"""Assembled-ready planar problem: stackup, grid, sheets, ports, via
levels, and the rooftop basis built from them."""
struct PlanarProblem
    stack::PlanarStackup
    grid::CellGrid
    sheets::Vector{SheetLevel}
    ports::Vector{PlanarPort}
    vias::Vector{ViaLevel}
    basis::PlanarBasisSet
    vols::Vector{VolLevel}
end

# convenience constructors: no via/volume levels, or via only
PlanarProblem(stack::PlanarStackup, grid::CellGrid,
    sheets::Vector{SheetLevel}, ports::Vector{PlanarPort},
    basis::PlanarBasisSet) =
    PlanarProblem(stack, grid, sheets, ports, ViaLevel[], basis,
        VolLevel[])
PlanarProblem(stack::PlanarStackup, grid::CellGrid,
    sheets::Vector{SheetLevel}, ports::Vector{PlanarPort},
    vias::Vector{ViaLevel}, basis::PlanarBasisSet) =
    PlanarProblem(stack, grid, sheets, ports, vias, basis, VolLevel[])

"""Validate the stackup/ports and build the rooftop basis; every port must
claim at least one wall-connected edge.  `vias` adds z-directed via
columns on `ViaLevel`s (uniform and up-tapered profiles per cell);
`vols` adds thick-metal volume rooftops on `VolLevel`s (x- and
y-directed volume current distributed uniformly through each level's
layer).  Every level's `layer` must lie inside the stackup."""
function build_planar_problem(stack::PlanarStackup, grid::CellGrid,
        sheets::Vector{SheetLevel}, ports::Vector{PlanarPort};
        vias::Vector{ViaLevel}=ViaLevel[],
        vols::Vector{VolLevel}=VolLevel[])
    planar_validate(stack)
    stack.a == grid.a && stack.b == grid.b || throw(ArgumentError(
        "stackup box dimensions must match the cell grid"))
    isempty(ports) &&
        throw(ArgumentError("at least one port is required"))
    L = length(stack.layers)
    for (vl, vlvl) in enumerate(vias)
        1 <= vlvl.layer <= L || throw(ArgumentError(
            "vias[$vl] layer $(vlvl.layer) outside 1:$L"))
    end
    for (vl, vlvl) in enumerate(vols)
        1 <= vlvl.layer <= L || throw(ArgumentError(
            "vols[$vl] layer $(vlvl.layer) outside 1:$L"))
    end
    basis = build_planar_basis(grid, sheets, ports; vias=vias, vols=vols)
    planar_basis_count(basis) == 0 &&
        throw(ArgumentError("no basis functions: check masks/connections"))
    # every port must claim at least one basis
    claimed = falses(length(ports))
    @inbounds for p in eachindex(basis.port)
        basis.port[p] > 0 && (claimed[basis.port[p]] = true)
    end
    for (i, c) in enumerate(claimed)
        c || throw(ArgumentError(
            "port $i claimed no basis functions; check cells/edge/mask"))
    end
    return PlanarProblem(stack, grid, sheets, ports, vias, basis, vols)
end

"""Result of `solve_planar`: retained problem, MoM matrix and LU
factorization, per-port coefficient columns, short-circuit admittance `y`,
and S-parameters `s` normalized to the port reference impedances."""
struct PlanarResult
    problem::PlanarProblem
    omega::ComplexF64              # complex (supports complex-step omega)
    freq::ComplexF64               # Hz
    z_mom::Union{Nothing,Matrix{ComplexF64}} # omitted with retain_matrix=false
    lu_fact::LinearAlgebra.LU{ComplexF64,Matrix{ComplexF64}}
    currents::Matrix{ComplexF64}   # nb x nport coefficient columns
    y::Matrix{ComplexF64}          # short-circuit port admittance [S]
    s::Matrix{ComplexF64}          # S-parameters normalized to port z0
    z0::Vector{ComplexF64}         # references evaluated at this frequency
end

# Preserve the original result constructor for callers retaining their
# own solved coefficients. New solver paths pass the evaluated refs directly.
PlanarResult(prob::PlanarProblem,omega,freq,Z,F,X,Y,S)=PlanarResult(prob,omega,freq,Z,F,X,Y,S,
    _planar_reference_values([p.z0 for p in prob.ports],length(prob.ports);freq=real(freq)))

"""Port-edge index lists from a basis set (basis.port marks port index)."""
function _port_basis_indices(basis::PlanarBasisSet, nports::Int)
    lists = [Int[] for _ in 1:nports]
    @inbounds for b in 1:planar_basis_count(basis)
        p = basis.port[b]
        p > 0 && push!(lists[p], b)
    end
    return lists
end

@inline _planar_port_sign(p::PlanarPort) =
    p.polarity * (p.wall in (:east,:north,:volume_east,:volume_north,
        :terminal_x_hi,:terminal_y_hi) ? -1.0 : 1.0)

# A distributed axial voltage V/h reacts with the average via profile:
# uniform = 1, up-taper = 1/2. Horizontal gap ports use the edge width.
@inline _planar_port_weight(basis::PlanarBasisSet, b::Int) =
    basis.width[b] * (basis.kind[b] == _BASIS_VIA_T ? 0.5 : 1.0)

"""
    solve_planar(problem, freq; kw...) -> PlanarResult

Assemble the shielded planar MoM system at `freq` Hz, drive each port with
a unit gap voltage, and extract the short-circuit admittance and
S-parameters. `retain_matrix=false` factors the assembly buffer in place
and returns `z_mom=nothing`, saving one dense matrix while retaining the
factorization and currents. `max_bytes` bounds operation-owned raw array
payloads and is checked before assembly.
"""
function solve_planar(prob::PlanarProblem, freq::Number;
        method::Symbol=:dense,
        retain_matrix::Bool=true,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES, kw...)
    method === :ufft && return solve_planar_ufft(prob, freq;
        max_bytes=max_bytes, kw...)
    method in (:dense,:dense_fft) || throw(ArgumentError("method must be :dense, :dense_fft or :ufft"))
    omega = 2pi * ComplexF64(freq)
    isfinite(omega) && real(omega) > 0 ||
        throw(ArgumentError(
            "freq must be finite with Re > 0, got $freq"))
    nb = planar_basis_count(prob.basis)
    nports = length(prob.ports)

    est = _checked_dense_lu_work_bytes(ComplexF64, nb,
        retain_matrix ? 2 : 1,
        _checked_array_payload_bytes(ComplexF64, nb, nports),
        # Y and the two buffers used by power-wave conversion.
        _checked_array_payload_bytes(ComplexF64, 3, nports, nports),
        _checked_array_payload_bytes(Int, nb), # port index lists
        _checked_array_payload_bytes(Int, nports), # conversion LU pivots
        _checked_array_payload_bytes(Float64, 2, nports),
        _checked_array_payload_bytes(ComplexF64, 2, nports);
        label="planar solve")
    _enforce_payload_limit(est, max_bytes, "planar solve", "max_bytes")
    # Validate reference impedances before an expensive assembly/solve.
    z0 = _planar_reference_values([p.z0 for p in prob.ports], nports;freq=real(freq))

    Z = if method===:dense_fft
        # LU/currents/port buffers coexist with the matrix. Reserve these
        # before constructing FFT kernels rather than granting both paths
        # the complete aggregate budget independently.
        matrix_bytes=_checked_array_payload_bytes(ComplexF64,nb,nb)
        assemble_planar_z_ufft(prob,freq;max_bytes=max_bytes-(est-matrix_bytes),kw...)
    else
        assemble_planar_z(prob.stack, prob.grid, prob.sheets,
            prob.basis, omega; vias=prob.vias, vols=prob.vols,
            max_bytes=max_bytes, kw...)
    end
    pbs = _port_basis_indices(prob.basis, nports)

    # a non-finite entry means an exact box resonance (infinite modal
    # voltage) -- LU would return silent garbage, so fail loudly
    all(isfinite, Z) || throw(ArgumentError(
        "planar impedance matrix is non-finite: frequency is at/near " *
        "a box resonance"))
    F = retain_matrix ? lu(Z) : lu!(Z)
    # s_p: +1 for :west/:south, -1 for :east/:north (into-network direction)
    sgn = [_planar_port_sign(p) for p in prob.ports]
    RHS = zeros(ComplexF64, nb, nports)
    @inbounds for q in 1:nports
        isempty(pbs[q]) && throw(ArgumentError(
            "port $q claimed no basis functions"))
        for b in pbs[q]
            RHS[b, q] = -sgn[q] * _planar_port_weight(prob.basis, b)
        end
    end
    X = ldiv!(F, RHS)   # solves in place over the RHS buffer

    # port currents and admittance
    Y = Matrix{ComplexF64}(undef, nports, nports)
    @inbounds for q in 1:nports, p in 1:nports
        acc = zero(ComplexF64)
        for b in pbs[p]
            acc += sgn[p] * _planar_port_weight(prob.basis, b) * X[b, q]
        end
        Y[p, q] = acc
    end

    S = planar_y_to_s(Y, z0)
    return PlanarResult(prob, omega, ComplexF64(freq),
        retain_matrix ? Z : nothing, F, X, Y, S, z0)
end

"""`planar_sparams(problem, freqs; kw...) -> (S vector of matrices)`.
Sweeps `freqs` (Hz) re-assembly per point."""
function planar_sparams(prob::PlanarProblem,
        freqs::AbstractVector{<:Number}; retain_matrix::Bool=false, kw...)
    return [solve_planar(prob, f; retain_matrix=retain_matrix, kw...).s
            for f in freqs]
end

function _planar_reference_roots(z0::AbstractVector)
    r = Vector{Float64}(undef, length(z0))
    for (p, z) in enumerate(z0)
        r[p] = sqrt(real(_planar_reference_number(z)))
    end
    return r
end

"""
    planar_y_to_s(Y, z0) -> S

Convert short-circuit admittance matrix to S-parameters against per-port
reference impedances `z0` (finite, positive real parts). Kurokawa power
waves use `a=(V+Z0*I)/(2sqrt(real(Z0)))` and
`b=(V-conj(Z0)*I)/(2sqrt(real(Z0)))`. For complex references, an exact
short reflects as `-conj(Z0)/Z0` and conjugate matching gives zero reflection.
"""
function planar_y_to_s(Y::AbstractMatrix{<:Number},
        z0::AbstractVector)
    n = size(Y, 1)
    size(Y, 2) == n && length(z0) == n ||
        throw(DimensionMismatch("Y must be n x n with n = length(z0)"))
    n > 0 || throw(ArgumentError("Y must have at least one port"))
    all(isfinite, Y) || throw(ArgumentError("Y must be finite"))
    r = _planar_reference_roots(z0)
    K = Matrix{ComplexF64}(undef, n, n)
    @inbounds for q in 1:n, p in 1:n
        K[p, q] = (Y[p,q] * ComplexF64(z0[q])) * (r[p]/r[q])
        p == q && (K[p, q] += 1)
    end
    all(isfinite,K) || throw(ArgumentError("power-wave admittance normalization exceeds finite ComplexF64 values"))
    # K = I + H*Q, H=√R Y√R, Q=Z0/ReZ0. Rewriting I-2K⁻¹H
    # as 2K⁻¹Q⁻¹-diag(conj(Z0)/Z0) preserves tiny transmission
    # entries near a short. The second buffer starts as the identity.
    S=Matrix{ComplexF64}(I,n,n)
    ldiv!(lu!(K), S)
    @inbounds for q in 1:n, p in 1:n
        ref=ComplexF64(z0[q])
        diagonal=p==q ? 1.0 : 0.0
        S[p, q] = diagonal+2*(real(ref)/ref)*(S[p,q]-diagonal)
    end
    all(isfinite,S) || throw(ArgumentError("power-wave scattering conversion is nonfinite"))
    return S
end

"""Write an `n`-port Touchstone (.sNp) file, S in real/imag pairs, Hz.
Two-port files list `S11 S21 S12 S22` on the frequency line; `n > 2`
files put the frequency on its own line followed by one matrix row per
line in row-major order."""
function write_touchstone(path::AbstractString,
        freqs::AbstractVector{<:Real},
        S::AbstractVector{<:AbstractMatrix},
        z0::Real=50.0)
    length(freqs) == length(S) ||
        throw(DimensionMismatch("freqs and S lengths differ"))
    isempty(S) && throw(ArgumentError("at least one frequency is required"))
    isfinite(z0) && z0 > 0 || throw(ArgumentError(
        "reference impedance must be finite and positive"))
    n = size(S[1], 1)
    n > 0 || throw(ArgumentError("S must have at least one port"))
    # Validate the complete dataset before opening an existing file.
    for k in eachindex(freqs)
        isfinite(freqs[k]) && freqs[k] >= 0 || throw(ArgumentError(
            "freqs[$k] must be finite and nonnegative"))
        size(S[k]) == (n, n) || throw(DimensionMismatch(
            "S[$k] is $(size(S[k])), expected $n x $n"))
        all(isfinite, S[k]) || throw(ArgumentError("S[$k] must be finite"))
    end
    open(path, "w") do io
        println(io, "! DiffMoM planar S-parameters")
        println(io, "# HZ S RI R ", Float64(z0))
        for k in eachindex(freqs)
            mat = S[k]
            if n <= 2
                print(io, Float64(freqs[k]))
                for q in 1:n, p in 1:n
                    print(io, "  ", real(mat[p, q]), "  ",
                        imag(mat[p, q]))
                end
                println(io)
            else
                println(io, Float64(freqs[k]))
                for p in 1:n
                    for q in 1:n
                        print(io, "  ", real(mat[p, q]), "  ",
                            imag(mat[p, q]))
                    end
                    println(io)
                end
            end
        end
    end
    return path
end
