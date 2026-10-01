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

"""Assembled-ready planar problem: stackup, grid, sheets, ports, and the
rooftop basis built from them."""
struct PlanarProblem
    stack::PlanarStackup
    grid::CellGrid
    sheets::Vector{SheetLevel}
    ports::Vector{PlanarPort}
    basis::PlanarBasisSet
end

"""Validate the stackup/ports and build the rooftop basis; every port must
claim at least one wall-connected edge."""
function build_planar_problem(stack::PlanarStackup, grid::CellGrid,
        sheets::Vector{SheetLevel}, ports::Vector{PlanarPort})
    planar_validate(stack)
    isempty(ports) &&
        throw(ArgumentError("at least one port is required"))
    basis = build_planar_basis(grid, sheets, ports)
    planar_basis_count(basis) == 0 &&
        throw(ArgumentError("no basis functions: check masks/connections"))
    # every port must claim at least one basis
    claimed = falses(length(ports))
    @inbounds for p in eachindex(basis.port)
        basis.port[p] > 0 && (claimed[basis.port[p]] = true)
    end
    for (i, c) in enumerate(claimed)
        c || throw(ArgumentError(
            "port $i claimed no wall edges; check cells/wall/mask"))
    end
    return PlanarProblem(stack, grid, sheets, ports, basis)
end

"""Result of `solve_planar`: retained problem, MoM matrix and LU
factorization, per-port coefficient columns, short-circuit admittance `y`,
and S-parameters `s` normalized to the port reference impedances."""
struct PlanarResult
    problem::PlanarProblem
    omega::ComplexF64              # complex (supports complex-step omega)
    freq::ComplexF64               # Hz
    z_mom::Matrix{ComplexF64}      # assembled impedance matrix
    lu_fact::LinearAlgebra.LU{ComplexF64,Matrix{ComplexF64}}
    currents::Matrix{ComplexF64}   # nb x nport coefficient columns
    y::Matrix{ComplexF64}          # short-circuit port admittance [S]
    s::Matrix{ComplexF64}          # S-parameters normalized to port z0
end

"""Port-edge index lists from a basis set (basis.port marks port index)."""
function _port_basis_indices(basis::PlanarBasisSet, nports::Int)
    lists = [Int[] for _ in 1:nports]
    @inbounds for b in 1:planar_basis_count(basis)
        p = basis.port[b]
        p > 0 && push!(lists[p], b)
    end
    return lists
end

"""
    solve_planar(problem, freq; kw...) -> PlanarResult

Assemble the shielded planar MoM system at `freq` Hz, drive each port with
a unit gap voltage, and extract the short-circuit admittance and
S-parameters.
"""
function solve_planar(prob::PlanarProblem, freq::Number; kw...)
    omega = 2pi * ComplexF64(freq)
    isfinite(omega) && real(omega) > 0 ||
        throw(ArgumentError(
            "freq must be finite with Re > 0, got $freq"))
    nb = planar_basis_count(prob.basis)
    nports = length(prob.ports)

    Z = assemble_planar_z(prob.stack, prob.grid, prob.sheets,
        prob.basis, omega; kw...)
    pbs = _port_basis_indices(prob.basis, nports)

    est = _checked_payload_sum("planar solve",
        _checked_array_payload_bytes(ComplexF64, nb, nb),   # Z
        _checked_array_payload_bytes(ComplexF64, nb, nb),   # lu(Z) copy
        _checked_array_payload_bytes(ComplexF64, nb, nports),
        _checked_array_payload_bytes(Int, nb))
    _enforce_payload_limit(est, get(kw, :max_bytes,
        _DEFAULT_MAX_DENSE_PAYLOAD_BYTES), "planar solve", "max_bytes")

    # a non-finite entry means an exact box resonance (infinite modal
    # voltage) -- LU would return silent garbage, so fail loudly
    all(isfinite, Z) || throw(ArgumentError(
        "planar impedance matrix is non-finite: frequency is at/near " *
        "a box resonance"))
    F = lu(Z)
    # s_p: +1 for :west/:south, -1 for :east/:north (into-network direction)
    sgn = [p.wall in (:west, :south) ? 1.0 : -1.0 for p in prob.ports]
    RHS = zeros(ComplexF64, nb, nports)
    @inbounds for q in 1:nports
        isempty(pbs[q]) && throw(ArgumentError(
            "port $q claimed no basis functions"))
        for b in pbs[q]
            RHS[b, q] = -sgn[q] * prob.basis.width[b]
        end
    end
    X = ldiv!(F, RHS)   # solves in place over the RHS buffer

    # port currents and admittance
    Y = Matrix{ComplexF64}(undef, nports, nports)
    @inbounds for q in 1:nports, p in 1:nports
        acc = zero(ComplexF64)
        for b in pbs[p]
            acc += sgn[p] * prob.basis.width[b] * X[b, q]
        end
        Y[p, q] = acc
    end

    z0 = ComplexF64[p.z0 for p in prob.ports]
    S = planar_y_to_s(Y, z0)
    return PlanarResult(prob, omega, ComplexF64(freq), Z, F, X, Y, S)
end

"""`planar_sparams(problem, freqs; kw...) -> (S vector of matrices)`.
Sweeps `freqs` (Hz) re-assembly per point."""
function planar_sparams(prob::PlanarProblem,
        freqs::AbstractVector{<:Number}; kw...)
    return [solve_planar(prob, f; kw...).s for f in freqs]
end

"""
    planar_y_to_s(Y, z0) -> S

Convert short-circuit admittance matrix to S-parameters against per-port
reference impedances `z0` (real, positive).  Power-wave normalization:
S = D^(-1/2) (I - Z0*Y) (I + Z0*Y)^(-1) D^(1/2)  with D = diag(z0);
the similarity factors cancel only when all ports share the same z0.
"""
function planar_y_to_s(Y::AbstractMatrix{<:Number},
        z0::AbstractVector{<:Number})
    n = size(Y, 1)
    size(Y, 2) == n && length(z0) == n ||
        throw(DimensionMismatch("Y must be n x n with n = length(z0)"))
    for (p, z) in enumerate(z0)
        isfinite(real(z)) && imag(z) == 0 && real(z) > 0 ||
            throw(ArgumentError(
                "port $p reference impedance must be real and positive, " *
                "got $z"))
    end
    Z0 = LinearAlgebra.Diagonal([ComplexF64(z) for z in z0])
    A = Matrix{ComplexF64}(I, n, n)
    P = (A - Z0 * Y) / (A + Z0 * Y)
    # similarity sandwich D^(-1/2) P D^(1/2): S[p,q] *= sqrt(z0_q/z0_p)
    r = sqrt.([real(z) for z in z0])
    S = Matrix{ComplexF64}(undef, n, n)
    @inbounds for q in 1:n, p in 1:n
        S[p, q] = P[p, q] * (r[q] / r[p])
    end
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
    n = size(S[1], 1)
    open(path, "w") do io
        println(io, "! DiffMoM planar S-parameters")
        println(io, "# HZ S RI R ", Float64(z0))
        for k in eachindex(freqs)
            mat = S[k]
            size(mat) == (n, n) || throw(DimensionMismatch(
                "S[$k] is $(size(mat)), expected $n x $n"))
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
