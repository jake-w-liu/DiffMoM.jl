# DensityInterpolation.jl — Per-triangle density design variables for topology optimization
#
# Material interpolation for PEC/void metasurface topology optimization.
# Each triangle t has density ρ_t ∈ [0,1]:
#   ρ = 1 → PEC (metal), no penalty
#   ρ = 0 → void, large impedance penalty kills surface currents
#
# Penalty model (SIMP):
#   Z_total = Z_efie + Σ_t (1 - ρ̄_t^p) * Z_max * M_t
#
# where M_t[m,n] = ∫_t f_m · f_n dS is the triangle mass matrix,
# p is the SIMP penalization power, and Z_max is the void penalty.
#
# Following Tucek, Capek, Jelinek (IEEE TAP 2023) approach.

export precompute_triangle_mass, assemble_Z_penalty, assemble_dZ_drhobar
export DensityConfig

"""
    DensityConfig

Configuration for density-based topology optimization.
"""
struct DensityConfig
    p::Float64          # SIMP penalization power (typically 3)
    Z_max::ComplexF64   # void penalty impedance (real = resistive, imaginary = reactive)
    vf_target::Float64  # target volume fraction (metal fraction)

    function DensityConfig(p::Real, Z_max::Number, vf_target::Real)
        p_value = Float64(p)
        Z_value = ComplexF64(Z_max)
        vf_value = Float64(vf_target)
        isfinite(p_value) && p_value >= 1 ||
            throw(ArgumentError(
                "SIMP power p must be finite and at least 1, got $p"))
        isfinite(Z_value) && !iszero(Z_value) ||
            throw(ArgumentError(
                "Z_max must be finite and nonzero, got $Z_max"))
        isfinite(vf_value) && 0 <= vf_value <= 1 ||
            throw(ArgumentError(
                "vf_target must be finite and lie in [0, 1], got $vf_target"))
        return new(p_value, Z_value, vf_value)
    end
end

"""
    DensityConfig(; p=3.0, Z_max_factor=1000.0, eta0=376.730313668, vf_target=0.5, reactive=false)

Construct DensityConfig with default parameters.
If `reactive=true`, Z_max is purely imaginary (jX penalty), preserving power conservation.
If `reactive=false` (default), Z_max is real (resistive penalty, introduces artificial absorption).
"""
function DensityConfig(; p::Float64=3.0, Z_max_factor::Float64=1000.0,
                       eta0::Float64=376.730313668, vf_target::Float64=0.5,
                       reactive::Bool=false)
    isfinite(Z_max_factor) && Z_max_factor > 0 ||
        throw(ArgumentError(
            "Z_max_factor must be finite and positive, got $Z_max_factor"))
    isfinite(eta0) && eta0 > 0 ||
        throw(ArgumentError(
            "eta0 must be finite and positive, got $eta0"))
    Z_max = reactive ? im * Z_max_factor * eta0 : ComplexF64(Z_max_factor * eta0)
    return DensityConfig(p, Z_max, vf_target)
end

function _validate_density_values(values::AbstractVector{<:Real},
                                  expected_length::Int,
                                  name::AbstractString)
    length(values) == expected_length ||
        throw(DimensionMismatch(
            "$name length $(length(values)) != $expected_length"))
    all(value -> isfinite(value) && 0 <= value <= 1, values) ||
        throw(ArgumentError(
            "$name entries must all be finite and lie in [0, 1]"))
    return nothing
end

function _validate_density_mass_inputs(
    Mt::AbstractVector{<:AbstractMatrix},
    rho_bar::AbstractVector{<:Real},
)
    _validate_density_values(rho_bar, length(Mt), "rho_bar")
    return _validate_mass_matrix_sizes(Mt)
end

"""
    precompute_triangle_mass(mesh, rwg; quad_order=3)

Precompute per-triangle mass matrices M_t[m,n] = ∫_t f_m · f_n dS
for all triangles t = 1:Nt.

Returns a vector of compact local matrices, one per triangle.
Only basis functions with support on triangle t have nonzero entries in M_t.
"""
function precompute_triangle_mass(mesh::TriMesh, rwg::RWGData; quad_order::Int=3)
    N = rwg.nedges
    Nt = ntriangles(mesh)
    Tcoef = promote_type(eltype(rwg.coeff_plus), eltype(rwg.coeff_minus))
    Tmass = Tcoef <: Real ? Float64 : ComplexF64

    xi, wq = tri_quad_rule(quad_order)
    Nq = length(wq)

    # Precompute quad points and areas
    quad_pts = [tri_quad_points(mesh, t, xi) for t in 1:Nt]
    areas = [triangle_area(mesh, t) for t in 1:Nt]

    # Map triangle → basis functions with support
    tri_to_basis = [Int[] for _ in 1:Nt]
    for n in 1:N
        push!(tri_to_basis[rwg.tplus[n]], n)
        push!(tri_to_basis[rwg.tminus[n]], n)
    end

    Mt = Vector{LocalMassMatrix{Tmass}}(undef, Nt)

    for t in 1:Nt
        A = areas[t]
        basis_on_t = tri_to_basis[t]
        nnz_hint = length(basis_on_t)^2
        rows = Vector{Int}()
        cols = Vector{Int}()
        vals = Vector{Tmass}()
        sizehint!(rows, nnz_hint)
        sizehint!(cols, nnz_hint)
        sizehint!(vals, nnz_hint)

        for bi in eachindex(basis_on_t)
            m = basis_on_t[bi]
            for bj in eachindex(basis_on_t)
                n = basis_on_t[bj]

                val = zero(Tmass)
                for q in 1:Nq
                    rq = quad_pts[t][q]
                    fm = eval_rwg(rwg, m, rq, t)
                    fn = eval_rwg(rwg, n, rq, t)
                    val += wq[q] * dot(fm, fn)
                end
                val *= 2 * A  # reference-to-physical Jacobian

                if val != zero(Tmass)
                    push!(rows, m)
                    push!(cols, n)
                    push!(vals, val)
                end
            end
        end
        Mt[t] = LocalMassMatrix(N, rows, cols, vals)
    end

    return Mt
end

"""
    assemble_Z_penalty(Mt, rho_bar, config)

Assemble the density penalty matrix:

    Z_penalty = Σ_t (1 - ρ̄_t^p) * Z_max * M_t

where ρ̄ are the filtered/projected densities.

When ρ̄_t = 1 (metal): penalty contribution = 0
When ρ̄_t = 0 (void):  penalty contribution = Z_max * M_t (large impedance)
"""
function assemble_Z_penalty(Mt::Vector{<:AbstractMatrix},
                            rho_bar::AbstractVector{<:Real},
                            config::DensityConfig)
    Nt = length(Mt)
    N = first(_validate_density_mass_inputs(Mt, rho_bar))
    CT = ComplexF64

    Z_pen = zeros(CT, N, N)
    for t in 1:Nt
        penalty = (1 - rho_bar[t]^config.p) * config.Z_max
        _add_scaled_matrix!(Z_pen, penalty, Mt[t])
    end

    return Z_pen
end

"""
    assemble_dZ_drhobar(Mt, rho_bar, config, t)

Compute the derivative ∂Z_penalty/∂ρ̄_t for triangle t:

    ∂Z/∂ρ̄_t = -p * ρ̄_t^(p-1) * Z_max * M_t

This is exact and closed-form (no finite differences needed).
"""
function assemble_dZ_drhobar(Mt::Vector{<:AbstractMatrix},
                             rho_bar::AbstractVector{<:Real},
                             config::DensityConfig, t::Int)
    1 <= t <= length(Mt) || throw(BoundsError(Mt, t))
    _validate_density_mass_inputs(Mt, rho_bar)
    coeff = -config.p * rho_bar[t]^(config.p - 1) * config.Z_max
    return coeff * Mt[t]
end
