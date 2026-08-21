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

@inline function _density_matrix_entries_are_finite(
        matrix::AbstractMatrix)
    values = if matrix isa StridedMatrix
        matrix
    elseif matrix isa SparseMatrixCSC
        nonzeros(matrix)
    elseif matrix isa LocalMassMatrix
        matrix.vals
    else
        matrix
    end
    return all(isfinite, values)
end

function _validate_density_mass_inputs(
    Mt::AbstractVector{<:AbstractMatrix},
    rho_bar::AbstractVector{<:Real},
)
    _validate_density_values(rho_bar, length(Mt), "rho_bar")
    matrix_size = _validate_mass_matrix_sizes(Mt)
    @inbounds for t in eachindex(Mt)
        _density_matrix_entries_are_finite(Mt[t]) ||
            throw(ArgumentError(
                "Mt[$t] must contain only finite values"))
    end
    return matrix_size
end

# The exceptional path must retain a density power that underflows in
# Float64 before multiplication by a large, but finite, impedance. The guard
# precision also makes conversion of the final ComplexF64 coefficient stable.
const _DENSITY_INTERPOLATION_FALLBACK_PRECISION = 4352

@noinline function _density_derivative_coefficient_bigfloat(
        rho::Real, config::DensityConfig)
    return setprecision(
            BigFloat, _DENSITY_INTERPOLATION_FALLBACK_PRECISION) do
        p = BigFloat(config.p)
        density_power = isone(config.p) ?
                        one(BigFloat) : BigFloat(rho)^(p - 1)
        coefficient = ComplexF64(
            -p * density_power * Complex{BigFloat}(config.Z_max))
        isfinite(coefficient) ||
            throw(OverflowError(
                "density derivative coefficient is outside the " *
                "representable ComplexF64 range"))
        return coefficient
    end
end

@inline function _density_derivative_coefficient(
        rho::Real, config::DensityConfig)
    density_power = isone(config.p) ? one(Float64) :
                    rho^(config.p - 1)
    coefficient = -config.p * density_power * config.Z_max
    if !isfinite(coefficient) ||
       (!iszero(rho) && iszero(coefficient))
        return _density_derivative_coefficient_bigfloat(rho, config)
    end
    return coefficient
end

"""
    precompute_triangle_mass(mesh, rwg;
                             quad_order=3,
                             max_work_bytes=536_870_912,
                             max_terms=200_000_000)

Precompute per-triangle mass matrices M_t[m,n] = ∫_t f_m · f_n dS
for all triangles t = 1:Nt.

Returns a vector of compact local matrices, one per triangle.
Only basis functions with support on triangle t have nonzero entries in M_t.

`max_work_bytes` bounds the raw payload of the quadrature cache, support map,
triplet builders, compact results, and constructor transients. `max_terms`
bounds local basis-pair/quadrature evaluations. Both limits are checked before
the quadrature cache and triplet builders are allocated.
"""
function precompute_triangle_mass(
        mesh::TriMesh, rwg::RWGData;
        quad_order::Int=3,
        max_work_bytes::Integer=_DEFAULT_MAX_MASS_PRECOMPUTE_WORK_BYTES,
        max_terms::Integer=_DEFAULT_MAX_MASS_PRECOMPUTE_TERMS)
    _validate_mesh_rwg_pair(mesh, rwg)
    N = rwg.nedges
    Nt = ntriangles(mesh)
    Tcoef = promote_type(eltype(rwg.coeff_plus), eltype(rwg.coeff_minus))
    Tmass = Tcoef <: Real ? Float64 : ComplexF64

    xi, wq = tri_quad_rule(quad_order)
    Nq = length(wq)
    profile = _mass_precompute_profile(
        rwg, Nt, Nq, Tmass, nothing, Nt,
        max_work_bytes, max_terms)

    # Precompute quad points and areas
    quad_pts = [tri_quad_points(mesh, t, xi) for t in 1:Nt]
    areas = [triangle_area(mesh, t) for t in 1:Nt]

    # Map triangle → basis functions with support
    tri_to_basis = [Int[] for _ in 1:Nt]
    @inbounds for t in 1:Nt
        sizehint!(tri_to_basis[t], profile.degrees[t])
    end
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

                val = _local_surface_mass_entry(
                    Tmass, rwg, m, n, t, quad_pts[t], wq, A)

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
    assemble_Z_penalty(Mt, rho_bar, config;
                       max_output_bytes=2_000_000_000)

Assemble the density penalty matrix:

    Z_penalty = Σ_t (1 - ρ̄_t^p) * Z_max * M_t

where ρ̄ are the filtered/projected densities.

When ρ̄_t = 1 (metal): penalty contribution = 0
When ρ̄_t = 0 (void):  penalty contribution = Z_max * M_t (large impedance)
"""
function assemble_Z_penalty(Mt::Vector{<:AbstractMatrix},
                            rho_bar::AbstractVector{<:Real},
                            config::DensityConfig;
                            max_output_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    Nt = length(Mt)
    N = first(_validate_density_mass_inputs(Mt, rho_bar))
    CT = ComplexF64

    output_bytes = _checked_array_payload_bytes(
        CT, N, N; label="density penalty matrix")
    _enforce_payload_limit(
        output_bytes, max_output_bytes,
        "density penalty matrix", "max_output_bytes")

    Z_pen = zeros(CT, N, N)
    penalty_at = function(t)
        penalty = (1 - rho_bar[t]^config.p) * config.Z_max
        isfinite(penalty) ||
            throw(OverflowError(
                "density penalty coefficient for triangle $t is outside " *
                "the representable ComplexF64 range"))
        return penalty
    end
    _accumulate_scaled_matrices!(
        Z_pen, nothing, Mt, penalty_at,
        "density penalty matrix")

    all(isfinite, Z_pen) ||
        throw(OverflowError(
            "density penalty matrix contains entries outside the " *
            "representable ComplexF64 range"))
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
    coefficient = _density_derivative_coefficient(rho_bar[t], config)
    derivative = coefficient * Mt[t]
    _density_matrix_entries_are_finite(derivative) ||
        throw(OverflowError(
            "density derivative for triangle $t contains entries outside " *
            "the representable range"))
    return derivative
end
