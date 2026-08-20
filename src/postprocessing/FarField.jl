# FarField.jl — Far-field radiation pattern computation
#
# E∞(r̂) = (ik η₀)/(4π) r̂ × [r̂ × ∫ J(r') exp(ik r̂·r') dS']
#        = Σ_n I_n g_n(r̂)

export make_sph_grid, radiation_vectors, compute_farfield, incident_farfield

const _DEFAULT_MAX_SPH_GRID_POINTS = 2_100_000
const _DEFAULT_MAX_SPH_GRID_RAW_BYTES = 134_400_000
const _SPH_GRID_RAW_BYTES_PER_POINT = 6sizeof(Float64)
const _DEFAULT_MAX_RADIATION_WORK_BYTES = 512 * 1024 * 1024
const _DEFAULT_MAX_RADIATION_TERMS = 200_000_000
const _DEFAULT_MAX_RADIATION_EXACT_WORK = 2_000_000
const _RADIATION_EXACT_PRECISION = 4352

function _radiation_vectors_work_bytes(
        Nt::Int, Nq::Int, NΩ::Int, output_bytes::Int)
    total = BigInt(output_bytes)
    # Quadrature points/areas, cached observation directions, quadrature-rule
    # arrays, and the outer quadrature-vector references coexist with G_mat.
    total += BigInt(sizeof(Ptr{Cvoid}) + sizeof(Float64)) * Nt
    total += BigInt(sizeof(Vec3)) * Nt * Nq
    total += BigInt(sizeof(Vec3)) * NΩ
    total += BigInt(sizeof(SVector{2,Float64}) + sizeof(Float64)) * Nq
    total <= typemax(Int) ||
        throw(ArgumentError(
            "radiation-vector raw-workspace estimate overflows Int"))
    return Int(total)
end

function _radiation_vectors_term_count(N::Int, Nq::Int, NΩ::Int)
    terms = BigInt(2) * N * Nq * NΩ
    terms <= typemax(Int) ||
        throw(ArgumentError("radiation-vector term estimate overflows Int"))
    return Int(terms)
end

@inline function _radiation_scale_requires_exact(k::Number, eta0::Number)
    product = k * eta0
    return !isfinite(product) ||
           (!iszero(k) && !iszero(eta0) &&
            (iszero(product) || abs(product) < floatmin(Float64)))
end

@inline function _radiation_geometry_requires_exact(
        mesh::TriMesh, rwg::RWGData, grid::SphGrid,
        k::Number, eta0::Number)
    (_ieee_dense_extreme_factor(k, Float64) ||
     _ieee_dense_extreme_factor(eta0, Float64)) && return true
    maximum_coordinates = MVector{3,Float64}(0.0, 0.0, 0.0)
    @inbounds for vertex in axes(mesh.xyz, 2), component in 1:3
        value = mesh.xyz[component, vertex]
        _ieee_dense_extreme_factor(value, Float64) && return true
        maximum_coordinates[component] = max(
            maximum_coordinates[component], abs(value))
    end
    @inbounds for values in (
            rwg.len, rwg.area_plus, rwg.area_minus,
            rwg.coeff_plus, rwg.coeff_minus)
        for value in values
            _ieee_dense_extreme_factor(value, Float64) && return true
        end
    end
    @inbounds for component in grid.rhat
        _ieee_dense_extreme_factor(component, Float64) && return true
    end
    coordinate_bound = Vec3(maximum_coordinates)
    real_k = Float64(real(k))
    imag_k = Float64(imag(k))
    @inbounds for direction in axes(grid.rhat, 2)
        rhat = Vec3(
            grid.rhat[1, direction],
            grid.rhat[2, direction],
            grid.rhat[3, direction])
        _source_phase_requires_fallback(
            real_k, rhat, coordinate_bound) && return true
        _source_phase_requires_fallback(
            imag_k, rhat, coordinate_bound) && return true
    end
    return false
end

@noinline function _radiation_vector_column_exact!(
        G_mat::Matrix{ComplexF64}, n::Int,
        rwg::RWGData, rhat_vec::Vector{Vec3},
        xi::Vector{<:SVector{2}}, areas::Vector{Float64},
        wq::Vector{Float64}, k::Number, eta0::Number)
    setprecision(BigFloat, _RADIATION_EXACT_PRECISION) do
        k_big = Complex{BigFloat}(k)
        eta_big = Complex{BigFloat}(eta0)
        prefactor_big = im * k_big * eta_big / (4 * BigFloat(pi))
        @inbounds for q_dir in eachindex(rhat_vec)
            rh_big = SVector{3,BigFloat}(BigFloat.(rhat_vec[q_dir]))
            total = zeros(Complex{BigFloat}, 3)
            for t in (rwg.tplus[n], rwg.tminus[n])
                weight_scale = 2 * BigFloat(areas[t])
                is_plus = t == rwg.tplus[n]
                coefficient_big = Complex{BigFloat}(
                    is_plus ? rwg.coeff_plus[n] : rwg.coeff_minus[n])
                edge_scale_big = coefficient_big * BigFloat(rwg.len[n]) /
                                 (2 * BigFloat(areas[t]))
                opposite_vertex = is_plus ?
                    rwg.vplus_opp[n] : rwg.vminus_opp[n]
                opposite_big = SVector{3,BigFloat}(
                    BigFloat.(Tuple(_mesh_vertex(rwg.mesh, opposite_vertex))))
                vertex1_big = SVector{3,BigFloat}(BigFloat.(Tuple(
                    _mesh_vertex(rwg.mesh, rwg.mesh.tri[1, t]))))
                vertex2_big = SVector{3,BigFloat}(BigFloat.(Tuple(
                    _mesh_vertex(rwg.mesh, rwg.mesh.tri[2, t]))))
                vertex3_big = SVector{3,BigFloat}(BigFloat.(Tuple(
                    _mesh_vertex(rwg.mesh, rwg.mesh.tri[3, t]))))
                for q_surf in eachindex(wq)
                    xi1 = BigFloat(xi[q_surf][1])
                    xi2 = BigFloat(xi[q_surf][2])
                    rp_big = vertex1_big * (1 - xi1 - xi2) +
                             vertex2_big * xi1 + vertex3_big * xi2
                    delta_big = is_plus ?
                        rp_big - opposite_big : opposite_big - rp_big
                    fn_big = edge_scale_big * delta_big
                    phase = exp(im * k_big * dot(rh_big, rp_big))
                    contribution = fn_big *
                        (BigFloat(wq[q_surf]) * weight_scale * phase)
                    projected =
                        rh_big * dot(rh_big, contribution) - contribution
                    total .+= prefactor_big .* projected
                end
            end
            idx = 3 * (q_dir - 1)
            for component in 1:3
                converted = ComplexF64(total[component])
                isfinite(converted) ||
                    throw(OverflowError(
                        "radiation-vector result is outside the ComplexF64 " *
                        "range at direction $q_dir, basis $n, component $component"))
                G_mat[idx + component, n] = converted
            end
        end
    end
    return nothing
end

"""
    make_sph_grid(Ntheta, Nphi;
                  max_points=2_100_000,
                  max_raw_bytes=134_400_000)

Create a spherical grid using a uniform midpoint rule in θ and φ, with
quadrature weights w = sin(θ) dθ dφ.
Returns a SphGrid.
"""
function make_sph_grid(
        Ntheta::Int, Nphi::Int;
        max_points::Int=_DEFAULT_MAX_SPH_GRID_POINTS,
        max_raw_bytes::Int=_DEFAULT_MAX_SPH_GRID_RAW_BYTES)
    Ntheta > 0 ||
        throw(ArgumentError("Ntheta must be positive, got $Ntheta"))
    Nphi > 0 ||
        throw(ArgumentError("Nphi must be positive, got $Nphi"))
    max_points > 0 ||
        throw(ArgumentError("max_points must be positive, got $max_points"))
    max_raw_bytes > 0 ||
        throw(ArgumentError(
            "max_raw_bytes must be positive, got $max_raw_bytes"))

    # Simple uniform grid (sufficient for moderate resolution)
    dtheta = π / Ntheta
    dphi   = 2π / Nphi

    NΩ = try
        Base.checked_mul(Ntheta, Nphi)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError(
            "Ntheta*Nphi overflows Int: $Ntheta*$Nphi"))
    end
    NΩ <= max_points ||
        throw(ArgumentError(
            "spherical grid requires $NΩ points, exceeding " *
            "max_points=$max_points"))
    raw_bytes = try
        Base.checked_mul(NΩ, _SPH_GRID_RAW_BYTES_PER_POINT)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError(
            "spherical-grid raw storage estimate overflows Int"))
    end
    raw_bytes <= max_raw_bytes ||
        throw(ArgumentError(
            "spherical grid requires $raw_bytes raw bytes, exceeding " *
            "max_raw_bytes=$max_raw_bytes"))
    rhat  = zeros(3, NΩ)
    theta = zeros(NΩ)
    phi   = zeros(NΩ)
    w     = zeros(NΩ)

    idx = 0
    for it in 1:Ntheta
        θ = (it - 0.5) * dtheta
        for ip in 1:Nphi
            φ = (ip - 0.5) * dphi
            idx += 1
            theta[idx] = θ
            phi[idx]   = φ
            rhat[1, idx] = sin(θ) * cos(φ)
            rhat[2, idx] = sin(θ) * sin(φ)
            rhat[3, idx] = cos(θ)
            w[idx] = sin(θ) * dtheta * dphi
        end
    end

    return SphGrid(rhat, theta, phi, w)
end

function _validate_sph_grid(grid::SphGrid)
    NΩ = length(grid.w)
    NΩ > 0 ||
        throw(ArgumentError("spherical grid must contain at least one direction"))
    size(grid.rhat) == (3, NΩ) ||
        throw(DimensionMismatch(
            "grid.rhat has size $(size(grid.rhat)), expected (3, $NΩ)"))
    length(grid.theta) == NΩ ||
        throw(DimensionMismatch(
            "grid.theta length $(length(grid.theta)) != $NΩ"))
    length(grid.phi) == NΩ ||
        throw(DimensionMismatch(
            "grid.phi length $(length(grid.phi)) != $NΩ"))

    @inbounds for q in 1:NΩ
        θ = grid.theta[q]
        ϕ = grid.phi[q]
        wq = grid.w[q]
        (isfinite(θ) && isfinite(ϕ)) ||
            throw(ArgumentError(
                "spherical grid angles must be finite; direction $q has theta=$θ, phi=$ϕ"))
        (isfinite(wq) && wq >= 0.0) ||
            throw(ArgumentError(
                "spherical grid weights must be finite and nonnegative; weight $q is $wq"))

        x = grid.rhat[1, q]
        y = grid.rhat[2, q]
        z = grid.rhat[3, q]
        (isfinite(x) && isfinite(y) && isfinite(z)) ||
            throw(ArgumentError(
                "spherical grid directions must be finite; direction $q is ($x, $y, $z)"))
        nrm = sqrt(x * x + y * y + z * z)
        (isfinite(nrm) && isapprox(nrm, 1.0; rtol=1e-10, atol=1e-12)) ||
            throw(ArgumentError(
                "spherical grid directions must be unit vectors; direction $q has norm $nrm"))
    end
    return NΩ
end

@inline function _validated_farfield_scalar(value::Number,
                                            label::AbstractString;
                                            positive_real::Bool=false)
    (isfinite(real(value)) && isfinite(imag(value))) ||
        throw(ArgumentError("$label must be finite, got $value"))
    if positive_real
        isreal(value) && real(value) > 0.0 ||
            throw(ArgumentError("$label must be real and positive, got $value"))
    else
        abs(value) > 0.0 ||
            throw(ArgumentError("$label must be nonzero, got $value"))
    end
    return value
end

@inline function _validated_farfield_direction(r_hat::Vec3)
    x = r_hat[1]
    y = r_hat[2]
    z = r_hat[3]
    (isfinite(x) && isfinite(y) && isfinite(z)) ||
        throw(ArgumentError("far-field direction components must be finite, got $r_hat"))

    # Scale before taking the norm so valid very large or subnormal direction
    # vectors do not overflow or underflow during normalization.
    scale = max(abs(x), abs(y), abs(z))
    scale > 0.0 ||
        throw(ArgumentError("far-field direction must be nonzero"))
    scaled = r_hat / scale
    scaled_norm = norm(scaled)
    (isfinite(scaled_norm) && scaled_norm > 0.0) ||
        throw(ArgumentError("far-field direction must have a finite, nonzero norm"))
    return scaled / scaled_norm
end

@inline function _validated_incident_farfield_args(r_hat::Vec3, k::Real)
    kf = Float64(k)
    _validated_farfield_scalar(kf, "far-field wavenumber"; positive_real=true)
    return _validated_farfield_direction(r_hat), kf
end

@inline function _validate_incident_farfield_frequency(frequency::Float64,
                                                       k::Float64,
                                                       label::AbstractString)
    (isfinite(frequency) && frequency > 0.0) ||
        throw(ArgumentError(
            "$label frequency must be finite and positive, got $frequency"))
    expected_k = _frequency_to_wavenumber(frequency, _C0, label)
    isapprox(k, expected_k; rtol=1e-8, atol=0.0) ||
        throw(ArgumentError(
            "$label frequency implies wavenumber $expected_k, got $k"))
    return nothing
end

@inline function _validate_incident_farfield_source(
    excitation::Union{DipoleExcitation,LoopExcitation,MonopoleExcitation},
    k::Float64,
)
    _validate_excitation_model(excitation)
    _validate_incident_farfield_frequency(
        excitation.frequency, k, _incident_farfield_source_label(excitation))
    return nothing
end

@inline _incident_farfield_source_label(::DipoleExcitation) = "DipoleExcitation"
@inline _incident_farfield_source_label(::LoopExcitation) = "LoopExcitation"
@inline _incident_farfield_source_label(::MonopoleExcitation) = "MonopoleExcitation"

@inline function _validate_incident_farfield_source(
    excitation::PatternFeedExcitation,
    k::Float64,
)
    # Keep the per-direction path O(1): construction and assembly perform the
    # full coefficient-grid scan, while interpolation output is checked below.
    (isfinite(excitation.frequency) && excitation.frequency > 0.0) ||
        throw(ArgumentError(
            "PatternFeedExcitation frequency must be finite and positive"))
    all(isfinite, excitation.phase_center) ||
        throw(ArgumentError("PatternFeedExcitation phase_center must be finite"))
    excitation.convention in (:exp_plus_iwt, :exp_minus_iwt) ||
        throw(ArgumentError(
            "Unsupported pattern convention: $(excitation.convention)"))
    _validate_incident_farfield_frequency(
        excitation.frequency, k, "PatternFeedExcitation")
    return nothing
end

"""
    radiation_vectors(mesh, rwg, grid, k;
                      quad_order=3,
                      eta0=376.730313668,
                      max_output_bytes=2_000_000_000,
                      max_work_bytes=536_870_912,
                      max_terms=200_000_000,
                      max_exact_work=2_000_000)

Compute the per-basis radiation vectors g_n(r̂_q) for all basis functions
and all grid directions.

`max_output_bytes` bounds the returned matrix. `max_work_bytes` bounds that
matrix together with retained quadrature and direction workspaces, while
`max_terms` bounds the basis/support/quadrature/direction loop count.
When exceptional scale or geometry requires exact accumulation,
`max_exact_work` bounds quadrature terms times BigFloat precision.

Returns G_mat of size (3*NΩ, N) such that
  G_mat[(3*(q-1)+1):(3*q), n] = g_n(r̂_q) ∈ C³
"""
function radiation_vectors(
        mesh::TriMesh, rwg::RWGData, grid::SphGrid, k;
        quad_order::Int=3,
        eta0=376.730313668,
        max_output_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
        max_work_bytes::Integer=_DEFAULT_MAX_RADIATION_WORK_BYTES,
        max_terms::Integer=_DEFAULT_MAX_RADIATION_TERMS,
        max_exact_work::Integer=_DEFAULT_MAX_RADIATION_EXACT_WORK)
    N = rwg.nedges
    NΩ = _validate_sph_grid(grid)
    _validated_farfield_scalar(k, "radiation_vectors wavenumber")
    _validated_farfield_scalar(eta0, "radiation_vectors eta0")
    output_rows = try
        Base.Checked.checked_mul(3, NΩ)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError(
            "radiation-vector row count overflows Int"))
    end
    output_bytes = _checked_array_payload_bytes(
        ComplexF64, output_rows, N; label="radiation-vector matrix")
    _enforce_payload_limit(
        output_bytes, max_output_bytes,
        "radiation-vector matrix", "max_output_bytes")
    Nt = ntriangles(mesh)

    xi, wq = tri_quad_rule(quad_order)
    Nq = length(wq)
    _enforce_payload_limit(
        _radiation_vectors_work_bytes(Nt, Nq, NΩ, output_bytes),
        max_work_bytes,
        "radiation-vector output and workspace", "max_work_bytes")
    term_count = _radiation_vectors_term_count(N, Nq, NΩ)
    term_limit = _validated_resource_limit("max_terms", max_terms)
    term_count <= term_limit ||
        throw(ArgumentError(
            "radiation-vector assembly requires $term_count terms, " *
            "exceeding max_terms=$term_limit"))
    exact_scale = _radiation_scale_requires_exact(k, eta0)
    exact_geometry = exact_scale ||
        _radiation_geometry_requires_exact(mesh, rwg, grid, k, eta0)
    exact_limit = _validated_resource_limit("max_exact_work", max_exact_work)
    if exact_geometry
        exact_work = BigInt(term_count) * _RADIATION_EXACT_PRECISION
        exact_work <= exact_limit ||
            throw(ArgumentError(
                "exact radiation-vector assembly requires $exact_work " *
                "precision-weighted terms, exceeding " *
                "max_exact_work=$exact_limit"))
    end
    prefactor = exact_geometry ? 0.0im : 1im * k * eta0 / (4π)

    areas = Vector{Float64}(undef, Nt)
    @inbounds for t in 1:Nt
        areas[t] = triangle_area(mesh, t)
    end
    quad_pts = exact_geometry ? Vector{Vector{Vec3}}() :
        [tri_quad_points(mesh, t, xi) for t in 1:Nt]

    # Precompute rhat Vec3
    rhat_vec = Vector{Vec3}(undef, NΩ)
    @inbounds for q in 1:NΩ
        rhat_vec[q] = Vec3(grid.rhat[1, q], grid.rhat[2, q], grid.rhat[3, q])
    end

    # Assemble G_mat with parallelization over basis functions. Each phase value
    # is consumed exactly once for a given basis/triangle/quadrature/direction
    # tuple, so compute it inline instead of allocating an NΩ × Nq buffer for
    # every basis function. Each task writes only to column n of G_mat, so there
    # are no data races.
    G_mat = zeros(ComplexF64, output_rows, N)

    if exact_geometry
        for n in 1:N
            _radiation_vector_column_exact!(
                G_mat, n, rwg, rhat_vec, xi, areas, wq, k, eta0)
        end
    else
        Threads.@threads for n in 1:N
            @inbounds for t in (rwg.tplus[n], rwg.tminus[n])
                A = areas[t]
                pts = quad_pts[t]

                for q_surf in 1:Nq
                    rp = pts[q_surf]
                    fn = eval_rwg(rwg, n, rp, t)
                    wt = wq[q_surf] * 2 * A

                    for q_dir in 1:NΩ
                        rh = rhat_vec[q_dir]
                        phase = exp(1im * k * dot(rh, rp))
                        contrib = fn * (wt * phase)
                        rh_cross_N_cross =
                            rh * dot(rh, contrib) - contrib
                        idx = 3 * (q_dir - 1)
                        G_mat[idx+1, n] += prefactor * rh_cross_N_cross[1]
                        G_mat[idx+2, n] += prefactor * rh_cross_N_cross[2]
                        G_mat[idx+3, n] += prefactor * rh_cross_N_cross[3]
                    end
                end
            end
        end
    end

    return G_mat
end

"""
    compute_farfield(G_mat, I_coeffs, NΩ)

Compute the far-field E∞(r̂_q) = Σ_n I_n g_n(r̂_q) for all grid points.
Returns a (3, NΩ) complex matrix.
"""
function compute_farfield(G_mat::Matrix{ComplexF64}, I_coeffs::Vector{ComplexF64}, NΩ::Int)
    NΩ > 0 ||
        throw(ArgumentError("NΩ must be positive, got $NΩ"))
    expected_rows = Base.checked_mul(3, NΩ)
    size(G_mat, 1) == expected_rows ||
        throw(DimensionMismatch(
            "G_mat has $(size(G_mat, 1)) rows, expected $expected_rows"))
    length(I_coeffs) == size(G_mat, 2) ||
        throw(DimensionMismatch(
            "I_coeffs length $(length(I_coeffs)) != $(size(G_mat, 2))"))
    all(isfinite, G_mat) ||
        throw(ArgumentError(
            "G_mat must contain only finite values"))
    all(isfinite, I_coeffs) ||
        throw(ArgumentError(
            "I_coeffs must contain only finite values"))

    E_flat = _finite_matrix_vector_product(
        G_mat, I_coeffs, "far-field product") # (3*NΩ,)
    E = reshape(E_flat, 3, NΩ)
    return E
end

# ══════════════════════════════════════════════════════════════════════
# Incident-field far-field amplitudes
# ══════════════════════════════════════════════════════════════════════
#
# `incident_farfield(excitation, r_hat, k)` returns the asymptotic
# amplitude `E_inc^∞(r̂)` such that `E_inc(r·r̂) ~ E_inc^∞(r̂)·e^{-ikr}/r`
# as `r → ∞`. Summed with `compute_farfield` (scattered contribution
# from solved currents) this gives the **total** far-field pattern of a
# finite scatterer illuminated by a point-like source — the true
# `r → ∞` limit, not a workaround evaluated at large finite `r`.
#
# `PlaneWave` has no 1/r decay in the incident field and is excluded
# (its "far-field" contribution is a delta function in the incident
# direction — not a radiation pattern).

"""
    incident_farfield(excitation, r_hat, k) -> CVec3

Asymptotic amplitude of the incident electric field radiated by
`excitation` in direction `r̂`, such that
`E_inc(r·r̂) ~ incident_farfield(...)·exp(-ikr)/r` as `r → ∞`.
"""
function incident_farfield(excitation::AbstractExcitation, ::Vec3, ::Real)
    error("incident_farfield is not defined for excitation type $(typeof(excitation)).")
end

@inline function _monopole_farfield_requires_exact(
        mono::MonopoleExcitation, k::Float64)
    return _source_scaling_extreme_value(mono.amplitude) ||
           _source_scaling_extreme_value(mono.height) ||
           _source_scaling_extreme_value(k)
end

@noinline function _monopole_farfield_amplitude_exact(
        mono::MonopoleExcitation, k::Float64,
        cosine::Float64, sine::Float64)
    return setprecision(BigFloat, _MONOPOLE_EXACT_BASE_PRECISION) do
        electrical_height = BigFloat(k) * BigFloat(mono.height)
        cosine_big = BigFloat(cosine)
        sine_big = BigFloat(sine)
        imaginary_unit = Complex{BigFloat}(0, 1)

        pattern = if mono.include_image
            # cos(x*cos(theta)) - cos(x), written as a product so the
            # electrically-small limit is retained before the source
            # amplitude is applied.
            2 * sin(electrical_height * (1 + cosine_big) / 2) *
            sin(electrical_height * (1 - cosine_big) / 2) / sine_big
        else
            first_delta = 1 - cosine_big
            second_delta = 1 + cosine_big
            first_term = iszero(first_delta) ?
                -imaginary_unit * electrical_height :
                -expm1(imaginary_unit * electrical_height * first_delta) /
                 first_delta
            second_term = iszero(second_delta) ?
                imaginary_unit * electrical_height :
                -expm1(-imaginary_unit * electrical_height * second_delta) /
                 second_delta
            (sine_big / 4) *
            exp(imaginary_unit * electrical_height * cosine_big) *
            (first_term + second_term)
        end

        value = ComplexF64(Complex{BigFloat}(mono.amplitude) * pattern)
        isfinite(value) ||
            throw(OverflowError(
                "MonopoleExcitation far-field amplitude is outside the " *
                "representable ComplexF64 range"))
        return value
    end
end

function incident_farfield(mono::MonopoleExcitation, r_hat::Vec3, k::Real)
    # Closed-form far-field `E_θ^∞(r̂) = iηk sinθ/(4π) · θ̂ · J`
    # with J the wire-current phase integral, computed by Simpson over
    # the appropriate axial range for each source model:
    #   include_image=true  : dipole-plus-image on z' ∈ [-h, +h]
    #   include_image=false : physical half-wire on z' ∈ [ 0, +h]
    # Returns zero below the ground plane when image theory is in
    # effect (cosθ < 0), and at the axial null (sinθ = 0).
    η0 = 376.730313668
    rh, kf = _validated_incident_farfield_args(r_hat, k)
    _validate_incident_farfield_source(mono, kf)
    ax = mono.axis
    cosθ = clamp(dot(rh, ax), -1.0, 1.0)

    mono.include_image && cosθ < 0 &&
        return CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)

    sinθ = sqrt(max(0.0, 1.0 - cosθ^2))
    sinθ > 1e-12 || return CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    θ_hat = (cosθ * rh - Vec3(ax)) / sinθ

    h = mono.height
    span_factor = mono.include_image ? 2.0 : 1.0
    N = _monopole_simpson_interval_count(
        h, kf, span_factor, 64, "MonopoleExcitation far field")
    Eθ_far = if _monopole_farfield_requires_exact(mono, kf)
        _monopole_farfield_amplitude_exact(mono, kf, cosθ, sinθ)
    else
        I_0 = -1im * 2π * mono.amplitude / η0
        dz = (span_factor / N) * h
        integ = 0.0 + 0im
        @inbounds for i in 0:N
            unit_position = mono.include_image ?
                            (-1.0 + 2.0 * (i / N)) : (i / N)
            z = h * unit_position
            I_z = I_0 * sin(kf * (h - abs(z)))  # abs(z) reduces to z for z ≥ 0
            w = (i == 0 || i == N) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
            integ += w * I_z * exp(1im * kf * z * cosθ)
        end
        integ *= dz / 3.0
        1im * η0 * kf * sinθ / (4π) * integ
    end
    phase = _source_phase(
        kf, rh, mono.position, 1.0, "MonopoleExcitation far field")
    return _check_finite_cvec3(
        CVec3(Eθ_far * θ_hat) * phase, "MonopoleExcitation far field")
end

function incident_farfield(dipole::DipoleExcitation, r_hat::Vec3, k::Real)
    # Cross-checked against `dipole_incident_field` in this file (keep the
    # 1/R term, multiply by R·exp(+ikR)):
    #   Electric: E∞(r̂) = +k²/(4πε₀) · (r̂×p)×r̂ = +k²/(4πε₀) · perp(p)
    #   Magnetic: E∞(r̂) = +η₀·k²/(4π) · (m × r̂)   (real coeff, dual to electric)
    rh, kf = _validated_incident_farfield_args(r_hat, k)
    _validate_incident_farfield_source(dipole, kf)
    E_far = _dipole_farfield_unphased(dipole, rh, kf)
    phase = _source_phase(
        kf, rh, dipole.position, 1.0, "DipoleExcitation far field")
    return _check_finite_cvec3(
        CVec3(E_far) * phase, "DipoleExcitation far field")
end

function incident_farfield(loop::LoopExcitation, r_hat::Vec3, k::Real)
    # A small circular current loop with current I and radius a radiates
    # in the far field as an equivalent magnetic dipole with moment
    # m = I·π·a²·n̂ placed at the loop centre.
    _, kf = _validated_incident_farfield_args(r_hat, k)
    _validate_incident_farfield_source(loop, kf)
    m = _loop_equivalent_moment(loop)
    dip = DipoleExcitation(loop.center, m, loop.normal, :magnetic, loop.frequency)
    return incident_farfield(dip, r_hat, kf)
end

function incident_farfield(pat::PatternFeedExcitation, r_hat::Vec3, k::Real)
    # `PatternFeedExcitation` is already a tabulated far-field:
    # E_inc(r·r̂) ~ (Fθ(θ,ϕ)·θ̂ + Fϕ(θ,ϕ)·ϕ̂) · exp(-ikR)/R
    # (with R = |r − phase_center|). At the far field, R ≈ r − r̂·pc,
    # so r·E·exp(+ikr) = E_far(r̂) · exp(+ik·r̂·phase_center).
    rh, kf = _validated_incident_farfield_args(r_hat, k)
    _validate_incident_farfield_source(pat, kf)

    θ = acos(clamp(rh[3], -1.0, 1.0))
    ϕ = atan(rh[2], rh[1])
    if ϕ < 0
        ϕ += 2π
    end
    _, eθ, eϕ = _spherical_basis(θ, ϕ)

    Fθ = _pattern_interp(pat, pat.Ftheta, θ, ϕ)
    Fϕ = _pattern_interp(pat, pat.Fphi, θ, ϕ)
    if pat.convention == :exp_minus_iwt
        Fθ = conj(Fθ)
        Fϕ = conj(Fϕ)
    end

    E_far = Fθ * eθ + Fϕ * eϕ
    phase = _source_phase(
        kf, rh, pat.phase_center, 1.0,
        "PatternFeedExcitation far field")
    return _check_finite_cvec3(
        CVec3(E_far) * phase, "PatternFeedExcitation far field")
end

function incident_farfield(multi::MultiExcitation, r_hat::Vec3, k::Real)
    _validated_incident_farfield_args(r_hat, k)
    length(multi.excitations) == length(multi.weights) ||
        error("MultiExcitation has mismatched excitation/weight lengths.")
    isempty(multi.excitations) &&
        throw(ArgumentError("MultiExcitation requires at least one child excitation."))
    E = CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    for (i, (exc, w)) in enumerate(zip(multi.excitations, multi.weights))
        isfinite(w) ||
            throw(ArgumentError(
                "MultiExcitation weight $i must be finite, got $w."))
        E = E + w * incident_farfield(exc, r_hat, k)
    end
    return _check_finite_cvec3(E, "MultiExcitation far field")
end
