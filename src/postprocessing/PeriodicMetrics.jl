# PeriodicMetrics.jl — Scattering metrics for periodic metasurfaces
#
# Computes reflection coefficients, Floquet mode efficiencies,
# and periodic RCS for metasurface unit cells.
#
# For a metasurface illuminated by a plane wave at angle (θ_inc, φ_inc),
# the reflected field is decomposed into Floquet harmonics (m,n) with
# propagation directions determined by grating equations:
#   kx_mn = kx_inc + 2πm/dx
#   ky_mn = ky_inc + 2πn/dy
#   kz_mn = sqrt(k² - kx_mn² - ky_mn²)  (propagating if real)
#
# Reflection coefficient normalization (Fix 1.3):
#   R_mn = -(η₀ k)/(2 κz_mn E₀) × (ê_pol · J̃_mn)
#   where J̃_mn = (1/A) ∫_cell J(r') exp(i κ_t·r') dS'
#   For PEC at normal incidence: R₀₀ = -1 (verified).

export floquet_modes, reflection_coefficients, reflection_coefficient_vectors
export reflected_power_fractions, transmission_coefficients, specular_rcs_objective
export power_balance
export FloquetMode

function _specular_objective_polarization(grid::SphGrid, polarization)
    if polarization isa Symbol
        if polarization in (:x, :theta, :tm)
            return pol_linear_x(grid)
        elseif polarization in (:y, :phi, :te)
            return pol_linear_y(grid)
        else
            throw(ArgumentError(
                "Unsupported polarization=$polarization " *
                "(expected :x/:theta/:tm, :y/:phi/:te, or a (3, NΩ) polarization matrix)."
            ))
        end
    elseif polarization isa AbstractMatrix
        NΩ = length(grid.w)
        size(polarization) == (3, NΩ) ||
            throw(DimensionMismatch(
                "Custom polarization matrix must have size (3, $NΩ); got $(size(polarization))."
            ))
        converted = Matrix{ComplexF64}(polarization)
        all(isfinite, converted) ||
            throw(ArgumentError(
                "Custom polarization matrix must contain only finite values."
            ))
        return converted
    else
        throw(ArgumentError(
            "Unsupported polarization input of type $(typeof(polarization)). " *
            "Pass a supported Symbol or a (3, NΩ) polarization matrix."
        ))
    end
end

function _assert_coplanar_periodic_metrics_mesh(mesh::TriMesh; atol::Float64=1e-12)
    isfinite(atol) && atol >= 0.0 ||
        throw(ArgumentError(
            "periodic-metrics coplanarity tolerance must be finite and " *
            "nonnegative, got $atol"))
    zvals = @view mesh.xyz[3, :]
    zmin = minimum(zvals)
    zmax = maximum(zvals)
    spread = abs(zmax - zmin)
    if !_periodic_coordinate_within(zmax, zmin, atol)
        throw(ArgumentError(
            "reflection_coefficients currently supports coplanar unit-cell meshes only " *
            "(max z spread <= $(atol)). Got z spread=$spread."
        ))
    end
end

"""
    FloquetMode

A single Floquet diffraction order (m, n).
"""
struct FloquetMode
    m::Int                      # x-order
    n::Int                      # y-order
    kx::Float64                 # kx of this mode
    ky::Float64                 # ky of this mode
    kz::ComplexF64              # kz (real = propagating, imaginary = evanescent)
    propagating::Bool           # true if kz is real (mode carries power)
    theta_r::Float64            # reflection angle theta (NaN if evanescent)
    phi_r::Float64              # reflection angle phi (NaN if evanescent)
end

function _validate_periodic_current_coefficients(rwg::RWGData,
                                                 I_coeffs::Vector{<:Number})
    length(I_coeffs) == rwg.nedges ||
        throw(DimensionMismatch(
            "I_coeffs length $(length(I_coeffs)) != rwg.nedges=$(rwg.nedges)"
        ))
    all(isfinite, I_coeffs) ||
        throw(ArgumentError("I_coeffs must contain only finite values"))
    return nothing
end

@inline function _validate_periodic_field_normalization(E0::Float64,
                                                        eta0::Float64)
    isfinite(E0) && !iszero(E0) ||
        throw(ArgumentError("E0 must be finite and nonzero, got $E0"))
    isfinite(eta0) && eta0 > 0.0 ||
        throw(ArgumentError("eta0 must be finite and positive, got $eta0"))
    return nothing
end

function _validate_periodic_polarization(pol::SVector{3,<:Real})
    all(isfinite, pol) ||
        throw(ArgumentError("pol components must be finite, got $pol"))
    any(x -> !iszero(x), pol) ||
        throw(ArgumentError("pol must be nonzero"))
    return nothing
end

function _validate_floquet_mode_orders(modes::Vector{FloquetMode})
    seen = Set{Tuple{Int,Int}}()
    for (i, mode) in enumerate(modes)
        order = (mode.m, mode.n)
        order in seen &&
            throw(ArgumentError(
                "modes contains duplicate Floquet order $order at index $i"
            ))
        push!(seen, order)
    end
    return nothing
end

@inline function _floquet_mode_flux_factor(mode::FloquetMode, k::Float64,
                                           index::Int)
    all(isfinite, (mode.kx, mode.ky, mode.kz)) ||
        throw(ArgumentError(
            "Floquet mode $index ($(mode.m), $(mode.n)) has non-finite wavevector components"
        ))

    kx_norm = mode.kx / k
    ky_norm = mode.ky / k
    kz_real_norm = real(mode.kz) / k
    kz_imag_norm = imag(mode.kz) / k
    all(isfinite, (kx_norm, ky_norm, kz_real_norm, kz_imag_norm)) ||
        throw(ArgumentError(
            "Floquet mode $index ($(mode.m), $(mode.n)) is not representable at k=$k"
        ))

    component_tol = 64eps(Float64)
    if mode.propagating
        kz_real_norm > 0.0 ||
            throw(ArgumentError(
                "propagating Floquet mode $index ($(mode.m), $(mode.n)) must have positive real kz"
            ))
        abs(kz_imag_norm) <= component_tol ||
            throw(ArgumentError(
                "propagating Floquet mode $index ($(mode.m), $(mode.n)) must have real kz"
            ))
        dispersion = kx_norm^2 + ky_norm^2 + kz_real_norm^2
        isfinite(dispersion) && isapprox(dispersion, 1.0; rtol=1e-10, atol=1e-12) ||
            throw(ArgumentError(
                "Floquet mode $index ($(mode.m), $(mode.n)) is inconsistent with k=$k"
            ))
        return kz_real_norm
    end

    abs(kz_real_norm) <= component_tol ||
        throw(ArgumentError(
            "evanescent Floquet mode $index ($(mode.m), $(mode.n)) must have imaginary kz"
        ))
    kz_imag_norm >= -component_tol ||
        throw(ArgumentError(
            "evanescent Floquet mode $index ($(mode.m), $(mode.n)) must use nonnegative imaginary kz"
        ))
    dispersion = kx_norm^2 + ky_norm^2 - kz_imag_norm^2
    isfinite(dispersion) && isapprox(dispersion, 1.0; rtol=1e-10, atol=1e-12) ||
        throw(ArgumentError(
            "Floquet mode $index ($(mode.m), $(mode.n)) is inconsistent with k=$k"
        ))
    return 0.0
end

@noinline function _periodic_scaled_intensity_exact(
    amplitudes,
    flux_factor::Float64,
    label::AbstractString,
)
    accumulator = _IEEEBilinearAccumulator(Float64)
    @inbounds for amplitude in amplitudes
        amplitude_real = Float64(real(amplitude))
        amplitude_imag = Float64(imag(amplitude))
        _ieee_bilinear_add_triple!(
            accumulator, flux_factor,
            amplitude_real, amplitude_real, 1)
        _ieee_bilinear_add_triple!(
            accumulator, flux_factor,
            amplitude_imag, amplitude_imag, 1)
    end
    return _ieee_bilinear_finish(accumulator, label)::Float64
end

@inline function _periodic_scaled_intensity(
    amplitudes,
    flux_factor::Float64,
    label::AbstractString,
)::Float64
    needs_fallback = _ieee_dense_extreme_factor(
        flux_factor, Float64) ||
        _ieee_bilinear_values_require_fallback(amplitudes, Float64)
    needs_fallback &&
        return _periodic_scaled_intensity_exact(
            amplitudes, flux_factor, label)

    intensity = 0.0
    @inbounds for amplitude in amplitudes
        intensity += abs2(amplitude)
    end
    power_factor = intensity * flux_factor
    isfinite(power_factor) ||
        return _periodic_scaled_intensity_exact(
            amplitudes, flux_factor, label)
    return power_factor
end

@noinline function _periodic_coefficient_power_sum_exact(
    modes::Vector{FloquetMode},
    coefficients::Vector{ComplexF64},
    k::Float64,
    label::AbstractString,
)
    accumulator = _IEEEBilinearAccumulator(Float64)
    @inbounds for (index, mode) in enumerate(modes)
        mode.propagating || continue
        flux_factor = _floquet_mode_flux_factor(mode, k, index)
        coefficient = coefficients[index]
        coefficient_real = real(coefficient)
        coefficient_imag = imag(coefficient)
        _ieee_bilinear_add_triple!(
            accumulator, flux_factor,
            coefficient_real, coefficient_real, 1)
        _ieee_bilinear_add_triple!(
            accumulator, flux_factor,
            coefficient_imag, coefficient_imag, 1)
    end
    return _ieee_bilinear_finish(accumulator, label)::Float64
end

function _periodic_coefficient_power_sum(
    modes::Vector{FloquetMode},
    coefficients::Vector{ComplexF64},
    k::Float64,
    label::AbstractString,
)
    needs_fallback = _ieee_bilinear_values_require_fallback(
        coefficients, Float64)
    power_sum = 0.0
    @inbounds for (index, mode) in enumerate(modes)
        mode.propagating || continue
        flux_factor = _floquet_mode_flux_factor(mode, k, index)
        needs_fallback |= _ieee_dense_extreme_factor(
            flux_factor, Float64)
        needs_fallback && continue
        power_sum += abs2(coefficients[index]) * flux_factor
        isfinite(power_sum) ||
            return _periodic_coefficient_power_sum_exact(
                modes, coefficients, k, label)
    end
    needs_fallback &&
        return _periodic_coefficient_power_sum_exact(
            modes, coefficients, k, label)
    return power_sum
end

@noinline function _periodic_incident_power_base_exact(
    E0::Float64,
    area::Float64,
    eta0::Float64,
)
    accumulator = _IEEEBilinearAccumulator(Float64)
    _ieee_bilinear_add_triple!(accumulator, area, E0, E0, 1)
    return _ieee_bilinear_finish(
        accumulator, "incident-power normalization", 0.5, eta0)
end

@inline function _periodic_incident_power_base(
    E0::Float64,
    area::Float64,
    eta0::Float64,
)
    needs_fallback = _ieee_dense_extreme_factor(E0, Float64) ||
                     _ieee_dense_extreme_factor(area, Float64) ||
                     _ieee_dense_extreme_factor(eta0, Float64)
    needs_fallback &&
        return _periodic_incident_power_base_exact(E0, area, eta0)
    base = abs2(E0) * area / (2 * eta0)
    (!isfinite(base) || iszero(base)) &&
        return _periodic_incident_power_base_exact(E0, area, eta0)
    return base
end

@noinline function _periodic_finite_product_exact(
    first::Float64,
    second::Float64,
    label::AbstractString,
)
    return setprecision(BigFloat, 256) do
        converted = Float64(BigFloat(first) * BigFloat(second))
        isfinite(converted) ||
            throw(OverflowError("$label is outside the Float64 range"))
        return converted
    end
end

@inline function _periodic_finite_product(
    first::Float64,
    second::Float64,
    label::AbstractString,
)
    product = first * second
    needs_fallback = _ieee_dense_extreme_factor(first, Float64) ||
                     _ieee_dense_extreme_factor(second, Float64) ||
                     !isfinite(product) ||
                     (!iszero(first) && !iszero(second) && iszero(product))
    needs_fallback &&
        return _periodic_finite_product_exact(first, second, label)
    return product
end

@inline function _periodic_reflection_requires_fallback(
    current_coefficient,
    eta0::Float64,
    k::Float64,
    kz::Float64,
    E0::Float64,
)
    return _ieee_bilinear_values_require_fallback(
               current_coefficient, Float64) ||
           _ieee_dense_extreme_factor(eta0, Float64) ||
           _ieee_dense_extreme_factor(k, Float64) ||
           _ieee_dense_extreme_factor(kz, Float64) ||
           _ieee_dense_extreme_factor(E0, Float64)
end

@noinline function _periodic_scalar_reflection_exact(
    polarization,
    current_coefficient,
    eta0::Float64,
    k::Float64,
    kz::Float64,
    E0::Float64,
    mode::FloquetMode,
)
    return setprecision(BigFloat, _PROJECTED_POWER_FALLBACK_PRECISION) do
        projection_real = zero(BigFloat)
        projection_imag = zero(BigFloat)
        @inbounds for component in eachindex(polarization, current_coefficient)
            polarization_component = BigFloat(polarization[component])
            current_value = current_coefficient[component]
            projection_real +=
                polarization_component * BigFloat(real(current_value))
            projection_imag +=
                polarization_component * BigFloat(imag(current_value))
        end
        scale = -BigFloat(eta0) * BigFloat(k) /
                (2 * BigFloat(kz) * BigFloat(E0))
        coefficient = ComplexF64(Complex{BigFloat}(
            scale * projection_real, scale * projection_imag))
        isfinite(coefficient) ||
            throw(OverflowError(
                "reflection coefficient is outside the ComplexF64 range " *
                "for order ($(mode.m), $(mode.n))"))
        return coefficient
    end
end

@noinline function _periodic_vector_reflection_exact(
    khat,
    current_coefficient,
    eta0::Float64,
    k::Float64,
    kz::Float64,
    E0::Float64,
    mode::FloquetMode,
)
    return setprecision(BigFloat, _PROJECTED_POWER_FALLBACK_PRECISION) do
        longitudinal_real = zero(BigFloat)
        longitudinal_imag = zero(BigFloat)
        @inbounds for component in eachindex(khat, current_coefficient)
            direction_component = BigFloat(khat[component])
            current_value = current_coefficient[component]
            longitudinal_real +=
                direction_component * BigFloat(real(current_value))
            longitudinal_imag +=
                direction_component * BigFloat(imag(current_value))
        end
        scale = -BigFloat(eta0) * BigFloat(k) /
                (2 * BigFloat(kz) * BigFloat(E0))
        coefficient = SVector{3,ComplexF64}(ntuple(component -> begin
            direction_component = BigFloat(khat[component])
            current_value = current_coefficient[component]
            transverse_real = BigFloat(real(current_value)) -
                              direction_component * longitudinal_real
            transverse_imag = BigFloat(imag(current_value)) -
                              direction_component * longitudinal_imag
            ComplexF64(Complex{BigFloat}(
                scale * transverse_real, scale * transverse_imag))
        end, 3))
        all(isfinite, coefficient) ||
            throw(OverflowError(
                "vector reflection coefficient is outside the ComplexF64 " *
                "range for order ($(mode.m), $(mode.n))"))
        return coefficient
    end
end

function _validate_floquet_modes(modes::Vector{FloquetMode}, k::Float64)
    _validate_floquet_mode_orders(modes)
    for (i, mode) in enumerate(modes)
        _floquet_mode_flux_factor(mode, k, i)
    end
    return nothing
end

function _incident_floquet_mode_index(modes::Vector{FloquetMode},
                                      incident_order::Tuple{Int,Int})
    inc_idx = findfirst(
        mode -> mode.m == incident_order[1] && mode.n == incident_order[2],
        modes,
    )
    inc_idx === nothing &&
        throw(ArgumentError(
            "incident Floquet order $incident_order is not present in modes"
        ))
    modes[inc_idx].propagating ||
        throw(ArgumentError(
            "incident Floquet order $incident_order must be propagating"
        ))
    return inc_idx
end

function _mode_transverse_projection(pol::SVector{3,<:Real}, mode::FloquetMode, k::Real)
    khat = SVector(mode.kx / k, mode.ky / k, real(mode.kz) / k)
    _validate_periodic_polarization(pol)
    # Only the polarization direction matters.  Scaling before projection keeps
    # valid very large and subnormal inputs away from overflow/underflow.
    pol_scale = max(abs(pol[1]), abs(pol[2]), abs(pol[3]))
    pol_scaled = SVector(
        Float64(pol[1] / pol_scale),
        Float64(pol[2] / pol_scale),
        Float64(pol[3] / pol_scale),
    )
    pol_mode_raw = pol_scaled - dot(pol_scaled, khat) * khat
    pol_mode_norm = norm(pol_mode_raw)
    isfinite(pol_mode_norm) ||
        throw(OverflowError(
            "periodic polarization projection produced a non-finite norm"))
    return pol_mode_norm < 1e-12 ? nothing : pol_mode_raw / pol_mode_norm
end

const _FLOQUET_MODE_FALLBACK_PRECISION = 256
const _FLOQUET_CURRENT_FALLBACK_PRECISION =
    _IEEE_BILINEAR_FALLBACK_PRECISION
const _DEFAULT_MAX_PERIODIC_REFLECTION_WORK_BYTES = 512 * 1024 * 1024
const _DEFAULT_MAX_PERIODIC_FOURIER_TERMS = 200_000_000
const _DEFAULT_MAX_PERIODIC_EXACT_FOURIER_TERMS = 2_000_000

function _periodic_reflection_base_bytes(mode_count::Int)
    # Modes, Fourier coefficients, and the largest public reflection output
    # coexist at a wrapper boundary.  Charging the vector-valued output also
    # safely covers the scalar and grounded wrappers.
    total = BigInt(sizeof(FloquetMode) +
                   2 * sizeof(SVector{3,ComplexF64})) * mode_count
    total <= typemax(Int) ||
        throw(ArgumentError(
            "periodic reflection raw-workspace estimate overflows Int"))
    return Int(total)
end

function _preflight_periodic_reflection_base(
        N_orders::Int, max_work_bytes::Integer)
    order = _periodic_truncation_order("N_orders", N_orders)
    mode_count = _periodic_term_count(order)
    work_limit = _validated_resource_limit("max_work_bytes", max_work_bytes)
    _enforce_payload_limit(
        _periodic_reflection_base_bytes(mode_count), work_limit,
        "periodic reflection modes and outputs", "max_work_bytes")
    return mode_count, work_limit
end

function _periodic_reflection_work_bytes(
        Nt::Int, N::Int, Nq::Int, mode_count::Int)
    total = BigInt(_periodic_reflection_base_bytes(mode_count))
    # Per-triangle quadrature points and cached currents.  The pointer payload
    # of the three outer vectors is included alongside their element payloads.
    total += BigInt(3 * sizeof(Ptr{Cvoid}) +
                    2 * sizeof(Float64)) * Nt
    total += BigInt(sizeof(Vec3) + sizeof(SVector{3,ComplexF64})) * Nt * Nq
    # Triangle-to-basis indices, exact-path basis integrals, and the tiny
    # quadrature-rule arrays.  Area ratios are included in the 2*Float64 term
    # above, even though they exist only on an exceptional path.
    total += BigInt(2 * sizeof(Int) +
                    sizeof(SVector{3,ComplexF64})) * N
    total += BigInt(sizeof(SVector{2,Float64}) + sizeof(Float64)) * Nq
    total <= typemax(Int) ||
        throw(ArgumentError(
            "periodic reflection raw-workspace estimate overflows Int"))
    return Int(total)
end

function _periodic_fourier_term_count(
        Nt::Int, N::Int, Nq::Int, propagating_count::Int;
        exact::Bool=false)
    count = if exact
        BigInt(propagating_count) *
        (BigInt(Nq) * (BigInt(Nt) + 2BigInt(N)) + 3BigInt(N))
    else
        BigInt(Nq) * (BigInt(propagating_count) * Nt + 2BigInt(N))
    end
    count <= typemax(Int) ||
        throw(ArgumentError(
            "periodic Fourier-term estimate overflows Int"))
    return Int(count)
end

function _enforce_periodic_fourier_term_limit(
        term_count::Int, max_terms::Integer, limit_name::AbstractString)
    limit = _validated_resource_limit(limit_name, max_terms)
    term_count <= limit ||
        throw(ArgumentError(
            "periodic current Fourier integration requires $term_count " *
            "terms, exceeding $limit_name=$limit"))
    return term_count
end

@inline function _periodic_phase_requires_fallback(
    mode::FloquetMode,
    point,
)
    return _source_phase_requires_fallback(
        1.0,
        Vec3(mode.kx, mode.ky, 0.0),
        Vec3(point[1], point[2], 0.0),
    )
end


@noinline function _periodic_phase_exact(
    mode::FloquetMode,
    point,
)
    return _source_phase_exact(
        1.0,
        Vec3(mode.kx, mode.ky, 0.0),
        Vec3(point[1], point[2], 0.0),
        1.0,
        "Floquet phase for order ($(mode.m), $(mode.n))",
    )
end


@inline function _periodic_phase(mode::FloquetMode, point)
    _periodic_phase_requires_fallback(mode, point) &&
        return _periodic_phase_exact(mode, point)
    argument = mode.kx * point[1] + mode.ky * point[2]
    isfinite(argument) || return _periodic_phase_exact(mode, point)
    return exp(im * argument)
end


function _periodic_fourier_requires_fallback(
    I_coeffs,
    A_cell::Float64,
    areas,
    modes,
    quad_pts,
    weights,
)
    _ieee_bilinear_values_require_fallback(I_coeffs, Float64) && return true
    _ieee_dense_extreme_factor(A_cell, Float64) && return true
    _ieee_bilinear_values_require_fallback(areas, Float64) && return true
    _ieee_bilinear_values_require_fallback(weights, Float64) && return true
    maximum_mode_x_exponent = typemin(Int)
    maximum_mode_y_exponent = typemin(Int)
    @inbounds for mode in modes
        mode.propagating || continue
        (_ieee_dense_extreme_factor(mode.kx, Float64) ||
         _ieee_dense_extreme_factor(mode.ky, Float64)) && return true
        !iszero(mode.kx) &&
            (maximum_mode_x_exponent = max(
                maximum_mode_x_exponent, exponent(abs(mode.kx))))
        !iszero(mode.ky) &&
            (maximum_mode_y_exponent = max(
                maximum_mode_y_exponent, exponent(abs(mode.ky))))
    end
    maximum_point_x_exponent = typemin(Int)
    maximum_point_y_exponent = typemin(Int)
    @inbounds for triangle_points in quad_pts
        for point in triangle_points
            _ieee_bilinear_values_require_fallback(point, Float64) &&
                return true
            !iszero(point[1]) &&
                (maximum_point_x_exponent = max(
                    maximum_point_x_exponent,
                    exponent(abs(point[1]))))
            !iszero(point[2]) &&
                (maximum_point_y_exponent = max(
                    maximum_point_y_exponent,
                    exponent(abs(point[2]))))
        end
    end
    # A finite phase term can still carry more absolute rounding error than
    # the unit-circle reduction can tolerate.  Use maxima by axis to classify
    # the whole Fourier job in O(number of modes + quadrature points), rather
    # than adding a second O(modes * points) preflight.
    maximum_mode_x_exponent != typemin(Int) &&
        maximum_point_x_exponent != typemin(Int) &&
        maximum_mode_x_exponent + maximum_point_x_exponent + 2 >
            _SOURCE_PHASE_FAST_TERM_EXPONENT && return true
    maximum_mode_y_exponent != typemin(Int) &&
        maximum_point_y_exponent != typemin(Int) &&
        maximum_mode_y_exponent + maximum_point_y_exponent + 2 >
            _SOURCE_PHASE_FAST_TERM_EXPONENT && return true
    return false
end


@inline function _periodic_accumulate_complex_product!(
    real_accumulator::_IEEEBilinearAccumulator{Float64},
    imag_accumulator::_IEEEBilinearAccumulator{Float64},
    first::Number,
    second::Number,
)
    first_real = Float64(real(first))
    first_imag = Float64(imag(first))
    second_real = Float64(real(second))
    second_imag = Float64(imag(second))
    unity = 1.0
    _ieee_bilinear_add_triple!(
        real_accumulator, first_real, second_real, unity, 1)
    _ieee_bilinear_add_triple!(
        real_accumulator, first_imag, second_imag, unity, -1)
    _ieee_bilinear_add_triple!(
        imag_accumulator, first_real, second_imag, unity, 1)
    _ieee_bilinear_add_triple!(
        imag_accumulator, first_imag, second_real, unity, 1)
    return nothing
end


function _periodic_finish_current_coefficient(
    I_coeffs,
    basis_integrals,
    area_scale::Float64,
    A_cell::Float64,
    mode::FloquetMode,
    component::Int,
)
    real_accumulator = _IEEEBilinearAccumulator(Float64)
    imag_accumulator = _IEEEBilinearAccumulator(Float64)
    @inbounds for basis_index in eachindex(I_coeffs, basis_integrals)
        _periodic_accumulate_complex_product!(
            real_accumulator, imag_accumulator,
            I_coeffs[basis_index],
            basis_integrals[basis_index][component])
    end
    label = "Floquet current coefficient component $component for order " *
            "($(mode.m), $(mode.n))"
    real_value = _ieee_bilinear_finish(
        real_accumulator, label, area_scale, A_cell)
    imag_value = _ieee_bilinear_finish(
        imag_accumulator, label, area_scale, A_cell)
    return ComplexF64(real_value, imag_value)
end


@noinline function _floquet_current_fourier_coefficients_bigfloat(
    rwg::RWGData,
    I_coeffs,
    modes,
    A_cell::Float64,
    quad_pts,
    areas,
    tri_to_basis,
    weights,
)
    return setprecision(BigFloat, _FLOQUET_CURRENT_FALLBACK_PRECISION) do
        zero_vec = SVector{3,ComplexF64}(0.0, 0.0, 0.0)
        coefficients = fill(zero_vec, length(modes))
        @inbounds for (mode_index, mode) in enumerate(modes)
            mode.propagating || continue
            integral = zeros(Complex{BigFloat}, 3)
            for triangle in eachindex(quad_pts, areas, tri_to_basis)
                area = BigFloat(areas[triangle])
                for quadrature_index in eachindex(weights)
                    point = quad_pts[triangle][quadrature_index]
                    argument = BigFloat(mode.kx) * BigFloat(point[1]) +
                               BigFloat(mode.ky) * BigFloat(point[2])
                    phase = exp(Complex{BigFloat}(0, argument))
                    current = zeros(Complex{BigFloat}, 3)
                    for basis_index in tri_to_basis[triangle]
                        basis_value = eval_rwg(
                            rwg, basis_index, point, triangle)
                        coefficient = Complex{BigFloat}(
                            I_coeffs[basis_index])
                        for component in 1:3
                            current[component] += coefficient *
                                BigFloat(basis_value[component])
                        end
                    end
                    weight = BigFloat(weights[quadrature_index]) * 2area
                    for component in 1:3
                        integral[component] +=
                            current[component] * phase * weight
                    end
                end
            end
            converted = SVector{3,ComplexF64}(ntuple(component ->
                ComplexF64(integral[component] / BigFloat(A_cell)), 3))
            all(isfinite, converted) ||
                throw(OverflowError(
                    "Floquet current coefficient is outside the ComplexF64 " *
                    "range for order ($(mode.m), $(mode.n))"))
            coefficients[mode_index] = converted
        end
        return coefficients
    end
end


function _floquet_current_fourier_coefficients_exact(
    rwg::RWGData,
    I_coeffs,
    modes,
    A_cell::Float64,
    quad_pts,
    areas,
    tri_to_basis,
    weights,
)
    _ieee_bilinear_supported_eltype(eltype(I_coeffs), Float64) ||
        return _floquet_current_fourier_coefficients_bigfloat(
            rwg, I_coeffs, modes, A_cell, quad_pts,
            areas, tri_to_basis, weights)

    maximum_area = maximum(areas)
    area_scale = ldexp(1.0, exponent(maximum_area))
    area_ratios = areas ./ area_scale
    if any(index -> !iszero(areas[index]) && iszero(area_ratios[index]),
           eachindex(areas))
        return _floquet_current_fourier_coefficients_bigfloat(
            rwg, I_coeffs, modes, A_cell, quad_pts,
            areas, tri_to_basis, weights)
    end

    zero_vec = SVector{3,ComplexF64}(0.0, 0.0, 0.0)
    coefficients = fill(zero_vec, length(modes))
    basis_integrals = fill(zero_vec, rwg.nedges)
    @inbounds for (mode_index, mode) in enumerate(modes)
        mode.propagating || continue
        fill!(basis_integrals, zero_vec)
        for triangle in eachindex(quad_pts, areas, tri_to_basis)
            area_weight = 2 * area_ratios[triangle]
            for quadrature_index in eachindex(weights)
                point = quad_pts[triangle][quadrature_index]
                phase = _periodic_phase(mode, point)
                weight = weights[quadrature_index] * area_weight
                for basis_index in tri_to_basis[triangle]
                    basis_value = eval_rwg(
                        rwg, basis_index, point, triangle)
                    contribution = basis_value * phase * weight
                    all(isfinite, contribution) ||
                        return _floquet_current_fourier_coefficients_bigfloat(
                            rwg, I_coeffs, modes, A_cell, quad_pts,
                            areas, tri_to_basis, weights)
                    basis_integrals[basis_index] += contribution
                    all(isfinite, basis_integrals[basis_index]) ||
                        return _floquet_current_fourier_coefficients_bigfloat(
                            rwg, I_coeffs, modes, A_cell, quad_pts,
                            areas, tri_to_basis, weights)
                end
            end
        end
        coefficients[mode_index] = SVector{3,ComplexF64}(ntuple(
            component -> _periodic_finish_current_coefficient(
                I_coeffs, basis_integrals,
                area_scale, A_cell, mode, component),
            3,
        ))
    end
    return coefficients
end

function _floquet_current_fourier_coefficients(mesh::TriMesh, rwg::RWGData,
                                               I_coeffs::Vector{<:Number},
                                               k::Real, lattice::PeriodicLattice;
                                               quad_order::Int=3,
                                               N_orders::Int=3,
                                               max_work_bytes::Integer=
                                                   _DEFAULT_MAX_PERIODIC_REFLECTION_WORK_BYTES,
                                               max_fourier_terms::Integer=
                                                   _DEFAULT_MAX_PERIODIC_FOURIER_TERMS,
                                                   max_exact_fourier_terms::Integer=
                                                   _DEFAULT_MAX_PERIODIC_EXACT_FOURIER_TERMS)
    _validate_mesh_rwg_pair(mesh, rwg)
    kw = _validated_lattice_wavenumber(k, lattice)
    mode_count, work_limit = _preflight_periodic_reflection_base(
        N_orders, max_work_bytes)
    _validated_resource_limit("max_fourier_terms", max_fourier_terms)
    _validated_resource_limit(
        "max_exact_fourier_terms", max_exact_fourier_terms)
    _validate_periodic_current_coefficients(rwg, I_coeffs)
    _assert_coplanar_periodic_metrics_mesh(mesh)
    _assert_boundary_touching_periodic_mesh_requires_bloch(
        mesh, lattice, rwg; max_cache_bytes=work_limit)

    modes = floquet_modes(kw, lattice; N_orders=N_orders)
    A_cell = lattice.dx * lattice.dy
    isfinite(A_cell) && A_cell > 0.0 ||
        throw(ArgumentError(
            "periodic cell area must be finite and positive, got $A_cell"
        ))

    xi, wq = tri_quad_rule(quad_order)
    Nq = length(wq)
    Nt = ntriangles(mesh)
    N = rwg.nedges
    _enforce_payload_limit(
        _periodic_reflection_work_bytes(Nt, N, Nq, mode_count), work_limit,
        "periodic reflection output and workspace", "max_work_bytes")
    propagating_count = count(mode -> mode.propagating, modes)
    ordinary_terms = _periodic_fourier_term_count(
        Nt, N, Nq, propagating_count)
    _enforce_periodic_fourier_term_limit(
        ordinary_terms, max_fourier_terms, "max_fourier_terms")

    quad_pts = [tri_quad_points(mesh, t, xi) for t in 1:Nt]
    areas = [triangle_area(mesh, t) for t in 1:Nt]

    tri_to_basis = [Int[] for _ in 1:Nt]
    for n in 1:N
        push!(tri_to_basis[rwg.tplus[n]], n)
        push!(tri_to_basis[rwg.tminus[n]], n)
    end

    zero_vec = SVector{3,ComplexF64}(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    if _periodic_fourier_requires_fallback(
            I_coeffs, A_cell, areas, modes, quad_pts, wq)
        exact_terms = _periodic_fourier_term_count(
            Nt, N, Nq, propagating_count; exact=true)
        _enforce_periodic_fourier_term_limit(
            exact_terms, max_exact_fourier_terms,
            "max_exact_fourier_terms")
        return modes, _floquet_current_fourier_coefficients_exact(
            rwg, I_coeffs, modes, A_cell, quad_pts,
            areas, tri_to_basis, wq)
    end

    J_tildes = fill(zero_vec, length(modes))

    # The surface current J(r_q) at each quadrature point is independent of the
    # Floquet mode, so precompute it once instead of recomputing inside every mode.
    J_at = [Vector{SVector{3,ComplexF64}}(undef, Nq) for _ in 1:Nt]
    for t in 1:Nt
        for q in 1:Nq
            rq = quad_pts[t][q]
            J_rq = zero_vec
            for n_idx in tri_to_basis[t]
                J_rq += I_coeffs[n_idx] * eval_rwg(rwg, n_idx, rq, t)
            end
            all(isfinite, J_rq) ||
                throw(OverflowError(
                    "surface-current evaluation became non-finite on triangle $t, quadrature point $q"
                ))
            J_at[t][q] = J_rq
        end
    end

    for (mi, mode) in enumerate(modes)
        mode.propagating || continue

        integral = zero_vec
        for t in 1:Nt
            At = areas[t]
            for q in 1:Nq
                rq = quad_pts[t][q]
                phase = _periodic_phase(mode, rq)
                integral += J_at[t][q] * phase * wq[q] * (2 * At)
                all(isfinite, integral) ||
                    throw(OverflowError(
                        "Floquet current integral became non-finite for order ($(mode.m), $(mode.n))"
                    ))
            end
        end

        J_tildes[mi] = integral / A_cell
        all(isfinite, J_tildes[mi]) ||
            throw(OverflowError(
                "Floquet current coefficient became non-finite for order ($(mode.m), $(mode.n))"
            ))
    end

    return modes, J_tildes
end

@noinline function _floquet_transverse_component_exact(
    bloch_component::Float64,
    order::Int,
    period::Float64,
    axis::AbstractString,
)
    return setprecision(BigFloat, _FLOQUET_MODE_FALLBACK_PRECISION) do
        # Preserve the Float64 value of 2π used by the ordinary path while
        # avoiding an overflowing reciprocal-lattice intermediate.
        component = BigFloat(bloch_component) +
                    BigFloat(2π) * BigFloat(order) / BigFloat(period)
        converted = Float64(component)
        isfinite(converted) ||
            throw(OverflowError(
                "Floquet $axis wavevector is outside the Float64 range " *
                "for order $order and period $period"))
        return converted
    end
end


@inline function _floquet_transverse_component(
    bloch_component::Float64,
    order::Int,
    period::Float64,
    axis::AbstractString,
)
    iszero(order) && return bloch_component
    component = bloch_component + (2π * order) / period
    isfinite(component) && return component
    return _floquet_transverse_component_exact(
        bloch_component, order, period, axis)
end


@inline function _floquet_longitudinal_component(
    k::Float64,
    kx::Float64,
    ky::Float64,
)
    magnitude, propagating = _periodic_longitudinal_magnitude(
        k, kx, ky, "Floquet longitudinal wavevector")
    kz = propagating ?
         ComplexF64(magnitude, 0.0) :
         ComplexF64(0.0, magnitude)
    return kz, propagating
end

"""
    floquet_modes(k, lattice; N_orders=3)

Enumerate all Floquet modes (m, n) for the given lattice and classify
them as propagating or evanescent.

`k` must match `lattice.k`, and `N_orders` must be between zero and 499,
inclusive. This bounds the two-dimensional enumeration to at most one million
lattice terms.

Returns a vector of FloquetMode structs.
"""
function floquet_modes(k::Real, lattice::PeriodicLattice; N_orders::Int=3)
    kw = _validated_lattice_wavenumber(k, lattice)
    order = _periodic_truncation_order("N_orders", N_orders)
    modes = FloquetMode[]
    sizehint!(modes, _periodic_term_count(order))

    for m in -order:order
        for n in -order:order
            kx_mn = _floquet_transverse_component(
                lattice.kx_bloch, m, lattice.dx, "x")
            ky_mn = _floquet_transverse_component(
                lattice.ky_bloch, n, lattice.dy, "y")
            kz, propagating = _floquet_longitudinal_component(
                kw, kx_mn, ky_mn)

            if propagating
                theta_r = acos(clamp(real(kz) / kw, 0.0, 1.0))
                phi_r = atan(ky_mn, kx_mn)
                push!(modes, FloquetMode(m, n, kx_mn, ky_mn, kz, true, theta_r, phi_r))
            else
                push!(modes, FloquetMode(m, n, kx_mn, ky_mn, kz, false, NaN, NaN))
            end
        end
    end

    return modes
end

"""
    reflection_coefficients(mesh, rwg, I_coeffs, k, lattice; kwargs...)

Compute properly normalized complex reflection coefficients for each
propagating Floquet mode.

The reflection coefficient for mode (m,n) is:
  R_mn = -(η₀ k)/(2 κz_mn E₀) × (ê_pol · J̃_mn)
where J̃_mn = (1/A) ∫_cell J(r') exp(i κ_t·r') dS' is the current
Fourier coefficient, ê_pol is the incident polarization, and E₀ is
the incident field amplitude.

Sanity check: for a PEC plate at normal incidence, R₀₀ = -1.

`max_work_bytes` bounds retained raw array payload. `max_fourier_terms` bounds
ordinary current-integration work, and `max_exact_fourier_terms` separately
bounds an exceptional exact retry.

Returns (modes, R_coeffs) where R_coeffs[i] is the complex reflection
coefficient for modes[i].
"""
function reflection_coefficients(mesh::TriMesh, rwg::RWGData,
                                 I_coeffs::Vector{<:Number},
                                 k::Real, lattice::PeriodicLattice;
                                 quad_order::Int=3, N_orders::Int=3,
                                 E0::Float64=1.0,
                                 pol::SVector{3,Float64}=SVector(1.0, 0.0, 0.0),
                                 eta0::Float64=376.730313668,
                                 max_work_bytes::Integer=
                                     _DEFAULT_MAX_PERIODIC_REFLECTION_WORK_BYTES,
                                 max_fourier_terms::Integer=
                                     _DEFAULT_MAX_PERIODIC_FOURIER_TERMS,
                                 max_exact_fourier_terms::Integer=
                                     _DEFAULT_MAX_PERIODIC_EXACT_FOURIER_TERMS)
    kw = _validated_lattice_wavenumber(k, lattice)
    _validate_periodic_field_normalization(E0, eta0)
    _validate_periodic_polarization(pol)
    _preflight_periodic_reflection_base(N_orders, max_work_bytes)
    modes, J_tildes = _floquet_current_fourier_coefficients(
        mesh, rwg, I_coeffs, kw, lattice;
        quad_order=quad_order,
        N_orders=N_orders,
        max_work_bytes=max_work_bytes,
        max_fourier_terms=max_fourier_terms,
        max_exact_fourier_terms=max_exact_fourier_terms,
    )

    R_coeffs = zeros(ComplexF64, length(modes))

    for (mi, mode) in enumerate(modes)
        if !mode.propagating
            continue
        end

        # Use a mode-transverse co-polar vector obtained by projecting the
        # incident polarization onto the mode's transverse plane.
        #
        # This avoids overestimating mode amplitudes when the global incident
        # polarization has a component parallel to the reflected mode direction.
        pol_mode = _mode_transverse_projection(pol, mode, kw)
        if isnothing(pol_mode)
            continue
        end

        # Co-polar reflection coefficient:
        #   R_mn = -(η₀ k)/(2 κz_mn E₀) × (ê_mode · J̃_mn)
        # where ê_mode is transverse to this mode's propagation direction.
        kz_mn = real(mode.kz)
        needs_fallback = _periodic_reflection_requires_fallback(
            J_tildes[mi], eta0, kw, kz_mn, E0)
        if needs_fallback
            R_coeffs[mi] = _periodic_scalar_reflection_exact(
                pol_mode, J_tildes[mi], eta0, kw, kz_mn, E0, mode)
        else
            R_coeffs[mi] = -(eta0 * kw) / (2 * kz_mn * E0) *
                           dot(pol_mode, J_tildes[mi])
            if !isfinite(R_coeffs[mi])
                R_coeffs[mi] = _periodic_scalar_reflection_exact(
                    pol_mode, J_tildes[mi],
                    eta0, kw, kz_mn, E0, mode)
            end
        end
    end

    return modes, R_coeffs
end

"""
    reflection_coefficient_vectors(mesh, rwg, I_coeffs, k, lattice; kwargs...)

Compute the full mode-transverse reflected electric-field amplitude vector for
each propagating Floquet order. Unlike `reflection_coefficients`, which reports
one scalar co-polar projection per order, this vector form retains both
orthogonal transverse polarizations and is therefore the correct quantity for
total reflected-power budgets.

Returns `(modes, R_vecs)`, where `R_vecs[i]` is a three-component complex vector
normalized by the incident field amplitude.

`max_work_bytes` bounds retained raw array payload. `max_fourier_terms` bounds
ordinary current-integration work, and `max_exact_fourier_terms` separately
bounds an exceptional exact retry.
"""
function reflection_coefficient_vectors(mesh::TriMesh, rwg::RWGData,
                                        I_coeffs::Vector{<:Number},
                                        k::Real, lattice::PeriodicLattice;
                                        quad_order::Int=3, N_orders::Int=3,
                                        E0::Float64=1.0,
                                        eta0::Float64=376.730313668,
                                        max_work_bytes::Integer=
                                            _DEFAULT_MAX_PERIODIC_REFLECTION_WORK_BYTES,
                                        max_fourier_terms::Integer=
                                            _DEFAULT_MAX_PERIODIC_FOURIER_TERMS,
                                        max_exact_fourier_terms::Integer=
                                            _DEFAULT_MAX_PERIODIC_EXACT_FOURIER_TERMS)
    kw = _validated_lattice_wavenumber(k, lattice)
    _validate_periodic_field_normalization(E0, eta0)
    _preflight_periodic_reflection_base(N_orders, max_work_bytes)
    modes, J_tildes = _floquet_current_fourier_coefficients(
        mesh, rwg, I_coeffs, kw, lattice;
        quad_order=quad_order,
        N_orders=N_orders,
        max_work_bytes=max_work_bytes,
        max_fourier_terms=max_fourier_terms,
        max_exact_fourier_terms=max_exact_fourier_terms,
    )

    zero_vec = SVector{3,ComplexF64}(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    R_vecs = fill(zero_vec, length(modes))

    for (mi, mode) in enumerate(modes)
        if !mode.propagating
            continue
        end

        kz_mn = real(mode.kz)
        khat = SVector(mode.kx / kw, mode.ky / kw, kz_mn / kw)
        needs_fallback = _periodic_reflection_requires_fallback(
            J_tildes[mi], eta0, kw, kz_mn, E0)
        if needs_fallback
            R_vecs[mi] = _periodic_vector_reflection_exact(
                khat, J_tildes[mi], eta0, kw, kz_mn, E0, mode)
        else
            J_transverse = J_tildes[mi] -
                           khat * dot(khat, J_tildes[mi])
            R_vecs[mi] = -(eta0 * kw) /
                         (2 * kz_mn * E0) * J_transverse
            if !all(isfinite, R_vecs[mi])
                R_vecs[mi] = _periodic_vector_reflection_exact(
                    khat, J_tildes[mi], eta0, kw, kz_mn, E0, mode)
            end
        end
    end

    return modes, R_vecs
end

"""
    reflected_power_fractions(modes, R_vecs, k)

Return the reflected power fraction carried by each Floquet order from full
vector reflection amplitudes. The total reflected fraction is `sum(p)`.
"""
function reflected_power_fractions(modes::Vector{FloquetMode},
                                   R_vecs::Vector{SVector{3,ComplexF64}},
                                   k::Real)
    length(modes) == length(R_vecs) ||
        throw(DimensionMismatch("modes length ($(length(modes))) != R_vecs length ($(length(R_vecs)))"))
    kw = _positive_periodic_parameter("k", k)
    _validate_floquet_modes(modes, kw)

    p = zeros(Float64, length(modes))
    for (i, mode) in enumerate(modes)
        all(isfinite, R_vecs[i]) ||
            throw(ArgumentError(
                "R_vecs[$i] for order ($(mode.m), $(mode.n)) must be finite"
            ))
        if mode.propagating
            flux_factor = _floquet_mode_flux_factor(mode, kw, i)
            p[i] = _periodic_scaled_intensity(
                R_vecs[i], flux_factor,
                "reflected power fraction for order ($(mode.m), $(mode.n))")
        end
    end
    return p
end

"""
    transmission_coefficients(modes, R_coeffs; incident_order=(0, 0))

Compute transmitted Floquet amplitudes from reflection amplitudes for a free-standing
electric-current sheet model in identical media above and below the sheet.

Exact thin-sheet relation (no branch selection):
- An infinitesimal electric surface current radiates a field whose tangential component is
  equal on both sides of the sheet, so the forward-scattered amplitude equals the
  backward-scattered (reflection) amplitude. Field continuity of the total tangential field
  then gives, for the incident order `(m,n) = incident_order`,
      T₀₀ = 1 + R₀₀
  (incident wave plus forward-scattered = 1 + R). This recovers the physical limits
  exactly: a PEC sheet `R = -1` ⇒ `T = 0`; a transparent sheet `R = 0` ⇒ `T = 1`.
  For a lossless (reactive) sheet it satisfies `|R₀₀|² + |T₀₀|² = 1` identically, so the
  power budget closes without any heuristic branch choice.
- Non-incident propagating orders are purely scattered; by the same forward/backward
  symmetry their transmitted amplitude equals the reflected amplitude (`T_mn = R_mn`).
"""
function transmission_coefficients(modes::Vector{FloquetMode},
                                   R_coeffs::Vector{ComplexF64};
                                   incident_order::Tuple{Int,Int}=(0, 0))
    length(modes) == length(R_coeffs) ||
        throw(DimensionMismatch("modes length ($(length(modes))) != R_coeffs length ($(length(R_coeffs)))"))
    _validate_floquet_mode_orders(modes)
    _incident_floquet_mode_index(modes, incident_order)
    all(isfinite, R_coeffs) ||
        throw(ArgumentError("R_coeffs must contain only finite values"))

    T_coeffs = similar(R_coeffs)
    m_inc, n_inc = incident_order
    for (i, mode) in enumerate(modes)
        if mode.m == m_inc && mode.n == n_inc
            # Exact incident-order transmission for a free-standing current sheet.
            T_coeffs[i] = 1 + R_coeffs[i]
        else
            # Non-incident orders: forward amplitude equals the reflected amplitude.
            T_coeffs[i] = R_coeffs[i]
        end
    end
    return T_coeffs
end

"""
    power_balance(I_coeffs, Z_pen, A_cell, k, modes, R_coeffs;
                  eta0=376.730313668, E0=1.0,
                  transmission=:none, T_coeffs=nothing, incident_order=(0, 0))

Compute the power balance for a periodic metasurface unit cell.

Returns a NamedTuple:
`(P_inc, P_refl, P_abs, P_trans, P_resid, refl_frac, abs_frac, trans_frac, resid_frac)`

- P_inc:   incident power through the unit cell = |E₀|² A / (2η₀)
- P_refl:  reflected power = Σ_mn |R_mn|² (kz_mn/k) P_inc  [propagating modes]
- P_abs:   absorbed by SIMP penalty impedance = ½ Re(I† Z_pen I)
- P_trans: transmitted power (depends on `transmission` mode)
  - `:none`: 0 (legacy behavior)
  - `:closure`: `max(P_inc - P_refl - P_abs, 0)` (conservation-based estimate)
  - `:floquet`: Σ_mn |T_mn|² (kz_mn/k) P_inc using `T_coeffs` or inferred from `R`
- P_resid: residual power = P_inc - P_refl - P_abs - P_trans
- refl_frac: P_refl / P_inc
- abs_frac:  P_abs / P_inc
- trans_frac: P_trans / P_inc
- resid_frac: P_resid / P_inc
"""
function power_balance(I_coeffs::Vector{<:Number},
                       Z_pen::AbstractMatrix,
                       A_cell::Real,
                       k::Real,
                       modes::Vector{FloquetMode},
                       R_coeffs::Vector{ComplexF64};
                       eta0::Float64=376.730313668,
                       E0::Float64=1.0,
                       transmission::Symbol=:none,
                       T_coeffs::Union{Nothing,Vector{ComplexF64}}=nothing,
                       incident_order::Tuple{Int,Int}=(0, 0))
    transmission in (:none, :closure, :floquet) ||
        throw(ArgumentError(
            "Unknown transmission mode: $transmission (expected :none, :closure, or :floquet)"
        ))
    ncurrents = length(I_coeffs)
    size(Z_pen) == (ncurrents, ncurrents) ||
        throw(DimensionMismatch(
            "Z_pen has size $(size(Z_pen)), expected ($ncurrents, $ncurrents)"
        ))
    length(modes) == length(R_coeffs) ||
        throw(DimensionMismatch(
            "modes length ($(length(modes))) != R_coeffs length ($(length(R_coeffs)))"
        ))
    all(isfinite, I_coeffs) ||
        throw(ArgumentError("I_coeffs must contain only finite values"))
    all(isfinite, Z_pen) ||
        throw(ArgumentError("Z_pen must contain only finite values"))
    all(isfinite, R_coeffs) ||
        throw(ArgumentError("R_coeffs must contain only finite values"))

    area = _positive_periodic_parameter("A_cell", A_cell)
    kw = _positive_periodic_parameter("k", k)
    _validate_periodic_field_normalization(E0, eta0)
    _validate_floquet_modes(modes, kw)
    inc_idx = _incident_floquet_mode_index(modes, incident_order)

    # Per-mode z-directed Poynting flux ∝ |coef|²·real(kz)/k. The incident power
    # crossing the cell is the z-flux of the incident order, which carries
    # cos(θ_inc) = real(kz_inc)/k. Normalizing the fractions by this (rather than
    # by the unprojected |E0|²A/2η) is required for energy conservation at oblique
    # incidence; at normal incidence kz_inc = k and nothing changes.
    base = _periodic_incident_power_base(E0, area, eta0)
    isfinite(base) && base > 0.0 ||
        throw(ArgumentError(
            "incident-power normalization is not representable for E0=$E0, A_cell=$area, eta0=$eta0"
        ))
    cos_inc = _floquet_mode_flux_factor(modes[inc_idx], kw, inc_idx)
    P_inc = _periodic_finite_product(
        base, cos_inc, "incident power")
    isfinite(P_inc) && P_inc > 0.0 ||
        throw(ArgumentError(
            "incident power is not representable for incident order $incident_order"
        ))

    # Reflected power from Floquet modes (z-directed flux)
    reflected_factor = _periodic_coefficient_power_sum(
        modes, R_coeffs, kw, "reflected power factor")
    P_refl = _periodic_finite_product(
        reflected_factor, base, "reflected power")

    # Power absorbed by SIMP penalty impedance
    P_abs = _finite_scaled_bilinear_component(
        I_coeffs, Z_pen, I_coeffs, Val(:real),
        0.5, "absorbed power")
    isfinite(P_abs) ||
        throw(OverflowError("absorbed-power evaluation produced a non-finite value"))

    # Transmitted power
    P_trans = 0.0
    if transmission == :none
        P_trans = 0.0
    elseif transmission == :closure
        # Conservation-based transmission estimate from unaccounted non-absorbed power.
        P_trans = clamp(P_inc - P_refl - P_abs, 0.0, P_inc)
    elseif transmission == :floquet
        Tc = isnothing(T_coeffs) ?
            transmission_coefficients(modes, R_coeffs; incident_order=incident_order) :
            T_coeffs
        length(Tc) == length(modes) ||
            throw(DimensionMismatch("T_coeffs length ($(length(Tc))) != modes length ($(length(modes)))"))
        all(isfinite, Tc) ||
            throw(ArgumentError("T_coeffs must contain only finite values"))

        transmitted_factor = _periodic_coefficient_power_sum(
            modes, Tc, kw, "transmitted power factor")
        P_trans = _periodic_finite_product(
            transmitted_factor, base, "transmitted power")
    end

    refl_frac = P_refl / P_inc
    abs_frac = P_abs / P_inc
    trans_frac = P_trans / P_inc
    P_resid = P_inc - P_refl - P_abs - P_trans
    resid_frac = P_resid / P_inc
    all(isfinite, (refl_frac, abs_frac, trans_frac, P_resid, resid_frac)) ||
        throw(OverflowError("power-balance result contains non-finite values"))

    return (P_inc=P_inc, P_refl=P_refl, P_abs=P_abs, P_trans=P_trans, P_resid=P_resid,
            refl_frac=refl_frac, abs_frac=abs_frac, trans_frac=trans_frac, resid_frac=resid_frac)
end

"""
    specular_rcs_objective(mesh, rwg, grid, k, lattice;
                           quad_order=3, half_angle=π/18, polarization=:x)

Build a Q matrix targeting the specular reflection direction.

For normal incidence: specular = broadside (θ=0).
For oblique incidence: specular direction from Snell's law.

`half_angle` sets the cone half-angle (radians) around the specular direction.
`polarization` selects the projection polarization. Supported symbols:

- `:x`, `:theta`, `:tm` → `pol_linear_x(grid)` (`θ̂` basis)
- `:y`, `:phi`, `:te` → `pol_linear_y(grid)` (`φ̂` basis)

You may also pass a custom polarization matrix of size `(3, NΩ)`.

Returns Q ∈ C^{N×N} such that J = Re(I† Q I) measures cone-integrated
specular scattered power in the chosen polarization.
"""
function specular_rcs_objective(mesh::TriMesh, rwg::RWGData,
                                grid::SphGrid, k::Real,
                                lattice::PeriodicLattice;
                                quad_order::Int=3,
                                half_angle::Float64=π/18,
                                polarization=:x)
    _validate_mesh_rwg_pair(mesh, rwg)
    kw = _validated_lattice_wavenumber(k, lattice)
    # Specular (0,0) reflected order keeps the incident transverse wavevector and
    # flips only kz, so r̂ = (kx_bloch, ky_bloch, +kz_inc)/k: θ_r = θ_inc and
    # φ_r = φ_inc = atan(ky_bloch, kx_bloch) (no +π — that would point at the
    # mirror azimuth and miss the specular lobe at oblique incidence).
    kz_inc = _kz_inc(kw, lattice)
    theta_spec = acos(clamp(kz_inc / kw, 0.0, 1.0))
    phi_spec = atan(lattice.ky_bloch, lattice.kx_bloch)

    # Build direction mask for specular cone
    spec_dir = Vec3(sin(theta_spec) * cos(phi_spec),
                    sin(theta_spec) * sin(phi_spec),
                    cos(theta_spec))
    mask = direction_mask(grid, spec_dir; half_angle=half_angle)

    pol = _specular_objective_polarization(grid, polarization)
    # Build radiation vectors and Q matrix
    G_mat = radiation_vectors(mesh, rwg, grid, kw; quad_order=quad_order)
    Q = build_Q(G_mat, grid, pol; mask=mask)

    return Q
end
