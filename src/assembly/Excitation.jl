# Excitation.jl — Incident field and excitation vector assembly
#
# v_m = -⟨f_m, E^inc_t⟩_Γ

export assemble_v_plane_wave, assemble_excitation, assemble_multiple_excitations
export PlaneWaveExcitation, PortExcitation, DeltaGapExcitation, DipoleExcitation,
       LoopExcitation, MonopoleExcitation, ImportedExcitation,
       PatternFeedExcitation, MultiExcitation
export make_plane_wave, make_delta_gap, make_dipole, make_loop, make_monopole,
       make_multi_excitation,
       make_pattern_feed, pattern_feed_field, plane_wave_field,
       make_analytic_dipole_pattern_feed, make_imported_excitation,
       monopole_incident_field

using LinearAlgebra
using SparseArrays
using StaticArrays

const _DEFAULT_MAX_EXCITATION_WORK_BYTES = 512 * 1024 * 1024
const _DEFAULT_MAX_EXCITATION_TERMS = 200_000_000
const _DEFAULT_MAX_MULTI_EXACT_BYTES = 512 * 1024 * 1024
const _MULTI_EXACT_BYTES_PER_ENTRY = 1536
const _MULTI_EXACT_BASE_BYTES = 4096
const _EXCITATION_SURFACE_FALLBACK_PRECISION = 4352
const _TRI_QUAD_ORDERS = (1, 3, 4, 7)
const _MAX_MULTI_EXCITATION_DEPTH = 64

function _multi_exact_accumulator_bytes(N::Int)
    bytes = BigInt(_MULTI_EXACT_BASE_BYTES) +
            BigInt(_MULTI_EXACT_BYTES_PER_ENTRY) * N
    bytes <= typemax(Int) ||
        throw(ArgumentError(
            "MultiExcitation exact-accumulator estimate overflows Int"))
    return Int(bytes)
end

function _multiple_excitation_work_bytes(
        N::Int, M::Int, Nt::Int, Nq::Int, output_bytes::Int)
    return _multiple_excitation_profile_work_bytes(
        N, Nt, (Nq,), 1, output_bytes)
end

function _multiple_excitation_profile_work_bytes(
        N::Int, Nt::Int, quadrature_point_counts,
        temporary_rhs_count::Int, output_bytes::Int)
    total = BigInt(output_bytes)
    # Each effective quadrature order is cached independently and retained for
    # the batch. Outer vector references, areas, and rule arrays are charged.
    for Nq in quadrature_point_counts
        total += BigInt(sizeof(Ptr{Cvoid}) + sizeof(Float64)) * Nt
        total += BigInt(sizeof(Vec3)) * Nt * Nq
        total += BigInt(sizeof(SVector{2,Float64}) + sizeof(Float64)) * Nq
    end
    total += BigInt(sizeof(ComplexF64)) * N * temporary_rhs_count
    total <= typemax(Int) ||
        throw(ArgumentError(
            "multiple-excitation raw-workspace estimate overflows Int"))
    return Int(total)
end

function _multiple_excitation_term_count(
        N::Int, M::Int, Nq::Int)
    count = BigInt(2) * N * M * Nq
    count <= typemax(Int) ||
        throw(ArgumentError(
            "multiple-excitation term estimate overflows Int"))
    return Int(count)
end

function _multiple_excitation_term_count(N::Int, quadrature_points::BigInt)
    count = BigInt(2) * N * quadrature_points
    count <= typemax(Int) ||
        throw(ArgumentError(
            "multiple-excitation term estimate overflows Int"))
    return Int(count)
end

# Abstract type for all excitations
abstract type AbstractExcitation end

"""
    PlaneWaveExcitation

Plane wave excitation with given propagation vector, amplitude, and polarization.
"""
struct PlaneWaveExcitation <: AbstractExcitation
    k_vec::Vec3           # Wave vector (rad/m)
    E0::Float64           # Amplitude (V/m)
    pol::Vec3             # Polarization (unit vector)
end

"""
    make_plane_wave(k_vec, E0, pol)

Create a PlaneWaveExcitation.
"""
function make_plane_wave(k_vec, E0, pol)
    excitation = PlaneWaveExcitation(k_vec, E0, pol)
    _validate_plane_wave_excitation(excitation)
    return excitation
end

"""
    PortExcitation

Port excitation defined by a set of RWG edges with applied voltage.
"""
struct PortExcitation <: AbstractExcitation
    port_edges::Vector{Int}       # RWG edges forming the port
    voltage::ComplexF64           # Port voltage (V)
    impedance::ComplexF64         # Port impedance (Ω)
end

"""
    DeltaGapExcitation

Delta-gap excitation across a single RWG edge.
"""
struct DeltaGapExcitation <: AbstractExcitation
    edge::Int                     # RWG edge with delta gap
    voltage::ComplexF64           # Gap voltage (V)
    gap_length::Float64           # Physical gap length (m)
end

"""
    DipoleExcitation

Electric or magnetic dipole source.
"""
struct DipoleExcitation <: AbstractExcitation
    position::Vec3                # Dipole position (m)
    moment::CVec3                 # Electric: C·m, magnetic: A·m²
    orientation::Vec3             # Preferred orientation metadata
    type::Symbol                  # :electric or :magnetic
    frequency::Float64            # Frequency (Hz), needed for wavenumber
end

"""
    LoopExcitation

Circular loop current source.
"""
struct LoopExcitation <: AbstractExcitation
    center::Vec3                  # Loop center (m)
    normal::Vec3                  # Loop normal (unit vector)
    radius::Float64               # Loop radius (m)
    current::ComplexF64           # Loop current (A)
    frequency::Float64            # Frequency (Hz), needed for wavenumber
end

"""
    MonopoleExcitation

Center-fed linear monopole of length `height` at `position` with axis along
unit vector `axis`. Two modes are supported via `include_image`:

* `include_image = true` (default) — equivalent source for a monopole on an
  infinite PEC ground. Image theory turns the physical wire into a
  length-`2·height` dipole with sinusoidal current
  `I(z') = I₀·sin(k·(height−|z'|))` on `z' ∈ [-height, height]`.
  `monopole_incident_field` returns zero below the ground plane. Use this
  when no explicit ground-plane scatterer is modeled.
* `include_image = false` — physical half-wire only, with current
  `I(z') = I₀·sin(k·(height−z'))` on `z' ∈ [0, height]`, radiating into all
  of free space. Use this when the finite ground plane is meshed as a MoM
  scatterer; the induced plate currents then reproduce the image (and the
  finite-ground truncation corrections).

The `amplitude` parameter follows the UTD `Monopole` convention: when
`include_image = true` the far-field satisfies
`|E_θ| = |amplitude|·|F(θ)|/R` with `F(θ) = [cos(k·h·cosθ) − cos(k·h)]/sinθ`,
and the internal current is `I₀ = -i · 2π · amplitude / η₀`.
"""
struct MonopoleExcitation <: AbstractExcitation
    position::Vec3                # Base of the monopole on the ground plane (m)
    axis::Vec3                    # Unit vector along the monopole
    height::Float64               # Physical length h (m)
    amplitude::ComplexF64         # UTD-consistent field amplitude scaling
    frequency::Float64            # Frequency (Hz), needed for wavenumber
    include_image::Bool           # true: dipole+image model (infinite-ground equivalent)
                                  # false: physical wire only (for use with meshed ground)
    function MonopoleExcitation(pos, axis, height, amplitude, frequency;
                                include_image::Bool = true)
        position = Vec3(pos)
        axv = Vec3(axis)
        height_f = Float64(height)
        amplitude_c = ComplexF64(amplitude)
        frequency_f = Float64(frequency)
        all(isfinite, position) ||
            throw(ArgumentError("MonopoleExcitation: position must be finite"))
        axis_scale = maximum(abs, axv)
        (all(isfinite, axv) && axis_scale > 0.0) ||
            throw(ArgumentError("MonopoleExcitation: axis must be finite and nonzero"))
        axis_scaled = axv / axis_scale
        axis_scaled_norm = norm(axis_scaled)
        (isfinite(axis_scaled_norm) && axis_scaled_norm > 0.0) ||
            throw(ArgumentError("MonopoleExcitation: axis must be finite and nonzero"))
        (isfinite(height_f) && height_f > 0.0) ||
            throw(ArgumentError(
                "MonopoleExcitation: height must be finite and positive, got $height"))
        isfinite(amplitude_c) ||
            throw(ArgumentError(
                "MonopoleExcitation: amplitude must be finite, got $amplitude"))
        (isfinite(frequency_f) && frequency_f > 0.0) ||
            throw(ArgumentError(
                "MonopoleExcitation: frequency must be finite and positive, got $frequency"))
        return new(
            position, axis_scaled / axis_scaled_norm, height_f,
            amplitude_c, frequency_f, include_image)
    end
end

"""
    ImportedExcitation

Canonical spatially imported excitation model for incident-field assembly.

- `kind=:electric_field`: `source_func(r)` returns the incident electric field
  phasor `E_inc(r)` directly.
- `kind=:surface_current_density`: `source_func(r)` returns an equivalent sheet
  current `J_s(r)` and the assembly uses the local map
  `E_inc(r) = eta_equiv * J_s(r)`.
"""
struct ImportedExcitation{F<:Function} <: AbstractExcitation
    source_func::F                # source(r) -> CVec3
    kind::Symbol                  # :electric_field or :surface_current_density
    eta_equiv::ComplexF64         # used only for :surface_current_density
    min_quad_order::Int           # minimum quadrature order target
    function ImportedExcitation(source_func::F;
                                kind::Symbol=:electric_field,
                                eta_equiv::Number=376.730313668 + 0im,
                                min_quad_order::Integer=3) where {F<:Function}
        1 <= min_quad_order <= last(_TRI_QUAD_ORDERS) ||
            throw(ArgumentError(
                "min_quad_order must be between 1 and " *
                "$(last(_TRI_QUAD_ORDERS)), got $min_quad_order."))
        kind in (:electric_field, :surface_current_density) ||
            throw(ArgumentError("Unsupported ImportedExcitation kind: $kind"))
        eta = ComplexF64(eta_equiv)
        isfinite(eta) ||
            throw(ArgumentError("eta_equiv must be finite, got $eta_equiv"))
        return new{F}(source_func, kind, eta, Int(min_quad_order))
    end
end

"""
    make_imported_excitation(source_func; kind=:electric_field,
                             eta_equiv=η0, min_quad_order=3)

Create an `ImportedExcitation`.
"""
make_imported_excitation(source_func::Function;
                         kind::Symbol=:electric_field,
                         eta_equiv::Number=376.730313668 + 0im,
                         min_quad_order::Integer=3) =
    ImportedExcitation(source_func;
                       kind=kind,
                       eta_equiv=eta_equiv,
                       min_quad_order=min_quad_order)

"""
    PatternFeedExcitation

Incident field synthesized from imported far-field coefficients on a spherical grid.
The complex coefficient matrices `Ftheta` and `Fphi` are defined so that, under
the `exp(+iωt)` convention,

E(r) = exp(-ikR)/R * (Fθ(θ,ϕ) * eθ + Fϕ(θ,ϕ) * eϕ),

where `(R, θ, ϕ)` are spherical coordinates of `r - phase_center`.
"""
struct PatternFeedExcitation <: AbstractExcitation
    theta::Vector{Float64}        # θ grid (rad), strictly increasing in [0, π]
    phi::Vector{Float64}          # ϕ grid (rad), strictly increasing over one 2π period (no duplicate endpoint)
    Ftheta::Matrix{ComplexF64}    # size (length(theta), length(phi))
    Fphi::Matrix{ComplexF64}      # size (length(theta), length(phi))
    frequency::Float64            # Hz
    phase_center::Vec3            # phase-center position (m)
    convention::Symbol            # :exp_plus_iwt or :exp_minus_iwt for imported coefficients
end

"""
    MultiExcitation

Combination of multiple excitations with weights.
"""
struct MultiExcitation <: AbstractExcitation
    excitations::Vector{AbstractExcitation}
    weights::Vector{ComplexF64}   # Weight for each excitation
end

"""
    make_multi_excitation(excitations, weights)

Create a MultiExcitation. If weights not provided, use equal weights.
"""
function make_multi_excitation(excitations, weights=nothing)
    if weights === nothing
        weights = ones(ComplexF64, length(excitations))
    end
    excitation = MultiExcitation(excitations, weights)
    _validate_excitation_model(excitation)
    return excitation
end

# Helper functions
function make_delta_gap(edge, voltage, gap_length)
    excitation = DeltaGapExcitation(edge, voltage, gap_length)
    _validate_excitation_model(excitation)
    return excitation
end

function make_dipole(position, moment, orientation, type, frequency=1e9)
    excitation = DipoleExcitation(position, moment, orientation, type, frequency)
    _validate_excitation_model(excitation)
    return excitation
end

function make_loop(center, normal, radius, current, frequency=1e9)
    excitation = LoopExcitation(center, normal, radius, current, frequency)
    _validate_excitation_model(excitation)
    return excitation
end

make_monopole(position, axis, height, amplitude, frequency=1e9;
              include_image::Bool = true) =
    MonopoleExcitation(position, axis, height, amplitude, frequency;
                       include_image=include_image)

const _C0 = 299792458.0
const _EPS0 = 8.854187817e-12
const _MU0 = 4π * 1e-7
const _ETA0 = sqrt(_MU0 / _EPS0)

# A monopole field evaluation performs this quadrature for every observation
# point. Bound the work before converting a floating-point estimate to Int so
# finite but electrically enormous sources cannot overflow the conversion or
# monopolize a solve with an effectively unbounded loop.
const _MAX_MONOPOLE_SIMPSON_INTERVALS = 100_000
const _MONOPOLE_SIMPSON_HALF_INTERVALS_PER_WAVELENGTH = 50.0
const _MAX_MONOPOLE_EXACT_WORK = 2_000_000
const _MONOPOLE_EXACT_BASE_PRECISION = 2304
const _SOURCE_SCALING_FALLBACK_PRECISION = 512
const _SOURCE_SCALING_SAFE_EXPONENT = 128
const _SOURCE_PHASE_PRODUCT_GUARD_BITS = 256
# Two complex Float64 factors span fewer than 4195 binary exponents.  This
# precision also covers their significands and fewer than 2^63 weighted child
# terms, so the exceptional source reductions retain every representable
# residual after arbitrarily ordered cancellation.
const _MULTI_SOURCE_EXACT_PRECISION = 4672
# Above this binary exponent, a single rounded Float64 dot-product term can
# carry more than roughly 2^-12 radians of absolute phase uncertainty.  Route
# such finite arguments through the exact-Float-input reducer as well as the
# outright overflow/underflow cases; otherwise a large but finite product can
# select an unrelated point on the unit circle.
const _SOURCE_PHASE_FAST_TERM_EXPONENT = 40

@inline function _frequency_to_wavenumber(
    frequency::Float64,
    wave_speed::Float64,
    label::AbstractString,
)
    angular_frequency = 2π * frequency
    k = isfinite(angular_frequency) ?
        angular_frequency / wave_speed :
        frequency * (2π / wave_speed)
    (isfinite(k) && k > 0.0) ||
        throw(ArgumentError(
            "$label frequency is outside the representable wavenumber range"))
    return k
end

@inline function _source_scaling_extreme_component(value::Real)
    magnitude = abs(Float64(value))
    iszero(magnitude) && return false
    value_exponent = exponent(magnitude)
    return value_exponent < -_SOURCE_SCALING_SAFE_EXPONENT ||
           value_exponent > _SOURCE_SCALING_SAFE_EXPONENT
end

@inline function _source_scaling_extreme_value(value::Number)
    return _source_scaling_extreme_component(real(value)) ||
           _source_scaling_extreme_component(imag(value))
end

@inline function _source_phase_requires_fallback(
    scale::Float64,
    first_vector::Vec3,
    second_vector::Vec3,
)
    _source_scaling_extreme_value(scale) && return true
    iszero(scale) && return false
    scale_exponent = exponent(abs(scale))
    @inbounds for component in 1:3
        first_value = first_vector[component]
        second_value = second_vector[component]
        (_source_scaling_extreme_value(first_value) ||
         _source_scaling_extreme_value(second_value)) && return true
        if !iszero(first_value) && !iszero(second_value)
            # The extra two bits conservatively cover both significand
            # products and the three-term dot reduction.
            term_exponent = scale_exponent +
                            exponent(abs(first_value)) +
                            exponent(abs(second_value)) + 2
            term_exponent > _SOURCE_PHASE_FAST_TERM_EXPONENT && return true
        end
    end
    return false
end

@inline function _source_phase_precision(
    scale::Float64,
    first_vector::Vec3,
    second_vector::Vec3,
)
    # Each term is the product of three binary Float64 values and therefore has
    # at most 159 significant bits. Include the full exponent span between
    # nonzero terms so a small phase contribution is not rounded away when it
    # is added to a much larger one. The guard exceeds the product width and
    # leaves room for normalization during the two additions.
    scale_exponent = exponent(abs(scale))
    minimum_term_exponent = typemax(Int)
    maximum_term_exponent = typemin(Int)
    @inbounds for component in 1:3
        first_value = first_vector[component]
        second_value = second_vector[component]
        if !iszero(first_value) && !iszero(second_value)
            term_exponent = scale_exponent +
                            exponent(abs(first_value)) +
                            exponent(abs(second_value))
            minimum_term_exponent = min(minimum_term_exponent, term_exponent)
            maximum_term_exponent = max(maximum_term_exponent, term_exponent)
        end
    end
    return minimum_term_exponent == typemax(Int) ?
        _SOURCE_SCALING_FALLBACK_PRECISION :
        max(
            _SOURCE_SCALING_FALLBACK_PRECISION,
            maximum_term_exponent - minimum_term_exponent +
            _SOURCE_PHASE_PRODUCT_GUARD_BITS,
        )
end

@noinline function _source_phase_exact(
    scale::Float64,
    first_vector::Vec3,
    second_vector::Vec3,
    phase_sign::Float64,
    label::AbstractString,
)
    precision = _source_phase_precision(
        scale, first_vector, second_vector)

    return setprecision(BigFloat, precision) do
        argument = BigFloat(scale) * sum(
            BigFloat(first_vector[component]) *
            BigFloat(second_vector[component]) for component in 1:3)
        value = ComplexF64(exp(Complex{BigFloat}(
            zero(BigFloat), BigFloat(phase_sign) * argument)))
        isfinite(value) ||
            throw(OverflowError("$label phase is not representable"))
        return value
    end
end

@inline function _source_scaled_vector_requires_fallback(
    amplitude::Number,
    vector::SVector{3,<:Number},
)
    _source_scaling_extreme_value(amplitude) && return true
    @inbounds for component in vector
        _source_scaling_extreme_value(component) && return true
    end
    return false
end

@inline function _source_product_requires_exact(
    first::Number,
    second::Number,
    product::Number,
)
    first_value = ComplexF64(first)
    second_value = ComplexF64(second)
    product_value = ComplexF64(product)
    real_magnitude =
        abs(real(first_value) * real(second_value)) +
        abs(imag(first_value) * imag(second_value))
    imag_magnitude =
        abs(real(first_value) * imag(second_value)) +
        abs(imag(first_value) * real(second_value))
    return _matrixfree_complex_reduction_requires_exact(
        product_value, real_magnitude, imag_magnitude, 2)
end

@inline function _source_vector_sum_requires_exact(
    first::SVector{3,<:Number},
    second::SVector{3,<:Number},
    combined::SVector{3,<:Number},
)
    @inbounds for component in 1:3
        _scaled_sum_requires_exact(
            first[component], second[component], combined[component]) &&
            return true
    end
    return false
end

@inline function _source_scaled_phase_vector_requires_fallback(
    amplitude::Number,
    vector::SVector{3,<:Number},
    phase_scale::Float64,
    first_vector::Vec3,
    second_vector::Vec3,
)
    _source_scaled_vector_requires_fallback(amplitude, vector) && return true
    return _source_phase_requires_fallback(
        phase_scale, first_vector, second_vector)
end

@noinline function _source_scaled_phase_vector_exact(
    amplitude::Number,
    vector::SVector{3,<:Number},
    phase_scale::Float64,
    first_vector::Vec3,
    second_vector::Vec3,
    phase_sign::Float64,
    label::AbstractString,
)
    precision = _source_phase_precision(
        phase_scale, first_vector, second_vector)
    return setprecision(BigFloat, precision) do
        argument = BigFloat(phase_scale) * sum(
            BigFloat(first_vector[component]) *
            BigFloat(second_vector[component]) for component in 1:3)
        phase = exp(Complex{BigFloat}(
            zero(BigFloat), BigFloat(phase_sign) * argument))
        amplitude_big = Complex{BigFloat}(amplitude)
        value = CVec3(ntuple(component -> ComplexF64(
            amplitude_big * Complex{BigFloat}(vector[component]) * phase), 3))
        all(isfinite, value) ||
            throw(OverflowError("$label is outside the ComplexF64 range"))
        return value
    end
end

@inline function _source_phase(
    scale::Float64,
    first_vector::Vec3,
    second_vector::Vec3,
    phase_sign::Float64,
    label::AbstractString,
)
    iszero(scale) && return 1.0 + 0.0im
    has_nonzero_term = false
    @inbounds for component in 1:3
        if !iszero(first_vector[component]) &&
           !iszero(second_vector[component])
            has_nonzero_term = true
            break
        end
    end
    has_nonzero_term || return 1.0 + 0.0im
    _source_phase_requires_fallback(scale, first_vector, second_vector) &&
        return _source_phase_exact(
            scale, first_vector, second_vector, phase_sign, label)
    argument = scale * dot(first_vector, second_vector)
    isfinite(argument) ||
        return _source_phase_exact(
            scale, first_vector, second_vector, phase_sign, label)
    return exp((phase_sign * 1im) * argument)
end

@inline function _source_directional_phase_precision(
    scale::Float64,
    direction::Vec3,
    center::Vec3,
)
    maximum_direction_exponent = typemin(Int)
    @inbounds for value in direction
        iszero(value) && continue
        maximum_direction_exponent = max(
            maximum_direction_exponent, exponent(abs(value)))
    end

    minimum_term_exponent = typemax(Int)
    maximum_term_exponent = typemin(Int)
    scale_exponent = exponent(abs(scale))
    @inbounds for component in 1:3
        direction_value = direction[component]
        center_value = center[component]
        (iszero(direction_value) || iszero(center_value)) && continue
        # Subtract the common direction scale because normalization removes it.
        term_exponent = scale_exponent +
                        exponent(abs(direction_value)) +
                        exponent(abs(center_value)) -
                        maximum_direction_exponent + 2
        minimum_term_exponent = min(minimum_term_exponent, term_exponent)
        maximum_term_exponent = max(maximum_term_exponent, term_exponent)
    end

    minimum_term_exponent == typemax(Int) &&
        return _SOURCE_SCALING_FALLBACK_PRECISION

    # Unlike a phase formed only from Float64 products, normalization contains
    # a square root.  Its relative error must remain below one radian divided
    # by the largest phase argument, in addition to retaining widely separated
    # terms in the dot product.
    return max(
        _SOURCE_SCALING_FALLBACK_PRECISION,
        maximum_term_exponent - minimum_term_exponent +
        _SOURCE_PHASE_PRODUCT_GUARD_BITS,
        max(0, maximum_term_exponent) +
        _SOURCE_PHASE_PRODUCT_GUARD_BITS,
    )
end

@inline function _source_direction_span_requires_fallback(
    direction::Vec3,
)
    minimum_exponent = typemax(Int)
    maximum_exponent = typemin(Int)
    @inbounds for value in direction
        iszero(value) && continue
        value_exponent = exponent(abs(value))
        minimum_exponent = min(minimum_exponent, value_exponent)
        maximum_exponent = max(maximum_exponent, value_exponent)
    end
    return maximum_exponent - minimum_exponent >
           _SOURCE_SCALING_SAFE_EXPONENT
end

@inline function _source_power_of_two_scaled_direction(
    direction::Vec3,
)
    maximum_exponent = typemin(Int)
    @inbounds for value in direction
        iszero(value) && continue
        maximum_exponent = max(maximum_exponent, exponent(abs(value)))
    end
    scale_exponent = -maximum_exponent
    return Vec3(
        ldexp(direction[1], scale_exponent),
        ldexp(direction[2], scale_exponent),
        ldexp(direction[3], scale_exponent),
    )
end

@noinline function _source_directional_phase_exact(
    scale::Float64,
    direction::Vec3,
    center::Vec3,
    phase_sign::Float64,
    label::AbstractString,
)
    precision = _source_directional_phase_precision(
        scale, direction, center)
    return setprecision(BigFloat, precision) do
        direction_big = SVector{3,BigFloat}(
            ntuple(component -> BigFloat(direction[component]), 3))
        direction_norm = sqrt(sum(abs2, direction_big))
        numerator = sum(
            direction_big[component] * BigFloat(center[component])
            for component in 1:3)
        argument = BigFloat(scale) * numerator / direction_norm
        value = ComplexF64(exp(Complex{BigFloat}(
            zero(BigFloat), BigFloat(phase_sign) * argument)))
        isfinite(value) ||
            throw(OverflowError("$label phase is not representable"))
        return value
    end
end

@inline function _source_directional_phase(
    scale::Float64,
    direction::Vec3,
    normalized_direction::Vec3,
    center::Vec3,
    phase_sign::Float64,
    label::AbstractString,
)
    iszero(scale) && return 1.0 + 0.0im
    nonzero_direction_components = 0
    has_nonzero_term = false
    @inbounds for component in 1:3
        !iszero(direction[component]) &&
            (nonzero_direction_components += 1)
        if !iszero(direction[component]) && !iszero(center[component])
            has_nonzero_term = true
        end
    end
    has_nonzero_term || return 1.0 + 0.0im

    # A single nonzero direction component normalizes exactly to ±1, so the
    # ordinary exact-Float phase reducer already has all supplied information.
    nonzero_direction_components == 1 &&
        return _source_phase(
            scale, normalized_direction, center, phase_sign, label)

    if _source_direction_span_requires_fallback(direction) ||
       _source_phase_requires_fallback(
           scale, normalized_direction, center)
        return _source_directional_phase_exact(
            scale, direction, center, phase_sign, label)
    end
    return _source_phase(
        scale, normalized_direction, center, phase_sign, label)
end

@inline function _dipole_scaling_requires_fallback(
    k::Float64,
    moment::CVec3,
)
    _source_scaling_extreme_value(k) && return true
    @inbounds for component in moment
        _source_scaling_extreme_value(component) && return true
    end
    return false
end

@inline function _dipole_scaling_requires_fallback(
    k::Float64,
    moment::CVec3,
    direction::Vec3,
)
    _dipole_scaling_requires_fallback(k, moment) && return true
    @inbounds for component in direction
        _source_scaling_extreme_value(component) && return true
    end
    return false
end

# Kahan's difference-of-products formula retains the product roundoff before
# the subtraction.  This matters when a dipole is almost parallel to the
# observation direction: the small transverse field can otherwise have the
# wrong sign or magnitude even though every intermediate value is finite.
@inline function _dipole_difference_of_products(
    a::Float64,
    b::Float64,
    c::Float64,
    d::Float64,
)
    cd = c * d
    return fma(a, b, -cd) + fma(-c, d, cd)
end

@inline function _dipole_difference_of_products(
    a::ComplexF64,
    b::Float64,
    c::ComplexF64,
    d::Float64,
)
    return ComplexF64(
        _dipole_difference_of_products(real(a), b, real(c), d),
        _dipole_difference_of_products(imag(a), b, imag(c), d),
    )
end

@inline function _dipole_cross(a::CVec3, b::Vec3)
    return CVec3(
        _dipole_difference_of_products(a[2], b[3], a[3], b[2]),
        _dipole_difference_of_products(a[3], b[1], a[1], b[3]),
        _dipole_difference_of_products(a[1], b[2], a[2], b[1]),
    )
end

@inline function _dipole_cross(a::Vec3, b::Vec3)
    return Vec3(
        _dipole_difference_of_products(a[2], b[3], a[3], b[2]),
        _dipole_difference_of_products(a[3], b[1], a[1], b[3]),
        _dipole_difference_of_products(a[1], b[2], a[2], b[1]),
    )
end

@inline _dipole_cross(a::Vec3, b::CVec3) = -_dipole_cross(b, a)

@inline function _dipole_angular_precision(
    moment::CVec3,
    direction::Vec3,
)
    direction_minimum = typemax(Int)
    direction_maximum = typemin(Int)
    @inbounds for value in direction
        iszero(value) && continue
        value_exponent = exponent(abs(value))
        direction_minimum = min(direction_minimum, value_exponent)
        direction_maximum = max(direction_maximum, value_exponent)
    end

    moment_minimum = typemax(Int)
    moment_maximum = typemin(Int)
    @inbounds for value in moment
        for part in (real(value), imag(value))
            iszero(part) && continue
            value_exponent = exponent(abs(part))
            moment_minimum = min(moment_minimum, value_exponent)
            moment_maximum = max(moment_maximum, value_exponent)
        end
    end

    moment_minimum == typemax(Int) &&
        return _SOURCE_SCALING_FALLBACK_PRECISION

    # An electric projection contains products of two direction components
    # and one moment component.  Cover the complete exponent span of those
    # Float64 products, their three significands, and both cross reductions.
    required_precision =
        2 * (direction_maximum - direction_minimum) +
        (moment_maximum - moment_minimum) +
        3 * precision(Float64) + 64
    return max(_SOURCE_SCALING_FALLBACK_PRECISION, required_precision)
end

@inline function _loop_scaling_requires_fallback(loop::LoopExcitation)
    _source_scaling_extreme_value(loop.radius) && return true
    _source_scaling_extreme_value(loop.current) && return true
    return false
end

@inline function _finite_source_vector(
    value,
    label::AbstractString,
)
    converted = CVec3(ntuple(index -> ComplexF64(value[index]), 3))
    all(isfinite, converted) ||
        throw(OverflowError("$label is outside the ComplexF64 range"))
    return converted
end

@noinline function _dipole_farfield_unphased_exact(
    dipole::DipoleExcitation,
    rhat::Vec3,
    k::Float64,
)
    precision = _dipole_angular_precision(dipole.moment, rhat)
    return setprecision(BigFloat, precision) do
        direction = SVector{3,BigFloat}(
            ntuple(index -> BigFloat(rhat[index]), 3))
        direction_norm_squared = sum(abs2, direction)
        moment = SVector{3,Complex{BigFloat}}(
            ntuple(index -> Complex{BigFloat}(dipole.moment[index]), 3))
        wavenumber = BigFloat(k)
        value = if dipole.type == :electric
            projection = cross(cross(direction, moment), direction) /
                         direction_norm_squared
            (wavenumber * wavenumber /
             (4 * BigFloat(pi) * BigFloat(_EPS0))) * projection
        elseif dipole.type == :magnetic
            (BigFloat(_ETA0) * wavenumber * wavenumber /
             (4 * BigFloat(pi))) *
            (cross(moment, direction) / sqrt(direction_norm_squared))
        else
            error(
                "Dipole type must be :electric or :magnetic, got $(dipole.type).")
        end
        return _finite_source_vector(value, "DipoleExcitation far field")
    end
end

@noinline function _dipole_farfield_exact(
    dipole::DipoleExcitation,
    rhat::Vec3,
    k::Float64,
)
    precision = max(
        _dipole_angular_precision(dipole.moment, rhat),
        _source_directional_phase_precision(k, rhat, dipole.position),
    )
    return setprecision(BigFloat, precision) do
        direction = SVector{3,BigFloat}(
            ntuple(index -> BigFloat(rhat[index]), 3))
        direction_norm_squared = sum(abs2, direction)
        direction_norm = sqrt(direction_norm_squared)
        moment = SVector{3,Complex{BigFloat}}(
            ntuple(index -> Complex{BigFloat}(dipole.moment[index]), 3))
        wavenumber = BigFloat(k)
        unphased = if dipole.type == :electric
            projection = cross(cross(direction, moment), direction) /
                         direction_norm_squared
            (wavenumber * wavenumber /
             (4 * BigFloat(pi) * BigFloat(_EPS0))) * projection
        elseif dipole.type == :magnetic
            (BigFloat(_ETA0) * wavenumber * wavenumber /
             (4 * BigFloat(pi))) *
            (cross(moment, direction) / direction_norm)
        else
            error(
                "Dipole type must be :electric or :magnetic, got $(dipole.type).")
        end
        phase_argument = wavenumber * sum(
            direction[index] * BigFloat(dipole.position[index])
            for index in 1:3) / direction_norm
        phase = exp(Complex{BigFloat}(0, phase_argument))
        return _finite_source_vector(
            unphased * phase, "DipoleExcitation far field")
    end
end

@inline function _dipole_farfield_unphased(
    dipole::DipoleExcitation,
    rhat::Vec3,
    k::Float64,
)
    _dipole_scaling_requires_fallback(k, dipole.moment, rhat) &&
        return _dipole_farfield_unphased_exact(dipole, rhat, k)

    direction_norm_squared = sum(abs2, rhat)
    value = if dipole.type == :electric
        projection = _dipole_cross(
            _dipole_cross(rhat, dipole.moment), rhat) /
                     direction_norm_squared
        (k^2 / (4π * _EPS0)) * projection
    elseif dipole.type == :magnetic
        (_ETA0 * k^2 / (4π)) *
        (_dipole_cross(dipole.moment, rhat) /
         sqrt(direction_norm_squared))
    else
        error(
            "Dipole type must be :electric or :magnetic, got $(dipole.type).")
    end
    all(isfinite, value) ||
        return _dipole_farfield_unphased_exact(dipole, rhat, k)
    return CVec3(value)
end

@noinline function _loop_equivalent_moment_exact(loop::LoopExcitation)
    return setprecision(BigFloat, _SOURCE_SCALING_FALLBACK_PRECISION) do
        radius = BigFloat(loop.radius)
        scale = Complex{BigFloat}(loop.current) * BigFloat(pi) *
                radius * radius
        value = SVector{3,Complex{BigFloat}}(ntuple(
            index -> scale * BigFloat(loop.normal[index]), 3))
        return _finite_source_vector(value, "LoopExcitation equivalent moment")
    end
end

@inline function _loop_equivalent_moment(loop::LoopExcitation)
    _loop_scaling_requires_fallback(loop) &&
        return _loop_equivalent_moment_exact(loop)
    value = loop.current * π * loop.radius^2 * loop.normal
    all(isfinite, value) || return _loop_equivalent_moment_exact(loop)
    return CVec3(value)
end

@inline function _monopole_simpson_interval_count(
    height::Float64,
    k::Float64,
    span_factor::Float64,
    minimum::Int,
    label::AbstractString,
)
    (isfinite(height) && height > 0.0) ||
        throw(ArgumentError("$label height must be finite and positive"))
    (isfinite(k) && k > 0.0) ||
        throw(ArgumentError(
            "$label wavenumber must be finite and positive, got $k"))

    wavelength = 2π / k
    max_half_intervals = _MAX_MONOPOLE_SIMPSON_INTERVALS ÷ 2
    max_span_wavelengths =
        max_half_intervals /
        _MONOPOLE_SIMPSON_HALF_INTERVALS_PER_WAVELENGTH

    # Compare before forming span_factor*height or height/wavelength. This
    # keeps the resource check finite even when either derived value would
    # overflow. If wavelength itself overflows, height*k remains bounded and
    # gives the electrical span without forming that wavelength.
    if isfinite(wavelength)
        max_height = (max_span_wavelengths / span_factor) * wavelength
        if isfinite(max_height) && height > max_height
            throw(ArgumentError(
                "$label would require more than " *
                "$_MAX_MONOPOLE_SIMPSON_INTERVALS Simpson intervals"))
        end
        raw_half_intervals =
            _MONOPOLE_SIMPSON_HALF_INTERVALS_PER_WAVELENGTH *
            span_factor * (height / wavelength)
        if !isfinite(raw_half_intervals) ||
           raw_half_intervals > max_half_intervals
            throw(ArgumentError(
                "$label would require more than " *
                "$_MAX_MONOPOLE_SIMPSON_INTERVALS Simpson intervals"))
        end
        half_intervals = ceil(Int, raw_half_intervals)
    else
        raw_half_intervals =
            (_MONOPOLE_SIMPSON_HALF_INTERVALS_PER_WAVELENGTH *
             span_factor / (2π)) * (height * k)
        half_intervals = ceil(Int, raw_half_intervals)
    end

    intervals = max(minimum, 2 * half_intervals)
    intervals <= _MAX_MONOPOLE_SIMPSON_INTERVALS ||
        throw(ArgumentError(
            "$label would require more than " *
            "$_MAX_MONOPOLE_SIMPSON_INTERVALS Simpson intervals"))
    return intervals
end

function _validate_pattern_grid(theta::Vector{Float64}, phi::Vector{Float64})
    length(theta) ≥ 2 || error("Pattern feed requires at least 2 theta samples.")
    length(phi) ≥ 2 || error("Pattern feed requires at least 2 phi samples.")
    all(isfinite, theta) || error("Theta grid contains non-finite values.")
    all(isfinite, phi) || error("Phi grid contains non-finite values.")
    @inbounds for i in 2:length(theta)
        theta[i] > theta[i - 1] ||
            error("Theta grid must be strictly increasing.")
    end
    @inbounds for i in 2:length(phi)
        phi[i] > phi[i - 1] ||
            error("Phi grid must be strictly increasing.")
    end

    θtol = 1e-12
    if theta[1] < -θtol || theta[end] > π + θtol
        error("Theta grid must lie in [0, π] (radians). Got range [$(theta[1]), $(theta[end])].")
    end

    ϕspan = phi[end] - phi[1]
    if ϕspan <= 0
        error("Phi grid span must be positive.")
    end
    if ϕspan >= 2π - 1e-12
        error("Phi grid should cover one open 2π period without a duplicate endpoint (e.g., 0:Δ:2π-Δ).")
    end
end

function _coerce_pattern_matrix(U, nθ::Int, nϕ::Int, name::AbstractString)
    if size(U) == (nθ, nϕ)
        return Matrix{ComplexF64}(U)
    elseif size(U) == (nϕ, nθ)
        @warn "$name matrix shape $(size(U)) appears transposed; auto-transposing to ($nθ, $nϕ)."
        M = Matrix{ComplexF64}(undef, nθ, nϕ)
        @inbounds for (j, source_row) in enumerate(axes(U, 1))
            for (i, source_col) in enumerate(axes(U, 2))
                M[i, j] = ComplexF64(U[source_row, source_col])
            end
        end
        return M
    else
        error("$name matrix shape $(size(U)) is incompatible with grids (Nθ=$nθ, Nϕ=$nϕ).")
    end
end

function _validate_pattern_feed_storage(
        nθ::Int, nϕ::Int, max_storage_bytes::Integer,
        label::AbstractString="pattern-feed")
    theta_bytes = _checked_array_payload_bytes(
        Float64, nθ; label="$label theta grid")
    phi_bytes = _checked_array_payload_bytes(
        Float64, nϕ; label="$label phi grid")
    matrix_bytes = _checked_array_payload_bytes(
        ComplexF64, nθ, nϕ; label="$label coefficient matrix")
    storage_bytes = try
        Base.Checked.checked_add(
            Base.Checked.checked_add(theta_bytes, phi_bytes),
            Base.Checked.checked_mul(2, matrix_bytes))
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError("$label persistent payload estimate overflows Int"))
    end
    return _enforce_payload_limit(
        storage_bytes, max_storage_bytes,
        "$label angle grids and coefficient matrices",
        "max_storage_bytes")
end

@inline function _validated_pattern_feed_metadata(
        frequency::Real, phase_center::Vec3, convention::Symbol)
    frequency_f = Float64(frequency)
    (isfinite(frequency_f) && frequency_f > 0.0) ||
        throw(ArgumentError(
            "Pattern feed frequency must be finite and positive after " *
            "Float64 conversion, got $frequency."))
    all(isfinite, phase_center) ||
        throw(ArgumentError("Pattern feed phase_center must be finite."))
    convention in (:exp_plus_iwt, :exp_minus_iwt) ||
        throw(ArgumentError(
            "Unsupported pattern convention: $convention " *
            "(expected :exp_plus_iwt or :exp_minus_iwt)."))
    return frequency_f
end

"""
    make_pattern_feed(theta, phi, Ftheta, Fphi, frequency;
                      phase_center=Vec3(0,0,0),
                      angles_in_degrees=false,
                      convention=:exp_plus_iwt)

Create a `PatternFeedExcitation` from explicit spherical grids and complex far-field
coefficient matrices.
"""
function make_pattern_feed(theta::AbstractVector{<:Real},
                           phi::AbstractVector{<:Real},
                           Ftheta::AbstractMatrix,
                           Fphi::AbstractMatrix,
                           frequency::Real;
                           phase_center::Vec3=Vec3(0.0, 0.0, 0.0),
                           angles_in_degrees::Bool=false,
                           convention::Symbol=:exp_plus_iwt,
                           max_storage_bytes::Integer=
                               _DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    frequency_f = _validated_pattern_feed_metadata(
        frequency, phase_center, convention)
    nθ = length(theta)
    nϕ = length(phi)
    _validate_pattern_feed_storage(nθ, nϕ, max_storage_bytes)

    θ = collect(Float64, theta)
    ϕ = collect(Float64, phi)
    if angles_in_degrees
        θ .*= π / 180
        ϕ .*= π / 180
    end

    _validate_pattern_grid(θ, ϕ)

    Fθ = _coerce_pattern_matrix(Ftheta, length(θ), length(ϕ), "Ftheta")
    Fϕ = _coerce_pattern_matrix(Fphi, length(θ), length(ϕ), "Fphi")
    all(isfinite, Fθ) ||
        throw(ArgumentError("Ftheta contains non-finite coefficients."))
    all(isfinite, Fϕ) ||
        throw(ArgumentError("Fphi contains non-finite coefficients."))

    excitation = PatternFeedExcitation(
        θ, ϕ, Fθ, Fϕ, frequency_f, phase_center, convention)
    _validate_excitation_model(excitation)
    return excitation
end

"""
    make_pattern_feed(Etheta_pattern, Ephi_pattern, frequency; ...)

Convenience constructor for importing two pattern objects (e.g., from
`RadiationPatterns.jl`), each exposing fields `.x` (theta), `.y` (phi), and `.U`
(complex values).
"""
function make_pattern_feed(Etheta_pattern, Ephi_pattern, frequency::Real;
                           phase_center::Vec3=Vec3(0.0, 0.0, 0.0),
                           angles_in_degrees::Bool=true,
                           convention::Symbol=:exp_plus_iwt,
                           max_storage_bytes::Integer=
                               _DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    frequency_f = _validated_pattern_feed_metadata(
        frequency, phase_center, convention)
    θ = getproperty(Etheta_pattern, :x)
    ϕ = getproperty(Etheta_pattern, :y)
    θ2 = getproperty(Ephi_pattern, :x)
    ϕ2 = getproperty(Ephi_pattern, :y)
    if length(θ) != length(θ2) || length(ϕ) != length(ϕ2)
        error("Etheta and Ephi patterns must share the same theta/phi grids.")
    end
    length(θ) > 0 || error("Pattern grid theta is empty.")
    length(ϕ) > 0 || error("Pattern grid phi is empty.")
    _validate_pattern_feed_storage(
        length(θ), length(ϕ), max_storage_bytes)
    θf = Float64.(θ)
    ϕf = Float64.(ϕ)
    θ2f = Float64.(θ2)
    ϕ2f = Float64.(ϕ2)
    if maximum(abs.(θf .- θ2f)) > 1e-12 || maximum(abs.(ϕf .- ϕ2f)) > 1e-12
        error("Etheta and Ephi patterns must share the same theta/phi grids.")
    end
    Fθ = getproperty(Etheta_pattern, :U)
    Fϕ = getproperty(Ephi_pattern, :U)
    return make_pattern_feed(θ, ϕ, Fθ, Fϕ, frequency_f;
                             phase_center=phase_center,
                             angles_in_degrees=angles_in_degrees,
                             convention=convention,
                             max_storage_bytes=max_storage_bytes)
end

@inline function _spherical_basis(θ::Float64, ϕ::Float64)
    sθ, cθ = sincos(θ)
    sϕ, cϕ = sincos(ϕ)
    rhat = Vec3(sθ * cϕ, sθ * sϕ, cθ)
    eθ = Vec3(cθ * cϕ, cθ * sϕ, -sθ)
    eϕ = Vec3(-sϕ, cϕ, 0.0)
    return rhat, eθ, eϕ
end

@inline function _bracket_nonperiodic(grid::Vector{Float64}, x::Float64)
    n = length(grid)
    if x <= grid[1]
        return 1, 1, 0.0
    elseif x >= grid[end]
        return n, n, 0.0
    end
    i = clamp(searchsortedlast(grid, x), 1, n - 1)
    t = (x - grid[i]) / (grid[i + 1] - grid[i])
    return i, i + 1, t
end

const _PATTERN_PHI_FAST_EXPONENT = 32

@inline function _pattern_phi_component_requires_exact(value::Float64)
    iszero(value) && return false
    return exponent(abs(value)) > _PATTERN_PHI_FAST_EXPONENT
end

@inline function _pattern_phi_requires_exact(
        phi::Vector{Float64}, ϕ::Float64)
    return _pattern_phi_component_requires_exact(phi[1]) ||
           _pattern_phi_component_requires_exact(phi[end]) ||
           _pattern_phi_component_requires_exact(ϕ)
end

@noinline function _bracket_periodic_phi_exact(
        phi::Vector{Float64}, ϕ::Float64)
    maximum_exponent = 0
    for value in (phi[1], phi[end], ϕ)
        iszero(value) && continue
        maximum_exponent = max(maximum_exponent, exponent(abs(value)))
    end
    precision = max(
        _SOURCE_SCALING_FALLBACK_PRECISION,
        maximum_exponent + _SOURCE_PHASE_PRODUCT_GUARD_BITS,
    )

    return setprecision(BigFloat, precision) do
        n = length(phi)
        origin = BigFloat(phi[1])
        period = 2 * BigFloat(pi)
        wrapped = mod(BigFloat(ϕ) - origin, period) + origin
        final_sample = BigFloat(phi[end])

        if wrapped <= final_sample
            index = searchsortedlast(phi, wrapped)
            if index < n
                lower = BigFloat(phi[index])
                upper = BigFloat(phi[index + 1])
                fraction = Float64((wrapped - lower) / (upper - lower))
                return index, index + 1, clamp(fraction, 0.0, 1.0)
            end
        end

        denominator = origin + period - final_sample
        fraction = Float64((wrapped - final_sample) / denominator)
        return n, 1, clamp(fraction, 0.0, 1.0)
    end
end

@inline function _bracket_periodic_phi(phi::Vector{Float64}, ϕ::Float64)
    _pattern_phi_requires_exact(phi, ϕ) &&
        return _bracket_periodic_phi_exact(phi, ϕ)
    n = length(phi)
    ϕ0 = phi[1]
    ϕw = mod(ϕ - ϕ0, 2π) + ϕ0
    if ϕw <= phi[end]
        i = searchsortedlast(phi, ϕw)
        if i == n
            denom = (phi[1] + 2π - phi[n])
            return n, 1, (ϕw - phi[n]) / denom
        else
            denom = (phi[i + 1] - phi[i])
            return i, i + 1, (ϕw - phi[i]) / denom
        end
    else
        denom = (phi[1] + 2π - phi[n])
        return n, 1, (ϕw - phi[n]) / denom
    end
end

@inline function _bilinear(M::Matrix{ComplexF64},
                           iθ::Int, jθ::Int, tθ::Float64,
                           iϕ::Int, jϕ::Int, tϕ::Float64)
    if iθ == jθ && iϕ == jϕ
        return M[iθ, iϕ]
    elseif iθ == jθ
        return (1 - tϕ) * M[iθ, iϕ] + tϕ * M[iθ, jϕ]
    elseif iϕ == jϕ
        return (1 - tθ) * M[iθ, iϕ] + tθ * M[jθ, iϕ]
    else
        c00 = M[iθ, iϕ]
        c10 = M[jθ, iϕ]
        c01 = M[iθ, jϕ]
        c11 = M[jθ, jϕ]
        return (1 - tθ) * (1 - tϕ) * c00 +
               tθ * (1 - tϕ) * c10 +
               (1 - tθ) * tϕ * c01 +
               tθ * tϕ * c11
    end
end

@inline function _pattern_interp(pat::PatternFeedExcitation, M::Matrix{ComplexF64}, θ::Float64, ϕ::Float64)
    iθ, jθ, tθ = _bracket_nonperiodic(pat.theta, θ)
    iϕ, jϕ, tϕ = _bracket_periodic_phi(pat.phi, ϕ)
    return _bilinear(M, iθ, jθ, tθ, iϕ, jϕ, tϕ)
end

@inline function _pattern_interp_big(
    pat::PatternFeedExcitation,
    M::Matrix{ComplexF64},
    θ::Float64,
    ϕ::Float64,
)
    iθ, jθ, tθ_float = _bracket_nonperiodic(pat.theta, θ)
    iϕ, jϕ, tϕ_float = _bracket_periodic_phi(pat.phi, ϕ)
    tθ = BigFloat(tθ_float)
    tϕ = BigFloat(tϕ_float)
    if iθ == jθ && iϕ == jϕ
        return Complex{BigFloat}(M[iθ, iϕ])
    elseif iθ == jθ
        return (1 - tϕ) * Complex{BigFloat}(M[iθ, iϕ]) +
               tϕ * Complex{BigFloat}(M[iθ, jϕ])
    elseif iϕ == jϕ
        return (1 - tθ) * Complex{BigFloat}(M[iθ, iϕ]) +
               tθ * Complex{BigFloat}(M[jθ, iϕ])
    end
    c00 = Complex{BigFloat}(M[iθ, iϕ])
    c10 = Complex{BigFloat}(M[jθ, iϕ])
    c01 = Complex{BigFloat}(M[iθ, jϕ])
    c11 = Complex{BigFloat}(M[jθ, jϕ])
    return (1 - tθ) * (1 - tϕ) * c00 +
           tθ * (1 - tϕ) * c10 +
           (1 - tθ) * tϕ * c01 +
           tθ * tϕ * c11
end

@inline function _source_radial_phase_precision(
    k::Float64,
    observation::Vec3,
    center::Vec3,
)
    maximum_coordinate_exponent = typemin(Int)
    @inbounds for component in 1:3
        for value in (observation[component], center[component])
            iszero(value) && continue
            maximum_coordinate_exponent = max(
                maximum_coordinate_exponent, exponent(abs(value)))
        end
    end
    maximum_coordinate_exponent == typemin(Int) &&
        return _SOURCE_SCALING_FALLBACK_PRECISION
    phase_exponent = exponent(abs(k)) + maximum_coordinate_exponent + 2
    return max(
        _SOURCE_SCALING_FALLBACK_PRECISION,
        _SOURCE_PHASE_PRODUCT_GUARD_BITS + max(0, phase_exponent),
    )
end

@noinline function _pattern_feed_field_exact(
    r::Vec3,
    pat::PatternFeedExcitation,
    k::Float64,
)
    precision = _source_radial_phase_precision(k, r, pat.phase_center)
    return setprecision(BigFloat, precision) do
        displacement = SVector{3,BigFloat}(ntuple(
            component ->
                BigFloat(r[component]) - BigFloat(pat.phase_center[component]),
            3,
        ))
        distance = sqrt(sum(abs2, displacement))
        iszero(distance) &&
            return CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)

        direction = Vec3(ntuple(
            component -> Float64(displacement[component] / distance), 3))
        all(isfinite, direction) ||
            throw(OverflowError(
                "pattern-feed observation direction is outside the Float64 range"))
        θ = acos(clamp(direction[3], -1.0, 1.0))
        ϕ = atan(direction[2], direction[1])
        ϕ < 0.0 && (ϕ += 2π)
        _, eθ, eϕ = _spherical_basis(θ, ϕ)

        Fθ = _pattern_interp_big(pat, pat.Ftheta, θ, ϕ)
        Fϕ = _pattern_interp_big(pat, pat.Fphi, θ, ϕ)
        if pat.convention == :exp_minus_iwt
            Fθ = conj(Fθ)
            Fϕ = conj(Fϕ)
        end

        phase = exp(Complex{BigFloat}(0, -BigFloat(k) * distance))
        value = SVector{3,Complex{BigFloat}}(ntuple(
            component ->
                (Fθ * BigFloat(eθ[component]) +
                 Fϕ * BigFloat(eϕ[component])) *
                phase / distance,
            3,
        ))
        return _finite_source_vector(value, "pattern-feed incident field")
    end
end

@inline function _pattern_feed_field_requires_exact(
    r::Vec3,
    pat::PatternFeedExcitation,
    k::Float64,
    distance::Float64,
)
    (!isfinite(distance) ||
     _source_scaling_extreme_value(k) ||
     _source_scaling_extreme_value(distance)) && return true
    @inbounds for component in 1:3
        (_source_scaling_extreme_value(r[component]) ||
         _source_scaling_extreme_value(pat.phase_center[component])) && return true
    end
    argument = k * distance
    (!isfinite(argument) ||
     (!iszero(argument) && exponent(abs(argument)) > 32)) && return true
    return false
end

"""
    pattern_feed_field(r, pat)

Evaluate the incident electric field at position `r` generated by
`PatternFeedExcitation`.
"""
function pattern_feed_field(r::Vec3, pat::PatternFeedExcitation)
    _validate_finite_vec3(r, "pattern-feed observation point")
    _validate_excitation_model(pat)
    return _pattern_feed_field_unchecked(r, pat)
end

@inline function _pattern_feed_field_unchecked(r::Vec3, pat::PatternFeedExcitation)
    Rvec = r - pat.phase_center
    R = hypot(hypot(Rvec[1], Rvec[2]), Rvec[3])
    if iszero(R)
        return CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    end

    k = _frequency_to_wavenumber(
        pat.frequency, _C0, "PatternFeedExcitation")
    _pattern_feed_field_requires_exact(r, pat, k, R) &&
        return _pattern_feed_field_exact(r, pat, k)

    θ = acos(clamp(Rvec[3] / R, -1.0, 1.0))
    ϕ = atan(Rvec[2], Rvec[1])
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

    theta_field = Fθ * eθ
    phi_field = Fϕ * eϕ
    angular_field = theta_field + phi_field
    _source_vector_sum_requires_exact(
        theta_field, phi_field, angular_field) &&
        return _pattern_feed_field_exact(r, pat, k)
    unphased_field = angular_field / R
    phase = exp(-1im * k * R)
    field = unphased_field * phase
    all(isfinite, field) || return _pattern_feed_field_exact(r, pat, k)
    @inbounds for component in 1:3
        _source_product_requires_exact(
            unphased_field[component], phase, field[component]) &&
            return _pattern_feed_field_exact(r, pat, k)
    end
    return CVec3(field)
end

"""
    make_analytic_dipole_pattern_feed(dipole, theta, phi; ...)

Generate a `PatternFeedExcitation` from the analytical far-field coefficients of
an electric or magnetic dipole.
"""
function make_analytic_dipole_pattern_feed(dipole::DipoleExcitation,
                                           theta::AbstractVector{<:Real},
                                           phi::AbstractVector{<:Real};
                                           phase_center::Vec3=dipole.position,
                                           angles_in_degrees::Bool=false,
                                           convention::Symbol=:exp_plus_iwt,
                                           max_storage_bytes::Integer=
                                               _DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    _validate_excitation_model(dipole)
    frequency = _validated_pattern_feed_metadata(
        dipole.frequency, phase_center, convention)
    k = _frequency_to_wavenumber(
        frequency, _C0, "DipoleExcitation")
    _validate_pattern_feed_storage(
        length(theta), length(phi), max_storage_bytes,
        "analytical pattern-feed")
    θ = collect(Float64, theta)
    ϕ = collect(Float64, phi)
    if angles_in_degrees
        θ .*= π / 180
        ϕ .*= π / 180
    end
    _validate_pattern_grid(θ, ϕ)
    nθ = length(θ)
    nϕ = length(ϕ)
    Fθ = zeros(ComplexF64, nθ, nϕ)
    Fϕ = zeros(ComplexF64, nθ, nϕ)

    for i in 1:nθ
        for j in 1:nϕ
            rhat, eθ, eϕ = _spherical_basis(θ[i], ϕ[j])
            Fvec = _dipole_farfield_unphased(dipole, rhat, k)
            Fθ[i, j] = dot(Fvec, eθ)
            Fϕ[i, j] = dot(Fvec, eϕ)
        end
    end

    excitation = PatternFeedExcitation(
        θ, ϕ, Fθ, Fϕ, frequency, phase_center, convention)
    _validate_excitation_model(excitation)
    return excitation
end

"""
    plane_wave_field(r, k_vec, E0, pol)

Evaluate a plane wave E^inc(r) = pol * E0 * exp(-i k_vec · r)
at point r. Convention: exp(+iωt).
"""
function plane_wave_field(r::Vec3, k_vec::Vec3, E0, pol::Vec3)
    _validate_finite_vec3(r, "plane-wave observation point")
    _validate_plane_wave_parameters(k_vec, E0, pol)
    return _plane_wave_field_unchecked(r, k_vec, E0, pol)
end

@inline function _plane_wave_field_unchecked(r::Vec3, k_vec::Vec3, E0, pol::Vec3)
    if _source_scaled_phase_vector_requires_fallback(
            E0, pol, 1.0, k_vec, r)
        return _source_scaled_phase_vector_exact(
            E0, pol, 1.0, k_vec, r, -1.0,
            "plane-wave incident field")
    end
    phase = _source_phase(
        1.0, k_vec, r, -1.0, "plane-wave incident field")
    scaled_field = pol * E0
    value = scaled_field * phase
    if all(isfinite, value)
        @inbounds for component in 1:3
            _source_product_requires_exact(
                scaled_field[component], phase, value[component]) &&
                return _source_scaled_phase_vector_exact(
                    E0, pol, 1.0, k_vec, r, -1.0,
                    "plane-wave incident field")
        end
        return value
    end
    return _source_scaled_phase_vector_exact(
        E0, pol, 1.0, k_vec, r, -1.0,
        "plane-wave incident field")
end

function _validate_plane_wave_parameters_and_geometry(
        k_vec::Vec3, E0, pol::Vec3)
    all(isfinite, k_vec) ||
        throw(ArgumentError("plane-wave k_vec components must be finite."))
    k_scale = maximum(abs, k_vec)
    k_scale > 0.0 ||
        throw(ArgumentError("plane-wave k_vec must have a finite, nonzero norm."))
    k_scaled = k_vec / k_scale
    k_scaled_norm = norm(k_scaled)
    isfinite(k_scaled_norm) && k_scaled_norm > 0.0 ||
        throw(ArgumentError("plane-wave k_vec must have a finite, nonzero norm."))
    k_fraction, k_exponent = frexp(k_scale)
    norm_fraction, norm_exponent = frexp(k_fraction * k_scaled_norm)
    combined_exponent = Base.checked_add(k_exponent, norm_exponent)
    k_norm = ldexp(norm_fraction, combined_exponent)
    isfinite(k_norm) && k_norm > 0.0 ||
        throw(ArgumentError(
            "plane-wave k_vec norm is outside the finite Float64 range."))

    E0 isa Number ||
        throw(ArgumentError(
            "plane-wave amplitude E0 must be numeric, got $(typeof(E0))."))
    isfinite(E0) ||
        throw(ArgumentError("plane-wave amplitude E0 must be finite, got $E0."))
    all(isfinite, pol) ||
        throw(ArgumentError("plane-wave polarization components must be finite."))
    pol_scale = maximum(abs, pol)
    pol_scale > 0.0 ||
        throw(ArgumentError(
            "plane-wave polarization must have a finite, nonzero norm."))
    pol_scaled = pol / pol_scale
    pol_scaled_norm = norm(pol_scaled)
    pol_norm = pol_scale * pol_scaled_norm
    isfinite(pol_norm) && pol_norm > 0 ||
        throw(ArgumentError(
            "plane-wave polarization must have a finite, nonzero norm."))
    isapprox(pol_norm, 1.0; rtol=1e-10, atol=1e-12) ||
        throw(ArgumentError(
            "plane-wave polarization must be a unit vector; norm=$pol_norm."))

    transverse_error = abs(dot(
        k_scaled / k_scaled_norm, pol_scaled / pol_scaled_norm))
    transverse_error <= 1e-10 ||
        throw(ArgumentError(
            "plane-wave polarization must be transverse to k_vec; normalized dot=$transverse_error."))
    return (
        magnitude=k_norm,
        direction=Vec3(k_scaled / k_scaled_norm),
    )
end

function _validate_plane_wave_parameters(k_vec::Vec3, E0, pol::Vec3)
    return _validate_plane_wave_parameters_and_geometry(
        k_vec, E0, pol).magnitude
end

function _validate_plane_wave_excitation_geometry(pw::PlaneWaveExcitation)
    return _validate_plane_wave_parameters_and_geometry(
        pw.k_vec, pw.E0, pw.pol)
end

function _validate_plane_wave_excitation(pw::PlaneWaveExcitation)
    return _validate_plane_wave_excitation_geometry(pw).magnitude
end

@inline function _validate_finite_vec3(value, label::AbstractString)
    all(isfinite, value) ||
        throw(ArgumentError("$label components must be finite."))
    return nothing
end

@inline function _check_finite_field(value, label::AbstractString)
    all(isfinite, value) ||
        error("$label produced non-finite values.")
    return value
end

function _validate_excitation_model(excitation::PlaneWaveExcitation)
    _validate_plane_wave_excitation(excitation)
    return nothing
end

function _validate_excitation_model(excitation::PortExcitation)
    isempty(excitation.port_edges) &&
        throw(ArgumentError("PortExcitation requires at least one port edge."))
    @inbounds for i in eachindex(excitation.port_edges)
        edge = excitation.port_edges[i]
        edge >= 1 ||
            throw(ArgumentError(
                "PortExcitation edge $i must be positive, got $edge."))
    end
    isfinite(excitation.voltage) ||
        throw(ArgumentError(
            "PortExcitation voltage must be finite, got $(excitation.voltage)."))
    isfinite(excitation.impedance) ||
        throw(ArgumentError(
            "PortExcitation impedance must be finite, got $(excitation.impedance)."))
    return nothing
end

function _validate_excitation_model(excitation::DeltaGapExcitation)
    excitation.edge >= 1 ||
        throw(ArgumentError(
            "DeltaGapExcitation edge must be positive, got $(excitation.edge)."))
    isfinite(excitation.voltage) ||
        throw(ArgumentError(
            "DeltaGapExcitation voltage must be finite, got $(excitation.voltage)."))
    (isfinite(excitation.gap_length) && excitation.gap_length > 0.0) ||
        throw(ArgumentError(
            "DeltaGapExcitation gap_length must be finite and positive, got $(excitation.gap_length)."))
    return nothing
end

function _validate_excitation_model(excitation::DipoleExcitation)
    _validate_finite_vec3(excitation.position, "DipoleExcitation position")
    _validate_finite_vec3(excitation.moment, "DipoleExcitation moment")
    _validate_finite_vec3(excitation.orientation, "DipoleExcitation orientation")
    orientation_norm = norm(excitation.orientation)
    (isfinite(orientation_norm) &&
     isapprox(orientation_norm, 1.0; rtol=1e-10, atol=1e-12)) ||
        throw(ArgumentError(
            "DipoleExcitation orientation must be a unit vector; norm=$orientation_norm."))
    excitation.type in (:electric, :magnetic) ||
        throw(ArgumentError(
            "DipoleExcitation type must be :electric or :magnetic, got $(excitation.type)."))
    (isfinite(excitation.frequency) && excitation.frequency > 0.0) ||
        throw(ArgumentError(
            "DipoleExcitation frequency must be finite and positive, got $(excitation.frequency)."))
    return nothing
end

function _validate_excitation_model(excitation::LoopExcitation)
    _validate_finite_vec3(excitation.center, "LoopExcitation center")
    _validate_finite_vec3(excitation.normal, "LoopExcitation normal")
    normal_norm = norm(excitation.normal)
    (isfinite(normal_norm) &&
     isapprox(normal_norm, 1.0; rtol=1e-10, atol=1e-12)) ||
        throw(ArgumentError(
            "LoopExcitation normal must be a unit vector; norm=$normal_norm."))
    (isfinite(excitation.radius) && excitation.radius > 0.0) ||
        throw(ArgumentError(
            "LoopExcitation radius must be finite and positive, got $(excitation.radius)."))
    isfinite(excitation.current) ||
        throw(ArgumentError(
            "LoopExcitation current must be finite, got $(excitation.current)."))
    (isfinite(excitation.frequency) && excitation.frequency > 0.0) ||
        throw(ArgumentError(
            "LoopExcitation frequency must be finite and positive, got $(excitation.frequency)."))
    return nothing
end

function _validate_excitation_model(excitation::MonopoleExcitation)
    _validate_finite_vec3(excitation.position, "MonopoleExcitation position")
    _validate_finite_vec3(excitation.axis, "MonopoleExcitation axis")
    isapprox(norm(excitation.axis), 1.0; rtol=1e-10, atol=1e-12) ||
        throw(ArgumentError("MonopoleExcitation axis must be a unit vector."))
    (isfinite(excitation.height) && excitation.height > 0.0) ||
        throw(ArgumentError("MonopoleExcitation height must be finite and positive."))
    isfinite(excitation.amplitude) ||
        throw(ArgumentError("MonopoleExcitation amplitude must be finite."))
    (isfinite(excitation.frequency) && excitation.frequency > 0.0) ||
        throw(ArgumentError("MonopoleExcitation frequency must be finite and positive."))
    return nothing
end

function _validate_excitation_model(excitation::ImportedExcitation)
    isfinite(excitation.eta_equiv) ||
        throw(ArgumentError(
            "ImportedExcitation eta_equiv must be finite, got $(excitation.eta_equiv)."))
    1 <= excitation.min_quad_order <= last(_TRI_QUAD_ORDERS) ||
        throw(ArgumentError(
            "ImportedExcitation min_quad_order must be between 1 and " *
            "$(last(_TRI_QUAD_ORDERS)), got $(excitation.min_quad_order)."))
    excitation.kind in (:electric_field, :surface_current_density) ||
        throw(ArgumentError(
            "Unsupported ImportedExcitation kind: $(excitation.kind)."))
    return nothing
end

function _validate_excitation_model(excitation::PatternFeedExcitation)
    _validate_pattern_grid(excitation.theta, excitation.phi)
    expected_size = (length(excitation.theta), length(excitation.phi))
    size(excitation.Ftheta) == expected_size ||
        throw(DimensionMismatch(
            "PatternFeedExcitation Ftheta has size $(size(excitation.Ftheta)), expected $expected_size."))
    size(excitation.Fphi) == expected_size ||
        throw(DimensionMismatch(
            "PatternFeedExcitation Fphi has size $(size(excitation.Fphi)), expected $expected_size."))
    all(isfinite, excitation.Ftheta) ||
        throw(ArgumentError("PatternFeedExcitation Ftheta must be finite."))
    all(isfinite, excitation.Fphi) ||
        throw(ArgumentError("PatternFeedExcitation Fphi must be finite."))
    (isfinite(excitation.frequency) && excitation.frequency > 0.0) ||
        throw(ArgumentError(
            "PatternFeedExcitation frequency must be finite and positive, got $(excitation.frequency)."))
    _validate_finite_vec3(
        excitation.phase_center, "PatternFeedExcitation phase_center")
    excitation.convention in (:exp_plus_iwt, :exp_minus_iwt) ||
        throw(ArgumentError(
            "Unsupported pattern convention: $(excitation.convention)."))
    return nothing
end

function _validate_multi_excitation(
        excitation::MultiExcitation,
        active_children::IdDict{Vector{AbstractExcitation},Nothing},
        depth::Int)
    depth <= _MAX_MULTI_EXCITATION_DEPTH ||
        throw(ArgumentError(
            "MultiExcitation nesting exceeds the supported depth " *
            "$_MAX_MULTI_EXCITATION_DEPTH."))
    length(excitation.excitations) == length(excitation.weights) ||
        throw(DimensionMismatch(
            "MultiExcitation has $(length(excitation.excitations)) excitations " *
            "but $(length(excitation.weights)) weights."))
    isempty(excitation.excitations) &&
        throw(ArgumentError("MultiExcitation requires at least one child excitation."))
    children = excitation.excitations
    haskey(active_children, children) &&
        throw(ArgumentError("MultiExcitation graph contains a cycle."))
    active_children[children] = nothing
    try
        @inbounds for i in eachindex(children)
            isfinite(excitation.weights[i]) ||
                throw(ArgumentError(
                    "MultiExcitation weight $i must be finite, got $(excitation.weights[i])."))
            try
                child = children[i]
                if child isa MultiExcitation
                    _validate_multi_excitation(
                        child, active_children, depth + 1)
                else
                    _validate_excitation_model(child)
                end
            catch err
                throw(ArgumentError(
                    "MultiExcitation child $i failed validation: " *
                    "$(sprint(showerror, err))"))
            end
        end
    finally
        delete!(active_children, children)
    end
    return nothing
end

function _validate_excitation_model(excitation::MultiExcitation)
    return _validate_multi_excitation(
        excitation,
        IdDict{Vector{AbstractExcitation},Nothing}(),
        1)
end

function _validate_plane_wave_wavenumber(pw::PlaneWaveExcitation,
                                         expected_k::Real,
                                         context::AbstractString)
    k_norm = _validate_plane_wave_excitation(pw)
    expected = Float64(expected_k)
    isfinite(expected) && expected > 0 ||
        throw(ArgumentError("$context expected wavenumber must be finite and positive, got $expected."))
    isapprox(k_norm, expected; rtol=1e-8, atol=0.0) ||
        throw(ArgumentError(
            "$context plane-wave |k_vec|=$k_norm does not match the problem wavenumber $expected."))
    return k_norm
end

@inline function _check_finite_cvec3(v::CVec3, label::AbstractString)
    for comp in v
        if !isfinite(real(comp)) || !isfinite(imag(comp))
            error("$label must return finite values, got component $comp.")
        end
    end
    return v
end

@inline function _to_cvec3(v::CVec3, label::AbstractString)
    return _check_finite_cvec3(v, label)
end

@inline function _to_cvec3(v::Vec3, label::AbstractString)
    return _check_finite_cvec3(CVec3(complex(v[1]), complex(v[2]), complex(v[3])), label)
end

@inline function _to_cvec3(v::SVector{3,<:Number}, label::AbstractString)
    return _check_finite_cvec3(CVec3(ComplexF64(v[1]), ComplexF64(v[2]), ComplexF64(v[3])), label)
end

@inline function _to_cvec3(v::NTuple{3,<:Number}, label::AbstractString)
    return _check_finite_cvec3(CVec3(ComplexF64(v[1]), ComplexF64(v[2]), ComplexF64(v[3])), label)
end

@inline function _to_cvec3(v::AbstractVector{<:Number}, label::AbstractString)
    length(v) == 3 || error("$label must return a 3-component vector, got length $(length(v)).")
    return _check_finite_cvec3(CVec3(ComplexF64(v[1]), ComplexF64(v[2]), ComplexF64(v[3])), label)
end

@inline function _to_cvec3(v, label::AbstractString)
    error("$label must return a 3-component numeric vector (e.g. `CVec3`, `Vec3`, tuple, or length-3 vector). Got $(typeof(v)).")
end

function _effective_quad_order(requested::Int, min_required::Int)
    requested >= 1 ||
        throw(ArgumentError(
            "requested quadrature order must be positive, got $requested"))
    min_required >= 1 ||
        throw(ArgumentError(
            "minimum quadrature order must be positive, got $min_required"))
    target = max(requested, min_required)
    target <= last(_TRI_QUAD_ORDERS) ||
        throw(ArgumentError(
            "quadrature order target $target exceeds the highest supported " *
            "order $(last(_TRI_QUAD_ORDERS))"))
    for q in _TRI_QUAD_ORDERS
        q >= target && return q
    end
    error("unreachable quadrature-order mapping")
end

function _excitation_quad_orders!(
        orders::Set{Int}, excitation::AbstractExcitation,
        requested_order::Int)
    push!(orders, requested_order)
    return orders
end

function _excitation_quad_orders!(
        orders::Set{Int}, ::Union{PortExcitation,DeltaGapExcitation},
        ::Int)
    return orders
end

function _excitation_quad_orders!(
        orders::Set{Int}, excitation::ImportedExcitation,
        requested_order::Int)
    push!(orders,
          _effective_quad_order(requested_order, excitation.min_quad_order))
    return orders
end

function _excitation_quad_orders!(
        orders::Set{Int}, excitation::MultiExcitation,
        requested_order::Int)
    for child in excitation.excitations
        _excitation_quad_orders!(orders, child, requested_order)
    end
    return orders
end

_excitation_rhs_vector_count(::AbstractExcitation) = 1

function _excitation_rhs_vector_count(excitation::MultiExcitation)
    maximum_child_count = 0
    for child in excitation.excitations
        maximum_child_count = max(
            maximum_child_count, _excitation_rhs_vector_count(child))
    end
    # The MultiExcitation output and the child that triggered exact fallback
    # can coexist with the child being reassembled. Charging two vectors per
    # nesting level therefore bounds the retained Float64 RHS stack.
    maximum_child_count <= typemax(Int) - 2 ||
        throw(ArgumentError(
            "multiple-excitation temporary RHS estimate overflows Int"))
    return maximum_child_count + 2
end

function _excitation_term_points(
        excitation::AbstractExcitation, requested_order::Int,
        requested_points::Int)
    return BigInt(requested_points)
end

function _excitation_term_points(
        excitation::ImportedExcitation, requested_order::Int,
        ::Int)
    effective_order =
        _effective_quad_order(requested_order, excitation.min_quad_order)
    return BigInt(length(last(tri_quad_rule(effective_order))))
end

function _excitation_term_points(
        excitation::MultiExcitation, requested_order::Int,
        requested_points::Int)
    child_points = sum(
        (_excitation_term_points(child, requested_order, requested_points)
         for child in excitation.excitations);
        init=BigInt(0))
    # A late exceptional child makes `assemble_multi` complete its normal pass
    # and then reassemble every child into the exact accumulator.
    return 2 * child_points
end

function _multiple_excitation_profile(
        excitations::AbstractVector{<:AbstractExcitation},
        requested_order::Int, requested_points::Int)
    quadrature_orders = Set{Int}()
    temporary_rhs_count = 0
    term_points = BigInt(0)
    for excitation in excitations
        _excitation_quad_orders!(
            quadrature_orders, excitation, requested_order)
        temporary_rhs_count = max(
            temporary_rhs_count,
            _excitation_rhs_vector_count(excitation))
        term_points += _excitation_term_points(
            excitation, requested_order, requested_points)
    end
    return quadrature_orders, temporary_rhs_count, term_points
end

function _excitation_quadrature_cache(mesh::TriMesh, quad_order::Int)
    xi, wq = tri_quad_rule(quad_order)
    Nt = ntriangles(mesh)
    quad_pts = Vector{Vector{Vec3}}(undef, Nt)
    areas = Vector{Float64}(undef, Nt)
    @inbounds for t in 1:Nt
        quad_pts[t] = tri_quad_points(mesh, t, xi)
        areas[t] = triangle_area(mesh, t)
    end
    return wq, quad_pts, areas
end

# Memoizing wrapper over `_excitation_quadrature_cache`. The mesh quadrature
# (points + areas) only depends on `(mesh, quad_order)`, so when many
# excitations are assembled against the same mesh (see `assemble_multi` and
# `assemble_multiple_excitations`) the per-order cache is built exactly once and
# shared. Different excitation types may request different orders (e.g.
# `ImportedExcitation` uses `_effective_quad_order`), so the cache is keyed by
# `quad_order`; each distinct order is materialized lazily on first use.
struct ExcitationQuadCache
    mesh::TriMesh
    by_order::Dict{Int,Tuple{Vector{Float64},Vector{Vector{Vec3}},Vector{Float64}}}
    cache_lock::ReentrantLock
end

ExcitationQuadCache(mesh::TriMesh) =
    ExcitationQuadCache(mesh,
        Dict{Int,Tuple{Vector{Float64},Vector{Vector{Vec3}},Vector{Float64}}}(),
        ReentrantLock())

ExcitationQuadCache(
    mesh::TriMesh,
    by_order::Dict{
        Int,Tuple{Vector{Float64},Vector{Vector{Vec3}},Vector{Float64}}},
) = ExcitationQuadCache(mesh, by_order, ReentrantLock())

# Return `(wq, quad_pts, areas)` for `quad_order`, computing and caching on miss.
function _quad_cache_for(cache::ExcitationQuadCache, mesh::TriMesh, quad_order::Int)
    (cache.mesh.xyz === mesh.xyz && cache.mesh.tri === mesh.tri) ||
        throw(ArgumentError(
            "quad_cache belongs to a different mesh; construct ExcitationQuadCache(mesh) for this mesh."))
    lock(cache.cache_lock)
    try
        cached = get(cache.by_order, quad_order, nothing)
        cached === nothing || return cached
        entry = _excitation_quadrature_cache(cache.mesh, quad_order)
        cache.by_order[quad_order] = entry
        return entry
    finally
        unlock(cache.cache_lock)
    end
end

"""
    dipole_incident_field(r, dipole)

Evaluate incident electric field from electric or magnetic dipole at point `r`.
Uses the `exp(+iωt)` convention and free-space Green-function fields.
"""
function dipole_incident_field(r::Vec3, dipole::DipoleExcitation)
    _validate_finite_vec3(r, "dipole observation point")
    _validate_excitation_model(dipole)
    return _dipole_incident_field_unchecked(r, dipole)
end

@inline function _dipole_incident_field_requires_exact(
    r::Vec3,
    dipole::DipoleExcitation,
    k::Float64,
    distance::Float64,
)
    !isfinite(distance) && return true
    @inbounds for component in 1:3
        (_source_scaling_extreme_value(r[component]) ||
         _source_scaling_extreme_value(dipole.position[component])) &&
            return true
    end
    argument = k * distance
    return !isfinite(argument) ||
           (!iszero(argument) && exponent(abs(argument)) > 32)
end

@noinline function _dipole_incident_field_exact(
    r::Vec3,
    dipole::DipoleExcitation,
    k::Float64,
)
    precision = _source_radial_phase_precision(k, r, dipole.position)
    return setprecision(BigFloat, precision) do
        displacement = SVector{3,BigFloat}(ntuple(
            component -> BigFloat(r[component]) -
                         BigFloat(dipole.position[component]),
            3,
        ))
        distance = sqrt(sum(abs2, displacement))
        iszero(distance) && return zero(CVec3)
        direction = displacement / distance
        moment = SVector{3,Complex{BigFloat}}(ntuple(
            component -> Complex{BigFloat}(dipole.moment[component]), 3))
        wavenumber = BigFloat(k)
        phase = exp(Complex{BigFloat}(0, -wavenumber * distance))
        value = if dipole.type == :electric
            transverse = cross(cross(direction, moment), direction) *
                         wavenumber^2
            near = (3 * direction * dot(direction, moment) - moment) *
                   (inv(distance^2) +
                    Complex{BigFloat}(0, 1) * wavenumber / distance)
            (transverse + near) * phase /
                (4 * BigFloat(pi) * BigFloat(_EPS0) * distance)
        elseif dipole.type == :magnetic
            (BigFloat(_ETA0) / (4 * BigFloat(pi))) *
            (wavenumber^2 / distance -
             Complex{BigFloat}(0, 1) * wavenumber / distance^2) *
            phase * cross(moment, direction)
        else
            error(
                "Dipole type must be :electric or :magnetic, got $(dipole.type).")
        end
        return _finite_source_vector(value, "dipole incident field")
    end
end

@inline function _dipole_incident_field_unchecked(r::Vec3, dipole::DipoleExcitation)
    ϵ0 = 8.854187817e-12
    μ0 = 4π * 1e-7
    η0 = sqrt(μ0 / ϵ0)
    c0 = 1 / sqrt(ϵ0 * μ0)

    k = _frequency_to_wavenumber(
        dipole.frequency, c0, "DipoleExcitation")

    R_vec = r - dipole.position
    R = hypot(hypot(R_vec[1], R_vec[2]), R_vec[3])
    if iszero(R)
        return CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    end

    _dipole_incident_field_requires_exact(r, dipole, k, R) &&
        return _dipole_incident_field_exact(r, dipole, k)

    p = dipole.moment
    projection_direction = _source_power_of_two_scaled_direction(R_vec)
    direction_norm_squared = sum(abs2, projection_direction)

    if dipole.type == :electric
        term1 = _dipole_cross(
            _dipole_cross(projection_direction, p),
            projection_direction) /
            direction_norm_squared * k^2
        radial_projection = projection_direction *
                            dot(projection_direction, p) /
                            direction_norm_squared
        term2 = (3 * radial_projection - p) *
                (1 / R^2 + 1im * k / R)
        value = (term1 + term2) * exp(-1im * k * R) / (4π * ϵ0 * R)
        return all(isfinite, value) ? CVec3(value) :
               _dipole_incident_field_exact(r, dipole, k)
    elseif dipole.type == :magnetic
        # E = (η0/4π) (k^2/R - i k/R^2) e^{-ikR} (m × R̂)
        # Derived from E = -i k η0 (∇G × m), G = e^{-ikR}/(4πR); radiating (1/R)
        # term has a REAL coefficient, matching the electric-dipole convention.
        value = (η0 / (4π)) * (k^2 / R - 1im * k / R^2) *
                exp(-1im * k * R) *
                (_dipole_cross(p, projection_direction) /
                 sqrt(direction_norm_squared))
        return all(isfinite, value) ? CVec3(value) :
               _dipole_incident_field_exact(r, dipole, k)
    else
        error("Dipole type must be :electric or :magnetic, got $(dipole.type).")
    end
end

"""
    loop_incident_field(r, loop)

Evaluate electric field from a small loop source via equivalent magnetic dipole.
"""
function loop_incident_field(r::Vec3, loop::LoopExcitation)
    _validate_finite_vec3(r, "loop observation point")
    _validate_excitation_model(loop)
    return _loop_incident_field_unchecked(r, loop)
end

@inline function _loop_incident_field_unchecked(r::Vec3, loop::LoopExcitation)
    m = _loop_equivalent_moment(loop)
    dip = DipoleExcitation(loop.center, m, loop.normal, :magnetic, loop.frequency)
    return _dipole_incident_field_unchecked(r, dip)
end

# Hertzian current element (Balanis eq 4-56a,b, exp(+iωt) convention).
# Returns dE/dz' for a unit-length axial element carrying current `I` at `r_p`.
@inline function _hertzian_per_length_field(r::Vec3, r_p::Vec3, axis::Vec3,
                                            I::ComplexF64, k::Float64)
    η0 = 376.730313668
    R_vec = r - r_p
    R = hypot(hypot(R_vec[1], R_vec[2]), R_vec[3])
    iszero(R) && return CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    projection_direction = _source_power_of_two_scaled_direction(R_vec)
    direction_norm_squared = sum(abs2, projection_direction)
    direction_norm = sqrt(direction_norm_squared)
    R_hat = projection_direction / direction_norm
    cosθ = clamp(
        dot(projection_direction, axis) / direction_norm, -1.0, 1.0)
    theta_numerator = _dipole_cross(
        projection_direction,
        _dipole_cross(projection_direction, axis)) /
        direction_norm_squared
    expfac = exp(-1im * k * R)
    kR = k * R

    E_r = η0 * I / (2π) * cosθ * expfac / R^2 * (1.0 + 1.0 / (1im * kR))
    transverse_scale = 1im * η0 * k * I / (4π) * expfac / R *
                       (1.0 + 1.0 / (1im * kR) - 1.0 / kR^2)
    return E_r * R_hat + transverse_scale * theta_numerator
end

"""
    monopole_incident_field(r, mono; max_exact_work=2_000_000)

Incident electric field at `r` from a `MonopoleExcitation`, computed by
rigorous Simpson integration of the sinusoidal current on the equivalent
dipole-plus-image distribution. Uses the exp(+iωt) convention. Returns zero
below the ground plane (axial projection of `r − position` is negative).
"""
function monopole_incident_field(
        r::Vec3,
        mono::MonopoleExcitation;
        max_exact_work::Integer=_MAX_MONOPOLE_EXACT_WORK)
    _validate_finite_vec3(r, "monopole observation point")
    _validate_excitation_model(mono)
    max_exact_work >= 0 ||
        throw(ArgumentError(
            "max_exact_work must be nonnegative, got $max_exact_work"))
    max_exact_work <= typemax(Int) ||
        throw(ArgumentError("max_exact_work exceeds the supported Int range"))
    return _monopole_incident_field_unchecked(
        r, mono, Int(max_exact_work))
end

@inline function _monopole_incident_field_requires_exact(
        r::Vec3,
        mono::MonopoleExcitation,
        k::Float64)
    displacement = r - mono.position
    distance = hypot(
        displacement[1], displacement[2], displacement[3])
    !isfinite(distance) && return true
    @inbounds for component in 1:3
        (_source_scaling_extreme_value(r[component]) ||
         _source_scaling_extreme_value(mono.position[component])) &&
            return true
    end
    (_source_scaling_extreme_value(k) ||
     _source_scaling_extreme_value(mono.height) ||
     _source_scaling_extreme_value(mono.amplitude)) && return true
    maximum_distance = distance + mono.height
    phase_bound = k * maximum_distance
    return !isfinite(maximum_distance) || !isfinite(phase_bound) ||
           (!iszero(phase_bound) && exponent(abs(phase_bound)) > 32)
end

@noinline function _monopole_below_ground_exact(
        r::Vec3,
        mono::MonopoleExcitation,
        k::Float64)
    precision = _monopole_exact_precision(r, mono, k)
    return setprecision(BigFloat, precision) do
        displacement = SVector{3,BigFloat}(ntuple(
            component -> BigFloat(r[component]) -
                         BigFloat(mono.position[component]), 3))
        axis = SVector{3,BigFloat}(BigFloat.(Tuple(mono.axis)))
        distance = sqrt(sum(abs2, displacement))
        projection = dot(displacement, axis)
        reference_length = max(
            BigFloat(mono.height), distance, one(BigFloat))
        return projection < -BigFloat(1.0e-12) * reference_length
    end
end

@inline function _monopole_exact_precision(
        r::Vec3,
        mono::MonopoleExcitation,
        k::Float64)
    radial_precision = _source_radial_phase_precision(
        k, r, mono.position)
    height_exponent = iszero(mono.height) ? 0 : exponent(mono.height)
    height_phase_exponent = exponent(abs(k)) + height_exponent + 2
    return max(
        _MONOPOLE_EXACT_BASE_PRECISION,
        radial_precision,
        _SOURCE_PHASE_PRODUCT_GUARD_BITS +
            max(0, height_phase_exponent),
    )
end

@inline function _monopole_exact_work_preflight(
        intervals::Int,
        precision::Int,
        max_exact_work::Int)
    terms = Base.checked_add(intervals, 1)
    (iszero(precision) || terms <= div(max_exact_work, precision)) ||
        throw(ArgumentError(
            "monopole incident-field exact evaluation requires more than " *
            "max_exact_work=$max_exact_work precision-weighted terms"))
    return nothing
end

@noinline function _monopole_incident_field_exact(
        r::Vec3,
        mono::MonopoleExcitation,
        k::Float64,
        intervals::Int,
        max_exact_work::Int)
    precision = _monopole_exact_precision(r, mono, k)
    _monopole_exact_work_preflight(
        intervals, precision, max_exact_work)
    return setprecision(BigFloat, precision) do
        observation = SVector{3,BigFloat}(BigFloat.(Tuple(r)))
        position = SVector{3,BigFloat}(BigFloat.(Tuple(mono.position)))
        axis = SVector{3,BigFloat}(BigFloat.(Tuple(mono.axis)))
        height = BigFloat(mono.height)
        wavenumber = BigFloat(k)

        eta = BigFloat(_ETA0)
        current_scale = Complex{BigFloat}(0, -1) * 2 * BigFloat(pi) *
                        Complex{BigFloat}(mono.amplitude) / eta
        span = mono.include_image ? BigFloat(2) : one(BigFloat)
        step = span * height / intervals
        total = zero(SVector{3,Complex{BigFloat}})
        @inbounds for index in 0:intervals
            unit_position = mono.include_image ?
                -one(BigFloat) + 2 * BigFloat(index) / intervals :
                BigFloat(index) / intervals
            axial_position = height * unit_position
            source = position + axial_position * axis
            current = current_scale * sin(
                wavenumber * (height - abs(axial_position)))
            displacement = observation - source
            distance = sqrt(sum(abs2, displacement))
            if !iszero(distance) && !iszero(current)
                direction = displacement / distance
                cosine = clamp(dot(direction, axis), -one(BigFloat), one(BigFloat))
                phase = exp(Complex{BigFloat}(0, -wavenumber * distance))
                kR = wavenumber * distance
                radial = eta * current / (2 * BigFloat(pi)) * cosine *
                         phase / distance^2 *
                         (1 + inv(Complex{BigFloat}(0, 1) * kR))
                contribution = radial * direction
                theta_numerator = cross(
                    displacement, cross(displacement, axis)) / distance^2
                transverse_scale = Complex{BigFloat}(0, 1) * eta *
                    wavenumber * current / (4 * BigFloat(pi)) *
                    phase / distance *
                    (1 + inv(Complex{BigFloat}(0, 1) * kR) -
                     inv(kR^2))
                contribution += transverse_scale * theta_numerator
                weight = (iszero(index) || index == intervals) ?
                         1 : (isodd(index) ? 4 : 2)
                total += weight * contribution
            end
        end
        value = (step / 3) * total
        return _finite_source_vector(value, "monopole incident field")
    end
end

function _monopole_incident_field_unchecked(
        r::Vec3,
        mono::MonopoleExcitation,
        max_exact_work::Int=_MAX_MONOPOLE_EXACT_WORK)
    k = _frequency_to_wavenumber(
        mono.frequency, _C0, "MonopoleExcitation")
    I_0 = -1im * 2π * mono.amplitude / _ETA0
    h = mono.height
    requires_exact = _monopole_incident_field_requires_exact(r, mono, k)

    if mono.include_image
        if requires_exact
            _monopole_below_ground_exact(r, mono, k) && return zero(CVec3)
        else
            d = r - mono.position
            proj = dot(d, mono.axis)
            ref_len = max(
                h, hypot(d[1], d[2], d[3]), 1.0)
            proj < -1e-12 * ref_len && return zero(CVec3)
        end
    end

    span_factor = mono.include_image ? 2.0 : 1.0
    minimum_intervals = mono.include_image ? 128 : 64
    N = _monopole_simpson_interval_count(
        h, k, span_factor, minimum_intervals,
        "monopole incident field")
    requires_exact &&
        return _monopole_incident_field_exact(
            r, mono, k, N, max_exact_work)

    if mono.include_image
        # Simpson on z' ∈ [-h, h]
        dz = (2.0 / N) * h
        E_sum = CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
        @inbounds for i in 0:N
            z_p = h * (-1.0 + 2.0 * (i / N))
            r_p = mono.position + z_p * mono.axis
            I_z = I_0 * sin(k * (h - abs(z_p)))
            w = (i == 0 || i == N) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
            E_sum = E_sum + w * _hertzian_per_length_field(r, r_p, mono.axis, I_z, k)
        end
        return _check_finite_field(
            (dz / 3.0) * E_sum,
            "monopole incident field",
        )
    else
        # physical half-wire only (no image): radiates into all 4π
        dz = h / N
        E_sum = CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
        @inbounds for i in 0:N
            z_p = h * (i / N)
            r_p = mono.position + z_p * mono.axis
            I_z = I_0 * sin(k * (h - z_p))
            w = (i == 0 || i == N) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
            E_sum = E_sum + w * _hertzian_per_length_field(r, r_p, mono.axis, I_z, k)
        end
        return _check_finite_field(
            (dz / 3.0) * E_sum,
            "monopole incident field",
        )
    end
end

@inline function _supported_incident_wavenumber_abs(k)
    real_magnitude = abs(real(k))
    imaginary_magnitude = abs(imag(k))
    (isfinite(real_magnitude) && isfinite(imaginary_magnitude) &&
     real_magnitude > 0.0) ||
        error(
            "compute_total_field: analytic incident-field models require a finite, " *
            "nonzero real free-space wavenumber; got k=$k.")
    relative_imaginary = imaginary_magnitude / real_magnitude
    relative_imaginary <= 1e-10 ||
        error("compute_total_field: analytic incident-field models require a real free-space wavenumber; got k=$k.")
    return real_magnitude
end

@inline function _check_incident_wavenumber_match(k_model::Real, k, label::AbstractString)
    k_abs = _supported_incident_wavenumber_abs(k)
    (isfinite(k_model) && k_model > 0.0) ||
        error(
            "compute_total_field: $label has an invalid model wavenumber $k_model.")
    larger = max(k_model, k_abs)
    smaller = min(k_model, k_abs)
    relative_error = (larger - smaller) / larger
    relative_error <= 1e-8 ||
        error(
            "compute_total_field: $label expects |k|=$k_model, got $k_abs " *
            "(relative error=$relative_error, tolerance=1.0e-8).")
    return nothing
end

function _validate_incident_electric_field_wavenumber(excitation::AbstractExcitation, k)
    error("compute_total_field: incident-field evaluation is not implemented for $(typeof(excitation)).")
end

function _validate_incident_electric_field_wavenumber(excitation::PlaneWaveExcitation, k)
    _validate_excitation_model(excitation)
    return _check_incident_wavenumber_match(norm(excitation.k_vec), k, "PlaneWaveExcitation")
end

function _validate_incident_electric_field_wavenumber(excitation::DipoleExcitation, k)
    _validate_excitation_model(excitation)
    k_model = _frequency_to_wavenumber(
        excitation.frequency, _C0, "DipoleExcitation")
    return _check_incident_wavenumber_match(k_model, k, "DipoleExcitation")
end

function _validate_incident_electric_field_wavenumber(excitation::LoopExcitation, k)
    _validate_excitation_model(excitation)
    k_model = _frequency_to_wavenumber(
        excitation.frequency, _C0, "LoopExcitation")
    return _check_incident_wavenumber_match(k_model, k, "LoopExcitation")
end

function _validate_incident_electric_field_wavenumber(excitation::MonopoleExcitation, k)
    _validate_excitation_model(excitation)
    k_model = _frequency_to_wavenumber(
        excitation.frequency, _C0, "MonopoleExcitation")
    return _check_incident_wavenumber_match(k_model, k, "MonopoleExcitation")
end

function _validate_incident_electric_field_wavenumber(excitation::PatternFeedExcitation, k)
    _validate_excitation_model(excitation)
    k_model = _frequency_to_wavenumber(
        excitation.frequency, _C0, "PatternFeedExcitation")
    return _check_incident_wavenumber_match(k_model, k, "PatternFeedExcitation")
end

function _validate_incident_electric_field_wavenumber(excitation::ImportedExcitation, k)
    _validate_excitation_model(excitation)
    if excitation.kind == :electric_field
        return nothing
    end
    error(
        "compute_total_field: ImportedExcitation(kind=:surface_current_density) is RHS-only " *
        "and does not define a rigorous observation-point incident electric field."
    )
end

function _validate_incident_electric_field_wavenumber(excitation::PortExcitation, k)
    error(
        "compute_total_field: PortExcitation is an excitation-vector surrogate, not a " *
        "pointwise incident-field model."
    )
end

function _validate_incident_electric_field_wavenumber(excitation::DeltaGapExcitation, k)
    error(
        "compute_total_field: DeltaGapExcitation is an excitation-vector surrogate, not a " *
        "pointwise incident-field model."
    )
end

function _validate_incident_electric_field_wavenumber(excitation::MultiExcitation, k)
    _validate_excitation_model(excitation)
    for (i, exc) in enumerate(excitation.excitations)
        try
            _validate_incident_electric_field_wavenumber(exc, k)
        catch err
            error(
                "compute_total_field: MultiExcitation child $i of type $(typeof(exc)) " *
                "failed incident-field validation: $(sprint(showerror, err))"
            )
        end
    end
    return nothing
end

function _incident_electric_field(excitation::AbstractExcitation, r::Vec3, k)
    error("compute_total_field: incident-field evaluation is not implemented for $(typeof(excitation)).")
end

function _incident_electric_field(excitation::PlaneWaveExcitation, r::Vec3, k)
    return _plane_wave_field_unchecked(
        r, excitation.k_vec, excitation.E0, excitation.pol)
end

function _incident_electric_field(excitation::DipoleExcitation, r::Vec3, k)
    return _dipole_incident_field_unchecked(r, excitation)
end

function _incident_electric_field(excitation::LoopExcitation, r::Vec3, k)
    return _loop_incident_field_unchecked(r, excitation)
end

function _incident_electric_field(excitation::MonopoleExcitation, r::Vec3, k)
    return _monopole_incident_field_unchecked(r, excitation)
end

function _incident_electric_field(excitation::PatternFeedExcitation, r::Vec3, k)
    return _pattern_feed_field_unchecked(r, excitation)
end

function _incident_electric_field(excitation::ImportedExcitation, r::Vec3, k)
    if excitation.kind == :electric_field
        return _to_cvec3(excitation.source_func(r), "ImportedExcitation source function")
    end
    error(
        "compute_total_field: ImportedExcitation(kind=:surface_current_density) is RHS-only " *
        "and does not define a rigorous observation-point incident electric field."
    )
end

function _incident_electric_field(excitation::PortExcitation, r::Vec3, k)
    error(
        "compute_total_field: PortExcitation is an excitation-vector surrogate, not a " *
        "pointwise incident-field model."
    )
end

function _incident_electric_field(excitation::DeltaGapExcitation, r::Vec3, k)
    error(
        "compute_total_field: DeltaGapExcitation is an excitation-vector surrogate, not a " *
        "pointwise incident-field model."
    )
end

@noinline function _multi_incident_electric_field_exact(
        excitation::MultiExcitation, r::Vec3, k)
    return setprecision(BigFloat, _MULTI_SOURCE_EXACT_PRECISION) do
        total = Complex{BigFloat}[
            zero(Complex{BigFloat}),
            zero(Complex{BigFloat}),
            zero(Complex{BigFloat}),
        ]
        @inbounds for index in eachindex(
                excitation.excitations, excitation.weights)
            child = try
                _incident_electric_field(
                    excitation.excitations[index], r, k)
            catch err
                error(
                    "compute_total_field: MultiExcitation child $index of " *
                    "type $(typeof(excitation.excitations[index])) failed " *
                    "incident-field evaluation: $(sprint(showerror, err))"
                )
            end
            weight = Complex{BigFloat}(excitation.weights[index])
            for component in 1:3
                total[component] +=
                    weight * Complex{BigFloat}(child[component])
            end
        end
        return _finite_source_vector(
            total, "compute_total_field MultiExcitation incident field")
    end
end

function _incident_electric_field(excitation::MultiExcitation, r::Vec3, k)
    length(excitation.excitations) == length(excitation.weights) ||
        error("compute_total_field: MultiExcitation has mismatched excitation/weight lengths.")
    E = CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    for (i, (exc, w)) in enumerate(zip(excitation.excitations, excitation.weights))
        child = (try
            _incident_electric_field(exc, r, k)
        catch err
            error(
                "compute_total_field: MultiExcitation child $i of type $(typeof(exc)) " *
                "failed incident-field evaluation: $(sprint(showerror, err))"
            )
        end)::CVec3
        _source_scaled_vector_requires_fallback(w, child) &&
            return _multi_incident_electric_field_exact(excitation, r, k)
        contribution = w * child
        @inbounds for component in 1:3
            _source_product_requires_exact(
                w, child[component], contribution[component]) &&
                return _multi_incident_electric_field_exact(excitation, r, k)
        end
        updated = E + contribution
        (all(isfinite, updated) &&
         !_source_vector_sum_requires_exact(E, contribution, updated)) ||
            return _multi_incident_electric_field_exact(excitation, r, k)
        E = updated
    end
    return E
end

"""
    assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol; quad_order=3)

Assemble the excitation vector v_m = -⟨f_m, E^inc_t⟩ for a plane wave.
"""
function assemble_v_plane_wave(mesh::TriMesh, rwg::RWGData,
                                k_vec::Vec3, E0, pol::Vec3;
                                quad_order::Int=3)
    # For backward compatibility, create PlaneWaveExcitation and call assemble_excitation
    pw = PlaneWaveExcitation(k_vec, E0, pol)
    return assemble_excitation(mesh, rwg, pw; quad_order=quad_order)
end

"""
    assemble_excitation(mesh, rwg, excitation;
                        quad_order=3,
                        max_exact_bytes=536_870_912)

Assemble RHS vector v for given excitation.

`max_exact_bytes` bounds the retained high-precision accumulator used only
when a `MultiExcitation` needs overflow, underflow, or cancellation recovery.
"""
function assemble_excitation(mesh::TriMesh, rwg::RWGData,
                             excitation::AbstractExcitation;
                             quad_order::Int=3,
                             quad_cache::Union{Nothing,ExcitationQuadCache}=nothing,
                             max_exact_bytes::Integer=
                                 _DEFAULT_MAX_MULTI_EXACT_BYTES)
    _validate_mesh_rwg_pair(mesh, rwg)
    _validate_excitation_model(excitation)
    exact_byte_limit =
        _validated_resource_limit("max_exact_bytes", max_exact_bytes)
    # Dispatch to specialized implementations. `quad_cache`, when provided,
    # lets the mesh quadrature be shared across many excitations (see
    # `assemble_multi`/`assemble_multiple_excitations`); the cache-free
    # quantities (delta gap, port) ignore it.
    v = if excitation isa PlaneWaveExcitation
        assemble_plane_wave(mesh, rwg, excitation; quad_order=quad_order, quad_cache=quad_cache)
    elseif excitation isa PortExcitation
        assemble_port(mesh, rwg, excitation)
    elseif excitation isa DeltaGapExcitation
        assemble_delta_gap(mesh, rwg, excitation)
    elseif excitation isa DipoleExcitation
        assemble_dipole(mesh, rwg, excitation; quad_order=quad_order, quad_cache=quad_cache)
    elseif excitation isa LoopExcitation
        assemble_loop(mesh, rwg, excitation; quad_order=quad_order, quad_cache=quad_cache)
    elseif excitation isa MonopoleExcitation
        assemble_monopole(mesh, rwg, excitation; quad_order=quad_order, quad_cache=quad_cache)
    elseif excitation isa ImportedExcitation
        assemble_imported_excitation(mesh, rwg, excitation; quad_order=quad_order, quad_cache=quad_cache)
    elseif excitation isa PatternFeedExcitation
        assemble_pattern_feed(mesh, rwg, excitation; quad_order=quad_order, quad_cache=quad_cache)
    elseif excitation isa MultiExcitation
        assemble_multi(
            mesh, rwg, excitation;
            quad_order=quad_order,
            quad_cache=quad_cache,
            max_exact_bytes=exact_byte_limit)
    else
        error("Unsupported excitation type: $(typeof(excitation))")
    end
    all(isfinite, v) ||
        error(
            "Excitation assembly for $(typeof(excitation)) produced non-finite RHS values.")
    return v
end

"""
    assemble_multiple_excitations(mesh, rwg, excitations;
                                  quad_order=3,
                                  max_output_bytes=2_000_000_000,
                                  max_work_bytes=536_870_912,
                                  max_terms=200_000_000)

Assemble RHS matrix V where each column corresponds to an excitation.
"""
function assemble_multiple_excitations(mesh::TriMesh, rwg::RWGData,
                                       excitations::Vector{<:AbstractExcitation};
                                       quad_order::Int=3,
                                       max_output_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
                                       max_work_bytes::Integer=
                                           _DEFAULT_MAX_EXCITATION_WORK_BYTES,
                                       max_terms::Integer=
                                           _DEFAULT_MAX_EXCITATION_TERMS)
    _validate_mesh_rwg_pair(mesh, rwg)
    # Public excitation structs can be constructed directly, so validate every
    # column before sizing or allocating the potentially large dense result.
    # `assemble_excitation` validates again to keep its standalone contract.
    for excitation in excitations
        _validate_excitation_model(excitation)
    end
    N = rwg.nedges
    M = length(excitations)
    output_bytes = _checked_array_payload_bytes(
        ComplexF64, N, M;
        label="multiple-excitation RHS matrix")
    _enforce_payload_limit(
        output_bytes, max_output_bytes,
        "multiple-excitation RHS matrix", "max_output_bytes")
    # Validate quadrature and all resource requests before allocating the
    # shared cache or output.  This conservative bound charges every child as
    # a surface-quadrature excitation; cache-free children only use less work.
    _, quadrature_weights = tri_quad_rule(quad_order)
    requested_points = length(quadrature_weights)
    quadrature_orders, temporary_rhs_count, term_points =
        _multiple_excitation_profile(
            excitations, quad_order, requested_points)
    quadrature_point_counts = (
        length(last(tri_quad_rule(order))) for order in quadrature_orders)
    work_bytes = _multiple_excitation_profile_work_bytes(
        N, ntriangles(mesh), quadrature_point_counts,
        temporary_rhs_count, output_bytes)
    work_limit = _validated_resource_limit(
        "max_work_bytes", max_work_bytes)
    work_bytes <= work_limit ||
        throw(ArgumentError(
            "multiple-excitation output and workspace requires " *
            "$work_bytes raw bytes, exceeding max_work_bytes=$work_limit"))
    available_exact_bytes = max(1, work_limit - work_bytes)
    term_count = _multiple_excitation_term_count(N, term_points)
    term_limit = _validated_resource_limit("max_terms", max_terms)
    term_count <= term_limit ||
        throw(ArgumentError(
            "multiple-excitation assembly requires $term_count terms, " *
            "exceeding max_terms=$term_limit"))
    V = zeros(ComplexF64, N, M)
    # Build the mesh quadrature once and share it across all excitations.
    quad_cache = ExcitationQuadCache(mesh)
    for m in 1:M
        V[:, m] .= assemble_excitation(mesh, rwg, excitations[m];
                                       quad_order=quad_order,
                                       quad_cache=quad_cache,
                                       max_exact_bytes=available_exact_bytes)
    end
    return V
end

# ------------------------------------------------------------------
# Specialized assembly functions
# ------------------------------------------------------------------

@noinline function _excitation_surface_term_exact(
    basis_value::SVector{3,<:Number},
    incident_field::SVector{3,<:Number},
    area::Float64,
    quadrature_weight::Float64,
)
    return setprecision(
            BigFloat, _EXCITATION_SURFACE_FALLBACK_PRECISION) do
        inner_product = zero(Complex{BigFloat})
        @inbounds for component in 1:3
            inner_product += conj(Complex{BigFloat}(
                                 basis_value[component])) *
                             Complex{BigFloat}(
                                 incident_field[component])
        end
        converted = ComplexF64(
            -2 * BigFloat(quadrature_weight) * BigFloat(area) *
            inner_product)
        isfinite(converted) ||
            throw(OverflowError(
                "excitation surface integral term is outside the " *
                "ComplexF64 range"))
        return converted
    end
end

@inline function _excitation_surface_term(
    basis_value::SVector{3,<:Number},
    incident_field::SVector{3,<:Number},
    area::Float64,
    quadrature_weight::Float64,
)
    first = conj(basis_value[1]) * incident_field[1]
    second = conj(basis_value[2]) * incident_field[2]
    third = conj(basis_value[3]) * incident_field[3]
    products_preserve_range =
        isfinite(first) && isfinite(second) && isfinite(third) &&
        (!iszero(first) || iszero(basis_value[1]) ||
         iszero(incident_field[1])) &&
        (!iszero(second) || iszero(basis_value[2]) ||
         iszero(incident_field[2])) &&
        (!iszero(third) || iszero(basis_value[3]) ||
         iszero(incident_field[3]))
    products_preserve_range ||
        return _excitation_surface_term_exact(
            basis_value, incident_field, area, quadrature_weight)

    inner_product = first + second + third
    isfinite(inner_product) ||
        return _excitation_surface_term_exact(
            basis_value, incident_field, area, quadrature_weight)
    area_product = area * inner_product
    weighted = -(2 * quadrature_weight) * area_product
    if isfinite(weighted) &&
       (!iszero(area_product) || iszero(area) || iszero(inner_product)) &&
       (!iszero(weighted) || iszero(quadrature_weight) ||
        iszero(area_product))
        return ComplexF64(weighted)
    end
    return _excitation_surface_term_exact(
        basis_value, incident_field, area, quadrature_weight)
end

function assemble_plane_wave(mesh::TriMesh, rwg::RWGData,
                             pw::PlaneWaveExcitation;
                             quad_order::Int=3,
                             quad_cache::Union{Nothing,ExcitationQuadCache}=nothing)
    _validate_plane_wave_excitation(pw)
    N = rwg.nedges
    wq, quad_pts, areas = quad_cache === nothing ?
        _excitation_quadrature_cache(mesh, quad_order) :
        _quad_cache_for(quad_cache, mesh, quad_order)
    Nq = length(wq)

    CT = ComplexF64
    v = zeros(CT, N)

    for n in 1:N
        for t in (rwg.tplus[n], rwg.tminus[n])
            A = areas[t]
            pts = quad_pts[t]

            for q in 1:Nq
                rq = pts[q]
                fn = eval_rwg(rwg, n, rq, t)
                Einc = _plane_wave_field_unchecked(
                    rq, pw.k_vec, pw.E0, pw.pol)
                v[n] += _excitation_surface_term(fn, Einc, A, wq[q])
            end
        end
    end

    return v
end

function assemble_port(mesh::TriMesh, rwg::RWGData, port::PortExcitation)
    @inbounds for e in port.port_edges
        1 <= e <= rwg.nedges ||
            throw(ArgumentError(
                "PortExcitation edge $e is outside the valid RWG range 1:$(rwg.nedges)."))
    end
    v = zeros(ComplexF64, rwg.nedges)
    # Simple delta-gap approximation across port edges
    for e in port.port_edges
        edge_len = rwg.len[e]
        v[e] = port.voltage / edge_len  # Approximate as uniform field across edge
    end
    return v
end

function assemble_delta_gap(mesh::TriMesh, rwg::RWGData, gap::DeltaGapExcitation)
    v = zeros(ComplexF64, rwg.nedges)
    if 1 <= gap.edge <= rwg.nedges
        gap.gap_length > 0 || error("Delta-gap length must be positive.")
        v[gap.edge] = gap.voltage / gap.gap_length
    else
        error("Delta-gap edge $(gap.edge) is out of bounds (1-$(rwg.nedges))")
    end
    return v
end

function assemble_dipole(mesh::TriMesh, rwg::RWGData,
                         dipole::DipoleExcitation;
                         quad_order::Int=3,
                         quad_cache::Union{Nothing,ExcitationQuadCache}=nothing)
    N = rwg.nedges
    wq, quad_pts, areas = quad_cache === nothing ?
        _excitation_quadrature_cache(mesh, quad_order) :
        _quad_cache_for(quad_cache, mesh, quad_order)
    Nq = length(wq)

    v = zeros(ComplexF64, N)

    for n in 1:N
        for t in (rwg.tplus[n], rwg.tminus[n])
            A = areas[t]
            pts = quad_pts[t]

            for q in 1:Nq
                rq = pts[q]
                fn = eval_rwg(rwg, n, rq, t)

                Einc = _dipole_incident_field_unchecked(rq, dipole)

                v[n] += _excitation_surface_term(fn, Einc, A, wq[q])
            end
        end
    end

    return v
end

function assemble_loop(mesh::TriMesh, rwg::RWGData,
                       loop::LoopExcitation;
                       quad_order::Int=3,
                       quad_cache::Union{Nothing,ExcitationQuadCache}=nothing)
    N = rwg.nedges
    wq, quad_pts, areas = quad_cache === nothing ?
        _excitation_quadrature_cache(mesh, quad_order) :
        _quad_cache_for(quad_cache, mesh, quad_order)
    Nq = length(wq)

    v = zeros(ComplexF64, N)
    for n in 1:N
        for t in (rwg.tplus[n], rwg.tminus[n])
            A = areas[t]
            pts = quad_pts[t]
            for q in 1:Nq
                rq = pts[q]
                fn = eval_rwg(rwg, n, rq, t)
                Einc = _loop_incident_field_unchecked(rq, loop)
                v[n] += _excitation_surface_term(fn, Einc, A, wq[q])
            end
        end
    end
    return v
end

function assemble_monopole(mesh::TriMesh, rwg::RWGData,
                           mono::MonopoleExcitation;
                           quad_order::Int=3,
                           quad_cache::Union{Nothing,ExcitationQuadCache}=nothing)
    N = rwg.nedges
    wq, quad_pts, areas = quad_cache === nothing ?
        _excitation_quadrature_cache(mesh, quad_order) :
        _quad_cache_for(quad_cache, mesh, quad_order)
    Nq = length(wq)

    v = zeros(ComplexF64, N)
    for n in 1:N
        for t in (rwg.tplus[n], rwg.tminus[n])
            A = areas[t]
            pts = quad_pts[t]
            for q in 1:Nq
                rq = pts[q]
                fn = eval_rwg(rwg, n, rq, t)
                Einc = _monopole_incident_field_unchecked(rq, mono)
                v[n] += _excitation_surface_term(fn, Einc, A, wq[q])
            end
        end
    end
    return v
end

function assemble_imported_excitation(mesh::TriMesh, rwg::RWGData,
                                      imported::ImportedExcitation;
                                      quad_order::Int=3,
                                      quad_cache::Union{Nothing,ExcitationQuadCache}=nothing)
    quad_order_eff = _effective_quad_order(quad_order, imported.min_quad_order)
    N = rwg.nedges
    wq, quad_pts, areas = quad_cache === nothing ?
        _excitation_quadrature_cache(mesh, quad_order_eff) :
        _quad_cache_for(quad_cache, mesh, quad_order_eff)
    Nq = length(wq)

    v = zeros(ComplexF64, N)

    for n in 1:N
        for t in (rwg.tplus[n], rwg.tminus[n])
            A = areas[t]
            pts = quad_pts[t]

            for q in 1:Nq
                rq = pts[q]
                fn = eval_rwg(rwg, n, rq, t)
                src = _to_cvec3(imported.source_func(rq), "ImportedExcitation source function")
                Einc = if imported.kind == :electric_field
                    src
                else
                    imported.eta_equiv * src
                end
                Einc = _check_finite_cvec3(
                    Einc, "ImportedExcitation mapped electric field")
                v[n] += _excitation_surface_term(fn, Einc, A, wq[q])
            end
        end
    end

    return v
end

function assemble_pattern_feed(mesh::TriMesh, rwg::RWGData,
                               pat::PatternFeedExcitation;
                               quad_order::Int=3,
                               quad_cache::Union{Nothing,ExcitationQuadCache}=nothing)
    N = rwg.nedges
    wq, quad_pts, areas = quad_cache === nothing ?
        _excitation_quadrature_cache(mesh, quad_order) :
        _quad_cache_for(quad_cache, mesh, quad_order)
    Nq = length(wq)

    v = zeros(ComplexF64, N)

    for n in 1:N
        for t in (rwg.tplus[n], rwg.tminus[n])
            A = areas[t]
            pts = quad_pts[t]

            for q in 1:Nq
                rq = pts[q]
                fn = eval_rwg(rwg, n, rq, t)
                Einc = _pattern_feed_field_unchecked(rq, pat)
                v[n] += _excitation_surface_term(fn, Einc, A, wq[q])
            end
        end
    end

    return v
end

@inline function _multi_assembly_child_requires_exact(
        weight::ComplexF64,
        child::AbstractVector{ComplexF64},
        output::AbstractVector{ComplexF64})
    iszero(weight) && return false
    weight_is_extreme = _source_scaling_extreme_value(weight)
    @inbounds for index in eachindex(child, output)
        value = child[index]
        iszero(value) && continue
        (weight_is_extreme || _source_scaling_extreme_value(value)) &&
            return true
        product = weight * value
        (!isfinite(product) || iszero(product) ||
         _source_product_requires_exact(weight, value, product)) &&
            return true
        combined = output[index] + product
        (!isfinite(combined) ||
         _scaled_sum_requires_exact(output[index], product, combined)) &&
            return true
    end
    return false
end

@noinline function _assemble_multi_exact!(
        output::Vector{ComplexF64}, mesh::TriMesh, rwg::RWGData,
        multi::MultiExcitation, quad_order::Int,
        cache::ExcitationQuadCache, max_exact_bytes::Int)
    required_bytes = _multi_exact_accumulator_bytes(length(output))
    _enforce_payload_limit(
        required_bytes, max_exact_bytes,
        "MultiExcitation exact accumulator", "max_exact_bytes")
    nested_exact_bytes = max(1, max_exact_bytes - required_bytes)

    return setprecision(BigFloat, _MULTI_SOURCE_EXACT_PRECISION) do
        total = [zero(Complex{BigFloat}) for _ in eachindex(output)]
        @inbounds for index in eachindex(
                multi.excitations, multi.weights)
            child = assemble_excitation(
                mesh, rwg, multi.excitations[index];
                quad_order=quad_order,
                quad_cache=cache,
                max_exact_bytes=nested_exact_bytes)
            weight = Complex{BigFloat}(multi.weights[index])
            for entry in eachindex(total, child)
                total[entry] +=
                    weight * Complex{BigFloat}(child[entry])
            end
        end
        @inbounds for entry in eachindex(output, total)
            converted = ComplexF64(total[entry])
            isfinite(converted) ||
                throw(OverflowError(
                    "MultiExcitation RHS entry $entry is outside the " *
                    "representable ComplexF64 range"))
            output[entry] = converted
        end
        return output
    end
end

function assemble_multi(mesh::TriMesh, rwg::RWGData,
                        multi::MultiExcitation;
                        quad_order::Int=3,
                        quad_cache::Union{Nothing,ExcitationQuadCache}=nothing,
                        max_exact_bytes::Integer=
                            _DEFAULT_MAX_MULTI_EXACT_BYTES)
    length(multi.excitations) == length(multi.weights) ||
        error("MultiExcitation has mismatched lengths: $(length(multi.excitations)) excitations vs $(length(multi.weights)) weights.")
    # Build (or reuse) the mesh quadrature once and share it across all children.
    cache = quad_cache === nothing ? ExcitationQuadCache(mesh) : quad_cache
    exact_byte_limit =
        _validated_resource_limit("max_exact_bytes", max_exact_bytes)
    v = zeros(ComplexF64, rwg.nedges)
    for (exc, w) in zip(multi.excitations, multi.weights)
        child = assemble_excitation(
            mesh, rwg, exc;
            quad_order=quad_order,
            quad_cache=cache,
            max_exact_bytes=exact_byte_limit)
        _multi_assembly_child_requires_exact(w, child, v) &&
            return _assemble_multi_exact!(
                v, mesh, rwg, multi, quad_order, cache,
                exact_byte_limit)
        axpy!(w, child, v)
        all(isfinite, v) ||
            return _assemble_multi_exact!(
                v, mesh, rwg, multi, quad_order, cache,
                exact_byte_limit)
    end
    return v
end
