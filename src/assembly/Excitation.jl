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
make_plane_wave(k_vec, E0, pol) = PlaneWaveExcitation(k_vec, E0, pol)

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
        axv = Vec3(axis)
        n = norm(axv)
        n > 1e-12 || error("MonopoleExcitation: axis must be nonzero")
        height > 0 || error("MonopoleExcitation: height must be positive, got $height")
        frequency > 0 || error("MonopoleExcitation: frequency must be positive, got $frequency")
        return new(Vec3(pos), axv / n, Float64(height),
                   ComplexF64(amplitude), Float64(frequency), include_image)
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
struct ImportedExcitation <: AbstractExcitation
    source_func::Function         # source(r) -> CVec3
    kind::Symbol                  # :electric_field or :surface_current_density
    eta_equiv::ComplexF64         # used only for :surface_current_density
    min_quad_order::Int           # minimum quadrature order target
    function ImportedExcitation(source_func::Function;
                                kind::Symbol=:electric_field,
                                eta_equiv::Number=376.730313668 + 0im,
                                min_quad_order::Integer=3)
        min_quad_order >= 1 || error("min_quad_order must be >= 1, got $min_quad_order.")
        kind in (:electric_field, :surface_current_density) ||
            error("Unsupported ImportedExcitation kind: $kind")
        return new(source_func, kind, ComplexF64(eta_equiv), Int(min_quad_order))
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
    if length(excitations) != length(weights)
        error("MultiExcitation requires matching lengths: $(length(excitations)) excitations, $(length(weights)) weights.")
    end
    return MultiExcitation(excitations, weights)
end

# Helper functions
make_delta_gap(edge, voltage, gap_length) = DeltaGapExcitation(edge, voltage, gap_length)
make_dipole(position, moment, orientation, type, frequency=1e9) = DipoleExcitation(position, moment, orientation, type, frequency)
make_loop(center, normal, radius, current, frequency=1e9) = LoopExcitation(center, normal, radius, current, frequency)
make_monopole(position, axis, height, amplitude, frequency=1e9;
              include_image::Bool = true) =
    MonopoleExcitation(position, axis, height, amplitude, frequency;
                       include_image=include_image)

const _C0 = 299792458.0
const _EPS0 = 8.854187817e-12
const _MU0 = 4π * 1e-7
const _ETA0 = sqrt(_MU0 / _EPS0)

function _validate_pattern_grid(theta::Vector{Float64}, phi::Vector{Float64})
    length(theta) ≥ 2 || error("Pattern feed requires at least 2 theta samples.")
    length(phi) ≥ 2 || error("Pattern feed requires at least 2 phi samples.")
    all(isfinite, theta) || error("Theta grid contains non-finite values.")
    all(isfinite, phi) || error("Phi grid contains non-finite values.")
    all(diff(theta) .> 0) || error("Theta grid must be strictly increasing.")
    all(diff(phi) .> 0) || error("Phi grid must be strictly increasing.")

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
    M = ComplexF64.(Matrix(U))
    if size(M) == (nθ, nϕ)
        return M
    elseif size(M) == (nϕ, nθ)
        @warn "$name matrix shape $(size(M)) appears transposed; auto-transposing to ($nθ, $nϕ)."
        return permutedims(M)
    else
        error("$name matrix shape $(size(M)) is incompatible with grids (Nθ=$nθ, Nϕ=$nϕ).")
    end
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
                           convention::Symbol=:exp_plus_iwt)
    θ = Float64.(collect(theta))
    ϕ = Float64.(collect(phi))
    if angles_in_degrees
        θ .*= π / 180
        ϕ .*= π / 180
    end

    _validate_pattern_grid(θ, ϕ)
    frequency > 0 || error("Pattern feed frequency must be positive.")
    convention in (:exp_plus_iwt, :exp_minus_iwt) ||
        error("Unsupported pattern convention: $convention (expected :exp_plus_iwt or :exp_minus_iwt).")

    Fθ = _coerce_pattern_matrix(Ftheta, length(θ), length(ϕ), "Ftheta")
    Fϕ = _coerce_pattern_matrix(Fphi, length(θ), length(ϕ), "Fphi")

    return PatternFeedExcitation(θ, ϕ, Fθ, Fϕ, Float64(frequency), phase_center, convention)
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
                           convention::Symbol=:exp_plus_iwt)
    θ = getproperty(Etheta_pattern, :x)
    ϕ = getproperty(Etheta_pattern, :y)
    θ2 = getproperty(Ephi_pattern, :x)
    ϕ2 = getproperty(Ephi_pattern, :y)
    if length(θ) != length(θ2) || length(ϕ) != length(ϕ2)
        error("Etheta and Ephi patterns must share the same theta/phi grids.")
    end
    length(θ) > 0 || error("Pattern grid theta is empty.")
    length(ϕ) > 0 || error("Pattern grid phi is empty.")
    θf = Float64.(θ)
    ϕf = Float64.(ϕ)
    θ2f = Float64.(θ2)
    ϕ2f = Float64.(ϕ2)
    if maximum(abs.(θf .- θ2f)) > 1e-12 || maximum(abs.(ϕf .- ϕ2f)) > 1e-12
        error("Etheta and Ephi patterns must share the same theta/phi grids.")
    end
    Fθ = getproperty(Etheta_pattern, :U)
    Fϕ = getproperty(Ephi_pattern, :U)
    return make_pattern_feed(θ, ϕ, Fθ, Fϕ, frequency;
                             phase_center=phase_center,
                             angles_in_degrees=angles_in_degrees,
                             convention=convention)
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

@inline function _bracket_periodic_phi(phi::Vector{Float64}, ϕ::Float64)
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

"""
    pattern_feed_field(r, pat)

Evaluate the incident electric field at position `r` generated by
`PatternFeedExcitation`.
"""
function pattern_feed_field(r::Vec3, pat::PatternFeedExcitation)
    pat.frequency > 0 || error("Pattern feed frequency must be positive.")
    Rvec = r - pat.phase_center
    R = norm(Rvec)
    if R < 1e-12
        return CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    end

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

    k = 2π * pat.frequency / _C0
    phase = exp(-1im * k * R) / R
    return phase * (Fθ * eθ + Fϕ * eϕ)
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
                                           convention::Symbol=:exp_plus_iwt)
    dipole.frequency > 0 || error("Dipole frequency must be positive.")
    θ = Float64.(collect(theta))
    ϕ = Float64.(collect(phi))
    if angles_in_degrees
        θ .*= π / 180
        ϕ .*= π / 180
    end
    _validate_pattern_grid(θ, ϕ)
    dipole.type in (:electric, :magnetic) ||
        error("Dipole type must be :electric or :magnetic, got $(dipole.type).")

    nθ = length(θ)
    nϕ = length(ϕ)
    Fθ = zeros(ComplexF64, nθ, nϕ)
    Fϕ = zeros(ComplexF64, nθ, nϕ)

    k = 2π * dipole.frequency / _C0
    p = dipole.moment

    for i in 1:nθ
        for j in 1:nϕ
            rhat, eθ, eϕ = _spherical_basis(θ[i], ϕ[j])
            Fvec = if dipole.type == :electric
                (k^2 / (4π * _EPS0)) * cross(rhat, cross(p, rhat))
            else
                1im * _ETA0 * (k^2 / (4π)) * cross(p, rhat)
            end
            Fθ[i, j] = dot(Fvec, eθ)
            Fϕ[i, j] = dot(Fvec, eϕ)
        end
    end

    return make_pattern_feed(θ, ϕ, Fθ, Fϕ, dipole.frequency;
                             phase_center=phase_center,
                             angles_in_degrees=false,
                             convention=convention)
end

"""
    plane_wave_field(r, k_vec, E0, pol)

Evaluate a plane wave E^inc(r) = pol * E0 * exp(-i k_vec · r)
at point r. Convention: exp(+iωt).
"""
function plane_wave_field(r::Vec3, k_vec::Vec3, E0, pol::Vec3)
    return pol * E0 * exp(-1im * dot(k_vec, r))
end

const _TRI_QUAD_ORDERS = (1, 3, 4, 7)

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
    target = max(requested, min_required)
    for q in _TRI_QUAD_ORDERS
        q >= target && return q
    end
    return last(_TRI_QUAD_ORDERS)
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

"""
    dipole_incident_field(r, dipole)

Evaluate incident electric field from electric or magnetic dipole at point `r`.
Uses the `exp(+iωt)` convention and free-space Green-function fields.
"""
function dipole_incident_field(r::Vec3, dipole::DipoleExcitation)
    ϵ0 = 8.854187817e-12
    μ0 = 4π * 1e-7
    η0 = sqrt(μ0 / ϵ0)
    c0 = 1 / sqrt(ϵ0 * μ0)

    dipole.frequency > 0 || error("Dipole frequency must be positive.")

    k = 2π * dipole.frequency / c0

    R_vec = r - dipole.position
    R = norm(R_vec)
    if R < 1e-12
        return CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    end

    R_hat = R_vec / R
    p = dipole.moment

    if dipole.type == :electric
        term1 = cross(R_hat, p)
        term1 = cross(term1, R_hat) * k^2
        term2 = (3 * R_hat * dot(R_hat, p) - p) * (1 / R^2 + 1im * k / R)
        return (term1 + term2) * exp(-1im * k * R) / (4π * ϵ0 * R)
    elseif dipole.type == :magnetic
        # E = (η0/4π) (k/R^2 + i k^2/R) e^{-ikR} (m × R̂)   [Balanis Ch. 5]
        return (η0 / (4π)) * (k / R^2 + 1im * k^2 / R) * exp(-1im * k * R) * cross(p, R_hat)
    else
        error("Dipole type must be :electric or :magnetic, got $(dipole.type).")
    end
end

"""
    loop_incident_field(r, loop)

Evaluate electric field from a small loop source via equivalent magnetic dipole.
"""
function loop_incident_field(r::Vec3, loop::LoopExcitation)
    m = loop.current * π * loop.radius^2 * loop.normal
    dip = DipoleExcitation(loop.center, m, loop.normal, :magnetic, loop.frequency)
    return dipole_incident_field(r, dip)
end

# Hertzian current element (Balanis eq 4-56a,b, exp(+iωt) convention).
# Returns dE/dz' for a unit-length axial element carrying current `I` at `r_p`.
@inline function _hertzian_per_length_field(r::Vec3, r_p::Vec3, axis::Vec3,
                                            I::ComplexF64, k::Float64)
    η0 = 376.730313668
    R_vec = r - r_p
    R = norm(R_vec)
    R < 1e-12 && return CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    R_hat = R_vec / R
    cosθ = clamp(dot(R_hat, axis), -1.0, 1.0)
    sin2θ = max(0.0, 1.0 - cosθ^2)
    sinθ = sqrt(sin2θ)
    expfac = exp(-1im * k * R)
    kR = k * R

    E_r = η0 * I / (2π) * cosθ * expfac / R^2 * (1.0 + 1.0 / (1im * kR))
    if sinθ > 1e-12
        θ_hat = (cosθ * R_hat - axis) / sinθ
        E_θ = 1im * η0 * k * I / (4π) * sinθ * expfac / R *
              (1.0 + 1.0 / (1im * kR) - 1.0 / kR^2)
        return E_r * R_hat + E_θ * θ_hat
    else
        return E_r * R_hat
    end
end

"""
    monopole_incident_field(r, mono)

Incident electric field at `r` from a `MonopoleExcitation`, computed by
rigorous Simpson integration of the sinusoidal current on the equivalent
dipole-plus-image distribution. Uses the exp(+iωt) convention. Returns zero
below the ground plane (axial projection of `r − position` is negative).
"""
function monopole_incident_field(r::Vec3, mono::MonopoleExcitation)
    k = 2π * mono.frequency / _C0
    I_0 = -1im * 2π * mono.amplitude / _ETA0
    λ = 2π / k
    h = mono.height

    if mono.include_image
        # dipole+image model: lower half-space is a true void
        d = r - mono.position
        proj = dot(d, mono.axis)
        ref_len = max(h, norm(d), 1.0)
        if proj < -1e-12 * ref_len
            return CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
        end
        # Simpson on z' ∈ [-h, h]
        N = max(128, 2 * Int(ceil(50.0 * 2 * h / λ)))
        iseven(N) || (N += 1)
        dz = 2h / N
        E_sum = CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
        @inbounds for i in 0:N
            z_p = -h + i * dz
            r_p = mono.position + z_p * mono.axis
            I_z = I_0 * sin(k * (h - abs(z_p)))
            w = (i == 0 || i == N) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
            E_sum = E_sum + w * _hertzian_per_length_field(r, r_p, mono.axis, I_z, k)
        end
        return (dz / 3.0) * E_sum
    else
        # physical half-wire only (no image): radiates into all 4π
        N = max(64, 2 * Int(ceil(50.0 * h / λ)))
        iseven(N) || (N += 1)
        dz = h / N
        E_sum = CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
        @inbounds for i in 0:N
            z_p = i * dz
            r_p = mono.position + z_p * mono.axis
            I_z = I_0 * sin(k * (h - z_p))
            w = (i == 0 || i == N) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
            E_sum = E_sum + w * _hertzian_per_length_field(r, r_p, mono.axis, I_z, k)
        end
        return (dz / 3.0) * E_sum
    end
end

@inline function _supported_incident_wavenumber_abs(k)
    tol_im = max(1e-10 * max(abs(real(k)), 1.0), 1e-12)
    abs(imag(k)) <= tol_im ||
        error("compute_total_field: analytic incident-field models require a real free-space wavenumber; got k=$k.")
    return abs(real(k))
end

@inline function _incident_wavenumber_match_tol(k)
    return max(1e-8 * max(abs(real(k)), 1.0), 1e-12)
end

@inline function _check_incident_wavenumber_match(k_model::Real, k, label::AbstractString)
    k_abs = _supported_incident_wavenumber_abs(k)
    tol = _incident_wavenumber_match_tol(k)
    abs(k_model - k_abs) <= tol ||
        error("compute_total_field: $label expects |k|=$k_model, got $(abs(real(k))) (tol=$tol).")
    return nothing
end

function _validate_incident_electric_field_wavenumber(excitation::AbstractExcitation, k)
    error("compute_total_field: incident-field evaluation is not implemented for $(typeof(excitation)).")
end

function _validate_incident_electric_field_wavenumber(excitation::PlaneWaveExcitation, k)
    return _check_incident_wavenumber_match(norm(excitation.k_vec), k, "PlaneWaveExcitation")
end

function _validate_incident_electric_field_wavenumber(excitation::DipoleExcitation, k)
    excitation.frequency > 0 || error("Dipole frequency must be positive.")
    k_model = 2π * excitation.frequency / _C0
    return _check_incident_wavenumber_match(k_model, k, "DipoleExcitation")
end

function _validate_incident_electric_field_wavenumber(excitation::LoopExcitation, k)
    excitation.frequency > 0 || error("Loop frequency must be positive.")
    k_model = 2π * excitation.frequency / _C0
    return _check_incident_wavenumber_match(k_model, k, "LoopExcitation")
end

function _validate_incident_electric_field_wavenumber(excitation::MonopoleExcitation, k)
    excitation.frequency > 0 || error("Monopole frequency must be positive.")
    k_model = 2π * excitation.frequency / _C0
    return _check_incident_wavenumber_match(k_model, k, "MonopoleExcitation")
end

function _validate_incident_electric_field_wavenumber(excitation::PatternFeedExcitation, k)
    excitation.frequency > 0 || error("Pattern feed frequency must be positive.")
    k_model = 2π * excitation.frequency / _C0
    return _check_incident_wavenumber_match(k_model, k, "PatternFeedExcitation")
end

function _validate_incident_electric_field_wavenumber(excitation::ImportedExcitation, k)
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
    length(excitation.excitations) == length(excitation.weights) ||
        error("compute_total_field: MultiExcitation has mismatched excitation/weight lengths.")
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
    return plane_wave_field(r, excitation.k_vec, excitation.E0, excitation.pol)
end

function _incident_electric_field(excitation::DipoleExcitation, r::Vec3, k)
    return dipole_incident_field(r, excitation)
end

function _incident_electric_field(excitation::LoopExcitation, r::Vec3, k)
    return loop_incident_field(r, excitation)
end

function _incident_electric_field(excitation::MonopoleExcitation, r::Vec3, k)
    return monopole_incident_field(r, excitation)
end

function _incident_electric_field(excitation::PatternFeedExcitation, r::Vec3, k)
    return pattern_feed_field(r, excitation)
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

function _incident_electric_field(excitation::MultiExcitation, r::Vec3, k)
    length(excitation.excitations) == length(excitation.weights) ||
        error("compute_total_field: MultiExcitation has mismatched excitation/weight lengths.")
    E = CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    for (i, (exc, w)) in enumerate(zip(excitation.excitations, excitation.weights))
        try
            E += w * _incident_electric_field(exc, r, k)
        catch err
            error(
                "compute_total_field: MultiExcitation child $i of type $(typeof(exc)) " *
                "failed incident-field evaluation: $(sprint(showerror, err))"
            )
        end
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
    assemble_excitation(mesh, rwg, excitation; quad_order=3)

Assemble RHS vector v for given excitation.
"""
function assemble_excitation(mesh::TriMesh, rwg::RWGData,
                             excitation::AbstractExcitation;
                             quad_order::Int=3)
    # Dispatch to specialized implementations
    if excitation isa PlaneWaveExcitation
        return assemble_plane_wave(mesh, rwg, excitation; quad_order=quad_order)
    elseif excitation isa PortExcitation
        return assemble_port(mesh, rwg, excitation)
    elseif excitation isa DeltaGapExcitation
        return assemble_delta_gap(mesh, rwg, excitation)
    elseif excitation isa DipoleExcitation
        return assemble_dipole(mesh, rwg, excitation; quad_order=quad_order)
    elseif excitation isa LoopExcitation
        return assemble_loop(mesh, rwg, excitation; quad_order=quad_order)
    elseif excitation isa MonopoleExcitation
        return assemble_monopole(mesh, rwg, excitation; quad_order=quad_order)
    elseif excitation isa ImportedExcitation
        return assemble_imported_excitation(mesh, rwg, excitation; quad_order=quad_order)
    elseif excitation isa PatternFeedExcitation
        return assemble_pattern_feed(mesh, rwg, excitation; quad_order=quad_order)
    elseif excitation isa MultiExcitation
        return assemble_multi(mesh, rwg, excitation; quad_order=quad_order)
    else
        error("Unsupported excitation type: $(typeof(excitation))")
    end
end

"""
    assemble_multiple_excitations(mesh, rwg, excitations; quad_order=3)

Assemble RHS matrix V where each column corresponds to an excitation.
"""
function assemble_multiple_excitations(mesh::TriMesh, rwg::RWGData,
                                       excitations::Vector{<:AbstractExcitation};
                                       quad_order::Int=3)
    N = rwg.nedges
    M = length(excitations)
    V = zeros(ComplexF64, N, M)
    for m in 1:M
        V[:, m] = assemble_excitation(mesh, rwg, excitations[m]; quad_order=quad_order)
    end
    return V
end

# ------------------------------------------------------------------
# Specialized assembly functions
# ------------------------------------------------------------------

function assemble_plane_wave(mesh::TriMesh, rwg::RWGData,
                             pw::PlaneWaveExcitation;
                             quad_order::Int=3)
    N = rwg.nedges
    wq, quad_pts, areas = _excitation_quadrature_cache(mesh, quad_order)
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
                Einc = plane_wave_field(rq, pw.k_vec, pw.E0, pw.pol)
                v[n] += -wq[q] * dot(fn, Einc) * (2 * A)
            end
        end
    end

    return v
end

function assemble_port(mesh::TriMesh, rwg::RWGData, port::PortExcitation)
    v = zeros(ComplexF64, rwg.nedges)
    # Simple delta-gap approximation across port edges
    for e in port.port_edges
        if 1 <= e <= rwg.nedges
            edge_len = rwg.len[e]
            v[e] = port.voltage / edge_len  # Approximate as uniform field across edge
        else
            @warn "Port edge $e is out of bounds (1-$(rwg.nedges)). Skipping."
        end
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
                         quad_order::Int=3)
    N = rwg.nedges
    wq, quad_pts, areas = _excitation_quadrature_cache(mesh, quad_order)
    Nq = length(wq)

    v = zeros(ComplexF64, N)

    for n in 1:N
        for t in (rwg.tplus[n], rwg.tminus[n])
            A = areas[t]
            pts = quad_pts[t]

            for q in 1:Nq
                rq = pts[q]
                fn = eval_rwg(rwg, n, rq, t)

                Einc = dipole_incident_field(rq, dipole)

                v[n] += -wq[q] * dot(fn, Einc) * (2 * A)
            end
        end
    end

    return v
end

function assemble_loop(mesh::TriMesh, rwg::RWGData,
                       loop::LoopExcitation;
                       quad_order::Int=3)
    N = rwg.nedges
    wq, quad_pts, areas = _excitation_quadrature_cache(mesh, quad_order)
    Nq = length(wq)

    v = zeros(ComplexF64, N)
    for n in 1:N
        for t in (rwg.tplus[n], rwg.tminus[n])
            A = areas[t]
            pts = quad_pts[t]
            for q in 1:Nq
                rq = pts[q]
                fn = eval_rwg(rwg, n, rq, t)
                Einc = loop_incident_field(rq, loop)
                v[n] += -wq[q] * dot(fn, Einc) * (2 * A)
            end
        end
    end
    return v
end

function assemble_monopole(mesh::TriMesh, rwg::RWGData,
                           mono::MonopoleExcitation;
                           quad_order::Int=3)
    N = rwg.nedges
    wq, quad_pts, areas = _excitation_quadrature_cache(mesh, quad_order)
    Nq = length(wq)

    v = zeros(ComplexF64, N)
    for n in 1:N
        for t in (rwg.tplus[n], rwg.tminus[n])
            A = areas[t]
            pts = quad_pts[t]
            for q in 1:Nq
                rq = pts[q]
                fn = eval_rwg(rwg, n, rq, t)
                Einc = monopole_incident_field(rq, mono)
                v[n] += -wq[q] * dot(fn, Einc) * (2 * A)
            end
        end
    end
    return v
end

function assemble_imported_excitation(mesh::TriMesh, rwg::RWGData,
                                      imported::ImportedExcitation;
                                      quad_order::Int=3)
    quad_order_eff = _effective_quad_order(quad_order, imported.min_quad_order)
    N = rwg.nedges
    wq, quad_pts, areas = _excitation_quadrature_cache(mesh, quad_order_eff)
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
                v[n] += -wq[q] * dot(fn, Einc) * (2 * A)
            end
        end
    end

    return v
end

function assemble_pattern_feed(mesh::TriMesh, rwg::RWGData,
                               pat::PatternFeedExcitation;
                               quad_order::Int=3)
    N = rwg.nedges
    wq, quad_pts, areas = _excitation_quadrature_cache(mesh, quad_order)
    Nq = length(wq)

    v = zeros(ComplexF64, N)

    for n in 1:N
        for t in (rwg.tplus[n], rwg.tminus[n])
            A = areas[t]
            pts = quad_pts[t]

            for q in 1:Nq
                rq = pts[q]
                fn = eval_rwg(rwg, n, rq, t)
                Einc = pattern_feed_field(rq, pat)
                v[n] += -wq[q] * dot(fn, Einc) * (2 * A)
            end
        end
    end

    return v
end

function assemble_multi(mesh::TriMesh, rwg::RWGData,
                        multi::MultiExcitation;
                        quad_order::Int=3)
    length(multi.excitations) == length(multi.weights) ||
        error("MultiExcitation has mismatched lengths: $(length(multi.excitations)) excitations vs $(length(multi.weights)) weights.")
    v = zeros(ComplexF64, rwg.nedges)
    for (exc, w) in zip(multi.excitations, multi.weights)
        v .+= w .* assemble_excitation(mesh, rwg, exc; quad_order=quad_order)
    end
    return v
end
