# PlanarTypes.jl — Data structures for shielded layered-media planar analysis
#
# Geometry model: a rectangular shielding box 0<=x<=a, 0<=y<=b with PEC
# (default) or PMC sidewalls.  A stack of `L` dielectric layers fills 0 =
# z_0 < z_1 < ... <
# z_L = h_box.  Infinitely thin metal sheets (cell masks on a uniform grid)
# live on selected interfaces; vertical terminations at z=0 and z=h_box are
# PEC, PMC, a surface impedance, or an open half-space.
#
# Scalar type parameter T supports Float64 for analysis and dual/complex
# perturbation types for gradients of layer parameters (epsr, mur, d, zs).
#
# Modal decomposition: fields expand in the discrete TE/TM modes of the
# rectangular waveguide formed by the sidewalls (kx = m*pi/a, ky = n*pi/b,
# gamma_l = sqrt(kc^2 - k_l^2)); the z-direction is an analytic
# transmission-line cascade, so coupling coefficients are 2-D modal sums.

export PlanarLayer, PlanarTerminator, PlanarStackup, CellGrid
export BoundaryKind, TERM_PEC, TERM_PMC, TERM_SURFACE, TERM_OPEN
export TERM_GND, TERM_SPACE
export SidewallKind, WALL_PEC, WALL_PMC
export planar_interfaces, planar_k0_layer, planar_validate

"""Layer of the stratified medium: complex eps_r, mu_r, thickness [m].
`epsr`/`mur` are the transverse (in-plane) constants; `epsr_z`/`mur_z`
are the axial constants of a uniaxial layer with optic axis z (they
default to the transverse values, i.e. isotropic).  `T` promotes to the
scalar type used for differentiation (Float64 for plain analysis)."""
struct PlanarLayer{T<:Number}
    epsr::T        # includes dielectric loss: epsr*(1 - i*tand) for e^{+iwt}
    mur::T
    thickness::T   # d > 0 [m]
    epsr_z::T
    mur_z::T
end

# promote naturally: Float64 inputs -> Float64 fields; dual/complex inputs
# keep their perturbation type so gradients flow through the cascade
PlanarLayer(epsr::Number, mur::Number, d::Number;
            epsr_z::Number=epsr, mur_z::Number=mur) =
    PlanarLayer(promote(epsr, mur, Float64(d), epsr_z, mur_z)...)
PlanarLayer(epsr::Number, mur::Number, d::Number,
            epsr_z::Number, mur_z::Number) =
    PlanarLayer(promote(epsr, mur, Float64(d), epsr_z, mur_z)...)

"""Box z-termination kind: `TERM_PEC`/`TERM_PMC`/`TERM_SURFACE`/`TERM_OPEN`."""
@enum BoundaryKind::UInt8 begin
    TERM_PEC
    TERM_PMC
    TERM_SURFACE
    TERM_OPEN
end
@doc "PEC (electric-wall) box termination." TERM_PEC
@doc "PMC (magnetic-wall) box termination." TERM_PMC
@doc "Surface-impedance box termination (`zs` field)." TERM_SURFACE
@doc "Open half-space box termination (`epsr`/`mur` fields)." TERM_OPEN

"""Vertical termination of a `PlanarStackup`: `kind` selects PEC, PMC,
a surface impedance `zs` [Ohm], or an open half-space with exterior
`epsr`/`mur`."""
struct PlanarTerminator{T<:Number}
    kind::BoundaryKind
    zs::T            # surface impedance [Ohm] for TERM_SURFACE
    epsr::T          # exterior medium for TERM_OPEN
    mur::T
end

PlanarTerminator(kind::BoundaryKind) =
    PlanarTerminator{ComplexF64}(kind, 0.0 + 0im, 1.0 + 0im, 1.0 + 0im)

"""Convenience PEC terminator (grounded box face)."""
const TERM_GND = PlanarTerminator(TERM_PEC)
"""Convenience open terminator (air half-space outside the box)."""
const TERM_SPACE = PlanarTerminator(TERM_OPEN)

"""
    PlanarStackup(layers, bottom, top, a, b)

`layers` ordered bottom -> top.  `bottom` terminates the cascade at z=0,
`top` at z = sum(thickness).  `a`, `b` are the box cross-section sides [m].
Interface `i` (0..L) is the top face of layer `i`.
"""
struct PlanarStackup{T<:Number}
    layers::Vector{PlanarLayer{T}}
    bottom::PlanarTerminator{T}
    top::PlanarTerminator{T}
    a::Float64
    b::Float64
end

_layer_scalar(::PlanarLayer{T}) where {T} = T
_term_scalar(::PlanarTerminator{T}) where {T} = T

function PlanarStackup(layers::Vector{<:PlanarLayer},
        bottom::PlanarTerminator, top::PlanarTerminator,
        a::Real, b::Real)
    T = promote_type(map(_layer_scalar, layers)...,
                     _term_scalar(bottom), _term_scalar(top))
    l2 = [PlanarLayer(
            convert(T, l.epsr), convert(T, l.mur),
            convert(T, l.thickness),
            convert(T, l.epsr_z), convert(T, l.mur_z)) for l in layers]
    bt = PlanarTerminator{T}(bottom.kind, convert(T, bottom.zs),
        convert(T, bottom.epsr), convert(T, bottom.mur))
    tp = PlanarTerminator{T}(top.kind, convert(T, top.zs),
        convert(T, top.epsr), convert(T, top.mur))
    return PlanarStackup{T}(l2, bt, tp, Float64(a), Float64(b))
end

"""z-coordinates of interfaces: z[1]=0 (bottom), z[end]=sum(thickness)."""
function planar_interfaces(stack::PlanarStackup{T}) where {T}
    L = length(stack.layers)
    z = Vector{T}(undef, L + 1)
    z[1] = zero(T)
    acc = zero(T)
    @inbounds for l in 1:L
        acc += stack.layers[l].thickness
        z[l + 1] = acc
    end
    return z
end

"""Validate a stackup: positive layer thickness, Re(epsr)/Re(mur) > 0,
positive box sides, finite terminator parameters."""
function planar_validate(stack::PlanarStackup)
    isempty(stack.layers) &&
        throw(ArgumentError("PlanarStackup requires at least one layer"))
    @inbounds for l in eachindex(stack.layers)
        layer = stack.layers[l]
        isfinite(real(layer.epsr)) ||
            throw(ArgumentError("layer $l epsr must be finite"))
        real(layer.epsr) > 0 ||
            throw(ArgumentError("layer $l epsr must have Re > 0"))
        isfinite(real(layer.mur)) ||
            throw(ArgumentError("layer $l mur must be finite"))
        real(layer.mur) > 0 ||
            throw(ArgumentError("layer $l mur must have Re > 0"))
        isfinite(real(layer.epsr_z)) && real(layer.epsr_z) > 0 ||
            throw(ArgumentError("layer $l epsr_z must have Re > 0"))
        isfinite(real(layer.mur_z)) && real(layer.mur_z) > 0 ||
            throw(ArgumentError("layer $l mur_z must have Re > 0"))
        isfinite(real(layer.thickness)) && real(layer.thickness) > 0 ||
            throw(ArgumentError("layer $l thickness must be positive"))
    end
    isfinite(stack.a) && stack.a > 0 ||
        throw(ArgumentError("box side a must be positive"))
    isfinite(stack.b) && stack.b > 0 ||
        throw(ArgumentError("box side b must be positive"))
    for (name, term) in (("bottom", stack.bottom), ("top", stack.top))
        if term.kind == TERM_SURFACE
            isfinite(real(term.zs)) ||
                throw(ArgumentError("$name terminator zs must be finite"))
        elseif term.kind == TERM_OPEN
            real(term.epsr) > 0 ||
                throw(ArgumentError("$name open terminator needs Re epsr > 0"))
            real(term.mur) > 0 ||
                throw(ArgumentError("$name open terminator needs Re mur > 0"))
        end
    end
    return nothing
end

# physical constants _C0/_EPS0/_MU0 are defined in assembly/Excitation.jl

"""k^2 of a layer at angular frequency omega: k2 = (omega/c0)^2 * epsr*mur."""
@inline function planar_k2_layer(omega::Number, epsr::Number, mur::Number)
    return (omega / _C0)^2 * epsr * mur
end

"""Transverse wavenumber of a layer at `omega`:
`sqrt(planar_k2_layer(omega, layer.epsr, layer.mur))` — the transverse
(in-plane) constants for a uniaxial layer."""
@inline planar_k0_layer(omega::Number, layer::PlanarLayer) =
    sqrt(planar_k2_layer(omega, layer.epsr, layer.mur))

"""Sidewall boundary condition of the analysis box: `WALL_PEC` (electric
wall, the default shielded box) or `WALL_PMC` (magnetic wall, an idealized
open/symmetry lateral boundary).  The PMC choice swaps the sin/cos parity
of every box-mode function and swaps the TE/TM existence masks."""
@enum SidewallKind::UInt8 begin
    WALL_PEC
    WALL_PMC
end
@doc "Electric sidewall: tangential E vanishes on the box sides." WALL_PEC
@doc "Magnetic sidewall: tangential H vanishes on the box sides (open/symmetry lateral boundary)." WALL_PMC

"""
    CellGrid(a, b, nx, ny; walls=WALL_PEC)

Uniform rectangular cell grid over the box cross-section, dx = a/nx,
dy = b/ny.  Cell (i,j), 1-based, covers
[(i-1)*dx, i*dx] x [(j-1)*dy, j*dy].  `walls` selects the sidewall
boundary condition of the box (`WALL_PEC` default, `WALL_PMC` for a
magnetic-wall / approximate open lateral boundary).  Gap ports on a
`WALL_PMC` boundary are mathematical ports driving the sheet edge
against the boundary surface; there is no wall conductor to carry a
return current.
"""
struct CellGrid
    nx::Int
    ny::Int
    dx::Float64
    dy::Float64
    a::Float64
    b::Float64
    walls::SidewallKind
end

function CellGrid(a::Real, b::Real, nx::Integer, ny::Integer;
        walls::SidewallKind=WALL_PEC)
    nx >= 1 && ny >= 1 ||
        throw(ArgumentError("CellGrid requires nx, ny >= 1"))
    af, bf = Float64(a), Float64(b)
    isfinite(af) && af > 0 || throw(ArgumentError("grid a must be positive"))
    isfinite(bf) && bf > 0 || throw(ArgumentError("grid b must be positive"))
    return CellGrid(nx, ny, af / nx, bf / ny, af, bf, walls)
end
