# PlanarSurface.jl — conductor surface-impedance models
#
# The `surface_zs` keyword of assemble_planar_z / solve_planar adds a
# surface-impedance Gram loss term Z -= Zs * Gram on every sheet.  This
# file produces that value from bulk conductor properties and published
# roughness correction factors:
#
#   smooth good conductor (e^{+i wt}):  Zs = Rs * (1 + i),
#     Rs = sqrt(omega * mu / (2 sigma)),  delta = sqrt(2/(omega*mu*sigma))
#
#   Hammerstad-Jensen (and modified form, Shlepnev's RF parameter):
#     K = 1 + (rf - 1) * (2/pi) * atan(1.4 * (rms/delta)^2)
#   K saturates at rf for rms >> delta (classical rf = 2).
#
#   Huray "snowball" (N spherical nodules of radius r per flat area A):
#     K = 1 + (3/2) * (N*4*pi*r^2/A) / (1 + delta/r + delta^2/(2 r^2))
#   the magnetic-screening term vanishes for delta << r (large nodules
#   behave flat) and saturates for delta >> r.
#
# Both factors correct the dissipative power; applying them to the full
# complex Zs (default) also scales the internal inductance, matching the
# common solver convention.  loss_only=true restricts the correction to
# Re(Zs), matching the models' literal attenuation-constant statement.

export PlanarRoughness, HammerstadRoughness, HurayRoughness
export planar_skin_depth, roughness_factor, planar_surface_zs

"""Abstract conductor-roughness model for `roughness_factor`."""
abstract type PlanarRoughness end

"""Hammerstad-Jensen roughness: `rms` profile height [m], `rf` asymptotic
loss multiplier (classical model: `rf = 2`)."""
struct HammerstadRoughness <: PlanarRoughness
    rms::Float64
    rf::Float64
    function HammerstadRoughness(rms::Real; rf::Real=2.0)
        rms = Float64(rms); rf = Float64(rf)
        (isfinite(rms) && rms >= 0) || throw(ArgumentError(
            "roughness rms must be finite and >= 0, got $rms"))
        (isfinite(rf) && rf >= 1) || throw(ArgumentError(
            "roughness rf must be finite and >= 1, got $rf"))
        return new(rms, rf)
    end
end

"""Huray "snowball" roughness: spherical nodules of `radius` [m] at
`density` nodules per unit conductor area [m^-2]."""
struct HurayRoughness <: PlanarRoughness
    radius::Float64
    density::Float64
    function HurayRoughness(radius::Real, density::Real)
        radius = Float64(radius); density = Float64(density)
        (isfinite(radius) && radius > 0) || throw(ArgumentError(
            "nodule radius must be finite and > 0, got $radius"))
        (isfinite(density) && density >= 0) || throw(ArgumentError(
            "nodule density must be finite and >= 0, got $density"))
        return new(radius, density)
    end
end

"""Skin depth delta = 1/sqrt(pi * f * mu0 * mur * sigma) [m]."""
function planar_skin_depth(f::Real, sigma::Real; mur::Real=1.0)
    (isfinite(f) && f > 0) || throw(ArgumentError(
        "frequency must be finite and > 0, got $f"))
    (isfinite(sigma) && sigma > 0) || throw(ArgumentError(
        "conductivity must be finite and > 0, got $sigma"))
    (isfinite(mur) && mur > 0) || throw(ArgumentError(
        "mur must be finite and > 0, got $mur"))
    return inv(sqrt(pi * f * _MU0 * mur * sigma))
end

"""Roughness correction factor K >= 1 at `f` [Hz] for conductor `sigma`
[S/m] (relative permeability `mur`)."""
roughness_factor

function roughness_factor(m::HammerstadRoughness, f::Real,
        sigma::Real; mur::Real=1.0)
    d = planar_skin_depth(f, sigma; mur=mur)
    return 1 + (m.rf - 1) * (2 / pi) * atan(1.4 * (m.rms / d)^2)
end

function roughness_factor(m::HurayRoughness, f::Real,
        sigma::Real; mur::Real=1.0)
    d = planar_skin_depth(f, sigma; mur=mur)
    x = d / m.radius
    return 1 + 1.5 * (4pi * m.radius^2 * m.density) /
        (1 + x + x^2 / 2)
end

"""Complex surface impedance [Ohm] of a conductor with bulk `sigma` [S/m]
at `f` [Hz] under e^{+i wt}:  Zs = Rs*(1 + i) with
Rs = sqrt(omega*mu0*mur / (2*sigma)).  `roughness` (a `PlanarRoughness`
model) scales Zs by its published factor; `loss_only=true` applies the
factor to Re(Zs) only (the models' literal attenuation statement)."""
function planar_surface_zs(f::Real, sigma::Real; mur::Real=1.0,
        roughness::Union{Nothing,PlanarRoughness}=nothing,
        loss_only::Bool=false)
    d = planar_skin_depth(f, sigma; mur=mur)
    rs = 0.5 * _MU0 * mur * 2pi * f * d   # = sqrt(omega*mu / (2*sigma))
    k = roughness === nothing ? 1.0 :
        roughness_factor(roughness, f, sigma; mur=mur)
    return loss_only ? complex(k * rs, rs) : k * rs * (1 + 1im)
end
