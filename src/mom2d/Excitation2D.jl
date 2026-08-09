# Excitation2D.jl — Incident field generation for 2D TM scattering
#
# Convention: exp(+iωt), plane wave E_z^inc = E₀ exp(-ik₀ k̂·r)

export planewave_2d, linesource_2d

"""
    planewave_2d(mesh, k0, phi_inc; E0=1.0)

Generate incident plane wave at cell centers.
Propagation direction: k̂ = (cos(phi_inc), sin(phi_inc)).
E_z^inc(r) = E₀ exp(-ik₀ k̂·r)
"""
function planewave_2d(mesh::Mesh2D, k0::Float64, phi_inc::Float64; E0::Float64=1.0)
    _validate_mesh_2d(mesh)
    _validate_positive_finite_2d(k0, "planewave_2d wavenumber")
    isfinite(phi_inc) ||
        throw(ArgumentError(
            "planewave_2d incidence angle must be finite, got $phi_inc."))
    isfinite(E0) ||
        throw(ArgumentError("planewave_2d amplitude must be finite, got $E0."))
    sin_phi, cos_phi = sincos(phi_inc)
    khat = Vec2(cos_phi, sin_phi)
    E_inc = Vector{ComplexF64}(undef, mesh.ncells)
    @inbounds for m in 1:mesh.ncells
        phase = k0 * dot(khat, mesh.centers[m])
        isfinite(phase) ||
            throw(ArgumentError(
                "planewave_2d phase must be finite at cell $m, got $phase."))
        E_inc[m] = E0 * exp(-im * phase)
    end
    all(isfinite, E_inc) ||
        error("planewave_2d produced non-finite incident-field values.")
    return E_inc
end

"""
    linesource_2d(mesh, k0, r_src)

Generate incident field from a 2D line source at position `r_src`.
E_z^inc(r) = (-i/4) H₀⁽²⁾(k₀|r - r_src|) (unit amplitude)
"""
function linesource_2d(mesh::Mesh2D, k0::Float64, r_src::Vec2)
    _validate_mesh_2d(mesh)
    _validate_positive_finite_2d(k0, "linesource_2d wavenumber")
    _validate_finite_vec2_2d(r_src, "linesource_2d source point")
    E_inc = Vector{ComplexF64}(undef, mesh.ncells)
    @inbounds for m in 1:mesh.ncells
        displacement = mesh.centers[m] - r_src
        separation = hypot(displacement[1], displacement[2])
        !iszero(separation) ||
            throw(DomainError(
                r_src,
                "linesource_2d is singular because the source coincides with cell center $m.",
            ))
        # Unit-amplitude 2D line source: E_z = (-i/4) H₀⁽²⁾(k₀R) = greens_2d.
        E_inc[m] = _greens_2d_unchecked(mesh.centers[m], r_src, k0)
    end
    all(isfinite, E_inc) ||
        error("linesource_2d produced non-finite incident-field values.")
    return E_inc
end
