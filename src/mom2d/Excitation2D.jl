# Excitation2D.jl — Incident field generation for 2D TM scattering
#
# Convention: exp(+iωt), plane wave E_z^inc = E₀ exp(-ik₀ k̂·r)

export planewave_2d, linesource_2d

const _DEFAULT_MAX_PLANEWAVE_EXACT_PHASE_WORK_2D = 2_000_000

@inline function _planewave_phase_requires_exact_2d(
        k0::Float64, khat::Vec2, center::Vec2)
    return _source_phase_requires_fallback(
        k0,
        Vec3(khat[1], khat[2], 0.0),
        Vec3(center[1], center[2], 0.0),
    )
end

@inline function _planewave_phase_precision_2d(
        k0::Float64, khat::Vec2, center::Vec2)
    return _source_phase_precision(
        k0,
        Vec3(khat[1], khat[2], 0.0),
        Vec3(center[1], center[2], 0.0),
    )
end

@noinline function _planewave_phase_big_2d(
        k0::Float64, phi_inc::Float64, center::Vec2, E0::Float64,
        precision::Int, cell::Int)
    return setprecision(BigFloat, precision) do
        sin_phi, cos_phi = sincos(BigFloat(phi_inc))
        phase = BigFloat(k0) *
                (cos_phi * BigFloat(center[1]) +
                 sin_phi * BigFloat(center[2]))
        value = ComplexF64(
            BigFloat(E0) *
            exp(Complex{BigFloat}(zero(BigFloat), -phase)))
        isfinite(value) ||
            error("planewave_2d produced a non-finite phase at cell $cell.")
        return value
    end
end

"""
    planewave_2d(mesh, k0, phi_inc;
                 E0=1.0, max_exact_phase_work=2_000_000)

Generate incident plane wave at cell centers.
Propagation direction: k̂ = (cos(phi_inc), sin(phi_inc)).
E_z^inc(r) = E₀ exp(-ik₀ k̂·r)
"""
function planewave_2d(
    mesh::Mesh2D,
    k0::Float64,
    phi_inc::Float64;
    E0::Float64=1.0,
    max_exact_phase_work::Integer=
        _DEFAULT_MAX_PLANEWAVE_EXACT_PHASE_WORK_2D,
)
    _validate_mesh_2d(mesh)
    _validate_positive_finite_2d(k0, "planewave_2d wavenumber")
    isfinite(phi_inc) ||
        throw(ArgumentError(
            "planewave_2d incidence angle must be finite, got $phi_inc."))
    isfinite(E0) ||
        throw(ArgumentError("planewave_2d amplitude must be finite, got $E0."))
    exact_work_limit = try
        Int(max_exact_phase_work)
    catch error
        error isa Union{InexactError,OverflowError} || rethrow()
        throw(ArgumentError(
            "max_exact_phase_work is outside the Int range"))
    end
    exact_work_limit >= 0 ||
        throw(ArgumentError(
            "max_exact_phase_work must be nonnegative, got " *
            "$max_exact_phase_work"))
    sin_phi, cos_phi = sincos(phi_inc)
    khat = Vec2(cos_phi, sin_phi)

    # Classify and charge every cold phase before allocating the output.  An
    # accepted large mesh therefore cannot turn a finite-coordinate recovery
    # case into unbounded arbitrary-precision work.
    exact_phase_work = 0
    @inbounds for center in mesh.centers
        _planewave_phase_requires_exact_2d(k0, khat, center) || continue
        precision = _planewave_phase_precision_2d(k0, khat, center)
        precision <= exact_work_limit - exact_phase_work ||
            throw(ArgumentError(
                "planewave_2d exact phase work exceeds " *
                "max_exact_phase_work=$exact_work_limit"))
        exact_phase_work += precision
    end

    E_inc = Vector{ComplexF64}(undef, mesh.ncells)
    @inbounds for m in 1:mesh.ncells
        center = mesh.centers[m]
        if _planewave_phase_requires_exact_2d(k0, khat, center)
            E_inc[m] = _planewave_phase_big_2d(
                k0, phi_inc, center, E0,
                _planewave_phase_precision_2d(k0, khat, center), m)
        else
            phase = k0 * dot(khat, center)
            E_inc[m] = E0 * exp(-im * phase)
        end
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
