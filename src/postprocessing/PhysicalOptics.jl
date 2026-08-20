# PhysicalOptics.jl — Physical Optics (PO) high-frequency RCS solver
#
# Computes scattered far-field and bistatic RCS using the PO approximation:
#   Illuminated face: J_s = 2(n̂ × H_inc)
#   Shadow face:      J_s = 0
#
# Works directly on triangle meshes — no RWG basis functions needed.
# Uses analytical phase integration (exact for linear phase over triangles),
# matching the POFacets 4.5 algorithm (Jenn, NPS).

export POResult, solve_po

const _PO_PHASE_FALLBACK_PRECISION = 4352
const _DEFAULT_MAX_PO_WORK_BYTES = 512 * 1024 * 1024

function _po_work_bytes(Nt::Int, direction_count::Int)
    # Retained raw arrays owned by solve_po. `POResult` retains E_ff, J_s, and
    # illuminated; the remaining arrays are construction workspaces.
    total = BigInt(0)
    total += BigInt(3 * sizeof(Vec3) + sizeof(Float64) + sizeof(Vec3) +
                    sizeof(CVec3)) * Nt
    total += BigInt(sizeof(Bool)) * Nt
    total += BigInt(sizeof(Vec3) + 3 * sizeof(ComplexF64)) * direction_count
    total <= typemax(Int) ||
        throw(ArgumentError("PO raw-workspace estimate overflows Int"))
    return Int(total)
end

function _preflight_po_work(
        Nt::Int, direction_count::Int, max_work_bytes::Integer)
    return _enforce_payload_limit(
        _po_work_bytes(Nt, direction_count), max_work_bytes,
        "PO output and workspace", "max_work_bytes")
end

@noinline function _po_surface_current_exact(
    amplitude::Float64,
    impedance::Float64,
    direction::Vec3,
    phase::ComplexF64,
    triangle::Int,
)
    return setprecision(BigFloat, _PO_PHASE_FALLBACK_PRECISION) do
        scale = 2 * BigFloat(amplitude) / BigFloat(impedance)
        phase_big = Complex{BigFloat}(phase)
        return CVec3(ntuple(component -> begin
            value = ComplexF64(
                scale * BigFloat(direction[component]) * phase_big)
            isfinite(value) ||
                throw(OverflowError(
                    "PO surface current is outside the ComplexF64 range " *
                    "on triangle $triangle"))
            value
        end, 3))
    end
end

@inline function _po_surface_current(
        amplitude::Float64,
        impedance::Float64,
        direction::Vec3,
        phase::ComplexF64,
        triangle::Int)
    needs_exact = _ieee_dense_extreme_factor(amplitude, Float64) ||
                  _ieee_dense_extreme_factor(impedance, Float64) ||
                  _ieee_dense_extreme_factor(phase, Float64)
    @inbounds for component in direction
        needs_exact |= _ieee_dense_extreme_factor(component, Float64)
    end
    needs_exact && return _po_surface_current_exact(
        amplitude, impedance, direction, phase, triangle)
    scale = (2.0 * amplitude) / impedance
    value = CVec3(complex.(scale * direction * phase))
    all(isfinite, value) && return value
    return _po_surface_current_exact(
        amplitude, impedance, direction, phase, triangle)
end

@noinline function _po_farfield_contribution_exact(
    k::Float64,
    amplitude::Float64,
    projection::Vec3,
    phase_integral::ComplexF64,
    triangle::Int,
    direction::Int,
)
    return setprecision(BigFloat, _PO_PHASE_FALLBACK_PRECISION) do
        scale = Complex{BigFloat}(0, 1) * BigFloat(k) *
                BigFloat(amplitude) / BigFloat(2π)
        integral = Complex{BigFloat}(phase_integral)
        value = CVec3(ntuple(component -> begin
            converted = ComplexF64(
                scale * BigFloat(projection[component]) * integral)
            isfinite(converted) ||
                throw(OverflowError(
                    "PO far-field contribution is outside the ComplexF64 " *
                    "range for triangle $triangle, direction $direction"))
            converted
        end, 3))
        return value
    end
end

@noinline function _po_farfield_contribution_geometry_exact(
    k::Float64,
    amplitude::Float64,
    projection::Vec3,
    delta_k::Vec3,
    p1::Vec3,
    p2::Vec3,
    p3::Vec3,
    area::Float64,
    triangle::Int,
    direction::Int,
)
    return setprecision(BigFloat, _PO_PHASE_FALLBACK_PRECISION) do
        vertices = sort((p1, p2, p3); by=Tuple)
        canonical_p1, canonical_p2, canonical_p3 =
            vertices[2], vertices[3], vertices[1]
        integral = _phase_integral_analytical_big_value(
            k, delta_k, canonical_p1, canonical_p2, canonical_p3,
            area, 1e-5, 5)
        scale = Complex{BigFloat}(0, 1) * BigFloat(k) *
                BigFloat(amplitude) / BigFloat(2π)
        value = CVec3(ntuple(component -> begin
            converted = ComplexF64(
                scale * BigFloat(projection[component]) * integral)
            isfinite(converted) ||
                throw(OverflowError(
                    "PO far-field contribution is outside the ComplexF64 " *
                    "range for triangle $triangle, direction $direction"))
            converted
        end, 3))
        return value
    end
end

@inline function _po_farfield_contribution(
    prefactor::ComplexF64,
    k::Float64,
    amplitude::Float64,
    projection::Vec3,
    phase_integral::ComplexF64,
    scale_requires_fallback::Bool,
    triangle::Int,
    direction::Int,
)
    needs_fallback = scale_requires_fallback ||
        _ieee_dense_extreme_factor(phase_integral, Float64)
    @inbounds for component in projection
        needs_fallback |= _ieee_dense_extreme_factor(component, Float64)
    end
    needs_fallback &&
        return _po_farfield_contribution_exact(
            k, amplitude, projection, phase_integral, triangle, direction)

    value = CVec3(complex.(prefactor * projection * phase_integral))
    all(isfinite, value) ||
        return _po_farfield_contribution_exact(
            k, amplitude, projection, phase_integral, triangle, direction)
    return value
end

"""
    POResult

Result from the PO solver containing far-field, surface currents,
illumination mask, and problem metadata.
"""
struct POResult
    E_ff::Matrix{ComplexF64}     # (3, NΩ) scattered far-field
    J_s::Vector{CVec3}           # (Nt,) PO surface current per triangle centroid
    illuminated::BitVector       # (Nt,) which triangles are illuminated
    grid::SphGrid
    freq_hz::Float64
    k::Float64
end

function _validate_po_inputs(grid::SphGrid, freq_hz::Real,
                             excitation::PlaneWaveExcitation,
                             c0::Float64, eta0::Float64)
    NΩ = _validate_sph_grid(grid)
    isfinite(freq_hz) && freq_hz > 0 ||
        throw(ArgumentError(
            "freq_hz must be finite and positive, got $freq_hz"))
    isfinite(c0) && c0 > 0 ||
        throw(ArgumentError("c0 must be finite and positive, got $c0"))
    isfinite(eta0) && eta0 > 0 ||
        throw(ArgumentError(
            "eta0 must be finite and positive, got $eta0"))
    plane_wave_geometry =
        _validate_plane_wave_excitation_geometry(excitation)
    k_norm = plane_wave_geometry.magnitude

    frequency = Float64(freq_hz)
    isfinite(frequency) && frequency > 0 ||
        throw(ArgumentError(
            "freq_hz must be representable as a finite positive Float64, got $freq_hz"))
    k = _frequency_to_wavenumber(
        frequency, c0, "Physical Optics wavenumber")
    isapprox(k_norm, k; rtol=1e-8, atol=0.0) ||
        throw(ArgumentError(
            "plane-wave |k_vec|=$k_norm does not match 2π*freq_hz/c0=$k"))

    return NΩ, frequency, k, plane_wave_geometry.direction
end

# ─── Analytical phase integral helpers (POFacets G.m / fact.m) ───

"""
Recursive helper G(n, w) for Taylor-series phase integral (POFacets G.m).
"""
function _po_G(n::Int, w)
    jw = 1im * w
    iszero(jw) && return one(jw) / (n + 1)
    if abs(jw) < 0.25
        # G_n(w) = integral_0^1 t^n exp(iwt) dt.  The moment series is
        # cancellation-free and its terms decrease geometrically here.
        term = one(jw)
        g = term / (n + 1)
        for order in 1:128
            term *= jw / order
            updated = g + term / (n + order + 1)
            updated == g && return updated
            g = updated
        end
        return g
    end
    exp_jw = exp(jw)
    g = expm1(jw) / jw
    for m in 1:n
        g_prev = g
        g = (exp_jw - m * g_prev) / jw
    end
    return g
end

function _phase_integral_analytical_big_value(
        k::Union{Float64,BigFloat},
        delta_k::Union{Vec3,SVector{3,BigFloat}},
        p1::Vec3, p2::Vec3, p3::Vec3, Area::Float64,
        Lt::Float64, Nt::Int)
    kb = BigFloat(k)
    delta = SVector{3,BigFloat}(BigFloat.(delta_k))
    vertex1 = SVector{3,BigFloat}(BigFloat.(p1))
    vertex2 = SVector{3,BigFloat}(BigFloat.(p2))
    vertex3 = SVector{3,BigFloat}(BigFloat.(p3))
    Dp = kb * dot(vertex1 - vertex3, delta)
    Dq = kb * dot(vertex2 - vertex3, delta)
    Do = kb * dot(vertex3, delta)
    DD = Dq - Dp
    expDo = exp(Complex{BigFloat}(0, Do))
    expDp = exp(Complex{BigFloat}(0, Dp))
    expDq = exp(Complex{BigFloat}(0, Dq))
    threshold = BigFloat(Lt)
    area = BigFloat(Area)
    imaginary = Complex{BigFloat}(0, 1)

    return if abs(Dp) < threshold && abs(Dq) >= threshold
        sum_value = zero(Complex{BigFloat})
        for order in 0:Nt
            sum_value += (imaginary * Dp)^order / factorial(order) *
                (-inv(BigFloat(order + 1)) +
                 expDq * _po_G(order, Complex{BigFloat}(-Dq)))
        end
        sum_value * 2area * expDo / (imaginary * Dq)
    elseif abs(Dp) < threshold && abs(Dq) < threshold
        sum_value = zero(Complex{BigFloat})
        for first_order in 0:Nt, second_order in 0:Nt
            sum_value += (imaginary * Dp)^first_order *
                (imaginary * Dq)^second_order /
                factorial(first_order + second_order + 2)
        end
        sum_value * 2area * expDo
    elseif abs(Dp) >= threshold && abs(Dq) < threshold
        sum_value = zero(Complex{BigFloat})
        for order in 0:Nt
            sum_value += (imaginary * Dq)^order / factorial(order) *
                _po_G(order + 1, Complex{BigFloat}(-Dp)) /
                BigFloat(order + 1)
        end
        sum_value * 2area * expDo * expDp
    elseif abs(Dp) >= threshold && abs(Dq) >= threshold &&
           abs(DD) < threshold
        sum_value = zero(Complex{BigFloat})
        for order in 0:Nt
            sum_value += (-imaginary * DD)^order / factorial(order) *
                (-_po_G(order, Complex{BigFloat}(Dq)) +
                 expDq / BigFloat(order + 1))
        end
        sum_value * 2area * expDo / (imaginary * Dq)
    else
        2area * expDo *
            (expDp / (Dp * DD) - expDq / (Dq * DD) - inv(Dp * Dq))
    end
end

@noinline function _po_farfield_direction_geometry_exact(
        k::Float64,
        amplitude::Float64,
        r_hat::Vec3,
        incident_wavevector::Vec3,
        directions::Vector{Vec3},
        vertex1::Vector{Vec3},
        vertex2::Vector{Vec3},
        vertex3::Vector{Vec3},
        areas::Vector{Float64},
        illuminated::BitVector,
        direction_index::Int)
    return setprecision(BigFloat, _PO_PHASE_FALLBACK_PRECISION) do
        totals = zeros(Complex{BigFloat}, 3)
        k_big = BigFloat(k)
        r_big = SVector{3,BigFloat}(BigFloat.(r_hat))
        r_norm_squared = sum(abs2, r_big)
        r_unit_big = r_big / sqrt(r_norm_squared)
        incident_big = SVector{3,BigFloat}(
            BigFloat.(incident_wavevector))
        incident_unit_big = incident_big / sqrt(sum(abs2, incident_big))
        delta_big = r_unit_big - incident_unit_big
        scale = Complex{BigFloat}(0, 1) * k_big *
                BigFloat(amplitude) / BigFloat(2π)
        @inbounds for triangle in eachindex(illuminated)
            illuminated[triangle] || continue
            vertices = sort(
                (vertex1[triangle], vertex2[triangle], vertex3[triangle]);
                by=Tuple)
            canonical_first, canonical_second, canonical_third =
                vertices[2], vertices[3], vertices[1]
            integral = _phase_integral_analytical_big_value(
                k_big, delta_big,
                canonical_first, canonical_second, canonical_third,
                areas[triangle], 1e-5, 5)
            surface_direction = SVector{3,BigFloat}(
                BigFloat.(directions[triangle]))
            projection = cross(r_big, cross(r_big, surface_direction)) /
                         r_norm_squared
            for component in 1:3
                totals[component] += scale *
                    projection[component] * integral
            end
        end
        return CVec3(ntuple(component -> begin
            converted = ComplexF64(totals[component])
            isfinite(converted) ||
                throw(OverflowError(
                    "PO far field is outside the ComplexF64 range at " *
                    "direction $direction_index"))
            converted
        end, 3))
    end
end

@inline function _po_phase_geometry_requires_exact(
        k::Float64,
        delta_k::Vec3,
        first::Vec3,
        second::Vec3,
        third::Vec3,
        area::Float64)
    _ieee_dense_extreme_factor(area, Float64) && return true
    @inbounds for vertex in (first, second, third)
        for component in vertex
            _ieee_dense_extreme_factor(component, Float64) && return true
        end
    end
    return _source_phase_requires_fallback(k, first - third, delta_k) ||
           _source_phase_requires_fallback(k, second - third, delta_k) ||
           _source_phase_requires_fallback(k, third, delta_k)
end

@noinline function _phase_integral_analytical_big(
        k::Float64, delta_k::Vec3,
        p1::Vec3, p2::Vec3, p3::Vec3, Area::Float64,
        Lt::Float64, Nt::Int)
    return setprecision(BigFloat, _PO_PHASE_FALLBACK_PRECISION) do
        integral = _phase_integral_analytical_big_value(
            k, delta_k, p1, p2, p3, Area, Lt, Nt)
        converted = ComplexF64(integral)
        isfinite(converted) ||
            throw(OverflowError(
                "PO phase integral is outside the ComplexF64 range"))
        converted
    end
end

"""
    _phase_integral_analytical(k, delta_k, v1, v2, v3, Area; Lt=1e-5, Nt=5)

Compute ∫_triangle exp(jk δk·r') dS' analytically using the POFacets formula.

Uses vertex-based decomposition with Taylor-series special cases for
small phase differences (avoids division by zero when δk is nearly
perpendicular to an edge).
"""
function _phase_integral_analytical(k::Float64, delta_k::Vec3,
                                    p1::Vec3, p2::Vec3, p3::Vec3,
                                    Area::Float64;
                                    Lt::Float64=1e-5, Nt::Int=5)
    Nt >= 0 || throw(ArgumentError("PO phase series order must be nonnegative"))
    (isfinite(Lt) && Lt > 0.0) ||
        throw(ArgumentError("PO phase threshold must be finite and positive"))
    # The closed form chooses one vertex as the phase origin.  Its truncated
    # near-pole expansions must not depend on the caller's cyclic triangle
    # ordering, so choose that origin and the two remaining vertices from the
    # geometry itself.
    vertices = sort((p1, p2, p3); by=Tuple)
    p1, p2, p3 = vertices[2], vertices[3], vertices[1]
    edge1 = p1 - p3
    edge2 = p2 - p3
    if _source_phase_requires_fallback(k, edge1, delta_k) ||
       _source_phase_requires_fallback(k, edge2, delta_k) ||
       _source_phase_requires_fallback(k, p3, delta_k)
        return _phase_integral_analytical_big(
            k, delta_k, p1, p2, p3, Area, Lt, Nt)
    end
    # Phase at vertices relative to v3
    Dp = k * dot(Vec3(p1 - p3), delta_k)
    Dq = k * dot(Vec3(p2 - p3), delta_k)
    Do = k * dot(p3, delta_k)

    DD = Dq - Dp
    expDo = exp(1im * Do)
    expDp = exp(1im * Dp)
    expDq = exp(1im * Dq)

    Ic::ComplexF64 = zero(ComplexF64)

    if abs(Dp) < Lt && abs(Dq) >= Lt
        # Special case 1: Dp small, Dq not
        sic = zero(ComplexF64)
        for n in 0:Nt
            sic += (1im * Dp)^n / factorial(n) *
                   (-1.0 / (n + 1) + expDq * _po_G(n, ComplexF64(-Dq)))
        end
        Ic = sic * 2 * Area * expDo / (1im * Dq)
    elseif abs(Dp) < Lt && abs(Dq) < Lt
        # Special case 2: both small
        sic = zero(ComplexF64)
        for n in 0:Nt
            for nn in 0:Nt
                sic += (1im * Dp)^n * (1im * Dq)^nn / factorial(nn + n + 2)
            end
        end
        Ic = sic * 2 * Area * expDo
    elseif abs(Dp) >= Lt && abs(Dq) < Lt
        # Special case 3: Dq small, Dp not
        sic = zero(ComplexF64)
        for n in 0:Nt
            sic += (1im * Dq)^n / factorial(n) *
                   _po_G(n + 1, ComplexF64(-Dp)) / (n + 1)
        end
        Ic = sic * 2 * Area * expDo * expDp
    elseif abs(Dp) >= Lt && abs(Dq) >= Lt && abs(DD) < Lt
        # Special case 4: DD small
        sic = zero(ComplexF64)
        for n in 0:Nt
            sic += (-1im * DD)^n / factorial(n) *
                   (-_po_G(n, ComplexF64(Dq)) + expDq / (n + 1))
        end
        Ic = sic * 2 * Area * expDo / (1im * Dq)
    else
        # General case: all phase differences large enough
        Ic = 2 * Area * expDo *
             (expDp / (Dp * DD) - expDq / (Dq * DD) - 1.0 / (Dp * Dq))
    end

    return Ic
end

"""
    solve_po(mesh, freq_hz, excitation; grid, c0=299792458.0, eta0=376.730313668)

Compute the Physical Optics scattered far-field for a PEC body.

# Arguments
- `mesh::TriMesh`: triangle mesh of the scatterer
- `freq_hz`: frequency in Hz
- `excitation::PlaneWaveExcitation`: incident plane wave
- `grid::SphGrid`: spherical observation grid (default: 36×72)
- `c0, eta0`: physical constants
- `max_work_bytes=536_870_912`: positive raw-payload ceiling for retained
  outputs and construction workspaces; checked before solver-owned arrays

# Returns
`POResult` with far-field `E_ff`, surface currents `J_s`, illumination mask, etc.

# Physics
For a plane wave E_inc = E₀ pol exp(-jk·r), the PO surface current on
illuminated faces is J_s = 2(n̂ × H_inc), where H_inc = (k̂ × E_inc)/η₀.
The scattered far-field is computed using the analytical phase integral
over each triangle (exact for the linear phase exp(jk δk·r')).
"""
function solve_po(mesh::TriMesh, freq_hz::Real, excitation::PlaneWaveExcitation;
                  grid::SphGrid=make_sph_grid(36, 72),
                  c0::Float64=299792458.0,
                  eta0::Float64=376.730313668,
                  max_work_bytes::Integer=_DEFAULT_MAX_PO_WORK_BYTES)
    NΩ, frequency, k, k_hat =
        _validate_po_inputs(grid, freq_hz, excitation, c0, eta0)
    assert_mesh_quality(
        mesh; allow_boundary=true, require_closed=false)
    Nt = ntriangles(mesh)
    _preflight_po_work(Nt, NΩ, max_work_bytes)
    k_vec = excitation.k_vec
    E0 = excitation.E0
    pol = excitation.pol

    # H_inc polarization direction: (k̂ × pol)
    h_pol = cross(k_hat, Vec3(pol))

    # ─── Phase 1: Determine illumination and PO surface currents ───
    illuminated = falses(Nt)
    J_s = Vector{CVec3}(undef, Nt)
    V_t = Vector{Vec3}(undef, Nt)

    # Precompute per-triangle vertices and areas
    tri_v1 = Vector{Vec3}(undef, Nt)
    tri_v2 = Vector{Vec3}(undef, Nt)
    tri_v3 = Vector{Vec3}(undef, Nt)
    tri_area = Vector{Float64}(undef, Nt)

    for t in 1:Nt
        v1 = _mesh_vertex(mesh, mesh.tri[1, t])
        v2 = _mesh_vertex(mesh, mesh.tri[2, t])
        v3 = _mesh_vertex(mesh, mesh.tri[3, t])
        tri_v1[t] = v1
        tri_v2[t] = v2
        tri_v3[t] = v3
        tri_area[t] = triangle_area(mesh, t)

        n_hat = triangle_normal(mesh, t)

        # Illumination: k̂ · n̂ ≤ 0 means wave propagates against normal
        if dot(k_hat, n_hat) <= 0.0
            illuminated[t] = true
            V_t[t] = cross(n_hat, h_pol)  # n̂ × (k̂ × pol)

            rc = triangle_center(mesh, t)
            phase = _source_phase(
                1.0, k_vec, rc, -1.0, "PO incident field")
            J_s[t] = _po_surface_current(
                E0, eta0, V_t[t], phase, t)
        else
            illuminated[t] = false
            V_t[t] = Vec3(0.0, 0.0, 0.0)
            J_s[t] = CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
        end
    end

    # ─── Phase 2: Far-field integration ───
    # E_scat(r̂) = (+jk E₀ / 2π) Σ_t [r̂ × (r̂ × V_t)] × I_t
    # where I_t = ∫_t exp(jk(r̂ - k̂)·r') dS'  (analytical phase integral).
    # The +jk sign matches the package far-field convention E∞ = +jkη₀/(4π) r̂×(r̂×N)
    # (FarField.radiation_vectors, validated against the near-field propagator), so
    # po.E_ff can be combined coherently with MoM/PTD fields; RCS = |E|² is unchanged.
    scale_requires_fallback =
        _ieee_dense_extreme_factor(k, Float64) ||
        _ieee_dense_extreme_factor(E0, Float64)
    prefactor = scale_requires_fallback ? 0.0im : 1im * k * E0 / (2π)

    # Precompute rhat Vec3 (avoids column-slice allocation in hot loop)
    rhat_vec = Vector{Vec3}(undef, NΩ)
    @inbounds for q in 1:NΩ
        rhat_vec[q] = Vec3(grid.rhat[1, q], grid.rhat[2, q], grid.rhat[3, q])
    end

    E_ff = zeros(ComplexF64, 3, NΩ)

    for q in 1:NΩ
        supplied_r_hat = rhat_vec[q]
        r_hat = _validated_farfield_direction(supplied_r_hat)
        # Phase: exp(jk(r̂ - k̂_prop)·r')
        delta_k = r_hat - k_hat

        direction_requires_exact = scale_requires_fallback
        if !direction_requires_exact
            @inbounds for t in 1:Nt
                !illuminated[t] && continue
                if _po_phase_geometry_requires_exact(
                        k, delta_k, tri_v1[t], tri_v2[t], tri_v3[t],
                        tri_area[t])
                    direction_requires_exact = true
                    break
                end
            end
        end
        if direction_requires_exact
            E_q = _po_farfield_direction_geometry_exact(
                k, E0, supplied_r_hat, k_vec, V_t,
                tri_v1, tri_v2, tri_v3, tri_area, illuminated, q)
            E_ff[1, q] = E_q[1]
            E_ff[2, q] = E_q[2]
            E_ff[3, q] = E_q[3]
            continue
        end

        E_q = CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)

        for t in 1:Nt
            !illuminated[t] && continue

            # Form the transverse projection as a double cross product.  The
            # algebraically equivalent dot/subtract form loses a parallel
            # null whenever the stored unit direction has a rounded norm.
            Vt = V_t[t]
            proj = _dipole_cross(r_hat, _dipole_cross(r_hat, Vt)) /
                   sum(abs2, r_hat)
            # Analytical phase integral over triangle
            I_t = _phase_integral_analytical(k, delta_k,
                      tri_v1[t], tri_v2[t], tri_v3[t], tri_area[t])
            isfinite(I_t) ||
                throw(OverflowError(
                    "PO phase integral is non-finite for triangle $t, direction $q"))
            contribution = _po_farfield_contribution(
                prefactor, k, E0, proj, I_t, false, t, q)
            E_q += contribution
            all(isfinite, E_q) ||
                throw(OverflowError(
                    "PO far-field accumulation overflowed at direction $q"))
        end

        E_ff[1, q] = E_q[1]
        E_ff[2, q] = E_q[2]
        E_ff[3, q] = E_q[3]
    end
    all(isfinite, E_ff) ||
        throw(OverflowError("PO far field contains non-finite values"))

    return POResult(E_ff, J_s, illuminated, grid, frequency, k)
end
