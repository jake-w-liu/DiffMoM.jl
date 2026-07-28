# Greens2D.jl — 2D scalar Green's function for Helmholtz equation
#
# Convention: exp(+iωt)
# G₂D(r,r') = (-i/4) H₀⁽²⁾(k|r-r'|)
# Satisfies: (∇² + k²) G = -δ(r-r')

export greens_2d, self_cell_integral_2d

"""
    greens_2d(r, rp, k)

2D scalar free-space Green's function:
  G(r,r') = (-i/4) H₀⁽²⁾(k|r-r'|)

Uses exp(+iωt) convention with outgoing H₀⁽²⁾.
"""
function greens_2d(r::Vec2, rp::Vec2, k::Float64)
    _validate_finite_vec2_2d(r, "greens_2d observation point")
    _validate_finite_vec2_2d(rp, "greens_2d source point")
    _validate_positive_finite_2d(k, "greens_2d wavenumber")
    return _greens_2d_unchecked(r, rp, k)
end

@inline function _greens_2d_unchecked(r::Vec2, rp::Vec2, k::Float64)
    R_vec = r - rp
    R = norm(R_vec)
    isfinite(R) ||
        throw(ArgumentError(
            "greens_2d point separation must be finite, got $R."))
    if R < 1e-30
        return zero(ComplexF64)
    end
    kR = k * R
    isfinite(kR) ||
        throw(ArgumentError(
            "greens_2d phase argument k*R must be finite, got $kR."))
    value = (-im / 4) * besselh(0, 2, kR)
    isfinite(value) ||
        error("greens_2d produced a non-finite Green's function value.")
    return value
end

"""
    self_cell_integral_2d(k, a_eq)

Analytical integral of G₂D over a circular cell of radius `a_eq`:

  ∫_{|r'|≤a_eq} G₂D(0, r') dA' = (-iπ/(2k²)) [k a_eq H₁⁽²⁾(k a_eq) - 2i/π]

Derived from: d/du[u H₁⁽²⁾(u)] = u H₀⁽²⁾(u).
"""
function self_cell_integral_2d(k::Float64, a_eq::Float64)
    _validate_positive_finite_2d(k, "self-cell wavenumber")
    _validate_positive_finite_2d(a_eq, "self-cell equivalent radius")
    ka = k * a_eq
    isfinite(ka) ||
        throw(ArgumentError(
            "self-cell size parameter k*a_eq must be finite, got $ka."))
    H1 = besselh(1, 2, ka)
    value = (-im * π / (2 * k^2)) * (ka * H1 - 2im / π)
    isfinite(value) ||
        error("self_cell_integral_2d produced a non-finite value.")
    return value
end

"""
    assemble_D_matrix(mesh::Mesh2D, k)

Assemble the Green's function integral matrix D where:
  D[m,n] = ∫_{cell_n} G₂D(r_m, r') dA'

For m ≠ n: midpoint approximation D[m,n] ≈ G₂D(r_m, r_n) × A_n
For m = n: analytical self-cell integral with equivalent circular cell.
"""
function assemble_D_matrix(mesh::Mesh2D, k::Float64)
    _validate_mesh_2d(mesh)
    _validate_positive_finite_2d(k, "assemble_D_matrix wavenumber")
    return _assemble_D_matrix_unchecked(mesh, k)
end

function _assemble_D_matrix_unchecked(mesh::Mesh2D, k::Float64)
    N = mesh.ncells
    A = mesh.cell_area
    a_eq = _equivalent_radius_unchecked(mesh)
    D_self = self_cell_integral_2d(k, a_eq)

    D = Matrix{ComplexF64}(undef, N, N)

    @inbounds for n in 1:N
        for m in 1:N
            if m == n
                D[m, n] = D_self
            else
                D[m, n] =
                    _greens_2d_unchecked(mesh.centers[m], mesh.centers[n], k) * A
            end
        end
    end

    all(isfinite, D) ||
        error("assemble_D_matrix produced non-finite matrix entries.")
    return D
end
