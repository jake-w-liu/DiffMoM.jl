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
    R_vec = r - rp
    R = sqrt(dot(R_vec, R_vec))
    if R < 1e-30
        return zero(ComplexF64)
    end
    return (-im / 4) * besselh(0, 2, k * R)
end

"""
    self_cell_integral_2d(k, a_eq)

Analytical integral of G₂D over a circular cell of radius `a_eq`:

  ∫_{|r'|≤a_eq} G₂D(0, r') dA' = (-iπ/(2k²)) [k a_eq H₁⁽²⁾(k a_eq) - 2i/π]

Derived from: d/du[u H₁⁽²⁾(u)] = u H₀⁽²⁾(u).
"""
function self_cell_integral_2d(k::Float64, a_eq::Float64)
    @assert k > 0 "Wavenumber must be positive"
    @assert a_eq > 0 "Equivalent radius must be positive"
    ka = k * a_eq
    H1 = besselh(1, 2, ka)
    return (-im * π / (2 * k^2)) * (ka * H1 - 2im / π)
end

"""
    assemble_D_matrix(mesh::Mesh2D, k)

Assemble the Green's function integral matrix D where:
  D[m,n] = ∫_{cell_n} G₂D(r_m, r') dA'

For m ≠ n: midpoint approximation D[m,n] ≈ G₂D(r_m, r_n) × A_n
For m = n: analytical self-cell integral with equivalent circular cell.
"""
function assemble_D_matrix(mesh::Mesh2D, k::Float64)
    N = mesh.ncells
    A = mesh.cell_area
    a_eq = equivalent_radius(mesh)
    D_self = self_cell_integral_2d(k, a_eq)

    D = Matrix{ComplexF64}(undef, N, N)

    for n in 1:N
        for m in 1:N
            if m == n
                D[m, n] = D_self
            else
                D[m, n] = greens_2d(mesh.centers[m], mesh.centers[n], k) * A
            end
        end
    end

    return D
end
