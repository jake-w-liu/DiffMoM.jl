# QMatrix.jl — Build the Q matrix for the quadratic far-field objective
#
# Q_mn = Σ_q w_q (p†·g_m)* (p†·g_n)
# J(θ) = I† Q I  (radiated power in selected direction/polarization)

export build_Q, apply_Q, pol_linear_x, pol_linear_y, cap_mask, direction_mask

"""
    build_Q(G_mat, grid, pol; mask=nothing)

Build the Hermitian PSD matrix Q from radiation vectors and polarization.

  G_mat: (3*NΩ, N) radiation vector matrix
  grid:  SphGrid with quadrature weights
  pol:   (3, NΩ) complex polarization vectors (unit, transverse to r̂)
  mask:  optional BitVector of length NΩ selecting target directions

Returns Q ∈ C^{N×N}, Hermitian positive semidefinite.
"""
function build_Q(G_mat::Matrix{ComplexF64}, grid::SphGrid,
                 pol::Matrix{ComplexF64}; mask=nothing)
    NΩ = length(grid.w)
    N = size(G_mat, 2)

    # Compute scalar projections: y_q_n = p†(r̂_q) · g_n(r̂_q)
    # y is (NΩ, N)
    y = zeros(ComplexF64, NΩ, N)
    for q in 1:NΩ
        if mask !== nothing && !mask[q]
            continue
        end
        p = pol[:, q]
        for n in 1:N
            idx = 3 * (q - 1)
            gn = SVector{3,ComplexF64}(G_mat[idx+1, n], G_mat[idx+2, n], G_mat[idx+3, n])
            y[q, n] = dot(p, gn)
        end
    end

    # Q_mn = Σ_q w_q conj(y_qm) y_qn
    Q = zeros(ComplexF64, N, N)
    for q in 1:NΩ
        if mask !== nothing && !mask[q]
            continue
        end
        wq = grid.w[q]
        for m in 1:N
            ym = conj(y[q, m])
            for n in 1:N
                Q[m, n] += wq * ym * y[q, n]
            end
        end
    end

    return Q
end

"""
    apply_Q(G_mat, grid, pol, I_coeffs; mask=nothing)

Apply Q*I without forming Q explicitly.
Returns Q*I ∈ C^N.
"""
function apply_Q(G_mat::Matrix{ComplexF64}, grid::SphGrid,
                 pol::Matrix{ComplexF64}, I_coeffs::Vector{ComplexF64};
                 mask=nothing)
    NΩ = length(grid.w)
    N = size(G_mat, 2)

    result = zeros(ComplexF64, N)
    for q in 1:NΩ
        if mask !== nothing && !mask[q]
            continue
        end
        p = pol[:, q]
        wq = grid.w[q]

        # Compute y_q = p† · E∞(r̂_q) = Σ_n I_n (p† · g_n)
        yq = zero(ComplexF64)
        for n in 1:N
            idx = 3 * (q - 1)
            gn = SVector{3,ComplexF64}(G_mat[idx+1, n], G_mat[idx+2, n], G_mat[idx+3, n])
            yq += dot(p, gn) * I_coeffs[n]
        end

        # Accumulate: (Q*I)_m += w_q conj(p†·g_m) y_q
        for m in 1:N
            idx = 3 * (q - 1)
            gm = SVector{3,ComplexF64}(G_mat[idx+1, m], G_mat[idx+2, m], G_mat[idx+3, m])
            result[m] += wq * conj(dot(p, gm)) * yq
        end
    end

    return result
end

"""
    pol_linear_x(grid)

Generate x-polarized far-field polarization vectors (θ̂ component for
broadside radiation along z).
Returns (3, NΩ) complex matrix.
"""
function pol_linear_x(grid::SphGrid)
    NΩ = length(grid.w)
    pol = zeros(ComplexF64, 3, NΩ)
    for q in 1:NΩ
        θ = grid.theta[q]
        φ = grid.phi[q]
        # θ̂ unit vector
        theta_hat = Vec3(cos(θ) * cos(φ), cos(θ) * sin(φ), -sin(θ))
        pol[:, q] = theta_hat
    end
    return pol
end

"""
    pol_linear_y(grid)

Generate the orthogonal far-field polarization vectors (`φ̂` component), which
correspond to y-polarized broadside radiation and the TE/s-polarized basis for
the common `φ = 0` incidence plane in periodic workflows.
Returns `(3, NΩ)` complex matrix.
"""
function pol_linear_y(grid::SphGrid)
    NΩ = length(grid.w)
    pol = zeros(ComplexF64, 3, NΩ)
    for q in 1:NΩ
        φ = grid.phi[q]
        phi_hat = Vec3(-sin(φ), cos(φ), 0.0)
        pol[:, q] = phi_hat
    end
    return pol
end

"""
    cap_mask(grid; theta_max=π/18)

Create a mask selecting directions within a cone of half-angle θ_max
around the z-axis (broadside).
"""
function cap_mask(grid::SphGrid; theta_max=π/18)
    return grid.theta .<= theta_max
end

"""
    direction_mask(grid, direction; half_angle=π/18)

Create a mask selecting directions within a cone of `half_angle` (radians)
around an arbitrary `direction` vector. Generalizes `cap_mask` to any direction.

# Example: backscatter mask for incidence from +z
```julia
mask = direction_mask(grid, Vec3(0,0,-1); half_angle=10*π/180)
```
"""
function direction_mask(grid::SphGrid, direction::Vec3; half_angle::Float64=π/18)
    d = direction / norm(direction)
    NΩ = length(grid.w)
    return BitVector([dot(Vec3(grid.rhat[:, q]), d) >= cos(half_angle) for q in 1:NΩ])
end
