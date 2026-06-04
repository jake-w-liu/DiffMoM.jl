# Scatter2D.jl — Scattered field computation and Jacobian for 2D VIE
#
# E_scat(r_obs) = k₀² Σ_n χ_n E_n G₂D(r_obs, r_n) A_n
#
# Jacobian: ∂E_scat/∂χ computed via implicit differentiation of the VIE system

export scattered_field_2d, green_obs_matrix, jacobian_scattered_field_2d

"""
    green_obs_matrix(r_obs, mesh, k0)

Compute the observation Green's function matrix G_obs[m,n] = G₂D(r_obs[m], r_n).
Observation points must be outside the scattering domain.
"""
function green_obs_matrix(r_obs::AbstractVector{Vec2}, mesh::Mesh2D, k0::Float64)
    M = length(r_obs)
    N = mesh.ncells
    G_obs = Matrix{ComplexF64}(undef, M, N)
    for n in 1:N
        for m in 1:M
            G_obs[m, n] = greens_2d(r_obs[m], mesh.centers[n], k0)
        end
    end
    return G_obs
end

"""
    scattered_field_2d(vie_result, r_obs)

Compute scattered field at observation points using solved VIE result.
E_scat(r_obs) = k₀² Σ_n χ_n E_n G₂D(r_obs, r_n) A_n
"""
function scattered_field_2d(vr::VIEResult2D, r_obs::AbstractVector{Vec2})
    G_obs = green_obs_matrix(r_obs, vr.mesh, vr.k0)
    A = vr.mesh.cell_area
    k0sq = vr.k0^2
    N = vr.mesh.ncells
    M = length(r_obs)

    E_scat = zeros(ComplexF64, M)
    for m in 1:M
        for n in 1:N
            E_scat[m] += k0sq * vr.chi[n] * vr.E_total[n] * G_obs[m, n] * A
        end
    end
    return E_scat
end

"""
    jacobian_scattered_field_2d(vie_result, r_obs)

Compute the Jacobian J[m,p] = ∂E_scat(r_obs[m])/∂χ_p.

Uses implicit differentiation: Z E = E^inc ⟹ ∂E/∂χ_p = k₀² E_p Z⁻¹ D[:,p].
Precomputes S = Z⁻¹ D for efficiency.

Returns (J, G_obs) where J is M_rx × N_cells.
"""
function jacobian_scattered_field_2d(vr::VIEResult2D, r_obs::AbstractVector{Vec2})
    G_obs = green_obs_matrix(r_obs, vr.mesh, vr.k0)
    A = vr.mesh.cell_area
    k0sq = vr.k0^2
    N = vr.mesh.ncells
    M = length(r_obs)

    # S = Z⁻¹ D: solve Z S = D column-by-column using existing LU factorization
    S = vr.Z_LU \ vr.D

    # Build Jacobian:
    # ∂E_scat(r_obs[m])/∂χ_p = k₀² A Σ_n G_obs[m,n] [δ_{np} E_p + χ_n k₀² E_p S[n,p]]
    #   = k₀² A E_p [G_obs[m,p] + k₀² Σ_n χ_n G_obs[m,n] S[n,p]]

    # Precompute W = I + k₀² diag(χ) S  (N×N)
    W = Matrix{ComplexF64}(undef, N, N)
    for p in 1:N
        for n in 1:N
            W[n, p] = k0sq * vr.chi[n] * S[n, p]
        end
        W[p, p] += 1.0
    end

    # J = k₀² A × G_obs × W × diag(E)
    J = Matrix{ComplexF64}(undef, M, N)
    GW = G_obs * W  # M×N
    for p in 1:N
        for m in 1:M
            J[m, p] = k0sq * A * GW[m, p] * vr.E_total[p]
        end
    end

    return J, G_obs
end
