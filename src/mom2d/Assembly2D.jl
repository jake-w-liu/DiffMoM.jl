# Assembly2D.jl — 2D TM Volume Integral Equation assembly and solve
#
# VIE for TM scattering from inhomogeneous dielectric:
#   E_z(r) = E_z^inc(r) + k₀² ∫_D χ(r') G₂D(r,r') E_z(r') dA'
#
# MoM discretization (pulse basis, point matching):
#   Z E = E^inc,  where Z = I - k₀² diag(χ) D

export assemble_vie_2d, solve_vie_2d

"""
    assemble_vie_2d(mesh, k0, chi)

Assemble VIE system matrix: Z[m,n] = δ[m,n] - k₀² χ[n] D[m,n]

Returns (Z, D) where D is the Green's function integral matrix.
"""
function assemble_vie_2d(mesh::Mesh2D, k0::Float64, chi::AbstractVector{Float64})
    @assert length(chi) == mesh.ncells "chi length must match number of cells"

    D = assemble_D_matrix(mesh, k0)
    N = mesh.ncells

    Z = Matrix{ComplexF64}(undef, N, N)
    k0sq = k0^2

    for n in 1:N
        for m in 1:N
            Z[m, n] = -k0sq * chi[n] * D[m, n]
        end
        Z[n, n] += 1.0  # add identity
    end

    return Z, D
end

"""
    solve_vie_2d(mesh, k0, chi, E_inc)

Solve the 2D VIE for internal total fields.
Returns `VIEResult2D` with all computed quantities for downstream use.
"""
function solve_vie_2d(mesh::Mesh2D, k0::Float64, chi::AbstractVector{Float64},
                      E_inc::AbstractVector{ComplexF64})
    @assert length(E_inc) == mesh.ncells "E_inc length must match number of cells"

    Z, D = assemble_vie_2d(mesh, k0, chi)
    Z_lu = lu(Z)
    E_total = Z_lu \ E_inc

    return VIEResult2D(E_total, Vector(E_inc), Vector(chi), D, Z, Z_lu, mesh, k0)
end
