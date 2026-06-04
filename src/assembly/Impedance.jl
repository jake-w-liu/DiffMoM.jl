# Impedance.jl — Impedance sheet term assembly and exact dZ/dθ
#
# Z_imp[m,n](θ) = -Σ_p θ_p ∫_{Γ_p} f_m · f_n dS
# ∂Z_mn/∂θ_p   = -∫_{Γ_p} f_m · f_n dS  =  -M_p[m,n]

export assemble_Z_impedance, precompute_patch_mass, assemble_dZ_dtheta

"""
    precompute_patch_mass(mesh, rwg, partition; quad_order=3)

Precompute the patch mass matrices M_p[m,n] = ∫_{Γ_p} f_m · f_n dS
for each patch p = 1:P.
Returns a vector of compact local matrices.
"""
function precompute_patch_mass(mesh::TriMesh, rwg::RWGData,
                               partition::PatchPartition; quad_order::Int=3)
    N = rwg.nedges
    Nt = ntriangles(mesh)
    P = partition.P
    Tcoef = promote_type(eltype(rwg.coeff_plus), eltype(rwg.coeff_minus))
    Tmass = Tcoef <: Real ? Float64 : ComplexF64

    xi, wq = tri_quad_rule(quad_order)
    Nq = length(wq)

    # Precompute quad points
    quad_pts = [tri_quad_points(mesh, t, xi) for t in 1:Nt]
    areas    = [triangle_area(mesh, t) for t in 1:Nt]

    # For each triangle, find which basis functions have support there
    tri_to_basis = [Int[] for _ in 1:Nt]
    for n in 1:N
        push!(tri_to_basis[rwg.tplus[n]], n)
        push!(tri_to_basis[rwg.tminus[n]], n)
    end

    rows_p = [Int[] for _ in 1:P]
    cols_p = [Int[] for _ in 1:P]
    vals_p = [Tmass[] for _ in 1:P]

    for t in 1:Nt
        p = partition.tri_patch[t]
        A = areas[t]
        basis_on_t = tri_to_basis[t]

        for bi in eachindex(basis_on_t)
            m = basis_on_t[bi]
            for bj in eachindex(basis_on_t)
                n = basis_on_t[bj]

                # Compute ∫_t f_m · f_n dS
                val = zero(Tmass)
                for q in 1:Nq
                    rq = quad_pts[t][q]
                    fm = eval_rwg(rwg, m, rq, t)
                    fn = eval_rwg(rwg, n, rq, t)
                    val += wq[q] * dot(fm, fn)
                end
                val *= 2 * A  # reference-to-physical scaling

                if val != zero(Tmass)
                    push!(rows_p[p], m)
                    push!(cols_p[p], n)
                    push!(vals_p[p], val)
                end
            end
        end
    end

    return [LocalMassMatrix(N, rows_p[p], cols_p[p], vals_p[p]) for p in 1:P]
end

"""
    assemble_Z_impedance(Mp, theta)

Assemble the impedance contribution to the MoM matrix:
Z_imp = -Σ_p θ_p M_p
"""
function assemble_Z_impedance(Mp::Vector{<:AbstractMatrix}, theta::AbstractVector)
    N = size(Mp[1], 1)
    CT = eltype(theta) <: Complex ? eltype(theta) : ComplexF64
    Z_imp = zeros(CT, N, N)
    for p in eachindex(theta)
        _add_scaled_matrix!(Z_imp, -theta[p], Mp[p])
    end
    return Z_imp
end

"""
    assemble_dZ_dtheta(Mp, p)

Return ∂Z/∂θ_p = -M_p (exact, closed-form derivative).
"""
function assemble_dZ_dtheta(Mp::Vector{<:AbstractMatrix}, p::Int)
    return -Mp[p]
end
