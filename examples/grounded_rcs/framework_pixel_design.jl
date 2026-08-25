# Ground-backed RCS-reduction helpers with separate design and analysis meshes.
# The coarse square-pixel design grid is mapped onto a finer analysis mesh.
using DiffMoM, LinearAlgebra, StaticArrays, Random, Printf
const C0 = 2.99792458e8

# Fast O(N) (0,0)-mode reflection vector s (R_cur00 = sᵀ I), normal incidence, x-pol.
# Equivalent to N unit-current calls to reflection_coefficients but one pass.
function svec_fast(mesh, rwg, k, lat; quad_order=3, E0=1.0, eta0=376.730313668)
    xi, wq = tri_quad_rule(quad_order); Nq = length(wq); Nt = ntriangles(mesh); N = rwg.nedges
    A_cell = lat.dx * lat.dy
    tri_to_basis = [Int[] for _ in 1:Nt]
    for n in 1:N
        push!(tri_to_basis[rwg.tplus[n]], n); push!(tri_to_basis[rwg.tminus[n]], n)
    end
    Fx = zeros(Float64, N)                       # x-component of ∫ f_n
    for t in 1:Nt
        At = triangle_area(mesh, t); qp = tri_quad_points(mesh, t, xi)
        for q in 1:Nq, n in tri_to_basis[t]
            fn = eval_rwg(rwg, n, qp[q], t)
            Fx[n] += real(fn[1]) * wq[q] * (2 * At)
        end
    end
    return ComplexF64.(-(eta0) / (2 * E0 * A_cell) .* Fx)
end

# Conic (cone) filter weight matrix on the Npix×Npix pixel grid.
function conic_filter_matrix(Npix, dpix, rmin)
    Hf = zeros(Float64, Npix*Npix, Npix*Npix)
    ctr(i) = ( (i-1) % Npix, (i-1) ÷ Npix )  # (col,row)
    for i in 1:Npix*Npix
        ci, ri = ctr(i); ssum = 0.0
        for j in 1:Npix*Npix
            cj, rj = ctr(j)
            d = hypot((ci-cj)*dpix, (ri-rj)*dpix)
            wij = max(0.0, 1 - d/rmin)
            Hf[i,j] = wij; ssum += wij
        end
        Hf[i,:] ./= ssum
    end
    return Hf
end
