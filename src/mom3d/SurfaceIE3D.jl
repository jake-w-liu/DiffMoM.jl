# SurfaceIE3D.jl -- Dense dielectric surface integral equations
#
# This file implements closed-surface PMCHWT and Muller block systems using RWG
# electric and magnetic equivalent surface currents. The electric-current EFIE
# blocks reuse the package's singularity-treated EFIE assembly. The magnetic
# field K operator is assembled by principal-value product quadrature, with
# higher-order quadrature on near-singular panel pairs (those sharing at least
# one vertex: self, edge-adjacent, and vertex-touching).

export DielectricMedium3D, DielectricSIEResult3D
export dielectric_medium_3d, assemble_magnetic_field_operator_3d
export MatrixFreeMagneticFieldOperator3D, MatrixFreeDielectricSIE3D
export matrixfree_magnetic_field_operator_3d, matrixfree_dielectric_sie_operator_3d
export assemble_dielectric_sie_rhs_3d, assemble_dielectric_sie_3d
export assemble_pmchwt_3d, assemble_muller_3d, solve_dielectric_sie_3d

struct DielectricMedium3D
    eps_r::ComplexF64
    mu_r::ComplexF64
    k::ComplexF64
    eta::ComplexF64
end

struct DielectricSIEResult3D{TA<:AbstractMatrix{ComplexF64},TLU,TStats}
    J::Vector{ComplexF64}
    M::Vector{ComplexF64}
    A::TA
    rhs::Vector{ComplexF64}
    A_LU::TLU
    solver::Symbol
    stats::TStats
    formulation::Symbol
    exterior::DielectricMedium3D
    interior::DielectricMedium3D
end

struct MatrixFreeMagneticFieldOperator3D <: AbstractMatrix{ComplexF64}
    mesh::TriMesh
    rwg::RWGData
    k::ComplexF64
    wq::Vector{Float64}
    pts::Vector{Vector{Vec3}}
    areas::Vector{Float64}
    wq_hi::Vector{Float64}
    pts_hi::Vector{Vector{Vec3}}
    areas_hi::Vector{Float64}
    near_pairs::BitMatrix
end

mutable struct MatrixFreeDielectricSIE3D{
        TZe<:MatrixFreeEFIEOperator,
        TZh<:MatrixFreeEFIEOperator,
        TK<:MatrixFreeMagneticFieldOperator3D} <: AbstractMatrix{ComplexF64}
    formulation::Symbol
    exterior::DielectricMedium3D
    interior::DielectricMedium3D
    Ze_ext::TZe
    Ze_int::TZe
    Zh_ext::TZh
    Zh_int::TZh
    K_ext::TK
    K_int::TK
    c_ze_ext::ComplexF64
    c_ze_int::ComplexF64
    c_zh_ext::ComplexF64
    c_zh_int::ComplexF64
    # Second-kind identity (nhat x Gram) residue omitted by the PV K operator.
    # c_g_e/c_g_h are the off-diagonal Gram coefficients for the E-/H-rows.
    # For PMCHWT they are zero (equal row weights); the matvec then skips Gram.
    Gram::Matrix{ComplexF64}
    c_g_e::ComplexF64
    c_g_h::ComplexF64
    work_J::Vector{ComplexF64}
    work_M::Vector{ComplexF64}
    tmp1::Vector{ComplexF64}
    tmp2::Vector{ComplexF64}
    tmp3::Vector{ComplexF64}
    tmp4::Vector{ComplexF64}
    tmp5::Vector{ComplexF64}
end

function _finite_complex_surface_3d(x, label::AbstractString)
    z = ComplexF64(x)
    isfinite(real(z)) && isfinite(imag(z)) ||
        error("$label must be finite, got $z.")
    return z
end

function dielectric_medium_3d(k0::Real, eps_r=1.0 + 0im, mu_r=1.0 + 0im;
                              eta0::Real=_ETA0_DDA)
    k0f = Float64(k0)
    k0f > 0 || error("k0 must be positive.")
    eta0f = Float64(eta0)
    eta0f > 0 || error("eta0 must be positive.")
    epsc = _finite_complex_surface_3d(eps_r, "eps_r")
    muc = _finite_complex_surface_3d(mu_r, "mu_r")
    abs(epsc) > 0 || error("eps_r must be nonzero.")
    abs(muc) > 0 || error("mu_r must be nonzero.")
    return DielectricMedium3D(epsc, muc, k0f * sqrt(epsc * muc),
                              eta0f * sqrt(muc / epsc))
end

function _assert_closed_surface_sie_3d(mesh::TriMesh, rwg::RWGData;
                                       mesh_precheck::Bool=true,
                                       area_tol_rel::Float64=1e-12)
    rwg.has_periodic_bloch &&
        error("Dielectric 3D SIE requires non-periodic closed-surface RWG basis functions.")
    mesh === rwg.mesh || error("RWG data must be built from the same mesh object.")
    if mesh_precheck
        assert_mesh_quality(mesh;
            allow_boundary=false,
            require_closed=true,
            area_tol_rel=area_tol_rel,
        )
    end
    return nothing
end

# Triangle pairs that are near-singular for the magnetic-field K kernel, i.e.
# pairs sharing at least one vertex (this subsumes edge-sharing pairs, which
# share two vertices). These pairs use the higher-order quadrature rule. The
# 1/R singularity of grad(G) makes both edge-touching AND vertex-touching pairs
# under-integrated by the coarse far rule; promoting vertex-touching pairs to
# the high-order rule measurably reduces the K-operator quadrature error toward
# a fine-quadrature reference (verified ~0.99%->0.27% on an icosphere).
function _triangle_near_pairs_3d(mesh::TriMesh)
    Nt = ntriangles(mesh)
    near = falses(Nt, Nt)
    vert_to_tri = Dict{Int, Vector{Int}}()
    for t in 1:Nt
        for iv in 1:3
            v = mesh.tri[iv, t]
            push!(get!(vert_to_tri, v, Int[]), t)
        end
    end
    for tris in values(vert_to_tri)
        for i in 1:length(tris), j in (i + 1):length(tris)
            near[tris[i], tris[j]] = true
            near[tris[j], tris[i]] = true
        end
    end
    return near
end

function _surface_quad_cache_3d(mesh::TriMesh, order::Int)
    xi, wq = tri_quad_rule(order)
    Nt = ntriangles(mesh)
    pts = Vector{Vector{Vec3}}(undef, Nt)
    areas = Vector{Float64}(undef, Nt)
    for t in 1:Nt
        pts[t] = tri_quad_points(mesh, t, xi)
        areas[t] = triangle_area(mesh, t)
    end
    return wq, pts, areas
end

"""
    _ncross_gram_3d(mesh, rwg; quad_order=3)

Assemble the second-kind identity overlap matrix
`G[m,n] = <f_m, nhat x f_n>` integrated over the surface. Because two RWG
functions overlap only on triangles they both support, the integral is taken
over the common triangle(s) using the outward triangle normal `nhat`. This is
the principal-value residue ("nhat x Gram") term that the magnetic-field K
operator omits; in the Muller system it appears off-diagonal with coefficient
`(c_ext - c_int)*(1/2)`. (It cancels identically in PMCHWT, where the two
off-diagonal coefficients are equal, so PMCHWT is unaffected.)
"""
function _ncross_gram_3d(mesh::TriMesh, rwg::RWGData; quad_order::Int=3)
    xi, wq = tri_quad_rule(quad_order)
    Nt = ntriangles(mesh)
    N = rwg.nedges
    pts = Vector{Vector{Vec3}}(undef, Nt)
    areas = Vector{Float64}(undef, Nt)
    normals = Vector{Vec3}(undef, Nt)
    for t in 1:Nt
        pts[t] = tri_quad_points(mesh, t, xi)
        areas[t] = triangle_area(mesh, t)
        normals[t] = triangle_normal(mesh, t)
    end
    G = zeros(ComplexF64, N, N)
    Threads.@threads for m in 1:N
        tmp = (rwg.tplus[m], rwg.tminus[m])
        @inbounds for n in 1:N
            tnp = (rwg.tplus[n], rwg.tminus[n])
            acc = 0.0 + 0.0im
            for t in tmp
                (t == tnp[1] || t == tnp[2]) || continue
                A = areas[t]
                nh = normals[t]
                ptst = pts[t]
                for q in eachindex(wq)
                    r = ptst[q]
                    fm = eval_rwg(rwg, m, r, t)
                    fn = eval_rwg(rwg, n, r, t)
                    acc += wq[q] * dot(fm, cross(nh, fn)) * (2 * A)
                end
            end
            G[m, n] = acc
        end
    end
    return G
end

@inline function _mfie_triangle_pair_entry_3d(rwg::RWGData, m::Int, n::Int,
                                             tm::Int, tn::Int,
                                             k, wq, pts, areas)
    val = 0.0 + 0.0im
    Am = areas[tm]
    An = areas[tn]
    @inbounds for qm in eachindex(wq)
        rm = pts[tm][qm]
        fm = eval_rwg(rwg, m, rm, tm)
        for qn in eachindex(wq)
            rn = pts[tn][qn]
            fn = eval_rwg(rwg, n, rn, tn)
            kernel = cross(grad_greens(rm, rn, k), fn)
            val += wq[qm] * wq[qn] * dot(fm, kernel) * (2 * Am) * (2 * An)
        end
    end
    return val
end

"""
    assemble_magnetic_field_operator_3d(mesh, rwg, k; quad_order=3)

Assemble the dense magnetic-field principal-value operator
`K[m,n] = <f_m, PV ∫ grad(G) x f_n dS'>`. This is the off-diagonal surface
current coupling used in PMCHWT/Muller systems.
"""
function assemble_magnetic_field_operator_3d(mesh::TriMesh, rwg::RWGData, k;
                                             quad_order::Int=3,
                                             singular_quad_order::Int=7,
                                             mesh_precheck::Bool=true,
                                             area_tol_rel::Float64=1e-12)
    _assert_closed_surface_sie_3d(mesh, rwg;
                                  mesh_precheck=mesh_precheck,
                                  area_tol_rel=area_tol_rel)
    N = rwg.nedges
    K = zeros(ComplexF64, N, N)
    wq, pts, areas = _surface_quad_cache_3d(mesh, quad_order)
    wq_hi, pts_hi, areas_hi = _surface_quad_cache_3d(mesh, singular_quad_order)
    near_pairs = _triangle_near_pairs_3d(mesh)

    Threads.@threads for m in 1:N
        @inbounds for n in 1:N
            acc = 0.0 + 0.0im
            for tm in (rwg.tplus[m], rwg.tminus[m])
                for tn in (rwg.tplus[n], rwg.tminus[n])
                    if tm == tn || near_pairs[tm, tn]
                        acc += _mfie_triangle_pair_entry_3d(
                            rwg, m, n, tm, tn, k, wq_hi, pts_hi, areas_hi,
                        )
                    else
                        acc += _mfie_triangle_pair_entry_3d(
                            rwg, m, n, tm, tn, k, wq, pts, areas,
                        )
                    end
                end
            end
            K[m, n] = acc
        end
    end
    return K
end

function matrixfree_magnetic_field_operator_3d(mesh::TriMesh, rwg::RWGData, k;
                                               quad_order::Int=3,
                                               singular_quad_order::Int=7,
                                               mesh_precheck::Bool=true,
                                               area_tol_rel::Float64=1e-12)
    _assert_closed_surface_sie_3d(mesh, rwg;
                                  mesh_precheck=mesh_precheck,
                                  area_tol_rel=area_tol_rel)
    wq, pts, areas = _surface_quad_cache_3d(mesh, quad_order)
    wq_hi, pts_hi, areas_hi = _surface_quad_cache_3d(mesh, singular_quad_order)
    near_pairs = _triangle_near_pairs_3d(mesh)
    return MatrixFreeMagneticFieldOperator3D(mesh, rwg, ComplexF64(k),
                                             wq, pts, areas,
                                             wq_hi, pts_hi, areas_hi,
                                             near_pairs)
end

Base.size(A::MatrixFreeMagneticFieldOperator3D) = (A.rwg.nedges, A.rwg.nedges)
Base.size(A::MatrixFreeMagneticFieldOperator3D, d::Int) = d <= 2 ? A.rwg.nedges : 1
Base.eltype(::Type{MatrixFreeMagneticFieldOperator3D}) = ComplexF64
Base.eltype(::MatrixFreeMagneticFieldOperator3D) = ComplexF64

@inline function _mfie_entry_3d(A::MatrixFreeMagneticFieldOperator3D, m::Int, n::Int)
    1 <= m <= A.rwg.nedges || throw(BoundsError(A, (m, n)))
    1 <= n <= A.rwg.nedges || throw(BoundsError(A, (m, n)))
    acc = 0.0 + 0.0im
    for tm in (A.rwg.tplus[m], A.rwg.tminus[m])
        for tn in (A.rwg.tplus[n], A.rwg.tminus[n])
            if tm == tn || A.near_pairs[tm, tn]
                acc += _mfie_triangle_pair_entry_3d(
                    A.rwg, m, n, tm, tn, A.k, A.wq_hi, A.pts_hi, A.areas_hi,
                )
            else
                acc += _mfie_triangle_pair_entry_3d(
                    A.rwg, m, n, tm, tn, A.k, A.wq, A.pts, A.areas,
                )
            end
        end
    end
    return acc
end

Base.getindex(A::MatrixFreeMagneticFieldOperator3D, i::Int, j::Int) =
    _mfie_entry_3d(A, i, j)

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                            A::MatrixFreeMagneticFieldOperator3D,
                            x::AbstractVector)
    N = size(A, 1)
    length(x) == N || throw(DimensionMismatch("x length $(length(x)) != $N"))
    length(y) == N || throw(DimensionMismatch("y length $(length(y)) != $N"))
    xread = y === x ? copy(x) : x
    @inbounds for m in 1:N
        acc = 0.0 + 0.0im
        for n in 1:N
            acc += _mfie_entry_3d(A, m, n) * xread[n]
        end
        y[m] = acc
    end
    return y
end

function Base.:*(A::MatrixFreeMagneticFieldOperator3D, x::AbstractVector)
    y = zeros(ComplexF64, size(A, 1))
    mul!(y, A, x)
    return y
end

function _assemble_plane_wave_h_rhs_3d(mesh::TriMesh, rwg::RWGData,
                                       pw::PlaneWaveExcitation, eta::Complex;
                                       quad_order::Int=3)
    N = rwg.nedges
    wq, quad_pts, areas = _excitation_quadrature_cache(mesh, quad_order)
    khat = pw.k_vec / norm(pw.k_vec)
    rhs = zeros(ComplexF64, N)
    for n in 1:N
        for t in (rwg.tplus[n], rwg.tminus[n])
            A = areas[t]
            pts = quad_pts[t]
            for q in eachindex(wq)
                rq = pts[q]
                fn = eval_rwg(rwg, n, rq, t)
                Einc = plane_wave_field(rq, pw.k_vec, pw.E0, pw.pol)
                Hinc = cross(khat, Einc) / eta
                rhs[n] += -wq[q] * dot(fn, Hinc) * (2 * A)
            end
        end
    end
    return rhs
end

"""
    assemble_dielectric_sie_rhs_3d(mesh, rwg, excitation, exterior; quad_order=3,
                                   formulation=:pmchwt, interior=nothing)

Assemble the PMCHWT/Muller right-hand side `[v_E; v_H]` for a plane-wave
incident field in the exterior medium.

The incident field lives only in the exterior region, so the tested forcing is
`[v_E; v_H]` for PMCHWT. For Muller the exterior equations are scaled by the
exterior row weights, so the RHS becomes `[c_ze_ext*v_E; c_zh_ext*v_H]`
consistently with the weighted block matrix. Pass the `interior` medium to
enable the Muller scaling.
"""
function assemble_dielectric_sie_rhs_3d(mesh::TriMesh, rwg::RWGData,
                                        excitation::PlaneWaveExcitation,
                                        exterior::DielectricMedium3D;
                                        quad_order::Int=3,
                                        formulation::Symbol=:pmchwt,
                                        interior::Union{Nothing,DielectricMedium3D}=nothing)
    vE = assemble_excitation(mesh, rwg, excitation; quad_order=quad_order)
    vH = _assemble_plane_wave_h_rhs_3d(mesh, rwg, excitation, exterior.eta;
                                       quad_order=quad_order)
    if formulation == :muller
        interior === nothing &&
            error("Muller RHS scaling requires the interior medium.")
        c_ze_ext, _, c_zh_ext, _ =
            _surface_sie_coefficients_3d(:muller, exterior, interior)
        return vcat(c_ze_ext .* vE, c_zh_ext .* vH)
    end
    return vcat(vE, vH)
end

function _surface_sie_blocks_3d(mesh::TriMesh, rwg::RWGData, k0::Real,
                                epsr_in, mur_in, epsr_ext, mur_ext;
                                formulation::Symbol,
                                quad_order::Int=3,
                                singular_quad_order::Int=7,
                                eta0::Real=_ETA0_DDA,
                                mesh_precheck::Bool=true,
                                area_tol_rel::Float64=1e-12)
    formulation in (:pmchwt, :muller) ||
        error("Unsupported dielectric SIE formulation: $formulation (expected :pmchwt or :muller).")
    _assert_closed_surface_sie_3d(mesh, rwg;
                                  mesh_precheck=mesh_precheck,
                                  area_tol_rel=area_tol_rel)

    exterior = dielectric_medium_3d(k0, epsr_ext, mur_ext; eta0=eta0)
    interior = dielectric_medium_3d(k0, epsr_in, mur_in; eta0=eta0)

    Ze_ext = assemble_Z_efie(mesh, rwg, exterior.k;
                             quad_order=quad_order,
                             eta0=exterior.eta,
                             mesh_precheck=false)
    Ze_int = assemble_Z_efie(mesh, rwg, interior.k;
                             quad_order=quad_order,
                             eta0=interior.eta,
                             mesh_precheck=false)
    Zh_ext = assemble_Z_efie(mesh, rwg, exterior.k;
                             quad_order=quad_order,
                             eta0=1 / exterior.eta,
                             mesh_precheck=false)
    Zh_int = assemble_Z_efie(mesh, rwg, interior.k;
                             quad_order=quad_order,
                             eta0=1 / interior.eta,
                             mesh_precheck=false)
    K_ext = assemble_magnetic_field_operator_3d(
        mesh, rwg, exterior.k;
        quad_order=quad_order,
        singular_quad_order=singular_quad_order,
        mesh_precheck=false,
    )
    K_int = assemble_magnetic_field_operator_3d(
        mesh, rwg, interior.k;
        quad_order=quad_order,
        singular_quad_order=singular_quad_order,
        mesh_precheck=false,
    )

    # Row weights (PMCHWT => all 1; Muller => mu/eps weights). The SAME weights
    # multiply the diagonal T blocks and the off-diagonal K blocks of each region.
    c_ze_ext, c_ze_int, c_zh_ext, c_zh_int =
        _surface_sie_coefficients_3d(formulation, exterior, interior)

    # Diagonal T blocks (mu/eps-weighted per row).
    A11 = c_ze_ext * Ze_ext + c_ze_int * Ze_int
    A22 = c_zh_ext * Zh_ext + c_zh_int * Zh_int

    # Off-diagonal K blocks: weighted principal-value K plus the second-kind
    # identity (nhat x Gram) residue that the PV K operator omits. The E-row
    # carries K with sign -, the H-row with sign +. The identity term enters
    # with coefficient (c_ext - c_int)*(1/2); it cancels in PMCHWT (equal
    # coefficients) and is essential for the Muller currents to match PMCHWT.
    K_E = c_ze_ext * K_ext + c_ze_int * K_int
    K_H = c_zh_ext * K_ext + c_zh_int * K_int
    if (c_ze_ext != c_ze_int) || (c_zh_ext != c_zh_int)
        Gram = _ncross_gram_3d(mesh, rwg; quad_order=quad_order)
        A12 = -(K_E + (c_ze_ext - c_ze_int) * 0.5 * Gram)
        A21 = K_H + (c_zh_ext - c_zh_int) * 0.5 * Gram
    else
        A12 = -K_E
        A21 = K_H
    end

    A = [A11 A12; A21 A22]
    return A, exterior, interior
end

function _surface_sie_coefficients_3d(formulation::Symbol,
                                      exterior::DielectricMedium3D,
                                      interior::DielectricMedium3D)
    if formulation == :pmchwt
        return (1.0 + 0im, 1.0 + 0im, 1.0 + 0im, 1.0 + 0im)
    elseif formulation == :muller
        mu_sum = exterior.mu_r + interior.mu_r
        eps_sum = exterior.eps_r + interior.eps_r
        abs(mu_sum) > 0 || error("Muller formulation is singular for mu_ext + mu_in = 0.")
        abs(eps_sum) > 0 || error("Muller formulation is singular for eps_ext + eps_in = 0.")
        return (interior.mu_r / mu_sum,
                exterior.mu_r / mu_sum,
                interior.eps_r / eps_sum,
                exterior.eps_r / eps_sum)
    else
        error("Unsupported dielectric SIE formulation: $formulation (expected :pmchwt or :muller).")
    end
end

function matrixfree_dielectric_sie_operator_3d(mesh::TriMesh, rwg::RWGData,
                                               k0::Real, epsr_in=1.0 + 0im;
                                               mur_in=1.0 + 0im,
                                               epsr_ext=1.0 + 0im,
                                               mur_ext=1.0 + 0im,
                                               formulation::Symbol=:pmchwt,
                                               quad_order::Int=3,
                                               singular_quad_order::Int=7,
                                               eta0::Real=_ETA0_DDA,
                                               mesh_precheck::Bool=true,
                                               area_tol_rel::Float64=1e-12)
    formulation in (:pmchwt, :muller) ||
        error("Unsupported dielectric SIE formulation: $formulation (expected :pmchwt or :muller).")
    _assert_closed_surface_sie_3d(mesh, rwg;
                                  mesh_precheck=mesh_precheck,
                                  area_tol_rel=area_tol_rel)
    exterior = dielectric_medium_3d(k0, epsr_ext, mur_ext; eta0=eta0)
    interior = dielectric_medium_3d(k0, epsr_in, mur_in; eta0=eta0)
    c_ze_ext, c_ze_int, c_zh_ext, c_zh_int =
        _surface_sie_coefficients_3d(formulation, exterior, interior)

    Ze_ext = matrixfree_efie_operator(mesh, rwg, exterior.k;
                                      quad_order=quad_order,
                                      eta0=exterior.eta,
                                      mesh_precheck=false)
    Ze_int = matrixfree_efie_operator(mesh, rwg, interior.k;
                                      quad_order=quad_order,
                                      eta0=interior.eta,
                                      mesh_precheck=false)
    Zh_ext = matrixfree_efie_operator(mesh, rwg, exterior.k;
                                      quad_order=quad_order,
                                      eta0=1 / exterior.eta,
                                      mesh_precheck=false)
    Zh_int = matrixfree_efie_operator(mesh, rwg, interior.k;
                                      quad_order=quad_order,
                                      eta0=1 / interior.eta,
                                      mesh_precheck=false)
    K_ext = matrixfree_magnetic_field_operator_3d(
        mesh, rwg, exterior.k;
        quad_order=quad_order,
        singular_quad_order=singular_quad_order,
        mesh_precheck=false,
    )
    K_int = matrixfree_magnetic_field_operator_3d(
        mesh, rwg, interior.k;
        quad_order=quad_order,
        singular_quad_order=singular_quad_order,
        mesh_precheck=false,
    )

    N = rwg.nedges
    # Off-diagonal second-kind identity (nhat x Gram) residue coefficients.
    # E-row K block enters with sign -, H-row with sign +.
    c_g_e = -(c_ze_ext - c_ze_int) * 0.5
    c_g_h = (c_zh_ext - c_zh_int) * 0.5
    Gram = (c_g_e != 0 || c_g_h != 0) ?
        _ncross_gram_3d(mesh, rwg; quad_order=quad_order) :
        zeros(ComplexF64, 0, 0)
    return MatrixFreeDielectricSIE3D(
        formulation, exterior, interior,
        Ze_ext, Ze_int, Zh_ext, Zh_int, K_ext, K_int,
        ComplexF64(c_ze_ext), ComplexF64(c_ze_int),
        ComplexF64(c_zh_ext), ComplexF64(c_zh_int),
        Gram, ComplexF64(c_g_e), ComplexF64(c_g_h),
        zeros(ComplexF64, N), zeros(ComplexF64, N),
        zeros(ComplexF64, N), zeros(ComplexF64, N),
        zeros(ComplexF64, N), zeros(ComplexF64, N),
        zeros(ComplexF64, N),
    )
end

Base.size(A::MatrixFreeDielectricSIE3D) = (2 * A.Ze_ext.cache.rwg.nedges,
                                           2 * A.Ze_ext.cache.rwg.nedges)
Base.size(A::MatrixFreeDielectricSIE3D, d::Int) =
    d <= 2 ? 2 * A.Ze_ext.cache.rwg.nedges : 1
Base.eltype(::Type{<:MatrixFreeDielectricSIE3D}) = ComplexF64
Base.eltype(::MatrixFreeDielectricSIE3D) = ComplexF64

function Base.getindex(A::MatrixFreeDielectricSIE3D, row::Int, col::Int)
    N = A.Ze_ext.cache.rwg.nedges
    1 <= row <= 2N || throw(BoundsError(A, (row, col)))
    1 <= col <= 2N || throw(BoundsError(A, (row, col)))
    if row <= N && col <= N
        return A.c_ze_ext * A.Ze_ext[row, col] + A.c_ze_int * A.Ze_int[row, col]
    elseif row <= N
        # E-row off-diagonal: -(c_ze_ext K_ext + c_ze_int K_int) + c_g_e * Gram
        c = col - N
        val = -(A.c_ze_ext * A.K_ext[row, c] + A.c_ze_int * A.K_int[row, c])
        if A.c_g_e != 0
            val += A.c_g_e * A.Gram[row, c]
        end
        return val
    elseif col <= N
        # H-row off-diagonal: (c_zh_ext K_ext + c_zh_int K_int) + c_g_h * Gram
        r = row - N
        val = A.c_zh_ext * A.K_ext[r, col] + A.c_zh_int * A.K_int[r, col]
        if A.c_g_h != 0
            val += A.c_g_h * A.Gram[r, col]
        end
        return val
    else
        r = row - N
        c = col - N
        return A.c_zh_ext * A.Zh_ext[r, c] + A.c_zh_int * A.Zh_int[r, c]
    end
end

@inline function _copy_block_inputs_3d!(J::Vector{ComplexF64}, M::Vector{ComplexF64},
                                       x::AbstractVector)
    N = length(J)
    @inbounds for j in 1:N
        J[j] = ComplexF64(x[j])
        M[j] = ComplexF64(x[N + j])
    end
    return nothing
end

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                            A::MatrixFreeDielectricSIE3D,
                            x::AbstractVector{ComplexF64},
                            alpha_scale::Number,
                            beta_scale::Number)
    N2 = size(A, 1)
    N = div(N2, 2)
    length(x) == N2 || throw(DimensionMismatch("x length must be $N2."))
    length(y) == N2 || throw(DimensionMismatch("y length must be $N2."))

    _copy_block_inputs_3d!(A.work_J, A.work_M, x)

    # E-row: c_ze_ext Ze_ext J + c_ze_int Ze_int J
    #        - (c_ze_ext K_ext + c_ze_int K_int) M + c_g_e Gram M
    mul!(A.tmp1, A.Ze_ext, A.work_J)
    mul!(A.tmp2, A.Ze_int, A.work_J)
    mul!(A.tmp3, A.K_ext, A.work_M)
    mul!(A.tmp4, A.K_int, A.work_M)
    if A.c_g_e != 0
        mul!(A.tmp5, A.Gram, A.work_M)
    end
    @inbounds for j in 1:N
        v = A.c_ze_ext * A.tmp1[j] + A.c_ze_int * A.tmp2[j] -
            (A.c_ze_ext * A.tmp3[j] + A.c_ze_int * A.tmp4[j])
        if A.c_g_e != 0
            v += A.c_g_e * A.tmp5[j]
        end
        y[j] = alpha_scale * v + beta_scale * y[j]
    end

    # H-row: (c_zh_ext K_ext + c_zh_int K_int) J + c_g_h Gram J
    #        + c_zh_ext Zh_ext M + c_zh_int Zh_int M
    mul!(A.tmp1, A.K_ext, A.work_J)
    mul!(A.tmp2, A.K_int, A.work_J)
    mul!(A.tmp3, A.Zh_ext, A.work_M)
    mul!(A.tmp4, A.Zh_int, A.work_M)
    if A.c_g_h != 0
        mul!(A.tmp5, A.Gram, A.work_J)
    end
    @inbounds for j in 1:N
        v = A.c_zh_ext * A.tmp1[j] + A.c_zh_int * A.tmp2[j] +
            A.c_zh_ext * A.tmp3[j] + A.c_zh_int * A.tmp4[j]
        if A.c_g_h != 0
            v += A.c_g_h * A.tmp5[j]
        end
        y[N + j] = alpha_scale * v + beta_scale * y[N + j]
    end
    return y
end

LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                   A::MatrixFreeDielectricSIE3D,
                   x::AbstractVector{ComplexF64}) =
    LinearAlgebra.mul!(y, A, x, one(ComplexF64), zero(ComplexF64))

function Base.:*(A::MatrixFreeDielectricSIE3D, x::AbstractVector)
    y = zeros(ComplexF64, size(A, 1))
    mul!(y, A, ComplexF64.(collect(x)))
    return y
end

"""
    assemble_dielectric_sie_3d(mesh, rwg, k0, epsr_in; formulation=:pmchwt, ...)

Assemble a dense closed-surface dielectric SIE matrix for isotropic homogeneous
interior/exterior media. Unknowns are stacked RWG coefficients `[J; M]`.
"""
function assemble_dielectric_sie_3d(mesh::TriMesh, rwg::RWGData, k0::Real,
                                    epsr_in=1.0 + 0im;
                                    mur_in=1.0 + 0im,
                                    epsr_ext=1.0 + 0im,
                                    mur_ext=1.0 + 0im,
                                    formulation::Symbol=:pmchwt,
                                    quad_order::Int=3,
                                    singular_quad_order::Int=7,
                                    eta0::Real=_ETA0_DDA,
                                    mesh_precheck::Bool=true,
                                    area_tol_rel::Float64=1e-12)
    A, _, _ = _surface_sie_blocks_3d(
        mesh, rwg, k0, epsr_in, mur_in, epsr_ext, mur_ext;
        formulation=formulation,
        quad_order=quad_order,
        singular_quad_order=singular_quad_order,
        eta0=eta0,
        mesh_precheck=mesh_precheck,
        area_tol_rel=area_tol_rel,
    )
    return A
end

assemble_pmchwt_3d(mesh::TriMesh, rwg::RWGData, k0::Real, epsr_in=1.0 + 0im; kwargs...) =
    assemble_dielectric_sie_3d(mesh, rwg, k0, epsr_in; formulation=:pmchwt, kwargs...)

# Second-kind Müller system: μ/ε-weighted diagonal T AND off-diagonal K blocks,
# weighted RHS, plus the (n̂× Gram) identity residue the principal-value K operator
# omits. Verified to match PMCHWT currents to <1% on a sphere (tightening under
# refinement), so it is a validated alternative to assemble_pmchwt_3d.
assemble_muller_3d(mesh::TriMesh, rwg::RWGData, k0::Real, epsr_in=1.0 + 0im; kwargs...) =
    assemble_dielectric_sie_3d(mesh, rwg, k0, epsr_in; formulation=:muller, kwargs...)

function _split_surface_currents_3d(x::AbstractVector{ComplexF64}, N::Int)
    length(x) == 2N || error("surface-current vector length ($(length(x))) must be 2N ($(2N)).")
    return x[1:N], x[(N + 1):(2N)]
end

"""
    solve_dielectric_sie_3d(mesh, rwg, k0, epsr_in, rhs; formulation=:pmchwt, solver=:direct, ...)

Solve a dense PMCHWT/Muller dielectric SIE system. `rhs` may be either a
length-`2N` vector or a `PlaneWaveExcitation`. Set `solver=:gmres` to use the
matrix-free operator instead of forming the dense matrix.
"""
function solve_dielectric_sie_3d(mesh::TriMesh, rwg::RWGData, k0::Real,
                                 epsr_in, rhs::AbstractVector{<:Number};
                                 mur_in=1.0 + 0im,
                                 epsr_ext=1.0 + 0im,
                                 mur_ext=1.0 + 0im,
                                 formulation::Symbol=:pmchwt,
                                 solver::Symbol=:direct,
                                 quad_order::Int=3,
                                 singular_quad_order::Int=7,
                                 eta0::Real=_ETA0_DDA,
                                 mesh_precheck::Bool=true,
                                 area_tol_rel::Float64=1e-12,
                                 tol::Float64=1e-8,
                                 maxiter::Int=200,
                                 memory::Int=20,
                                 verbose::Bool=false)
    rhsv = ComplexF64.(collect(rhs))
    if solver == :direct
        A, exterior, interior = _surface_sie_blocks_3d(
            mesh, rwg, k0, epsr_in, mur_in, epsr_ext, mur_ext;
            formulation=formulation,
            quad_order=quad_order,
            singular_quad_order=singular_quad_order,
            eta0=eta0,
            mesh_precheck=mesh_precheck,
            area_tol_rel=area_tol_rel,
        )
        length(rhsv) == size(A, 1) ||
            error("rhs length ($(length(rhsv))) must match dielectric SIE size ($(size(A, 1))).")
        fac = lu(A)
        x = fac \ rhsv
        J, M = _split_surface_currents_3d(x, rwg.nedges)
        return DielectricSIEResult3D(copy(J), copy(M), A, rhsv, fac,
                                     :direct, nothing,
                                     formulation, exterior, interior)
    elseif solver == :gmres
        A = matrixfree_dielectric_sie_operator_3d(
            mesh, rwg, k0, epsr_in;
            mur_in=mur_in,
            epsr_ext=epsr_ext,
            mur_ext=mur_ext,
            formulation=formulation,
            quad_order=quad_order,
            singular_quad_order=singular_quad_order,
            eta0=eta0,
            mesh_precheck=mesh_precheck,
            area_tol_rel=area_tol_rel,
        )
        length(rhsv) == size(A, 1) ||
            error("rhs length ($(length(rhsv))) must match dielectric SIE size ($(size(A, 1))).")
        x, stats = Krylov.gmres(A, rhsv;
                                memory=memory,
                                rtol=tol,
                                atol=0.0,
                                itmax=maxiter,
                                verbose=(verbose ? 1 : 0))
        J, M = _split_surface_currents_3d(x, rwg.nedges)
        return DielectricSIEResult3D(copy(J), copy(M), A, rhsv, nothing,
                                     :gmres, stats,
                                     formulation, A.exterior, A.interior)
    else
        error("Unsupported dielectric SIE solver: $solver (expected :direct or :gmres).")
    end
end

function solve_dielectric_sie_3d(mesh::TriMesh, rwg::RWGData, k0::Real,
                                 epsr_in, excitation::PlaneWaveExcitation;
                                 mur_in=1.0 + 0im,
                                 epsr_ext=1.0 + 0im,
                                 mur_ext=1.0 + 0im,
                                 formulation::Symbol=:pmchwt,
                                 solver::Symbol=:direct,
                                 quad_order::Int=3,
                                 singular_quad_order::Int=7,
                                 eta0::Real=_ETA0_DDA,
                                 mesh_precheck::Bool=true,
                                 area_tol_rel::Float64=1e-12,
                                 tol::Float64=1e-8,
                                 maxiter::Int=200,
                                 memory::Int=20,
                                 verbose::Bool=false)
    exterior = dielectric_medium_3d(k0, epsr_ext, mur_ext; eta0=eta0)
    interior = dielectric_medium_3d(k0, epsr_in, mur_in; eta0=eta0)
    rhs = assemble_dielectric_sie_rhs_3d(
        mesh, rwg, excitation, exterior;
        quad_order=quad_order,
        formulation=formulation,
        interior=interior,
    )
    return solve_dielectric_sie_3d(
        mesh, rwg, k0, epsr_in, rhs;
        mur_in=mur_in,
        epsr_ext=epsr_ext,
        mur_ext=mur_ext,
        formulation=formulation,
        solver=solver,
        quad_order=quad_order,
        singular_quad_order=singular_quad_order,
        eta0=eta0,
        mesh_precheck=mesh_precheck,
        area_tol_rel=area_tol_rel,
        tol=tol,
        maxiter=maxiter,
        memory=memory,
        verbose=verbose,
    )
end
