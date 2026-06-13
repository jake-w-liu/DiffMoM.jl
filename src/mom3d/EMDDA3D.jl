# EMDDA3D.jl -- Coupled electric-magnetic 3D discrete-dipole solver
#
# Convention: exp(+i omega t), matching DDA3D.jl. Unknowns are total
# electric and magnetic fields at voxel centers. The induced normalized dipoles
# are [q; m] = alpha6 * [E; H], where q = p / eps0 and m is the magnetic dipole
# moment. The free-space interaction uses the same electric dyadic as DDA3D and
# its electromagnetic dual.

export BianisotropicPolarizability3D
export magnetic_clausius_mossotti_polarizability
export bianisotropic_clausius_mossotti_polarizability, em_dda_polarizabilities
export em_dda_operator_3d, assemble_em_dda_3d, solve_em_dda_3d
export planewave_em_dda_3d, induced_dipoles_em_dda_3d
export scattered_fields_em_dda_3d, farfield_em_dda_3d

const _ETA0_DDA = 376.730313668
const _CVec6DDA = SVector{6,ComplexF64}
const _CMat6DDA = SMatrix{6,6,ComplexF64,36}

"""
    BianisotropicPolarizability3D(alpha6)

Validated per-voxel `6 x 6` polarizability mapping total fields
`(Ex,Ey,Ez,Hx,Hy,Hz)` to normalized electric and magnetic dipoles
`(qx,qy,qz,mx,my,mz)`.
"""
struct BianisotropicPolarizability3D
    alpha::SMatrix{6,6,ComplexF64,36}
    function BianisotropicPolarizability3D(alpha6)
        return new(_as_cmat6_3d(alpha6, "alpha6"))
    end
end

@inline _em_index(voxel::Int, comp::Int) = 6 * (voxel - 1) + comp
@inline _em_voxel(linear_index::Int) = div(linear_index - 1, 6) + 1
@inline _em_component(linear_index::Int) = mod(linear_index - 1, 6) + 1

function _as_cmat6_3d(m, label::AbstractString)
    size(m) == (6, 6) || error("$label must be a 6x6 tensor.")
    vals = ntuple(i -> ComplexF64(m[i]), 36)
    M = _CMat6DDA(vals)
    for c in M
        isfinite(real(c)) && isfinite(imag(c)) ||
            error("$label contains a non-finite component: $c.")
    end
    return M
end

@inline function _read_em_field6(x::AbstractVector, j::Int)
    return _CVec6DDA(x[_em_index(j, 1)], x[_em_index(j, 2)], x[_em_index(j, 3)],
                     x[_em_index(j, 4)], x[_em_index(j, 5)], x[_em_index(j, 6)])
end

@inline _split_em_field(v::_CVec6DDA) =
    (CVec3(v[1], v[2], v[3]), CVec3(v[4], v[5], v[6]))

@inline _join_em_field(E::CVec3, H::CVec3) =
    _CVec6DDA(E[1], E[2], E[3], H[1], H[2], H[3])

function _flatten_em_fields_3d(E_fields::AbstractVector, H_fields::AbstractVector,
                               n::Int, E_label::AbstractString,
                               H_label::AbstractString)
    length(E_fields) == n || error("$E_label length ($(length(E_fields))) must match nvoxels ($n).")
    length(H_fields) == n || error("$H_label length ($(length(H_fields))) must match nvoxels ($n).")
    out = Vector{ComplexF64}(undef, 6n)
    for j in 1:n
        E = _as_cvec3(E_fields[j], "$E_label[$j]")
        H = _as_cvec3(H_fields[j], "$H_label[$j]")
        out[_em_index(j, 1)] = E[1]
        out[_em_index(j, 2)] = E[2]
        out[_em_index(j, 3)] = E[3]
        out[_em_index(j, 4)] = H[1]
        out[_em_index(j, 5)] = H[2]
        out[_em_index(j, 6)] = H[3]
    end
    return out
end

function _unflatten_em_fields_3d(x::AbstractVector{ComplexF64}, n::Int)
    length(x) == 6n || error("field vector length ($(length(x))) must be 6*nvoxels ($(6n)).")
    E = Vector{CVec3}(undef, n)
    H = Vector{CVec3}(undef, n)
    for j in 1:n
        E[j] = CVec3(x[_em_index(j, 1)], x[_em_index(j, 2)], x[_em_index(j, 3)])
        H[j] = CVec3(x[_em_index(j, 4)], x[_em_index(j, 5)], x[_em_index(j, 6)])
    end
    return E, H
end

function _coerce_mur_material_3d(mu_r, n::Int)
    if mu_r isa Number
        return fill(ComplexF64(mu_r), n)
    elseif mu_r isa AbstractMatrix
        return fill(_as_cmat3(mu_r, "mu_r"), n)
    elseif _is_diag_tensor_tuple(mu_r)
        return fill(_diag_tensor_from_tuple(mu_r, "mu_r"), n)
    else
        length(mu_r) == n ||
            error("mu_r length ($(length(mu_r))) must match nvoxels ($n).")
        first_mu = mu_r[1]
        if first_mu isa Number
            muv = ComplexF64.(collect(mu_r))
            for j in 1:n
                muj = muv[j]
                isfinite(real(muj)) && isfinite(imag(muj)) ||
                    error("mu_r[$j] is not finite: $muj.")
            end
            return muv
        elseif first_mu isa AbstractMatrix
            return [_as_cmat3(mu_r[j], "mu_r[$j]") for j in 1:n]
        elseif _is_diag_tensor_tuple(first_mu)
            return [_diag_tensor_from_tuple(mu_r[j], "mu_r[$j]") for j in 1:n]
        else
            error("Unsupported mu_r material specification: $(typeof(first_mu)).")
        end
    end
end

function _coerce_alpha6_3d(alpha6, n::Int)
    if alpha6 isa BianisotropicPolarizability3D
        return fill(alpha6.alpha, n)
    elseif alpha6 isa AbstractMatrix
        return fill(_as_cmat6_3d(alpha6, "alpha6"), n)
    else
        length(alpha6) == n ||
            error("alpha6 length ($(length(alpha6))) must match nvoxels ($n).")
        first_alpha = alpha6[1]
        if first_alpha isa BianisotropicPolarizability3D
            return [alpha6[j].alpha for j in 1:n]
        elseif first_alpha isa AbstractMatrix
            return [_as_cmat6_3d(alpha6[j], "alpha6[$j]") for j in 1:n]
        else
            error("Unsupported alpha6 specification: $(typeof(first_alpha)).")
        end
    end
end

@inline function _scale6_matrix_3d(eta0::Float64)
    eta0 > 0 || error("eta0 must be positive.")
    return SMatrix{6,6,ComplexF64,36}(ntuple(idx -> begin
        row = mod1(idx, 6)
        col = div(idx - 1, 6) + 1
        row == col ? (row <= 3 ? 1.0 + 0im : eta0 + 0im) : 0.0 + 0im
    end, 36))
end

@inline function _inv_scale6_matrix_3d(eta0::Float64)
    eta0 > 0 || error("eta0 must be positive.")
    return SMatrix{6,6,ComplexF64,36}(ntuple(idx -> begin
        row = mod1(idx, 6)
        col = div(idx - 1, 6) + 1
        row == col ? (row <= 3 ? 1.0 + 0im : (1 / eta0) + 0im) : 0.0 + 0im
    end, 36))
end

@inline _alpha3_matrix(alpha::Number) = ComplexF64(alpha) * _CI3_DDA
@inline _alpha3_matrix(alpha::_CMat3DDA) = alpha

function _blockdiag_alpha6(alpha_e, alpha_m)
    ae = _alpha3_matrix(alpha_e)
    am = _alpha3_matrix(alpha_m)
    return _CMat6DDA((ae[1, 1], ae[2, 1], ae[3, 1], 0, 0, 0,
                      ae[1, 2], ae[2, 2], ae[3, 2], 0, 0, 0,
                      ae[1, 3], ae[2, 3], ae[3, 3], 0, 0, 0,
                      0, 0, 0, am[1, 1], am[2, 1], am[3, 1],
                      0, 0, 0, am[1, 2], am[2, 2], am[3, 2],
                      0, 0, 0, am[1, 3], am[2, 3], am[3, 3]))
end

"""
    magnetic_clausius_mossotti_polarizability(mu_r, volume; k0=0, radiative_correction=false)

Return magnetic polarizability `alpha_m` for an isotropic or tensor
relative permeability voxel. It uses the same Clausius-Mossotti form as the
electric path, with `mu_r` replacing `eps_r`, so the induced magnetic dipole is
`m = alpha_m * H`.
"""
function magnetic_clausius_mossotti_polarizability(mu_r::Number, volume::Real;
                                                   k0::Real=0.0,
                                                   radiative_correction::Bool=false)
    return clausius_mossotti_polarizability(mu_r, volume;
                                            k0=k0,
                                            radiative_correction=radiative_correction)
end

function magnetic_clausius_mossotti_polarizability(mu_r::AbstractMatrix, volume::Real;
                                                   k0::Real=0.0,
                                                   radiative_correction::Bool=false)
    return clausius_mossotti_polarizability(mu_r, volume;
                                            k0=k0,
                                            radiative_correction=radiative_correction)
end

"""
    bianisotropic_clausius_mossotti_polarizability(C6, volume; k0=0, radiative_correction=false)

Return a `6 x 6` coupled electric-magnetic polarizability from a normalized
bianisotropic relative material matrix `C6`. `C6` acts on `[E; eta0*H]`; the
returned polarizability acts on the solver fields `[E; H]` and returns
`[q; m]`.
"""
function bianisotropic_clausius_mossotti_polarizability(C6, volume::Real;
                                                        k0::Real=0.0,
                                                        radiative_correction::Bool=false,
                                                        eta0::Real=_ETA0_DDA)
    V = Float64(volume)
    V > 0 || error("volume must be positive.")
    k = Float64(k0)
    k >= 0 || error("k0 must be nonnegative.")
    eta = Float64(eta0)
    C = C6 isa BianisotropicMaterial3D ? C6.C6 : _as_cmat6_3d(C6, "C6")
    denom = C + 2 * SMatrix{6,6,ComplexF64,36}(I)
    abs(det(denom)) > 100 * eps(Float64) ||
        error("Bianisotropic Clausius-Mossotti polarizability is singular for C6 + 2I.")
    alpha_norm = 3 * V * ((C - SMatrix{6,6,ComplexF64,36}(I)) / denom)
    if radiative_correction
        I6 = SMatrix{6,6,ComplexF64,36}(I)
        alpha_norm = alpha_norm / (I6 + 1im * k^3 * alpha_norm / (6π))
    end
    return _inv_scale6_matrix_3d(eta) * alpha_norm * _scale6_matrix_3d(eta)
end

"""
    em_dda_polarizabilities(grid, k0, eps_r, mu_r; radiative_correction=false)

Compute block-diagonal coupled electric-magnetic polarizability matrices for
magnetodielectric voxels.
"""
function em_dda_polarizabilities(grid::VoxelGrid3D, k0::Real, eps_r, mu_r;
                                 radiative_correction::Bool=false)
    k = Float64(k0)
    k >= 0 || error("k0 must be nonnegative.")
    epsv = _coerce_epsr_material_3d(eps_r, grid.nvoxels)
    muv = _coerce_mur_material_3d(mu_r, grid.nvoxels)
    alpha = Vector{_CMat6DDA}(undef, grid.nvoxels)
    for j in 1:grid.nvoxels
        alpha_e = clausius_mossotti_polarizability(
            epsv[j], grid.volumes[j];
            k0=k,
            radiative_correction=radiative_correction,
        )
        alpha_m = magnetic_clausius_mossotti_polarizability(
            muv[j], grid.volumes[j];
            k0=k,
            radiative_correction=radiative_correction,
        )
        alpha[j] = _blockdiag_alpha6(alpha_e, alpha_m)
    end
    return alpha
end

function em_dda_polarizabilities(grid::VoxelGrid3D, k0::Real, alpha6;
                                 radiative_correction::Bool=false)
    Float64(k0) >= 0 || error("k0 must be nonnegative.")
    return _coerce_alpha6_3d(alpha6, grid.nvoxels)
end

function em_dda_polarizabilities(grid::VoxelGrid3D, k0::Real,
                                 material::BianisotropicMaterial3D;
                                 radiative_correction::Bool=false,
                                 eta0::Real=_ETA0_DDA)
    k = Float64(k0)
    k >= 0 || error("k0 must be nonnegative.")
    alpha = bianisotropic_clausius_mossotti_polarizability(
        material, grid.volumes[1];
        k0=k,
        radiative_correction=radiative_correction,
        eta0=eta0,
    )
    out = Vector{_CMat6DDA}(undef, grid.nvoxels)
    for j in 1:grid.nvoxels
        out[j] = grid.volumes[j] == grid.volumes[1] ? alpha :
            bianisotropic_clausius_mossotti_polarizability(
                material, grid.volumes[j];
                k0=k,
                radiative_correction=radiative_correction,
                eta0=eta0,
            )
    end
    return out
end

function em_dda_polarizabilities(grid::VoxelGrid3D, k0::Real,
                                 materials::AbstractVector{<:BianisotropicMaterial3D};
                                 radiative_correction::Bool=false,
                                 eta0::Real=_ETA0_DDA)
    length(materials) == grid.nvoxels ||
        error("bianisotropic material length ($(length(materials))) must match nvoxels ($(grid.nvoxels)).")
    k = Float64(k0)
    k >= 0 || error("k0 must be nonnegative.")
    return [bianisotropic_clausius_mossotti_polarizability(
                materials[j], grid.volumes[j];
                k0=k,
                radiative_correction=radiative_correction,
                eta0=eta0,
            ) for j in 1:grid.nvoxels]
end

@inline function _grad_g_cross_3d(r::Vec3, rp::Vec3, k::Float64, q::CVec3)
    R_vec = r - rp
    R = norm(R_vec)
    R > 0 || error("cross Green interaction is singular for coincident points.")
    Rhat = R_vec / R
    G = exp(-1im * k * R) / (4π * R)
    dGdR = (-1im * k - 1 / R) * G
    return dGdR * cross(Rhat, q)
end

@inline electric_dipole_magnetic_field_3d(r::Vec3, rp::Vec3, k::Float64, q::CVec3;
                                          eta0::Float64=_ETA0_DDA) =
    (1im * k / eta0) * _grad_g_cross_3d(r, rp, k, q)

@inline magnetic_dipole_electric_field_3d(r::Vec3, rp::Vec3, k::Float64, m::CVec3;
                                          eta0::Float64=_ETA0_DDA) =
    (-1im * eta0 * k) * _grad_g_cross_3d(r, rp, k, m)

@inline function _em_interaction_apply_3d(ri::Vec3, rj::Vec3, k::Float64,
                                          q::CVec3, m::CVec3)
    E = _electric_dipole_apply_3d(ri, rj, k, q) +
        magnetic_dipole_electric_field_3d(ri, rj, k, m)
    H = electric_dipole_magnetic_field_3d(ri, rj, k, q) +
        _electric_dipole_apply_3d(ri, rj, k, m)
    return E, H
end

"""
    em_dda_operator_3d(grid, k0, eps_r, mu_r; radiative_correction=false)

Construct a matrix-free coupled electric-magnetic DDA operator for
magnetodielectric material voxels.
"""
function em_dda_operator_3d(grid::VoxelGrid3D, k0::Real, eps_r, mu_r;
                            radiative_correction::Bool=false)
    k = Float64(k0)
    k > 0 || error("k0 must be positive.")
    alpha = em_dda_polarizabilities(
        grid, k, eps_r, mu_r;
        radiative_correction=radiative_correction,
    )
    return EMDDAOperator3D(grid, k, alpha, radiative_correction)
end

"""
    em_dda_operator_3d(grid, k0, alpha6)

Construct a matrix-free coupled electric-magnetic DDA operator from explicit
per-voxel `6 x 6` polarizabilities. Use `BianisotropicMaterial3D` when starting
from a normalized bianisotropic constitutive tensor instead.
"""
function em_dda_operator_3d(grid::VoxelGrid3D, k0::Real, alpha6;
                            radiative_correction::Bool=false)
    k = Float64(k0)
    k > 0 || error("k0 must be positive.")
    alpha = em_dda_polarizabilities(
        grid, k, alpha6;
        radiative_correction=radiative_correction,
    )
    return EMDDAOperator3D(grid, k, alpha, radiative_correction)
end

function em_dda_operator_3d(grid::VoxelGrid3D, k0::Real,
                            material::Union{BianisotropicMaterial3D,
                                            AbstractVector{<:BianisotropicMaterial3D}};
                            radiative_correction::Bool=false,
                            eta0::Real=_ETA0_DDA)
    k = Float64(k0)
    k > 0 || error("k0 must be positive.")
    alpha = em_dda_polarizabilities(
        grid, k, material;
        radiative_correction=radiative_correction,
        eta0=eta0,
    )
    return EMDDAOperator3D(grid, k, alpha, radiative_correction)
end

Base.size(A::EMDDAOperator3D) = (6 * A.grid.nvoxels, 6 * A.grid.nvoxels)
Base.size(A::EMDDAOperator3D, d::Int) = d <= 2 ? 6 * A.grid.nvoxels : 1
Base.eltype(::Type{<:EMDDAOperator3D}) = ComplexF64
Base.eltype(::EMDDAOperator3D) = ComplexF64

function Base.getindex(A::EMDDAOperator3D, row::Int, col::Int)
    1 <= row <= size(A, 1) || throw(BoundsError(A, (row, col)))
    1 <= col <= size(A, 2) || throw(BoundsError(A, (row, col)))
    i = _em_voxel(row)
    j = _em_voxel(col)
    a = _em_component(row)
    b = _em_component(col)
    if i == j
        return a == b ? 1.0 + 0im : 0.0 + 0im
    end

    alphaj = A.alpha[j]
    iszero(alphaj) && return 0.0 + 0im
    basis = _CVec6DDA(ntuple(c -> c == b ? 1.0 + 0im : 0.0 + 0im, 6))
    q6 = alphaj * basis
    q, m = _split_em_field(q6)
    E, H = _em_interaction_apply_3d(A.grid.centers[i], A.grid.centers[j],
                                    A.k0, q, m)
    return -(a <= 3 ? E[a] : H[a - 3])
end

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                            A::EMDDAOperator3D,
                            x::AbstractVector{ComplexF64},
                            alpha_scale::Number,
                            beta_scale::Number)
    length(x) == size(A, 2) || throw(DimensionMismatch("x length must be $(size(A, 2))."))
    length(y) == size(A, 1) || throw(DimensionMismatch("y length must be $(size(A, 1))."))

    xread = y === x ? copy(x) : x
    N = A.grid.nvoxels
    for i in 1:N
        Ei, Hi = _split_em_field(_read_em_field6(xread, i))
        ri = A.grid.centers[i]
        for j in 1:N
            i == j && continue
            alphaj = A.alpha[j]
            iszero(alphaj) && continue
            q, m = _split_em_field(alphaj * _read_em_field6(xread, j))
            Es, Hs = _em_interaction_apply_3d(ri, A.grid.centers[j], A.k0, q, m)
            Ei -= Es
            Hi -= Hs
        end

        for a in 1:3
            y[_em_index(i, a)] = alpha_scale * Ei[a] + beta_scale * y[_em_index(i, a)]
            y[_em_index(i, a + 3)] = alpha_scale * Hi[a] + beta_scale * y[_em_index(i, a + 3)]
        end
    end
    return y
end

LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                   A::EMDDAOperator3D,
                   x::AbstractVector{ComplexF64}) =
    LinearAlgebra.mul!(y, A, x, one(ComplexF64), zero(ComplexF64))

"""
    assemble_em_dda_3d(grid, k0, eps_r, mu_r; radiative_correction=false)

Assemble the dense coupled electric-magnetic DDA system. Prefer
`em_dda_operator_3d` for larger grids to avoid `O(N^2)` dense storage.
"""
function assemble_em_dda_3d(grid::VoxelGrid3D, k0::Real, eps_r, mu_r;
                            radiative_correction::Bool=false)
    Aop = em_dda_operator_3d(
        grid, k0, eps_r, mu_r;
        radiative_correction=radiative_correction,
    )
    return Matrix(Aop), Aop.alpha
end

function assemble_em_dda_3d(grid::VoxelGrid3D, k0::Real, alpha6;
                            radiative_correction::Bool=false)
    Aop = em_dda_operator_3d(
        grid, k0, alpha6;
        radiative_correction=radiative_correction,
    )
    return Matrix(Aop), Aop.alpha
end

"""
    planewave_em_dda_3d(grid, k_vec, E0, pol; eta0=376.730313668)

Evaluate transverse plane-wave electric and magnetic incident fields at voxel
centers. Returns `(E_inc, H_inc)` with `H = khat x E / eta0`.
"""
function planewave_em_dda_3d(grid::VoxelGrid3D, k_vec::Vec3, E0, pol;
                             eta0::Real=_ETA0_DDA)
    eta = Float64(eta0)
    eta > 0 || error("eta0 must be positive.")
    Einc = planewave_dda_3d(grid, k_vec, E0, pol)
    khat = k_vec / norm(k_vec)
    Hinc = Vector{CVec3}(undef, grid.nvoxels)
    for j in 1:grid.nvoxels
        Hinc[j] = cross(khat, Einc[j]) / eta
    end
    return Einc, Hinc
end

function _solve_em_dda_from_operator(grid::VoxelGrid3D, k0::Real, Aop,
                                     E_inc::AbstractVector, H_inc::AbstractVector;
                                     solver::Symbol=:direct,
                                     reported_solver::Symbol=solver,
                                     tol::Float64=1e-8,
                                     maxiter::Int=200,
                                     memory::Int=20,
                                     verbose::Bool=false)
    rhs = _flatten_em_fields_3d(E_inc, H_inc, grid.nvoxels, "E_inc", "H_inc")

    if solver == :direct
        A = Matrix(Aop)
        fac = lu(A)
        total_flat = fac \ rhs
        E_total, H_total = _unflatten_em_fields_3d(total_flat, grid.nvoxels)
        E_rhs, H_rhs = _unflatten_em_fields_3d(rhs, grid.nvoxels)
        return EMDDAResult3D(E_total, H_total, E_rhs, H_rhs,
                             Aop.alpha, A, fac, :direct, nothing,
                             grid, Float64(k0), Aop.radiative_correction)
    elseif solver == :gmres
        total_flat, stats = Krylov.gmres(Aop, rhs;
                                         memory=memory,
                                         rtol=tol,
                                         atol=0.0,
                                         itmax=maxiter,
                                         verbose=(verbose ? 1 : 0))
        E_total, H_total = _unflatten_em_fields_3d(total_flat, grid.nvoxels)
        E_rhs, H_rhs = _unflatten_em_fields_3d(rhs, grid.nvoxels)
        return EMDDAResult3D(E_total, H_total, E_rhs, H_rhs,
                             Aop.alpha, Aop, nothing, reported_solver, stats,
                             grid, Float64(k0), Aop.radiative_correction)
    else
        error("Unsupported EM DDA solver: $solver (expected :direct or :gmres).")
    end
end

"""
    solve_em_dda_3d(grid, k0, eps_r, mu_r, E_inc, H_inc; solver=:direct)

Solve coupled electric-magnetic volume DDA for magnetodielectric voxels.
"""
function solve_em_dda_3d(grid::VoxelGrid3D, k0::Real, eps_r, mu_r,
                         E_inc::AbstractVector, H_inc::AbstractVector;
                         radiative_correction::Bool=false,
                         solver::Symbol=:direct,
                         tol::Float64=1e-8,
                         maxiter::Int=200,
                         memory::Int=20,
                         verbose::Bool=false)
    solve_mode = solver == :fft_gmres ? :gmres : solver
    Aop = solver == :fft_gmres ?
        fft_em_dda_operator_3d(
            grid, k0, eps_r, mu_r;
            radiative_correction=radiative_correction,
        ) :
        em_dda_operator_3d(
            grid, k0, eps_r, mu_r;
            radiative_correction=radiative_correction,
        )
    return _solve_em_dda_from_operator(
        grid, k0, Aop, E_inc, H_inc;
        solver=solve_mode,
        reported_solver=solver,
        tol=tol,
        maxiter=maxiter,
        memory=memory,
        verbose=verbose,
    )
end

"""
    solve_em_dda_3d(grid, k0, alpha6, E_inc, H_inc; solver=:direct)

Solve coupled electric-magnetic volume DDA from explicit `6 x 6`
polarizabilities. Use this for bianisotropic voxel polarizabilities.
"""
function solve_em_dda_3d(grid::VoxelGrid3D, k0::Real, alpha6,
                         E_inc::AbstractVector, H_inc::AbstractVector;
                         radiative_correction::Bool=false,
                         solver::Symbol=:direct,
                         tol::Float64=1e-8,
                         maxiter::Int=200,
                         memory::Int=20,
                         verbose::Bool=false)
    solve_mode = solver == :fft_gmres ? :gmres : solver
    Aop = solver == :fft_gmres ?
        fft_em_dda_operator_3d(
            grid, k0, alpha6;
            radiative_correction=radiative_correction,
        ) :
        em_dda_operator_3d(
            grid, k0, alpha6;
            radiative_correction=radiative_correction,
        )
    return _solve_em_dda_from_operator(
        grid, k0, Aop, E_inc, H_inc;
        solver=solve_mode,
        reported_solver=solver,
        tol=tol,
        maxiter=maxiter,
        memory=memory,
        verbose=verbose,
    )
end

function solve_em_dda_3d(grid::VoxelGrid3D, k0::Real,
                         material::Union{BianisotropicMaterial3D,
                                         AbstractVector{<:BianisotropicMaterial3D}},
                         E_inc::AbstractVector, H_inc::AbstractVector;
                         radiative_correction::Bool=false,
                         solver::Symbol=:direct,
                         tol::Float64=1e-8,
                         maxiter::Int=200,
                         memory::Int=20,
                         verbose::Bool=false,
                         eta0::Real=_ETA0_DDA)
    solve_mode = solver == :fft_gmres ? :gmres : solver
    Aop = solver == :fft_gmres ?
        fft_em_dda_operator_3d(
            grid, k0, material;
            radiative_correction=radiative_correction,
            eta0=eta0,
        ) :
        em_dda_operator_3d(
            grid, k0, material;
            radiative_correction=radiative_correction,
            eta0=eta0,
        )
    return _solve_em_dda_from_operator(
        grid, k0, Aop, E_inc, H_inc;
        solver=solve_mode,
        reported_solver=solver,
        tol=tol,
        maxiter=maxiter,
        memory=memory,
        verbose=verbose,
    )
end

"""
    induced_dipoles_em_dda_3d(result)

Return `(q, m)`, the normalized induced electric dipoles and magnetic dipoles
from a coupled EM DDA result.
"""
function induced_dipoles_em_dda_3d(res::EMDDAResult3D)
    q = Vector{CVec3}(undef, res.grid.nvoxels)
    m = Vector{CVec3}(undef, res.grid.nvoxels)
    for j in 1:res.grid.nvoxels
        q[j], m[j] = _split_em_field(res.alpha[j] * _join_em_field(res.E_total[j], res.H_total[j]))
    end
    return q, m
end

"""
    scattered_fields_em_dda_3d(result, r_obs)

Compute scattered electric and magnetic fields at observation points by
summing induced electric and magnetic dipoles. Returns `(E_scat, H_scat)`.
"""
function scattered_fields_em_dda_3d(res::EMDDAResult3D, r_obs::AbstractVector{Vec3})
    q, m = induced_dipoles_em_dda_3d(res)
    Eout = Vector{CVec3}(undef, length(r_obs))
    Hout = Vector{CVec3}(undef, length(r_obs))
    for p in eachindex(r_obs)
        E = CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
        H = CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
        for j in 1:res.grid.nvoxels
            iszero(q[j]) && iszero(m[j]) && continue
            Es, Hs = _em_interaction_apply_3d(r_obs[p], res.grid.centers[j],
                                              res.k0, q[j], m[j])
            E += Es
            H += Hs
        end
        Eout[p] = E
        Hout[p] = H
    end
    return Eout, Hout
end

"""
    farfield_em_dda_3d(result, rhat)

Return `(F_E, F_H)` such that `E_scat ~= exp(-ikr) F_E / r` and
`H_scat ~= exp(-ikr) F_H / r` in observation direction `rhat`.
"""
function farfield_em_dda_3d(res::EMDDAResult3D, rhat::Vec3;
                            eta0::Real=_ETA0_DDA)
    eta = Float64(eta0)
    eta > 0 || error("eta0 must be positive.")
    rn = norm(rhat)
    rn > 0 || error("rhat must be nonzero.")
    n = rhat / rn
    proj = _I3_DDA - n * transpose(n)
    FE = CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    FH = CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    q, m = induced_dipoles_em_dda_3d(res)
    prefac = res.k0^2 / (4π)
    for j in 1:res.grid.nvoxels
        phase = exp(1im * res.k0 * dot(n, res.grid.centers[j]))
        FE += prefac * phase * (proj * q[j] - eta * cross(n, m[j]))
        FH += prefac * phase * ((1 / eta) * cross(n, q[j]) + proj * m[j])
    end
    return FE, FH
end

function farfield_em_dda_3d(res::EMDDAResult3D, rhat::AbstractVector{Vec3};
                            eta0::Real=_ETA0_DDA)
    FE = Vector{CVec3}(undef, length(rhat))
    FH = Vector{CVec3}(undef, length(rhat))
    for j in eachindex(rhat)
        FE[j], FH[j] = farfield_em_dda_3d(res, rhat[j]; eta0=eta0)
    end
    return FE, FH
end
