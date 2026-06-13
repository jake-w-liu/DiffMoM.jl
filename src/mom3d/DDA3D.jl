# DDA3D.jl -- 3D vector material scattering by discrete dipoles
#
# Convention: exp(+i omega t), matching the rest of the package.
# The scalar Green phase is exp(-i k R). The electric field from a normalized
# electric dipole q = p / eps0 is
#
#   E(r) = G_EE(r, r') q
#
# where G_EE is the free-space electric dipole dyadic without the 1/eps0 factor.

export clausius_mossotti_polarizability, dda_polarizabilities
export dda_operator_3d
export electric_dipole_dyadic_3d, assemble_dda_3d, solve_dda_3d
export planewave_dda_3d, induced_dipoles_dda_3d
export scattered_field_dda_3d, farfield_dda_3d

const _I3_DDA = @SMatrix [1.0 0.0 0.0;
                          0.0 1.0 0.0;
                          0.0 0.0 1.0]

const _CI3_DDA = @SMatrix [1.0 + 0im 0.0 + 0im 0.0 + 0im;
                           0.0 + 0im 1.0 + 0im 0.0 + 0im;
                           0.0 + 0im 0.0 + 0im 1.0 + 0im]
const _CMat3DDA = SMatrix{3,3,ComplexF64,9}

@inline _dda_index(voxel::Int, comp::Int) = 3 * (voxel - 1) + comp

@inline _dda_voxel(linear_index::Int) = div(linear_index - 1, 3) + 1
@inline _dda_component(linear_index::Int) = mod(linear_index - 1, 3) + 1

@inline function _read_field_component(x::AbstractVector, j::Int)
    return CVec3(x[_dda_index(j, 1)], x[_dda_index(j, 2)], x[_dda_index(j, 3)])
end

function _as_cvec3(v, label::AbstractString)
    length(v) == 3 || error("$label must have exactly three components.")
    out = CVec3(ComplexF64(v[1]), ComplexF64(v[2]), ComplexF64(v[3]))
    for c in out
        isfinite(real(c)) && isfinite(imag(c)) ||
            error("$label contains a non-finite component: $c.")
    end
    return out
end

function _flatten_fields_3d(fields::AbstractVector, n::Int, label::AbstractString)
    length(fields) == n || error("$label length ($(length(fields))) must match nvoxels ($n).")
    out = Vector{ComplexF64}(undef, 3n)
    for j in 1:n
        v = _as_cvec3(fields[j], "$label[$j]")
        out[_dda_index(j, 1)] = v[1]
        out[_dda_index(j, 2)] = v[2]
        out[_dda_index(j, 3)] = v[3]
    end
    return out
end

function _unflatten_fields_3d(x::AbstractVector{ComplexF64}, n::Int)
    length(x) == 3n || error("field vector length ($(length(x))) must be 3*nvoxels ($(3n)).")
    out = Vector{CVec3}(undef, n)
    for j in 1:n
        out[j] = CVec3(x[_dda_index(j, 1)], x[_dda_index(j, 2)], x[_dda_index(j, 3)])
    end
    return out
end

function _coerce_epsr_3d(eps_r, n::Int)
    if eps_r isa Number
        epsv = fill(ComplexF64(eps_r), n)
    else
        length(eps_r) == n ||
            error("eps_r length ($(length(eps_r))) must match nvoxels ($n).")
        epsv = ComplexF64.(collect(eps_r))
    end
    for j in 1:n
        epsj = epsv[j]
        isfinite(real(epsj)) && isfinite(imag(epsj)) ||
            error("eps_r[$j] is not finite: $epsj.")
    end
    return epsv
end

function _as_cmat3(m, label::AbstractString)
    size(m) == (3, 3) || error("$label must be a 3x3 tensor.")
    vals = ntuple(i -> ComplexF64(m[i]), 9)
    M = _CMat3DDA(vals)
    for c in M
        isfinite(real(c)) && isfinite(imag(c)) ||
            error("$label contains a non-finite component: $c.")
    end
    return M
end

function _diag_tensor_from_tuple(t, label::AbstractString)
    length(t) == 3 || error("$label diagonal tensor tuple must have three entries.")
    vals = ComplexF64.(t)
    return _CMat3DDA((vals[1], 0, 0,
                      0, vals[2], 0,
                      0, 0, vals[3]))
end

_is_diag_tensor_tuple(x) = x isa Tuple && length(x) == 3 && all(v -> v isa Number, x)

function _coerce_epsr_material_3d(eps_r, n::Int)
    if eps_r isa Number
        return fill(ComplexF64(eps_r), n)
    elseif eps_r isa AbstractMatrix
        return fill(_as_cmat3(eps_r, "eps_r"), n)
    elseif _is_diag_tensor_tuple(eps_r)
        return fill(_diag_tensor_from_tuple(eps_r, "eps_r"), n)
    else
        length(eps_r) == n ||
            error("eps_r length ($(length(eps_r))) must match nvoxels ($n).")
        first_eps = eps_r[1]
        if first_eps isa Number
            return _coerce_epsr_3d(eps_r, n)
        elseif first_eps isa AbstractMatrix
            return [_as_cmat3(eps_r[j], "eps_r[$j]") for j in 1:n]
        elseif _is_diag_tensor_tuple(first_eps)
            return [_diag_tensor_from_tuple(eps_r[j], "eps_r[$j]") for j in 1:n]
        else
            error("Unsupported eps_r material specification: $(typeof(first_eps)).")
        end
    end
end

@inline _alpha_apply(alpha::Number, E::CVec3) = alpha * E
@inline _alpha_apply(alpha::_CMat3DDA, E::CVec3) = alpha * E
@inline _alpha_adjoint_apply(alpha::Number, E::CVec3) = conj(alpha) * E
@inline _alpha_adjoint_apply(alpha::_CMat3DDA, E::CVec3) = adjoint(alpha) * E
@inline _alpha_block(G, alpha::Number) = G * alpha
@inline _alpha_block(G, alpha::_CMat3DDA) = G * alpha

"""
    clausius_mossotti_polarizability(eps_r, volume; k0=0, radiative_correction=false)

Return normalized electric polarizability `alpha = p / (eps0 * E)` for an
isotropic voxel of relative permittivity `eps_r` and volume `volume`.

The default is the Clausius-Mossotti polarizability

    alpha0 = 3V * (eps_r - 1) / (eps_r + 2)

which gives the exact electrostatic polarizability of a sphere with the same
volume. If `radiative_correction=true`, a leading-order radiation-reaction
correction consistent with the package's `exp(+i omega t)` convention is used.
"""
function clausius_mossotti_polarizability(eps_r::Number, volume::Real;
                                          k0::Real=0.0,
                                          radiative_correction::Bool=false)
    V = Float64(volume)
    V > 0 || error("volume must be positive.")
    k = Float64(k0)
    k >= 0 || error("k0 must be nonnegative.")
    epsc = ComplexF64(eps_r)
    abs(epsc + 2) > 100 * eps(Float64) ||
        error("Clausius-Mossotti polarizability is singular for eps_r near -2.")
    alpha0 = 3 * V * (epsc - 1) / (epsc + 2)
    if radiative_correction
        return alpha0 / (1 + 1im * k^3 * alpha0 / (6π))
    end
    return alpha0
end

function clausius_mossotti_polarizability(eps_r::AbstractMatrix, volume::Real;
                                          k0::Real=0.0,
                                          radiative_correction::Bool=false)
    V = Float64(volume)
    V > 0 || error("volume must be positive.")
    k = Float64(k0)
    k >= 0 || error("k0 must be nonnegative.")
    epsm = _as_cmat3(eps_r, "eps_r")
    denom = epsm + 2 * _CI3_DDA
    abs(det(denom)) > 100 * eps(Float64) ||
        error("Tensor Clausius-Mossotti polarizability is singular for eps_r + 2I.")
    alpha0 = 3 * V * ((epsm - _CI3_DDA) / denom)
    if radiative_correction
        return alpha0 / (_CI3_DDA + 1im * k^3 * alpha0 / (6π))
    end
    return alpha0
end

"""
    dda_polarizabilities(grid, k0, eps_r; radiative_correction=false)

Compute normalized electric polarizability for every voxel in `grid`.
"""
function dda_polarizabilities(grid::VoxelGrid3D, k0::Real, eps_r;
                              radiative_correction::Bool=false)
    epsv = _coerce_epsr_material_3d(eps_r, grid.nvoxels)
    alpha = Vector{typeof(clausius_mossotti_polarizability(epsv[1], grid.volumes[1];
                                                           k0=k0,
                                                           radiative_correction=radiative_correction))}(undef, grid.nvoxels)
    for j in 1:grid.nvoxels
        alpha[j] = clausius_mossotti_polarizability(
            epsv[j], grid.volumes[j];
            k0=k0,
            radiative_correction=radiative_correction,
        )
    end
    return alpha
end

"""
    dda_operator_3d(grid, k0, eps_r; radiative_correction=false)

Construct the matrix-free DDA material operator. This is the memory-efficient
counterpart to `assemble_dda_3d`.
"""
function dda_operator_3d(grid::VoxelGrid3D, k0::Real, eps_r;
                         radiative_correction::Bool=false)
    k = Float64(k0)
    k > 0 || error("k0 must be positive.")
    epsv = _coerce_epsr_material_3d(eps_r, grid.nvoxels)
    alpha = dda_polarizabilities(grid, k, epsv; radiative_correction=radiative_correction)
    return DDAOperator3D(grid, k, epsv, alpha, radiative_correction)
end

"""
    electric_dipole_dyadic_3d(r, rp, k0)

Free-space electric dipole dyadic for the package convention `exp(+i omega t)`.
It maps normalized dipole moment `q = p / eps0` at `rp` to electric field at
`r`. The singular self term is not defined and must be handled by the chosen
polarizability model.
"""
function electric_dipole_dyadic_3d(r::Vec3, rp::Vec3, k0::Real)
    k = Float64(k0)
    k >= 0 || error("k0 must be nonnegative.")
    R_vec = r - rp
    R = norm(R_vec)
    R > 0 || error("electric_dipole_dyadic_3d is singular for coincident points.")
    Rhat = R_vec / R
    rr = Rhat * transpose(Rhat)
    expfac = exp(-1im * k * R) / (4π)

    transverse = (k^2 / R) * (_I3_DDA - rr)
    near = (1 / R^3 + 1im * k / R^2) * (3 * rr - _I3_DDA)
    return expfac * (transverse + near)
end

@inline function _electric_dipole_apply_3d(r::Vec3, rp::Vec3, k::Float64, q::CVec3)
    R_vec = r - rp
    R = norm(R_vec)
    R > 0 || error("electric dipole application is singular for coincident points.")
    Rhat = R_vec / R
    rq = dot(Rhat, q)
    expfac = exp(-1im * k * R) / (4π)

    transverse = (k^2 / R) * (q - rq * Rhat)
    near = (1 / R^3 + 1im * k / R^2) * (3 * rq * Rhat - q)
    return expfac * (transverse + near)
end

Base.size(A::DDAOperator3D) = (3 * A.grid.nvoxels, 3 * A.grid.nvoxels)
Base.size(A::DDAOperator3D, d::Int) = d <= 2 ? 3 * A.grid.nvoxels : 1
Base.eltype(::Type{<:DDAOperator3D}) = ComplexF64
Base.eltype(::DDAOperator3D) = ComplexF64

function Base.getindex(A::DDAOperator3D, row::Int, col::Int)
    1 <= row <= size(A, 1) || throw(BoundsError(A, (row, col)))
    1 <= col <= size(A, 2) || throw(BoundsError(A, (row, col)))
    i = _dda_voxel(row)
    j = _dda_voxel(col)
    a = _dda_component(row)
    b = _dda_component(col)
    if i == j
        return a == b ? 1.0 + 0im : 0.0 + 0im
    end
    iszero(A.alpha[j]) && return 0.0 + 0im
    G = electric_dipole_dyadic_3d(A.grid.centers[i], A.grid.centers[j], A.k0)
    return -_alpha_block(G, A.alpha[j])[a, b]
end

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                            A::DDAOperator3D,
                            x::AbstractVector{ComplexF64},
                            alpha_scale::Number,
                            beta_scale::Number)
    length(x) == size(A, 2) || throw(DimensionMismatch("x length must be $(size(A, 2))."))
    length(y) == size(A, 1) || throw(DimensionMismatch("y length must be $(size(A, 1))."))

    xread = y === x ? copy(x) : x
    N = A.grid.nvoxels
    # Output voxels are independent, but the matrix-free alloc budget
    # (test/test_mom3d.jl asserts < 1024 bytes) is tighter than the task-spawn
    # overhead of `Threads.@threads` (>= ~1.5 KB even for an empty loop), so
    # threading here would break it. `@inbounds` over the verified-safe indexing
    # (1 <= i,j <= N, components 1..3, xread/y length 3N) keeps the loop at zero
    # allocations while removing per-access bounds checks.
    @inbounds for i in 1:N
        Ei = _read_field_component(xread, i)
        ri = A.grid.centers[i]
        for j in 1:N
            i == j && continue
            alphaj = A.alpha[j]
            iszero(alphaj) && continue
            qj = _alpha_apply(alphaj, _read_field_component(xread, j))
            Ei -= _electric_dipole_apply_3d(ri, A.grid.centers[j], A.k0, qj)
        end

        for a in 1:3
            idx = _dda_index(i, a)
            y[idx] = alpha_scale * Ei[a] + beta_scale * y[idx]
        end
    end
    return y
end

LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                   A::DDAOperator3D,
                   x::AbstractVector{ComplexF64}) =
    LinearAlgebra.mul!(y, A, x, one(ComplexF64), zero(ComplexF64))

Base.adjoint(A::DDAOperator3D) = DDAAdjointOperator3D(A)
Base.size(A::DDAAdjointOperator3D) = reverse(size(A.parent))
Base.size(A::DDAAdjointOperator3D, d::Int) = size(A.parent, d <= 2 ? 3 - d : d)
Base.eltype(::Type{DDAAdjointOperator3D}) = ComplexF64
Base.eltype(::DDAAdjointOperator3D) = ComplexF64
Base.getindex(A::DDAAdjointOperator3D, row::Int, col::Int) = conj(A.parent[col, row])

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                            Aadj::DDAAdjointOperator3D,
                            x::AbstractVector{ComplexF64},
                            alpha_scale::Number,
                            beta_scale::Number)
    A = Aadj.parent
    length(x) == size(Aadj, 2) || throw(DimensionMismatch("x length must be $(size(Aadj, 2))."))
    length(y) == size(Aadj, 1) || throw(DimensionMismatch("y length must be $(size(Aadj, 1))."))

    xread = y === x ? copy(x) : x
    N = A.grid.nvoxels
    # Left serial for the same alloc-budget reason as the forward operator;
    # `@inbounds` is applied over the verified-safe indexing.
    @inbounds for i in 1:N
        Ei = _read_field_component(xread, i)
        alphai = conj(A.alpha[i])
        if !iszero(alphai)
            ri = A.grid.centers[i]
            acc = CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
            for j in 1:N
                i == j && continue
                xj = _read_field_component(xread, j)
                acc += conj(_electric_dipole_apply_3d(ri, A.grid.centers[j], A.k0, conj(xj)))
            end
            Ei -= _alpha_adjoint_apply(A.alpha[i], acc)
        end

        for a in 1:3
            idx = _dda_index(i, a)
            y[idx] = alpha_scale * Ei[a] + beta_scale * y[idx]
        end
    end
    return y
end

LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                   A::DDAAdjointOperator3D,
                   x::AbstractVector{ComplexF64}) =
    LinearAlgebra.mul!(y, A, x, one(ComplexF64), zero(ComplexF64))

"""
    assemble_dda_3d(grid, k0, eps_r; radiative_correction=false)

Assemble the dense coupled-dipole system

    E_i - sum_{j != i} G_EE(r_i, r_j) alpha_j E_j = E_inc_i

for isotropic complex relative permittivity `eps_r`.

Returns `(A, alpha, epsv)`, where `A` is a `3N x 3N` dense matrix.
"""
function assemble_dda_3d(grid::VoxelGrid3D, k0::Real, eps_r;
                         radiative_correction::Bool=false)
    k = Float64(k0)
    k > 0 || error("k0 must be positive.")
    N = grid.nvoxels
    epsv = _coerce_epsr_material_3d(eps_r, N)
    alpha = dda_polarizabilities(grid, k, epsv; radiative_correction=radiative_correction)

    A = Matrix{ComplexF64}(I, 3N, 3N)
    # Each source voxel j writes a disjoint block of columns, so the assembly is
    # threaded over j when worker threads exist. There is no alloc budget on the
    # dense build, unlike the matrix-free matvec.
    if Threads.nthreads() > 1
        Threads.@threads for j in 1:N
            _dda_fill_block_column!(A, grid, alpha, k, j, N)
        end
    else
        for j in 1:N
            _dda_fill_block_column!(A, grid, alpha, k, j, N)
        end
    end

    return A, alpha, epsv
end

# Fill the 3 columns owned by source voxel j. Each j writes disjoint columns,
# so this is safe to run concurrently across j.
@inline function _dda_fill_block_column!(A::Matrix{ComplexF64}, grid::VoxelGrid3D,
                                         alpha, k::Float64, j::Int, N::Int)
    @inbounds begin
        alphaj = alpha[j]
        iszero(alphaj) && return nothing
        rj = grid.centers[j]
        for i in 1:N
            i == j && continue
            G = electric_dipole_dyadic_3d(grid.centers[i], rj, k)
            block = _alpha_block(G, alphaj)
            for c in 1:3
                col = _dda_index(j, c)
                for a in 1:3
                    A[_dda_index(i, a), col] -= block[a, c]
                end
            end
        end
    end
    return nothing
end

"""
    planewave_dda_3d(grid, k_vec, E0, pol)

Evaluate a transverse plane wave at voxel centers:

    E_inc(r) = pol * E0 * exp(-i k_vec dot r)
"""
function planewave_dda_3d(grid::VoxelGrid3D, k_vec::Vec3, E0, pol)
    kn = norm(k_vec)
    kn > 0 || error("k_vec must be nonzero.")
    polv = _as_cvec3(pol, "pol")
    pn = norm(polv)
    pn > 0 || error("pol must be nonzero.")
    transverse_error = abs(dot(k_vec / kn, polv / pn))
    transverse_error <= 1e-10 ||
        error("Plane-wave polarization must be transverse to k_vec; normalized dot=$transverse_error.")

    amp = ComplexF64(E0)
    Einc = Vector{CVec3}(undef, grid.nvoxels)
    for j in 1:grid.nvoxels
        Einc[j] = polv * amp * exp(-1im * dot(k_vec, grid.centers[j]))
    end
    return Einc
end

"""
    solve_dda_3d(grid, k0, eps_r, E_inc; radiative_correction=false)

Solve the 3D vector material scattering problem for total electric fields at
voxel centers.
"""
function solve_dda_3d(grid::VoxelGrid3D, k0::Real, eps_r, E_inc::AbstractVector;
                      radiative_correction::Bool=false,
                      solver::Symbol=:direct,
                      tol::Float64=1e-8,
                      maxiter::Int=200,
                      memory::Int=20,
                      verbose::Bool=false)
    rhs = _flatten_fields_3d(E_inc, grid.nvoxels, "E_inc")

    if solver == :direct
        A, alpha, epsv = assemble_dda_3d(
            grid, k0, eps_r;
            radiative_correction=radiative_correction,
        )
        fac = lu(A)
        E_total_flat = fac \ rhs
        E_total = _unflatten_fields_3d(E_total_flat, grid.nvoxels)
        return DDAResult3D(E_total, _unflatten_fields_3d(rhs, grid.nvoxels),
                           epsv, alpha, A, fac, :direct, nothing,
                           grid, Float64(k0), radiative_correction)
    elseif solver == :gmres
        A = dda_operator_3d(
            grid, k0, eps_r;
            radiative_correction=radiative_correction,
        )
        E_total_flat, stats = Krylov.gmres(A, rhs;
                                           memory=memory,
                                           rtol=tol,
                                           atol=0.0,
                                           itmax=maxiter,
                                           verbose=(verbose ? 1 : 0))
        E_total = _unflatten_fields_3d(E_total_flat, grid.nvoxels)
        return DDAResult3D(E_total, _unflatten_fields_3d(rhs, grid.nvoxels),
                           A.eps_r, A.alpha, A, nothing, :gmres, stats,
                           grid, Float64(k0), radiative_correction)
    else
        error("Unsupported DDA solver: $solver (expected :direct or :gmres).")
    end
end

"""
    induced_dipoles_dda_3d(result)

Return normalized induced electric dipoles `q_j = p_j / eps0 = alpha_j E_j`.
"""
function induced_dipoles_dda_3d(res::DDAResult3D)
    q = Vector{CVec3}(undef, res.grid.nvoxels)
    for j in 1:res.grid.nvoxels
        q[j] = _alpha_apply(res.alpha[j], res.E_total[j])
    end
    return q
end

"""
    scattered_field_dda_3d(result, r_obs)

Compute scattered electric field at observation points by summing the radiated
field of all induced dipoles. Observation points must not coincide with voxel
centers.
"""
function scattered_field_dda_3d(res::DDAResult3D, r_obs::AbstractVector{Vec3})
    out = Vector{CVec3}(undef, length(r_obs))
    for m in eachindex(r_obs)
        E = CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
        for j in 1:res.grid.nvoxels
            iszero(res.alpha[j]) && continue
            G = electric_dipole_dyadic_3d(r_obs[m], res.grid.centers[j], res.k0)
            E += G * _alpha_apply(res.alpha[j], res.E_total[j])
        end
        out[m] = E
    end
    return out
end

"""
    farfield_dda_3d(result, rhat)

Return the far-field amplitude `F(rhat)` such that

    E_scat(r) ~= exp(-i k r) / r * F(rhat)

for unit observation direction `rhat`.
"""
function farfield_dda_3d(res::DDAResult3D, rhat::Vec3)
    rn = norm(rhat)
    rn > 0 || error("rhat must be nonzero.")
    n = rhat / rn
    proj = _I3_DDA - n * transpose(n)
    F = CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    prefac = res.k0^2 / (4π)
    for j in 1:res.grid.nvoxels
        iszero(res.alpha[j]) && continue
        phase = exp(1im * res.k0 * dot(n, res.grid.centers[j]))
        F += prefac * phase * (proj * _alpha_apply(res.alpha[j], res.E_total[j]))
    end
    return F
end

function farfield_dda_3d(res::DDAResult3D, rhat::AbstractVector{Vec3})
    return [farfield_dda_3d(res, dir) for dir in rhat]
end
