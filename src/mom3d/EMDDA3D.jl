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
        muc = ComplexF64(mu_r)
        isfinite(muc) ||
            throw(ArgumentError("mu_r must be finite, got $mu_r."))
        return fill(muc, n)
    elseif mu_r isa AbstractMatrix
        return fill(_as_cmat3(mu_r, "mu_r"), n)
    elseif _is_diag_tensor_tuple(mu_r)
        return fill(_diag_tensor_from_tuple(mu_r, "mu_r"), n)
    else
        length(mu_r) == n ||
            error("mu_r length ($(length(mu_r))) must match nvoxels ($n).")
        first_mu = mu_r[1]
        if first_mu isa Number
            return _copy_finite_complex_vector_3d(mu_r, n, "mu_r")
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
    isfinite(eta0) && eta0 > 0 ||
        throw(ArgumentError("eta0 must be finite and positive, got $eta0."))
    return SMatrix{6,6,ComplexF64,36}(ntuple(idx -> begin
        row = mod1(idx, 6)
        col = div(idx - 1, 6) + 1
        row == col ? (row <= 3 ? 1.0 + 0im : eta0 + 0im) : 0.0 + 0im
    end, 36))
end

@inline function _inv_scale6_matrix_3d(eta0::Float64)
    isfinite(eta0) && eta0 > 0 ||
        throw(ArgumentError("eta0 must be finite and positive, got $eta0."))
    return SMatrix{6,6,ComplexF64,36}(ntuple(idx -> begin
        row = mod1(idx, 6)
        col = div(idx - 1, 6) + 1
        row == col ? (row <= 3 ? 1.0 + 0im : (1 / eta0) + 0im) : 0.0 + 0im
    end, 36))
end

@inline function _transform_normalized_alpha6_3d(
    alpha::_CMat6DDA,
    eta0::Float64,
)
    inverse_eta0 = inv(eta0)
    factors = @SMatrix [
        1.0 1.0 1.0 eta0 eta0 eta0
        1.0 1.0 1.0 eta0 eta0 eta0
        1.0 1.0 1.0 eta0 eta0 eta0
        inverse_eta0 inverse_eta0 inverse_eta0 1.0 1.0 1.0
        inverse_eta0 inverse_eta0 inverse_eta0 1.0 1.0 1.0
        inverse_eta0 inverse_eta0 inverse_eta0 1.0 1.0 1.0
    ]
    return alpha .* factors
end

@inline function _transformed_alpha6_isfinite_3d(
    alpha::_CMat6DDA,
    eta0::Float64,
)
    inverse_eta0 = inv(eta0)
    @inbounds for col in 1:6, row in 1:6
        factor = if row <= 3
            col <= 3 ? 1.0 : eta0
        else
            col <= 3 ? inverse_eta0 : 1.0
        end
        isfinite(alpha[row, col] * factor) || return false
    end
    return true
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
@inline function bianisotropic_clausius_mossotti_polarizability(
    C6,
    volume::Real;
    k0::Real=0.0,
    radiative_correction::Bool=false,
    eta0::Real=_ETA0_DDA,
)::_CMat6DDA
    V = _finite_positive_real_3d(volume, "volume")
    k = _finite_nonnegative_k0_3d(k0)
    eta = _finite_positive_real_3d(eta0, "eta0")
    C = C6 isa BianisotropicMaterial3D ? C6.C6 : _as_cmat6_3d(C6, "C6")
    denom = C + 2 * SMatrix{6,6,ComplexF64,36}(I)
    inverse_denom = _validated_clausius_mossotti_inverse_3d(
        denom, "Bianisotropic Clausius-Mossotti denominator C6 + 2I")
    I6 = SMatrix{6,6,ComplexF64,36}(I)
    alpha_norm = V * (3 * ((C - I6) * inverse_denom))
    if all(isfinite, alpha_norm)
        if radiative_correction
            alpha_norm = alpha_norm / (I6 + 1im * k^3 * alpha_norm / (6π))
        end
        # Check scalar entries before materializing the 576-byte transformed
        # SMatrix, so heterogeneous voxel builds do not box a temporary matrix.
        all(isfinite, alpha_norm) &&
            _transformed_alpha6_isfinite_3d(alpha_norm, eta) &&
            return _transform_normalized_alpha6_3d(alpha_norm, eta)
    end

    converted = setprecision(BigFloat, _POLARIZABILITY_FALLBACK_PRECISION) do
        Cb = Matrix{Complex{BigFloat}}(C)
        identity_b = Matrix{Complex{BigFloat}}(I, 6, 6)
        alpha = 3 * BigFloat(V) * ((Cb - identity_b) / (Cb + 2 * identity_b))
        if radiative_correction
            denominator = identity_b +
                          Complex{BigFloat}(0, 1) * BigFloat(k)^3 * alpha /
                          (6 * BigFloat(pi))
            alpha /= denominator
        end
        scales = (BigFloat(1), BigFloat(1), BigFloat(1),
                  BigFloat(eta), BigFloat(eta), BigFloat(eta))
        for col in 1:6, row in 1:6
            alpha[row, col] *= scales[col] / scales[row]
        end
        out = ComplexF64.(alpha)
        all(isfinite, out) ||
            throw(OverflowError(
                "Bianisotropic Clausius-Mossotti polarizability is outside the representable ComplexF64 range."))
        out
    end
    return _CMat6DDA(Tuple(converted))
end

"""
    em_dda_polarizabilities(grid, k0, eps_r, mu_r; radiative_correction=false)

Compute block-diagonal coupled electric-magnetic polarizability matrices for
magnetodielectric voxels.
"""
function em_dda_polarizabilities(grid::VoxelGrid3D, k0::Real, eps_r, mu_r;
                                 radiative_correction::Bool=false)
    k = _finite_nonnegative_k0_3d(k0)
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
    _finite_nonnegative_k0_3d(k0)
    return _coerce_alpha6_3d(alpha6, grid.nvoxels)
end

function em_dda_polarizabilities(grid::VoxelGrid3D, k0::Real,
                                 material::BianisotropicMaterial3D;
                                 radiative_correction::Bool=false,
                                 eta0::Real=_ETA0_DDA)
    k = _finite_nonnegative_k0_3d(k0)
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
    k = _finite_nonnegative_k0_3d(k0)
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
    _electric_dipole_geometry_requires_exact_3d(ri, rj, k) &&
        return _em_interaction_apply_bigfloat_3d(ri, rj, k, q, m)
    E = _electric_dipole_apply_3d(ri, rj, k, q) +
        magnetic_dipole_electric_field_3d(ri, rj, k, m)
    H = electric_dipole_magnetic_field_3d(ri, rj, k, q) +
        _electric_dipole_apply_3d(ri, rj, k, m)
    return E, H
end

function _em_interaction_apply_bigfloat_3d(
        ri::Vec3,
        rj::Vec3,
        k::Float64,
        q::CVec3,
        m::CVec3)
    precision = _electric_dipole_exact_precision_3d(ri, rj, k)
    return setprecision(BigFloat, precision) do
        qb = SVector{3,Complex{BigFloat}}(ntuple(
            component -> Complex{BigFloat}(q[component]), 3))
        mb = SVector{3,Complex{BigFloat}}(ntuple(
            component -> Complex{BigFloat}(m[component]), 3))
        E, H = _em_interaction_value_bigfloat_3d(ri, rj, k, qb, mb)
        converted_E = CVec3(ntuple(component -> ComplexF64(E[component]), 3))
        converted_H = CVec3(ntuple(component -> ComplexF64(H[component]), 3))
        all(isfinite, converted_E) && all(isfinite, converted_H) ||
            throw(OverflowError(
                "EM DDA interaction field is outside the representable ComplexF64 range."))
        return converted_E, converted_H
    end
end

function _em_interaction_value_bigfloat_3d(
        ri::Vec3,
        rj::Vec3,
        k::Float64,
        q::SVector{3,Complex{BigFloat}},
        m::SVector{3,Complex{BigFloat}})
    R_vec = SVector{3,BigFloat}(
        BigFloat(ri[1]) - BigFloat(rj[1]),
        BigFloat(ri[2]) - BigFloat(rj[2]),
        BigFloat(ri[3]) - BigFloat(rj[3]),
    )
    R = sqrt(sum(abs2, R_vec))
    R > 0 || error("cross Green interaction is singular for coincident points.")
    Rhat = R_vec / R
    kb = BigFloat(k)
    scalar_green = exp(Complex{BigFloat}(0, -kb * R)) /
                   (4 * BigFloat(pi) * R)
    radial_derivative =
        (Complex{BigFloat}(0, -kb) - inv(R)) * scalar_green
    gradient_cross_q = radial_derivative * cross(Rhat, q)
    gradient_cross_m = radial_derivative * cross(Rhat, m)
    eta = BigFloat(_ETA0_DDA)
    electric_q = _electric_dipole_value_bigfloat_3d(ri, rj, k, q)
    electric_m = _electric_dipole_value_bigfloat_3d(ri, rj, k, m)
    E = electric_q + Complex{BigFloat}(0, -eta * kb) * gradient_cross_m
    H = Complex{BigFloat}(0, kb / eta) * gradient_cross_q + electric_m
    return E, H
end

function _em_alpha_interaction_apply_bigfloat_3d(
        ri::Vec3,
        rj::Vec3,
        k::Float64,
        alpha::_CMat6DDA,
        field::_CVec6DDA)
    precision = _electric_dipole_exact_precision_3d(ri, rj, k)
    return setprecision(BigFloat, precision) do
        dipoles = _alpha_apply_bigfloat_vector_3d(alpha, field)
        q = SVector{3,Complex{BigFloat}}(
            dipoles[1], dipoles[2], dipoles[3])
        m = SVector{3,Complex{BigFloat}}(
            dipoles[4], dipoles[5], dipoles[6])
        E, H = _em_interaction_value_bigfloat_3d(ri, rj, k, q, m)
        converted_E = CVec3(
            ComplexF64(E[1]), ComplexF64(E[2]), ComplexF64(E[3]))
        converted_H = CVec3(
            ComplexF64(H[1]), ComplexF64(H[2]), ComplexF64(H[3]))
        all(isfinite, converted_E) && all(isfinite, converted_H) ||
            throw(OverflowError(
                "EM DDA interaction field is outside the representable ComplexF64 range."))
        return converted_E, converted_H
    end
end

@inline function _em_alpha_interaction_apply_3d(
        ri::Vec3,
        rj::Vec3,
        k::Float64,
        alpha::_CMat6DDA,
        field::_CVec6DDA)
    dipoles = alpha * field
    if all(isfinite, dipoles) &&
       !_alpha_field_product_loses_range_3d(alpha, field)
        q, m = _split_em_field(dipoles)
        try
            E, H = _em_interaction_apply_3d(ri, rj, k, q, m)
            all(isfinite, E) && all(isfinite, H) && return E, H
        catch err
            err isa OverflowError || rethrow()
        end
    end
    return _em_alpha_interaction_apply_bigfloat_3d(
        ri, rj, k, alpha, field)
end

"""
    em_dda_operator_3d(grid, k0, eps_r, mu_r; radiative_correction=false)

Construct a matrix-free coupled electric-magnetic DDA operator for
magnetodielectric material voxels.
"""
function em_dda_operator_3d(grid::VoxelGrid3D, k0::Real, eps_r, mu_r;
                            radiative_correction::Bool=false)
    k = _finite_positive_k0_3d(k0)
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
    k = _finite_positive_k0_3d(k0)
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
    k = _finite_positive_k0_3d(k0)
    alpha = em_dda_polarizabilities(
        grid, k, material;
        radiative_correction=radiative_correction,
        eta0=eta0,
    )
    return EMDDAOperator3D(grid, k, alpha, radiative_correction)
end

Base.size(A::EMDDAOperator3D) = (6 * A.grid.nvoxels, 6 * A.grid.nvoxels)
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
    E, H = _em_alpha_interaction_apply_3d(
        A.grid.centers[i], A.grid.centers[j], A.k0, alphaj, basis)
    return -(a <= 3 ? E[a] : H[a - 3])
end

@noinline function _em_dda_operator_field_bigfloat_3d(
        A::EMDDAOperator3D,
        x::AbstractVector{ComplexF64},
        voxel::Int)
    return setprecision(
            BigFloat, _DDA_ACCUMULATION_FALLBACK_PRECISION) do
        total = SVector{6,Complex{BigFloat}}(ntuple(
            component -> Complex{BigFloat}(
                x[_em_index(voxel, component)]), 6))
        observation = A.grid.centers[voxel]
        @inbounds for source_voxel in 1:A.grid.nvoxels
            source_voxel == voxel && continue
            alpha = A.alpha[source_voxel]
            iszero(alpha) && continue
            dipoles = _alpha_apply_bigfloat_vector_3d(
                alpha, _read_em_field6(x, source_voxel))
            q = SVector{3,Complex{BigFloat}}(
                dipoles[1], dipoles[2], dipoles[3])
            m = SVector{3,Complex{BigFloat}}(
                dipoles[4], dipoles[5], dipoles[6])
            E, H = _em_interaction_value_bigfloat_3d(
                observation,
                A.grid.centers[source_voxel],
                A.k0,
                q,
                m,
            )
            total -= SVector{6,Complex{BigFloat}}(
                E[1], E[2], E[3], H[1], H[2], H[3])
        end
        converted = _CVec6DDA(ntuple(
            component -> ComplexF64(total[component]), 6))
        all(isfinite, converted) ||
            throw(OverflowError(
                "EM DDA operator output at voxel $voxel is outside the " *
                "representable ComplexF64 range."))
        return converted
    end
end

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                            A::EMDDAOperator3D,
                            x::AbstractVector{ComplexF64},
                            alpha_scale::Number,
                            beta_scale::Number)
    length(x) == size(A, 2) || throw(DimensionMismatch("x length must be $(size(A, 2))."))
    length(y) == size(A, 1) || throw(DimensionMismatch("y length must be $(size(A, 1))."))

    if iszero(alpha_scale)
        if iszero(beta_scale)
            fill!(y, zero(ComplexF64))
        elseif beta_scale != one(beta_scale)
            y .*= beta_scale
        end
        return y
    end

    xread = Base.mightalias(y, x) ? copy(x) : x
    overwrite = iszero(beta_scale)
    N = A.grid.nvoxels
    # Output voxels are independent, but threading is intentionally avoided here:
    # test/test_mom3d_em.jl asserts (@allocated mul!(y, A_op, x)) < 4096 and the
    # `Threads.@threads` task-spawn overhead allocates regardless of thread count,
    # which would break that budget. `@inbounds` over the verified-safe indexing
    # (1 <= i,j <= N, components 1..6, xread/y length 6N) keeps the loop at zero
    # allocations while removing per-access bounds checks.
    @inbounds for i in 1:N
        Ei, Hi = _split_em_field(_read_em_field6(xread, i))
        ri = A.grid.centers[i]
        needs_fallback = false
        try
            for j in 1:N
                i == j && continue
                alphaj = A.alpha[j]
                iszero(alphaj) && continue
                Es, Hs = _em_alpha_interaction_apply_3d(
                    ri, A.grid.centers[j], A.k0, alphaj,
                    _read_em_field6(xread, j))
                next_Ei = Ei - Es
                next_Hi = Hi - Hs
                if !all(isfinite, next_Ei) || !all(isfinite, next_Hi)
                    needs_fallback = true
                    break
                end
                Ei = next_Ei
                Hi = next_Hi
            end
        catch err
            err isa OverflowError || rethrow()
            needs_fallback = true
        end
        if needs_fallback
            Ei, Hi = _split_em_field(
                _em_dda_operator_field_bigfloat_3d(A, xread, i))
        end

        for a in 1:3
            idx_e = _em_index(i, a)
            idx_h = _em_index(i, a + 3)
            y[idx_e] = _dda_scaled_output_3d(
                Ei[a], y[idx_e], alpha_scale, beta_scale, overwrite, idx_e)
            y[idx_h] = _dda_scaled_output_3d(
                Hi[a], y[idx_h], alpha_scale, beta_scale, overwrite, idx_h)
        end
    end
    return y
end

LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                   A::EMDDAOperator3D,
                   x::AbstractVector{ComplexF64}) =
    LinearAlgebra.mul!(y, A, x, one(ComplexF64), zero(ComplexF64))

# Compute the 6x6 off-diagonal interaction block coupling source voxel j to
# observation voxel i (i != j). Column b is -[E; H] produced by the dipoles
# alpha_j * e_b, matching `getindex(A, row, col)` exactly but evaluating the
# pair interaction once per block instead of once per scalar entry.
@inline function _em_dense_block(A::EMDDAOperator3D, i::Int, j::Int)
    alphaj = A.alpha[j]
    ri = A.grid.centers[i]
    rj = A.grid.centers[j]
    k = A.k0
    cols = ntuple(b -> begin
        basis = _CVec6DDA(ntuple(c -> c == b ? 1.0 + 0im : 0.0 + 0im, 6))
        q, m = _split_em_field(alphaj * basis)
        E, H = _em_interaction_apply_3d(ri, rj, k, q, m)
        -_join_em_field(E, H)
    end, 6)
    return _CMat6DDA(ntuple(idx -> begin
        b = div(idx - 1, 6) + 1
        a = mod1(idx, 6)
        cols[b][a]
    end, 36))
end

# Fill the 6 columns owned by source voxel j (i.e. the block column of the dense
# operator). Each j writes a disjoint set of columns, so this is safe to run
# concurrently across j.
@inline function _em_fill_block_column!(M::Matrix{ComplexF64}, A::EMDDAOperator3D,
                                        j::Int, N::Int)
    @inbounds begin
        alphaj = A.alpha[j]
        iszero(alphaj) && return nothing
        for i in 1:N
            i == j && continue
            block = _em_dense_block(A, i, j)
            for b in 1:6
                col = _em_index(j, b)
                for a in 1:6
                    M[_em_index(i, a), col] = block[a, b]
                end
            end
        end
    end
    return nothing
end

# Dense materialization that fills each voxel-pair block once. The generic
# `AbstractMatrix` fallback would call `getindex` per scalar entry, recomputing
# every 6x6 voxel-pair interaction 36 times. The work for distinct source voxels
# j writes disjoint columns, so it is threaded when worker threads exist.
function _validated_dense_em_dda_system_size(
        grid::VoxelGrid3D,
        max_output_bytes::Integer)
    system_size = try
        Base.Checked.checked_mul(6, grid.nvoxels)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError("EM DDA system dimension overflows Int"))
    end
    matrix_bytes = _checked_array_payload_bytes(
        ComplexF64, system_size, system_size;
        label="dense EM DDA system matrix")
    _enforce_payload_limit(
        matrix_bytes, max_output_bytes,
        "dense EM DDA system matrix", "max_output_bytes")
    return system_size
end

function _dense_em_dda_matrix(
        A::EMDDAOperator3D,
        max_output_bytes::Integer)
    N = A.grid.nvoxels
    system_size = _validated_dense_em_dda_system_size(
        A.grid, max_output_bytes)
    M = Matrix{ComplexF64}(I, system_size, system_size)
    if Threads.nthreads() > 1
        Threads.@threads for j in 1:N
            _em_fill_block_column!(M, A, j, N)
        end
    else
        for j in 1:N
            _em_fill_block_column!(M, A, j, N)
        end
    end
    return M
end

Base.Matrix(A::EMDDAOperator3D) =
    _dense_em_dda_matrix(A, _DEFAULT_MAX_DENSE_PAYLOAD_BYTES)

"""
    assemble_em_dda_3d(grid, k0, eps_r, mu_r; radiative_correction=false)

Assemble the dense coupled electric-magnetic DDA system. Prefer
`em_dda_operator_3d` for larger grids to avoid `O(N^2)` dense storage.
"""
function assemble_em_dda_3d(
        grid::VoxelGrid3D, k0::Real, eps_r, mu_r;
        radiative_correction::Bool=false,
        max_output_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    _validated_dense_em_dda_system_size(grid, max_output_bytes)
    Aop = em_dda_operator_3d(
        grid, k0, eps_r, mu_r;
        radiative_correction=radiative_correction,
    )
    return _dense_em_dda_matrix(Aop, max_output_bytes), Aop.alpha
end

function assemble_em_dda_3d(
        grid::VoxelGrid3D, k0::Real, alpha6;
        radiative_correction::Bool=false,
        max_output_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    _validated_dense_em_dda_system_size(grid, max_output_bytes)
    Aop = em_dda_operator_3d(
        grid, k0, alpha6;
        radiative_correction=radiative_correction,
    )
    return _dense_em_dda_matrix(Aop, max_output_bytes), Aop.alpha
end

"""
    planewave_em_dda_3d(grid, k_vec, E0, pol; eta0=376.730313668)

Evaluate transverse plane-wave electric and magnetic incident fields at voxel
centers. Returns `(E_inc, H_inc)` with `H = khat x E / eta0`.
"""
function planewave_em_dda_3d(grid::VoxelGrid3D, k_vec::Vec3, E0, pol;
                             eta0::Real=_ETA0_DDA)
    eta = Float64(eta0)
    isfinite(eta) && eta > 0 ||
        throw(ArgumentError(
            "eta0 must be finite and positive, got $eta0."))
    Einc = planewave_dda_3d(grid, k_vec, E0, pol)
    khat = _normalized_real_direction_dda_3d(k_vec, "k_vec")
    Hinc = Vector{CVec3}(undef, grid.nvoxels)
    for j in 1:grid.nvoxels
        field = cross(khat, Einc[j]) / eta
        all(isfinite, field) ||
            throw(OverflowError(
                "EM DDA plane-wave magnetic field is non-finite at voxel $j."))
        Hinc[j] = field
    end
    return Einc, Hinc
end

function _solve_em_dda_from_operator(grid::VoxelGrid3D, k0::Real, Aop,
                                     E_inc::AbstractVector, H_inc::AbstractVector;
                                     solver::Symbol=:direct,
                                     reported_solver::Symbol=solver,
                                     max_matrix_bytes::Integer=
                                         _DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
                                     tol::Float64=1e-8,
                                     maxiter::Int=200,
                                     memory::Int=20,
                                     verbose::Bool=false,
                                     check_gmres_convergence::Bool=true)
    rhs = _flatten_em_fields_3d(E_inc, H_inc, grid.nvoxels, "E_inc", "H_inc")

    if solver == :direct
        A = _dense_em_dda_matrix(Aop, max_matrix_bytes)
        fac = _factor_dense_linear_system(
            A, ComplexF64, "direct EM DDA factorization")
        total_flat = _solve_factored_linear_system(
            fac, A, rhs, "direct EM DDA solution")
        E_total, H_total = _unflatten_em_fields_3d(total_flat, grid.nvoxels)
        E_rhs, H_rhs = _unflatten_em_fields_3d(rhs, grid.nvoxels)
        return EMDDAResult3D(E_total, H_total, E_rhs, H_rhs,
                             Aop.alpha, A, fac, :direct, nothing,
                             grid, Float64(k0), Aop.radiative_correction)
    elseif solver == :gmres
        total_flat, stats = solve_gmres(
            Aop, rhs;
            memory=memory,
            tol=tol,
            maxiter=maxiter,
            verbose=verbose,
            check_gmres_convergence=check_gmres_convergence,
        )
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
                         max_matrix_bytes::Integer=
                             _DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
                         tol::Float64=1e-8,
                         maxiter::Int=200,
                         memory::Int=20,
                         verbose::Bool=false,
                         check_gmres_convergence::Bool=true)
    solve_mode = solver == :fft_gmres ? :gmres : solver
    solve_mode == :direct && _validated_dense_em_dda_system_size(
        grid, max_matrix_bytes)
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
        max_matrix_bytes=max_matrix_bytes,
        tol=tol,
        maxiter=maxiter,
        memory=memory,
        verbose=verbose,
        check_gmres_convergence=check_gmres_convergence,
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
                         max_matrix_bytes::Integer=
                             _DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
                         tol::Float64=1e-8,
                         maxiter::Int=200,
                         memory::Int=20,
                         verbose::Bool=false,
                         check_gmres_convergence::Bool=true)
    solve_mode = solver == :fft_gmres ? :gmres : solver
    solve_mode == :direct && _validated_dense_em_dda_system_size(
        grid, max_matrix_bytes)
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
        max_matrix_bytes=max_matrix_bytes,
        tol=tol,
        maxiter=maxiter,
        memory=memory,
        verbose=verbose,
        check_gmres_convergence=check_gmres_convergence,
    )
end

function solve_em_dda_3d(grid::VoxelGrid3D, k0::Real,
                         material::Union{BianisotropicMaterial3D,
                                         AbstractVector{<:BianisotropicMaterial3D}},
                         E_inc::AbstractVector, H_inc::AbstractVector;
                         radiative_correction::Bool=false,
                         solver::Symbol=:direct,
                         max_matrix_bytes::Integer=
                             _DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
                         tol::Float64=1e-8,
                         maxiter::Int=200,
                         memory::Int=20,
                         verbose::Bool=false,
                         check_gmres_convergence::Bool=true,
                         eta0::Real=_ETA0_DDA)
    solve_mode = solver == :fft_gmres ? :gmres : solver
    solve_mode == :direct && _validated_dense_em_dda_system_size(
        grid, max_matrix_bytes)
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
        max_matrix_bytes=max_matrix_bytes,
        tol=tol,
        maxiter=maxiter,
        memory=memory,
        verbose=verbose,
        check_gmres_convergence=check_gmres_convergence,
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
        dipoles = _scaled_alpha_apply_3d(
            res.alpha[j], _join_em_field(res.E_total[j], res.H_total[j]),
            1.0, "EM DDA induced dipole", j)
        q[j], m[j] = _split_em_field(dipoles)
    end
    return q, m
end

"""
    scattered_fields_em_dda_3d(result, r_obs)

Compute scattered electric and magnetic fields at observation points by
summing induced electric and magnetic dipoles. Returns `(E_scat, H_scat)`.
"""
@noinline function _scattered_fields_sum_bigfloat_em_dda_3d(
        res::EMDDAResult3D,
        observation::Vec3)
    return setprecision(
            BigFloat, _DDA_ACCUMULATION_FALLBACK_PRECISION) do
        total_E = zero(SVector{3,Complex{BigFloat}})
        total_H = zero(SVector{3,Complex{BigFloat}})
        @inbounds for j in 1:res.grid.nvoxels
            iszero(res.alpha[j]) && continue
            dipoles = _alpha_apply_bigfloat_vector_3d(
                res.alpha[j],
                _join_em_field(res.E_total[j], res.H_total[j]))
            q = SVector{3,Complex{BigFloat}}(
                dipoles[1], dipoles[2], dipoles[3])
            m = SVector{3,Complex{BigFloat}}(
                dipoles[4], dipoles[5], dipoles[6])
            contribution_E, contribution_H =
                _em_interaction_value_bigfloat_3d(
                    observation, res.grid.centers[j], res.k0, q, m)
            total_E += contribution_E
            total_H += contribution_H
        end
        converted_E = CVec3(
            ComplexF64(total_E[1]),
            ComplexF64(total_E[2]),
            ComplexF64(total_E[3]),
        )
        converted_H = CVec3(
            ComplexF64(total_H[1]),
            ComplexF64(total_H[2]),
            ComplexF64(total_H[3]),
        )
        all(isfinite, converted_E) && all(isfinite, converted_H) ||
            throw(OverflowError(
                "EM DDA scattered field is outside the representable ComplexF64 range."))
        return converted_E, converted_H
    end
end

function _scattered_fields_sum_em_dda_3d(
        res::EMDDAResult3D,
        observation::Vec3)
    total_E = zero(CVec3)
    total_H = zero(CVec3)
    try
        @inbounds for j in 1:res.grid.nvoxels
            iszero(res.alpha[j]) && continue
            contribution_E, contribution_H =
                _em_alpha_interaction_apply_3d(
                    observation, res.grid.centers[j], res.k0,
                    res.alpha[j],
                    _join_em_field(res.E_total[j], res.H_total[j]))
            next_E = total_E + contribution_E
            next_H = total_H + contribution_H
            all(isfinite, next_E) && all(isfinite, next_H) ||
                return _scattered_fields_sum_bigfloat_em_dda_3d(
                    res, observation)
            total_E = next_E
            total_H = next_H
        end
    catch err
        err isa OverflowError || rethrow()
        return _scattered_fields_sum_bigfloat_em_dda_3d(
            res, observation)
    end
    return total_E, total_H
end

function scattered_fields_em_dda_3d(res::EMDDAResult3D, r_obs::AbstractVector{Vec3})
    Eout = Vector{CVec3}(undef, length(r_obs))
    Hout = Vector{CVec3}(undef, length(r_obs))
    @inbounds for p in eachindex(r_obs)
        Eout[p], Hout[p] =
            _scattered_fields_sum_em_dda_3d(res, r_obs[p])
    end
    return Eout, Hout
end

function _em_farfield_alpha_contribution_bigfloat_3d(
        alpha::_CMat6DDA,
        field::_CVec6DDA,
        k::Float64,
        direction::Vec3,
        center::Vec3,
        eta::Float64)
    return setprecision(BigFloat, _POLARIZABILITY_FALLBACK_PRECISION) do
        dipoles = _alpha_apply_bigfloat_vector_3d(alpha, field)
        q = SVector{3,Complex{BigFloat}}(
            dipoles[1], dipoles[2], dipoles[3])
        m = SVector{3,Complex{BigFloat}}(
            dipoles[4], dipoles[5], dipoles[6])
        n = SVector{3,BigFloat}(
            BigFloat(direction[1]),
            BigFloat(direction[2]),
            BigFloat(direction[3]),
        )
        projected_q = q - n * sum(n[index] * q[index] for index in 1:3)
        projected_m = m - n * sum(n[index] * m[index] for index in 1:3)
        eta_big = BigFloat(eta)
        E_term = projected_q - eta_big * cross(n, m)
        H_term = cross(n, q) / eta_big + projected_m
        phase_argument = BigFloat(k) * sum(
            n[index] * BigFloat(center[index]) for index in 1:3)
        phase = exp(Complex{BigFloat}(0, phase_argument))
        prefactor = BigFloat(k)^2 / (4 * BigFloat(pi))
        E_value = phase * prefactor * E_term
        H_value = phase * prefactor * H_term
        converted_E = CVec3(
            ComplexF64(E_value[1]),
            ComplexF64(E_value[2]),
            ComplexF64(E_value[3]),
        )
        converted_H = CVec3(
            ComplexF64(H_value[1]),
            ComplexF64(H_value[2]),
            ComplexF64(H_value[3]),
        )
        all(isfinite, converted_E) && all(isfinite, converted_H) ||
            throw(OverflowError(
                "EM DDA far-field contribution is outside the representable ComplexF64 range."))
        return converted_E, converted_H
    end
end

@inline function _em_farfield_alpha_contribution_3d(
        alpha::_CMat6DDA,
        field::_CVec6DDA,
        k::Float64,
        direction::Vec3,
        center::Vec3,
        eta::Float64,
        projection)
    dipoles = alpha * field
    if all(isfinite, dipoles) &&
       !_alpha_field_product_loses_range_3d(alpha, field)
        q, m = _split_em_field(dipoles)
        phase = _source_phase(
            k, direction, center, 1.0, "EM DDA far-field phase")
        E_contribution = phase * (
            projection * q - eta * cross(direction, m))
        H_contribution = phase * (
            (1 / eta) * cross(direction, q) + projection * m)
        prefactor, scale_fraction, scale_exponent =
            _farfield_prefactor_dda_3d(k)
        if iszero(scale_fraction)
            E_value = prefactor * E_contribution
            H_value = prefactor * H_contribution
        else
            E_value = _scale_farfield_vector_dda_3d(
                E_contribution, scale_fraction, scale_exponent)
            H_value = _scale_farfield_vector_dda_3d(
                H_contribution, scale_fraction, scale_exponent)
        end
        all(isfinite, E_value) && all(isfinite, H_value) &&
            return E_value, H_value
    end
    return _em_farfield_alpha_contribution_bigfloat_3d(
        alpha, field, k, direction, center, eta)
end

@noinline function _em_farfield_sum_bigfloat_3d(
        res::EMDDAResult3D,
        direction::Vec3,
        eta::Float64)
    return setprecision(
            BigFloat, _DDA_ACCUMULATION_FALLBACK_PRECISION) do
        n = SVector{3,BigFloat}(
            BigFloat(direction[1]),
            BigFloat(direction[2]),
            BigFloat(direction[3]),
        )
        eta_big = BigFloat(eta)
        prefactor = BigFloat(res.k0)^2 / (4 * BigFloat(pi))
        total_E = zero(SVector{3,Complex{BigFloat}})
        total_H = zero(SVector{3,Complex{BigFloat}})
        @inbounds for j in 1:res.grid.nvoxels
            iszero(res.alpha[j]) && continue
            dipoles = _alpha_apply_bigfloat_vector_3d(
                res.alpha[j],
                _join_em_field(res.E_total[j], res.H_total[j]))
            q = SVector{3,Complex{BigFloat}}(
                dipoles[1], dipoles[2], dipoles[3])
            m = SVector{3,Complex{BigFloat}}(
                dipoles[4], dipoles[5], dipoles[6])
            projected_q = q - n * sum(
                n[index] * q[index] for index in 1:3)
            projected_m = m - n * sum(
                n[index] * m[index] for index in 1:3)
            phase_argument = BigFloat(res.k0) * sum(
                n[index] * BigFloat(res.grid.centers[j][index])
                for index in 1:3)
            phase_prefactor =
                exp(Complex{BigFloat}(0, phase_argument)) * prefactor
            total_E += phase_prefactor *
                       (projected_q - eta_big * cross(n, m))
            total_H += phase_prefactor *
                       (cross(n, q) / eta_big + projected_m)
        end
        converted_E = CVec3(
            ComplexF64(total_E[1]),
            ComplexF64(total_E[2]),
            ComplexF64(total_E[3]),
        )
        converted_H = CVec3(
            ComplexF64(total_H[1]),
            ComplexF64(total_H[2]),
            ComplexF64(total_H[3]),
        )
        all(isfinite, converted_E) && all(isfinite, converted_H) ||
            throw(OverflowError(
                "EM DDA far-field amplitude is outside the representable ComplexF64 range."))
        return converted_E, converted_H
    end
end

function _em_farfield_sum_3d(
        res::EMDDAResult3D,
        direction::Vec3,
        eta::Float64,
        projection)
    @inbounds for center in res.grid.centers
        if _source_phase_requires_fallback(res.k0, direction, center)
            return _em_farfield_sum_bigfloat_3d(res, direction, eta)
        end
    end
    total_E = zero(CVec3)
    total_H = zero(CVec3)
    try
        @inbounds for j in 1:res.grid.nvoxels
            iszero(res.alpha[j]) && continue
            contribution_E, contribution_H =
                _em_farfield_alpha_contribution_3d(
                    res.alpha[j],
                    _join_em_field(res.E_total[j], res.H_total[j]),
                    res.k0, direction, res.grid.centers[j], eta,
                    projection)
            next_E = total_E + contribution_E
            next_H = total_H + contribution_H
            all(isfinite, next_E) && all(isfinite, next_H) ||
                return _em_farfield_sum_bigfloat_3d(
                    res, direction, eta)
            total_E = next_E
            total_H = next_H
        end
    catch err
        err isa OverflowError || rethrow()
        return _em_farfield_sum_bigfloat_3d(res, direction, eta)
    end
    return total_E, total_H
end

"""
    farfield_em_dda_3d(result, rhat)

Return `(F_E, F_H)` such that `E_scat ~= exp(-ikr) F_E / r` and
`H_scat ~= exp(-ikr) F_H / r` in observation direction `rhat`.
"""
function farfield_em_dda_3d(res::EMDDAResult3D, rhat::Vec3;
                            eta0::Real=_ETA0_DDA)
    eta = _finite_positive_real_3d(eta0, "eta0")
    n = _normalized_real_direction_dda_3d(rhat, "rhat")
    proj = _I3_DDA - n * transpose(n)
    return _em_farfield_sum_3d(res, n, eta, proj)
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
