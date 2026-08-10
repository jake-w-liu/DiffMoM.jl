# FFTDDA3D.jl -- FFT-accelerated DDA matvecs on uniform Cartesian voxel grids

export FFTDDAKernel3D, FFTDDAOperator3D
export fft_dda_kernel_3d, fft_dda_operator_3d
export FFTEMDDAKernel3D, FFTEMDDAOperator3D
export fft_em_dda_kernel_3d, fft_em_dda_operator_3d

"""
    FFTDDAKernel3D

Fourier-space block Toeplitz embedding of the free-space DDA dyadic for a
uniform `VoxelGrid3D`. The stored kernel excludes the singular self offset and
is multiplied by `interaction_scale`; the operator divides induced dipoles by
the same scale before convolution.
"""
struct FFTDDAKernel3D
    pad_dims::NTuple{3,Int}
    interaction_scale::Float64
    kernel_hat::Array{ComplexF64,5}
end

function _fft_array_component_scale_3d(values)
    scale = 0.0
    @inbounds for value in values
        component_scale = _complex_component_scale_3d(value)
        isfinite(component_scale) || return Inf
        scale = max(scale, component_scale)
    end
    return scale
end

"""
    FFTDDAOperator3D

FFT-accelerated coupled-dipole operator for a uniform `VoxelGrid3D`. It applies

    y = x - G * (alpha .* x)

with the self interaction excluded, matching `DDAOperator3D`.
"""
struct FFTDDAOperator3D{TEps<:AbstractVector,TAlpha<:AbstractVector} <: AbstractMatrix{ComplexF64}
    grid::VoxelGrid3D
    k0::Float64
    eps_r::TEps
    alpha::TAlpha
    radiative_correction::Bool
    kernel::FFTDDAKernel3D
    qhat::Array{ComplexF64,4}
    conv::Array{ComplexF64,3}
    work_lock::ReentrantLock
end

FFTDDAOperator3D(grid::VoxelGrid3D, k0::Float64,
                 eps_r::TEps, alpha::TAlpha, radiative_correction::Bool,
                 kernel::FFTDDAKernel3D, qhat::Array{ComplexF64,4},
                 conv::Array{ComplexF64,3}) where
                {TEps<:AbstractVector,TAlpha<:AbstractVector} =
    FFTDDAOperator3D{TEps,TAlpha}(grid, k0, eps_r, alpha,
                                 radiative_correction, kernel, qhat, conv,
                                 ReentrantLock())

"""
    FFTEMDDAKernel3D

Fourier-space block Toeplitz embedding of the coupled electric-magnetic DDA
interaction. The stored kernel maps induced `[q; m]` dipoles to scattered
`[E; H]` fields, excludes the singular self offset, and is multiplied by
`interaction_scale`; the operator applies the inverse scale to the dipoles.
"""
struct FFTEMDDAKernel3D
    pad_dims::NTuple{3,Int}
    interaction_scale::Float64
    kernel_hat::Array{ComplexF64,5}
end

"""
    FFTEMDDAOperator3D

FFT-accelerated coupled electric-magnetic DDA operator. It applies

    y = x - G_em * (alpha6 * x)

with the self interaction excluded, matching `EMDDAOperator3D`.
"""
struct FFTEMDDAOperator3D{TAlpha<:AbstractVector} <: AbstractMatrix{ComplexF64}
    grid::VoxelGrid3D
    k0::Float64
    alpha::TAlpha
    radiative_correction::Bool
    kernel::FFTEMDDAKernel3D
    qhat::Array{ComplexF64,4}
    conv::Array{ComplexF64,3}
    work_lock::ReentrantLock
end

FFTEMDDAOperator3D(grid::VoxelGrid3D, k0::Float64,
                   alpha::TAlpha, radiative_correction::Bool,
                   kernel::FFTEMDDAKernel3D, qhat::Array{ComplexF64,4},
                   conv::Array{ComplexF64,3}) where
                  {TAlpha<:AbstractVector} =
    FFTEMDDAOperator3D{TAlpha}(grid, k0, alpha, radiative_correction,
                              kernel, qhat, conv, ReentrantLock())

@inline _fft_dda_mod_index(offset::Int, nfft::Int) = mod(offset, nfft) + 1

@inline function _fft_convolution_range_safe_3d(
        kernel_scale::Float64,
        transformed_dipole_scale::Float64,
        channels::Int,
        transform_size::Int)
    isfinite(kernel_scale) && isfinite(transformed_dipole_scale) ||
        return false
    (iszero(kernel_scale) || iszero(transformed_dipole_scale)) &&
        return true
    # A complex product needs at most two real products per component. The
    # channel reduction contributes `channels` terms and an unnormalized FFT
    # butterfly can transiently sum up to `transform_size` frequency values.
    factor = 2.0 * Float64(channels) * Float64(transform_size)
    isfinite(factor) || return false
    component_limit = floatmax(Float64) / factor
    return kernel_scale <= component_limit / transformed_dipole_scale
end

function fft_dda_kernel_3d(grid::VoxelGrid3D, k0::Real)
    k = _finite_positive_k0_3d(k0)
    interaction_scale = grid.volumes[1]

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    px, py, pz = 2nx - 1, 2ny - 1, 2nz - 1
    kernel_hat = Array{ComplexF64}(undef, px, py, pz, 3, 3)
    kernel = zeros(ComplexF64, px, py, pz)

    for a in 1:3, b in 1:3
        fill!(kernel, 0.0 + 0.0im)
        for oz in -(nz - 1):(nz - 1)
            iz = _fft_dda_mod_index(oz, pz)
            z = oz * grid.dz
            for oy in -(ny - 1):(ny - 1)
                iy = _fft_dda_mod_index(oy, py)
                y = oy * grid.dy
                for ox in -(nx - 1):(nx - 1)
                    ox == 0 && oy == 0 && oz == 0 && continue
                    ix = _fft_dda_mod_index(ox, px)
                    x = ox * grid.dx
                    scaled_G = _electric_dipole_alpha_block_3d(
                        Vec3(x, y, z), Vec3(0.0, 0.0, 0.0), k,
                        ComplexF64(interaction_scale))
                    kernel[ix, iy, iz] = scaled_G[a, b]
                end
            end
        end
        copyto!(view(kernel_hat, :, :, :, a, b), FFTW.fft(kernel))
    end

    return FFTDDAKernel3D((px, py, pz), interaction_scale, kernel_hat)
end

"""
    fft_dda_operator_3d(grid, k0, eps_r; radiative_correction=false)

Construct an FFT-accelerated DDA material operator for a uniform `VoxelGrid3D`.
The matvec matches `dda_operator_3d` while replacing the dense all-pairs sum by
a zero-padded block Toeplitz convolution over Cartesian grid offsets.
"""
function fft_dda_operator_3d(grid::VoxelGrid3D, k0::Real, eps_r;
                             radiative_correction::Bool=false)
    k = _finite_positive_k0_3d(k0)
    epsv = _coerce_epsr_material_3d(eps_r, grid.nvoxels)
    alpha = _dda_polarizabilities_from_coerced(
        grid, k, epsv, radiative_correction)
    kernel = fft_dda_kernel_3d(grid, k)
    px, py, pz = kernel.pad_dims
    qhat = zeros(ComplexF64, px, py, pz, 3)
    conv = zeros(ComplexF64, px, py, pz)
    return FFTDDAOperator3D(grid, k, epsv, alpha, radiative_correction, kernel, qhat, conv)
end

Base.size(A::FFTDDAOperator3D) = (3 * A.grid.nvoxels, 3 * A.grid.nvoxels)
Base.eltype(::Type{<:FFTDDAOperator3D}) = ComplexF64
Base.eltype(::FFTDDAOperator3D) = ComplexF64

function Base.getindex(A::FFTDDAOperator3D, row::Int, col::Int)
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
                            A::FFTDDAOperator3D,
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
    needs_direct_fallback = false
    lock(A.work_lock)
    try
        grid = A.grid
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        qhat = A.qhat
        conv = A.conv
        fill!(qhat, 0.0 + 0.0im)

        idx = 0
        try
            for iz in 1:nz, iy in 1:ny, ix in 1:nx
                idx += 1
                qvec = _scaled_alpha_apply_3d(
                    A.alpha[idx], _read_field_component(xread, idx),
                    A.kernel.interaction_scale, "scaled FFT DDA dipole", idx)
                for b in 1:3
                    qhat[ix, iy, iz, b] = qvec[b]
                end
            end
        catch err
            err isa OverflowError || rethrow()
            needs_direct_fallback = true
        end

        if !needs_direct_fallback
            for b in 1:3
                FFTW.fft!(view(qhat, :, :, :, b))
            end

            transformed_scale = _fft_array_component_scale_3d(qhat)
            needs_direct_fallback = !_fft_convolution_range_safe_3d(
                _fft_array_component_scale_3d(A.kernel.kernel_hat),
                transformed_scale,
                3,
                length(conv),
            )
        end

        if !needs_direct_fallback
            for a in 1:3
                fill!(conv, 0.0 + 0.0im)
                for b in 1:3
                    conv .+=
                        view(A.kernel.kernel_hat, :, :, :, a, b) .*
                        view(qhat, :, :, :, b)
                end
                FFTW.ifft!(conv)
                idx = 0
                for iz in 1:nz, iy in 1:ny, ix in 1:nx
                    idx += 1
                    row = _dda_index(idx, a)
                    value = xread[row] - conv[ix, iy, iz]
                    y[row] = overwrite ?
                        alpha_scale * value :
                        alpha_scale * value + beta_scale * y[row]
                end
            end
        end
    finally
        unlock(A.work_lock)
    end

    if needs_direct_fallback
        direct = DDAOperator3D(
            A.grid, A.k0, A.eps_r, A.alpha, A.radiative_correction)
        return LinearAlgebra.mul!(
            y, direct, xread, alpha_scale, beta_scale)
    end

    return y
end

LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                   A::FFTDDAOperator3D,
                   x::AbstractVector{ComplexF64}) =
    LinearAlgebra.mul!(y, A, x, one(ComplexF64), zero(ComplexF64))

@inline function _em_interaction_field_fft_3d(
    r::Vec3,
    k::Float64,
    b::Int,
    interaction_scale::Float64,
)
    origin = Vec3(0.0, 0.0, 0.0)
    scale = ComplexF64(interaction_scale)
    q = b <= 3 ? CVec3(ntuple(a -> a == b ? scale : 0.0 + 0im, 3)) :
        CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    m = b > 3 ? CVec3(ntuple(a -> a == b - 3 ? scale : 0.0 + 0im, 3)) :
        CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    return _em_interaction_apply_3d(r, origin, k, q, m)
end

function fft_em_dda_kernel_3d(grid::VoxelGrid3D, k0::Real)
    k = _finite_positive_k0_3d(k0)
    interaction_scale = grid.volumes[1]

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    px, py, pz = 2nx - 1, 2ny - 1, 2nz - 1
    kernel_hat = Array{ComplexF64}(undef, px, py, pz, 6, 6)
    # Build and transform the spatial kernel one source column at a time,
    # reusing a single padded buffer instead of materializing all 36 blocks
    # simultaneously. The interaction for source component `b` carries all six
    # output components `a`, so a single offset sweep fills `block[:, :, :, :]`
    # for that `b`; the buffer is then transformed in place, column by column.
    block = zeros(ComplexF64, px, py, pz, 6)

    for b in 1:6
        fill!(block, 0.0 + 0.0im)
        for oz in -(nz - 1):(nz - 1)
            iz = _fft_dda_mod_index(oz, pz)
            z = oz * grid.dz
            for oy in -(ny - 1):(ny - 1)
                iy = _fft_dda_mod_index(oy, py)
                y = oy * grid.dy
                for ox in -(nx - 1):(nx - 1)
                    ox == 0 && oy == 0 && oz == 0 && continue
                    ix = _fft_dda_mod_index(ox, px)
                    x = ox * grid.dx
                    E, H = _em_interaction_field_fft_3d(
                        Vec3(x, y, z), k, b, interaction_scale)
                    for a in 1:3
                        block[ix, iy, iz, a] = E[a]
                        block[ix, iy, iz, a + 3] = H[a]
                    end
                end
            end
        end
        for a in 1:6
            FFTW.fft!(view(block, :, :, :, a))
            copyto!(view(kernel_hat, :, :, :, a, b), view(block, :, :, :, a))
        end
    end

    return FFTEMDDAKernel3D((px, py, pz), interaction_scale, kernel_hat)
end

"""
    fft_em_dda_operator_3d(grid, k0, eps_r, mu_r; radiative_correction=false)

Construct an FFT-accelerated coupled electric-magnetic DDA operator for a
uniform `VoxelGrid3D`.
"""
function fft_em_dda_operator_3d(grid::VoxelGrid3D, k0::Real, eps_r, mu_r;
                                radiative_correction::Bool=false)
    k = _finite_positive_k0_3d(k0)
    alpha = em_dda_polarizabilities(
        grid, k, eps_r, mu_r;
        radiative_correction=radiative_correction,
    )
    kernel = fft_em_dda_kernel_3d(grid, k)
    px, py, pz = kernel.pad_dims
    qhat = zeros(ComplexF64, px, py, pz, 6)
    conv = zeros(ComplexF64, px, py, pz)
    return FFTEMDDAOperator3D(grid, k, alpha, radiative_correction, kernel, qhat, conv)
end

function fft_em_dda_operator_3d(grid::VoxelGrid3D, k0::Real, alpha6;
                                radiative_correction::Bool=false)
    k = _finite_positive_k0_3d(k0)
    alpha = em_dda_polarizabilities(
        grid, k, alpha6;
        radiative_correction=radiative_correction,
    )
    kernel = fft_em_dda_kernel_3d(grid, k)
    px, py, pz = kernel.pad_dims
    qhat = zeros(ComplexF64, px, py, pz, 6)
    conv = zeros(ComplexF64, px, py, pz)
    return FFTEMDDAOperator3D(grid, k, alpha, radiative_correction, kernel, qhat, conv)
end

function fft_em_dda_operator_3d(grid::VoxelGrid3D, k0::Real,
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
    kernel = fft_em_dda_kernel_3d(grid, k)
    px, py, pz = kernel.pad_dims
    qhat = zeros(ComplexF64, px, py, pz, 6)
    conv = zeros(ComplexF64, px, py, pz)
    return FFTEMDDAOperator3D(grid, k, alpha, radiative_correction, kernel, qhat, conv)
end

Base.size(A::FFTEMDDAOperator3D) = (6 * A.grid.nvoxels, 6 * A.grid.nvoxels)
Base.eltype(::Type{<:FFTEMDDAOperator3D}) = ComplexF64
Base.eltype(::FFTEMDDAOperator3D) = ComplexF64

function Base.getindex(A::FFTEMDDAOperator3D, row::Int, col::Int)
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
    q, m = _split_em_field(alphaj * basis)
    E, H = _em_interaction_apply_3d(A.grid.centers[i], A.grid.centers[j],
                                    A.k0, q, m)
    return -(a <= 3 ? E[a] : H[a - 3])
end

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                            A::FFTEMDDAOperator3D,
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
    needs_direct_fallback = false
    lock(A.work_lock)
    try
        grid = A.grid
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        qhat = A.qhat
        conv = A.conv
        fill!(qhat, 0.0 + 0.0im)

        idx = 0
        try
            for iz in 1:nz, iy in 1:ny, ix in 1:nx
                idx += 1
                q6 = _scaled_alpha_apply_3d(
                    A.alpha[idx], _read_em_field6(xread, idx),
                    A.kernel.interaction_scale,
                    "scaled FFT EM-DDA dipole", idx)
                for b in 1:6
                    qhat[ix, iy, iz, b] = q6[b]
                end
            end
        catch err
            err isa OverflowError || rethrow()
            needs_direct_fallback = true
        end

        if !needs_direct_fallback
            for b in 1:6
                FFTW.fft!(view(qhat, :, :, :, b))
            end

            transformed_scale = _fft_array_component_scale_3d(qhat)
            needs_direct_fallback = !_fft_convolution_range_safe_3d(
                _fft_array_component_scale_3d(A.kernel.kernel_hat),
                transformed_scale,
                6,
                length(conv),
            )
        end

        if !needs_direct_fallback
            for a in 1:6
                fill!(conv, 0.0 + 0.0im)
                for b in 1:6
                    conv .+=
                        view(A.kernel.kernel_hat, :, :, :, a, b) .*
                        view(qhat, :, :, :, b)
                end
                FFTW.ifft!(conv)
                idx = 0
                for iz in 1:nz, iy in 1:ny, ix in 1:nx
                    idx += 1
                    row = _em_index(idx, a)
                    value = xread[row] - conv[ix, iy, iz]
                    y[row] = overwrite ?
                        alpha_scale * value :
                        alpha_scale * value + beta_scale * y[row]
                end
            end
        end
    finally
        unlock(A.work_lock)
    end

    if needs_direct_fallback
        direct = EMDDAOperator3D(
            A.grid, A.k0, A.alpha, A.radiative_correction)
        return LinearAlgebra.mul!(
            y, direct, xread, alpha_scale, beta_scale)
    end

    return y
end

LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                   A::FFTEMDDAOperator3D,
                   x::AbstractVector{ComplexF64}) =
    LinearAlgebra.mul!(y, A, x, one(ComplexF64), zero(ComplexF64))
