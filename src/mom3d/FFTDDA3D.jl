# FFTDDA3D.jl -- FFT-accelerated DDA matvecs on uniform Cartesian voxel grids

export FFTDDAKernel3D, FFTDDAOperator3D
export fft_dda_kernel_3d, fft_dda_operator_3d
export FFTEMDDAKernel3D, FFTEMDDAOperator3D
export fft_em_dda_kernel_3d, fft_em_dda_operator_3d

"""
    FFTDDAKernel3D

Fourier-space block Toeplitz embedding of the free-space DDA dyadic for a
uniform `VoxelGrid3D`. The stored kernel excludes the singular self offset.
"""
struct FFTDDAKernel3D
    pad_dims::NTuple{3,Int}
    kernel_hat::Array{ComplexF64,5}
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
end

"""
    FFTEMDDAKernel3D

Fourier-space block Toeplitz embedding of the coupled electric-magnetic DDA
interaction. The stored kernel maps induced `[q; m]` dipoles to scattered
`[E; H]` fields and excludes the singular self offset.
"""
struct FFTEMDDAKernel3D
    pad_dims::NTuple{3,Int}
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
end

@inline _fft_dda_mod_index(offset::Int, nfft::Int) = mod(offset, nfft) + 1

function fft_dda_kernel_3d(grid::VoxelGrid3D, k0::Real)
    k = Float64(k0)
    k > 0 || error("k0 must be positive.")

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
                    G = electric_dipole_dyadic_3d(Vec3(x, y, z), Vec3(0.0, 0.0, 0.0), k)
                    kernel[ix, iy, iz] = G[a, b]
                end
            end
        end
        copyto!(view(kernel_hat, :, :, :, a, b), FFTW.fft(kernel))
    end

    return FFTDDAKernel3D((px, py, pz), kernel_hat)
end

"""
    fft_dda_operator_3d(grid, k0, eps_r; radiative_correction=false)

Construct an FFT-accelerated DDA material operator for a uniform `VoxelGrid3D`.
The matvec matches `dda_operator_3d` while replacing the dense all-pairs sum by
a zero-padded block Toeplitz convolution over Cartesian grid offsets.
"""
function fft_dda_operator_3d(grid::VoxelGrid3D, k0::Real, eps_r;
                             radiative_correction::Bool=false)
    k = Float64(k0)
    k > 0 || error("k0 must be positive.")
    epsv = _coerce_epsr_material_3d(eps_r, grid.nvoxels)
    alpha = dda_polarizabilities(grid, k, epsv; radiative_correction=radiative_correction)
    kernel = fft_dda_kernel_3d(grid, k)
    px, py, pz = kernel.pad_dims
    qhat = zeros(ComplexF64, px, py, pz, 3)
    conv = zeros(ComplexF64, px, py, pz)
    return FFTDDAOperator3D(grid, k, epsv, alpha, radiative_correction, kernel, qhat, conv)
end

Base.size(A::FFTDDAOperator3D) = (3 * A.grid.nvoxels, 3 * A.grid.nvoxels)
Base.size(A::FFTDDAOperator3D, d::Int) = d <= 2 ? 3 * A.grid.nvoxels : 1
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

    xread = y === x ? copy(x) : x
    grid = A.grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    px, py, pz = A.kernel.pad_dims

    qhat = A.qhat
    conv = A.conv
    fill!(qhat, 0.0 + 0.0im)

    idx = 0
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        idx += 1
        qvec = _alpha_apply(A.alpha[idx], _read_field_component(xread, idx))
        for b in 1:3
            qhat[ix, iy, iz, b] = qvec[b]
        end
    end

    for b in 1:3
        FFTW.fft!(view(qhat, :, :, :, b))
    end

    for a in 1:3
        fill!(conv, 0.0 + 0.0im)
        for b in 1:3
            conv .+= view(A.kernel.kernel_hat, :, :, :, a, b) .* view(qhat, :, :, :, b)
        end
        FFTW.ifft!(conv)
        idx = 0
        for iz in 1:nz, iy in 1:ny, ix in 1:nx
            idx += 1
            row = _dda_index(idx, a)
            value = xread[row] - conv[ix, iy, iz]
            y[row] = alpha_scale * value + beta_scale * y[row]
        end
    end

    return y
end

LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                   A::FFTDDAOperator3D,
                   x::AbstractVector{ComplexF64}) =
    LinearAlgebra.mul!(y, A, x, one(ComplexF64), zero(ComplexF64))

@inline function _em_interaction_field_fft_3d(r::Vec3, k::Float64, b::Int)
    origin = Vec3(0.0, 0.0, 0.0)
    q = b <= 3 ? CVec3(ntuple(a -> a == b ? 1.0 + 0im : 0.0 + 0im, 3)) :
        CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    m = b > 3 ? CVec3(ntuple(a -> a == b - 3 ? 1.0 + 0im : 0.0 + 0im, 3)) :
        CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    return _em_interaction_apply_3d(r, origin, k, q, m)
end

function fft_em_dda_kernel_3d(grid::VoxelGrid3D, k0::Real)
    k = Float64(k0)
    k > 0 || error("k0 must be positive.")

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
                    E, H = _em_interaction_field_fft_3d(Vec3(x, y, z), k, b)
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

    return FFTEMDDAKernel3D((px, py, pz), kernel_hat)
end

"""
    fft_em_dda_operator_3d(grid, k0, eps_r, mu_r; radiative_correction=false)

Construct an FFT-accelerated coupled electric-magnetic DDA operator for a
uniform `VoxelGrid3D`.
"""
function fft_em_dda_operator_3d(grid::VoxelGrid3D, k0::Real, eps_r, mu_r;
                                radiative_correction::Bool=false)
    k = Float64(k0)
    k > 0 || error("k0 must be positive.")
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
    k = Float64(k0)
    k > 0 || error("k0 must be positive.")
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
    k = Float64(k0)
    k > 0 || error("k0 must be positive.")
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
Base.size(A::FFTEMDDAOperator3D, d::Int) = d <= 2 ? 6 * A.grid.nvoxels : 1
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

    xread = y === x ? copy(x) : x
    grid = A.grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    qhat = A.qhat
    conv = A.conv
    fill!(qhat, 0.0 + 0.0im)

    idx = 0
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        idx += 1
        q6 = A.alpha[idx] * _read_em_field6(xread, idx)
        for b in 1:6
            qhat[ix, iy, iz, b] = q6[b]
        end
    end

    for b in 1:6
        FFTW.fft!(view(qhat, :, :, :, b))
    end

    for a in 1:6
        fill!(conv, 0.0 + 0.0im)
        for b in 1:6
            conv .+= view(A.kernel.kernel_hat, :, :, :, a, b) .* view(qhat, :, :, :, b)
        end
        FFTW.ifft!(conv)
        idx = 0
        for iz in 1:nz, iy in 1:ny, ix in 1:nx
            idx += 1
            row = _em_index(idx, a)
            value = xread[row] - conv[ix, iy, iz]
            y[row] = alpha_scale * value + beta_scale * y[row]
        end
    end

    return y
end

LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                   A::FFTEMDDAOperator3D,
                   x::AbstractVector{ComplexF64}) =
    LinearAlgebra.mul!(y, A, x, one(ComplexF64), zero(ComplexF64))
