# Types3D.jl -- Core data structures for 3D material volume solvers
#
# The 3D material path uses a vector discrete-dipole / volume-integral
# discretization. Unknowns are the three components of the total electric field
# at voxel centers.

export VoxelGrid3D, DDAOperator3D, DDAAdjointOperator3D, DDAResult3D
export EMDDAOperator3D, EMDDAResult3D, make_voxel_grid_3d

@inline function _checked_voxel_count_3d(nx::Int, ny::Int, nz::Int)
    try
        return Base.checked_mul(Base.checked_mul(nx, ny), nz)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError(
            "VoxelGrid3D dimensions overflow Int: nx=$nx, ny=$ny, nz=$nz."))
    end
end

"""
    VoxelGrid3D

Uniform Cartesian voxel grid for 3D material scattering. Each voxel stores a
center and volume; material properties are supplied separately as one value per
voxel so the same grid can be reused for sweeps.
"""
struct VoxelGrid3D
    centers::Vector{Vec3}
    volumes::Vector{Float64}
    nvoxels::Int
    nx::Int
    ny::Int
    nz::Int
    dx::Float64
    dy::Float64
    dz::Float64
    x0::Float64
    y0::Float64
    z0::Float64
    function VoxelGrid3D(centers::Vector{Vec3},
                         volumes::Vector{Float64},
                         nvoxels::Int,
                         nx::Int,
                         ny::Int,
                         nz::Int,
                         dx::Float64,
                         dy::Float64,
                         dz::Float64,
                         x0::Float64,
                         y0::Float64,
                         z0::Float64)
        nx >= 1 && ny >= 1 && nz >= 1 ||
            throw(ArgumentError(
                "VoxelGrid3D requires nx, ny, nz >= 1."))
        expected_nvoxels = _checked_voxel_count_3d(nx, ny, nz)
        nvoxels == expected_nvoxels ||
            throw(DimensionMismatch(
                "VoxelGrid3D nvoxels=$nvoxels, expected nx*ny*nz=$expected_nvoxels."))
        length(centers) == nvoxels ||
            throw(DimensionMismatch(
                "VoxelGrid3D has $(length(centers)) centers, expected $nvoxels."))
        length(volumes) == nvoxels ||
            throw(DimensionMismatch(
                "VoxelGrid3D has $(length(volumes)) volumes, expected $nvoxels."))
        all(isfinite, (dx, dy, dz)) &&
            dx > 0.0 && dy > 0.0 && dz > 0.0 ||
            throw(ArgumentError(
                "VoxelGrid3D spacings must be finite and positive."))
        all(isfinite, (x0, y0, z0)) ||
            throw(ArgumentError(
                "VoxelGrid3D origins must be finite."))

        expected_volume = dx * dy * dz
        isfinite(expected_volume) && expected_volume > 0.0 ||
            throw(ArgumentError(
                "VoxelGrid3D spacing product must be finite and positive."))
        @inbounds for j in eachindex(centers, volumes)
            all(isfinite, centers[j]) ||
                throw(ArgumentError(
                    "VoxelGrid3D center $j must be finite, got $(centers[j])."))
            (isfinite(volumes[j]) && volumes[j] == expected_volume) ||
                throw(ArgumentError(
                    "VoxelGrid3D volume $j must equal dx*dy*dz=$expected_volume, got $(volumes[j])."))
        end

        index = 0
        @inbounds for iz in 1:nz
            expected_z = z0 + (iz - 0.5) * dz
            for iy in 1:ny
                expected_y = y0 + (iy - 0.5) * dy
                for ix in 1:nx
                    expected_x = x0 + (ix - 0.5) * dx
                    index += 1
                    center = centers[index]
                    (isfinite(expected_x) &&
                     isfinite(expected_y) &&
                     isfinite(expected_z) &&
                     isapprox(center[1], expected_x;
                              rtol=8eps(Float64), atol=0.0) &&
                     isapprox(center[2], expected_y;
                              rtol=8eps(Float64), atol=0.0) &&
                     isapprox(center[3], expected_z;
                              rtol=8eps(Float64), atol=0.0)) ||
                        throw(ArgumentError(
                            "VoxelGrid3D center $index is inconsistent with the Cartesian grid."))
                end
            end
        end

        return new(
            centers, volumes, nvoxels, nx, ny, nz,
            dx, dy, dz, x0, y0, z0)
    end
end

"""
    EMDDAOperator3D

Matrix-free coupled electric-magnetic DDA operator. Unknowns are the total
electric and magnetic fields at each voxel, stored as six components per voxel:
`(Ex,Ey,Ez,Hx,Hy,Hz)`.
"""
struct EMDDAOperator3D{TAlpha<:AbstractVector} <: AbstractMatrix{ComplexF64}
    grid::VoxelGrid3D
    k0::Float64
    alpha::TAlpha
    radiative_correction::Bool
end

"""
    EMDDAResult3D

Result from a coupled electric-magnetic DDA solve.
"""
struct EMDDAResult3D{TAlpha<:AbstractVector,TA<:AbstractMatrix{ComplexF64},TLU,TStats}
    E_total::Vector{CVec3}
    H_total::Vector{CVec3}
    E_inc::Vector{CVec3}
    H_inc::Vector{CVec3}
    alpha::TAlpha
    A::TA
    A_LU::TLU
    solver::Symbol
    stats::TStats
    grid::VoxelGrid3D
    k0::Float64
    radiative_correction::Bool
end

"""
    VoxelGrid3D(x_range, y_range, z_range, nx, ny, nz)

Create a uniform Cartesian voxel grid over
`[x_range[1], x_range[2]] x [y_range[1], y_range[2]] x [z_range[1], z_range[2]]`.
"""
function VoxelGrid3D(x_range::Tuple{<:Real,<:Real},
                     y_range::Tuple{<:Real,<:Real},
                     z_range::Tuple{<:Real,<:Real},
                     nx::Int, ny::Int, nz::Int)
    nx >= 1 && ny >= 1 && nz >= 1 || error("VoxelGrid3D requires nx, ny, nz >= 1.")
    x1, x2 = Float64(x_range[1]), Float64(x_range[2])
    y1, y2 = Float64(y_range[1]), Float64(y_range[2])
    z1, z2 = Float64(z_range[1]), Float64(z_range[2])
    all(isfinite, (x1, x2, y1, y2, z1, z2)) ||
        throw(ArgumentError("VoxelGrid3D range endpoints must be finite."))
    x2 > x1 || error("x_range must be increasing.")
    y2 > y1 || error("y_range must be increasing.")
    z2 > z1 || error("z_range must be increasing.")

    nvoxels = _checked_voxel_count_3d(nx, ny, nz)
    dx = (x2 - x1) / nx
    dy = (y2 - y1) / ny
    dz = (z2 - z1) / nz
    all(isfinite, (dx, dy, dz)) && dx > 0 && dy > 0 && dz > 0 ||
        throw(ArgumentError(
            "VoxelGrid3D spacings must be finite and positive."))
    volume = dx * dy * dz
    isfinite(volume) && volume > 0 ||
        throw(ArgumentError(
            "VoxelGrid3D voxel volume must be finite and positive, got $volume."))

    centers = Vector{Vec3}(undef, nvoxels)
    volumes = fill(volume, nvoxels)

    idx = 0
    for iz in 1:nz
        zc = z1 + (iz - 0.5) * dz
        for iy in 1:ny
            yc = y1 + (iy - 0.5) * dy
            for ix in 1:nx
                xc = x1 + (ix - 0.5) * dx
                idx += 1
                centers[idx] = Vec3(xc, yc, zc)
            end
        end
    end

    return VoxelGrid3D(centers, volumes, nvoxels, nx, ny, nz, dx, dy, dz,
                       x1, y1, z1)
end

make_voxel_grid_3d(x_range, y_range, z_range, nx::Int, ny::Int, nz::Int) =
    VoxelGrid3D(x_range, y_range, z_range, nx, ny, nz)

"""
    DDAOperator3D

Matrix-free coupled-dipole operator for 3D material scattering. It represents
the same linear system as `assemble_dda_3d` without storing the dense
`3N x 3N` matrix.
"""
struct DDAOperator3D{TEps<:AbstractVector,TAlpha<:AbstractVector} <: AbstractMatrix{ComplexF64}
    grid::VoxelGrid3D
    k0::Float64
    eps_r::TEps
    alpha::TAlpha
    radiative_correction::Bool
end

"""
    DDAAdjointOperator3D

Hermitian-adjoint wrapper for `DDAOperator3D`.
"""
struct DDAAdjointOperator3D <: AbstractMatrix{ComplexF64}
    parent::DDAOperator3D
end

"""
    DDAResult3D

Result from a 3D discrete-dipole material solve. `alpha` is the normalized
electric polarizability `p / (eps0 * E)` in cubic meters, so induced dipoles are
`alpha[j] * E_total[j]`.
"""
struct DDAResult3D{TEps<:AbstractVector,TAlpha<:AbstractVector,
                   TA<:AbstractMatrix{ComplexF64},TLU,TStats}
    E_total::Vector{CVec3}
    E_inc::Vector{CVec3}
    eps_r::TEps
    alpha::TAlpha
    A::TA
    A_LU::TLU
    solver::Symbol
    stats::TStats
    grid::VoxelGrid3D
    k0::Float64
    radiative_correction::Bool
end
