# Types2D.jl — Core data structures for 2D TM Method of Moments
#
# Convention: exp(+iωt), 2D scalar (TM polarization, E_z only)

export Vec2, CVec2, Mesh2D, VIEResult2D, equivalent_radius

const Vec2 = SVector{2,Float64}
const CVec2 = SVector{2,ComplexF64}

@inline function _validate_finite_vec2_2d(value::Vec2, label::AbstractString)
    all(isfinite, value) ||
        throw(ArgumentError("$label components must be finite, got $value."))
    return nothing
end

@inline function _validate_positive_finite_2d(value::Real, label::AbstractString)
    isfinite(value) && value > 0 ||
        throw(ArgumentError("$label must be finite and positive, got $value."))
    return nothing
end

@inline function _checked_cell_count_2d(nx::Int, ny::Int)
    try
        return Base.checked_mul(nx, ny)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError("nx * ny overflows Int: nx=$nx, ny=$ny."))
    end
end

@inline _mesh_axis_center_2d(origin::Float64, spacing::Float64, index::Int) =
    muladd(Float64(index) - 0.5, spacing, origin)

function _validate_mesh_axis_2d(origin::Float64, spacing::Float64,
                                count::Int, label::AbstractString)
    previous = _mesh_axis_center_2d(origin, spacing, 1)
    isfinite(previous) ||
        throw(ArgumentError("Mesh2D $label-axis center 1 is non-finite."))
    @inbounds for index in 2:count
        current = _mesh_axis_center_2d(origin, spacing, index)
        (isfinite(current) && current > previous) ||
            throw(ArgumentError(
                "Mesh2D $label-axis centers are not representably distinct " *
                "at indices $(index - 1) and $index."))
        previous = current
    end
    return nothing
end

"""
    Mesh2D

Rectangular grid discretization of a 2D domain for volume integral equation (VIE).
Each cell has constant material properties and field values (pulse basis).
"""
struct Mesh2D
    centers::Vector{Vec2}       # cell center coordinates
    cell_area::Float64          # uniform cell area (dx * dy)
    ncells::Int
    nx::Int                     # cells in x
    ny::Int                     # cells in y
    dx::Float64                 # cell width
    dy::Float64                 # cell height
    x0::Float64                 # domain lower-left x
    y0::Float64                 # domain lower-left y
    function Mesh2D(centers::Vector{Vec2}, cell_area::Float64,
                    ncells::Int, nx::Int, ny::Int,
                    dx::Float64, dy::Float64,
                    x0::Float64, y0::Float64)
        mesh = new(centers, cell_area, ncells, nx, ny, dx, dy, x0, y0)
        _validate_mesh_2d(mesh)
        return mesh
    end
end

"""
    Mesh2D(x_range, y_range, nx, ny)

Create a uniform rectangular grid over [x_range[1], x_range[2]] × [y_range[1], y_range[2]]
with `nx × ny` cells.
"""
function Mesh2D(x_range::Tuple{Float64,Float64}, y_range::Tuple{Float64,Float64},
                nx::Int, ny::Int)
    nx >= 1 && ny >= 1 ||
        throw(ArgumentError(
            "Grid must have at least 1 cell in each direction, got nx=$nx, ny=$ny."))
    all(isfinite, x_range) ||
        throw(ArgumentError("x_range endpoints must be finite, got $x_range."))
    all(isfinite, y_range) ||
        throw(ArgumentError("y_range endpoints must be finite, got $y_range."))
    x_range[2] > x_range[1] ||
        throw(ArgumentError("x_range must be strictly increasing, got $x_range."))
    y_range[2] > y_range[1] ||
        throw(ArgumentError("y_range must be strictly increasing, got $y_range."))

    ncells = _checked_cell_count_2d(nx, ny)
    dx = (x_range[2] - x_range[1]) / nx
    dy = (y_range[2] - y_range[1]) / ny
    _validate_positive_finite_2d(dx, "Mesh2D dx")
    _validate_positive_finite_2d(dy, "Mesh2D dy")
    _validate_mesh_axis_2d(x_range[1], dx, nx, "x")
    _validate_mesh_axis_2d(y_range[1], dy, ny, "y")
    cell_area = dx * dy
    _validate_positive_finite_2d(cell_area, "Mesh2D cell area")

    centers = Vector{Vec2}(undef, ncells)
    idx = 0
    for iy in 1:ny
        yc = _mesh_axis_center_2d(y_range[1], dy, iy)
        for ix in 1:nx
            xc = _mesh_axis_center_2d(x_range[1], dx, ix)
            idx += 1
            centers[idx] = Vec2(xc, yc)
        end
    end

    return Mesh2D(centers, cell_area, ncells, nx, ny, dx, dy, x_range[1], y_range[1])
end

function _validate_mesh_2d(mesh::Mesh2D)
    mesh.nx >= 1 && mesh.ny >= 1 ||
        throw(ArgumentError(
            "Mesh2D dimensions must be positive, got nx=$(mesh.nx), ny=$(mesh.ny)."))
    expected_ncells = _checked_cell_count_2d(mesh.nx, mesh.ny)
    mesh.ncells == expected_ncells ||
        throw(DimensionMismatch(
            "Mesh2D ncells=$(mesh.ncells), expected nx*ny=$expected_ncells."))
    length(mesh.centers) == mesh.ncells ||
        throw(DimensionMismatch(
            "Mesh2D has $(length(mesh.centers)) centers, expected $(mesh.ncells)."))
    _validate_positive_finite_2d(mesh.dx, "Mesh2D dx")
    _validate_positive_finite_2d(mesh.dy, "Mesh2D dy")
    _validate_positive_finite_2d(mesh.cell_area, "Mesh2D cell area")
    isfinite(mesh.x0) ||
        throw(ArgumentError("Mesh2D x0 must be finite, got $(mesh.x0)."))
    isfinite(mesh.y0) ||
        throw(ArgumentError("Mesh2D y0 must be finite, got $(mesh.y0)."))
    _validate_mesh_axis_2d(mesh.x0, mesh.dx, mesh.nx, "x")
    _validate_mesh_axis_2d(mesh.y0, mesh.dy, mesh.ny, "y")
    expected_area = mesh.dx * mesh.dy
    isfinite(expected_area) && mesh.cell_area == expected_area ||
        throw(ArgumentError(
            "Mesh2D cell_area=$(mesh.cell_area) is inconsistent with dx*dy=$expected_area."))

    index = 0
    @inbounds for iy in 1:mesh.ny
        expected_y = _mesh_axis_center_2d(mesh.y0, mesh.dy, iy)
        for ix in 1:mesh.nx
            expected_x = _mesh_axis_center_2d(mesh.x0, mesh.dx, ix)
            index += 1
            center = mesh.centers[index]
            (center[1] == expected_x && center[2] == expected_y) ||
                throw(ArgumentError(
                    "Mesh2D center $index is inconsistent with the Cartesian grid."))
        end
    end
    return nothing
end

"""
    equivalent_radius(mesh::Mesh2D)

Equivalent circular cell radius for self-cell integration: πa² = cell_area.
"""
function equivalent_radius(mesh::Mesh2D)
    _validate_mesh_2d(mesh)
    return _equivalent_radius_unchecked(mesh)
end

@inline _equivalent_radius_unchecked(mesh::Mesh2D) =
    sqrt(mesh.cell_area) / sqrt(π)

"""
    VIEResult2D

Result from 2D VIE forward solve.
"""
struct VIEResult2D{F}
    E_total::Vector{ComplexF64}     # total field at cell centers
    E_inc::Vector{ComplexF64}       # incident field at cell centers
    chi::Vector{Float64}            # contrast profile (εᵣ - 1)
    D::Matrix{ComplexF64}           # Green's function integral matrix
    Z::Matrix{ComplexF64}           # system matrix (I - k² D diag(χ))
    Z_LU::F
    mesh::Mesh2D
    k0::Float64                     # free-space wavenumber
end

function _validate_vie_result_2d(vr::VIEResult2D; require_system::Bool=false)
    _validate_mesh_2d(vr.mesh)
    _validate_positive_finite_2d(vr.k0, "VIEResult2D k0")
    N = vr.mesh.ncells
    length(vr.E_total) == N ||
        throw(DimensionMismatch(
            "VIEResult2D E_total length $(length(vr.E_total)) != $N."))
    length(vr.E_inc) == N ||
        throw(DimensionMismatch(
            "VIEResult2D E_inc length $(length(vr.E_inc)) != $N."))
    length(vr.chi) == N ||
        throw(DimensionMismatch(
            "VIEResult2D chi length $(length(vr.chi)) != $N."))
    all(isfinite, vr.E_total) ||
        throw(ArgumentError("VIEResult2D E_total must contain only finite values."))
    all(isfinite, vr.E_inc) ||
        throw(ArgumentError("VIEResult2D E_inc must contain only finite values."))
    all(isfinite, vr.chi) ||
        throw(ArgumentError("VIEResult2D chi must contain only finite values."))
    if require_system
        size(vr.D) == (N, N) ||
            throw(DimensionMismatch(
                "VIEResult2D D has size $(size(vr.D)), expected ($N, $N)."))
        size(vr.Z) == (N, N) ||
            throw(DimensionMismatch(
                "VIEResult2D Z has size $(size(vr.Z)), expected ($N, $N)."))
        size(vr.Z_LU) == (N, N) ||
            throw(DimensionMismatch(
                "VIEResult2D Z_LU has size $(size(vr.Z_LU)), expected ($N, $N)."))
        all(isfinite, vr.D) ||
            throw(ArgumentError("VIEResult2D D must contain only finite values."))
        all(isfinite, vr.Z) ||
            throw(ArgumentError("VIEResult2D Z must contain only finite values."))
        issuccess(vr.Z_LU) ||
            throw(ArgumentError("VIEResult2D Z_LU is not a successful factorization."))
    end
    return nothing
end
