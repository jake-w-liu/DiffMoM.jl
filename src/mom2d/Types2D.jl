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

    dx = (x_range[2] - x_range[1]) / nx
    dy = (y_range[2] - y_range[1]) / ny
    _validate_positive_finite_2d(dx, "Mesh2D dx")
    _validate_positive_finite_2d(dy, "Mesh2D dy")
    cell_area = dx * dy
    _validate_positive_finite_2d(cell_area, "Mesh2D cell area")
    ncells = _checked_cell_count_2d(nx, ny)

    centers = Vector{Vec2}(undef, ncells)
    idx = 0
    for iy in 1:ny
        yc = y_range[1] + (iy - 0.5) * dy
        for ix in 1:nx
            xc = x_range[1] + (ix - 0.5) * dx
            idx += 1
            centers[idx] = Vec2(xc, yc)
        end
    end
    all(center -> all(isfinite, center), centers) ||
        throw(ArgumentError(
            "Mesh2D center construction produced non-finite coordinates."))

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
    expected_area = mesh.dx * mesh.dy
    isfinite(expected_area) && mesh.cell_area == expected_area ||
        throw(ArgumentError(
            "Mesh2D cell_area=$(mesh.cell_area) is inconsistent with dx*dy=$expected_area."))
    isfinite(mesh.x0) ||
        throw(ArgumentError("Mesh2D x0 must be finite, got $(mesh.x0)."))
    isfinite(mesh.y0) ||
        throw(ArgumentError("Mesh2D y0 must be finite, got $(mesh.y0)."))
    all(center -> all(isfinite, center), mesh.centers) ||
        throw(ArgumentError("Mesh2D centers must contain only finite coordinates."))
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
struct VIEResult2D
    E_total::Vector{ComplexF64}     # total field at cell centers
    E_inc::Vector{ComplexF64}       # incident field at cell centers
    chi::Vector{Float64}            # contrast profile (εᵣ - 1)
    D::Matrix{ComplexF64}           # Green's function integral matrix
    Z::Matrix{ComplexF64}           # system matrix (I - k² D diag(χ))
    Z_LU::LinearAlgebra.LU{ComplexF64, Matrix{ComplexF64}, Vector{Int64}}
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
