# Types2D.jl — Core data structures for 2D TM Method of Moments
#
# Convention: exp(+iωt), 2D scalar (TM polarization, E_z only)

export Vec2, CVec2, Mesh2D, VIEResult2D, equivalent_radius

const Vec2 = SVector{2,Float64}
const CVec2 = SVector{2,ComplexF64}

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
    @assert nx >= 1 && ny >= 1 "Grid must have at least 1 cell in each direction"
    @assert x_range[2] > x_range[1] && y_range[2] > y_range[1] "Ranges must be increasing"

    dx = (x_range[2] - x_range[1]) / nx
    dy = (y_range[2] - y_range[1]) / ny
    cell_area = dx * dy
    ncells = nx * ny

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

    return Mesh2D(centers, cell_area, ncells, nx, ny, dx, dy, x_range[1], y_range[1])
end

"""
    equivalent_radius(mesh::Mesh2D)

Equivalent circular cell radius for self-cell integration: πa² = cell_area.
"""
equivalent_radius(mesh::Mesh2D) = sqrt(mesh.cell_area / π)

"""
    VIEResult2D

Result from 2D VIE forward solve.
"""
struct VIEResult2D
    E_total::Vector{ComplexF64}     # total field at cell centers
    E_inc::Vector{ComplexF64}       # incident field at cell centers
    chi::Vector{Float64}            # contrast profile (εᵣ - 1)
    D::Matrix{ComplexF64}           # Green's function integral matrix
    Z::Matrix{ComplexF64}           # system matrix (I - k² diag(χ) D)
    Z_LU::LinearAlgebra.LU{ComplexF64, Matrix{ComplexF64}, Vector{Int64}}
    mesh::Mesh2D
    k0::Float64                     # free-space wavenumber
end
