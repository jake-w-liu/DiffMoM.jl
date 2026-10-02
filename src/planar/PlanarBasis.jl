# PlanarBasis.jl — Cell rasterization and subsection rooftop basis functions
#
# Metal on each interface is a Boolean cell mask over the uniform `CellGrid`.
# Basis functions are one-cell-wide rooftops placed on grid edges:
#   * an interior vertical edge e (1..nx-1) carries an x-directed rooftop
#     spanning cells (e,j) and (e+1,j) iff BOTH cells are metal;
#   * a boundary edge carries a half rooftop only where the sheet is
#     electrically connected to that sidewall (short or port; on a PMC
#     wall this is a mathematical connection — the wall carries no
#     return current);
#   * horizontal edges carry y-directed rooftops by the same rule.
# An edge that borders metal on only one side is an open boundary and gets no
# basis function, so J x n_hat = 0 holds naturally at metal edges.
#
# Modal inner products: a rooftop <f_p, e_mn> factors as an x-transform times
# a y-transform times trig factors at the basis centre.  Full rooftops use
#   T_full(k,h) = h * sinc^2(k*h/2)          (triangle, support 2h)
#   R(k,h)      = h * sinc(k*h/2)            (rectangle, width h)
# and half rooftops at a wall use the one-sided transforms
#   Hc(k,h) = (1 - cos(k*h)) / (k^2*h)       (cos transform of 1 - t/h)
#   Hs(k,h) = (k*h - sin(k*h)) / (k^2*h)     (sin transform)
# all evaluated with the small-argument series when |k*h| is tiny.

export SheetLevel, PlanarPort, PlanarBasisSet
export sheet_level, rasterize_rect!, rasterize_poly!
export build_planar_basis, planar_basis_count

"""Metal sheet on one interface: cell mask plus wall-connection flags.
`connect_west[j]`/`connect_east[j]` mark x=0/x=a edges of cell row j;
`connect_south[i]`/`connect_north[i]` mark y=0/y=b edges of cell column i."""
struct SheetLevel
    interface::Int
    mask::BitMatrix
    connect_west::BitVector
    connect_east::BitVector
    connect_south::BitVector
    connect_north::BitVector
end

"""Empty `SheetLevel` on `interface` for an `nx` x `ny` cell grid."""
function sheet_level(interface::Integer, nx::Integer, ny::Integer)
    return SheetLevel(interface, falses(nx, ny),
        falses(ny), falses(ny), falses(nx), falses(nx))
end

"""A co-calibrated port: a contiguous range of wall-connected edges.
`cells` indexes cell rows (west/east walls) or columns (south/north)."""
struct PlanarPort
    level::Int                 # index into the sheets array
    wall::Symbol               # :west :east :south :north
    cells::UnitRange{Int}
    z0::ComplexF64             # reference impedance for S-parameters [Ohm]
end

# ---------------- rasterization ----------------

"""Fill cells overlapped by axis-aligned rectangle [x0,x1] x [y0,y1].
A cell is metal when its centre lies inside the rectangle; `connected=true`
also marks wall-connection flags where the rectangle touches a sidewall."""
function rasterize_rect!(sheet::SheetLevel, grid::CellGrid,
        x0::Real, x1::Real, y0::Real, y1::Real; connected::Bool=false)
    0 <= x0 < x1 <= grid.a || throw(ArgumentError(
        "rectangle x-range [$x0,$x1] outside box [0,$(grid.a)]"))
    0 <= y0 < y1 <= grid.b || throw(ArgumentError(
        "rectangle y-range [$y0,$y1] outside box [0,$(grid.b)]"))
    @inbounds for j in 1:grid.ny
        yc = (j - 0.5) * grid.dy
        (y0 <= yc < y1) || continue
        for i in 1:grid.nx
            xc = (i - 0.5) * grid.dx
            if x0 <= xc < x1
                sheet.mask[i, j] = true
            end
        end
    end
    if connected
        half_dy = 0.5 * grid.dy
        half_dx = 0.5 * grid.dx
        if x0 <= half_dx
            @inbounds for j in 1:grid.ny
                yc = (j - 0.5) * grid.dy
                y0 <= yc < y1 && (sheet.connect_west[j] = true)
            end
        end
        if x1 >= grid.a - half_dx
            @inbounds for j in 1:grid.ny
                yc = (j - 0.5) * grid.dy
                y0 <= yc < y1 && (sheet.connect_east[j] = true)
            end
        end
        if y0 <= half_dy
            @inbounds for i in 1:grid.nx
                xc = (i - 0.5) * grid.dx
                x0 <= xc < x1 && (sheet.connect_south[i] = true)
            end
        end
        if y1 >= grid.b - half_dy
            @inbounds for i in 1:grid.nx
                xc = (i - 0.5) * grid.dx
                x0 <= xc < x1 && (sheet.connect_north[i] = true)
            end
        end
    end
    return sheet
end

"""Fill cells whose centre lies inside an arbitrary polygon given by vertex
lists `xs`, `ys` (ray-casting rule).  Connection flags are not modified."""
function rasterize_poly!(sheet::SheetLevel, grid::CellGrid,
        xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real})
    nv = length(xs)
    nv == length(ys) && nv >= 3 ||
        throw(ArgumentError("polygon needs >= 3 vertices"))
    @inbounds for j in 1:grid.ny
        yc = (j - 0.5) * grid.dy
        for i in 1:grid.nx
            xc = (i - 0.5) * grid.dx
            inside = false
            k = nv
            for v in 1:nv
                if (ys[v] > yc) != (ys[k] > yc)
                    xint = xs[k] + (xs[v] - xs[k]) * (yc - ys[k]) /
                                   (ys[v] - ys[k])
                    xc < xint && (inside = !inside)
                end
                k = v
            end
            inside && (sheet.mask[i, j] = true)
        end
    end
    return sheet
end

# ---------------- basis set ----------------

const _BASIS_X_FULL = UInt8(1)
const _BASIS_X_LO   = UInt8(2)   # half rooftop on the x=0 wall
const _BASIS_X_HI   = UInt8(3)   # half rooftop on the x=a wall
const _BASIS_Y_FULL = UInt8(4)
const _BASIS_Y_LO   = UInt8(5)   # half rooftop on the y=0 wall
const _BASIS_Y_HI   = UInt8(6)   # half rooftop on the y=b wall

"""Flat storage of rooftop basis functions built by `build_planar_basis`:
kind (direction/full/half-wall), cell-edge indices, sheet level, centre,
lateral width, and owning port index."""
struct PlanarBasisSet
    kind::Vector{UInt8}
    ei::Vector{Int}        # x-basis: edge index e (0..nx); y-basis: cell col i
    ej::Vector{Int}        # x-basis: cell row j;        y-basis: edge index f
    level::Vector{Int}     # sheet index
    x0::Vector{Float64}    # basis centre (peak for full, wall for half)
    y0::Vector{Float64}
    width::Vector{Float64} # lateral extent (port current weighting)
    port::Vector{Int}      # port index or 0
end

"""Number of basis functions in the set."""
planar_basis_count(b::PlanarBasisSet) = length(b.kind)

function _push_x!(b::PlanarBasisSet, kind, e, j, lv, x0, y0, dy, port)
    push!(b.kind, kind); push!(b.ei, e); push!(b.ej, j)
    push!(b.level, lv);  push!(b.x0, x0); push!(b.y0, y0)
    push!(b.width, dy);  push!(b.port, port)
    return nothing
end

function _push_y!(b::PlanarBasisSet, kind, i, f, lv, x0, y0, dx, port)
    push!(b.kind, kind); push!(b.ei, i); push!(b.ej, f)
    push!(b.level, lv);  push!(b.x0, x0); push!(b.y0, y0)
    push!(b.width, dx);  push!(b.port, port)
    return nothing
end

# edge->port lookup tables from the port list
function _wall_port_maps(ports::Vector{PlanarPort}, lv::Int,
        nx::Int, ny::Int)
    west = zeros(Int, ny); east = zeros(Int, ny)
    south = zeros(Int, nx); north = zeros(Int, nx)
    for (pidx, p) in enumerate(ports)
        p.level == lv || continue
        if p.wall === :west
            for j in p.cells
                1 <= j <= ny || throw(ArgumentError(
                    "port $pidx west cells $j outside 1:$ny"))
                west[j] == 0 || throw(ArgumentError(
                    "ports overlap on west edge row $j"))
                west[j] = pidx
            end
        elseif p.wall === :east
            for j in p.cells
                1 <= j <= ny || throw(ArgumentError(
                    "port $pidx east cells $j outside 1:$ny"))
                east[j] == 0 || throw(ArgumentError(
                    "ports overlap on east edge row $j"))
                east[j] = pidx
            end
        elseif p.wall === :south
            for i in p.cells
                1 <= i <= nx || throw(ArgumentError(
                    "port $pidx south cells $i outside 1:$nx"))
                south[i] == 0 || throw(ArgumentError(
                    "ports overlap on south edge col $i"))
                south[i] = pidx
            end
        elseif p.wall === :north
            for i in p.cells
                1 <= i <= nx || throw(ArgumentError(
                    "port $pidx north cells $i outside 1:$nx"))
                north[i] == 0 || throw(ArgumentError(
                    "ports overlap on north edge col $i"))
                north[i] = pidx
            end
        else
            throw(ArgumentError(
                "port $pidx wall must be :west/:east/:south/:north"))
        end
    end
    return west, east, south, north
end

"""
    build_planar_basis(grid, sheets, ports) -> PlanarBasisSet

Enumerate all x- and y-directed rooftop basis functions on every metal
level.  `sheets[lv].interface` is the stackup interface index the mask lives
on; `ports` claim connected wall edges (an edge marked `connect_*` but not
in any port is a galvanic short to the sidewall).
"""
function build_planar_basis(grid::CellGrid, sheets::Vector{SheetLevel},
        ports::Vector{PlanarPort})
    nx, ny = grid.nx, grid.ny
    b = PlanarBasisSet(UInt8[], Int[], Int[], Int[], Float64[],
                       Float64[], Float64[], Int[])
    for (lv, sheet) in enumerate(sheets)
        size(sheet.mask) == (nx, ny) || throw(DimensionMismatch(
            "sheet $lv mask size $(size(sheet.mask)) != grid ($nx,$ny)"))
        west, east, south, north = _wall_port_maps(ports, lv, nx, ny)

        # --- x-directed rooftops on vertical edges e = 0..nx ---
        @inbounds for j in 1:ny, e in 0:nx
            left = e >= 1 ? sheet.mask[e, j] : false
            right = e <= nx - 1 ? sheet.mask[e + 1, j] : false
            if e == 0
                if right && sheet.connect_west[j]
                    _push_x!(b, _BASIS_X_LO, e, j, lv,
                             0.0, (j - 0.5) * grid.dy, grid.dy, west[j])
                end
            elseif e == nx
                if left && sheet.connect_east[j]
                    _push_x!(b, _BASIS_X_HI, e, j, lv,
                             grid.a, (j - 0.5) * grid.dy, grid.dy, east[j])
                end
            else
                if left && right
                    _push_x!(b, _BASIS_X_FULL, e, j, lv,
                             e * grid.dx, (j - 0.5) * grid.dy, grid.dy, 0)
                end
            end
        end

        # --- y-directed rooftops on horizontal edges f = 0..ny ---
        @inbounds for f in 0:ny, i in 1:nx
            below = f >= 1 ? sheet.mask[i, f] : false
            above = f <= ny - 1 ? sheet.mask[i, f + 1] : false
            if f == 0
                if above && sheet.connect_south[i]
                    _push_y!(b, _BASIS_Y_LO, i, f, lv,
                             (i - 0.5) * grid.dx, 0.0, grid.dx, south[i])
                end
            elseif f == ny
                if below && sheet.connect_north[i]
                    _push_y!(b, _BASIS_Y_HI, i, f, lv,
                             (i - 0.5) * grid.dx, grid.b, grid.dx, north[i])
                end
            else
                if below && above
                    _push_y!(b, _BASIS_Y_FULL, i, f, lv,
                             (i - 0.5) * grid.dx, f * grid.dy, grid.dx, 0)
                end
            end
        end
    end
    return b
end
