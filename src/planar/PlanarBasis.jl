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
# Via columns are z-directed volume-current unknowns on single cells: a
# `ViaLevel` on layer l contributes a uniform (constant in z) and/or an
# up-tapered (0 at bottom -> max at top) basis per marked cell, spanning
# interface l-1 (bottom face) to l (top face).  They couple to sheets only
# through the TM modal E_z field, and to each other through it plus the
# filament contact term on the shared layer.
#
# Modal inner products: a rooftop <f_p, e_mn> factors as an x-transform times
# a y-transform times trig factors at the basis centre.  Full rooftops use
#   T_full(k,h) = h * sinc^2(k*h/2)          (triangle, support 2h)
#   R(k,h)      = h * sinc(k*h/2)            (rectangle, width h)
# and half rooftops at a wall use the one-sided transforms
#   Hc(k,h) = (1 - cos(k*h)) / (k^2*h)       (cos transform of 1 - t/h)
#   Hs(k,h) = (k*h - sin(k*h)) / (k^2*h)     (sin transform)
# all evaluated with the small-argument series when |k*h| is tiny.

export SheetLevel, PlanarPort, PlanarReferencePlane, PlanarBasisSet
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

"""Per-port physical reference-plane shift. `length` is signed distance
[m] removed from the raw port launch; `zc` [Ω] and `gamma` [1/m] are
numbers or `f_hz -> value` providers, so broadband sweeps evaluate the
appropriate propagation at every frequency. Raw solves retain their gap
plane; [`planar_reference_planes`](@ref) produces calibrated results."""
struct PlanarReferencePlane{Z,G}
    length::Float64
    zc::Z
    gamma::G
    function PlanarReferencePlane(length::Real,zc,gamma)
        isfinite(length) || throw(ArgumentError("reference-plane distance must be finite"))
        for (provider,label) in ((zc,"zc"),(gamma,"gamma"))
            provider isa Number ? isfinite(provider) || throw(ArgumentError("$label must be finite")) :
                applicable(provider,1.0) || throw(ArgumentError("$label must be a number or frequency provider"))
        end
        return new{typeof(zc),typeof(gamma)}(Float64(length),zc,gamma)
    end
end

"""A planar gap-voltage port. Wall ports use `:west/:east/:south/:north`.
Internal zero-width gaps use `:internal_x/:internal_y` and an interior
grid `edge`; `cells` indexes transverse rows/columns. A `:via` port drives
the axial column on a via level, with `cells` indexing column-major cell
numbers. `:volume_west/:volume_east/:volume_south/:volume_north` drive
the corresponding wall of a `VolLevel`; `level` is its volume ordinal.
Its normalized axial profile gives trace weight equal to lateral width,
so the extracted current is the integral through the physical thickness.
`polarity` is +1 or -1. `refplane` optionally holds a physical
per-frequency reference-plane contract. The four-argument wall constructor
is preserved; raw solver results retain their original gap plane. `z0` may
be a finite complex impedance with positive real part or a frequency [Hz]
provider, including `PlanarPortImpedance`. Solves retain the evaluated
reference impedances; the physical gap voltage/current basis is unchanged."""
struct PlanarPort
    level::Int                 # index into the sheets array
    wall::Symbol               # :west :east :south :north
    cells::UnitRange{Int}
    z0::Any                    # finite positive-Re impedance or frequency provider
    edge::Int                 # internal shared-cell edge, otherwise 0
    polarity::Int             # terminal voltage/current polarity
    refplane::Union{Nothing,PlanarReferencePlane}
end

PlanarPort(level::Int,wall::Symbol,cells::UnitRange{Int},z0,
    edge::Int,polarity::Int) = PlanarPort(level,wall,cells,z0;edge=edge,polarity=polarity)

function PlanarPort(level::Integer, wall::Symbol, cells::UnitRange{Int},
        z0; edge::Integer=0, polarity::Integer=1,
        refplane::Union{Nothing,PlanarReferencePlane}=nothing)
    polarity in (-1, 1) || throw(ArgumentError("port polarity must be +1 or -1"))
    return PlanarPort(Int(level), wall, cells, _planar_store_reference(z0),
        Int(edge), Int(polarity),refplane)
end

"""`PlanarPort(level, direction, edge, cells, z0; polarity=1)` places an
internal infinitesimal gap on edge `edge`, flowing along `direction=:x`
or `:y`. Metal must occupy the cells on both sides of each claimed edge.
Positive voltage drives current along the positive coordinate direction.
Use `direction=:terminal_x/:terminal_y` and
`metal_side=:positive/:negative` for a one-sided rooftop on an open
sheet edge. `edge` identifies its physical grid plane and `cells` its
full transverse pad span. Positive terminal current flows into its metal
side. This mathematical gap-field source retains endpoint charge; it
does not create a conductive return to ground. Use
[`planar_terminal_returns`](@ref) for physical box-cover sources, or supply
explicit return geometry. `refplane` holds a supplied launch calibration;
Sonnet automatic local-ground calibration is not inferred.
"""
function PlanarPort(level::Integer, direction::Symbol, edge::Integer,
        cells::UnitRange{Int}, z0; polarity::Integer=1,
        refplane::Union{Nothing,PlanarReferencePlane}=nothing,
        metal_side::Symbol=:positive)
    direction in (:x,:y,:terminal_x,:terminal_y) || throw(ArgumentError(
        "port direction must be :x/:y or :terminal_x/:terminal_y"))
    wall = if direction in (:terminal_x,:terminal_y)
        metal_side in (:positive,:negative) || throw(ArgumentError(
            "terminal metal_side must be :positive or :negative"))
        direction===:terminal_x ? (metal_side===:positive ? :terminal_x_lo : :terminal_x_hi) :
            (metal_side===:positive ? :terminal_y_lo : :terminal_y_hi)
    else
        direction===:x ? :internal_x : :internal_y
    end
    return PlanarPort(level, wall,
        cells, z0; edge=edge, polarity=polarity,refplane=refplane)
end

@inline _is_planar_terminal(wall::Symbol) = wall in
    (:terminal_x_lo,:terminal_x_hi,:terminal_y_lo,:terminal_y_hi)
@inline _is_planar_x_terminal(wall::Symbol) = wall in (:terminal_x_lo,:terminal_x_hi)
@inline _is_planar_positive_terminal(wall::Symbol) = wall in (:terminal_x_lo,:terminal_y_lo)
@inline _is_planar_volume_port(wall::Symbol) = wall in
    (:volume_west,:volume_east,:volume_south,:volume_north)
@inline _planar_volume_wall(wall::Symbol) = wall===:volume_west ? :west :
    wall===:volume_east ? :east : wall===:volume_south ? :south : :north

# ---------------- rasterization ----------------

function _validate_sheet_grid(sheet::Union{SheetLevel,VolLevel}, grid::CellGrid)
    nx, ny = grid.nx, grid.ny
    size(sheet.mask) == (nx, ny) || throw(DimensionMismatch(
        "metal mask size $(size(sheet.mask)) != grid ($nx,$ny)"))
    length(sheet.connect_west) == ny && length(sheet.connect_east) == ny &&
        length(sheet.connect_south) == nx && length(sheet.connect_north) == nx ||
        throw(DimensionMismatch("metal wall-connection vectors must match the grid"))
    return nothing
end

"""Fill cells overlapped by axis-aligned rectangle [x0,x1] x [y0,y1].
A cell is metal when its centre lies inside the rectangle; `connected=true`
also marks wall-connection flags where the rectangle touches a sidewall."""
function rasterize_rect!(sheet::Union{SheetLevel,VolLevel}, grid::CellGrid,
        x0::Real, x1::Real, y0::Real, y1::Real; connected::Bool=false)
    _validate_sheet_grid(sheet, grid)
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
function rasterize_poly!(sheet::Union{SheetLevel,VolLevel}, grid::CellGrid,
        xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real})
    _validate_sheet_grid(sheet, grid)
    nv = length(xs)
    nv == length(ys) && nv >= 3 ||
        throw(ArgumentError("polygon needs >= 3 vertices"))
    all(isfinite, xs) && all(isfinite, ys) ||
        throw(ArgumentError("polygon vertices must be finite"))
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
const _BASIS_VIA_U  = UInt8(7)   # uniform z-directed via column
const _BASIS_VIA_T  = UInt8(8)   # up-tapered via column
const _BASIS_VX_FULL = UInt8(9)  # thick-metal volume rooftop, x-directed
const _BASIS_VX_LO   = UInt8(10)
const _BASIS_VX_HI   = UInt8(11)
const _BASIS_VY_FULL = UInt8(12)
const _BASIS_VY_LO   = UInt8(13)
const _BASIS_VY_HI   = UInt8(14)

# same-direction map volume -> sheet kind for transform/weight reuse
@inline _sheet_kind(kind::UInt8) = kind <= _BASIS_Y_HI ? kind :
    kind >= _BASIS_VX_FULL ? kind - UInt8(8) : kind
@inline _is_vol_kind(kind::UInt8) = kind >= _BASIS_VX_FULL
@inline _is_via_kind(kind::UInt8) =
    kind == _BASIS_VIA_U || kind == _BASIS_VIA_T

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

function _push_via!(b::PlanarBasisSet, kind, i, j, lv, x0, y0, area)
    push!(b.kind, kind); push!(b.ei, i); push!(b.ej, j)
    push!(b.level, lv);  push!(b.x0, x0); push!(b.y0, y0)
    push!(b.width, area);  push!(b.port, 0)
    return nothing
end

# edge->port lookup tables from the port list
function _wall_port_maps(ports::Vector{PlanarPort}, lv::Int,
        nx::Int, ny::Int; volume::Bool=false)
    west = zeros(Int, ny); east = zeros(Int, ny)
    south = zeros(Int, nx); north = zeros(Int, nx)
    for (pidx, p) in enumerate(ports)
        (p.wall in (:internal_x,:internal_y,:via) || _is_planar_terminal(p.wall)) && continue
        _is_planar_volume_port(p.wall)==volume || continue
        p.level == lv || continue
        wall=volume ? _planar_volume_wall(p.wall) : p.wall
        if wall === :west
            for j in p.cells
                1 <= j <= ny || throw(ArgumentError(
                    "port $pidx west cells $j outside 1:$ny"))
                west[j] == 0 || throw(ArgumentError(
                    "ports overlap on west edge row $j"))
                west[j] = pidx
            end
        elseif wall === :east
            for j in p.cells
                1 <= j <= ny || throw(ArgumentError(
                    "port $pidx east cells $j outside 1:$ny"))
                east[j] == 0 || throw(ArgumentError(
                    "ports overlap on east edge row $j"))
                east[j] = pidx
            end
        elseif wall === :south
            for i in p.cells
                1 <= i <= nx || throw(ArgumentError(
                    "port $pidx south cells $i outside 1:$nx"))
                south[i] == 0 || throw(ArgumentError(
                    "ports overlap on south edge col $i"))
                south[i] = pidx
            end
        elseif wall === :north
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
    build_planar_basis(grid, sheets, ports; vias=ViaLevel[], vols=VolLevel[])
        -> PlanarBasisSet

Enumerate all x- and y-directed rooftop basis functions on every metal
level plus the via columns in `vias` and the volume rooftops in `vols`.
`sheets[lv].interface` is the stackup interface index the mask lives on;
`ports` claim connected wall edges (an edge marked `connect_*` but not in
any port is a galvanic short to the sidewall).  `vias[vl].layer` is the
stackup layer the column spans; each marked cell contributes one basis
per via kind present.  `vols[vl].layer` is the stackup layer the thick
metal occupies through its full thickness; each marked cell contributes
volume rooftops that carry a normalized uniform z-profile across the
layer. Volume wall ports claim their wall-connected half rooftops.
"""
function build_planar_basis(grid::CellGrid, sheets::Vector{SheetLevel},
        ports::Vector{PlanarPort}; vias::Vector{ViaLevel}=ViaLevel[],
        vols::Vector{VolLevel}=VolLevel[])
    nx, ny = grid.nx, grid.ny
    b = PlanarBasisSet(UInt8[], Int[], Int[], Int[], Float64[],
                       Float64[], Float64[], Int[])
    for (pidx, p) in enumerate(ports)
        levels = p.wall === :via ? length(vias) :
            _is_planar_volume_port(p.wall) ? length(vols) : length(sheets)
        1 <= p.level <= levels || throw(ArgumentError(
            "port $pidx level $(p.level) outside 1:$levels"))
        isempty(p.cells) && throw(ArgumentError("port $pidx cell range is empty"))
        p.polarity in (-1, 1) || throw(ArgumentError("port $pidx polarity must be +1 or -1"))
        if p.wall in (:internal_x, :internal_y)
            axis_count = p.wall === :internal_x ? nx : ny
            1 <= p.edge < axis_count || throw(ArgumentError(
                "port $pidx internal edge outside 1:$(axis_count - 1)"))
            transverse_count = p.wall === :internal_x ? ny : nx
            1 <= first(p.cells) <= last(p.cells) <= transverse_count ||
                throw(ArgumentError("port $pidx internal cells outside grid"))
        elseif _is_planar_terminal(p.wall)
            axis_count = _is_planar_x_terminal(p.wall) ? nx : ny
            positive = _is_planar_positive_terminal(p.wall)
            (positive ? 0<=p.edge<axis_count : 0<p.edge<=axis_count) ||
                throw(ArgumentError("port $pidx terminal edge has no adjacent metal cell"))
            transverse_count = _is_planar_x_terminal(p.wall) ? ny : nx
            1<=first(p.cells)<=last(p.cells)<=transverse_count ||
                throw(ArgumentError("port $pidx terminal span lies outside the grid"))
        elseif p.wall === :via
            1 <= first(p.cells) <= last(p.cells) <= nx * ny ||
                throw(ArgumentError("port $pidx via cells outside grid"))
        elseif !(p.wall in (:west, :east, :south, :north) || _is_planar_volume_port(p.wall))
            throw(ArgumentError("port $pidx has unknown kind $(p.wall)"))
        end
    end
    for (lv, sheet) in enumerate(sheets)
        _validate_sheet_grid(sheet, grid)
        west, east, south, north = _wall_port_maps(ports, lv, nx, ny)
        for (wall, ids, connections) in ((:west, west, sheet.connect_west),
                (:east, east, sheet.connect_east),
                (:south, south, sheet.connect_south),
                (:north, north, sheet.connect_north))
            for idx in eachindex(ids)
                ids[idx] == 0 && continue
                metal = wall === :west ? sheet.mask[1, idx] :
                    wall === :east ? sheet.mask[nx, idx] :
                    wall === :south ? sheet.mask[idx, 1] : sheet.mask[idx, ny]
                metal && connections[idx] || throw(ArgumentError(
                    "port $(ids[idx]) includes unconnected $wall edge $idx"))
            end
        end

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
    _append_planar_terminal_ports!(b,ports,sheets,grid)
    for (vl, vlvl) in enumerate(vias)
        (size(vlvl.uni) == (nx, ny) && size(vlvl.tap) == (nx, ny)) ||
            throw(DimensionMismatch(
                "vias[$vl] masks must be $nx x $ny"))
        @inbounds for j in 1:ny, i in 1:nx
            x0 = (i - 0.5) * grid.dx
            y0 = (j - 0.5) * grid.dy
            vlvl.uni[i, j] && _push_via!(b, _BASIS_VIA_U, i, j, vl,
                x0, y0, grid.dx * grid.dy)
            vlvl.tap[i, j] && _push_via!(b, _BASIS_VIA_T, i, j, vl,
                x0, y0, grid.dx * grid.dy)
        end
    end
    # volume rooftops: same x/y edge enumeration as sheets on the layer
    # mask, shifted to the VX/VY kind range.
    for (vl, vol) in enumerate(vols)
        _validate_sheet_grid(vol, grid)
        west,east,south,north=_wall_port_maps(ports,vl,nx,ny;volume=true)
        for (wall,ids,connections) in ((:west,west,vol.connect_west),
                (:east,east,vol.connect_east),(:south,south,vol.connect_south),
                (:north,north,vol.connect_north))
            for idx in eachindex(ids)
                ids[idx]==0 && continue
                metal=wall===:west ? vol.mask[1,idx] : wall===:east ? vol.mask[nx,idx] :
                    wall===:south ? vol.mask[idx,1] : vol.mask[idx,ny]
                metal && connections[idx] || throw(ArgumentError(
                    "volume port $(ids[idx]) includes unconnected $wall edge $idx"))
            end
        end
        @inbounds for j in 1:ny, e in 0:nx
            left = e >= 1 ? vol.mask[e, j] : false
            right = e <= nx - 1 ? vol.mask[e + 1, j] : false
            if e == 0
                if right && vol.connect_west[j]
                    _push_x!(b, _BASIS_VX_LO, e, j, vl,
                             0.0, (j - 0.5) * grid.dy, grid.dy, west[j])
                end
            elseif e == nx
                if left && vol.connect_east[j]
                    _push_x!(b, _BASIS_VX_HI, e, j, vl,
                             grid.a, (j - 0.5) * grid.dy, grid.dy, east[j])
                end
            else
                if left && right
                    _push_x!(b, _BASIS_VX_FULL, e, j, vl,
                             e * grid.dx, (j - 0.5) * grid.dy, grid.dy, 0)
                end
            end
        end
        @inbounds for f in 0:ny, i in 1:nx
            below = f >= 1 ? vol.mask[i, f] : false
            above = f <= ny - 1 ? vol.mask[i, f + 1] : false
            if f == 0
                if above && vol.connect_south[i]
                    _push_y!(b, _BASIS_VY_LO, i, f, vl,
                             (i - 0.5) * grid.dx, 0.0, grid.dx, south[i])
                end
            elseif f == ny
                if below && vol.connect_north[i]
                    _push_y!(b, _BASIS_VY_HI, i, f, vl,
                             (i - 0.5) * grid.dx, grid.b, grid.dx, north[i])
                end
            else
                if below && above
                    _push_y!(b, _BASIS_VY_FULL, i, f, vl,
                             (i - 0.5) * grid.dx, f * grid.dy, grid.dx, 0)
                end
            end
        end
    end
    _assign_planar_internal_ports!(b, ports, grid)
    return b
end

function _append_planar_terminal_ports!(basis::PlanarBasisSet,
        ports::Vector{PlanarPort},sheets::Vector{SheetLevel},grid::CellGrid)
    for (pidx,p) in enumerate(ports)
        _is_planar_terminal(p.wall) || continue
        xaxis=_is_planar_x_terminal(p.wall)
        positive=_is_planar_positive_terminal(p.wall)
        sheet=sheets[p.level]
        kind=xaxis ? (positive ? _BASIS_X_LO : _BASIS_X_HI) :
            (positive ? _BASIS_Y_LO : _BASIS_Y_HI)
        for transverse in p.cells
            cell=p.edge+(positive ? 1 : 0)
            opposite=p.edge+(positive ? 0 : 1)
            i,j=xaxis ? (cell,transverse) : (transverse,cell)
            sheet.mask[i,j] || throw(ArgumentError(
                "terminal port $pidx includes edge without metal on its requested side"))
            if 1<=opposite<=(xaxis ? grid.nx : grid.ny)
                oi,oj=xaxis ? (opposite,transverse) : (transverse,opposite)
                sheet.mask[oi,oj] && throw(ArgumentError(
                    "terminal port $pidx must lie on an open metal edge; use an internal gap for continuous metal"))
            end
            ei,ej=xaxis ? (p.edge,transverse) : (transverse,p.edge)
            existing=findfirst(k -> basis.kind[k]==kind && basis.level[k]==p.level &&
                basis.ei[k]==ei && basis.ej[k]==ej,eachindex(basis.kind))
            if existing!==nothing
                basis.port[existing]==0 || throw(ArgumentError(
                    "terminal port $pidx overlaps port $(basis.port[existing])"))
                basis.port[existing]=pidx
            elseif xaxis
                _push_x!(basis,kind,ei,ej,p.level,p.edge*grid.dx,
                    (transverse-0.5)*grid.dy,grid.dy,pidx)
            else
                _push_y!(basis,kind,ei,ej,p.level,(transverse-0.5)*grid.dx,
                    p.edge*grid.dy,grid.dx,pidx)
            end
        end
    end
    return nothing
end

function _assign_planar_internal_ports!(basis::PlanarBasisSet,
        ports::Vector{PlanarPort}, grid::CellGrid)
    for (pidx, p) in enumerate(ports)
        p.wall in (:internal_x, :internal_y, :via) || continue
        claimed = Set{Int}()
        for b in eachindex(basis.kind)
            basis.level[b] == p.level || continue
            kind = basis.kind[b]
            cell = if p.wall === :internal_x && kind == _BASIS_X_FULL &&
                    basis.ei[b] == p.edge
                basis.ej[b]
            elseif p.wall === :internal_y && kind == _BASIS_Y_FULL &&
                    basis.ej[b] == p.edge
                basis.ei[b]
            elseif p.wall === :via && _is_via_kind(kind)
                basis.ei[b] + grid.nx * (basis.ej[b] - 1)
            else
                continue
            end
            cell in p.cells || continue
            basis.port[b] == 0 || throw(ArgumentError(
                "port $pidx overlaps port $(basis.port[b])"))
            basis.port[b] = pidx
            push!(claimed, cell)
        end
        length(claimed) == length(p.cells) || throw(ArgumentError(
            "port $pidx contains edges/cells without conductor basis functions"))
    end
    return nothing
end
