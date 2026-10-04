# Adapted from ASCENT.jl, copyright (c) 2026 jake-w-liu.
# MIT license: ASCENT_LIBRARY_LICENSE in this directory.
# -----------------------------------------------------------------------------
# Planar 2-D geometry: polygon/via records, allocation-free geometry kernels,
# and polygon normalization (dedup, simplicity check, CCW orientation).
# -----------------------------------------------------------------------------

export PlanarPolygon, PlanarVia, planar_normalize_polygon

const _P2 = SVector{2,Float64}

"""
    PlanarPolygon — a normalized metal polygon on interface `level` (0..L).
`vertices` are CCW, deduplicated, in SI metres; `bbox` = (xmin, ymin, xmax,
ymax). `net` is "" when undeclared.
"""
struct PlanarPolygon
    name::String
    level::Int
    metal::String
    net::String
    vertices::Vector{_P2}
    bbox::NTuple{4,Float64}
end

"""
    PlanarVia — a via instance from `from_level` to `to_level`
(`to_level = -1` encodes interface zero, the bottom cover). `vertices` is the
normalized CCW footprint polygon.
"""
struct PlanarVia
    name::String
    via_type::String
    from_level::Int
    to_level::Int
    vertices::Vector{_P2}
    bbox::NTuple{4,Float64}
end

# ── allocation-free kernels ──

# Signed polygon area (positive = CCW). Allocation-free.
function _p2_signed_area(v::Vector{_P2})
    n = length(v)
    a = 0.0
    n < 3 && return 0.0
    origin = v[1]
    @inbounds for i in 1:n
        p = v[i] - origin
        q = v[i == n ? 1 : i + 1] - origin
        a += p[1] * q[2] - q[1] * p[2]
    end
    return 0.5 * a
end

# 2-D cross product of (a - o) x (b - o).
_p2_cross(o::_P2, a::_P2, b::_P2) =
    (a[1] - o[1]) * (b[2] - o[2]) - (a[2] - o[2]) * (b[1] - o[1])

# Point p on segment a-b within perpendicular/endpoint tolerance `tol` (a
# length, in metres)? The cross product has units of length², so the
# collinearity test compares against tol * |b-a| and the parameter range
# against tol / |b-a|, keeping `tol` a scale-consistent distance.
function _p2_on_seg(p::_P2, a::_P2, b::_P2, tol::Float64)
    dx = b[1] - a[1]
    dy = b[2] - a[2]
    len2 = dx * dx + dy * dy
    len = sqrt(len2)
    len <= tol && return abs(p[1] - a[1]) <= tol && abs(p[2] - a[2]) <= tol
    abs(_p2_cross(a, b, p)) > tol * len && return false
    t = ((p[1] - a[1]) * dx + (p[2] - a[2]) * dy) / len2
    ttol = tol / len
    return -ttol <= t <= 1.0 + ttol
end

# Segment Euclidean length. Allocation-free.
_p2_seg_len(a::_P2, b::_P2) = sqrt((b[1] - a[1])^2 + (b[2] - a[2])^2)

# Segments a-b and c-d intersect or touch? Allocation-free. `tol` is a
# distance (metres); each cross product is compared against tol * |segment|.
function _p2_seg_intersect(a::_P2, b::_P2, c::_P2, d::_P2, tol::Float64)
    d1 = _p2_cross(c, d, a)
    d2 = _p2_cross(c, d, b)
    d3 = _p2_cross(a, b, c)
    d4 = _p2_cross(a, b, d)
    tcd = tol * max(_p2_seg_len(c, d), eps(Float64))
    tab = tol * max(_p2_seg_len(a, b), eps(Float64))
    if ((d1 > tcd && d2 < -tcd) || (d1 < -tcd && d2 > tcd)) &&
       ((d3 > tab && d4 < -tab) || (d3 < -tab && d4 > tab))
        return true                     # proper crossing
    end
    (abs(d1) <= tcd && _p2_on_seg(a, c, d, tol)) && return true
    (abs(d2) <= tcd && _p2_on_seg(b, c, d, tol)) && return true
    (abs(d3) <= tab && _p2_on_seg(c, a, b, tol)) && return true
    (abs(d4) <= tab && _p2_on_seg(d, a, b, tol)) && return true
    return false
end

# Point-in-polygon (ray cast); points on the boundary count as inside.
# Allocation-free.
function _p2_point_in_poly(p::_P2, v::Vector{_P2}, tol::Float64)
    n = length(v)
    inside = false
    @inbounds for i in 1:n
        a = v[i]
        b = v[i == n ? 1 : i + 1]
        _p2_on_seg(p, a, b, tol) && return true
        if (a[2] > p[2]) != (b[2] > p[2])
            x = a[1] + (p[2] - a[2]) * (b[1] - a[1]) / (b[2] - a[2])
            x > p[1] && (inside = !inside)
        end
    end
    return inside
end

# Axis-aligned bbox overlap (bboxes as (xmin, ymin, xmax, ymax)).
_p2_bbox_overlap(a::NTuple{4,Float64}, b::NTuple{4,Float64}, tol::Float64) =
    a[1] <= b[3] + tol && b[1] <= a[3] + tol &&
    a[2] <= b[4] + tol && b[2] <= a[4] + tol

_p2_bbox(v::Vector{_P2}) =
    (minimum(p -> p[1], v), minimum(p -> p[2], v),
     maximum(p -> p[1], v), maximum(p -> p[2], v))

function _p2_bbox_diag2(b::NTuple{4,Float64})
    dx = b[3] - b[1]
    dy = b[4] - b[2]
    return dx * dx + dy * dy
end

# ── polygon normalization ──

# Drop a closing duplicate and consecutive duplicate vertices (cyclic).
function _planar_dedup(raw::Vector{_P2}, tol::Float64)
    isempty(raw) && return raw
    pts = _P2[]
    @inbounds for p in raw
        if isempty(pts) ||
           abs(p[1] - pts[end][1]) > tol || abs(p[2] - pts[end][2]) > tol
            push!(pts, p)
        end
    end
    while length(pts) > 1 &&
          abs(pts[1][1] - pts[end][1]) <= tol &&
          abs(pts[1][2] - pts[end][2]) <= tol
        pop!(pts)
    end
    return pts
end

function _planar_scale(v::Vector{_P2})
    b = _p2_bbox(v)
    return sqrt(max(_p2_bbox_diag2(b), eps(Float64)))
end

"""
    planar_normalize_polygon(raw; label="polygon") -> (vertices, bbox)

Normalize a raw vertex list: drop a closing duplicate and consecutive
duplicates, require >= 3 vertices, nonzero area, a simple boundary (no
non-adjacent edge intersections or touches), and reorient to CCW. Returns
the normalized `Vector{SVector{2,Float64}}` and its bbox.
"""
function planar_normalize_polygon(raw::Vector{_P2}; label::AbstractString = "polygon")
    length(raw) >= 3 || throw(ArgumentError("$label requires at least three vertices"))
    all(p -> all(isfinite, p), raw) ||
        throw(ArgumentError("$label vertices must be finite"))
    scale = _planar_scale(raw)
    tol = 1e-12 * scale
    pts = _planar_dedup(raw, tol)
    n = length(pts)
    n >= 3 || throw(ArgumentError("$label requires at least 3 distinct vertices"))
    bbox = _p2_bbox(pts)
    area = _p2_signed_area(pts)
    abs(area) > 1e-12 * _p2_bbox_diag2(bbox) ||
        throw(ArgumentError("$label has (near-)zero area"))
    # Reject zero-angle spikes: consecutive edges that are collinear and
    # reverse direction (the adjacent-edge pairs the simplicity loop skips).
    @inbounds for i in 1:n
        p = pts[i]
        q = pts[i == n ? 1 : i + 1]
        r = pts[i == 1 ? n : i - 1]
        vx = p[1] - r[1]; vy = p[2] - r[2]
        ux = q[1] - p[1]; uy = q[2] - p[2]
        lv = sqrt(vx * vx + vy * vy)
        lu = sqrt(ux * ux + uy * uy)
        (lv <= tol || lu <= tol) && continue
        if abs(vx * uy - vy * ux) <= 1e-12 * lv * lu && vx * ux + vy * uy < 0
            throw(ArgumentError(
                "$label is self-intersecting (zero-angle spike at vertex $i)"))
        end
    end
    # Simplicity: no pair of non-adjacent edges may intersect or touch.
    etol = 1e-12 * scale
    @inbounds for i in 1:n
        a = pts[i]
        b = pts[i == n ? 1 : i + 1]
        for j in (i + 1):n
            # skip adjacent edge pairs (share a vertex)
            (j == i || j == i + 1 || (i == 1 && j == n)) && continue
            c = pts[j]
            d = pts[j == n ? 1 : j + 1]
            _p2_seg_intersect(a, b, c, d, etol) && throw(ArgumentError(
                "$label is self-intersecting (edges $i and $j cross or touch)"))
        end
    end
    area < 0 && reverse!(pts)
    return pts, _p2_bbox(pts)
end

planar_normalize_polygon(raw::AbstractVector; kw...) =
    planar_normalize_polygon(_P2[_P2(v[1], v[2]) for v in raw]; kw...)

function _planar_regular_ngon(center::_P2, diameter::Real, n::Integer)
    n >= 3 || throw(ArgumentError("polygon needs at least three sides"))
    return [center + (diameter / 2) * _P2(cospi(2k/n), sinpi(2k/n))
        for k in 0:n-1]
end
