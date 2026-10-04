# Adapted from ASCENT.jl, copyright (c) 2026 jake-w-liu.
# MIT license: ASCENT_LIBRARY_LICENSE in this directory.
# -----------------------------------------------------------------------------
# P0' — parametric planar layout library.
#
# Each primitive returns a `PlanarShape`: raw polygon / via footprints in SI
# metres plus named `PlanarPin`s (terminal edges: owning polygon, 1-based edge
# index in the emitted vertex order, midpoint, outward unit direction, width).
# Shapes are placed with `planar_transform` / chained with `planar_attach`,
# and lowered directly with `planar_layout_problem`, or serialized as
# SI layout tables with `planar_layout_dict` / `planar_pin_port_dict`.
#
# Every polygon is validated (simple, nonzero area) at construction, so an
# infeasible parameter set fails loudly here rather than at mesh time.
# Closed-form references (spiral inductance, parallel-plate capacitance) are
# provided for the solve-level acceptance gates.
# -----------------------------------------------------------------------------

export PlanarPin, PlanarShapePolygon, PlanarShapeVia, PlanarShape
export planar_line, planar_bend, planar_curved_bend, planar_taper, planar_step
export planar_tee, planar_cross, planar_coupled_lines, planar_broadside_coupled_lines
export planar_radial_stub, planar_spiral, planar_spiral_inductance
export planar_mim_capacitor, planar_parallel_plate_capacitance
export planar_via_array, planar_air_bridge, planar_pin, planar_transform, planar_attach
export planar_interdigital_capacitor
export planar_layout_dict, planar_pin_port_dict, planar_layout_bbox

"""
    PlanarPin — a connection terminal of a [`PlanarShape`](@ref): the edge
`edge` (1-based, emitted vertex order) of polygon `polygon` on `level`,
with midpoint `point`, outward unit `direction`, and edge length `width`.
"""
struct PlanarPin
    name::String
    polygon::String
    edge::Int
    level::Int
    point::_P2
    direction::_P2
    width::Float64
end

"""One emitted metal polygon of a [`PlanarShape`](@ref) (raw vertex order)."""
struct PlanarShapePolygon
    name::String
    level::Int
    metal::String
    net::String
    vertices::Vector{_P2}
end

"""One emitted via of a [`PlanarShape`](@ref); `to_level = -1` is "gnd"."""
struct PlanarShapeVia
    name::String
    via_type::String
    from_level::Int
    to_level::Int
    vertices::Vector{_P2}
end

"""
    PlanarShape — output of a library primitive: polygons, vias, pins, and
`meta` reference quantities (SI) used by acceptance tests, e.g.
`"centerline_length"`, `"area"`, `"d_in"`, `"overlap_area"`.
"""
struct PlanarShape
    name::String
    polygons::Vector{PlanarShapePolygon}
    vias::Vector{PlanarShapeVia}
    pins::Vector{PlanarPin}
    meta::Dict{String,Float64}
end

# ── small vector helpers ──

@inline _lib_norm(v::_P2) = sqrt(v[1] * v[1] + v[2] * v[2])
@inline _lib_perp(v::_P2) = _P2(-v[2], v[1])           # left normal
@inline _lib_rot(v::_P2, c::Float64, s::Float64) =
    _P2(c * v[1] - s * v[2], s * v[1] + c * v[2])

function _lib_unit(v::_P2, label::AbstractString)
    n = _lib_norm(v)
    n > 0 || throw(ArgumentError("$label: zero-length segment"))
    return v / n
end

function _lib_pos(x::Real, label::AbstractString)
    value=_circuit_stored_real(x,label)
    value>0 || throw(ArgumentError("$label must be finite and > 0 (got $x)"))
    return value
end
function _lib_nonneg(x::Real, label::AbstractString)
    value=_circuit_stored_real(x,label)
    value>=0 || throw(ArgumentError("$label must be finite and >= 0 (got $x)"))
    return value
end

# Range-safe evaluation of a positive reference product. Separate binary
# exponents avoid overflowing/underflowing intermediates with a finite result.
function _lib_reference_product(factors::Tuple,divisor::Float64,label::AbstractString)
    isfinite(divisor) && divisor!=0 || throw(ArgumentError("$label denominator must be finite and nonzero"))
    mantissa=1.;exponent=0
    for factor in factors
        part,power=frexp(factor)
        mantissa*=part;exponent+=power
        mantissa,power=frexp(mantissa);exponent+=power
    end
    part,power=frexp(abs(divisor))
    mantissa/=part;exponent-=power
    value=copysign(ldexp(mantissa,exponent),divisor)
    isfinite(value) && value!=0 || throw(ArgumentError("$label must remain finite and nonzero in Float64"))
    return value
end

function _lib_level(level::Integer, label::AbstractString)
    level >= 0 || throw(ArgumentError("$label must be >= 0 (got $level)"))
    return Int(level)
end

# Validate a raw polygon (simplicity, area) without altering emitted order.
function _lib_polygon(name::AbstractString, level::Int, metal::AbstractString,
                      net::AbstractString, v::Vector{_P2})
    isempty(metal) && throw(ArgumentError("$name: metal must be non-empty"))
    planar_normalize_polygon(v; label = String(name))
    return PlanarShapePolygon(String(name), level, String(metal), String(net), v)
end

# Pin on edge `e` of polygon `poly` whose outward side is `dir`.
function _lib_pin(name::AbstractString, poly::PlanarShapePolygon, e::Int,
                  dir::_P2)
    v = poly.vertices
    n = length(v)
    a = v[e]
    b = v[e == n ? 1 : e + 1]
    return PlanarPin(String(name), poly.name, e, poly.level, (a + b) / 2,
                     _lib_unit(dir, "pin $name direction"), _p2_seg_len(a, b))
end

"""
    _lib_ribbon(center, widths, label; t_start, t_end) -> (vertices, end_edge, start_edge)

Constant- or variable-width trace along an open polyline `center` with
miter joins: `vertices = [right side forward; left side backward]` (CCW for
a left-turning walk). Edge `n` is the end cap, edge `2n` the start cap. The
caps are perpendicular to the first/last chord unless explicit unit
tangents `t_start`/`t_end` are given (used where the polyline samples a
smooth curve, so a cap is square to the true exit direction). With chord
caps, a constant-width ribbon has area exactly `w * length(center)`.
Turns sharper than ~157° are rejected (the miter would spike).
"""
function _lib_ribbon(c::Vector{_P2}, w::Vector{Float64}, label::AbstractString;
                     t_start::Union{Nothing,_P2} = nothing,
                     t_end::Union{Nothing,_P2} = nothing)
    n = length(c)
    n >= 2 || throw(ArgumentError("$label: centerline needs >= 2 points"))
    length(w) == n || throw(ArgumentError("$label: one width per centerline point"))
    v = Vector{_P2}(undef, 2n)
    @inbounds for i in 1:n
        wi = w[i]
        (isfinite(wi) && wi > 0) ||
            throw(ArgumentError("$label: width must be > 0 at point $i"))
        h = 0.5 * wi
        if i == 1 || i == n
            t = i == 1 ? t_start : t_end
            d = t !== nothing ? _lib_unit(t, label) :
                i == 1 ? _lib_unit(c[2] - c[1], label) :
                         _lib_unit(c[n] - c[n - 1], label)
            off = h * _lib_perp(d)
        else
            n1 = _lib_perp(_lib_unit(c[i] - c[i - 1], label))
            n2 = _lib_perp(_lib_unit(c[i + 1] - c[i], label))
            m = n1 + n2
            mn = _lib_norm(m)
            mn > 1e-9 || throw(ArgumentError("$label: centerline reverses at point $i"))
            m = m / mn
            cm = m[1] * n1[1] + m[2] * n1[2]
            cm > 0.2 || throw(ArgumentError(
                "$label: turn at point $i is too sharp for a miter join"))
            off = (h / cm) * m
        end
        v[i] = c[i] - off
        v[2n + 1 - i] = c[i] + off
    end
    return v, n, 2n
end

function _lib_polyline_length(c::Vector{_P2})
    s = 0.0
    @inbounds for i in 2:length(c)
        s += _p2_seg_len(c[i - 1], c[i])
    end
    return s
end

# Two-pin ribbon shape (start pin "p1", end pin "p2"); optional exact end
# tangents (see `_lib_ribbon`).
function _lib_ribbon_shape(name, level, metal, net, c::Vector{_P2},
                           w::Vector{Float64}, meta::Dict{String,Float64};
                           t_start::Union{Nothing,_P2} = nothing,
                           t_end::Union{Nothing,_P2} = nothing)
    verts, eend, estart = _lib_ribbon(c, w, String(name); t_start, t_end)
    poly = _lib_polygon(name, level, metal, net, verts)
    d0 = t_start !== nothing ? -t_start : _lib_unit(c[1] - c[2], "$name start")
    d1 = t_end !== nothing ? t_end : _lib_unit(c[end] - c[end - 1], "$name end")
    pins = [_lib_pin("p1", poly, estart, d0), _lib_pin("p2", poly, eend, d1)]
    meta["centerline_length"] = _lib_polyline_length(c)
    meta["area"] = abs(_p2_signed_area(verts))
    return PlanarShape(String(name), [poly], PlanarShapeVia[], pins, meta)
end

# ── transmission-line primitives ──

"""
    planar_line(; length, width, level, metal, name="line", net="")

Straight trace from (0,0) to (length,0). Pins `p1` (x=0, facing -x) and `p2`
(x=length, facing +x).
"""
function planar_line(; length::Real, width::Real, level::Integer,
                     metal::AbstractString, name::AbstractString = "line",
                     net::AbstractString = "")
    L = _lib_pos(length, "planar_line length")
    w = _lib_pos(width, "planar_line width")
    return _lib_ribbon_shape(name, _lib_level(level, "level"), metal, net,
                             _P2[_P2(0, 0), _P2(L, 0)], [w, w],
                             Dict{String,Float64}())
end

"""
    planar_bend(; width, arm, miter=0.0, level, metal, name="bend", net="")

Right-angle (left-turning) bend: centerline (0,0) -> (arm,0) -> (arm,arm).
`miter` ∈ [0,1) is the chamfer as a fraction of the corner diagonal
`width*√2` (Douville–James convention; optimal ≈ 0.5–0.7). Pins `p1` (x=0,
facing -x) and `p2` (y=arm, facing +y). `meta["area"] = 2*arm*width - c²/2`
with chamfer leg `c = 2*miter*width`.
"""
function planar_bend(; width::Real, arm::Real, miter::Real = 0.0,
                     level::Integer, metal::AbstractString,
                     name::AbstractString = "bend", net::AbstractString = "")
    w = _lib_pos(width, "planar_bend width")
    a = _lib_pos(arm, "planar_bend arm")
    m = _lib_nonneg(miter, "planar_bend miter")
    m < 1 || throw(ArgumentError("planar_bend miter must be < 1"))
    a > w / 2 || throw(ArgumentError("planar_bend arm must exceed width/2"))
    h = w / 2
    c = 2 * m * w
    c < a + h || throw(ArgumentError("planar_bend miter cuts past the arm ends"))
    v = c > 0 ?
        _P2[_P2(0, -h), _P2(a + h - c, -h), _P2(a + h, -h + c), _P2(a + h, a),
            _P2(a - h, a), _P2(a - h, h), _P2(0, h)] :
        _P2[_P2(0, -h), _P2(a + h, -h), _P2(a + h, a),
            _P2(a - h, a), _P2(a - h, h), _P2(0, h)]
    poly = _lib_polygon(name, _lib_level(level, "level"), metal, net, v)
    nv = length(v)
    pins = [_lib_pin("p1", poly, nv, _P2(-1, 0)),
            _lib_pin("p2", poly, nv - 3, _P2(0, 1))]
    meta = Dict("area" => 2 * a * w - c^2 / 2, "centerline_length" => 2a)
    return PlanarShape(String(name), [poly], PlanarShapeVia[], pins, meta)
end

"""
    planar_curved_bend(; width, radius, angle=π/2, segments=16, level, metal,
                        name="curved_bend", net="")

Circular-arc (radial) bend of centerline `radius`, turning left by `angle`
(0 < angle ≤ π), starting at (0,0) heading +x; the arc is a `segments`-chord
polyline. Pins `p1`, `p2`.
"""
function planar_curved_bend(; width::Real, radius::Real, angle::Real = π / 2,
                            segments::Integer = 16, level::Integer,
                            metal::AbstractString,
                            name::AbstractString = "curved_bend",
                            net::AbstractString = "")
    w = _lib_pos(width, "planar_curved_bend width")
    R = _lib_pos(radius, "planar_curved_bend radius")
    θ = _lib_pos(angle, "planar_curved_bend angle")
    θ <= π || throw(ArgumentError("planar_curved_bend angle must be <= π"))
    R > w / 2 || throw(ArgumentError("planar_curved_bend radius must exceed width/2"))
    segments >= 1 || throw(ArgumentError("planar_curved_bend segments must be >= 1"))
    c = Vector{_P2}(undef, segments + 1)
    @inbounds for k in 0:segments
        φ = θ * k / segments
        c[k + 1] = _P2(R * sin(φ), R * (1 - cos(φ)))
    end
    meta = Dict("arc_area" => w * R * θ)        # exact annular-sector area
    return _lib_ribbon_shape(name, _lib_level(level, "level"), metal, net, c,
                             fill(w, segments + 1), meta;
                             t_start = _P2(1, 0), t_end = _P2(cos(θ), sin(θ)))
end

"""
    planar_taper(; width1, width2, length, profile=:linear, segments=16,
                  level, metal, name="taper", net="")

Width taper from `width1` at x=0 to `width2` at x=length. `profile` is
`:linear` or `:exponential` (w(x) = w1 (w2/w1)^(x/L), sampled at `segments`).
Pins `p1`, `p2`. `meta["area"]` is the exact polygon area.
"""
function planar_taper(; width1::Real, width2::Real, length::Real,
                      profile::Symbol = :linear, segments::Integer = 16,
                      level::Integer, metal::AbstractString,
                      name::AbstractString = "taper", net::AbstractString = "")
    w1 = _lib_pos(width1, "planar_taper width1")
    w2 = _lib_pos(width2, "planar_taper width2")
    L = _lib_pos(length, "planar_taper length")
    ns = profile === :linear ? 1 :
         profile === :exponential ? Int(segments) :
         throw(ArgumentError("planar_taper profile must be :linear or :exponential"))
    ns >= 1 || throw(ArgumentError("planar_taper segments must be >= 1"))
    c = [_P2(L * k / ns, 0) for k in 0:ns]
    w = profile === :linear ? [w1, w2] : [w1 * (w2 / w1)^(k / ns) for k in 0:ns]
    return _lib_ribbon_shape(name, _lib_level(level, "level"), metal, net, c, w,
                             Dict{String,Float64}())
end

"""
    planar_step(; width1, width2, length1, length2, level, metal,
                 name="step", net="")

Width step: `width1` over x∈[0,length1], `width2` over [length1,
length1+length2], both centred on y=0. Pins `p1`, `p2`.
"""
function planar_step(; width1::Real, width2::Real, length1::Real, length2::Real,
                     level::Integer, metal::AbstractString,
                     name::AbstractString = "step", net::AbstractString = "")
    w1 = _lib_pos(width1, "planar_step width1")
    w2 = _lib_pos(width2, "planar_step width2")
    L1 = _lib_pos(length1, "planar_step length1")
    L2 = _lib_pos(length2, "planar_step length2")
    w1 != w2 || throw(ArgumentError("planar_step widths are equal; use planar_line"))
    h1, h2 = w1 / 2, w2 / 2
    v = _P2[_P2(0, -h1), _P2(L1, -h1), _P2(L1, -h2), _P2(L1 + L2, -h2),
            _P2(L1 + L2, h2), _P2(L1, h2), _P2(L1, h1), _P2(0, h1)]
    poly = _lib_polygon(name, _lib_level(level, "level"), metal, net, v)
    pins = [_lib_pin("p1", poly, 8, _P2(-1, 0)), _lib_pin("p2", poly, 4, _P2(1, 0))]
    meta = Dict("area" => w1 * L1 + w2 * L2)
    return PlanarShape(String(name), [poly], PlanarShapeVia[], pins, meta)
end

"""
    planar_tee(; width_main, width_branch, length_main, length_branch,
                level, metal, name="tee", net="")

T-junction: main trace x∈[0,length_main] (width `width_main`, centred on
y=0) and a branch of width `width_branch` rising from the main's top edge at
x = length_main/2 by `length_branch`. Pins `p1` (left), `p2` (right),
`p3` (branch end, facing +y).
"""
function planar_tee(; width_main::Real, width_branch::Real, length_main::Real,
                    length_branch::Real, level::Integer, metal::AbstractString,
                    name::AbstractString = "tee", net::AbstractString = "")
    w1 = _lib_pos(width_main, "planar_tee width_main")
    w2 = _lib_pos(width_branch, "planar_tee width_branch")
    Lm = _lib_pos(length_main, "planar_tee length_main")
    Lb = _lib_pos(length_branch, "planar_tee length_branch")
    w2 < Lm || throw(ArgumentError("planar_tee width_branch must be < length_main"))
    h1, h2, xc = w1 / 2, w2 / 2, Lm / 2
    v = _P2[_P2(0, -h1), _P2(Lm, -h1), _P2(Lm, h1), _P2(xc + h2, h1),
            _P2(xc + h2, h1 + Lb), _P2(xc - h2, h1 + Lb), _P2(xc - h2, h1),
            _P2(0, h1)]
    poly = _lib_polygon(name, _lib_level(level, "level"), metal, net, v)
    pins = [_lib_pin("p1", poly, 8, _P2(-1, 0)), _lib_pin("p2", poly, 2, _P2(1, 0)),
            _lib_pin("p3", poly, 5, _P2(0, 1))]
    meta = Dict("area" => w1 * Lm + w2 * Lb)
    return PlanarShape(String(name), [poly], PlanarShapeVia[], pins, meta)
end

"""
    planar_cross(; width_x, width_y, arm_x, arm_y, level, metal,
                  name="cross", net="")

Cross junction centred at the origin: an x-directed trace of width `width_x`
spanning ±`arm_x` and a y-directed trace of width `width_y` spanning
±`arm_y`. Pins `p1` (-x), `p2` (+x), `p3` (+y), `p4` (-y).
"""
function planar_cross(; width_x::Real, width_y::Real, arm_x::Real, arm_y::Real,
                      level::Integer, metal::AbstractString,
                      name::AbstractString = "cross", net::AbstractString = "")
    wx = _lib_pos(width_x, "planar_cross width_x")
    wy = _lib_pos(width_y, "planar_cross width_y")
    ax = _lib_pos(arm_x, "planar_cross arm_x")
    ay = _lib_pos(arm_y, "planar_cross arm_y")
    hx, hy = wx / 2, wy / 2
    ax > hy && ay > hx ||
        throw(ArgumentError("planar_cross arms must extend past the crossing trace"))
    v = _P2[_P2(-ax, -hx), _P2(-hy, -hx), _P2(-hy, -ay), _P2(hy, -ay),
            _P2(hy, -hx), _P2(ax, -hx), _P2(ax, hx), _P2(hy, hx),
            _P2(hy, ay), _P2(-hy, ay), _P2(-hy, hx), _P2(-ax, hx)]
    poly = _lib_polygon(name, _lib_level(level, "level"), metal, net, v)
    pins = [_lib_pin("p1", poly, 12, _P2(-1, 0)), _lib_pin("p2", poly, 6, _P2(1, 0)),
            _lib_pin("p3", poly, 9, _P2(0, 1)), _lib_pin("p4", poly, 3, _P2(0, -1))]
    meta = Dict("area" => 2ax * wx + 2ay * wy - wx * wy)
    return PlanarShape(String(name), [poly], PlanarShapeVia[], pins, meta)
end

"""
    planar_coupled_lines(; length, width, gap, width2=width, level, metal,
                          name="coupled", net1="", net2="")

Edge-coupled pair: line A (y<0) and line B (y>0) of length `length`
separated by `gap` (edge to edge). Pins `p1`,`p2` (A: x=0, x=L) and
`p3`,`p4` (B: x=0, x=L).
"""
function planar_coupled_lines(; length::Real, width::Real, gap::Real,
                              width2::Real = width, level::Integer,
                              metal::AbstractString,
                              name::AbstractString = "coupled",
                              net1::AbstractString = "", net2::AbstractString = "")
    L = _lib_pos(length, "planar_coupled_lines length")
    wa = _lib_pos(width, "planar_coupled_lines width")
    wb = _lib_pos(width2, "planar_coupled_lines width2")
    g = _lib_pos(gap, "planar_coupled_lines gap")
    lev = _lib_level(level, "level")
    ya = -(g + wa) / 2
    yb = (g + wb) / 2
    return _lib_coupled_pair(String(name), L, wa, wb, ya, yb, lev, lev, metal,
                             metal, net1, net2, Dict("gap" => g))
end

"""
    planar_broadside_coupled_lines(; length, width, upper_level, lower_level,
                                    offset=0.0, width2=width, metal,
                                    metal2=metal, name="broadside", ...)

Broadside pair: line A on `upper_level`, line B on `lower_level` (larger
index = lower in the stackup), laterally offset by `offset` (centre to
centre, +y for B). Pins as [`planar_coupled_lines`](@ref).
"""
function planar_broadside_coupled_lines(; length::Real, width::Real,
                                        upper_level::Integer, lower_level::Integer,
                                        offset::Real = 0.0, width2::Real = width,
                                        metal::AbstractString,
                                        metal2::AbstractString = metal,
                                        name::AbstractString = "broadside",
                                        net1::AbstractString = "",
                                        net2::AbstractString = "")
    L = _lib_pos(length, "planar_broadside_coupled_lines length")
    wa = _lib_pos(width, "planar_broadside_coupled_lines width")
    wb = _lib_pos(width2, "planar_broadside_coupled_lines width2")
    isfinite(offset) || throw(ArgumentError("offset must be finite"))
    lu = _lib_level(upper_level, "upper_level")
    ll = _lib_level(lower_level, "lower_level")
    lu > ll || throw(ArgumentError(
        "upper_level must be above (larger interface index than) lower_level"))
    return _lib_coupled_pair(String(name), L, wa, wb, 0.0, Float64(offset), lu,
                             ll, metal, metal2, net1, net2,
                             Dict("offset" => Float64(offset)))
end

function _lib_coupled_pair(name, L, wa, wb, ya, yb, la, lb, ma, mb, na, nb, meta)
    va, ea, sa = _lib_ribbon(_P2[_P2(0, ya), _P2(L, ya)], [wa, wa], name)
    vb, eb, sb = _lib_ribbon(_P2[_P2(0, yb), _P2(L, yb)], [wb, wb], name)
    pa = _lib_polygon("$(name)_a", la, ma, na, va)
    pb = _lib_polygon("$(name)_b", lb, mb, nb, vb)
    pins = [_lib_pin("p1", pa, sa, _P2(-1, 0)), _lib_pin("p2", pa, ea, _P2(1, 0)),
            _lib_pin("p3", pb, sb, _P2(-1, 0)), _lib_pin("p4", pb, eb, _P2(1, 0))]
    return PlanarShape(name, [pa, pb], PlanarShapeVia[], pins, meta)
end

"""
    planar_radial_stub(; radius, angle, feed_width, segments=24, level, metal,
                        name="radial_stub", net="")

Radial (butterfly-half) stub: a circular sector of outer `radius` and
opening `angle` (< π) opening toward +x, truncated at its apex by the feed
chord of length `feed_width`. The feed pin `p1` sits at the origin facing
-x. `meta["area"]` is the exact polygon area.
"""
function planar_radial_stub(; radius::Real, angle::Real, feed_width::Real,
                            segments::Integer = 24, level::Integer,
                            metal::AbstractString,
                            name::AbstractString = "radial_stub",
                            net::AbstractString = "")
    R = _lib_pos(radius, "planar_radial_stub radius")
    θ = _lib_pos(angle, "planar_radial_stub angle")
    w = _lib_pos(feed_width, "planar_radial_stub feed_width")
    # Float64(π) < π holds in Julia (the float rounds below π), so compare
    # with an explicit margin: a 180° "stub" has no apex.
    θ < Float64(π) - 1e-9 ||
        throw(ArgumentError("planar_radial_stub angle must be < π"))
    segments >= 2 || throw(ArgumentError("planar_radial_stub segments must be >= 2"))
    α = θ / 2
    r0 = (w / 2) / sin(α)
    R > r0 || throw(ArgumentError("planar_radial_stub radius must exceed the feed apex distance"))
    x0 = r0 * cos(α)
    v = Vector{_P2}(undef, segments + 3)
    v[1] = _P2(r0 * cos(-α) - x0, r0 * sin(-α))
    @inbounds for k in 0:segments
        φ = -α + θ * k / segments
        v[k + 2] = _P2(R * cos(φ) - x0, R * sin(φ))
    end
    v[end] = _P2(r0 * cos(α) - x0, r0 * sin(α))
    poly = _lib_polygon(name, _lib_level(level, "level"), metal, net, v)
    pins = [_lib_pin("p1", poly, length(v), _P2(-1, 0))]
    # "area": exact area of the emitted (chorded) polygon; "ideal_area": the
    # true stub (circular arc, straight feed chord).
    meta = Dict("area" => 0.5 * R^2 * segments * sin(θ / segments) -
                          0.5 * r0^2 * sin(θ),
                "ideal_area" => 0.5 * θ * R^2 - 0.5 * r0^2 * sin(θ))
    return PlanarShape(String(name), [poly], PlanarShapeVia[], pins, meta)
end

# ── spiral inductors ──

"""
    planar_spiral(; shape=:rectangular, turns, width, spacing, d_out,
                   segments_per_turn=64, level, metal, name="spiral", net="")

Single-layer spiral inductor trace (outer end `p1`, inner end `p2`) wound
inward and counter-clockwise from the outer edge. `shape` ∈ `:rectangular`
/ `:hexagonal` / `:octagonal` (turns in multiples of 1/4, 1/6, 1/8; adjacent
turns are exactly `spacing` apart) or `:circular` (Archimedean,
`segments_per_turn` chords; the gap falls short of `spacing` by the chord
sagitta ≈ r·π²/(2m²)). `d_out` is the outer-edge diameter (across flats for
the polygonal shapes). The inner end needs an underpass/air bridge to reach
the outside. `meta` carries the Mohan-convention `d_out`,
`d_in = d_out - 2n·w - 2(n-1)·s`, `turns`, and `centerline_length`. See
[`planar_spiral_inductance`](@ref).
"""
function planar_spiral(; shape::Symbol = :rectangular, turns::Real, width::Real,
                       spacing::Real, d_out::Real, segments_per_turn::Integer = 64,
                       level::Integer, metal::AbstractString,
                       name::AbstractString = "spiral", net::AbstractString = "")
    n = _lib_pos(turns, "planar_spiral turns")
    w = _lib_pos(width, "planar_spiral width")
    s = _lib_pos(spacing, "planar_spiral spacing")
    D = _lib_pos(d_out, "planar_spiral d_out")
    p = w + s
    d_in = D - 2n * w - 2(n - 1) * s
    d_in > 0 || throw(ArgumentError(
        "planar_spiral: $n turns of pitch $p do not fit in d_out $D (d_in = $d_in)"))
    c = if shape === :rectangular
        _lib_spiral_poly(n, w, p, D, 4, "rectangular spiral")
    elseif shape === :hexagonal
        _lib_spiral_poly(n, w, p, D, 6, "hexagonal spiral")
    elseif shape === :octagonal
        _lib_spiral_poly(n, w, p, D, 8, "octagonal spiral")
    elseif shape === :circular
        segments_per_turn >= 8 ||
            throw(ArgumentError("planar_spiral segments_per_turn must be >= 8"))
        _lib_spiral_circ(n, w, p, D, Int(segments_per_turn))
    else
        throw(ArgumentError("planar_spiral shape must be :rectangular, :hexagonal, :octagonal, or :circular"))
    end
    meta = Dict("turns" => n, "d_out" => D, "d_in" => d_in, "width" => w,
                "spacing" => s)
    return _lib_ribbon_shape(name, _lib_level(level, "level"), metal, net, c,
                             fill(w, length(c)), meta)
end

function _lib_quantize_turns(n::Float64, per_turn::Int, label::AbstractString)
    k = round(Int, n * per_turn)
    abs(k - n * per_turn) <= 1e-9 * max(1.0, n * per_turn) || throw(ArgumentError(
        "$label turns must be a multiple of 1/$per_turn (got $n)"))
    k >= 2 || throw(ArgumentError("$label needs at least 2/$per_turn turns"))
    return k
end

# Inward M-sided polygonal spiral (M = 4 rectangular, 6 hexagonal, 8
# octagonal). Edge k (1-based) lies on the line x·N_k = A_k with outward
# normal N_k at angle -π/2 + (k-1)·2π/M and centreline apothem
# A_k = a0 - p·floor((k-1)/M): edges one turn apart are parallel and exactly
# one pitch apart, so the edge-to-edge gap is exactly `spacing`. Corners are
# intersections of consecutive edge lines; the start/end points use the
# virtual edges k = 0 and k = nseg+1. For M = 4 this is the classic
# R a, U a, L a, D a-p, R a-p, ... square spiral.
function _lib_spiral_poly(n, w, p, D, M::Int, label::AbstractString)
    nseg = _lib_quantize_turns(n, M, label)
    a0 = D / 2 - w / 2
    # virtual start edge k = 0 belongs to the outer ring (apothem a0)
    line(k) = (θ = -π / 2 + (k - 1) * 2π / M;
               (_P2(cos(θ), sin(θ)), a0 - p * floor(max(k - 1, 0) / M)))
    function corner(i, j)
        (ni, ai), (nj, aj) = line(i), line(j)
        det = ni[1] * nj[2] - ni[2] * nj[1]
        return _P2((ai * nj[2] - aj * ni[2]) / det, (ni[1] * aj - nj[1] * ai) / det)
    end
    c = Vector{_P2}(undef, nseg + 1)
    c[1] = corner(0, 1)
    @inbounds for k in 1:nseg
        c[k + 1] = corner(k, k + 1)
        Lk = _p2_seg_len(c[k], c[k + 1])
        Lk > w || throw(ArgumentError(
            "$label: segment $k length $Lk collapses (too many turns for d_out)"))
    end
    return c
end

# Archimedean spiral r(φ) = r0 - p φ / 2π.
function _lib_spiral_circ(n, w, p, D, m)
    nseg = max(2, round(Int, n * m))
    r0 = D / 2 - w / 2
    c = Vector{_P2}(undef, nseg + 1)
    @inbounds for k in 0:nseg
        φ = 2π * n * k / nseg
        r = r0 - p * φ / (2π)
        r > w / 2 || throw(ArgumentError("circular spiral collapses (turns too many)"))
        c[k + 1] = r * _P2(sin(φ), -cos(φ))
    end
    return c
end

const _MOHAN_WHEELER = Dict(:rectangular => (2.34, 2.75),
                            :hexagonal => (2.33, 3.82),
                            :octagonal => (2.25, 3.55))
const _MOHAN_CURRENT_SHEET = Dict(:rectangular => (1.27, 2.07, 0.18, 0.13),
                                  :hexagonal => (1.09, 2.23, 0.00, 0.17),
                                  :octagonal => (1.07, 2.29, 0.00, 0.19),
                                  :circular => (1.00, 2.46, 0.00, 0.20))

"""
    planar_spiral_inductance(; shape, turns, d_out, d_in, method=:wheeler) -> H

Closed-form planar spiral inductance (Mohan, Hershenson, Boyd & Lee, IEEE
JSSC 34(10), 1999). `method = :wheeler` (modified Wheeler,
`K1 μ0 n² d_avg / (1 + K2 ρ)`; rectangular/hexagonal/octagonal) or
`:current_sheet` (`μ0 n² d_avg c1/2 (ln(c2/ρ) + c3 ρ + c4 ρ²)`; also
circular). `d_avg = (d_out+d_in)/2`, fill ratio `ρ = (d_out-d_in)/(d_out+d_in)`.
Typical accuracy is a few percent against field solvers.
"""
function planar_spiral_inductance(; shape::Symbol, turns::Real, d_out::Real,
                                  d_in::Real, method::Symbol = :wheeler)
    n = _lib_pos(turns, "turns")
    Do = _lib_pos(d_out, "d_out")
    Di = _lib_pos(d_in, "d_in")
    Di < Do || throw(ArgumentError("d_in must be < d_out"))
    diameter_ratio=Di/Do
    mean_factor=(1+diameter_ratio)/2
    ρ=((Do-Di)/Do)/(1+diameter_ratio)
    if method === :wheeler
        haskey(_MOHAN_WHEELER, shape) || throw(ArgumentError(
            "modified Wheeler is defined for rectangular/hexagonal/octagonal, not $shape"))
        K1, K2 = _MOHAN_WHEELER[shape]
        denominator=1+K2*ρ
        return _lib_reference_product((K1,_MU0,n,n,Do,mean_factor),denominator,"spiral inductance")
    elseif method === :current_sheet
        haskey(_MOHAN_CURRENT_SHEET, shape) ||
            throw(ArgumentError("current-sheet formula has no coefficients for $shape"))
        c1, c2, c3, c4 = _MOHAN_CURRENT_SHEET[shape]
        correction=log(c2/ρ)+c3*ρ+c4*ρ^2
        return _lib_reference_product((_MU0,n,n,Do,mean_factor,c1,correction),2.,"spiral inductance")
    end
    throw(ArgumentError("method must be :wheeler or :current_sheet"))
end

# ── multi-level primitives ──

"""
    planar_mim_capacitor(; width, length, upper_level, lower_level,
                          upper_metal, lower_metal=upper_metal, overhang,
                          lead_width, lead_length, name="mim", net_top="",
                          net_bottom="")

MIM / overlay capacitor centred at the origin: top plate `length`×`width`
on `upper_level` with a lead toward -x (pin `p1`), bottom plate enlarged by
`overhang` on every side on `lower_level` with a lead toward +x (pin `p2`).
`meta["overlap_area"]` = plate overlap + the top lead's overlap with the
bottom plate. See [`planar_parallel_plate_capacitance`](@ref).
"""
function planar_mim_capacitor(; width::Real, length::Real, upper_level::Integer,
                              lower_level::Integer, upper_metal::AbstractString,
                              lower_metal::AbstractString = upper_metal,
                              overhang::Real, lead_width::Real, lead_length::Real,
                              name::AbstractString = "mim",
                              net_top::AbstractString = "",
                              net_bottom::AbstractString = "")
    W = _lib_pos(width, "planar_mim_capacitor width")
    L = _lib_pos(length, "planar_mim_capacitor length")
    o = _lib_pos(overhang, "planar_mim_capacitor overhang")
    lw = _lib_pos(lead_width, "planar_mim_capacitor lead_width")
    ll = _lib_pos(lead_length, "planar_mim_capacitor lead_length")
    lu = _lib_level(upper_level, "upper_level")
    lo = _lib_level(lower_level, "lower_level")
    lu > lo || throw(ArgumentError(
        "upper_level must be above (larger interface index than) lower_level"))
    lw < W || throw(ArgumentError("lead_width must be < plate width"))
    ll > o || throw(ArgumentError("lead_length must exceed overhang (lead must clear the bottom plate)"))
    hx, hy, hl = L / 2, W / 2, lw / 2
    top = _P2[_P2(-hx, -hy), _P2(hx, -hy), _P2(hx, hy), _P2(-hx, hy),
              _P2(-hx, hl), _P2(-hx - ll, hl), _P2(-hx - ll, -hl), _P2(-hx, -hl)]
    bx, by = hx + o, hy + o
    bot = _P2[_P2(-bx, -by), _P2(bx, -by), _P2(bx, -hl), _P2(bx + ll, -hl),
              _P2(bx + ll, hl), _P2(bx, hl), _P2(bx, by), _P2(-bx, by)]
    pt = _lib_polygon("$(name)_top", lu, upper_metal, net_top, top)
    pb = _lib_polygon("$(name)_bottom", lo, lower_metal, net_bottom, bot)
    pins = [_lib_pin("p1", pt, 6, _P2(-1, 0)), _lib_pin("p2", pb, 4, _P2(1, 0))]
    meta = Dict("overlap_area" => W * L + lw * o, "plate_area" => W * L)
    return PlanarShape(String(name), [pt, pb], PlanarShapeVia[], pins, meta)
end

"""
    planar_parallel_plate_capacitance(stackup, upper_level, lower_level, area;
                                       f_hz=1e9) -> F

Parallel-plate capacitance of `area` between metal interfaces `upper_level` >
`lower_level`, using the series combination of axial dielectric constants:
`C = ε0 A / Σ (t_i / ε_zi)`. No fringing.
"""
function planar_parallel_plate_capacitance(s::PlanarStackup, upper_level::Integer,
                                           lower_level::Integer, area::Real;
                                           f_hz::Real = 1e9)
    A = _lib_pos(area, "area")
    planar_validate(s)
    n = length(s.layers)
    0 <= lower_level < upper_level <= n || throw(ArgumentError(
        "need 0 <= lower_level < upper_level <= $n"))
    acc = 0.0
    for i in (lower_level + 1):upper_level
        layer = s.layers[i]
        acc += real(layer.thickness) / real(layer.epsr_z)
    end
    return _lib_reference_product((_EPS0,A),Float64(acc),"parallel-plate capacitance")
end

"""
    planar_via_array(; nx, ny, pitch_x, pitch_y, via_size, from_level,
                      to_level, via_type, footprint=:square, pad_metal="",
                      pad_margin=0.0, name="via_array", net="")

`nx`×`ny` via grid centred at the origin. `to_level = -1` targets the
bottom cover ("gnd"). `footprint` is `:square` (side `via_size`) or
`:circle` (16-gon, diameter `via_size`). A non-empty `pad_metal` adds a
landing pad on `from_level` covering the array plus `pad_margin`, and on
`to_level` when that is not "gnd"; the pads carry pins `p1` (-x edge of the
upper pad) and, for non-gnd arrays, `p2` (+x edge of the lower pad).
"""
function planar_via_array(; nx::Integer, ny::Integer, pitch_x::Real = 0.0,
                          pitch_y::Real = 0.0, via_size::Real,
                          from_level::Integer, to_level::Integer,
                          via_type::AbstractString, footprint::Symbol = :square,
                          pad_metal::AbstractString = "", pad_margin::Real = 0.0,
                          name::AbstractString = "via_array",
                          net::AbstractString = "")
    nx >= 1 && ny >= 1 || throw(ArgumentError("planar_via_array nx, ny must be >= 1"))
    d = _lib_pos(via_size, "planar_via_array via_size")
    px = nx > 1 ? _lib_pos(pitch_x, "planar_via_array pitch_x") : 0.0
    py = ny > 1 ? _lib_pos(pitch_y, "planar_via_array pitch_y") : 0.0
    (nx == 1 || px > d) && (ny == 1 || py > d) ||
        throw(ArgumentError("planar_via_array pitch must exceed via_size (vias would merge)"))
    fl = _lib_level(from_level, "from_level")
    to_level == -1 || to_level >= 0 && to_level != fl ||
        throw(ArgumentError("to_level must be a different interface or -1 (gnd)"))
    isempty(via_type) && throw(ArgumentError("via_type must be non-empty"))
    footprint in (:square, :circle) ||
        throw(ArgumentError("footprint must be :square or :circle"))
    m = _lib_nonneg(pad_margin, "pad_margin")
    vias = PlanarShapeVia[]
    sizehint!(vias, nx * ny)
    x0 = -px * (nx - 1) / 2
    y0 = -py * (ny - 1) / 2
    for j in 1:ny, i in 1:nx
        ctr = _P2(x0 + px * (i - 1), y0 + py * (j - 1))
        fp = footprint === :circle ? _planar_regular_ngon(ctr, d, 16) :
            _P2[ctr + _P2(-d / 2, -d / 2), ctr + _P2(d / 2, -d / 2),
                ctr + _P2(d / 2, d / 2), ctr + _P2(-d / 2, d / 2)]
        planar_normalize_polygon(fp; label = "$(name) via ($i,$j)")
        push!(vias, PlanarShapeVia("$(name)_v$(i)_$(j)", String(via_type), fl,
                                   Int(to_level), fp))
    end
    polys = PlanarShapePolygon[]
    pins = PlanarPin[]
    if !isempty(pad_metal)
        hx = px * (nx - 1) / 2 + d / 2 + m
        hy = py * (ny - 1) / 2 + d / 2 + m
        rect = _P2[_P2(-hx, -hy), _P2(hx, -hy), _P2(hx, hy), _P2(-hx, hy)]
        up = _lib_polygon("$(name)_pad_top", fl, pad_metal, net, rect)
        push!(polys, up)
        push!(pins, _lib_pin("p1", up, 4, _P2(-1, 0)))
        if to_level != -1
            lo = _lib_polygon("$(name)_pad_bottom", Int(to_level), pad_metal, net,
                              copy(rect))
            push!(polys, lo)
            push!(pins, _lib_pin("p2", lo, 2, _P2(1, 0)))
        end
    end
    meta = Dict("count" => Float64(nx * ny))
    return PlanarShape(String(name), polys, vias, pins, meta)
end

"""
    planar_air_bridge(; span, width, landing, bridge_level, base_level,
                       via_type, metal, bridge_metal=metal, via_margin=0.0,
                       name="bridge", net="")

Air bridge / crossover: a `width`-wide strip on `bridge_level` spanning
`span` between two `landing`-long pads on `base_level` (below it), joined by
one via per pad (pad footprint shrunk by `via_margin`). Centred at the
origin along x; pins `p1` (outer -x edge of the left pad) and `p2` (outer +x
edge of the right pad) on `base_level`.
"""
function planar_air_bridge(; span::Real, width::Real, landing::Real,
                           bridge_level::Integer, base_level::Integer,
                           via_type::AbstractString, metal::AbstractString,
                           bridge_metal::AbstractString = metal,
                           via_margin::Real = 0.0,
                           name::AbstractString = "bridge",
                           net::AbstractString = "")
    S = _lib_pos(span, "planar_air_bridge span")
    w = _lib_pos(width, "planar_air_bridge width")
    lg = _lib_pos(landing, "planar_air_bridge landing")
    vm = _lib_nonneg(via_margin, "planar_air_bridge via_margin")
    lb = _lib_level(bridge_level, "bridge_level")
    lv = _lib_level(base_level, "base_level")
    lb > lv || throw(ArgumentError(
        "bridge_level must be above (larger interface index than) base_level"))
    2vm < min(lg, w) || throw(ArgumentError("via_margin leaves no via footprint"))
    isempty(via_type) && throw(ArgumentError("via_type must be non-empty"))
    h, a, b = w / 2, S / 2, S / 2 + lg
    rect(x0, x1, y0, y1) = _P2[_P2(x0, y0), _P2(x1, y0), _P2(x1, y1), _P2(x0, y1)]
    strip = _lib_polygon("$(name)_strip", lb, bridge_metal, net, rect(-b, b, -h, h))
    padl = _lib_polygon("$(name)_pad1", lv, metal, net, rect(-b, -a, -h, h))
    padr = _lib_polygon("$(name)_pad2", lv, metal, net, rect(a, b, -h, h))
    vias = [PlanarShapeVia("$(name)_via1", String(via_type), lb, lv,
                           rect(-b + vm, -a - vm, -h + vm, h - vm)),
            PlanarShapeVia("$(name)_via2", String(via_type), lb, lv,
                           rect(a + vm, b - vm, -h + vm, h - vm))]
    pins = [_lib_pin("p1", padl, 4, _P2(-1, 0)), _lib_pin("p2", padr, 2, _P2(1, 0))]
    meta = Dict("span" => S, "strip_length" => 2b)
    return PlanarShape(String(name), [strip, padl, padr], vias, pins, meta)
end

# ── placement ──

"""    planar_pin(shape, name) -> PlanarPin"""
function planar_pin(sh::PlanarShape, name::AbstractString)
    for p in sh.pins
        p.name == name && return p
    end
    throw(ArgumentError("shape `$(sh.name)` has no pin `$name` " *
                        "(pins: $(join((p.name for p in sh.pins), ", ")))"))
end

_lib_rename(s::String, old::String, new::String) =
    startswith(s, old) ? new * s[(ncodeunits(old) + 1):end] : s

"""
    planar_transform(shape; offset=(0,0), angle=0.0, mirror=false, name)

Rigid placement: optional mirror y -> -y, then rotation by `angle` (rad,
CCW) about the origin, then translation by `offset` (m). Vertex order (and
so every pin's edge index) is preserved. `name` renames the shape and the
shape-name prefix of its polygons/vias.
"""
function planar_transform(sh::PlanarShape; offset = (0.0, 0.0), angle::Real = 0.0,
                          mirror::Bool = false, name::AbstractString = sh.name)
    length(offset) == 2 && all(v->v isa Real && isfinite(v), offset) ||
        throw(ArgumentError("offset must be a finite 2-tuple"))
    stored_angle=_circuit_stored_real(angle,"placement angle")
    c, s = cos(stored_angle), sin(stored_angle)
    t = _P2(_circuit_stored_real(offset[1],"placement x offset"),
        _circuit_stored_real(offset[2],"placement y offset"))
    mp(v::_P2) = mirror ? _P2(v[1], -v[2]) : v
    function xf(v::_P2)
        point=_lib_rot(mp(v),c,s)+t
        all(isfinite,point) || throw(ArgumentError("placement produces nonfinite coordinates"))
        return point
    end
    function placed_vertices(vertices,label)
        placed=map(xf,vertices)
        planar_normalize_polygon(placed;label=label)
        return placed
    end
    xd(v::_P2) = _lib_rot(mp(v), c, s)
    old, new = sh.name, String(name)
    polys = [PlanarShapePolygon(_lib_rename(p.name, old, new), p.level, p.metal,
                                p.net, placed_vertices(p.vertices,p.name)) for p in sh.polygons]
    vias = [PlanarShapeVia(_lib_rename(v.name, old, new), v.via_type, v.from_level,
                           v.to_level, placed_vertices(v.vertices,v.name)) for v in sh.vias]
    pins = [PlanarPin(p.name, _lib_rename(p.polygon, old, new), p.edge, p.level,
                      xf(p.point), xd(p.direction), p.width) for p in sh.pins]
    return PlanarShape(new, polys, vias, pins, copy(sh.meta))
end

"""
    planar_attach(shape, pin_name, target::PlanarPin; name=shape.name)

Place `shape` so its pin `pin_name` meets `target`: the pin's outward
direction is turned to oppose `target.direction` and its midpoint moved onto
`target.point`. The pins must be on the same level.
"""
function planar_attach(sh::PlanarShape, pin_name::AbstractString,
                       target::PlanarPin; name::AbstractString = sh.name)
    pin = planar_pin(sh, pin_name)
    pin.level == target.level || throw(ArgumentError(
        "cannot attach pin on level $(pin.level) to a pin on level $(target.level)"))
    ang = atan(-target.direction[2], -target.direction[1]) -
          atan(pin.direction[2], pin.direction[1])
    rotated = planar_transform(sh; angle = ang, name = name)
    rp = planar_pin(rotated, pin_name).point
    return planar_transform(rotated; offset = (target.point[1] - rp[1],
                                               target.point[2] - rp[2]))
end

# ── lowering to schema tables ──

"""
    planar_layout_dict(shapes; unit="m") -> Dict("polygons" => ..., "vias" => ...)

Lower shapes to `[[polygons]]` / `[[vias]]` schema tables (vertex numbers in
`unit`, the project's default length unit) for
layout tables. Names must be unique across shapes.
"""
function planar_layout_dict(shapes::AbstractVector{PlanarShape}; unit::AbstractString = "m")
    units = Dict("m"=>1.0, "mm"=>1e-3, "um"=>1e-6, "nm"=>1e-9,
        "mil"=>25.4e-6, "in"=>.0254)
    haskey(units, unit) || throw(ArgumentError("unsupported layout length unit $unit"))
    s = 1.0 / units[unit]
    polys = Any[]
    vias = Any[]
    seen = Set{String}()
    claim(n) = (n in seen && throw(ArgumentError("duplicate layout name `$n`"));
                push!(seen, n))
    for sh in shapes
        for p in sh.polygons
            claim(p.name)
            t = Dict{String,Any}("name" => p.name, "level" => p.level,
                                 "metal" => p.metal,
                                 "vertices" => Any[Any[v[1] * s, v[2] * s]
                                                   for v in p.vertices])
            isempty(p.net) || (t["net"] = p.net)
            push!(polys, t)
        end
        for v in sh.vias
            claim(v.name)
            push!(vias, Dict{String,Any}(
                "name" => v.name, "via_type" => v.via_type,
                "from_level" => v.from_level,
                "to_level" => v.to_level == -1 ? "gnd" : v.to_level,
                "vertices" => Any[Any[q[1] * s, q[2] * s] for q in v.vertices]))
        end
    end
    return Dict{String,Any}("polygons" => polys, "vias" => vias)
end

planar_layout_dict(sh::PlanarShape; kwargs...) = planar_layout_dict([sh]; kwargs...)

"""
    planar_pin_port_dict(pin; number, type=:box_wall, kwargs...) -> Dict

A `[[ports]]` table attaching port `number` to `pin`'s polygon edge. Extra
keyword arguments (e.g. `resistance = 50.0`, `refplane = "1 mm"`) are copied
verbatim.
"""
function planar_pin_port_dict(pin::PlanarPin; number::Integer,
                              type::Symbol = :box_wall, kwargs...)
    number != 0 || throw(ArgumentError("port number must be nonzero"))
    d = Dict{String,Any}("number" => Int(number), "type" => String(type),
                         "polygon" => pin.polygon, "edge" => pin.edge)
    for (k, v) in kwargs
        d[String(k)] = v
    end
    return d
end

"""    planar_layout_bbox(shapes) -> (xmin, ymin, xmax, ymax) over polygons and vias."""
function planar_layout_bbox(shapes::AbstractVector{PlanarShape})
    xmin = ymin = Inf
    xmax = ymax = -Inf
    for sh in shapes, vs in Iterators.flatten(((p.vertices for p in sh.polygons),
                                               (v.vertices for v in sh.vias))),
        q in vs
        xmin = min(xmin, q[1]); ymin = min(ymin, q[2])
        xmax = max(xmax, q[1]); ymax = max(ymax, q[2])
    end
    isfinite(xmin) || throw(ArgumentError("layout has no geometry"))
    return (xmin, ymin, xmax, ymax)
end

"""
    planar_interdigital_capacitor(; fingers, finger_length, width, gap,
                                  bus_width=width, level, metal, name="idc")

Two interleaved planar comb electrodes centred in y, with `fingers` fingers
on each electrode, edge spacing `gap`, and an opposite-bus clearance of
`gap`. The terminal pins lie on the outer bus edges. Polygons belonging to
one comb meet its bus and retain their declared net name. The capacitance
is obtained by EM analysis; no lumped capacitance is inserted in geometry.
"""
function planar_interdigital_capacitor(; fingers::Integer, finger_length::Real,
        width::Real, gap::Real, bus_width::Real=width, level::Integer,
        metal::AbstractString, name::AbstractString="idc",
        net1::AbstractString="", net2::AbstractString="")
    1 <= fingers <= 100_000 || throw(ArgumentError("fingers must be in 1:100000"))
    f = Int(fingers)
    l, w, g, bw = _lib_pos(finger_length,"finger_length"),
        _lib_pos(width,"width"), _lib_pos(gap,"gap"), _lib_pos(bus_width,"bus_width")
    lev = _lib_level(level,"level")
    height = 2*f*w+(2*f-1)*g
    xright = bw+l+g
    rect(a,b,c,d) = _P2[_P2(a,c),_P2(b,c),_P2(b,d),_P2(a,d)]
    left = _lib_polygon("$(name)_left_bus",lev,metal,net1,rect(0.,bw,-height/2,height/2))
    right = _lib_polygon("$(name)_right_bus",lev,metal,net2,
        rect(xright,xright+bw,-height/2,height/2))
    polys = [left,right]
    for k in 0:2*f-1
        low = -height/2+k*(w+g)
        side = iseven(k) ? "left" : "right"
        a,b = iseven(k) ? (bw,bw+l) : (bw+g,xright)
        push!(polys,_lib_polygon("$(name)_$(side)_$(k+1)",lev,metal,
            iseven(k) ? net1 : net2,rect(a,b,low,low+w)))
    end
    pins = [_lib_pin("p1",left,4,_P2(-1,0)),_lib_pin("p2",right,2,_P2(1,0))]
    return PlanarShape(String(name),polys,PlanarShapeVia[],pins,
        Dict("fingers"=>Float64(f),"finger_length"=>l,"gap"=>g,
            "height"=>height,"length"=>xright+bw))
end
