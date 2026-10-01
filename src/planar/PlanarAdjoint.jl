# PlanarAdjoint.jl — Parameter gradients for the planar solver
#
# Objective J(Y) is a real scalar of the short-circuit admittance (or of
# derived quantities such as S = planar_y_to_s(Y, z0)).  The implicit
# dependence through the solve gives the Wirtinger-form pullback
#     dJ = Re sum(G .* dY),   dY = -transpose(Λmat) dZ X,
#     Λmat = Z^{-T} E,     X = Z^{-1} B,
# where E and B are the port extraction/excitation matrices and
# G[p,q] = dJ/dRe(Y_pq) - i*dJ/dIm(Y_pq).  With the modal decomposition
# Z = -sum_m V_m w_m w_m' this collapses to a per-mode contraction
#     dJ/dθ = Re sum_{m,pol,π} (u'_{m,π} G v_{m,π}) * dV^{pol}_{m,π}/dθ
# with u = W' Λmat restricted to the pair's field-side bases and
# v = W' X restricted to its source-side bases.  Modal-voltage
# derivatives dV/dθ are computed exactly by rerunning the transmission-line
# cascade on a _PlanarDual-typed stackup (forward-mode), so each parameter
# costs one extra cascade sweep and no factorization.
#
# Complex-stepping V(θ) directly would only recover Re(dV/dθ) once Im(V)
# is nonzero, so the dual type is required for a machine-precision adjoint.

export PlanarParam
export planar_default_params, planar_param_values, planar_with_params
export planar_objective_gradient

# ---------------- minimal forward-mode dual ----------------

"""Scalar (value, derivative) pair used to propagate d/dθ through the
modal cascade.  `d` is the derivative of `v` with respect to one selected
stackup parameter; branch predicates inspect `v` only."""
struct _PlanarDual{T<:Number} <: Number
    v::T
    d::T
end

_PlanarDual{T}(x::Number) where {T<:Number} = _PlanarDual{T}(T(x), zero(T))
_PlanarDual(v::Number, d::Number) = _PlanarDual(promote(v, d)...)

Base.promote_rule(::Type{_PlanarDual{T}}, ::Type{S}) where {T<:Number,
    S<:Number} = _PlanarDual{promote_type(T, S)}
Base.convert(::Type{_PlanarDual{T}}, x::Number) where {T<:Number} =
    _PlanarDual{T}(x)
Base.convert(::Type{_PlanarDual{T}}, x::_PlanarDual) where {T<:Number} =
    _PlanarDual{T}(convert(T, x.v), convert(T, x.d))

Base.zero(::Type{_PlanarDual{T}}) where {T<:Number} =
    _PlanarDual{T}(zero(T), zero(T))
Base.zero(x::_PlanarDual) = zero(typeof(x))
Base.one(::Type{_PlanarDual{T}}) where {T<:Number} =
    _PlanarDual{T}(one(T), zero(T))
Base.one(x::_PlanarDual) = one(typeof(x))
Base.iszero(x::_PlanarDual) = iszero(x.v)
Base.isone(x::_PlanarDual) = isone(x.v)
Base.isfinite(x::_PlanarDual) = isfinite(x.v)
Base.real(x::_PlanarDual) = real(x.v)
Base.imag(x::_PlanarDual) = imag(x.v)
Base.show(io::IO, x::_PlanarDual) = print(io, "(", x.v, ")+(", x.d, ")ε")

@inline Base.:+(x::_PlanarDual, y::_PlanarDual) =
    _PlanarDual(x.v + y.v, x.d + y.d)
@inline Base.:-(x::_PlanarDual, y::_PlanarDual) =
    _PlanarDual(x.v - y.v, x.d - y.d)
@inline Base.:-(x::_PlanarDual) = _PlanarDual(-x.v, -x.d)
@inline Base.:*(x::_PlanarDual, y::_PlanarDual) =
    _PlanarDual(x.v * y.v, x.d * y.v + x.v * y.d)
# a division by an infinite factor means exact decoupling: the value is
# zero to first order and the derivative contribution is nil
@inline function Base.:/(x::D, y::D) where {D<:_PlanarDual}
    if _planarisinf(y.v)
        _planarisinf(x.v) && return D(x.v / y.v, x.v / y.v)
        return zero(D)
    end
    iy = inv(y.v)
    return D(x.v * iy, (x.d - x.v * iy * y.d) * iy)
end
@inline Base.inv(x::_PlanarDual) =
    _PlanarDual(inv(x.v), -x.d * inv(x.v)^2)
# zero seed part must not turn sqrt(0) into NaN through 0/0
@inline Base.sqrt(x::_PlanarDual) =
    (s = sqrt(x.v);
     _PlanarDual(s, iszero(x.d) ? zero(x.d) : x.d / (2 * s)))
@inline Base.exp(x::_PlanarDual) =
    (e = exp(x.v); _PlanarDual(e, x.d * e))
@inline Base.conj(x::_PlanarDual) = _PlanarDual(conj(x.v), conj(x.d))

# derivative-carrying limits at the e1 = 1 / e2 = 1 transparent-section
# points: the value branch in PlanarImmittance is exact, but a parameter
# that moves gamma*d through the degenerate point contributes
#   d(inv_tau) = u * d(gamma*d)      (u = Zc / load at the far face)
#   d(Zin)     = (Zc - zl^2/Zc) * d(gamma*d)
# with e1 = exp(-gd) -> d(gd) = -e1.d and e2 = exp(-2gd) -> -e2.d/2.
@inline function _planar_e1_one_limit(u::D, e1::D) where {D<:_PlanarDual}
    iszero(e1.d) && return one(D)
    return D(one(u.v), -u.v * e1.d)
end

@inline function _planar_e2_one_limit(zc::D, e2::D, zl::D) where
        {D<:_PlanarDual}
    iszero(e2.d) && return zl
    return D(zl.v, zl.d - (zc.v - zl.v * zl.v / zc.v) * e2.d / 2)
end

# ---------------- differentiable stackup parameters ----------------

"""One scalar degree of freedom of a `PlanarStackup`: `index` selects a
layer (`1..L`) or a box terminator (`0` = bottom, `L+1` = top); `field` is
`:epsr`, `:mur`, `:thickness` for layers and `:zs`, `:epsr`, `:mur` for
terminators; `part` is `:re` or `:im`."""
struct PlanarParam
    index::Int
    field::Symbol
    part::Symbol
    function PlanarParam(index::Integer, field::Symbol, part::Symbol)
        index >= 0 || throw(ArgumentError(
            "PlanarParam index must be >= 0, got $index"))
        field in (:epsr, :mur, :thickness, :zs) || throw(ArgumentError(
            "PlanarParam field must be :epsr, :mur, :thickness or :zs, " *
            "got :$field"))
        part in (:re, :im) || throw(ArgumentError(
            "PlanarParam part must be :re or :im, got :$part"))
        return new(Int(index), field, part)
    end
end

@inline _planar_param_fields_layer() = (:epsr, :mur, :thickness)
@inline _planar_param_fields_term() = (:zs, :epsr, :mur)

function _planar_param_check(stack::PlanarStackup, p::PlanarParam)
    L = length(stack.layers)
    0 <= p.index <= L + 1 || throw(ArgumentError(
        "PlanarParam index $(p.index) outside 0:$(L + 1) for $L layers"))
    if 1 <= p.index <= L
        p.field in _planar_param_fields_layer() || throw(ArgumentError(
            "layer parameter field must be :epsr, :mur or :thickness, " *
            "got :$(p.field)"))
    else
        p.field in _planar_param_fields_term() || throw(ArgumentError(
            "terminator parameter field must be :zs, :epsr or :mur, " *
            "got :$(p.field)"))
    end
    return nothing
end

"""Current Float64 value of `p` in `stack` (real or imaginary part)."""
function planar_param_values(stack::PlanarStackup,
        params::AbstractVector{PlanarParam})
    theta = Vector{Float64}(undef, length(params))
    @inbounds for j in eachindex(params)
        p = params[j]
        _planar_param_check(stack, p)
        obj = if 1 <= p.index <= length(stack.layers)
            stack.layers[p.index]
        elseif p.index == 0
            stack.bottom
        else
            stack.top
        end
        f = getfield(obj, p.field)
        theta[j] = p.part === :re ? Float64(real(f)) : Float64(imag(f))
    end
    return theta
end

"""Default differentiable parameters: every layer's epsr (re, im),
mur (re, im), and thickness; surface-terminator `zs` and open-terminator
exterior `epsr`/`mur` (re, im) when the stackup uses those kinds."""
function planar_default_params(stack::PlanarStackup)
    params = PlanarParam[]
    for l in eachindex(stack.layers)
        for f in (:epsr, :mur)
            push!(params, PlanarParam(l, f, :re))
            push!(params, PlanarParam(l, f, :im))
        end
        push!(params, PlanarParam(l, :thickness, :re))
    end
    for (idx, term) in ((0, stack.bottom), (length(stack.layers) + 1,
                        stack.top))
        if term.kind === TERM_SURFACE
            push!(params, PlanarParam(idx, :zs, :re))
            push!(params, PlanarParam(idx, :zs, :im))
        elseif term.kind === TERM_OPEN
            for f in (:epsr, :mur)
                push!(params, PlanarParam(idx, f, :re))
                push!(params, PlanarParam(idx, f, :im))
            end
        end
    end
    return params
end

# new field value with one component replaced by x (x may be real,
# complex, or dual — promotion carries the perturbation type)
@inline _set_part(f::Number, ::Val{:re}, x::Number) = x + 1im * imag(f)
@inline _set_part(f::Number, ::Val{:im}, x::Number) = real(f) + 1im * x

"""Rebuild `stack` with `params[j] = theta[j]` for all j.  The returned
stackup scalar type promotes with `eltype(theta)`, so `ComplexF64` theta
selects complex-step evaluation."""
function planar_with_params(stack::PlanarStackup,
        params::AbstractVector{PlanarParam},
        theta::AbstractVector{<:Number})
    length(theta) == length(params) || throw(DimensionMismatch(
        "theta length $(length(theta)) != params $(length(params))"))
    L = length(stack.layers)
    upd = Dict{Int,Vector{Tuple{Symbol,Symbol,Number}}}()
    @inbounds for j in eachindex(params)
        p = params[j]
        _planar_param_check(stack, p)
        isfinite(real(theta[j])) && isfinite(imag(theta[j])) ||
            throw(ArgumentError("theta[$j] is non-finite: $(theta[j])"))
        push!(get!(upd, p.index, Tuple{Symbol,Symbol,Number}[]),
            (p.field, p.part, theta[j]))
    end
    layers = map(1:L) do l
        haskey(upd, l) || return stack.layers[l]
        lay = stack.layers[l]
        e, m, d = lay.epsr, lay.mur, lay.thickness
        for (field, part, x) in upd[l]
            if field === :epsr
                e = _set_part(e, Val(part), x)
            elseif field === :mur
                m = _set_part(m, Val(part), x)
            else
                d = _set_part(d, Val(part), x)
            end
        end
        return PlanarLayer(e, m, d)
    end
    function term_for(idx, term)
        haskey(upd, idx) || return term
        zs, e, m = term.zs, term.epsr, term.mur
        for (field, part, x) in upd[idx]
            if field === :zs
                zs = _set_part(zs, Val(part), x)
            elseif field === :epsr
                e = _set_part(e, Val(part), x)
            else
                m = _set_part(m, Val(part), x)
            end
        end
        return PlanarTerminator(term.kind, zs, e, m)
    end
    return PlanarStackup(PlanarLayer[layers...],
        term_for(0, stack.bottom), term_for(L + 1, stack.top),
        stack.a, stack.b)
end

# stackup with every scalar as _PlanarDual{ComplexF64}; the field named by
# `p` carries unit seed so cascade outputs hold (value, d/dtheta_p)
function _planar_dual_stackup(stack::PlanarStackup, p::PlanarParam)
    _planar_param_check(stack, p)
    D(v) = _PlanarDual{ComplexF64}(ComplexF64(v), zero(ComplexF64))
    L = length(stack.layers)
    layers = map(1:L) do l
        lay = stack.layers[l]
        e, m, d = D(lay.epsr), D(lay.mur), D(lay.thickness)
        if l == p.index
            if p.field === :epsr
                e = _PlanarDual{ComplexF64}(ComplexF64(lay.epsr),
                    _seed(p.part))
            elseif p.field === :mur
                m = _PlanarDual{ComplexF64}(ComplexF64(lay.mur),
                    _seed(p.part))
            else
                d = _PlanarDual{ComplexF64}(ComplexF64(lay.thickness),
                    _seed(p.part))
            end
        end
        return PlanarLayer(e, m, d)
    end
    function term_for(idx, term)
        zs, e, m = D(term.zs), D(term.epsr), D(term.mur)
        if idx == p.index
            if p.field === :zs
                zs = _PlanarDual{ComplexF64}(ComplexF64(term.zs),
                    _seed(p.part))
            elseif p.field === :epsr
                e = _PlanarDual{ComplexF64}(ComplexF64(term.epsr),
                    _seed(p.part))
            else
                m = _PlanarDual{ComplexF64}(ComplexF64(term.mur),
                    _seed(p.part))
            end
        end
        return PlanarTerminator(term.kind, zs, e, m)
    end
    return PlanarStackup(collect(PlanarLayer{_PlanarDual{ComplexF64}},
            layers),
        term_for(0, stack.bottom), term_for(L + 1, stack.top),
        stack.a, stack.b)
end

# unit seed direction: :re seeds d/dtheta on the real part, :im on the
# imaginary part (field + i*dtheta)
@inline _seed(part::Symbol) =
    part === :re ? one(ComplexF64) : ComplexF64(0, 1)

# ---------------- Wirtinger gradient of the port objective ----------------

# G[p,q] = dJ/dRe(Y[p,q]) - i*dJ/dIm(Y[p,q]) via central differences of
# the real-valued objective on each Y entry
function _planar_wirtinger_fd(f, Y::Matrix{ComplexF64}, h::Float64)
    n = size(Y, 1)
    G = Matrix{ComplexF64}(undef, n, n)
    Yw = copy(Y)
    for q in 1:n, p in 1:n
        hp = h * max(abs(Y[p, q]), one(Float64))
        y0 = Y[p, q]
        Yw[p, q] = y0 + hp
        fp = _checked_objective(f, Yw)
        Yw[p, q] = y0 - hp
        fm = _checked_objective(f, Yw)
        Yw[p, q] = y0 + 1im * hp
        gi = _checked_objective(f, Yw)
        Yw[p, q] = y0 - 1im * hp
        gm = _checked_objective(f, Yw)
        Yw[p, q] = y0
        G[p, q] = ((fp - fm) - 1im * (gi - gm)) / (2 * hp)
    end
    return G
end

function _checked_objective(f, Y::Matrix{ComplexF64})
    v = f(Y)
    v isa Real ||
        throw(ArgumentError(
            "planar objective must return a real scalar, got " *
            "$(typeof(v))"))
    isfinite(v) ||
        throw(ArgumentError("planar objective returned $v"))
    return Float64(v)
end

# ---------------- objective gradient ----------------

"""
    planar_objective_gradient(prob, freq, f; kw...) -> (J, grad)

Value and stackup-parameter gradient of the real scalar objective
`f(Y)` evaluated on the port admittance from `solve_planar(prob, freq)`.

`params` (default `planar_default_params(prob.stack)`) selects the
differentiated scalars.  `gY` optionally supplies the analytic Wirtinger
gradient `Y -> G` with `G[p,q] = dJ/dRe Y_pq - i dJ/dIm Y_pq`; otherwise a
central finite difference on the `f` evaluation is used (`h_fd`).  Other
keywords (`mx`, `my`, `block`, `surface_zs`, `max_bytes`) are forwarded to
the forward assembly.  Returns `(J, grad)` with `grad` aligned to
`params`.

Note that plain complex-stepping cannot recover `dY/dθ` because `Y` is
already complex (the i*eps perturbation cancels against the imaginary
baseline); the `_PlanarDual` cascade exists precisely for this reason.
`planar_param_values` + `planar_with_params` remain available for
finite-difference verification of both the solve and this gradient.
"""
function planar_objective_gradient(prob::PlanarProblem, freq::Number,
        f; params::AbstractVector{PlanarParam}=
        planar_default_params(prob.stack),
        gY=nothing, h_fd::Real=1e-6,
        mx::Integer=2 * prob.grid.nx, my::Integer=2 * prob.grid.ny,
        block::Integer=512,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES, kw...)
    h = Float64(h_fd)
    isfinite(h) && h > 0 ||
        throw(ArgumentError("h_fd must be finite and positive"))
    r = solve_planar(prob, freq; mx=mx, my=my, block=block,
        max_bytes=max_bytes, kw...)
    J = _checked_objective(f, r.y)
    G = gY === nothing ? _planar_wirtinger_fd(f, r.y, h) :
        begin
            gv = gY(r.y)
            (gv isa AbstractMatrix && size(gv) == size(r.y) &&
             all(isfinite, gv)) ||
                throw(ArgumentError(
                    "gY(Y) must return a finite $(size(r.y)) matrix"))
            ComplexF64.(gv)
        end

    nb = planar_basis_count(prob.basis)
    np = length(prob.ports)
    L = length(prob.stack.layers)
    mg = planar_mode_grid(prob.grid, mx, my)
    nmode = mg.mx * mg.my
    blk = clamp(Int(block), 1, nmode)
    iface = Vector{Int}(undef, nb)
    @inbounds for b in 1:nb
        iface[b] = prob.sheets[prob.basis.level[b]].interface
    end
    pairs = _level_pairs(iface)
    npair = length(pairs)
    pbs = _port_basis_indices(prob.basis, np)

    est = _checked_payload_sum("planar gradient",
        _checked_array_payload_bytes(ComplexF64, nb, np),   # E/adjoint
        _checked_array_payload_bytes(ComplexF64, np, np),   # G
        _checked_array_payload_bytes(ComplexF64, np, np),   # Yw scratch
        _checked_array_payload_bytes(Float64, mg.mx, nb),   # fxb
        _checked_array_payload_bytes(Float64, mg.my, nb),   # fyb
        _checked_array_payload_bytes(Float64, blk, nb),     # Wte
        _checked_array_payload_bytes(Float64, blk, nb),     # Wtm
        _checked_array_payload_bytes(ComplexF64, blk, np),  # Ub
        _checked_array_payload_bytes(ComplexF64, blk, np),  # Vb
        _checked_array_payload_bytes(ComplexF64, blk, npair),   # Cte
        _checked_array_payload_bytes(ComplexF64, blk, npair),   # Ctm
        _checked_array_payload_bytes(_PlanarDual{ComplexF64}, blk, npair),
        _checked_array_payload_bytes(_PlanarDual{ComplexF64}, blk, npair),
        _checked_array_payload_bytes(Int, blk),
        _checked_array_payload_bytes(Int, blk),
        _checked_array_payload_bytes(Int, nb),              # iface
        _checked_array_payload_bytes(Int, nb),              # port lists
        _checked_array_payload_bytes(Int, 2 * npair, nb),   # pair rows/cols
        _checked_array_payload_bytes(Float64, 2 * length(params) + np),
        # dual cascade state: 2 pols x (2*(L+1) + 2*L) + 3*L duals
        _checked_array_payload_bytes(_PlanarDual{ComplexF64},
            11 * (L + 1)),
        # per-parameter dual stackups: 3 duals/layer + 3/terminator x2
        _checked_array_payload_bytes(_PlanarDual{ComplexF64},
            (3 * L + 6) * length(params)))
    _enforce_payload_limit(est, max_bytes, "planar gradient", "max_bytes")

    # port signs: +1 west/south, -1 east/north (solve_planar convention)
    sgn = [p.wall in (:west, :south) ? 1.0 : -1.0 for p in prob.ports]
    E = zeros(ComplexF64, nb, np)
    @inbounds for p in 1:np, b in pbs[p]
        E[b, p] = sgn[p] * prob.basis.width[b]
    end
    Λmat = ldiv!(transpose(r.lu_fact), E)

    pair_rows = Dict{Tuple{Int,Int},Vector{Int}}()
    pair_cols = Dict{Tuple{Int,Int},Vector{Int}}()
    for pr in pairs
        pair_rows[pr] = findall(==(pr[1]), iface)
        pair_cols[pr] = findall(==(pr[2]), iface)
    end

    fxb = Matrix{Float64}(undef, mg.mx, nb)
    fyb = Matrix{Float64}(undef, mg.my, nb)
    @inbounds for b in 1:nb
        _basis_fx!(view(fxb, :, b), prob.basis, b, mg, prob.grid)
        _basis_fy!(view(fyb, :, b), prob.basis, b, mg, prob.grid)
    end

    Wte = Matrix{Float64}(undef, blk, nb)
    Wtm = Matrix{Float64}(undef, blk, nb)
    Ub = Matrix{ComplexF64}(undef, blk, np)
    Vb = Matrix{ComplexF64}(undef, blk, np)
    Cte = Matrix{ComplexF64}(undef, blk, npair)
    Ctm = Matrix{ComplexF64}(undef, blk, npair)
    D = _PlanarDual{ComplexF64}
    dvte = Matrix{D}(undef, blk, npair)
    dvtm = Matrix{D}(undef, blk, npair)
    casc_te = PlanarCascade(Vector{D}(undef, L + 1), Vector{D}(undef, L + 1),
        Vector{D}(undef, L), Vector{D}(undef, L))
    casc_tm = PlanarCascade(Vector{D}(undef, L + 1), Vector{D}(undef, L + 1),
        Vector{D}(undef, L), Vector{D}(undef, L))
    scratch = (Vector{D}(undef, L), Vector{D}(undef, L),
               Vector{D}(undef, L))
    mlist = Vector{Int}(undef, blk)
    nlist = Vector{Int}(undef, blk)
    ab4 = prob.grid.a * prob.grid.b / 4
    g = zeros(Float64, length(params))
    dual_stacks = [_planar_dual_stackup(prob.stack, p) for p in params]

    c = 0
    for n in 1:mg.my, m in 1:mg.mx
        c += 1
        mlist[c] = m
        nlist[c] = n
        if c == blk || (m == mg.mx && n == mg.my)
            cblk = c
            _planar_weight_block!(Wte, Wtm, view(mlist, 1:cblk),
                view(nlist, 1:cblk), mg, fxb, fyb, prob.basis)
            # contraction coefficients per (mode, pair): C = u' * G * v
            @inbounds for pi_ in eachindex(pairs)
                rows = pair_rows[pairs[pi_]]
                cols = pair_cols[pairs[pi_]]
                for (W, C) in ((Wte, Cte), (Wtm, Ctm))
                    mul!(view(Ub, 1:cblk, :), view(W, 1:cblk, rows),
                        view(Λmat, rows, :))
                    mul!(view(Vb, 1:cblk, :), view(W, 1:cblk, cols),
                        view(r.currents, cols, :))
                    for cm in 1:cblk
                        acc = zero(ComplexF64)
                        for q in 1:np, p in 1:np
                            acc += G[p, q] * Ub[cm, p] * Vb[cm, q]
                        end
                        C[cm, pi_] = acc
                    end
                end
            end
            # per-parameter modal-voltage derivatives via dual cascade
            for j in eachindex(params)
                ds = dual_stacks[j]
                _planar_mode_voltages!(dvte, dvtm, casc_te, casc_tm,
                    scratch, ds, r.omega, mg, view(mlist, 1:cblk),
                    view(nlist, 1:cblk), pairs, ab4)
                acc = zero(ComplexF64)
                @inbounds for pi_ in 1:npair, cm in 1:cblk
                    acc += Cte[cm, pi_] * dvte[cm, pi_].d +
                           Ctm[cm, pi_] * dvtm[cm, pi_].d
                end
                g[j] += real(acc)
            end
            c = 0
        end
    end

    all(isfinite, g) || throw(ArgumentError(
        "planar gradient is non-finite: the frequency may sit on a " *
        "box resonance or a parameter may leave the model space"))
    return (J, g)
end
