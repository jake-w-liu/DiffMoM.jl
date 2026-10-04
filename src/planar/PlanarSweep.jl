# PlanarSweep.jl — Adaptive rational frequency sweep and box-resonance scan
#
# Adaptive band synthesis (ABS) after Rautio, "Generating Spectrally Rich
# Data Sets Using Adaptive Band Synthesis Interpolation" (Sonnet Software,
# 2003) and Microwaves & RF, May 2002:
#
#   1. analyze the two band edges and the mid frequency,
#   2. form rational (Pade-family) interpolation models of each
#      S-parameter entry through the analyzed points,
#   3. estimate interpolation error as the difference between the full
#      model and the model with the newest point dropped,
#   4. find the frequency of worst estimated error on a dense grid,
#   5. quit when the estimated error is at least `rel_db` below the data
#      at every evaluation frequency (Sonnet uses 40 dB; the true error
#      can be up to 20 dB worse than the estimate),
#   6. otherwise analyze the worst frequency and repeat.
#
# Interpolation uses a greedy barycentric rational representation with
# Loewner-SVD weights. Small or unresolved sample sets retain the Thiele
# continued-fraction and Floater-Hormann paths. Every accepted barycentric
# model must reproduce all analyzed data to roundoff; the adaptive error
# estimate remains the difference of the full and reduced models.
#
# Box resonances are natural modes of the shielded stackup: the parallel
# modal impedance zdn(f) || zup(f) at a reference interface develops a
# pole, i.e. zdn + zup -> 0 (see `planar_modal_voltage`).  Each resonance
# is reported with its transverse mode numbers (m, n) and polarization.

export PlanarSweep, planar_sweep_abs
export PlanarResonance, planar_box_resonances

# ---------------- Thiele rational interpolation ----------------

# Continued-fraction rational interpolant through (x[i], y[i]):
#   R(x) = a[1] + (x - x[1])/(a[2] + (x - x[2])/(a[3] + ... /a[ndeg]))
# a[k] is the reciprocal difference of order k-1.  A nonfinite
# reciprocal difference (numerically duplicated ordinates) truncates the
# fraction at ndeg < n: the truncated model still interpolates the first
# ndeg points and the sweep treats the result as high-error so a fresh
# analysis point is selected.
struct _ThieleCF
    x::Vector{Float64}
    a::Vector{ComplexF64}
    ndeg::Int
    nfull::Int            # number of samples requested
end

function _thiele_build(x::AbstractVector{<:Real},
        y::AbstractVector{<:Number})
    n = length(x)
    n == length(y) && n >= 1 ||
        throw(ArgumentError("Thiele interpolant needs matching nonempty x and y"))
    a = Vector{ComplexF64}(undef, n)
    a[1] = y[1]
    ndeg = 1
    # coefficient k is the tail value the chain must reach at x[k]:
    #   t_1 = y[k];  t_j = (x[k] - x[j-1]) / (t_{j-1} - a[j-1]);  a[k] = t_k
    @inbounds for k in 2:n
        u = ComplexF64(y[k])
        degenerate = false
        for j in 2:k
            den = u - a[j - 1]
            if iszero(den) || !isfinite(den)
                degenerate = true
                break
            end
            u = (x[k] - x[j - 1]) / den
        end
        degenerate && break
        a[k] = u
        ndeg = k
    end
    resize!(a, ndeg)
    return _ThieleCF(Float64.(x[1:ndeg]), a, ndeg, n)
end

@inline function _thiele_eval(t::_ThieleCF, x::Float64)
    n = t.ndeg
    n == 1 && return t.a[1]
    acc = t.a[n]
    @inbounds for k in (n - 1):-1:2
        iszero(acc) && return ComplexF64(Inf, Inf)
        acc = t.a[k] + (x - t.x[k]) / acc
    end
    iszero(acc) && return ComplexF64(Inf, Inf)
    return t.a[1] + (x - t.x[1]) / acc
end

# Floater-Hormann rational interpolation handles equal ordinates and
# low-degree data where reciprocal differences terminate. With sorted
# real nodes its denominator has no real poles. The normalized frequency
# coordinate prevents products of GHz-valued differences from dominating.
struct _SweepRational
    x::Vector{Float64}
    y::Vector{ComplexF64}
    w::Vector{Float64}
end

# Greedy barycentric rational interpolation following Nakatsukasa, Sète
# and Trefethen, SIAM J. Sci. Comput. 40 (2018), A1494–A1522,
# doi:10.1137/16M1106122. The smallest right singular vector of the
# Loewner matrix supplies weights. Only a model reproducing every
# analyzed value to the same roundoff gate is accepted here.
struct _SweepAAA
    x::Vector{Float64}
    y::Vector{ComplexF64}
    w::Vector{ComplexF64}
    scale::Float64
end

function _sweep_eval(t::_SweepAAA,x::Float64)
    near=argmin(abs(x-xi) for xi in t.x)
    distance=x-t.x[near]
    iszero(distance) && return t.scale*t.y[near]
    numerator=zero(ComplexF64);denominator=zero(ComplexF64)
    @inbounds for k in eachindex(t.x)
        # Scaling by the closest-node distance avoids large partial
        # fractions arbitrarily near a support point.
        term=t.w[k]*(distance/(x-t.x[k]))
        numerator+=term*t.y[k];denominator+=term
    end
    return t.scale*(numerator/denominator)
end

function _sweep_aaa_model(x::Vector{Float64},y::Vector{ComplexF64})
    n=length(x);n>3 || return nothing
    scale=maximum(z->max(abs(real(z)),abs(imag(z))),y)
    iszero(scale) && return nothing
    values=y./scale
    tolerance=64eps(Float64)*max(maximum(abs,values),floatmin(Float64))
    approximation=fill(sum(values)/n,n);support=Int[]
    for _ in 1:cld(n+1,2)
        remaining=[k for k in 1:n if !(k in support)]
        isempty(remaining) && break
        chosen=remaining[argmax(abs.(values[remaining]-approximation[remaining]))]
        push!(support,chosen)
        remaining=[k for k in 1:n if !(k in support)]
        loewner=ComplexF64[(values[i]-values[j])/(x[i]-x[j]) for i in remaining,j in support]
        all(isfinite,loewner) || return nothing
        decomposition=LinearAlgebra.svd(loewner;full=true)
        weights=Vector{ComplexF64}(decomposition.V[:,end])
        nonzero=findall(!iszero,weights)
        isempty(nonzero) && return nothing
        selected=support[nonzero]
        model=_SweepAAA(x[selected],values[selected],weights[nonzero],1.)
        approximation=[_sweep_eval(model,xi) for xi in x]
        if all(isfinite,approximation) && maximum(abs,approximation-values)<=tolerance
            return _SweepAAA(model.x,model.y,model.w,scale)
        end
    end
    return nothing
end

function _sweep_model(x::Vector{Float64}, y::Vector{ComplexF64})
    barycentric=_sweep_aaa_model(x,y)
    barycentric===nothing || return barycentric
    t = _thiele_build(x, y)
    # A terminated continued fraction may represent every sample exactly
    # (constant/linear/low-degree rational response). Only trust it after
    # checking the requested data, including the samples it did not use.
    tolerance = 64eps(Float64) * max(maximum(abs, y), floatmin(Float64))
    all(abs(_thiele_eval(t, x[k]) - y[k]) <= tolerance for k in eachindex(x)) &&
        return t
    order = sortperm(x)
    xs, ys = x[order], y[order]
    n = length(xs)
    d = min(3, n - 1)
    w = zeros(Float64, n)
    for i in 1:n, k in max(1, i - d):min(i, n - d)
        term = isodd(k) ? 1.0 : -1.0
        for j in k:(k + d)
            j == i || (term /= xs[i] - xs[j])
        end
        w[i] += term
    end
    w ./= maximum(abs, w)
    return _SweepRational(xs, ys, w)
end

@inline _sweep_eval(t::_ThieleCF, x::Float64) = _thiele_eval(t, x)
function _sweep_eval(t::_SweepRational, x::Float64)
    near = argmin(abs(x - xi) for xi in t.x)
    distance = x - t.x[near]
    iszero(distance) && return t.y[near]
    numerator = zero(ComplexF64)
    denominator = 0.0
    @inbounds for k in eachindex(t.x)
        term = t.w[k] * (distance / (x - t.x[k]))
        numerator += term * t.y[k]
        denominator += term
    end
    return numerator / denominator
end

# ---------------- adaptive sweep ----------------

"""Result of [`planar_sweep_abs`](@ref): the analyzed frequency set, the
S-matrix at each analysis frequency, the dense interpolated grid, and the
worst-entry relative error estimate per dense frequency."""
struct PlanarSweep
    freqs::Vector{Float64}              # analyzed frequencies [Hz]
    s::Vector{Matrix{ComplexF64}}       # S-matrix per analyzed frequency
    dense_freqs::Vector{Float64}        # dense evaluation grid [Hz]
    dense_s::Vector{Matrix{ComplexF64}} # interpolated S on the dense grid
    est_err::Vector{Float64}            # worst-entry |R - R_lo|/|R| per dense f
    converged::Bool                     # est_err < rel_tol everywhere
    z0::Union{Nothing,Vector{ComplexF64}} # fixed wave references, when supplied
end
PlanarSweep(f,s,df,ds,e,c)=PlanarSweep(f,s,df,ds,e,c,nothing)

"""
    planar_sweep_abs(prob, fmin, fmax; kw...) -> PlanarSweep

Adaptive band synthesis of `prob`'s port S-parameters on
`[fmin, fmax]`.  Each full EM analysis runs `solve_planar(prob, f; kw...)`;
new analysis frequencies are placed at the largest estimated interpolation
error until every dense-grid estimate is `rel_tol` below the data
(Sonnet ABS rule: estimated error at least 40 dB under the S-parameter
magnitude, i.e. `rel_tol = 1e-2`).

Keywords: `rel_tol=1e-2`, `n_eval=257` dense candidates, `max_points=32`
analysis cap (the sweep then returns `converged=false`),
`solve_kw...` forwarded to `solve_planar`.
Frequency endpoints and candidate frequencies must be distinct, positive
and finite in Float64 storage. `max_bytes` includes the sequential
barycentric interpolation workspace before any response analysis.
`converged` certifies the reported interpolation estimate on the declared
grid; it does not guarantee discovery of an unsampled narrow resonance.
"""
function planar_sweep_abs(prob::PlanarProblem,
        fmin::Real, fmax::Real; rel_tol::Real=1e-2,
        n_eval::Integer=257, max_points::Integer=32,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
        solve_kw...)
    fmin,fmax,n_eval,max_points=_planar_abs_parameters(fmin,fmax,rel_tol,n_eval,max_points)
    n=length(prob.ports)
    storage = _checked_payload_sum("physical sweep storage",
        _planar_sweep_storage_bytes(n,n_eval,max_points),
        _checked_array_payload_bytes(ComplexF64,8,n,n),
        _checked_array_payload_bytes(ComplexF64,3,n))
    _enforce_payload_limit(storage, max_bytes, "planar sweep", "max_bytes")
    _planar_abs_frequency_grid(fmin,fmax,n_eval)
    remaining = _validated_resource_limit("max_bytes", max_bytes) - storage
    options = merge((retain_matrix=false, max_bytes=remaining), (; solve_kw...))
    refs=_planar_reference_values([port.z0 for port in prob.ports],n;freq=fmin)
    response(f)=begin
        result=solve_planar(prob,f;options...)
        planar_renormalize_s(result.s,result.z0,refs)
    end
    return _planar_sweep_abs(response,n,fmin,fmax;rel_tol,n_eval,
        max_points,max_bytes,z0=refs)
end

function _planar_sweep_storage_bytes(nports::Integer, n_eval::Integer,
        max_points::Integer)
    return _checked_payload_sum("sweep storage",
        _checked_array_payload_bytes(ComplexF64, n_eval, nports, nports),
        _checked_array_payload_bytes(ComplexF64, max_points, nports, nports),
        # Full/reduced interpolation coefficients and their data.
        _checked_array_payload_bytes(ComplexF64, 6, max_points, nports, nports),
        # One sequential Loewner/SVD construction, factors and numerical
        # workspace; stored entry models remain linear in sample count.
        _checked_array_payload_bytes(ComplexF64, 12, max_points, max_points),
        _checked_array_payload_bytes(Float64, 3, n_eval),
        _checked_array_payload_bytes(Float64, 3, max_points))
end

"""`planar_sweep_abs(response, fmin, fmax; nports, kw...)` adaptively
samples a callable returning a finite `nports × nports` S-matrix. This
shares the interpolation path with the EM solver and can sweep calibrated
networks or combined circuit/EM responses."""
function planar_sweep_abs(response::Function, fmin::Real, fmax::Real;
        nports::Integer, kw...)
    return _planar_sweep_abs(response, Int(nports), fmin, fmax; kw...)
end

function _planar_abs_parameters(fmin,fmax,rel_tol,n_eval,max_points)
    isfinite(fmin) && isfinite(fmax) && 0<fmin<fmax ||
        throw(ArgumentError("sweep band needs finite 0 < fmin < fmax"))
    lo,hi=try
        Float64(fmin),Float64(fmax)
    catch
        throw(ArgumentError("sweep band must be representable in Float64"))
    end
    isfinite(lo) && isfinite(hi) && 0<lo<hi ||
        throw(ArgumentError("sweep band needs finite distinct positive Float64 endpoints"))
    lo<lo+(hi-lo)/2<hi ||
        throw(ArgumentError("sweep band has no distinct Float64 midpoint"))
    isfinite(rel_tol) && rel_tol>0 ||
        throw(ArgumentError("rel_tol must be finite and positive, got $rel_tol"))
    ne,mp=Int(n_eval),Int(max_points)
    ne>=8 || throw(ArgumentError("n_eval must be >= 8, got $ne"))
    mp>=3 || throw(ArgumentError("max_points must be >= 3, got $mp"))
    return lo,hi,ne,mp
end

# Validate the lazy grid after storage preflight and before reference or
# response callbacks. A fixed Float64 result cannot represent repeated
# candidate frequencies as distinct analysis samples.
function _planar_abs_frequency_grid(fmin::Float64,fmax::Float64,n_eval::Int)
    grid=range(fmin,fmax;length=n_eval)
    previous=first(grid)
    for k in 2:n_eval
        current=grid[k]
        isfinite(current) && current>previous ||
            throw(ArgumentError("sweep candidate grid needs distinct finite Float64 frequencies"))
        previous=current
    end
    return grid
end

function _planar_sweep_abs(response, nports::Int,
        fmin::Real, fmax::Real; rel_tol::Real=1e-2,
        n_eval::Integer=257, max_points::Integer=32,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,z0=nothing)
    fmin,fmax,n_eval,max_points=_planar_abs_parameters(fmin,fmax,rel_tol,n_eval,max_points)
    nports >= 1 || throw(ArgumentError("nports must be positive"))
    _enforce_payload_limit(
        _checked_payload_sum("sweep reference storage",
            _planar_sweep_storage_bytes(nports,n_eval,max_points),
            _checked_array_payload_bytes(ComplexF64,2,nports)),
        max_bytes, "planar sweep", "max_bytes")
    dense=collect(_planar_abs_frequency_grid(fmin,fmax,n_eval))
    refs=z0===nothing ? nothing : _planar_reference_values(z0,nports)
    midpoint = fmin + (fmax - fmin) / 2
    freqs = Float64[fmin, midpoint, fmax]
    function analyze(f)
        S = response(f)
        S isa AbstractMatrix && size(S) == (nports, nports) ||
            throw(ArgumentError("sweep response must return a $(nports)x$(nports) S-matrix"))
        all(isfinite, S) || throw(ArgumentError("sweep response contains non-finite entries"))
        stored=Matrix{ComplexF64}(S)
        all(isfinite,stored) || throw(ArgumentError("sweep response is not finite in ComplexF64 storage"))
        return stored
    end
    svals = Matrix{ComplexF64}[analyze(f) for f in freqs]
    converged = false
    est_err = fill(Inf, n_eval)
    dense_s = [Matrix{ComplexF64}(undef, nports, nports) for _ in 1:n_eval]
    entries = [(p, q) for p in 1:nports for q in 1:nports]
    scaled = f -> 2 * ((f - fmin) / (fmax - fmin)) - 1

    while true
        n = length(freqs)
        xs = scaled.(freqs)
        # full model vs the model missing the newest analysis point
        hi = [_sweep_model(xs, [s[p, q] for s in svals])
              for (p, q) in entries]
        lo = [_sweep_model(xs[1:(n - 1)], [svals[k][p, q] for k in 1:(n - 1)])
              for (p, q) in entries]
        fill!(est_err, 0.0)
        @inbounds for j in 1:n_eval
            f = scaled(dense[j])
            worst = 0.0
            fill!(dense_s[j], zero(ComplexF64))
            for (e, (p, q)) in enumerate(entries)
                vh = _sweep_eval(hi[e], f)
                dense_s[j][p, q] = vh
                isfinite(real(vh)) && isfinite(imag(vh)) ||
                    (worst = Inf; continue)
                vl = _sweep_eval(lo[e], f)
                isfinite(vl) || (worst = Inf; continue)
                d = abs(vh - vl) / max(abs(vh), 1e-12)
                d > worst && (worst = d)
            end
            est_err[j] = worst
            known = findfirst(==(dense[j]), freqs)
            if known !== nothing
                copyto!(dense_s[j], svals[known])
                est_err[j] = 0.0
            end
        end
        # never pick an already-analyzed frequency (interpolated exactly)
        worst_err, jstar = 0.0, 0
        for j in 1:n_eval
            dense[j] in freqs && continue
            est_err[j] > worst_err &&
                (worst_err = est_err[j]; jstar = j)
        end
        jstar == 0 &&
            (converged = all(e -> e <= rel_tol, est_err); break)
        worst_err <= rel_tol && (converged = all(e -> e <= rel_tol, est_err); break)
        # the cap bounds the number of EM analyses; the current model
        # still reports its dense evaluation before giving up
        length(freqs) >= max_points && break
        push!(freqs, dense[jstar])
        push!(svals, analyze(dense[jstar]))
    end
    return PlanarSweep(freqs,svals,dense,dense_s,est_err,converged,refs)
end

# ---------------- box-resonance pole scan ----------------

"""One natural resonance of the shielded stackup: transverse mode numbers
`(m, n)` of the box lattice, polarization, and resonant frequency [Hz]."""
struct PlanarResonance
    freq::Float64
    m::Int
    n::Int
    pol::PlanarPol
end

# transverse mode existence mask: mirrors the norm^2 factors used by the
# assemble loop (norm zero marks a mode that does not exist under the
# sidewall parity -- PEC TM needs m,n >= 1; PMC TE needs m,n >= 1 and PMC
# TM drops only the uniform mode)
function _mode_exists(walls::SidewallKind, a::Float64, b::Float64,
        m0::Int, n0::Int, pol::PlanarPol)
    kx, ky = m0 * pi / a, n0 * pi / b
    ic = m0 == 0 ? a : a / 2
    isv = m0 == 0 ? 0.0 : a / 2
    jc = n0 == 0 ? b : b / 2
    jsv = n0 == 0 ? 0.0 : b / 2
    nte2, ntm2 = if walls == WALL_PEC
        (ky * ky * ic * jsv + kx * kx * isv * jc,
         kx * kx * ic * jsv + ky * ky * isv * jc)
    else
        (ky * ky * isv * jc + kx * kx * ic * jsv,
         kx * kx * isv * jc + ky * ky * ic * jsv)
    end
    kc2 = kx * kx + ky * ky
    return pol === TE_POL ? (kc2 != 0.0 && nte2 != 0.0) : ntm2 != 0.0
end

# Propagate a bottom-boundary voltage/current state and test the top
# boundary. Unlike zdn+zup this characteristic residual remains finite at
# PMC covers and at impedance poles, and does not depend on observation
# interface. Layer coefficients share the stable cutoff/evanescent path.
function _resonance_residual(stack::PlanarStackup, omega::Number,
        kc2::Float64, pol::PlanarPol, iface::Int)
    zref = sqrt(_MU0 / _EPS0)
    zb = _planar_term_impedance(stack.bottom, pol, omega, kc2)
    v, qi = _planarisinf(zb) ? (1.0 + 0im, 0.0 + 0im) :
                              (ComplexF64(zb / zref), -1.0 + 0im)
    for layer in stack.layers
        a, b, c, _ = _planar_layer_abcd(pol, omega, kc2, layer)
        v, qi = a * v - (b / zref) * qi, -(c * zref) * v + a * qi
        norm = max(abs(v), abs(qi), floatmin(Float64))
        v /= norm
        qi /= norm
    end
    zt = _planar_term_impedance(stack.top, pol, omega, kc2)
    return _planarisinf(zt) ? qi : (v - (zt / zref) * qi) /
        max(1.0, abs(zt / zref))
end

"""
    planar_box_resonances(stack, fmin, fmax; kw...) -> Vector{PlanarResonance}

Scan the shielded stackup for natural resonances in `[fmin, fmax]`: for
every existing transverse mode `(m, n, pol)` with `m <= mmax`,
`n <= nmax`, find frequencies where the parallel modal impedance
the bottom and top boundary conditions agree. Candidates are refined by
bracketed bisection/golden-section on a bounded characteristic residual.
Lossless PEC, PMC and reactive covers have real resonances; lossy/open
stacks only report a real-frequency candidate when its residual meets
`rtol`. Complex resonance frequencies are not extracted by this scan.

Keywords: `walls=WALL_PEC`, `mmax=8`, `nmax=8`, `nsamp=1024` scan points,
`rtol=1e-9` residual acceptance relative to the modal scale.
"""
function planar_box_resonances(stack::PlanarStackup{T},
        fmin::Real, fmax::Real; walls::SidewallKind=WALL_PEC,
        mmax::Integer=8, nmax::Integer=8, nsamp::Integer=1024,
        rtol::Real=1e-9, iface::Integer=length(stack.layers)) where {T}
    planar_validate(stack)
    isfinite(fmin) && isfinite(fmax) && 0 < fmin < fmax ||
        throw(ArgumentError("resonance band needs 0 < fmin < fmax"))
    mmax >= 0 && nmax >= 0 ||
        throw(ArgumentError("mode caps must be nonnegative"))
    nsamp = Int(nsamp)
    nsamp >= 16 ||
        throw(ArgumentError("nsamp must be >= 16, got $nsamp"))
    isfinite(rtol) && rtol > 0 || throw(ArgumentError(
        "rtol must be finite and positive"))
    L = length(stack.layers)
    0 <= iface <= L ||
        throw(ArgumentError("iface must be in 0:$L, got $iface"))
    fs = collect(range(fmin, fmax; length=nsamp))
    out = PlanarResonance[]
    @inbounds for pol in (TE_POL, TM_POL), n0 in 0:Int(nmax),
            m0 in 0:Int(mmax)
        _mode_exists(walls, stack.a, stack.b, m0, n0, pol) || continue
        kx, ky = m0 * pi / stack.a, n0 * pi / stack.b
        kc2 = kx * kx + ky * ky
        r = [let om = 2pi * f
                 _resonance_residual(stack, om, kc2, pol, Int(iface))
             end for f in fs]
        scale = maximum(abs, r)
        isfinite(scale) && scale > 0 || continue
        for j in 2:(nsamp - 1)
            # sign change of the reactive part (lossless stacks carry a
            # purely imaginary residual) or a strict |r| local minimum
            bracket = if sign(imag(r[j - 1])) != sign(imag(r[j + 1])) &&
                         imag(r[j - 1]) != 0 && imag(r[j + 1]) != 0
                (fs[j - 1], fs[j + 1])
            elseif abs(r[j]) < abs(r[j - 1]) && abs(r[j]) < abs(r[j + 1]) &&
                    abs(r[j]) < 0.02 * scale
                (fs[j - 1], fs[j + 1])
            else
                nothing
            end
            bracket === nothing && continue
            lo, hi = bracket
            fres = if sign(imag(r[j - 1])) != sign(imag(r[j + 1]))
                _bisect_residual(stack, kc2, pol, Int(iface), lo, hi)
            else
                _golden_residual(stack, kc2, pol, Int(iface), lo, hi)
            end
            fres === nothing && continue
            rstar = _resonance_residual(stack, 2pi * fres, kc2, pol,
                Int(iface))
            abs(rstar) <= rtol * scale || abs(rstar) < 1e-8 * scale ||
                continue
            any(o -> abs(o.freq - fres) <= 1e-6 * (fmax - fmin) &&
                     o.m == m0 && o.n == n0 && o.pol == pol, out) ||
                push!(out, PlanarResonance(fres, m0, n0, pol))
        end
    end
    sort!(out; by=o -> o.freq)
    return out
end

function _bisect_residual(stack, kc2, pol, iface, lo, hi)
    flo = imag(_resonance_residual(stack, 2pi * lo, kc2, pol, iface))
    for _ in 1:80
        mid = (lo + hi) / 2
        fmid = imag(_resonance_residual(stack, 2pi * mid, kc2, pol, iface))
        (isfinite(flo) && isfinite(fmid)) || return nothing
        if sign(fmid) == sign(flo)
            lo, flo = mid, fmid
        else
            hi = mid
        end
        hi - lo < 1e-12 * max(1.0, lo) && break
    end
    return (lo + hi) / 2
end

function _golden_residual(stack, kc2, pol, iface, lo, hi)
    phi = (sqrt(5.0) - 1.0) / 2
    x1, x2 = hi - phi * (hi - lo), lo + phi * (hi - lo)
    f1 = abs(_resonance_residual(stack, 2pi * x1, kc2, pol, iface))
    f2 = abs(_resonance_residual(stack, 2pi * x2, kc2, pol, iface))
    for _ in 1:120
        (isfinite(f1) && isfinite(f2)) || return nothing
        hi - lo < 1e-12 * max(1.0, lo) && break
        if f1 < f2
            hi, x2, f2 = x2, x1, f1
            x1 = hi - phi * (hi - lo)
            f1 = abs(_resonance_residual(stack, 2pi * x1, kc2, pol, iface))
        else
            lo, x1, f1 = x1, x2, f2
            x2 = lo + phi * (hi - lo)
            f2 = abs(_resonance_residual(stack, 2pi * x2, kc2, pol, iface))
        end
    end
    return (lo + hi) / 2
end
