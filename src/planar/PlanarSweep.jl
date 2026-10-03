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
# The interpolant is a Thiele continued fraction (reciprocal-difference
# form), the closed-form equivalent of the Pade quotient R(f) =
# (a0 + a1 f + ...)/(1 + b1 f + ...) without the f^N Vandermonde matrix
# that paper warns is ill-conditioned for large N.
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
end

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
"""
function planar_sweep_abs(prob::PlanarProblem,
        fmin::Real, fmax::Real; rel_tol::Real=1e-2,
        n_eval::Integer=257, max_points::Integer=32,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
        solve_kw...)
    isfinite(fmin) && isfinite(fmax) ||
        throw(ArgumentError("sweep band must be finite"))
    0 < fmin < fmax ||
        throw(ArgumentError("sweep band needs 0 < fmin < fmax, " *
            "got [$fmin, $fmax]"))
    0 < rel_tol ||
        throw(ArgumentError("rel_tol must be positive, got $rel_tol"))
    n_eval = Int(n_eval)
    n_eval >= 8 ||
        throw(ArgumentError("n_eval must be >= 8, got $n_eval"))
    max_points = Int(max_points)
    max_points >= 3 ||
        throw(ArgumentError("max_points must be >= 3, got $max_points"))
    nports = length(prob.ports)
    _enforce_payload_limit(
        _checked_array_payload_bytes(ComplexF64, n_eval * max_points,
            max(nports * nports, 1); label="sweep storage"),
        max_bytes, "planar sweep", "max_bytes")

    dense = collect(range(fmin, fmax; length=n_eval))
    freqs = Float64[fmin, (fmin + fmax) / 2, fmax]
    svals = Matrix{ComplexF64}[solve_planar(prob, f; solve_kw...).s
                               for f in freqs]
    converged = false
    est_err = fill(Inf, n_eval)
    dense_s = Matrix{ComplexF64}[]

    while true
        n = length(freqs)
        entries = [(p, q) for p in 1:nports for q in 1:nports]
        # full model vs the model missing the newest analysis point
        hi = [_thiele_build(freqs, [s[p, q] for s in svals])
              for (p, q) in entries]
        lo = [_thiele_build(freqs[1:(n - 1)], [s[p, q] for s in svals[1:(n - 1)]])
              for (p, q) in entries]
        dense_s = [Matrix{ComplexF64}(undef, nports, nports)
                   for _ in 1:n_eval]
        fill!(est_err, 0.0)
        @inbounds for j in 1:n_eval
            f = dense[j]
            worst = 0.0
            fill!(dense_s[j], zero(ComplexF64))
            for (e, (p, q)) in enumerate(entries)
                vh = _thiele_eval(hi[e], f)
                dense_s[j][p, q] = vh
                # a truncated (degenerate) model does not reach every
                # sample: its estimate is untrusted, force refinement
                (hi[e].ndeg < hi[e].nfull ||
                 lo[e].ndeg < lo[e].nfull) && (worst = Inf; break)
                isfinite(real(vh)) && isfinite(imag(vh)) ||
                    (worst = Inf; continue)
                vl = _thiele_eval(lo[e], f)
                d = abs(vh - vl) / max(abs(vh), floatmin(Float64))
                d > worst && (worst = d)
            end
            est_err[j] = worst
        end
        # never pick an already-analyzed frequency (interpolated exactly)
        worst_err, jstar = 0.0, 0
        for j in 1:n_eval
            any(fk -> abs(dense[j] - fk) <= 0.5 * (fmax - fmin) / n_eval,
                freqs) && continue
            est_err[j] > worst_err &&
                (worst_err = est_err[j]; jstar = j)
        end
        jstar == 0 &&
            (converged = all(e -> e <= rel_tol, est_err); break)
        worst_err <= rel_tol && (converged = true; break)
        # the cap bounds the number of EM analyses; the current model
        # still reports its dense evaluation before giving up
        length(freqs) >= max_points && break
        push!(freqs, dense[jstar])
        push!(svals, solve_planar(prob, dense[jstar]; solve_kw...).s)
    end
    return PlanarSweep(freqs, svals, dense, dense_s, est_err, converged)
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

# parallel-impedance residual zdn + zup at the reference interface; a zero
# is a natural resonance.  Interface L (top of stack) is used; the zero
# set is interface-independent for a lossless stack.
function _resonance_residual(stack::PlanarStackup, omega::Number,
        kc2::Float64, pol::PlanarPol, iface::Int)
    casc = planar_mode_cascade(stack, omega, kc2, pol)
    return casc.zdn[iface + 1] + casc.zup[iface + 1]
end

"""
    planar_box_resonances(stack, fmin, fmax; kw...) -> Vector{PlanarResonance}

Scan the shielded stackup for natural resonances in `[fmin, fmax]`: for
every existing transverse mode `(m, n, pol)` with `m <= mmax`,
`n <= nmax`, find frequencies where the parallel modal impedance
`zdn || zup` diverges (`zdn + zup -> 0`).  Candidates are refined by
bracketed bisection/golden-section on the residual.

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
