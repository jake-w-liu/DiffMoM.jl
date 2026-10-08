# Convergent small-q conductor transfer relations

The finite-film conductor relations used a fixed third-order polynomial
inside a hard-coded `abs2(q) < 1e-8` window. The truncation error is an
order-`q^4` term, so caller precision beyond roughly `|q|^-4` digits was
lost: at `|q| ~ 6e-5` the public open-back slab impedance carried a
relative error of `3.1e-22` (about `1.8e55` units of the caller's 256-bit
epsilon) where the exact relation is representable to caller precision.

`_planar_conductor_even_series(q)` now accumulates the even `cosh` and
`sinh(x)/x` series until adding a term no longer changes the running sum
in the working arithmetic (nonprogress termination, no fixed order or
cutoff). The series-domain gate is the whole unit disk `abs2(q) <= 1`,
where successive term ratios are bounded by `1/2` and `1/6` and decrease
factorially; a binary64 reference model confirms every observed error
stays inside the exact geometric envelope of the discarded tail plus one
working epsilon per retained term.

For the public layered path, the thin-film ABCD components keep the
original gamma-squared component structure (`a = cosh(x)`,
`series = sinh(x)/x`, `b = i*omega*mu*h*series`, `c = sigma*h*series`),
which preserves the separately representable thin-film terms that a
direct `expm1` rewrite loses (the rejected all-`expm1` variant missed the
original `1e-12` two-sheet gate by `4e-6`).

Outside the unit disk, both the layered path and the internal two-face
relation use the exponent-balanced `expm1` difference instead of
`1 - exp(-2x)`, removing the cancellation when `exp(-2x)` rounds toward
one, and the propagation sign is folded so the exponential never grows.
`_planar_conductor_face_zs` uses the same series pair through the
identities `x*coth(x) = a/series` and `x*csch(x) = 1/series`.

Measured evidence: 256-bit caller errors within about one epsilon on the
public path and the internal relation; Float64 errors bounded by the
`coth`/`csch` conditioning envelope; ForwardDiff partials within derived
bounds; zero allocation per call; per-call latency same order as the
fixed polynomial (observed ratio <= 1.83x); all original tests retained.
