# Finite rational evaluation and exact IEEE narrowing

Both Julia versions reproduce three finite-input range errors in the
previous evaluation: a partial sum becomes Inf even though the final
response is maximum finite; a denominator difference exceeds storage
and erases a finite real component; individually rounded-away half-
subnormal terms lose a representable subnormal sum. These are real
stable models with finite stored coefficients and finite frequencies.

Keep ordinary arithmetic and allocation. Recompute only nonfinite
intermediates/results or zero components with potentially nonzero
numerators. Structural zeros retain the ordinary path. Exact dyadic
integer arithmetic shares four pole coordinate/divisor/weight vectors
across port entries and fourteen named scalar buffers. A common positive
denominator replaces repeated rational reductions. Caller matrices and
pole/residue coefficients are borrowed without mutation.

Direct integer quotient/remainder quantization selects the IEEE neighbor
with nearest ties to even, including subnormals and true overflow.
Five reused scalar roles grow from actual numerator/denominator bit
counts and the IEEE subnormal spacing. No selected BigFloat precision
or caller-state change narrows the IEEE result. Dyadic entry range,
polynomial degree, pole count and output lattice derive integer capacity
and raw payload before allocation. Ordinary evaluation remains 144
allocated bytes in the tested two-port case. The tested correct rare
fallback drops from 64656 rational-prototype bytes to 10440 integer-
kernel bytes on each Julia version. These are scoped measurements,
excluding object headers/opaque library workspace from the raw budget;
they do not prove universal optimization or peak RSS.

Non-IEEE big/rational frequency fallback retains exact rationals and
caller output type. The original natural MPFR exponent-range boundary
still raises ArgumentError before an enormous integer expansion. That
generic path has separate resource/performance acceptance still open.
V1/V2 prototypes changed the original exponent-limit behavior; their
failed scoped controls remain preserved, and V3 restored the contract.
The first new private-helper inference assertion incorrectly excluded
its legitimate Nothing pole sentinel; the revised test permits that
documented optional return while original public inference gates remain.

All 1345 parent scoped checks plus 69 independent closed forms, 182
independent IEEE midpoint/neighbor/state checks and 8 derived-resource,
buffer-identity and caller-ownership checks pass on both Julia versions
(1604 each). Every original scientific tolerance is unchanged. The
durable tests use package-compatible isolated modules and no output
artifact is required by Pkg.test.

This commit descends from the current public-GMP exact-storage parent
aef. Native, docs, static, import, proper, complete, current parent hosted
CI, fresh clean-tree and main/own-CI qualification is still required.
Do not publish unpublished known-bad intermediate heads. The broad
Sonnet/RFIC, numeric and memory/performance audit remains active.
