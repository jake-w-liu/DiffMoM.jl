# Reviewed excitation storage range guard

The actual native attachment project and its raw solved geometry accepted a
finite BigFloat voltage of 2^-1075, then silently stored it as zero. Independent
high precision scaling of their unit-voltage current fields produces 32
nonzero, representable ComplexF64 entries; both original routes returned none.
The before capture and immutable source hashes are retained under ignored
data/sonnet_current_excitation_intake_probe_v1_20261005.toml.

A shared scalar conversion rejects nonfinite storage and the loss of any
nonzero real or imaginary component. Current, radiation and power-wave entry
points use it before transferring supplied excitations. This declares the
existing ComplexF64 storage domain explicitly; it does not implement a higher
precision solver or certify all subnormal arithmetic inside field transfers.
The vector conversion retains one-based output and its original single output
allocation. Ordinary, zero and representable subnormal scalar inputs remain
valid. The scalar guard has zero warmed allocation.

Both Julia 1.13.1 and 1.12.7 focused runs pass the new native/raw range and
independent linearity controls, plus the original current-map, power-wave,
calibration, subdivision and radiation suites. Full frozen tests and GitHub
verification remain required before publishing. Original gates, fixtures,
workflow and duplication ceiling are unchanged.

The pinned line counter reports 212743 code lines, a reviewed increase of 53
from 212690 for the guard, integrations and registered regressions. The line
ceiling adjustment is recorded separately. Blocking smells and the bounded
400-group duplication census pass; that census is not exhaustive coverage.
