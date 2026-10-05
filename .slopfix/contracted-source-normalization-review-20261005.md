# Reviewed contracted FFT source normalization

Both Julia 1.13.1 and 1.12.7 reproduce a public contracted FFT failure for
finite, independent voltage contracts scaled by 1e-16 and 1e-18. Krylov stops
after one iteration with a recomputed physical voltage residual of 0.602637,
above the unchanged 1e-9 gate. Dense solves and independently normalized FFT
solves pass that gate and agree on physical currents. The corrected before
captures retain all 139 actual source hashes. An earlier capture with an
incorrect independent wall-source sign remains archived as a harness error.

The FFT solve divides its source by the largest source component using the
existing residual buffer, then restores the physical current scale before
recomputing the original full-operator residual. No array is added and the
original iteration limit, tolerance and payload reservation remain unchanged.
Finite positive and negative source scales are checked against an independent
dense matrix, physical wall RHS and reference-impedance transformation, with
and without preconditioning. A one-iteration budget must still fail the gate.

Both supported Julia versions pass all eight focused suites, including 49
new amplitude-invariance and iteration-budget assertions. The initial focused
runner omitted required TMM fixture producers; its failure logs and the
subsequent runner setup failure are preserved separately from passing results.
The pinned line counter measures 212781 code lines, a reviewed increase of 38
from 212743. The line ceiling adjustment is recorded in a separate commit;
the duplication ceiling and original workflow are unchanged. Full frozen tests
and GitHub verification remain required before publishing. This change does
not establish complete Sonnet native parity or certify every floating-point
range.
