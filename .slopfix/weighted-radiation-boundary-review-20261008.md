# Physical radiation directions and representable weighted fields

The former grazing cosine floor changed the observation direction. Near
grazing, subtracting transverse and total wave-number squares also erased
the axial dispersion of a matched layer, and subtracting PEC image phases
lost their antisymmetric component. Directions now retain their sine and
cosine, matched moments retain the axial wave number, and PEC images use
expm1. The independent original sheet and volume field gates are retained.

At the smallest nonzero Float64 angle, a unit vertical-current reaction
rounded away before weighting by the current and spherical-wave factor.
The factor now weights the linear receiving fields first. Zero or subnormal
via reactions use the physical sine/current binary mantissas, restoring
their exponents after weighting. The guard follows machine representation;
no angle cutoff, chosen precision or new numerical tolerance is introduced.

Both Julia versions passed the expanded boundary cases, original radiation
modules, generic layered receiving controls and archived native-material
matrix at their original gates. Measured warmed allocations on those
boundary fixtures match the preceding main. Broader cancellation/range,
full-suite, documentation, committed checks and hosted CI remain required.
