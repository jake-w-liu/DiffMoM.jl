# Actual ngspice 47 admittance references

These unmodified netlists, raw currents and engine logs were produced by
`validation/planar_audit/validate_spice_external.jl` in
`data/sonnet_validation/external_spice_CrHcMO`. Four cases cover two fitted
two-port models at grounded and complex common-mode floating references:
a real pole with affine capacitance and a complex conjugate pole pair.
Each has 37 AC frequencies from 1 MHz to 30 GHz and two independent drives.
Negative ideal-source currents give the complete admittance columns.
All 444 frequency/current comparison checks pass the unchanged `1e-9`
relative gate; maximum complete-matrix relative error is `1.025e-15`.

The engine is the official Windows console ngspice 47, creation date
2026-08-11. `comparison.toml` retains its version, executable SHA-256,
source hashes and generated-subcircuit hashes. The release archive's
published SHA-256 was verified before extraction:
`59225971bd68cdd1199443649aa4615a9e6d684933f205ab49006a3942518f5a`.
Release: <https://sourceforge.net/projects/ngspice/files/ng-spice-rework/47/>.
The engine binaries are external validation tools and are not fixtures.

This accepts these exported real R/C/G/E/F/V realizations. It does not
certify arbitrary vendor or nonlinear SPICE ingestion, measured RFIC
models, or a complete Sonnet workflow. Earlier command-construction,
global-ground and floating-state conditioning failures remain in their
separate audit logs and artifacts.
