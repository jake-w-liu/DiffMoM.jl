# Actual ngspice47 linear transmission-line evidence

These are exact hand-authored linear libraries, loaded-wave excitation decks,
engine logs and original 17-digit voltage/current output. `sha256.toml` lists
272 exact files and the ngspice47 executable/manual hashes. The thirteen
cases cover T delay, F/NL, zero delay, LTRA RLC/RC/LC/RG, nested parameters,
forward/global/local model lookup, local shadowing and transient controls.
Three additional mixed networks verify forward F/H voltage-source controls
and K inductor branch numbering when a preceding line owns two currents.

Each AC case has 23 frequencies from 1 MHz to 12 GHz, unequal 50/75 ohm
loads and two independently shifted return potentials. Seven cases also
retain actual DC engine output. No production DiffMoM helper generated
the reference voltages or currents.

The default RG engine stamp multiplies its B and C coefficients by
`1+gmin`; at large attenuation this breaks reciprocity. Default, zero and
small-gmin output is retained separately. A 512-bit independent equation
oracle matches the default perturbed stamp within 5.43e-14. The physical
comparison uses explicit `gmin=0` and retains the same 1e-10 full-S gate;
its largest error is 9.23e-13. This is a native numerical-regularization
finding, not a change to DiffMoM physics or to the gate.

This proves the explicit small-signal SPICE T/O-LTRA subset. It does not
certify arbitrary Spectre line grammar, nonlinear/vendor models, transient
initial conditions, URC/W/multiconductor line cards or arbitrary RLCG.
