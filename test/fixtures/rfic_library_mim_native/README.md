# Actual native public RFIC library MIM proof

The public `planar_mim_capacitor` and two library wall leads form two
independent physical metal interfaces across a 1 um, epsilon=7.5 dielectric,
above a 100 um epsilon=4 substrate and below 100 um of air, with PEC covers.
Native GEO is six independently declared rectangles in millimetres.

Eighteen actual Sonnet Lite cases cover 16/32/64-cell grids, 1/10/20 GHz,
and PEC or a constant 0.1 ohm/square film. The full complex S gate remains
0.06, inherited from the existing native MIM validator; the independently
reconstructed original-voltage residual gate remains 1e-9. Native output
basis guards, commands, logs, matrices, current DiffMoM outputs and twelve
byte-exact final source snapshots are retained.

`initial_eighteen/` preserves the earlier complete eighteen-case proof.
`final_repeat/` repeats it after the Library pin and capacitance-ratio fixes
and the native scalar grammar upgrade. The earlier proof records captured
source hashes; complete byte-exact source snapshots are provided for the
final repeat. No earlier source result is inherited by a later revision.

`initial_mask_oracle_failure/` preserves the original two failed assertions
and passed physical matrix checks. The assertions compared sheet array
positions instead of physical interfaces: the public library sorts sheets,
whereas the importer retains native polygon encounter order. The corrected
oracle compares by interface, without changing production geometry or gates.

This is a bounded raw wall-port finite-grid MIM comparison. Interior pin
calibration, measured-device accuracy, broader loss/technology classes and
continuum convergence remain separate full-parity work.
