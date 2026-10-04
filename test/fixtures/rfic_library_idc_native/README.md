# Actual native RFIC library IDC reference

The public `planar_interdigital_capacitor` constructs two fingers per electrode, joined to explicit library wall leads. Independent native GEO contains eight literal rectangles. All masks and topology agree on 16/32/64 cell grids. Actual Sonnet Lite 18.53 runs at 1/10/20 GHz compare every entry of the raw two-port S matrix for PEC and 0.1 Ω/square resistive film.

`pec_resistive/comparison.toml` records eighteen passing cases under a 0.005 full complex S gate and the unchanged 1e-9 original voltage equation gate. Native outputs are checked against the intended raw log section and reference basis. The source snapshots and all native input/output/metadata files are byte hashed. The initial nine-case PEC proof is retained separately; its original 0.06 gate was tightened for the extended proof.

`library_domain_repeat` retains a fresh eighteen-case native rerun after the Library scalar/reference/placement fixes. Its separately owned source snapshot and unchanged gates prevent the earlier source version from earning credit for later edits.

`placement_ratio_repeat/` retains another eighteen-case native repeat after
the emitted-pin, broadside-offset, via-footprint and capacitance-ratio fixes.
Twelve source snapshots also capture the native scalar grammar and modal
assembly at that run. Its parsed native matrices equal the earlier matrices
exactly. The archive contains 941 hashed files and the current production
replay passes 1377 checks under the same gates.

The two earlier failures were validation harness errors: an incorrect `sonnet_planar_problem` call signature and a nonexistent result residual field. Their logs and native folders remain preserved and earn no solve-accuracy acceptance. The corrected proof independently reconstructs the original unit-voltage wall RHS and checks the retained physical matrix against solved currents.

This is a matched finite-grid EM proof for this IDC and declared constant film. It does not certify continuum convergence, measured RFIC accuracy, other IDC geometries, spiral/MIM/Lange libraries, general conductor loss, or full Sonnet parity.

Reproduce with `julia --project=. --startup-file=no validation/sonnet_stripline/validate_sonnet_library_idc.jl`. The package replay uses the live shared fixture helper and current production solver against these immutable native responses.
