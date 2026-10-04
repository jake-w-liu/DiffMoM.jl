# Independent remaining Library domain proofs

This archive records the independent review of `PlanarLibrary.jl` after the
earlier stored-domain fixes. It preserves that earlier archive without changing
any file in `../library_stored_domain`.

## Captured source versions

- `source_before.jl`: exact SHA-256
  `b69db254f721e9c0b57637b36849f1184619407859439ab559977f6f914cdab3`.
  This is also the exact earlier archive's `source_after.jl`.
- `source_after.jl`: exact SHA-256
  `54d3328bb16cf70bb4c4597bebc7e9172e52ea7ad3619dbcd2ee8196c747ae53`.
- `ASCENT_LIBRARY_LICENSE` preserves the MIT attribution accompanying the
  adapted source snapshots.

`proofs.toml` maps successful reports to these captured versions. Its two
intermediate reports explicitly record missing complete source snapshots; no
reconstructed source is credited for them. `sha256.toml` hashes every other
file in this archive, including this description and the source mapping.

## Confirmed failures and corrected checks

1. Broadside placement accepted a nonzero BigFloat lateral offset that became
   zero in Float64. Original failure assertions: 24, also covering stale widths.
2. Transformed pins retained widths, midpoints and directions that contradicted
   their actual emitted polygon edges. The isolated archived-source replay
   retains midpoint and direction witnesses with two successful assertions.
3. An ordinary air bridge with `via_margin=prevfloat(0.00005)` accepted via
   polygons with only two distinct vertices. The archived replay also confirms
   capacitance failures below, with 14 successful original-failure assertions.
4. A capacitance reference rejected a representable finite result because an
   intermediate dielectric thickness/permittivity ratio became zero or infinity.
   Independent expected values use 4096-bit arithmetic in the corrected replay.
5. An intermediate pin correction used `a/2+b/2`; each half of a minimum
   subnormal offset became zero. Its original report retains the exact hash
   `be566fa5838091fad9a009ecab6a31924131352abfb70cb1bc216436be31d96b`
   and six successful assertions. **The complete intermediate source was not
   captured.** This report proves the observed contradiction, not the complete
   behavior of that source version.

The final public-path independent proof passes 1,109 assertions: 7 broadside
storage checks, 800 pin edge/normal/exact rational midpoint checks, 14 via
geometry checks and 288 capacitance cases against a 4096-bit oracle. Its final
separate assertion verifies the live source did not change during that run.
The registered test replays these contracts without writing to the archive.

## Intermediate and discarded probes

- `library_air_bridge_capacitance_before.*` reports source hash
  `db883488fbdd425699b17c11521272dd574ab93b0a456002137fc25e304d0bd1`.
  That complete intermediate source was not captured. The corrected isolated
  archived replay, whose report maps to the captured `b69db254...` bytes, is
  the credited original source proof for these failures.
- `library_pin_geometry_before.log` is a raced harness: the package loaded an
  already corrected pin implementation, produced no old contradictory rows,
  and failed its two assertions that expected the old bug. No source hash was
  recorded. It is retained as a harness disposition, not a production failure.
- `library_pin_midpoint_subnormal_probe.log` prints the midpoint witness, then
  fails because the command omitted importing `SHA`. The separate six-check
  clean midpoint report is the credited observation.
- The translation probe also passes invalid values to the capacitance helper's
  unused `f_hz` keyword. The formula uses a static stack and is frequency
  independent; this was not classified as incorrect numerical output.
- The review noticed range-sensitive norm calculations in other constructors.
  It did not prove an accepted invalid geometry from that candidate; rejected
  extreme geometry is not credited as an additional confirmed bug here.

The scripts are preserved as the exact bytes used in the original validation
location and keep their original relative paths and report-writing statements.
Run `test/test_planar_library_remaining_proof.jl` to check integrity and repeat
the physical contracts. This archive makes no native renderer or full Sonnet
parity claim.
