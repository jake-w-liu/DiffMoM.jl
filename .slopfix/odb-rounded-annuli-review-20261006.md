# Rounded ODB++ annulus implementation review

The importer rejected the rounded square and rounded rectangular annuli listed in Siemens ODB++ 8.1 Update 3, printed page 200. Both families now support optional corner selectors through the existing rectangle and transparent-composite geometry. The inner radius follows the concentric parallel-wall construction shown in the primary illustrations: max(outer radius - wall width, 0). Zero-sized holes become empty geometry.

Primary source: https://odbplusplus.com/wp-content/uploads/sites/2/2021/02/odb_spec_user.pdf (printed pp. 36, 200 and 214).

The isolated implementation passes 18,857 geometry, scaling, corner, orientation, polarity and resource checks plus ten raster/analytic-area/swept-line checks on Julia 1.13.1 and 1.12.7. The original three ODB suites retain all 249,045 assertions. Repeated membership allocates zero bytes in five warmed samples; the two tested shapes have 80-byte owned payload and two fixed children. Dimensions, radii and duplicate corner lists reject before drawing.

The v1 resource harness used a nonconstant module alias and failed its allocation assertion. The unchanged implementation passes the corrected v2 harness. Both original failed logs remain preserved. Full committed-source, hosted CI and native EM acceptance are separate requirements. This change closes these two symbol-family implementation gaps; it does not complete other thermal/stencil symbols, dimensional resizing, barcodes or the broader Sonnet/RFIC plan.
