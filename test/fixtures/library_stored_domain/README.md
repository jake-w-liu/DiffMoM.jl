# RFIC library stored-domain and placement proof

The original public reference APIs accepted positive BigFloat inputs that became zero or infinity in Float64. Finite spiral inputs also returned zero or infinity when intermediate squares or diameter sums exceeded the floating range, despite a representable reference value. Valid BigFloat rotation angles caused a method error; wide offsets and finite translations that collapsed the polygon were accepted.

The fixed library validates stored scalar values, evaluates reference products with separate binary exponents, and validates placed polygons while preserving their emitted vertex order and pin edge indices. The reference product has zero warmed Julia allocations. Tests independently evaluate the documented spiral equations at 256-bit precision, cover all declared shape/method coefficients and wide ranges, and retain capacitance, ordinary placement and mirror controls.

Original/final source hashes and sixteen public reproduction rows are retained alongside the 1004 numerical/domain/allocation checks and the completed neighboring library/layout tests. The native IDC proof separately rechecks current public placement and solving against actual Sonnet outputs. These archives establish the stated checks; full RFIC library and Sonnet parity remain separate completion work.
