# Actual Sonnet D-Pack reference

Nine fresh Sonnet Lite 18.53 raw four-port geometry runs use 16/32/64
actual cells and 1/5/10 GHz. Imported ODB D-Pack pads plus two buses are
compared with independently declared native rectangular protrusions and
buses. Native geometry is never generated from imported masks.

The original gates are 0.06 in complete complex S and `1e-9` in the
original voltage residual. All 39 checks passed: maximum full S error
`6.0981800093341824e-5`, maximum residual `1.0686446699108635e-10`.
This accepts the declared physical GEO workflow; native ODB translator
compatibility and mesh convergence remain separate requirements.

The first overlapping-wall-pad coupon failed the native port-count limit
and remains archived separately as `odb_dpack_native_MiuIyy`. The revised
pads stay away from walls and touch bus polygons without overlap.

Raw projects, engine logs, full matrices, selected raw-log guard outcomes,
metadata, comparison and source hashes are preserved byte for byte.
Reproduction: `validation/sonnet_stripline/validate_sonnet_odb_dpack.jl`.
The manifest hashes all files except itself.
