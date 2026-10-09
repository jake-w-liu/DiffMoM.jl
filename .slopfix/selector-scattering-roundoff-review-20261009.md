# Stored scattering and physical wave accuracy

Current main8102 failed the unchanged radiation mapping assertion at
1e-13 on macOS and all three Linux configurations. Windows passed.
Actual macOS arithmetic diagnostic37876311027 preserved the operands:
the roundoff-only selector replaced ordinary stored-S voltage by a
relative2.8e-15 change; coefficient cancellation amplified the field
difference to1.15e-12. Declared-S contraction matched the original oracle.

The selector now also accepts the existing independent Kurokawa incident
wave accuracy requirement3e-14 from test_planar_power_waves.jl. This is an
existing integration requirement, not a universal conditioned-voltage
forward-error theorem or a newly fitted tolerance. Derived roundoff and
least-subnormal bounds remain for zero drives. Quantized cases outside
the physical wave requirement still recover from retained admittance.

All445 unchanged nearby assertions and54 captured-bit/adjacent-coordinate
regressions pass on both local Julia versions. The original physical,
field, ownership, inference, normalized-range and quantization gates stay.
Actual four-platform branch validation, complete suites, static gates,
main publication and own full hosted CI require separate qualification.
Historical failure logs and candidate commits remain preserved.
