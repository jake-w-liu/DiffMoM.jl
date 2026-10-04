# Native SPARAM model-file fixtures

`manifest.json` preserves SHA256 and original paths for twelve exact files.
The native manufactured coupon, known series-resistor device model, IDEAL
control, actual native control response and engine/license logs are retained
without relabeling. The engine explicitly rejects SPARAM data-file components
under Sonnet Lite; its IDEAL control succeeds. This is a license diagnosis,
not a native SPARAM response reference.

Two installed example/model pairs retain the observed automatic AUTO/FEED
schema (`amp`) and the unresolved custom-width CUST40 schema (`7GHz_amp`).
The test independently generates an unequal passive two-pin model on the
same coupon and compares raw physical EM+device attachment against manual
MNA wiring at knots and interior frequencies. Model pin indices, geometry
labels, global common-return shunts and complex reference bases are distinct.

The historical validation prototype and failed/accepted comparison logs are
under `validation/planar_audit/native_sparam_adapter*`. The final prototype
passes145 checks with maximum full-S delta4.441434166503682e-16. Production
checks add public construction, provenance, caller ownership, resource and
transactional/provider tests. Complete native pin-group calibration and
licensed vendor rendering compatibility remain unverified.
