# Planar Native Projects

Native project readers retain unsupported records as explicit errors.
See the [parity ledger](../advanced/sonnet-parity.md) for implementation
coverage, validation evidence and remaining acceptance work.

`sonnet_component_files` stages SPARAM datasets and literal linear CKT
SPROJ children with explicit PEC AUTO+FEED pins. Project children support
R/L/C, earlier DEF invocations, Touchstone data and recursive PRJ blocks.
They evaluate at the parent requested frequency, retaining INHSWP N/Y;
saved child sweeps are not interpolated. Sources and compiled circuits belong
to the frequency-specific snapshot. Root, dependency, recursion, node,
element and aggregate storage limits apply before optional output-reference
callbacks. Geometry children and parameter bindings need explicit adapters.
This physical attachment does not imply licensed native SMD pin calibration.

Native dielectric layer `RSVY` selects resistivity in ohm centimetres;
the unmarked field is conductivity in siemens per metre. These dielectric
units are distinct from metal `SRVY` and primitive `DIM RES` fields.
Unsupported dielectric loss/anisotropy flags reject.

Native `NOR` loss selectors use `CDVY` (or no selector) for conductivity
in siemens per metre, `RSVY` for resistivity in ohm centimetres and `SRVY`
for sheet DC resistance in ohms per square. Thickness uses `DIM LNG`.
Zero `RSVY`/`SRVY` agrees with native PEC controls; negative or
unrepresentable material values reject before geometry allocation.

The expression variable `FREQ` is always frequency in hertz, independent
of `DIM FREQ`. Native functions `h2p`/`p2h` convert between hertz and project
frequency units; `m2p`/`p2m` convert between metres and project length units.
Each takes one scalar argument.
Native `ln` and `log10` each take one value and compute the logarithm
of its magnitude. Zero/nonfinite inputs reject. `log` is unsupported native
syntax and rejects; the engine's warned zero fallback is preserved as failure
evidence rather than treated as a physical material definition.

## Material expressions

The bounded native grammar preserves parentheses and native left association
of powers: `2^3^2` is 64, while `2^(3^2)` is 512. A negative exponent consumes
its power operand (`2^-3^2` is `2^(-(3^2))`). Actual native rejected syntax,
including comparisons, conditionals and `2^+3^2`, remains unsupported.

Material expressions support the installed mathematical functions:

| Group | Functions |
|---|---|
| Magnitude logarithms | `ln`, `log10`, `db10`, `db20`, `exp` |
| Complex values | `cmplx`, `real`, `imag`, `abs`, `mag`, `conj`, `deg`, `rad` |
| Trigonometry | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2` |
| Hyperbolic functions | `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` |
| General mathematics | `sqrt`, `hypot`, `int`, `fmod`, `min`, `max` |

Trigonometric arguments and inverse outputs use radians independently of
`DIM ANG`. `deg` and `rad` extract complex phase in degrees and radians;
they are not scalar angle-conversion functions. `cmplx(x,y)` computes `x+i*y`
with complex operands, and `hypot(x,y)` computes `hypot(abs(x),abs(y))`.
`int` truncates the real part toward zero. `fmod`, `min` and `max` use signed
magnitudes: magnitude is negative exactly when the operand's real part is
negative. `min` and `max` preserve the selected complex operand and choose
the second operand on ties. Exact-zero `int` and `fmod` results reset to
positive zero. Documented arities are enforced. Complex `atan2`, conversion
function arguments and table keys remain outside the verified evaluator.

Complex intermediate values are supported, with native real-axis branch and
signed-zero phase conventions. Native named quantity variables store the real
projection of a finite defining expression, resetting imaginary signed zero.
Actual SRES controls distinguish `Inner=cmplx(3,4); abs(Inner)` (3) from
`abs(cmplx(3,4))` (5). Finite final quantities also store their real projection:
quoted/bare inline NOR fields `cmplx(3,4)`, `sqrt(-1)` and
`cmplx(3,4)/sqrt(-1)` agree with independently declared resistances 3, 0 and 4.
Nonfinite complex expressions reject before projection. For example,
`imag(sqrt(-4))` is 2 while the final quantity `sqrt(-4)` is 0.
Byte, nesting, AST and
variable-chain bounds are enforced before host stack exhaustion. Graph-only
`FORMAT` and arbitrary host-language syntax are outside this evaluator.

`sonnet_scalar_files` captures the effective project, exact parent SON and
literal-file CSV dependencies before material/geometry/model evaluation.
Native escaped filename quotes are decoded faithfully. `table1` uses linear
interpolation and `table2` bilinear interpolation; their finite axes must be
strictly increasing. Sources, files, nodes, line bytes and estimated storage
have explicit limits, and resolved dependency paths stay inside the selected
root. Snapshots retain original bytes and separate configuration identities.
Nonzero decimal literals that underflow stored precision reject.
Expressions have a 16 KiB byte limit and a 128-level nesting/AST/evaluation
depth limit, including variable chains, so unsupported depth rejects cleanly.

Parent intake compares decoded SON records and their diagnostic line positions
with the parsed baseline before CSV reads. Whitespace and inline comments can
change when those records/positions remain identical; hashes still differ.
Detached effective edits are allowed with a separate configuration identity.
A stale parsed parent cannot be rebound to changed SON records. The linked-STF
overload uses its captured SON/XML bytes, verifies decoded XML identities and
records the explicit static conversion and owned XML source in the snapshot.

The default table domain policy rejects outside keys. `outside=:hold` on an
explicit snapshot selects the installed engine's warned endpoint behavior.
Pass the snapshot through `scalar_files` to native geometry/component solves,
or evaluate it with `sonnet_variable_value`. Results retain the owned snapshot.
These CSV functions do not establish STF XML interpolation semantics.

```@docs
SonnetRecord
SonnetPolygon
SonnetPortSpec
SonnetProject
SonnetNetlistProject
SonnetPlanarResult
read_sonnet_project
sonnet_planar_problem
sonnet_variable_value
sonnet_planar_circuit
solve_sonnet_project
sonnet_metal_zs
SonnetScalarSource
SonnetScalarTable
SonnetScalarFiles
sonnet_scalar_files
SonnetComponentResult
SonnetModelSource
SonnetComponentBinding
SonnetComponentFiles
sonnet_component_files
circuit_add_sonnet_files!
SonnetFloatingResult
sonnet_floating_model
sonnet_component_model
sonnet_component_current_maps
sonnet_conformal_layout
solve_sonnet_conformal
```

## Sonnet technology snapshots

STF intake preserves exact SON/XML sources, dependency hashes, public
material/model/mesh/bias identities and native stack order. XML syntax is
handled by EzXML. Source, element, depth, estimated retained-storage and
table-node budgets are explicit; DTD/entities and external XML/XSD
dependencies reject. An optional local XSD provides recorded validation,
and a schema-location URL is never fetched.

Static providers cover schema defaults, isotropic dielectrics, scalar
conductivity/resistivity and exact tabulated nodes. Off-node interpolation,
etch/rho/RPV geometry application, private data and unknown physical
semantics remain explicit errors. Static linked materialization admits PEC
polygons and Lossless covers; it preserves original records. Sonnet Lite
rejects linked STF EM projects, so the native inline control does not certify
licensed linked-STF behavior.

Native editor maintain-physical exports establish `OHMM` as ohm metres
(`70000 OHUM` becomes `0.07 OHMM`) and `MOSQ` as milliohms per square
(`2 OHSQ` becomes `2000 MOSQ`). Scalar material providers use these SI
conversions; table-driven geometry adjustments remain unsupported.
The actual editor writes `MOSQ` although its stated official 1.4 XSD lists
`MOHSQ`; requesting validation against that unmodified XSD rejects these
native exports. Intake retains this discrepancy without a schema rewrite.
The editor also accepts `MOHSQ` as milliohms per square and saves the same
numeric value as `MOSQ`, so both scalar spellings are supported. The root
`writeable` Boolean is retained as authoring metadata with its exact XML
lexical domain; it does not change the static physical stack.

```@docs
SonnetTechnologyNode
SonnetTechnology
SonnetTechnologyTable
SonnetLinkedProject
read_sonnet_technology
sonnet_technology_value
sonnet_technology_material
sonnet_technology_table
sonnet_technology_lookup
sonnet_technology_stack
read_sonnet_linked_project
sonnet_materialize_project
```
