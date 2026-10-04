# Planar Networks and Outputs

Network databanks retain Kurokawa power-wave references with positive real
parts. A fixed reference is stored in `data.z0`; frequency-dependent data
also stores `data.reference_series`. The S matrices and their references
describe the same samples.

```julia
z = PlanarPortImpedance(r=50., x=10., l=1e-9, c=2e-12, topology=:parallel)
frequencies = [1e9, 2e9, 3e9]
data = PlanarNetworkData(frequencies, samples; z0=[z, 75-10im])
at_frequency = planar_network_response(data, 1.5e9; z0=[50., 75.])
planar_write_touchstone("response.s2p", data; z0=[50., 75.])
```

`samples` must be calculated with the declared references at each sample
frequency. The constructor evaluates providers once per sample, after its
numeric payload preflight. Interpolation first converts adjacent matrices
to a common basis; it is not an electromagnetic analysis or an accuracy
certificate.

Standard Touchstone and databank CSV exports use one fixed real reference
per port. Complex or frequency-dependent input is renormalized before
writing, defaulting to the real parts of the first sample's references.
The array export overloads use `z0` for input references and
`output_z0` for the file's references. Export conversion is prevalidated
before the destination is opened and streamed one matrix at a time.

Touchstone export defaults to `version="2.1"`; `"1.0"`, `"1.1"` and
`"2.0"` select the corresponding syntax. Version 1.0 needs a common
reference at every port, so use `z0=50.` for a databank or
`output_z0=50.` for the array overload when exporting to older tools:

```julia
planar_write_touchstone("legacy.s2p", data; version="1.0", z0=50.)
```

Legacy two-port values use column order; larger matrices use row order
with at most four complex pairs per physical line. Version 1.1 retains
per-port real references on the option line. These rules follow the
[IBIS Touchstone specification](https://ibis.org/touchstone_ver2.1/touchstone_ver2_1.pdf).
Legacy Y/Z/H/G data also supports unequal per-port references. Its
normalized terminal coordinates are `V/sqrt(R)` and `I*sqrt(R)`;
conversion directly in those coordinates avoids a repeated physical
denormalization. Version 2 non-S data uses SI values. Literal RI/MA/DB
records are checked against an independent 256-bit constitutive oracle
and archived external-engine voltages. Ill-conditioned format conversion
can lose information: one near-open Z-to-DB fixture changes full S by
`1.704e-10` from its external source before reading. That representation
error is measured separately from reader error.

Circuit transmission lines use bounded travelling-wave equations. Both
terminal currents point into the line; zero-length and half-wave limits
keep their voltage/current constraints. Attenuation does not require a
growing ABCD matrix. Constant and provider-returned `zc`/`gamma*length`
must remain finite and preserve nonzero components in ComplexF64, with
nonzero `zc`. A finite line response can exist when its ABCD representation
overflows. Arbitrary active-network poles or an unrepresentable response
still reject at the circuit solve.

The reader counts numeric tokens and checks aggregate storage before
allocating numeric buffers, including option references and native
termination records. Valid arbitrarily long physical data lines remain
supported. An explicit empty `[Reference]` block rejects; it never
silently falls back to the option-line reference.

The reader also preserves native Sonnet `! TERM` and `! FTERM` extensions.
TERM contains R/X; FTERM contains R/X/L/C in SI units and supports wrapped
`&` continuation records. The native C is parallel to R+jX+jωL. Malformed,
duplicate or incomplete extensions reject. Native Graph exports verify
this convention; Lite's restrictions still prevent the corresponding
non-50 Ω electromagnetic comparison.

Cartesian overlays check references at every plotted frequency, or use
explicit `renormalize=true`. Impedance Smith charts convert the terminated
input to a fixed real display reference while retaining other port loads.
Equation-curve reflection dB uses the declared power-wave reference; SWR
uses a real line reference equal to its real part. Reports list every
sample's reference when it varies.

```@docs
PlanarNetworkData
planar_read_touchstone
planar_write_touchstone
planar_network_response
planar_renormalize_s
planar_write_databank_csv
planar_read_sparam_csv
planar_compare_sweeps
planar_equation_curves
planar_write_equation_curves
planar_convergence_certificate
planar_stripline_benchmark
planar_write_report
planar_dc_sparams
plot_planar_sparams
plot_planar_smith
```
