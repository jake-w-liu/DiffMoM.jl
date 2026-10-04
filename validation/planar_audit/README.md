# Planar audit reproducers

Run each script in a fresh Julia process from the repository root:

```powershell
julia --project=. --startup-file=no validation/planar_audit/gerber_legacy_allocations.jl
```

The nine maintained audit scripts write reports to ignored `data/planar_audit/`.
An optional argument selects an explicit output file; its parent directory is
created. `layout_modal_derivative.jl` writes a text log; the others write TOML.
`rfic_library.jl` uses the report name `rfic_library_mim.toml`.

The artwork allocation scripts measure warmed cumulative Julia allocations,
including returned masks. They do not establish peak process memory or native
engine resources. The sweep, modal derivative, and MIM scripts retain their
declared analytic or finite-grid scope. A completed process and its reported
checks establish replay success; a partial output file alone does not.

Original script bytes and SHA-256 records from the output-path review are
preserved locally under ignored `data/prior_worker_validation_20261004/`, using
their original repository-relative paths. Historical evidence remains separate
from a new execution's report.

`odb_compression_fixture.jl` is a separate deterministic fixture producer. It
rebuilds the tracked compression fixture and should be run only when updating
that fixture.

`run_radial_geovar_reference.jl` replays the tracked RAD parameter/literal
pairs, including all axis, direction and scaling headers, against the installed
native engine. It also replays RAD whole-polygon selectors with one explicit
moving reference under all headers. It checks full matrix identity and source
stability.

`run_geovar_multiplicity_reference.jl` replays ten ordinary repeated-point and
two NSCD anchored/radial moving-reference parameter/literal pairs, preserving
their recorded movements and checking
complete native matrices and source stability in a fresh output directory.

`run_geovar_reference_variants_reference.jl` replays expansion/contraction at
the explicit moving-reference count boundary. It verifies four single-occurrence
literal matches, three multiple-occurrence mismatches and one native rejection
against retained matrices and logs; all eight expected outcomes must pass.

`run_geovar_reference_count_law.jl` replays 94 native parameter/literal pairs
for anchored reference-coordinate direction and repeated ANC/RAD references.
It checks the retained complete matrices and the unchanged 1e-12 native
identity gate in a fresh directory. Earlier failed sequential and direction
hypotheses remain in their original fixtures and captures.

`run_radial_adjustment_sequence.jl` replays 36 native parameter/literal pairs
for three successive radial adjustments, including anchor crossings, diagonal
rays, repeated ordinary points and point-group/header variations. It checks
the retained complete matrices and the unchanged 1e-12 identity gate in a fresh
directory. The separate engine memory-allocation failure remains unverified.

`run_box_port_extent_reference.jl` replays 80 native inputs: six ordinary edge
controls, 68 boundary-allowance controls and three active ANC parameter/literal
pairs. It checks exact rejection messages, complete retained matrices at the
unchanged 1e-12 identity gate and public raster outcomes. Both the original
false acceptances and the overly strict candidate's false rejections remain
in separate immutable fixtures.

`run_port_attachment_reference.jl` replays 20 native controls for logical port
edges, annotation coordinates, active geometry midpoint updates, collinear
wall excitation and undriven wall grounding. It preserves complete matrices
at the unchanged 1e-12 identity gate and checks importer acceptance/rejection.
The original incorrect responses remain in the immutable fixture reports.

`run_internal_port_attachment_reference.jl` replays 21 native shared-edge,
partial-overlap, annotation, kind-alias and invalid-return controls. Both
axes have two-port phase controls referenced from either adjacent polygon.
It checks complete retained matrices at the unchanged 1e-12 identity gate
and raster/conformal acceptance/rejection, including four diagonal source
rejections. Original failures remain immutable.

`run_whole_geovar_reference.jl` replays sixteen whole-polygon zero-count,
explicit vertex-list and independent literal triples. It checks native full
matrix identity, source stability, both axes/directions and contraction/expansion.

`run_scaled_geovar_reference.jl` replays the tracked SCUNI/SCXY parameter/literal
pairs using an installed native Sonnet engine. It creates a fresh evidence
directory under `data/planar_audit/`; its optional argument selects an output
root. It retains logs, complete matrices, metadata and source-before/after
hashes, and fails on matrix disagreement or source drift.

`run_folded_ufft_resource_audit.jl` compares retained modal arrays with folded
family-pair spectra on mixed sheet/via/volume PEC and PMC problems. Independent
modal matrices gate complete actions and diagonals; alternating warmed samples
record construction and matvec costs. Reports use a fresh ignored directory;
an optional argument selects its parent. These measurements exclude peak
process memory and opaque FFTW plan storage.
