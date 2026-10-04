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

`run_scaled_geovar_reference.jl` replays the tracked SCUNI/SCXY parameter/literal
pairs using an installed native Sonnet engine. It creates a fresh evidence
directory under `data/planar_audit/`; its optional argument selects an output
root. It retains logs, complete matrices, metadata and source-before/after
hashes, and fails on matrix disagreement or source drift.
