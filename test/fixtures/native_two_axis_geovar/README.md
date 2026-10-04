# Native two-axis geometry scaling

Fresh Sonnet 18.53 `SCXY` runs use a notched polygon with references on its
port wall. Sixteen parameter/literal pairs cover ANC/SYM, XDIR/YDIR, signed
reference directions, expansion and contraction. All complete native S
matrices are bit-identical. Four retained axis-only hypotheses disagree by
more than .014 in full S and establish movement in both coordinates.

Native sources, logs, errors, metadata, full Touchstone and SID responses,
comparison records and relevant source snapshots remain exact. Historical
source hashes describe the capture, not a requirement that current code equal
the old implementation. `sha256.toml` covers every payload and this readme.

`literal_physical_before.toml` records public literal solves before adding the
adapter. The previously declared fixed gates are .005 absolute full-S error
and 1e-9 original voltage residual, at mx=my=128. All sixteen pass, with maxima
2.554e-5 and 4.984e-11. These are fixed-grid coupon comparisons, not continuum
convergence. The current regression resolves actual parameter records through
public lowering, compares both coordinate axes, attached ports and masks to
independent literal controls, and retains ownership, resource and override
checks.

`rejected_diagonal_port/` retains an actual native error when scaling turns a
wall attachment into a diagonal port. Public lowering rejects that circuit.
Supporting SCXY does not make an invalid physical port configuration valid.
Active RAD, ordered dependent/overlapping dimensions and moved component,
interior or via attachments remain separate implementation work.

```powershell
julia --project=. --startup-file=no -e 'include("test/test_planar_sonnet_two_axis_geometry_variables.jl")'
```
