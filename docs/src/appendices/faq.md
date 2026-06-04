# Appendix: FAQ

## Why does my solve fail right after importing OBJ?

Usually mesh quality issues (non-manifold edges, degenerates, orientation
conflicts). Run `mesh_quality_report` and then
`repair_mesh_for_simulation`.

## Why is my run very slow?

Dense MoM scales poorly with unknown count. Check `rwg.nedges` and
`estimate_dense_matrix_gib`. Coarsen mesh or reduce scenario count.

## Does preconditioning improve accuracy?

Primarily numerical robustness/efficiency of linear solves, not physical-model
accuracy by itself.

## Why does beam steering degrade at small frequency offsets?

Single-frequency optimized reactive maps are often narrowband; detuning shifts
effective phase progression.

## Why do global dB residuals look worse than beam metrics?

Deep-null regions can dominate absolute dB residuals. For steering workflows,
beam-centric metrics (main beam, sidelobes, target-angle gain) are usually more
informative.

## Can I use arbitrary meshes?

Yes, if they pass mesh-quality checks and are repaired/coarsened to a feasible
unknown count.

## Where should I start as a new user?

1. Getting Started → Installation/Quickstart  
2. Tutorial: First PEC Plate  
3. Tutorial: Adjoint Gradient Check  
4. Tutorial: Beam Steering Design

## How do I verify my custom workflow?

Use internal consistency gates first, then gradient checks (if optimized), then
external validation (e.g., Bempp or sphere-vs-Mie where applicable).

## How do I validate near-field or total-field results?

Start with the internal consistency checks in `validation/01`, then use the
Rayleigh small-sphere benchmark in `validation/06` and
`examples/21_near_total_field_rayleigh_sphere.jl` for an analytical
near-/total-field reference.
