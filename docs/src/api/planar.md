# Planar Spectral-Domain MoM

API for the shielded multilayer planar solver. See
[Shielded Planar Spectral-Domain MoM](../formulations/06-planar-spectral-mom.md)
for the formulation, conventions, and a worked example.

```@docs
PlanarLayer
PlanarTerminator
PlanarStackup
CellGrid
BoundaryKind
TERM_PEC
TERM_PMC
TERM_SURFACE
TERM_OPEN
TERM_GND
TERM_SPACE
SheetLevel
PlanarPort
planar_interfaces
planar_k0_layer
planar_validate
sheet_level
rasterize_rect!
rasterize_poly!
build_planar_basis
planar_basis_count
PlanarBasisSet
PlanarModeGrid
planar_mode_grid
assemble_planar_z
PlanarPol
TE_POL
TM_POL
PlanarCascade
planar_mode_cascade
planar_modal_voltage
PlanarProblem
PlanarResult
build_planar_problem
solve_planar
planar_sparams
planar_y_to_s
write_touchstone
```
