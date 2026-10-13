# Nexxim FLOAT file-model attach-law oracle

Independent circuit-simulator verification of the FLOAT/BOX
floating-common attach law used by `sonnet_component_model` /
`solve_sonnet_project` for file-model (SPARAM) Sonnet components.

## Coupon

A synthetic Sonnet project (`gen.jl`, `float_test_project`) carries a
FLOAT SMD component with a SPARAM file model. DiffMoM lowers it to a
5-node `PlanarCircuit`:

- EM network: 3-port S block on `source_terminals = [(1,0), (2,0), (3,4)]`
  — two differential feed ports plus one per-pin port pair to the
  floating common node.
- File model: 2-port S block on `[(3,5), (4,5)]` — both device
  terminals reference the floating common node 5.
- External ports `[(1,0), (2,0)]` at 50 ohm.

`attach.sp` is a hand-independent replay of that exact wiring in Nexxim
multi-reference S-element form (`Sxxx n1 ref1 n2 ref2 ... MNAME=`):
`SE 1 0 2 0 3 4 MNAME=emmod` reproduces the EM block terminal map and
`SM 3 5 4 5 MNAME=dev` reproduces the file-model attach. `Rb*` 1e9-ohm
bleeders reproduce the `:auto` floating-gauge DC reference so the
Nexxim MNA system is non-singular, matching the solver's gauge
treatment.

## Regeneration

    julia --project=. gen.jl            # writes attach.sp, expected.txt, *_s.txt, topology.txt
    nexxim attach.sp                    # AnsysEM v251 Nexxim, writes attach.sp.sdf
    sdf2csv attach.sp.sdf -o attach.csv

`gen.jl` rebuilds the same project as
`test_planar_sonnet_floating.jl`, solves it at 0.1/0.15 GHz, and dumps
the EM block (`em_s.txt`), file-model block (`dev_s.txt`), composite
result (`expected.txt`), and wiring topology (`topology.txt`).

## Result

Nexxim LNA (0.1/0.15 GHz) vs `solve_sonnet_project`:

| f (GHz) | element | DiffMoM | Nexxim | |diff| |
|---|---|---|---|---|
| 0.1 | S11 | 0.11003115+0.00170419i | 0.110031+0.00170419i | ~1e-7 |
| 0.1 | S21 | 0.88991582-0.01200190i | 0.889916-0.0120019i | ~1e-7 |
| 0.15 | S11 | 0.09007744+0.00306231i | 0.0900774+0.00306231i | ~1e-7 |
| 0.15 | S21 | 0.90980326-0.01850858i | 0.909803-0.0185086i | ~1e-7 |

The composite is symmetric (S12=S21) even though the file-model block
is non-reciprocal, because the EM geometry is symmetric in the two
device pins — a physical sanity check the regression test asserts.

The test gate is 1e-3 (CSV emits ~6 significant digits); the observed
agreement is ~1e-7. Native Sonnet Lite cannot exercise this path
(SPARAM component models are licensed-solve gated), so Nexxim is the
independent oracle for the floating-common attach law.
