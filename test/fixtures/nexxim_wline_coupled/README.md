# Nexxim W-element coupled-line oracle (F1)

Sonnet Lite has no native coupled SPICE-line primitive, so the W-card
multiconductor implementation is verified against an independent
transmission-line engine: Ansys Nexxim 2025 R1 standalone
(`nexxim.exe` batch netlist solve + `sdf2csv` S-parameter export).

## Circuit

- `wline.sp`: `W1 a1 a2 0 b1 b2 0 N=2 L=0.05 RLGCMODEL=cplw` with
  `.MODEL cplw W MODELTYPE=RLGC N=2 FDIE=1e9` and genuinely asymmetric
  lossy per-unit-length matrices:

      Lo = [4.0e-7 1.0e-7; 1.0e-7 3.8e-7]  H/m
      Co = [1.3e-10 -3.5e-11; -3.5e-11 1.4e-10]  F/m
      Ro = [2.0 0.5; 0.5 2.0]  Ohm/m
      Go = [1.0e-5 -2.0e-6; -2.0e-6 1.2e-5]  S/m

  off-diagonal entries are nonzero in every matrix, so the result
  exercises the full modal decomposition rather than two independent
  lines. All four signal nodes are driven through 50-ohm port
  impedances; `.LNA POI 2 0.5e9 1.5e9` reports complex S at 0.5 and
  1.5 GHz.
- `cpl.spc`: the equivalent DiffMoM subckt library (`w1` + `.model cplw
  w modeltype=rlgc`) read by `planar_read_spice` in the regression test.

## Method

1. `nexxim.exe wline.sp -ng -o wline_sdf` (log in `wline.run.log`).
2. `sdf2csv.exe wline_sdf -o wline.csv` -> ~6 significant digits.
3. In `test_planar_spice_lines.jl` the compiled `PlanarSpiceModel` is
   solved with `planar_spice_sparams` on pin pairs `a1 a2 b1 b2` and
   compared elementwise (gate 1e-3, well above the CSV print precision
   but far below structural-error scale). The N=1 degenerate case is
   checked against the closed-form scalar RLGC admittance to machine
   precision.
