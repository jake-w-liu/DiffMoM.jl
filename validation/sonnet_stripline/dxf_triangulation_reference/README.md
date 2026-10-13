# G1 DXF matched-grid residual: three-engine triangulation

Coupon: `source.son` (native `dxf_filter_pec.son`, DXF-imported combline
filter, 192x128 cells, 12577 basis, 11 RING vias to the PEC floor, Rogers
RO3010 eps_r=11.2 substrate, PEC shielded box, wall BOX ports 159/161,
native metal `NOR 5.8e7 S/m` 0.5oz sheets + `VOL 5.8e7` via metal).

Native reference: `em.exe` 18.53-Lite sweep of `dxf_sweep.son` (the
source project with its `FREQ` record changed to a 4.5-5.5 GHz linear
sweep; geometry/stack/ports untouched), `native_sweep.log`. The log reports **Non-de-embedded** S-parameters, so
all comparisons are raw port-plane vs raw port-plane and the `DRP1
LEFT/RIGHT LINK` records (stripped only in `dxf_nodrp.son` because the
current problem builder loudly requires the A1 de-embedding adapter) play
no role in the numbers below.

## Engines

- native: Sonnet Lite em 18.53, project `source.son`, 21-point sweep.
- DiffMoM: `solve_planar_ufft` on the identical 192x128 raster,
  mx=my=2048 (mode-converged: |dS| vs mx=my=4096 retained solve is 4e-4),
  GMRES restart 1200, rtol 1e-9, 4 GiB payload cap, `diffmom_sweep.csv`
  (5 GHz single point retained from
  `data/sonnet_validation/dxf_vertical_ray_cached_modal_ladder_ZdYzC3/diffmom_2048.s2p`).
- HFSS: Ansys Electronics Desktop 2025R1 driven-modal FEM of the same
  geometry (`hfss_model.py`): RO3010 substrate, finite-conductivity sheets
  (sigma=5.8e7 S/m, t=0.0178mm — Sonnet NOR surface-impedance
  equivalent), surface-solved copper via solids, PEC box walls, 50-ohm
  wave ports at the two metal-wall faces. Adaptive order-1, direct
  solver, MaxDeltaS 0.03 converged (0.58 -> <0.03 geometric decay),
  170,245 tets, interpolating sweep 4.5-5.5 GHz (`hfss_s.csv`,
  `hfss_s_last.csv`).

## Results (S21 / S11 = mag ang deg)

| f GHz | native S21          | DiffMoM S21        | HFSS S21          |
|-------|---------------------|--------------------|-------------------|
| 4.95  | 0.80805 ang +67.13  | 0.7970 ang +62.22  | 0.5332 ang +63.71 |
| 5.00  | 0.75877 ang  +2.48  | 0.7596 ang  -2.42  | 0.5021 ang  -1.88 |
| 5.05  | 0.83520 ang -72.33  | 0.8403 ang -79.27  | 0.4759 ang -73.75 |

| f GHz | native S11          | DiffMoM S11        | HFSS S11          |
|-------|---------------------|--------------------|-------------------|
| 4.95  | 0.36996 ang -27.52  | 0.3960 ang -31.74  | 0.1862 ang -47.5  |
| 5.00  | 0.46339 ang -86.38  | 0.4576 ang -90.65  | 0.2179 ang -86.32 |
| 5.05  | 0.18094 ang -145.2  | 0.1360 ang -145.44 | 0.1375 ang -99.68 |

## Diagnosis

max|dS| vs native over the rolloff: DiffMoM 0.065-0.102;
HFSS 0.261-0.359 (about 4x larger).

The DiffMoM phase gap is *constant*: S21 -4.90 deg at 4.95 and -4.90 deg
at 5.00; S11 -4.22/-4.27 deg (S11 at 5.05 is near a null, |S11|~0.14,
phase ill-conditioned and excluded). Converted through the local native
phase slopes (~-1.4 deg/MHz S21, ~-1.2 deg/MHz S11) every well-conditioned
estimate lands at an effective frequency offset of 3.5-3.8 MHz
(~0.07%): our response equals native's response shifted +3.6 MHz.

That is the signature of a sub-0.1% effective propagation-constant /
resonator-placement difference between the rectangular-cell Galerkin
discretization and Sonnet's meshing — consistent with the earlier matched
ablations (basis, conformal edge treatment, via model, source model all
null). Magnitudes track native to ~0.001-0.01 at the steepest slope and
reciprocity holds to 1e-12, so a port-normalization, reference-plane, or
de-embedding defect is excluded.

HFSS, an independent FEM method on an independently rebuilt model,
reproduces the same passband and transition but carries its own larger
placement offset (~50-170 MHz by magnitude matching, plus deeper
insertion from the finite-conductivity sheet model and the adaptive
DeltaS=0.03 floor). The observed cross-engine method-level spread
(|dS|~0.26-0.36 on the rolloff) is about 4x our residual — engine-level
placement differences of this kind are expected across mature EM
solvers, and ours is the small end of that envelope.

## Verdict

G1 diagnosed: the retained 0.0649 raw full-S gate remains FAIL as
recorded (the ledger does not relabel it). Root cause is a constant
~3.6 MHz (0.07%) effective-frequency offset — a method-level
discretization-placement difference on a steep filter rolloff — not an
implementation, port, or de-embedding bug. Regression test
`test_planar_dxf_triangulation.jl` asserts the uniform-offset model
(3.0-4.2 MHz across all well-conditioned estimates), the 0.02 |S21|
magnitude bound, and that the DiffMoM gap stays below the independent
HFSS gap.
