# Nexxim coupled-group co-calibration oracle (A3)

Sonnet Lite 18.53 rejects calibration groups outright
("Sonnet Lite does not allow calibration groups"), so the coupled
pin-group calibration kernel `deembed_cocal_group` is verified against
an independent circuit oracle: Ansys Nexxim 2025 R1 standalone
(`nexxim.exe` batch netlist solve + `sdf2csv` S-parameter export).

## Circuits

- `launch.sp`: genuinely coupled 4-port launch error box, port order
  `c1 c2 | r1 r2` (cal-side | raw-side). A resistive lattice with
  cross-links `Rx1`/`Rx2` and shared shunt/series branches produces
  nonzero off-diagonal blocks in the 4x4 chain — independent per-port
  launches cannot reproduce it. `C1`/`L2` make the box fully complex.
- `dut.sp`: arbitrary 3-port DUT (ports `r1 r2 d3`), star junction with
  shunts and a bridging capacitance.
- `composite.sp`: launch raw nodes `r1 r2` wired into the DUT; the
  composite is measured at the cal plane `c1 c2 d3`.

All three use port-impedance resistors (`PORTNUM/RZ/IZ` + `.PORT`)
and `.LNA POI 2 1e9 2e9`, so each run reports complex S at 1 and 2 GHz.

## Method

1. `nexxim.exe <name>.sp` -> `<name>.sp.sdf` (run logs kept as
   `<name>.run.log`).
2. `sdf2csv <name>.sp.sdf -o <name>.csv` -> printed-precision complex S.
3. In the regression test (`test_planar_calibration.jl`):

   - `Ylaunch = planar_s_to_y(Slaunch)` is partitioned into
     `[Ycc Ycr; Yrc Yrr]` and converted to the voltage/current chain
     `M` satisfying the documented `[Vcal;Ical] = M [Vraw;Iraw]`
     embedding relation:

         M = [ -Yrc\Yrr        -Yrc\I      ;
               Ycr - Ycc Yrc\Yrr  -Ycc/Yrc ]

   - `deembed_cocal_group(Ycomposite, [1,2], M)` must recover
     `planar_s_to_y(Sdut)` — the passthrough port `d3` rides along.
   - The forward identity `Ycal = (C + D Y)(A + B Y)^-1` is also
     checked against the measured composite directly.

## Tolerance

CSV export carries ~6 significant digits; the double admittance
inversion amplifies that input quantization ~50x (observed relative
residual <= 7.4e-5 across both frequencies). The gate is 1e-3 relative
Y error — far above fixture noise but orders of magnitude below any
structural convention/port-order error (which are O(1)).

## Regenerating

Requires an Ansys Electronics Desktop 2025 R1 (or compatible) install
with a Nexxim license:

```
set PATH=C:\Program Files\ANSYS Inc\v251\AnsysEM;%PATH%
nexxim.exe launch.sp && nexxim.exe dut.sp && nexxim.exe composite.sp
sdf2csv.exe launch.sp.sdf -o launch.csv
sdf2csv.exe dut.sp.sdf -o dut.csv
sdf2csv.exe composite.sp.sdf -o composite.csv
```

Sanity anchors for the netlist syntax were verified independently:
a series `1 pF` capacitance between two 50-ohm ports at 1 GHz returns
S11 = 0.716957 - 0.450477i, matching the analytic thin-series
two-port value to the printed 6 digits.
