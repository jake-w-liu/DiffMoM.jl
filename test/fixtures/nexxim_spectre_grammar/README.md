# F2: broader Spectre grammar — Nexxim netlist-equivalence oracle

`ctrl.scs` exercises the extended bounded Spectre subset in one solveable
two-port: file-scope `parameters`, subckt-scope `parameters` (Spectre
overridable cell parameters), a nested instance `X1 (in n1) gainstage`,
`vccs`/`vcvs` node-pair controlled sources, `vsource`/`isource` `type=dc`,
`cccs`/`ccvs` `probe=` controlled sources and parenthesized node lists.

`ctrl_equiv.sp` is the hand-lowered equivalent SPICE deck: identical
topology written in the SPICE dialect the `.scs` lowering produces
(`E_`/`G_`/`F_`/`H_`/`V_`/`I_`/`X_` class-prefixed names). The DiffMoM
spectre-ingested result is bit-identical to the spice-ingested result,
which proves the grammar lowering preserves semantics.

`nexxim_deck.sp` instantiates the same `top` subckt with two 50-ohm
`.PORT` terminations and a `.LNA` sweep at 0.5 GHz and 1.5 GHz.

## Oracle result (Nexxim 2025 R1, `nexxim -ng nexxim_deck.sp`)

| freq  | Nexxim S21 (r,i)      | DiffMoM (from ctrl.scs) | |ΔS|max |
|-------|-----------------------|-------------------------|--------|
| 0.5G  | 1.09516, -0.182837    | 1.095156, -0.182837     | 4.4e-6 |
| 1.5G  | 0.975699, -0.517237   | 0.975699, -0.517237     | 2.9e-7 |

Differences are bounded by the CSV's six-digit print precision. The
`oracle_sdf` binary output and `oracle.csv` conversion are retained.
Nexxim's LNA pass prints a benign floating-node Z-parameter warning for
this active feedback topology; S-parameter output is valid.

## Reproduce

    "/c/Program Files/ANSYS Inc/v251/AnsysEM/nexxim.exe" -ng nexxim_deck.sp
    "/c/Program Files/ANSYS Inc/v251/AnsysEM/CMA/sdf2csv.exe" nexxim_deck.sp.sdf -o oracle.csv
