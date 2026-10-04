# Native material field-order oracle

The installed Sonnet 18.53 Lite GUI decoded the independently authored
`planar_order_probe.son` on 2026-10-04. The raw source, GUI tables, editor
screenshots and hashes are retained here.

| Native record | Observed GUI fields |
| --- | --- |
| `SUP 1 2 3 4` | Rdc 1, Rrf 2, Xdc 3, Ls 4 |
| `SFC 1 1e-5 .2` | Rdc 1, Xdc 1e-5, Rrf 0.2 |

The editor labels confirm RF resistance in Ohm Hz^(-1/2)/square,
DC resistance/reactance in Ohm/square and planar Ls in pH/square.
The nonzero planar Ls triggers the archived Lite kinetic-inductance
restriction. These artifacts validate material field order and units;
they do not establish native EM agreement for Surface vias or nonzero Ls.
Surface-via material lowering remains explicitly unsupported.

The old `surface_physical_bridge_McTbg2` run emitted SUP ordering for SFC.
Its original outputs remain intact with a separate diagnosis. Its declared
RF/reactance settings were wrong, so it provides no physical skin-law
comparison evidence.
