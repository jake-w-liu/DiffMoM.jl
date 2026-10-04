# Native dielectric resistivity oracle

The original native GUI source and loaded STF are preserved. Six fresh native
controls retain the same literal PEC geometry and raw 50-ohm sources, with
the unused sheet technology declarations explicitly removed. At each of
1/5/10 GHz, RSVY=7 and conductivity=100/7 S/m agree within 1.01e-13 in full S.
The engine prints `Rho = 7 Ohm-cm`, agreeing with the installed primary help.

The old importer treated 7 as S/m and failed the unchanged full-S gate by
0.1656–0.1754. The corrected importer and independent manually specified
stack agree with the actual native matrices within 0.000546; original source
residuals are below 2e-13. Both before and after reports remain retained.
Technology geometry and conductor NOR/SRVY flags remain separate contracts.
