# Native STF unit editor oracle

The inputs are literal, manufactured XML controls. Sonnet18.53-Lite opened
them and saved the native exports after selecting **Maintain physical**,
**Ohm-m** and **mOhms/sq**. Exact source bytes and screenshots are retained.

`70000 OHUM` becomes `.07 OHMM`: OHMM means ohm metres.
`2 OHSQ` becomes `2000 MOSQ`: MOSQ means milliohms per square.
The original reader failure is preserved separately. These exports prove
scalar unit conversion; they do not establish table interpolation, applied
technology geometry, or licensed linked-STF EM equivalence.

The official1.4 XSD lists `MOHSQ` rather than the editor's output `MOSQ`.
The schema alias input was accepted by the editor as mOhms/sq; its variable
table shows `2000.0 mOhms/sq`, and the exact fresh roundtrip writes `MOSQ`
with unchanged `2000.0`. Both spellings have the same SI conversion. Strict
validation against the unmodified captured XSD continues to reject `MOSQ`.
