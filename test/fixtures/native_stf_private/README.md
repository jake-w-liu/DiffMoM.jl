# Native STF private-section materialization probe (D6)

`input.stf` is `native_stf_linked_solve/rich.stf` modified to declare
`has_private="true"` and carry a schema-valid `<private>` section
containing a deliberately divergent `materials` + `stackup` tree (a
single 100 um Air layer with `erel=9.99` instead of the public stack).

`input_link.son` links it via `STF input.stf`.

`native_public_only.son` is the output of the installed

    soncmd -LoadTechnology input_link.son input.stf out.son

run on Sonnet Lite 18.53. It is byte-identical (modulo timestamp and
`STF` name records) to the output produced from the unmodified public
STF — native `-LoadTechnology` materializes only the `<public>`
section and ignores `<private>`/`<privenc>` entirely. The same probe
on the shipped `vnc_mom_cap.stf` example (which carries a real
`privenc`-capable layout) confirmed identical behavior.

DiffMoM therefore treats declared root-level `private`/`privenc`
sections as opaque and materializes the public section identically
(`_stf_static_guard`). Undeclared private sections, malformed
`has_private` values, and private payloads nested inside
grammar-gated elements (dielectrics, technology layers) still reject
loudly — only name-and-kind-keyed public lookups drive material
resolution, so a private-only reference resolves to nothing.

Encrypted *parameter* payloads (real foundry `privenc` blobs whose
materials are referenced by the stack) remain DocOnly: they cannot be
decrypted, and any public reference to them rejects as an unresolved
material name.
