# Scalar STF native controls

Exact archived sources and outputs from installed Sonnet 18.53-Lite.
`provenance.json` hashes every retained file. The manufactured `mini.stf`
declares two scalar dielectric layers in UM and builtin Lossless covers;
there is no etch, RPV table or encrypted material.

The independent inline project was actually simulated at 1/10 GHz. Its full
complex S agrees with a separately hand-built stack/layout within 0.000108
(unchanged 0.005 gate). Both linked projects are rejected by the native
engine: "Project includes linked STF file. Sonnet Lite does not allow use
of linked STF files." The serialized 10 Ω covers are a precedence probe;
their linked EM response remains unverified due to that same restriction.

Native stdout independently reports the XML 100 UM TopH/BottomH variables as
0.1 mm in the SON length context before the linked license rejection.
Production static materialization is checked against the inline control;
this does not establish native linked-STF EM equivalence.

Complete original run, version/command metadata and benchmark source:
`data/sonnet_validation/native_stf_scalar_rZbgen`.
Primary installed three-STF/schema/help archive:
`data/sonnet_validation/sonnet_stf_primary_rd055bdl`.
