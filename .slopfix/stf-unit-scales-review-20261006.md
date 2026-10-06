# STF unit evaluation allocation review

STF scalar evaluation rebuilt four fixed unit dictionaries on every non-dimensionless call. The implementation now uses immutable unit definitions and explicit quantity dispatch. Healthy attribute validation checks keys directly and constructs the original detailed diagnostic only when an unknown attribute exists.

All 16 native unit factors and the existing rejection messages remain unchanged. Mutable technology units are still read on each evaluation. Existing STF tests, including native materialization and the previous coordinate-underflow regression, pass all 295 assertions on Julia 1.13.1 and 1.12.7.

Five warmed samples per path show non-dimensionless scalar allocation falling from 3,392 to 544 bytes and the tested two-layer stack falling from 62,624 to 23,760 bytes on both versions. The sampled scalar results and all tested stack layer fields are bit-for-bit identical. The dimensionless path remains at its original allocation. These measurements concern the tested calls, not peak process memory or universal optimization.

Independent review rehashes all 141 package inputs and both registered fixture snapshots. Full-package and hosted CI verification are separate requirements. No STF interpolation, technology geometry bias, general native skin-profile or complete RFIC parity claim is made by this change.
