# Result serialization boundary and verified recovery

The frozen INV-047 contract permits field/ordering compatibility or an explicit
migration/breaking policy. Current checked-current result fields/type parameters
change the previous opaque Julia cache layout. Actual prior-main PlanarResult
and PlanarSourceResult caches round-trip in the original implementation, then
fail under current46 on Julia 1.13.1 and 1.12.7 at the same patch version.

The public types and wide-current pages state that Julia result caches require
a compatible recorded environment and provide the supported recovery boundary:
load with the original code, export supported network data through Touchstone,
or reconstruct inputs and solve again. Both actual representative result kinds
recover and export/import every frequency/reference/complex S value exactly on
both Julia versions. A network export contains no geometry, current/factor state
or evidence that the old coefficients meet current physical requirements.

No binary upgrader, cross-revision opaque-state compatibility or general
serialized operator migration is claimed. Original scientific/native/resource
and constructor gates remain unchanged. Historical inventory/report unverified
entries are retained; this dated policy resolves the documented policy
alternative for the current representation change once published.
