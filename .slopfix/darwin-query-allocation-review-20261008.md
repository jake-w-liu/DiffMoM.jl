# Darwin memory-query allocation repair

The actual macOS job on f389ef01 failed the original zero-allocation check:
the OS query allocated 288 bytes. The exact statistics initializer reproduced
that allocation on both supported Julia versions; its type-derived Val
tuple length allocated zero. Compiler output shows the original dynamic
tuple and splatted constructor calls, while the Mach output references were
already stack allocated. The initializer now fixes its tuple shape using
the ABI field count. No numeric tuple length is introduced.

The successful byte calculation is bounded by Sys.total_memory before
conversion to UInt64, matching the existing fallback return type and
removing the union of unsigned return types. The original allocation gate,
Mach ABI, reclaimable-page equation, send-right cleanup, conservative
failure fallback and explicit caller limits remain. A durable check covers
zero initialization and its allocation. Actual macOS CI is still required.
