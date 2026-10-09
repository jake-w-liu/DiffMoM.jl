# Stable DXF curves and derived layout tessellation

Both Julia versions reject valid DXF bulges of either sign whose entire
arc deviation is below the requested sagitta. A nearby accepted small
arc also moves both recorded endpoints because reconstruction subtracts
large circle-center coordinates. Public read_dxf reproduces the rejection.

Compare chord*abs(bulge)/2 with the caller's sagitta, use the stable
inverse sin identity for the angular step, and rotate relative to the
chord instead of constructing a remote circle center. Preserve the
recorded endpoints exactly. A strict segment-count inequality corrects
the rounded integer boundary caught by four independent high-precision
oracle tests in V1. V1 and all evidence remain preserved; tolerance
assertions were not loosened. Tiny-bulge, huge-chord tests cover a radius
that exceeds Float64 storage while the generated arc remains finite.

Circle and arc coordinates use one directly filled matrix. Remove the
unexplained twelve-segment minimum and million-vertex cap: the geometric
minimum is three vertices for a closed circle and one segment for an arc.
Actual coordinate bytes and caller/OS available-memory budgets determine
the admissible count before allocating. This raw per-curve budget does
not include parser objects or all retained geometry and does not reserve
memory. Aggregate hierarchy/resource auditing remains open.

Remove the selected 128000000-byte and million-polygon import defaults.
Use OS available memory and addressable integer range respectively,
while retaining explicit caller limits. DXF curved edges require caller
accuracy; DXF circles also accept explicit segment counts. GDSII derives
default cap sagitta from half its declared database lattice spacing.
No selected SI curve tolerance is imposed. These deliberate API defaults
are documented in the exported docstrings and covered by public tests.

All 215 new geometry/resource/public-API assertions and 45 original
layout import/physical-solve/trace assertions pass on Julia1.13.1 and
Julia1.12.7 (260 each). The measured 64-vertex circle allocates 1424
bytes rather than 18864 on Julia1.13.1; raw coordinates are1024 bytes.
This scoped improvement does not establish universal optimization.

Native/docs/static/proper/full/current-parent hosted checks, all known
clean worktrees, main push and the new hosted run remain required.
Original scientific and documentation gates are retained. Miter-join
thresholds, aggregate layout budgets, generic rational evaluation,
counter helper deadlines, and broader Sonnet/RFIC acceptance remain open.
