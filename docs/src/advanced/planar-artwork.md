# Fabrication artwork imports

Fabrication files retain their analytic drawing operations until an explicit
analysis grid is selected. This preserves circular boundaries, rotated pads,
clear image operations and transparent holes without an intermediate curve
polygonization. All imported dimensions are in metres. Unknown conductive
records raise an error.

## Gerber

```julia
using DiffMoM
artwork = read_gerber("top_copper.gbr"; layer="top")
grid = CellGrid(10e-3, 8e-3, 200, 160)
masks = artwork_cell_masks(artwork, grid; offset=(0.0, 0.0))
```

The current reader supports C/R/O/P apertures, arithmetic aperture macros,
nested aperture blocks, circular drawing, closed linear/circular regions,
step and repeat, dark/clear image polarity, aperture mirror/rotation/scale,
attributes, inch/mm units, leading/trailing zero suppression and legacy
incremental coordinates. Macro and standard aperture holes are transparent:
they preserve objects underneath. Aperture blocks append ordered image
operations, so clear objects in a block erase earlier image copper; flashing
a block under clear polarity toggles every contained object's polarity.

Negative images are complemented over the explicitly selected analysis
window. Legacy MI/SF/OF/IR transforms execute in the required order,
independent of their record order. MI and SF change coordinate paths while
preserving aperture geometry and repeat spacing; IR rotates the complete
image. Nonuniform SF retains analytic elliptical strokes and curved regions.
AS is retained as plotting-device metadata and leaves the exchanged image
unchanged. Legacy solid standard
rectangle drawing is implemented for straight and circular paths, retaining
the fixed aperture orientation, square end caps and analytic circular
boundaries. The specification does not allow drawing with a holed rectangle,
other standard shapes or macros that happen to look rectangular. An unsupported
command
rejects the file rather than producing an incomplete circuit.

The implementation follows the
[Ucamco Gerber Layer Format Specification, revision 2026.05](https://www.ucamco.com/files/downloads/file_en/554/gerber-layer-format-specification-revision-2026-05_en.pdf).
The independent regressions check analytic membership, macro arithmetic,
polarity and solver lowering. Another 5180 rectangle-line/arc checks use
independent line clipping and circle/rectangle min/max-distance oracles,
including clockwise paths, full circles, single-quadrant zero-length arcs
and rotated apertures. Another 13680 legacy-transform checks cover fixed
apertures, all transform-order permutations, blocks/repeats, independent
ellipse distance bounds, circle-normal witnesses, rectangle/ellipse
intersection, affine regions and a raster allocation gate. All eleven
[official Ucamco test files](https://www.ucamco.com/files/downloads/file_en/423/gerber-layer-format-test-files_en.zip)
parse and rasterize. Four resulting images have also been visually compared
with the supplied previews; their different margins and antialiasing prevent
using that check as a quantitative pixel accuracy gate.

## ODB++

```julia
artwork = read_odb("unpacked_product";
    step="board", layers=["top", "bottom", "drill"])
```

`read_odb_features` reads a single features file; `read_odb` reads a product
directory or tar/tar.gz/tgz/tar.Z archive with its matrix, product units, step
headers, user symbols and step repetitions. Plain, gzip and UNIX-compress
`.Z` entities can be mixed within a product. Decompression and archive
extraction use temporary directories that are removed when the call ends.
Expanded entity bytes, extracted file bytes and owned decoder/archive
metadata are checked against `max_bytes`; `max_entities` bounds archive
entries. Archives require at least 2 MiB for the standard tar copying
buffer. Links, special files, duplicate paths and paths outside the product
directory reject. Per-symbol inch/micron declarations override the
features file's default symbol units. Pad and step rotations follow ODB++'s
clockwise convention; rotation precedes its mirror of the x coordinates.
Negative layers are inverted inside the step's own profile before placement,
so repetition does not turn the surrounding parent panel into copper.
Matrix rows and step columns must be unique positive integers; gaps and
out-of-order records are allowed.
Uppercase logical names in matrix/repetition records are canonicalized to
their legal lowercase entity references, as in the primary specification's
matrix example. Raw names remain in `ODB.matrix` JSON metadata. Case-folded
duplicate names reject, and filesystem/entity name validation stays strict.

`FLIP=YES` mirrors a repeated step and reverses its complete physical BOARD
buildup, which must have symmetric layer types and polarities. Drill/rout
layers map by their reflected `START_NAME`/`END_NAME` span and subtype;
through spans remain through. Documentation and MISC layers retain their
names by default. If equal spans make the layer correspondence ambiguous,
provide `flip_layer_map=Dict("blind_top"=>"blind_bottom",
"blind_bottom"=>"blind_top")`; the complete mapping must be an involution
and preserve the physical reversal. Layer selection names the destination:
selecting top loads the flipped bottom input, including nested repetitions.
FLIP and MIRROR compose as two reflections, so their joint XY reflection
cancels while FLIP still reverses the layer buildup. Datum and repetition
pitch use that combined transform. Negative layers retain their own profile
under the same placement. Matrix/info records, recursive headers, repeated
attributes and streamed JSON metadata share the owned payload budget.

Current standard symbols include circles, squares, rectangles with selected
rounded/chamfered corners, obrounds, diamonds, triangles, octagons, hexagons,
round/square butterflies, ellipses, and basic round/square/rectangle/obround
donuts. Drill-wheel symbols retain their diameter, plating classification and
positive/negative tolerances in `ODB.drill.*` feature attributes. Null symbols
retain their extension and feature attributes without drawing geometry.
Additional analytic symbols include half ovals with width at least half
their height; squared-cut round, square, square/round, rectangular and oval
thermals (`ths`, `s_ths`, `sr_ths`, `rc_ths`, `o_ths`); rounded round
thermals (`thr`) and line thermals (`s_thr`); and standard,
inverted and flat home plates (`hplate`, `rhplate`, `fhplate`). Home-plate
corner fillets preserve convex and concave circular boundaries and enforce
the primary strict adjacent-corner limit. Ordered numeric optional radii in
the tables and named `ra`/`ro` tokens in the figures are both accepted.
D-Pack grids (`dpack`) retain straight or rounded individual pads without
allocating an object per pad. `hn` counts pads along the horizontal side
and `vn` along the vertical side, following the primary diagram and the
side identity `count*pad + (count-1)*gap = side`. Radii may reach half the
smaller pad dimension; infeasible pad dimensions or unrepresentable counts
reject explicitly. Geometry storage and membership work are constant with
respect to both counts, including a billion-by-billion zero-gap grid.
Thermal gap angles are
counter-clockwise; pad orientations and rotation suffixes are clockwise.
The older [primary v7 angle definition](https://odbplusplus.com/wp-content/uploads/sites/2/2020/03/ODB_Format_Description_v7.pdf)
states that distinction explicitly.
Rounded thermal gaps measure the nearest clearance between circular caps.
Circular thermals retain swept circular centerline arcs and rounded end caps;
line thermals retain four capped straight strokes. The line thermal end-cap
diameter is `(outside-inside)/2`, as corrected in Update 3, and its intrinsic
parameters require 45 degrees and four spokes. Infeasible gaps reject rather
than producing negative arc or line lengths. Gap count does not allocate a
proportional list of arcs.
Feature files accept modern `UNITS=INCH`/`UNITS=MM` and the literal legacy
`U INCH`/`U MM` records found in the installed Sonnet example archive.
Both forms require one declaration before symbols or geometry; repeated,
mixed, malformed and unsupported unit declarations reject. Their original
syntax and evaluated unit are retained in metadata.
Curved surfaces retain their
analytic island and hole contours in natural containment order, including
conductive islands inside holes. Holes stay transparent to previously drawn
artwork, and complete circular contours give the same filled region in either
arc direction. Feature attributes retain their encoded
values and optional text-table lookup separately. Numeric attributes cannot
be reclassified as text by a coinciding lookup-table index.

Vector fonts from the product's `fonts` folder lower quoted `T` records into
analytic round/square strokes. Single-file imports accept `font_directory`.
Character advance and capital height use the font metrics; text stroke width
is absolute in 12 mil units, independent of feature units. Both insertion
versions, feature polarity, attributes and all ten orientation codes are
retained. Reused user symbols resolve dynamic text at each placement.
`text_context=Dict("DATE-MMDDYY"=>"100326")` supplies explicit values for
dates, times or other dynamic variables. Product/step/layer names and
placement coordinates supply `JOB`, `STEP`, `LAYER`, `X`, `Y`, `X_MM` and
`Y_MM` when known; unresolved or recursive substitutions reject. Input,
font caches, active reader prefixes and expanded strokes share `max_bytes`.
`max_objects` also bounds font strokes and expanded text strokes.

Independent literal vector glyphs check metrics, physical widths, insertion,
orientation and hierarchy. Vendor standard-font rendering, diagonal square
stroke compatibility and native dynamic-coordinate formatting remain
unverified; these tests do not establish pixel-identical vendor rendering.

Open-corner and further rounded thermal families, rounded annuli, further
radiused stencil/cross/dogbone/moire symbols, extreme half-oval
dimensions, barcodes,
dimensional pad resizing
remain implementation work. These records currently raise explicit errors.
Missing feature files and ambiguous plain/compressed variants reject.
Entity names and resolved paths are restricted to the
selected product directory, and circular symbol/step references reject.

These rules come from the
[Siemens ODB++Design Format Specification, 8.1 Update 3](https://odbplusplus.com/wp-content/uploads/sites/2/2021/02/odb_spec_user.pdf).
The present acceptance uses independent geometric/unit/transform/hierarchy
fixtures and 77 compression/archive regressions on Windows (76 elsewhere).
An independent libarchive
3.8.8 producer supplies a UNIX-compress fixture; decoding reproduces the
SHA-256 of the separately produced 837120-byte uncompressed tar. These
checks do not certify a complete vendor manufacturing corpus. A separate
27-point end-to-end fixture checks clear-window Gerber, legacy transformed
Gerber and an ODB++ job against independently hand-authored native Sonnet
geometry on three actual grids at three frequencies. Maximum complete
complex S error is `1.18886e-4`, with unchanged `0.06` and recomputed-residual
gates. Native geometry was not generated from the reader's mask. Evidence:
`data/sonnet_validation/artwork_native_yoK3Zq`.

The additional symbol suites have 40207 standard-symbol and 19185 rounded
thermal geometry, reader, orientation, polarity and allocation checks.
Existing ODB++, legacy unit records and new text
checks contribute 1751, 39 and 272 respectively; FLIP, matrix references and
metadata preflight contribute 470; compression contributes 77
on Windows. This is 62001 ODB++ checks on Windows, separately from the
19185 Gerber checks. A further 55604 D-Pack checks cover independently
enumerated pads, units, rounded corners, orientations, polarity, resource
rejection and constant geometry/membership allocations. Vendor renderer
compatibility remains scoped above.

## Analytic closed boundaries

Circular and polygonal flashes and their ordered translation sweeps use the
same closed analytic boundary. Ordinary membership is certified with outward
floating intervals. Uncertain tangencies and overlapping boundary events use
exact dyadic or quadratic witnesses, followed by the original dark/clear
operation order. Construction bounds primitive events and nesting; the rare
exact sweep workspace is checked against the remaining `max_bytes` before
arbitrary-precision operands are created. Ordinary typed queries retain their
zero-allocation path.

This deliberately removes the former circular `hypot` rounding and region
`8eps` boundary band. A point just outside a circle is outside even when its
rounded distance equals the radius. Arc centerline radii use the computed
stored Float64 radius, and literal arc endpoints remain closed. Zero-length
and reversed strokes follow the same convention as a flash. Affine membership
uses the stored inverse matrix, with outward cached bounds derived from that
inverse so raster clipping cannot discard accepted points. Extreme finite
scales use normalized or separate diagonal inversion; an inverse that cannot
be represented in Float64 is rejected.

The original tangency, overflow and clipping failures and the independent
regressions are retained in `test/test_planar_artwork_exact_boundaries.jl` and
`data/prior_worker_validation_20261004/validation/planar_audit/artwork_*_before` artifacts. These are geometry and
resource checks; native electromagnetic comparisons have separate source
hashes and acceptance gates.

## Lowering to the solver

```julia
artwork = read_gerber("top_copper.gbr"; layer="top")
stack = PlanarStackup([PlanarLayer(3.5, 1.0, 0.5e-3)],
    TERM_GND, TERM_GND, grid.a, grid.b)
technology = Dict("top" => (; kind=:sheet, interface=1))
ports = [PlanarPort(1, :west, 40:60, 50.0),
         PlanarPort(1, :east, 40:60, 50.0)]
problem = artwork_planar_problem(artwork, stack, grid, technology, ports)
result = solve_planar(problem, 1e9; method=:ufft)
```

Ports must cover actual connected conductor cells on the selected grid.
Sheet mappings use explicit bottom-to-top interface indices; via mappings
use `(; kind=:via, from_interface, to_interface)`. Multiple artwork layers
mapped to one sheet are combined after each layer's ordered polarity has
been evaluated. Material impedances/conductivities and port calibration
remain explicit solver inputs. `max_bytes` bounds owned input/geometry/grid
payloads; the resource limit is checked before raster or basis construction.
Actual physical widths, terminal spans, clearances and mode/grid convergence
must be checked before accepting an RF design result.

## Raster allocation evidence

A warmed raster of one analytic circular region on a 400×400 grid allocates
28749 Julia bytes, including the returned 20000-byte bit mask, with 125676
occupied cells. The original loop allocated 33308781 bytes for the identical
geometry. Static arc cuts and a shape-specific inner loop remove the repeated
per-cell temporary arrays and boxing. This measures cumulative Julia
allocation for that fixture, not peak process memory or every shape family.
The reproducer and before/after measurements are in
`validation/planar_audit/artwork_allocations.jl`. New TOML outputs go to ignored
`data/planar_audit`; original measurements remain in the local historical
archive `data/prior_worker_validation_20261004/validation/planar_audit`.

The new analytic rectangle-arc raster has the same 28749-byte allocation
scope on a 400×400 grid. The UNIX-compress fixture decoder allocates 331204
Julia bytes while expanding 837120 bytes to a null destination. Removing
per-code arbitrary-precision resource additions reduced that value from
72220604 bytes; a subtraction guard preserves the same expansion limit
without overflowing integer addition. The decoder's output SHA and all
transport regressions are unchanged. These measurements and their scope
are in `validation/planar_audit/artwork_extended_allocations.jl`.

A warmed nonuniform ellipse-stroke raster on a 400×400 grid allocates
28781 Julia bytes including its 20000-byte returned mask, with 37296
occupied cells. Membership and its quartic candidate helper each allocate
zero bytes. The initial 80×80 implementation allocated 334597096 bytes;
fixed tuple return types and a homogeneous angle loop remove that repeated
scratch allocation. This is cumulative Julia allocation, excluding peak
process memory and native runtime allocations. Reproduction:
`validation/planar_audit/gerber_legacy_allocations.jl`.

Warmed 400×400 squared-thermal, rounded-home-plate and half-oval rasters
allocate approximately 28.8 KB each, including the returned 20000-byte mask.
Retaining concrete payload types in private transform/thermal wrappers
removes repeated dispatch allocations: the original thermal fixtures used
12.6–13.2 MB. Separate preserved abstract-field validation types reproduce
the original path and verify equality of every mask cell. Typed membership
allocates zero bytes. This has the same cumulative Julia allocation scope,
with no peak process-memory claim. Reproduction:
`validation/planar_audit/odb_extra_symbol_allocation_equivalence.jl`.

Mixed round/square vector-font text on a 400×400 grid now allocates
approximately 28.8 KB including its returned mask. The original eight-glyph
fixture allocated 149580973 bytes. Concrete stroke references preserve the
same ordered dark/clear geometry and avoid copying union-valued strokes
at each sample. Independent preserved abstract composites check every
mask cell; the shared composite regressions include negative strokes.
Evidence: `data/prior_worker_validation_20261004/validation/planar_audit/odb_font_raster_allocations.log` (original),
`odb_font_raster_allocations_after.log` and
`artwork_typed_composite_prototype.toml`.

Small heterogeneous composites of at most eight parts retain concrete
tuple fields. A square pad with a circular transparent hole originally
allocated 15350365 bytes for a 400×400 raster; the optimized path allocates
approximately 28.8 KB and returns the same 79336 occupied cells.
Membership allocates zero bytes. Larger heterogeneous collections retain
the general vector path, so this is a scoped performance result. The tuple
size cap bounds compilation growth. All these figures measure warmed
cumulative Julia allocation, excluding peak process memory. Reproduction:
`validation/planar_audit/artwork_general_composite_allocations.jl` and
the locally archived `artwork_small_composite_prototype.jl`. The 9826 shared composite checks are
separate from the format-specific Gerber/ODB++ totals.

## Exact transformed bounds

Cached bounds for transformed artwork enclose the stored coordinates through
exact rational arithmetic. Their final Float64 conversion rounds the lower
bound downward and the upper bound upward, including subnormal and overflow
boundaries. Numerator and denominator precision follows their actual integer
bit lengths; quotient precision follows the Float64 significand. Caller
BigFloat precision and rounding settings remain unchanged. Public ODB symbol
imports retain these bounds when raster membership is evaluated.
