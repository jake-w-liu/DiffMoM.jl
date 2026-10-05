# Project layout budget validation

`planar_project_layout` previously validated `max_bytes` only after building
layer-dependent project data. An invalid zero budget therefore allocated more
memory as the layer count increased, although no layout could be accepted.

The entry point now calls the existing resource-limit validator immediately
after frequency validation. The valid-input layout body is unchanged, and an
invalid frequency retains its original error priority. The resource validator
continues to enforce the existing positive, machine-integer budget domain.

Five warmed measurements for each of 2, 128 and 1024 layers produced these
allocation counts on Windows:

| Julia | Before, bytes by layer count | After, bytes by layer count |
| --- | --- | --- |
| 1.13.1 | 13216, 107968, 798224 | 720, 720, 720 |
| 1.12.7 | 13344, 108096, 798352 | 848, 848, 848 |

The same zero-budget `ArgumentError` was returned in every measurement.
Independent comparison rehashed all 138 tracked Julia source files, verified
unchanged inputs during each capture, and checked that the production change
consists solely of the early validator call. The original project test file
remains an exact prefix of the updated file.

Thirteen appended assertions cover allocation at the three layer counts,
unchanged project input, negative and out-of-range budgets, and frequency
error priority. Nine focused project, component-return, native FLOAT and power
wave suites passed on both Julia versions. The first focused runner omitted a
fixture-producing suite; its failed logs are retained, and the corrected runner
includes that dependency.

Ignored audit artifacts in `data/project_invalid_budget_*`,
`data/project_budget_allocation_comparison_v1_20261005.*`, and
`data/project_budget_focused_v2_*` contain the measurements and source hashes.
This result concerns rejection of invalid budgets. It does not establish peak
resident memory, performance for positive budgets, or complete Sonnet parity.
Existing numerical tolerances, reference fixtures and CI workflow are unchanged.

The pinned Julia 1.12.7 code counter measures 212835 lines, an increase of 28
from the existing 212807 ceiling. The ceiling adjustment is a separate commit
for this measured addition. Blocking smells pass; the unchanged duplication
ceiling is 58394 lines, with a bounded 400-group census rather than an exhaustive
duplication proof.
