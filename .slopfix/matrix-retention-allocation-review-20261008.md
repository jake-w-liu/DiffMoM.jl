# Complete matrix-payload retention allocation check

An unchanged full Julia test run missed the whole-call allocation-difference
threshold by 92 bytes. Complete allocation tracing on both supported Julia
versions showed that the retained solve allocates two full matrix buffers
and the compact solve allocates one. The required full matrix payload is
saved; whole-call metadata and allocation-counter variation obscured it.

The allocation assertion now measures every matrix-sized data-buffer event
from the same two complete solve paths and still requires one full matrix
payload saved. All original numerical, ownership, resource-rejection and
file-preservation assertions remain. The arbitrary batch and repeat counts
are removed, with no byte allowance or skipped gate. Profiler buffers are
cleared in finally. Profile is declared as a standard-library test dependency.
Both original focused resource modules pass; fresh full suites and hosted CI
remain required. No production solver or physical equation changes here.
