# Counting helpers follow subprocess completion

The original SCC identity/batch and Julia identity/batch helpers pass
selected 30/900/120/900-second deadlines to subprocess.run even though
the top-level quality commands have no elapsed deadline. Boundary spies
verify those four actual arguments. Remove the two selected batch-limit
constants and pass None in all four subprocess calls. Completed outputs,
missing/malformed records, nonzero exits and reported SCC failures retain
their existing validation. If an external invocation raises TimeoutExpired,
its actual exception timeout supplies the diagnostic instead of an internal
selected number. No interpreter/library identity or token classification
algorithm changes; the Julia helper and all142 runtime inputs are identical
to the layout parent. No scientific tolerance or CI gate changes.

All41 Python validation tests pass in the full tracked source tree, including
four new regression methods with identity/batch, completion, nonzero/report,
missing/ERR and spawn-error subcases. The first new SCC test used an already
prefixed mock version string; its preserved failure was corrected with a
realistic SCC version fixture and unchanged production semantics. An initial
scripts/tests-only projection lacked validation dependencies and could not
run the entire Python suite. The full tracked tree corrects that projection.
The first complete runner used -X utf8 without propagating UTF-8 to child
Python processes on Windows; the preserved decoding failure was corrected
by inheriting the runner encoding and UTF-8 flag in its child environment.
No product-test assertion was weakened to pass these preparation corrections.

Exact pinned count, all static quality gates, actual interpreter count parity,
the complete layout parent and its hosted CI, fresh clean-tree verification,
main push and the new hosted run remain required. Unchanged Julia runtime,
native, docs and full-suite results can be reused only after the exact parent
has completed them; this Python-only change does not require another local
scientific rerun. New hosted CI still executes the original workflow.

This closes these four helper defaults. It does not claim all selected
numerics in the repository have been removed or that the broad
Sonnet/RFIC/resource/performance audit is complete.
