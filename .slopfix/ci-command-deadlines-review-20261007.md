# Explicit command deadlines and complete CI evidence

The exact main run37595089374 passed all five platform jobs and documentation.
Its strict Julia test command was killed at the inherited3600-second deadline.
The retained artifact shows continuing test output through later testsets and
no protected-source changes. All other strict command gates passed. Explicit
package initialization did not resolve the cutoff; that failed evidence is
preserved rather than relabeled as a billing failure.

Quality command configurations now require an explicit deadline choice:
`null` waits for completion; a positive integer remains a caller-specified
budget within Python's platform wait limit. Julia templates and this
repository's command gates use `null` because no project requirement justified
their former generic durations. The quality job uses GitHub's documented
runner deadline instead of a separately invented repository cutoff. This can
allow a hung command to run until the hosting service ends the job; it does
not establish termination of arbitrary programs.

Every required command, numerical/static/import/resource assertion, captured
output, protected-source check and owned-process cleanup remains enforced.
Regression controls cover null/explicit/malformed/missing choices, preserved
nonzero exit failures, explicit timeout cleanup, surviving descendants and
protected-file mutation. Windows retains its explicit command-runner limitation;
mocked ownership controls do not establish POSIX execution. The actual hosted
strict contract must finish before the published commit is called green.

Primary API references:

- https://docs.python.org/3/library/subprocess.html#subprocess.Popen.wait
- https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idtimeout-minutes

The exact deadline-fix main run37611819035 subsequently passed all seven
hosted jobs and all four complete strict commands. Independent log review
verified original assertion counts after stripping ANSI presentation.

The platform120-minute override followed genuine Windows cancellations
at30 and60 minutes, but those observations do not derive120 as a bound.
No project requirement was recorded for the docs20-minute override. Both
repository overrides are removed; every job now uses the documented
GitHub hosting deadline. No new numeric timeout is introduced, and no
job, step, numerical target or required command is skipped or softened.
GitHub service policy remains an external operational constraint.

https://docs.github.com/en/actions/reference/limits
