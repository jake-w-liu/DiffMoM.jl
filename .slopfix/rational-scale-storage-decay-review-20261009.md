# Rational fitting scale, stored inputs and finite decay

Uniform response scaling cancels from the shared-pole relocation equation.
The implementation normalizes from the actual finite component maximum
before QR projection, retains an exact zero-data shortcut, and uses the
standard LinearAlgebra numerical-rank convention: the smaller matrix
dimension times working epsilon and the largest singular value. This
removes absolute response floors that stopped small but finite fits.

Online scaled sums of squares preserve finite fit-error metadata when
ordinary squaring underflows or overflows. Binary exponents retain the
representable final relative ratio. Statistics reuse predicted matrices
without allocating a separate difference matrix. Measured statistics
allocation is 14688 versus 7344 bytes across four amplitudes on both Julia
versions; this is a scoped allocation result, not a global timing claim.

Dense Y/S intake checks each component through the existing stored-phasor
contract before conversion can erase a nonzero wide value or introduce
infinity. Dense, sparse and actual views plus ordinary Float32/BigFloat
inputs remain supported. Zero-byte preflight and rejection occur before
frequency-dependent reference providers are evaluated.

Relocated unstable pole real parts are reflected without a fabricated
normalized decay floor. Transient zero real parts may occur during the
caller's existing iteration sequence. The final poles must be finite and
strictly decaying. Epsilon-scaled slow-decay data now passes the unchanged
original fit residual gate. The ill-conditioned example does not establish
accurate pole identification or extrapolation outside the sampled band.

The exact isolated runtime passes both Julia versions: original 69 rational
assertions, 40 coupled/rank/amplitude cases, independent promoted RMS
references, intake and slow-decay regressions, proper package guards with
1101 passes and the upstream expected negative-guard broken assertion,
original conformal 4532 and project 422 assertions, archived native 130
cases per version, and original documentation. New durable tests add 148
assertions. Original tolerances, fixtures, quality gates and CI are retained.

Full original suites and current parent own CI, measured/static/accounting
qualification, main publication and this candidate's own hosted CI remain
required. This candidate adds only PlanarVectorFit.jl to the preceding
retained-excitation candidate. Remaining iteration, initialization,
conjugate-pair sorting, sampled passivity and certificate numeric policies
and the broad resource/physics/RFIC audit remain open.

This candidate is rebased onto the corrected retained-wave selector after
the earlier parent failed the unchanged original radiation comparison.
The original scoped rational results above describe the earlier isolated
runtime. The corrected combined runtime is being requalified with proper
package 1635 assertions, current native references, documentation, complete
suites and static checks. The obsolete old-parent accounting commit remains
in the archive reference; the new ceiling requires an exact fresh count.
No original fixture, tolerance, test gate or CI configuration is changed.

The retained parent is now revised again to test terminal consistency before
forming unnormalized retained-Y coefficients. The earlier cf1d parent and
ce10 rational candidate passed both complete suites but a new passive
normalized-reference control demonstrated eager coefficient overflow.
Those full-pass logs and archived commits remain; they do not qualify the
new source. This rebased candidate preserves the canonical per-coordinate
15-assertion range regression and the unchanged four rational fixes/148 tests.
Current combined proper1650, native130, documentation240, complete suites,
exact static accounting and own hosted verification are required anew.
No original fixture, tolerance, test gate or CI configuration was changed.
