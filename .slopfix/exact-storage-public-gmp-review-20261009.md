# Public GMP and MPFR dependency boundary

The exact stored-energy and KYP candidate fixes four independently
reproduced finite-input false certificates. Its first complete suite
stopped at the original ExplicitImports public-access gate: production
code used Base.GMP and Base.MPFR implementation bindings. The failed
full run and complete source/log/report are retained; Julia 1.12 full
was not run after that actual failure. No hosted run or main publication
was started for that candidate.

Use the declared public GMP_jll library and documented GMP C arithmetic
and quotient/remainder functions for the same owned BigInt buffers.
GMP_jll is already an MPFR dependency; declare its direct dependency and
major ABI compatibility. Read limb width from the loaded public GMP ABI
and derive bytes from UInt8 width. The standard MPFR_RNDU enumeration
value is checked through the loaded library's mpfr_print_rnd_mode.
No private-access exception, import allowlist, scientific tolerance,
resource gate or default shutdown deadline is changed.

All original 865 scoped controls plus 480 exact-energy/storage/ABI tests
pass on both Julia versions (1345 each). Original full import/Aqua
checks are separately exercised before complete qualification. These
scoped passes do not establish native, docs, allocation, proper, full
or hosted success for the new input hashes. Each of those is required
fresh before publication. Preserve the linked rational API page and its
default documentation size guard. Broad Sonnet/RFIC, numeric and
resource/performance audit remains active.
