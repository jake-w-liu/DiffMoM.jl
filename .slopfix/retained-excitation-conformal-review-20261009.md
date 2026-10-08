# Retained terminal excitation, power and geometry rounding

Finite physical excitation and loss can survive after rounded S parameters
reach an exact short or lossless boundary. Radiation now uses retained
admittance for terminal excitation and peak V/I power, and loaded projects
use external differential terminal voltages and currents from their actual
MNA solution. Explicit no-circuit voltages remain authoritative. The S-only
route retains its declared S representation. Every computed positive power
remains available for gain normalization.

The retained-admittance excitation solves the physical Kurokawa boundary
equation. Its row scale comes from actual coefficients, and its RHS uses
binary exponents. Already stored incident waves and references are borrowed
read-only; outputs own their storage. Budget accounting follows the actual
helper branch and typed array payloads. No selected excitation threshold,
precision, retry or iteration limit is added.

Conformal Float64 geometry now explicitly retains nearest-even conversion
under unrelated caller MPFR directed rounding. Exact orientation and
intersection precision derives from the IEEE exponent span and polynomial
degree, including the final Float64 midpoint rounding guard.

Independent exact geometry and public mesh references, all original 4532
conformal and 422 project assertions, actual native 130 cases per Julia
version, original documentation, and proper package guards pass on the
exact isolated runtime. New durable regressions add 18 geometry and 224
terminal excitation/power/coupled/reference/derivative/ownership/rejection
assertions. Original fixtures and scientific accuracy gates remain intact.
Retained EM witnesses use coherent caller-provided LU/source/port-current
data; they do not claim an assembled EM solve. Loaded resistor witnesses
use the real native MNA solver.

Scoped allocation measurements improve for both versions: five-port
forward conversion falls from 1120 to 832 bytes and inverse conversion
from 528 to 384 bytes relative to the correct earlier implementation.
This is not a whole-program timing, peak-memory or optimization claim.

Complete original suites, measured accounting/static gates, publication
checks and this candidate's own CI remain required. Existing negative
roundoff guards and the broader numeric/resource/RFIC audit remain open.
Later rational-fit repairs are a separate isolated candidate scope.
