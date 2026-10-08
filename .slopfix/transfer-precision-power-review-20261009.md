# Layer transfer precision and accepted power

The former fixed cubic in layer ABCD transfer lost the public BigFloat
working-type approximation contract. Layer transfer now uses the existing
convergent even series in its unit disk, with coefficients and termination
in working arithmetic. No selected tolerance, precision, order or retry
limit is introduced.

Receiving moments now choose endpoint arithmetic from the actual
exponential and subtraction conditioning. The former real(x)>20 split
lost decaying Maxwell integrals under the original field tolerance. The
selector preserves the existing representation at propagating nodes.

The result-based far-field route retains every computed positive accepted
power. A coherent caller-retained linear model reproduces the loss of
small positive admittance conductance; the actual retained LU solution,
port extraction and residual are checked. This fixture does not claim an
assembled electromagnetic solve. The existing negative-roundoff guard
and project power-wave policies require separate audit.

The durable 121 assertions cover decaying moments, public layer cascade
working precision and the unit boundary, native dual cutoff slopes,
coherent accepted power, and caller precision restoration. Original
physical fixtures and field/residual tolerances are retained. Historical
20 and former polynomial witnesses appear only in regression inputs.

Fresh scoped checks, all 130 native-reference cases per Julia version,
proper package tests and original documentation checks pass on the exact
combined runtime. Every measured Float64 private transfer and receiving
call allocates zero bytes; public cascade workspace allocations match
the parent. BigFloat allocates more for the precision-correct recurrence;
the measured cost is recorded without a global efficiency claim.

Complete original suites, exact accounting and static gates, publication
checks and this candidate's own hosted CI remain required. The public
layer-state conditioning investigation and the broader numerical,
resource and RFIC audit remain open.
