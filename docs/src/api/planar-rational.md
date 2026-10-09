# Planar Rational Models

Shared-pole rational admittance models describe real, stable N-port
responses and export an equivalent linear SPICE subcircuit. See the
[planar formulation](../formulations/06-planar-spectral-mom.md) for
solver conventions and [Planar Networks and Outputs](planar-networks.md)
for native network and circuit APIs.

`planar_rational_passivity` reports the supplied frequency grid.
`planar_rational_certificate` checks sufficient global positive-real
conditions and, when needed, verifies a storage-energy inequality for
the original stored coefficients. A returned false certificate can
mean that the proof is inconclusive. Passivity repair records its
conductance and capacitance changes and recomputes fitting error.

The storage inequality follows the positive-real lemma in
[Boyd et al., section 2.7.2](https://stanford.edu/~boyd/lmibook/lmibook.pdf).
Potential crossing diagnostics follow the full Hamiltonian criterion in
[Semlyen and Gustavsen (2009)](https://doi.org/10.1109/TPWRD.2008.923406).
Numerical absence of a crossing alone cannot certify stored energy.

```@docs
PlanarRationalModel
planar_fit_rational
planar_rational_eval
planar_write_spice
planar_rational_passivity
planar_rational_certificate
```
