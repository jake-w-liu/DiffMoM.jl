# Finite-Space Error and RCS Screening

Use these functions to model the field difference between nested RWG spaces
on the same faceted surface. The Gaussian model describes an unresolved
finite-space correction. It does not by itself bound the continuum solution,
geometry error, or the probability of a correct mask decision.

## Prepare the nested problem

Retain a forward solve with `solve_scattering(...; return_state=true)`, then
construct its fixed-facet refinement with `build_nested_rwg_pair`. The pair
contains the coarse injection `P` and coordinate complement `Q`.

`prepare_galerkin_error` takes the retained state, pair, fine operator, and
fine right-hand side. It checks the restricted coarse equations. If independently
assembled coarse and fine quadratures disagree, the default behavior constructs
a separate restricted coarse solve; the original forward state is unchanged.
Set `rebuild_on_mismatch=false` to reject that discrepancy instead. Inspect
`restriction_report` before interpreting the correction.

For the transformed fine operator with blocks `A`, `B`, `C`, and `D`, the
unresolved system is `S*w=r`, where `S=D-C*(A\B)`. The complete field correction
uses `H=Gf*Q-Gf*P*(A\B)`, including the radiation of the unresolved basis.
The implementation obtains selected rows through coarse adjoint solves and
fine-operator actions rather than forming a dense current covariance.

## Condition and propagate

Call `condition_discretization_error(system, probes; tau=...)` with a positive
prior scale and either unresolved row indices or a probe matrix `R`. The
observations are `R*S*w=R*r`. Optional positive triangle weights define the
field-mass precision; omitted weights give unit precision.

`rcs_output_map` supplies two complex transverse rows per look, ordered as
theta then phi. Use `prepare_error_outputs` once when comparing several probe
sets with the same system and output map. Pass that object to
`evaluate_error_outputs` to reuse its adjoint work.

`output_residual_probes` constructs operator/output-dependent rows from a
diagonal approximation. It does not use a reference solution to select them.
Their usefulness depends on the problem; the function does not promise an
accuracy or cost improvement.

`sample_conditioned_error` and `sample_error_outputs` require an explicit random
number generator. Output samples share latent coordinates across all looks.
`quadratic_output_moments` returns moments under the working Gaussian model,
not calibrated RCS decision probabilities.

Retained states, nested pairs, conditioning data, and prepared outputs must not
be mutated behind the API. A new forward right-hand side invalidates error
objects tied to the old revision. Rebuild those objects before reuse.

Cancellation-sensitive residual checks have a separate work budget:
`max_true_residual_exact_terms`. Set it when retaining the initial solve if a
larger coarse system needs more than the default two million terms per residual
evaluation. Restricted coarse states inherit that limit. Repeated forward and
adjoint solves can override it for one call; increasing the work budget does not
relax the residual tolerance or bypass verification.

The runnable `examples/experiments.jl` checks a small full-information case
against a directly solved fine system. That solve is an algebraic comparison,
not a demonstration of savings over fine enrichment.

## Calibrate whole cases

`rcs_case_score` takes the maximum normalized complex-field error over every
look and every candidate level in one case. Split training and calibration by
case, not by individual look or level. `calibrate_rcs` uses the augmented
order statistic, including an infinite endpoint when the calibration sample
is too small for the requested error rate.

`RCSCalibrationContext` identifies the frozen algorithm, training revision,
case distribution, reference, looks, levels, and numerical settings. The caller
must supply the actual provenance identifiers and determine whether a new case
belongs to that distribution. A well-formed hash alone is not evidence of its
provenance.

`screen_rcs_mask` returns pass, fail, or unresolved from field-ball RCS bounds.
It requires a matching context, a supported-case assertion, and a
`NumericalErrorBudget` covering algebraic, quadrature, compression, restriction,
reference, and geometry errors. Missing checks do not count as zero error.
The checks can be scalar or per look. Outside those conditions, the result is
unresolved rather than a mask decision.

Casewise calibration concerns marginal coverage for exchangeable cases under
the frozen procedure. It does not provide a conditional-on-pass guarantee or
a guarantee for an out-of-distribution case.

## Source reference

```@autodocs
Modules = [DiffMoM]
Pages = ["error_estimation/GalerkinError.jl", "error_estimation/Conditioning.jl", "error_estimation/Calibration.jl"]
Private = false
```
