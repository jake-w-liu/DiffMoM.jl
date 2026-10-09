# Constant nonreciprocal certificate and feedthrough preservation

Fresh public checks find two regressions in the unpublished scaled
certificate. A pole-free real nonreciprocal feedthrough has a PSD
Hermitian part but the new normalization returns uncertified because
the absent poles give a zero frequency scale. A negative minimum
subnormal feedthrough beside a maximum finite positive conductance
disappears during normalization and incorrectly certifies the model.
Both Julia versions reproduce both cases before this correction.

Constant feedthrough needs no pole scale. Check its Hermitian part
with the existing PSD predicate, including the negative-diagonal guard.
Share the stable real/complex Hermitian average with sampled passivity;
the helper precedes the original public docstring. A normalized nonzero
feedthrough that becomes zero cannot certify the original model.
No chosen numerical threshold or tolerance is introduced.

All 46 new constant/nonreciprocal/affine-capacitance/signed-subnormal,
ownership/inference/caller-state checks pass on both Julia versions.
The complete scoped rational controls also pass: 865 assertions each.
The known-bad 55620/7cfd publisher/own-CI queues and their idle whole-suite
successors were stopped after verifying identities. Their commits,
snapshots, earlier qualifications and failed local guards stay archived.

This corrected commit descends from the archived unpublished candidates
and preserves every prior current/power/selector/rational/pole/evaluation/
sampling/borrowing fix. Its publication base is the qualified physical
power parent 364e6; the intermediate known-bad heads will not become
separate main tips. Proper, native, docs, static, allocation, complete
suite and own hosted CI qualification still precede publication.
Broad numeric, resource, performance and planar/RFIC acceptance stays open.
