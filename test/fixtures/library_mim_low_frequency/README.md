# Public MIM low-frequency overlap reference

Six current public solves cover 16/32/64 cells at 100/500 MHz. Literal plate
plus lead overlap is 7.03125e-8 m2; epsilon0*7.5*area/1e-6 gives
4.6692006066210936 pF independently of production capacitance estimators.
The unchanged approximation gate is 5%; original voltage equations use 1e-9.
Observed errors are 1.14-4.19%, residuals below 1.55e-12. Lead and fringing
effects are part of the solved response. This is a bounded finite-grid
electrostatic approximation check, not measured-device or continuum proof.

The initial successful run guards six named production files. The final
successful run additionally guards both fixture and validation helpers;
all eight before/after hashes match the retained source snapshots. These
guards do not certify every package source. The full immutable package
checkpoint is separate. No native engine is invoked by this analytic check;
the native MIM matrix archive is a distinct acceptance reference.

The registered test replays the complete current public solves, original
voltage residuals, reciprocity/passivity and immutable archive hashes.
