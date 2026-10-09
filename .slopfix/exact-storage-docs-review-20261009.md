# Rational API page and original documentation guard

The exact storage candidate passed all 1343 scoped rational assertions
on each Julia version, all 130 native/material/physical cases on each,
and the committed static gates. Its first local Documenter run failed:
the planar API page reached 200.38 KiB, exceeding Documenter's existing
200 KiB limit. The complete failed build/log/root remain preserved.

Move all six rational-model docstrings to a linked Planar Rational Models
API page and register it in the existing navigation. Keep every exported
docstring, doctest, signature, and the default page-size guard. The new
page explains sampled versus global certificate semantics and links the
primary positive-real lemma and Hamiltonian reference. No page threshold
or documentation gate is raised or skipped.

Production source and tests match the verified exact-storage parent byte
for byte. Their qualifications are retained with exact input hashes.
This documentation follow-up still needs its own complete doc build,
rendered-content review and fresh committed static qualification; whole
package/source and hosted/main publication checks remain required.
