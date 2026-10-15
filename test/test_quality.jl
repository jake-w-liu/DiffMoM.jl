using LinearAlgebra
using SparseArrays
using StaticArrays
using Statistics
using Random
using Test
using Aqua
using SpecialFunctions: erfc, sphericalbesselj, sphericalbessely
using ExplicitImports: check_all_explicit_imports_are_public,
                       check_all_explicit_imports_via_owners,
                       check_all_qualified_accesses_are_public,
                       check_all_qualified_accesses_via_owners,
                       check_no_implicit_imports,
                       check_no_self_qualified_accesses,
                       check_no_stale_explicit_imports,
                       improper_qualified_accesses
using CSV
using DataFrames
using DiffMoM

Aqua.test_all(DiffMoM; stale_deps=(ignore=[:CSV, :DataFrames, :JSON],))

# Reviewed non-public implementation boundaries; new names fail this gate.
# EzXML 1.2.3 documents StreamReader/XMLError as public but does not export them.
const _NONPUBLIC_QUALIFIED_ACCESS_ALLOWLIST = (
    :AbstractSparseMatrixCSC, :IEEEFloat, :ILUFactorization, :RefValue,
    :StreamReader, :UMFPACK, :XMLError, :checked_add, :checked_mul,
    :decompose, :gecon!, :mightalias, :promote_op, :setindex,
)

@test check_no_implicit_imports(DiffMoM) === nothing
@test check_no_stale_explicit_imports(DiffMoM) === nothing
@test check_all_explicit_imports_via_owners(DiffMoM) === nothing
@test check_all_explicit_imports_are_public(DiffMoM) === nothing
@test check_all_qualified_accesses_via_owners(DiffMoM) === nothing
@test check_no_self_qualified_accesses(DiffMoM) === nothing
@test check_all_qualified_accesses_are_public(
    DiffMoM; ignore=_NONPUBLIC_QUALIFIED_ACCESS_ALLOWLIST) === nothing

nonpublic_qualified_accesses = Set(
    access.name
    for (_, accesses) in improper_qualified_accesses(DiffMoM)
    for access in accesses
    if !access.public_access
)
@test nonpublic_qualified_accesses == Set(_NONPUBLIC_QUALIFIED_ACCESS_ALLOWLIST)
@test Docs.hasdoc(DiffMoM, :planewave_dda_3d)
