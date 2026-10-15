# Each full-validation group owns a fresh process and releases its compiler state.
using LinearAlgebra, SparseArrays, StaticArrays, Statistics, Random, Test
using SpecialFunctions: erfc, sphericalbesselj, sphericalbessely
using CSV, DataFrames, DiffMoM

files = copy(ARGS)
empty!(ARGS)
for file in files
    println("\nFull validation file: ", file)
    include(joinpath(@__DIR__, file))
end
println("Full validation group passed; peak RSS bytes: ", Sys.maxrss())
