# Isolated runtime overlay: no repository production file is changed.
# Source diagnostics/0-or-1 trailing field behavior otherwise remain unchanged.
Base.include_string(DiffMoM, read(joinpath(@__DIR__,"circuit_return_function.jl"),String),
    joinpath(@__DIR__,"circuit_return_function.jl"))
Base.include(DiffMoM, joinpath(@__DIR__,"project_return_function.jl"))
