# I5 peak-process-RSS acceptance: the payload budgets bound declared array
# workspace only. This drives `peak_rss_probe.jl` in a fresh subprocess and
# gates its printed `Sys.maxrss()` high-water marks against an absolute
# ceiling calibrated on the workload, so an opaque-memory explosion (FFTW
# plans, BLAS workspace, leaked retained buffers) fails loudly. The
# stage-to-stage high-water delta additionally gates per-solve retention.

using Test

@testset "peak process RSS stays bounded across representative solve paths" begin
    probe = joinpath(@__DIR__, "peak_rss_probe.jl")
    load_path = join(("@", "@stdlib"), Sys.iswindows() ? ';' : ':')
    buffer = IOBuffer()
    proc = run(pipeline(
        addenv(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__, ".."))
            --startup-file=no $probe`,
            "JULIA_LOAD_PATH" => load_path,
            "DIFFMOM_REPO" => joinpath(@__DIR__, "..")),
        stdout = buffer, stderr = stderr))
    @test success(proc)
    lines = filter(l -> startswith(l, "RSSPROBE"), split(String(take!(buffer)), '\n'; keepempty = false))
    stages = Dict{String,Int}()
    for line in lines
        parts = split(line)
        stages[parts[2]] = parse(Int, parts[3])
    end
    @test haskey(stages, "baseline") && haskey(stages, "ufft_again")
    if !haskey(stages, "baseline")
        return
    end
    base = stages["baseline"]
    # The three probe stages complete under a ~1 GB peak over the loaded
    # baseline on the reference machine; the 6 GB absolute ceiling leaves
    # room for OS/runtime variance while still rejecting the multi-GB
    # runaway the payload budgets cannot express.
    ceiling = base + Int64(6) << 30
    for stage in ("dense", "ufft_drp", "ufft_again")
        @test stages[stage] <= ceiling
    end
    # Re-solving the same geometry at a new frequency must reuse the cached
    # operator footprint rather than accumulate a second copy.
    @test stages["ufft_again"] - stages["ufft_drp"] <= Int64(1) << 30
    # And the de-embedding feed-standard cascade (two extra solves per
    # unique wall section) stays within the dense solve's own envelope.
    @test stages["ufft_drp"] - stages["dense"] <= Int64(2) << 30
end
