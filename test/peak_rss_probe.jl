# Peak-process-RSS probe for the I5 acceptance item. Runs representative
# heavy solve paths in one fresh process and prints `RSSPROBE <stage> <bytes>`
# high-water marks after each stage so the driver test can gate process-level
# memory against the declared payload budgets (which exclude opaque FFTW
# plans, BLAS workspaces and runtime overhead).

using DiffMoM
using Printf

repo = get(ENV, "DIFFMOM_REPO", joinpath(@__DIR__, ".."))
fixture = joinpath(repo, "test", "fixtures", "native_drp_reference_planes")

function report(stage, f)
    t = time()
    r = f()
    GC.gc(true)
    @printf("RSSPROBE %s %d %.1f\n", stage, Sys.maxrss(), time() - t)
    return r
end

println("RSSPROBE baseline ", Sys.maxrss(), " 0.0")

# Stage 1: dense native solve — BLAS/LU opaque workspace plus the retained
# dense operator.
proj = DiffMoM.read_sonnet_project(joinpath(fixture, "drp_base.son"))
report("dense", () -> solve_sonnet_project(proj, 2.5e9; raw = true))

# Stage 2: the de-embedding path — main UFFT solve plus the two generated
# feed-standard solves per unique wall cross-section (FFTW plans + Krylov
# workspace on top of the payload budget).
report("ufft_drp", () -> solve_sonnet_project(proj, 2.5e9; method = :ufft))

# Stage 3: same geometry at a second frequency — the retained operator and
# cached standards must not duplicate their process footprint.
report("ufft_again", () -> solve_sonnet_project(proj, 3.0e9; method = :ufft))
