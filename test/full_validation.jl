# Full scientific, native-reference, stress and regression validation.
# Run: julia --project=. --startup-file=no -e 'using Pkg; Pkg.test(test_args=["full"])'
# Sequential processes release compiler state between groups. A failed child
# propagates to Pkg.test; the success banner is printed only after every group.
include("test_quality.jl")
include("full_reference_files.jl")

function run_full_validation_group(files)
    worker = joinpath(@__DIR__, "full_validation_worker.jl")
    project = dirname(Base.active_project())
    run(`$(Base.julia_cmd()) --startup-file=no --project=$project
         --threads=$(Threads.nthreads()) --heap-size-hint=1G $worker $files`)
end

function run_full_validation_files(files; group_size=8)
    for first in 1:group_size:length(files)
        last = min(first + group_size - 1, length(files))
        run_full_validation_group(files[first:last])
    end
end

run_full_validation_files(FULL_REFERENCE_BEFORE_CORE)
run_full_validation_group(["full_core.jl"])
run_full_validation_files(FULL_REFERENCE_AFTER_CORE)
println("\nFULL VALIDATION PASSED")
