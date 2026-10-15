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

shard=parse(Int,get(ENV,"DIFFMOM_FULL_SHARD","1"))
shards=parse(Int,get(ENV,"DIFFMOM_FULL_SHARDS","1"))
groups=full_reference_groups()
1<=shard<=shards<=length(groups) || throw(ArgumentError("invalid full-validation shard selection"))
for (index,files) in enumerate(groups)
    mod(index-1,shards)+1==shard || continue
    println("\nFull validation group $index/$(length(groups)), shard $shard/$shards")
    run_full_validation_group(files)
end
println(shards==1 ? "\nFULL VALIDATION PASSED" : "\nFULL VALIDATION SHARD $shard/$shards PASSED")
