using DiffMoM, Test, SHA, TOML, LinearAlgebra, Krylov
BLAS.set_num_threads(1)
root=realpath(pwd())
files=split(read(`git -C $root ls-files -z -- src`,String),'\0';keepempty=false)
append!(files,["test/runtests.jl","test/test_planar_conformal_defect.jl",
    "test/test_planar_conformal_defect_refinement.jl"])
hashes()=Dict(name=>bytes2hex(sha256(read(joinpath(root,name)))) for name in files)
before=hashes()
include(joinpath(root,"test/test_planar_conformal_defect.jl"))
include(joinpath(root,"test/test_planar_conformal_defect_refinement.jl"))
after=hashes();@assert before==after
report=Dict("status"=>"PASS","version"=>string(VERSION),"arch"=>string(Sys.ARCH),
    "cpu"=>Sys.CPU_NAME,"krylov_version"=>string(pkgversion(Krylov)),
    "head"=>readchomp(`git -C $root rev-parse HEAD`),"source_before"=>before,
    "source_after"=>after,"source_unchanged"=>true,"inputs"=>length(files),
    "scope"=>"Actual production module, existing conformal tests and all new assertions; no process-only replacement or tolerance change.")
println("BEGIN_CONFORMAL_PRODUCTION_ARM_PROOF_TOML")
TOML.print(stdout,report)
println("END_CONFORMAL_PRODUCTION_ARM_PROOF_TOML")
println("ACTUAL_COMPENSATED_PRODUCTION_ARM_PROOF_PASS")
