using DiffMoM,LinearAlgebra,Statistics,SHA,TOML

const repo=dirname(dirname(pathof(DiffMoM)))
const fixture=joinpath(repo,"test/test_planar_fft_dense_workspace.jl")
include_string(Main,first(split(read(fixture,String),"@testset")),fixture)

function main()
    root=isempty(ARGS) ? joinpath(repo,"data/planar_audit") : abspath(only(ARGS))
    mkpath(root)
    directory=mktempdir(root;prefix="folded_ufft_resources_",cleanup=false)
    cp(@__FILE__,joinpath(directory,"producer.jl"))
    paths=vcat([@__FILE__,fixture],[joinpath(d,f) for (d,_,fs) in walkdir(joinpath(repo,"src")) for f in fs])
    hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
    report=Dict{String,Any}("scope"=>"Alternating warmed retained-modal and folded-spectrum construction/matvec measurements. Julia array " *
        "payload and cumulative allocations exclude opaque FFT plans and peak process memory; independent " *
        "direct-modal action and diagonal gates remain2e-12.",
        "source_before"=>hashes(),"cases"=>Any[])
    for walls in (WALL_PEC,WALL_PMC)
        prob=_dense_workspace_fixture(walls)
        kw=(;mx=128,my=96,surface_zs=[.3+.2im,.5+.1im],via_sigma=5.8e7,volume_sigma=4e7)
        construct(fold)=planar_ufft_operator(prob,8e9;kw...,_fold_iterative=fold)
        retained=construct(false);folded=construct(true)
        @assert folded.folded!==nothing && retained.folded===nothing
        x=ComplexF64[sin(p)+im*cos(p) for p in 1:retained.n];out=similar(x)
        modal=assemble_planar_z(prob.stack,prob.grid,prob.sheets,prob.basis,2pi*8e9;
            vias=prob.vias,vols=prob.vols,kw...)
        error=norm(folded*x-modal*x)/norm(modal*x)
        derror=norm(DiffMoM._planar_ufft_diagonal(folded)-diag(modal))/norm(diag(modal))
        @assert error<2e-12 && derror<2e-12
        for _ in 1:3;construct(false);construct(true);mul!(out,retained,x);mul!(out,folded,x);end
        constructors=Dict(false=>Float64[],true=>Float64[])
        actions=Dict(false=>Float64[],true=>Float64[])
        for sample in 1:10,fold in (isodd(sample) ? (false,true) : (true,false))
            GC.gc();push!(constructors[fold],@elapsed construct(fold))
            A=fold ? folded : retained
            push!(actions[fold],@elapsed mul!(out,A,x))
        end
        oldbytes=minimum(@allocated(construct(false)) for _ in 1:3)
        bytes=minimum(@allocated(construct(true)) for _ in 1:3)
        actionbytes=minimum(@allocated(mul!(out,folded,x)) for _ in 1:3)
        @assert bytes<.3oldbytes && actionbytes==0
        row=Dict("walls"=>string(walls),"relative_action_error"=>error,"relative_diagonal_error"=>derror,
            "retained_owned_array_bytes"=>DiffMoM._subdivision_retained_payload(retained),
            "folded_owned_array_bytes"=>DiffMoM._subdivision_retained_payload(folded),
            "retained_construction_gc_bytes"=>oldbytes,"folded_construction_gc_bytes"=>bytes,
            "folded_matvec_gc_bytes"=>actionbytes,"retained_construction_seconds"=>constructors[false],
            "folded_construction_seconds"=>constructors[true],"retained_matvec_seconds"=>actions[false],
            "folded_matvec_seconds"=>actions[true],"retained_construction_median"=>median(constructors[false]),
            "folded_construction_median"=>median(constructors[true]),"retained_matvec_median"=>median(actions[false]),
            "folded_matvec_median"=>median(actions[true]))
        push!(report["cases"],row);println(row);flush(stdout)
    end
    report["source_after"]=hashes()
    report["source_unchanged"]=report["source_before"]==report["source_after"]
    report["status"]=report["source_unchanged"] ? "PASS" : "FAIL"
    open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
    println("Retained folded FFT resource evidence: ",directory)
    @assert report["status"]=="PASS"
end
main()
