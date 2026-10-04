using DiffMoM,LinearAlgebra,SHA,TOML
const repo=normpath(joinpath(@__DIR__,".."))
function main()
    previous=joinpath(repo,"data/sonnet_validation/geovar_zero_nominal_axes_GAt2Wu")
    directory=mktempdir(joinpath(repo,"data/planar_audit");prefix="zero_nominal_axes_literal_physics_",cleanup=false)
    cp(@__FILE__,joinpath(directory,"producer.jl"))
    paths=vcat([@__FILE__],[joinpath(dir,file) for (dir,_,files) in walkdir(joinpath(repo,"src")) for file in files],
        [joinpath(dir,file) for (dir,_,files) in walkdir(previous) for file in files])
    hashes()=Dict(path=>bytes2hex(sha256(read(path))) for path in paths)
    report=Dict{String,Any}("scope"=>"Before zero-NOM implementation, independent literals through complete public denseFFT solver; fixed .005 full-S and1e-9 original voltage residual gates. Not continuum convergence.","source_before"=>hashes(),"cases"=>Any[])
    try
        for axis in ("XDIR","YDIR"),direction in (-1,1),target in (.0625,.125)
            tag="$(axis)_dir_$(direction)_target_$(target)_literal"
            source=joinpath(previous,tag*".son")
            reference=only(planar_read_touchstone(joinpath(previous,tag,"native_raw.s2p")).s)
            p=read_sonnet_project(source)
            result=solve_sonnet_project(p,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
            raw=result.raw;rhs=zeros(ComplexF64,size(raw.currents))
            for b in eachindex(raw.problem.basis.port)
                port=raw.problem.basis.port[b];iszero(port) && continue
                rhs[b,port]=(port==1 ? -1. : 1.)*raw.problem.basis.width[b]
            end
            row=Dict{String,Any}("axis"=>axis,"direction"=>direction,"target"=>target,"full_s_error"=>maximum(abs,result.s-reference),
                "original_voltage_residual"=>norm(raw.z_mom*raw.currents-rhs)/norm(rhs))
            row["status"]=row["full_s_error"]<=.005 && row["original_voltage_residual"]<=1e-9 ? "PASS" : "FAIL"
            push!(report["cases"],row);println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        report["status"]=all(row["status"]=="PASS" for row in report["cases"]) ? "PASS" : "FAIL"
        open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained literal physics: ",directory)
    end
    @assert report["source_unchanged"]
    @assert length(report["cases"])==8
end
main()
