using DiffMoM,LinearAlgebra,TOML,SHA
function main()
    repo=normpath(joinpath(@__DIR__,".."));native=abspath(ARGS[1])
    sources=[joinpath(dir,file) for (dir,_,files) in walkdir(joinpath(repo,"src")) for file in files if endswith(file,".jl")]
    push!(sources,@__FILE__)
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in sources)
    report=Dict{String,Any}("scope"=>"current public raw independent RAD literal coupons before implementation; full-S gate .005, original voltage residual 1e-9, mx=my=128", "source_before"=>hashes(),"cases"=>Any[])
    try
        for tag in ("expand_radial_literal","contract_radial_literal")
            p=read_sonnet_project(joinpath(native,tag*".son"))
            result=solve_sonnet_project(p,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
            reference=only(planar_read_touchstone(joinpath(native,tag,"native_raw.s2p")).s)
            raw=result.raw;rhs=zeros(ComplexF64,size(raw.currents))
            for b in eachindex(raw.problem.basis.port)
                port=raw.problem.basis.port[b];iszero(port) && continue
                rhs[b,port]=(port==1 ? -1. : 1.)*raw.problem.basis.width[b]
            end
            full_s_error=maximum(abs,result.s-reference);residual=norm(raw.z_mom*raw.currents-rhs)/norm(rhs)
            row=Dict("case"=>tag,"full_s_error"=>full_s_error,"original_voltage_residual"=>residual,"status"=>full_s_error<=.005 && residual<=1e-9 ? "PASS" : "FAIL")
            push!(report["cases"],row);println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(@__DIR__,"radial_literal_physical_gate_20261004.toml"),"w") do io;TOML.print(io,report);end
    end
    @assert report["source_unchanged"] && all(row->row["status"]=="PASS",report["cases"])
end
main()
