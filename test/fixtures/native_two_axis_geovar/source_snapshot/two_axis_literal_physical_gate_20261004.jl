using DiffMoM,LinearAlgebra,TOML,SHA
function main()
    repo=normpath(joinpath(@__DIR__,".."))
    native=joinpath(repo,"data/sonnet_validation/two_axis_geovar_HagV1m")
    sources=[joinpath(dir,file) for (dir,_,files) in walkdir(joinpath(repo,"src")) for file in files if endswith(file,".jl")]
    push!(sources,@__FILE__)
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in sources)
    report=Dict{String,Any}("scope"=>"current public raw independent SCXY literal coupons before implementing the mode; declared full-S gate .005 and original voltage residual 1e-9; fixed mx=my=128",
        "source_before"=>hashes(),"cases"=>Any[])
    try
        for native_case in TOML.parsefile(joinpath(native,"comparison.toml"))["cases"]
            tag=native_case["tag"]
            p=read_sonnet_project(joinpath(native,tag*"_proportional_literal.son"))
            result=solve_sonnet_project(p,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
            reference=only(planar_read_touchstone(joinpath(native,tag*"_proportional_literal/native_raw.s2p")).s)
            raw=result.raw;rhs=zeros(ComplexF64,size(raw.currents))
            for b in eachindex(raw.problem.basis.port)
                port=raw.problem.basis.port[b];iszero(port) && continue
                rhs[b,port]=(port==1 ? -1. : 1.)*raw.problem.basis.width[b]
            end
            full_s_error=maximum(abs,result.s-reference)
            original_voltage_residual=norm(raw.z_mom*raw.currents-rhs)/norm(rhs)
            row=Dict("case"=>tag,"full_s_error"=>full_s_error,"original_voltage_residual"=>original_voltage_residual,
                "status"=>full_s_error<=.005 && original_voltage_residual<=1e-9 ? "PASS" : "FAIL")
            push!(report["cases"],row);println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(@__DIR__,"two_axis_literal_physical_gate_20261004.toml"),"w") do io;TOML.print(io,report);end
    end
    @assert report["source_unchanged"] && all(row->row["status"]=="PASS",report["cases"])
end
main()
