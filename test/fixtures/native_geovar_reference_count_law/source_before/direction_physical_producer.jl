using DiffMoM,LinearAlgebra,SHA,TOML
const repo=normpath(joinpath(@__DIR__,".."))
function main()
    output=mktempdir(joinpath(repo,"data/planar_audit");prefix="geovar_reference_direction_baseline_",cleanup=false)
    cp(@__FILE__,joinpath(output,"producer.jl"))
    cp(joinpath(repo,"src/planar/PlanarSonnetGeometryVariables.jl"),joinpath(output,"geometry_before.jl"))
    cases=Tuple{String,String,String}[]
    resolved=joinpath(repo,"data/planar_audit/geovar_reference_orientation_resolved_3drdml")
    for row in TOML.parsefile(joinpath(resolved,"comparison.toml"))["cases"]
        tag=row["case"];push!(cases,(tag,joinpath(resolved,tag*"_factor-1.son"),joinpath(resolved,tag*"_parameter/native_raw.s2p")))
    end
    paths=vcat([@__FILE__],[joinpath(d,f) for (d,_,fs) in walkdir(joinpath(repo,"src")) for f in fs],
        [p for (_,literal,native) in cases for p in (literal,native)])
    hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
    report=Dict{String,Any}("scope"=>"Confirmed native ANC direction bug: independent literal physical comparisons and current wrong-direction parameter full-S failures,128x128 modes; unchanged.005 full-S and1e-9 original voltage-residual gates; no production edits.","source_before"=>hashes(),"cases"=>Any[])
    try
        for (tag,literal,native) in cases
            p=read_sonnet_project(literal);reference=only(planar_read_touchstone(native).s)
            row=Dict{String,Any}("case"=>tag,"literal"=>relpath(literal,repo),"native"=>relpath(native,repo))
            try
                result=solve_sonnet_project(p,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
                raw=result.raw;rhs=zeros(ComplexF64,size(raw.currents))
                for b in eachindex(raw.problem.basis.port)
                    port=raw.problem.basis.port[b];iszero(port) && continue
                    rhs[b,port]=(port==1 ? -1. : 1.)*raw.problem.basis.width[b]
                end
                row["full_s_error"]=maximum(abs,result.s-reference)
                parameter=replace(literal,"_factor-1.son"=>"_parameter.son")
                previous=solve_sonnet_project(read_sonnet_project(parameter),1e9;raw=true,method=:dense_fft,mx=128,my=128)
                row["before_parameter_full_s_error"]=maximum(abs,previous.s-reference)
                row["original_voltage_residual"]=norm(raw.z_mom*raw.currents-rhs)/norm(rhs)
                row["reciprocity_error"]=maximum(abs,result.s-transpose(result.s));row["max_s_singular_value"]=opnorm(result.s)
                row["status"]=row["full_s_error"]<=.005 && row["original_voltage_residual"]<=1e-9 ? "PASS" : "FAIL"
            catch err
                row["status"]="REJECT";row["error"]=sprint(showerror,err)
            end
            push!(report["cases"],row);println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained direction bug baseline: ",output)
    end
    @assert report["source_unchanged"] && length(report["cases"])==16
end
main()
