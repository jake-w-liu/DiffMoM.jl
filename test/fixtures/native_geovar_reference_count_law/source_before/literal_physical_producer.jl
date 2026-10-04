using DiffMoM,LinearAlgebra,SHA,TOML
const repo=normpath(joinpath(@__DIR__,".."))
function main()
    output=mktempdir(joinpath(repo,"data/planar_audit");prefix="geovar_reference_literal_baseline_",cleanup=false)
    cp(@__FILE__,joinpath(output,"producer.jl"))
    cp(joinpath(repo,"src/planar/PlanarSonnetGeometryVariables.jl"),joinpath(output,"geometry_before.jl"))
    cases=Tuple{String,String,String}[]
    extended=joinpath(repo,"data/planar_audit/geovar_reference_count_law_extended_CttacU")
    for row in TOML.parsefile(joinpath(extended,"comparison.toml"))["cases"]
        row["status"]=="PASS" || continue
        tag=row["case"];push!(cases,(tag,joinpath(extended,tag*"_literal.son"),joinpath(extended,tag*"_parameter/native_raw.s2p")))
    end
    countlaw=joinpath(repo,"data/planar_audit/geovar_reference_count_law_probe_2xgDYY")
    for row in TOML.parsefile(joinpath(countlaw,"comparison.toml"))["cases"]
        tag=row["case"];factor=row["explicit_reference_count"]==3 ? 7 : 13
        push!(cases,(tag,joinpath(countlaw,tag*"_factor_$(factor).son"),joinpath(countlaw,tag*"_parameter/native_raw.s2p")))
    end
    orientation=joinpath(repo,"data/planar_audit/geovar_reference_orientation_DrCyGL")
    for row in TOML.parsefile(joinpath(orientation,"comparison.toml"))["cases"]
        startswith(row["case"],"zero_") || continue
        tag=row["case"];factor=1+row["r"]*(row["r"]-1)
        push!(cases,(tag,joinpath(orientation,tag*"_factor$(factor).son"),joinpath(orientation,tag*"_parameter/native_raw.s2p")))
    end
    factor3=joinpath(repo,"data/planar_audit/geovar_multiple_reference_factor3_probe_XqABap")
    for tag in ("nscd_expand_2","rad_expand_2","rad_contract_2")
        # The independent factor3 capture has its own input/output naming.
        push!(cases,(tag,joinpath(factor3,tag*"_literal.son"),joinpath(repo,"test/fixtures/native_geovar_reference_variants/native",tag*"_parameter/native_raw.s2p")))
    end
    paths=vcat([@__FILE__],[joinpath(d,f) for (d,_,fs) in walkdir(joinpath(repo,"src")) for f in fs],
        [p for (_,literal,native) in cases for p in (literal,native)])
    hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
    report=Dict{String,Any}("scope"=>"Independent literals for native count/orientation controls, pre-implementation public physical gate at128x128 modes; unchanged full-S.005 and original voltage-residual1e-9 gates, no tolerance relaxation or production edits.","source_before"=>hashes(),"cases"=>Any[])
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
        println("Retained independent literal baseline: ",output)
    end
    @assert report["source_unchanged"] && length(report["cases"])==44
end
main()
