using DiffMoM,SHA,TOML,LinearAlgebra
const repo=normpath(joinpath(@__DIR__,".."))
include(joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
function main()
    fixture=joinpath(repo,"test/fixtures/native_geovar_whole_polygon/variants")
    tags=[row["tag"] for row in TOML.parsefile(joinpath(fixture,"comparison.toml"))["cases"] if startswith(row["tag"],"rad_")]
    directory=mktempdir(joinpath(repo,"data/sonnet_validation");prefix="geovar_rad_reference_headers_",cleanup=false)
    paths=[@__FILE__,joinpath(repo,"src/Types.jl"),joinpath(repo,"src/planar/PlanarSonnetGeometryVariables.jl"),joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")]
    append!(paths,[joinpath(fixture,tag*"_"*suffix*".son") for tag in tags for suffix in ("parameter","literal")])
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in paths)
    report=Dict{String,Any}("scope"=>"native RAD whole-polygon explicit moving reference under NSCD/SCUNI/SCXY headers;8 axis/direction/contract-expand cases, native literal exact full-S identity, independent public literal .005/full voltage1e-9 gates before extension", "source_before"=>hashes(),"cases"=>Any[])
    cp(@__FILE__,joinpath(directory,"producer.jl"))
    cp(joinpath(repo,"src/planar/PlanarSonnetGeometryVariables.jl"),joinpath(directory,"geometry_source_before.jl"))
    try
        for tag in tags
            literal_source=joinpath(fixture,tag*"_literal.son")
            literal=reference_run(find_em(),literal_source;output_dir=joinpath(directory,tag*"_literal"),deembedded=false)
            native_literal=only(checked_native_touchstone(literal,joinpath(literal.output_dir,"native_raw.s2p");deembedded=false).s)
            public_literal=solve_sonnet_project(read_sonnet_project(literal_source),1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
            raw=public_literal.raw;rhs=zeros(ComplexF64,size(raw.currents))
            for b in eachindex(raw.problem.basis.port)
                port=raw.problem.basis.port[b];iszero(port) && continue
                rhs[b,port]=(port==1 ? -1. : 1.)*raw.problem.basis.width[b]
            end
            residual=norm(raw.z_mom*raw.currents-rhs)/norm(rhs)
            literal_error=maximum(abs,public_literal.s-native_literal)
            @assert residual<=1e-9 && literal_error<=.005
            for mode in ("NSCD","SCUNI","SCXY")
                name=tag*"_"*lowercase(mode)
                source=joinpath(directory,name*".son")
                original=read(joinpath(fixture,tag*"_parameter.son"),String)
                write(source,replace(original," NSCD\n"=>" $mode\n"))
                native=reference_run(find_em(),source;output_dir=joinpath(directory,name),deembedded=false)
                matrix=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
                public_status="ACCEPT";public_error=""
                try
                    DiffMoM._sonnet_geometry_project(read_sonnet_project(source),1e9)
                catch err
                    err isa ArgumentError || rethrow()
                    public_status="REJECT";public_error=sprint(showerror,err)
                end
                row=Dict("tag"=>tag,"name"=>name,"mode"=>mode,"native_full_s_error"=>maximum(abs,matrix-native_literal),"native_bit_identical"=>matrix==native_literal,"literal_public_full_s_error"=>literal_error,"literal_original_voltage_residual"=>residual,"before_public_status"=>public_status,"before_public_error"=>public_error)
                push!(report["cases"],row);println(row);flush(stdout)
            end
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained RAD reference header evidence: ",directory)
    end
    @assert report["source_unchanged"] && length(report["cases"])==24
    @assert all(row->row["native_bit_identical"] && row["before_public_status"]==(row["mode"]=="NSCD" ? "ACCEPT" : "REJECT"),report["cases"])
end
main()
