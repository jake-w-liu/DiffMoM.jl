using DiffMoM,SHA,TOML,LinearAlgebra
function main()
    repo=normpath(joinpath(@__DIR__,".."));label=isempty(ARGS) ? "before" : only(ARGS)
    capture=joinpath(repo,"data/sonnet_validation/geovar_whole_polygon_PIH1cm")
    paths=[@__FILE__,joinpath(repo,"src/planar/PlanarSonnetGeometryVariables.jl")]
    append!(paths,[joinpath(capture,mode*"_"*suffix*".son") for mode in ("nscd","rad"),suffix in ("parameter","double_reference")])
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in paths)
    report=Dict{String,Any}("scope"=>"whole-polygon count0 native selector public path; independently verified native sequential literals, .005 full-S and 1e-9 original voltage-residual gates at mx=my128; warmed returned-geometry GC", "source_before"=>hashes(),"cases"=>Any[])
    try
        for mode in ("nscd","rad")
            p=read_sonnet_project(joinpath(capture,mode*"_parameter.son"))
            q=read_sonnet_project(joinpath(capture,mode*"_double_reference.son"))
            native=only(planar_read_touchstone(joinpath(capture,mode*"_parameter/native_raw.s2p")).s)
            row=Dict{String,Any}("mode"=>mode);push!(report["cases"],row)
            for (name,project) in (("literal",q),("parameter",p))
                try
                    result=solve_sonnet_project(project,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
                    raw=result.raw;rhs=zeros(ComplexF64,size(raw.currents))
                    for b in eachindex(raw.problem.basis.port)
                        port=raw.problem.basis.port[b];iszero(port) && continue
                        rhs[b,port]=(port==1 ? -1. : 1.)*raw.problem.basis.width[b]
                    end
                    row[name*"_full_s_error"]=maximum(abs,result.s-native)
                    row[name*"_voltage_residual"]=norm(raw.z_mom*raw.currents-rhs)/norm(rhs)
                    row[name*"_status"]="PASS"
                    @assert row[name*"_full_s_error"]<=.005 && row[name*"_voltage_residual"]<=1e-9
                catch err
                    row[name*"_status"]="REJECT";row[name*"_error"]=sprint(showerror,err)
                    name=="literal" && rethrow()
                end
            end
            if row["parameter_status"]=="PASS"
                effective=DiffMoM._sonnet_geometry_project(p,1e9)
                row["geometry_error"]=maximum(maximum(abs,a.vertices-b.vertices) for (a,b) in zip(effective.polygons,q.polygons))
                for _ in 1:100;DiffMoM._sonnet_geometry_project(p,1e9);end
                row["geometry_gc_bytes"]=@allocated DiffMoM._sonnet_geometry_project(p,1e9)
            end
            println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(@__DIR__,"geovar_whole_polygon_"*label*"_20261004.toml"),"w") do io;TOML.print(io,report);end
    end
    @assert report["source_unchanged"]
end
main()
