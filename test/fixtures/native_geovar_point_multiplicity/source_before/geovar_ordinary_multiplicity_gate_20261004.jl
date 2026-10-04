using DiffMoM,LinearAlgebra,TOML,SHA,Statistics
function main()
    repo=normpath(joinpath(@__DIR__,".."));label=isempty(ARGS) ? "before" : only(ARGS)
    paths=[joinpath(repo,"src/planar/PlanarSonnetGeometryVariables.jl"),@__FILE__]
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in paths)
    captures=[("anc","data/sonnet_validation/geovar_point_multiplicity_FjM71n","nscd_ordinary","multiplicity"),
        ("anc","data/sonnet_validation/geovar_point_multiplicity_FjM71n","scuni_ordinary","multiplicity"),
        ("anc","data/sonnet_validation/geovar_point_multiplicity_FjM71n","scxy_ordinary","multiplicity"),
        ("rad","data/sonnet_validation/geovar_point_multiplicity_FjM71n","rad_ordinary","multiplicity")]
    for row in TOML.parsefile(joinpath(repo,"data/sonnet_validation/geovar_symmetric_multiplicity_kHodLQ/comparison.toml"))["cases"]
        push!(captures,("sym","data/sonnet_validation/geovar_symmetric_multiplicity_kHodLQ",row["tag"],"literal"))
    end
    report=Dict{String,Any}("scope"=>"public repeated ordinary vertices; independent actual native literal gate .005 full S and 1e-9 original voltage residual, mx=my=128; pre/post effective geometry and warmed cumulative GC allocation", "source_before"=>hashes(),"cases"=>Any[])
    try
        for (kind,path,tag,literal) in captures
            directory=joinpath(repo,path);p=read_sonnet_project(joinpath(directory,tag*"_parameter.son"))
            q=read_sonnet_project(joinpath(directory,tag*"_"*literal*".son"))
            reference=only(planar_read_touchstone(joinpath(directory,tag*"_parameter/native_raw.s2p")).s)
            row=Dict{String,Any}("tag"=>tag,"kind"=>kind,"directory"=>path,"literal_suffix"=>literal);push!(report["cases"],row)
            for (name,project) in (("literal",q),("parameter",p))
                try
                    result=solve_sonnet_project(project,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
                    raw=result.raw;rhs=zeros(ComplexF64,size(raw.currents))
                    for b in eachindex(raw.problem.basis.port)
                        port=raw.problem.basis.port[b];iszero(port) && continue
                        rhs[b,port]=(port==1 ? -1. : 1.)*raw.problem.basis.width[b]
                    end
                    row[name*"_full_s_error"]=maximum(abs,result.s-reference)
                    row[name*"_voltage_residual"]=norm(raw.z_mom*raw.currents-rhs)/norm(rhs)
                    row[name*"_status"]="PASS"
                catch err
                    row[name*"_status"]="REJECT";row[name*"_error"]=sprint(showerror,err)
                end
            end
            resolved=DiffMoM._sonnet_geometry_project(p,1e9)
            row["geometry_error"]=maximum(abs,only(resolved.polygons).vertices-only(q.polygons).vertices)
            for _ in 1:100;DiffMoM._sonnet_geometry_project(p,1e9);end
            row["geometry_gc_bytes"]=@allocated DiffMoM._sonnet_geometry_project(p,1e9)
            println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(@__DIR__,"geovar_ordinary_multiplicity_"*label*"_20261004.toml"),"w") do io;TOML.print(io,report);end
    end
    @assert report["source_unchanged"]
end
main()
