using DiffMoM,Test,SHA,TOML,LinearAlgebra
const repo=normpath(joinpath(@__DIR__,".."))
function main()
    fixture=joinpath(repo,"data/planar_audit/radial_reference_three_pass_Cjn6o9")
    paths=vcat([@__FILE__],[joinpath(d,f) for folder in (joinpath(repo,"src"),fixture) for (d,_,fs) in walkdir(folder) for f in fs])
    hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
    report=Dict{String,Any}("status"=>"RUNNING","source_before"=>hashes(),"cases"=>Any[])
    try
        @testset "Native RAD three-pass anchor crossings" begin
            for row in TOML.parsefile(joinpath(fixture,"comparison.toml"))["cases"]
                tag=row["case"];p=read_sonnet_project(joinpath(fixture,tag*"_parameter.son"));saved=deepcopy(p.polygons)
                if haskey(row,"native_error")
                    @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9)
                    push!(report["cases"],Dict("case"=>tag,"status"=>"REJECT: coincident radial point; native full-S unverified"));continue
                end
                q=read_sonnet_project(joinpath(fixture,tag*"_literal.son"));effective=DiffMoM._sonnet_geometry_project(p,1e9)
                geometry_error=maximum(maximum(abs,a.vertices-b.vertices) for (a,b) in zip(effective.polygons,q.polygons))
                @test geometry_error<=2e-18
                a=sonnet_planar_problem(p;freq=1e9);b=sonnet_planar_problem(q;freq=1e9)
                @test all(x.mask==y.mask for (x,y) in zip(a.sheets,b.sheets))
                native=only(planar_read_touchstone(joinpath(fixture,tag*"_parameter/native_raw.s2p")).s)
                result=solve_sonnet_project(p,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
                raw=result.raw;rhs=zeros(ComplexF64,size(raw.currents))
                for b in eachindex(raw.problem.basis.port)
                    port=raw.problem.basis.port[b];iszero(port) && continue
                    rhs[b,port]=(port==1 ? -1. : 1.)*raw.problem.basis.width[b]
                end
                full_s_error=maximum(abs,result.s-native);residual=norm(raw.z_mom*raw.currents-rhs)/norm(rhs)
                @test full_s_error<=.005 && residual<=1e-9
                @test maximum(abs,result.s-transpose(result.s))<=1e-10 && opnorm(result.s)<=1+1e-9
                @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved))
                push!(report["cases"],Dict("case"=>tag,"status"=>"PASS","geometry_error"=>geometry_error,"full_s_error"=>full_s_error,"original_voltage_residual"=>residual));println(last(report["cases"]));flush(stdout)
            end
        end
        report["status"]="PASS"
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(repo,"data/radial_three_pass_current_validation_$(VERSION)_20261005.toml"),"w") do io;TOML.print(io,report);end
    end
    @assert report["status"]=="PASS" && report["source_unchanged"]
end
main()
