using Test,DiffMoM,LinearAlgebra

@testset "Project frequency storage is checked before workflow evaluation" begin
    path=joinpath(@__DIR__,"..","examples","planar_project_line.toml")
    project=load_planar_project(path)
    project.data["metals"]["lossless"]=Dict("type"=>"lossless")
    invalid=(-1.,0.,NaN,Inf,BigFloat("1e1000"),BigFloat("1e-1000"))
    for frequency in invalid
        @test_throws ArgumentError planar_project_layout(project;freq=frequency)
        @test_throws ArgumentError planar_project_metal_zs(project,"conductor",frequency)
        @test_throws ArgumentError planar_project_metal_zs(project,"lossless",frequency)
        @test_throws ArgumentError solve_planar_project(project,frequency)
        @test_throws ArgumentError solve_planar_project(path,frequency)
        error=try solve_planar_project("missing_frequency_sentinel.toml",frequency);nothing catch e;e end
        @test error isa ArgumentError && occursin("frequency",sprint(showerror,error))
    end
    # These checks precede even schema traversal on a caller-edited project.
    malformed=deepcopy(project);malformed.data["unsupported_field"]=1
    error=try planar_project_layout(malformed;freq=BigFloat("1e-1000"));nothing catch e;e end
    @test error isa ArgumentError && occursin("frequency",sprint(showerror,error))
    for frequency in (BigFloat("1e9"),BigFloat("1.125e9"))
        expected=planar_project_layout(project;freq=Float64(frequency))
        actual=planar_project_layout(project;freq=frequency)
        @test actual.layout.problem.stack.layers[1].epsr==expected.layout.problem.stack.layers[1].epsr
        index=findfirst(==("conductor"),actual.layout.material_names)
        @test actual.layout.materials[index](Float64(frequency))==expected.layout.materials[index](Float64(frequency))
        @test planar_project_metal_zs(project,"conductor",frequency)==planar_project_metal_zs(project,"conductor",Float64(frequency))
        @test planar_project_metal_zs(project,"lossless",frequency)==0im
        big=solve_planar_project(project,frequency;mx=16,my=12)
        small=solve_planar_project(project,Float64(frequency);mx=16,my=12)
        @test big.freq isa Float64 && big.freq==Float64(frequency)
        @test big.s==small.s
    end
    loaded=deepcopy(project)
    for port in loaded.data["ports"];port["external"]=true;end
    loaded.data["components"]=[Dict("name"=>"external","type"=>"vendor","ports"=>[1,2])]
    calls=Ref(0);frequencies=Any[]
    response=(project,record,f)->begin
        calls[]+=1;push!(frequencies,f)
        (response=Matrix{ComplexF64}(I,2,2)*.1,format=:s,z0=50.)
    end
    for frequency in invalid
        @test_throws ArgumentError solve_planar_project(loaded,frequency;component_response=response)
        @test calls[]==0
    end
    result=solve_planar_project(loaded,BigFloat("1e9");component_response=response,mx=16,my=12)
    @test result.freq==1e9 && calls[]==1 && only(frequencies)===1e9
end
