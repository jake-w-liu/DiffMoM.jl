module SonnetCUPSheetOwnershipTests
using DiffMoM,Test

function cup_project()
    record(t)=SonnetRecord(1,String.(t))
    # The upper pad overlaps the lower launch in xy. Its own east launch
    # and characterization contact have clear paths to the bottom cover.
    lower=SonnetPolygon(:sheet,1,-1,1,
        [1 4 4 1 1;1 1 6 6 1].*.001,"","","",String[])
    upper=SonnetPolygon(:sheet,0,-1,2,
        [1 10 10 1 1;1 1 6 6 1].*.001,"","","",String[])
    ports=SonnetPortSpec[]
    for (polygon,edge,number) in ((1,3,1),(2,1,2))
        records=[record(["POR1","CUP","g"]),record(["GNDREF","BOX","BELOW"]),
            record(["TERMW","FEED"]),record(["CUPGRP","g"])]
        push!(ports,SonnetPortSpec(:cup,polygon,edge,number,
            [string(number),"50","0","0","0","0",".5"],records))
    end
    SonnetProject("cup_levels.son",Dict("LNG"=>"MM"),.001,1e9,
        ["2","12","8","24","16"],fill(["1","1","1","0","0","0"],3),
        Vector{String}[],["PEC","0","RES","0"],["PEC","0","RES","0"],
        [lower,upper],ports,Dict{String,String}(),Vector{SonnetRecord}[],SonnetRecord[],SonnetRecord[])
end

@testset "CUP-only sources and sheet-scoped junction marginals" begin
    project=cup_project()
    model=sonnet_planar_problem(project;freq=1e9,_materials=true,_details=true,max_bytes=16*1024^2)
    standard=only(model.cup_standards).standard
    @test model.labels==[1,2]
    @test !isempty(model.problem.ports)
    @test standard.kind===:marginal
    full,minus_lower,minus_upper=standard.problems
    @test minus_lower.sheets[2].mask==full.sheets[2].mask
    @test minus_upper.sheets[1].mask==full.sheets[1].mask
    @test findall(full.sheets[1].mask .!= minus_lower.sheets[1].mask)==[CartesianIndex(2,j) for j in 2:6]
    @test findall(full.sheets[2].mask .!= minus_upper.sheets[2].mask)==[CartesianIndex(10,j) for j in 2:6]
    @test project.polygons[2].vertices==[1 10 10 1 1;1 1 6 6 1].*.001
end
end
