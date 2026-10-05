using DiffMoM,Test

function _conformal_transform_budget_rejection(project)
    try
        sonnet_conformal_layout(project;freq=1e9,edge_size=.00025,
            interior_size=.0005,max_bytes=1)
        nothing
    catch err
        err isa ArgumentError || rethrow()
        err
    end
end

@testset "Conformal resource rejection precedes GEOVAR copies" begin
    p=read_sonnet_project(joinpath(@__DIR__,"fixtures","native_geovar_whole_polygon",
        "variants","nscd_ydir_positive_expand_parameter.son"))
    saved=deepcopy(p)
    for subdivisions in (1,128,1024)
        old=p.polygons[2];v=old.vertices;n=size(v,2)-1
        points=[(1-t).*v[:,k].+t.*v[:,k+1] for k in 1:n
            for t in (i/subdivisions for i in 0:subdivisions-1)]
        push!(points,first(points));vertices=hcat(points...)
        poly=SonnetPolygon(old.kind,old.level,old.material,old.id,vertices,
            old.target,old.technology,old.flags)
        polygons=copy(p.polygons);polygons[2]=poly
        project=SonnetProject(p.source,p.units,p.length_scale,p.frequency_scale,
            p.box,p.layers,p.metals,p.top,p.bottom,polygons,p.ports,p.variables,
            p.components,p.sweeps,p.records)
        original=copy(vertices)
        transformed=DiffMoM._sonnet_geometry_project(project,1e9)
        @test transformed!==project
        @test size(transformed.polygons[2].vertices)==size(vertices)
        err=_conformal_transform_budget_rejection(project)
        @test err isa ArgumentError
        @test occursin("native geometry variable workspace",sprint(showerror,err))
        bytes=minimum(@allocated(_conformal_transform_budget_rejection(project)) for _ in 1:3)
        @test bytes<25000
        @test vertices==original
    end
    @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved.polygons))
    @test all(a.values==b.values for (a,b) in zip(p.ports,saved.ports))
end
