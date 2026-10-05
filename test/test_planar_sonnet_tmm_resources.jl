using DiffMoM,Test

function _native_tmm_resource_project(base,count;wall=false)
    polygons=copy(base.polygons)
    if wall
        old=polygons[2]
        polygons[2]=SonnetPolygon(old.kind,old.level,0,old.id,old.vertices,
            old.target,old.technology,old.flags)
    else
        vertices=[.00025 .0004375 .0004375 .00025 .00025;
            .00075 .00075 .000875 .000875 .00075]
        base.box[2]=="6" && (vertices.*=4)
        push!(polygons,SonnetPolygon(:sheet,0,0,3,vertices,"","",String[]))
    end
    box=copy(base.box);box[1]=string(count-1)
    return SonnetProject(base.source,base.units,base.length_scale,base.frequency_scale,
        box,[copy(base.layers[1]) for _ in 1:count],
        [["Cu","1","TMM","58000000",".5",".02","2"]],base.top,base.bottom,
        polygons,base.ports,Dict{String,String}(),base.components,base.sweeps,SonnetRecord[])
end

function _native_tmm_resource_rejection(project,kind;max_bytes=kind===:conformal ? 1 : 2048)
    try
        if kind===:conformal
            sonnet_conformal_layout(project;freq=1e9,edge_size=.00025,
                interior_size=.0005,max_bytes)
        elseif kind===:floating
            sonnet_floating_model(project,1e9;max_bytes)
        else
            sonnet_component_model(project,1e9;max_bytes)
        end
        nothing
    catch err
        err isa ArgumentError || rethrow()
        err
    end
end

@testset "Native TMM budgets reject before physical source expansion" begin
    wall=read_sonnet_project(joinpath(@__DIR__,"fixtures","native_geovar_whole_polygon",
        "variants","nscd_ydir_positive_expand_parameter.son"))
    floating=_native_floating_pad_project(;cup=true)
    component=_synthetic_smd_project()
    @test DiffMoM._sonnet_thick_geometry(floating,1e9,Dict{String,Float64}();max_bytes=1)===floating
    small=_native_tmm_resource_project(wall,2;wall=true)
    conformal=sonnet_conformal_layout(small;freq=1e9,edge_size=.00025,interior_size=.0005)
    @test conformal.labels==[1,2]
    @test length(conformal.geometry_project.layers)==3
    @test any(poly->poly.kind===:via,conformal.geometry_project.polygons)
    float_control=sonnet_floating_model(_native_tmm_resource_project(floating,2),1e9)
    @test float_control.external_labels==[1]
    @test length(float_control.problem.stack.layers)==3
    component_control=sonnet_component_model(_native_tmm_resource_project(component,2),1e9)
    @test component_control.labels==[1,2,3,4]
    @test length(component_control.problem.stack.layers)==3
    source_err=_native_tmm_resource_rejection(small,:conformal;max_bytes=384)
    @test source_err isa ArgumentError
    @test occursin("native TMM source workspace",sprint(showerror,source_err))
    source_alloc=minimum(@allocated(_native_tmm_resource_rejection(small,:conformal;max_bytes=384)) for _ in 1:3)
    @test source_alloc<25000
    for count in (128,1024),kind in (:conformal,:floating,:component)
        base=kind===:conformal ? wall : kind===:floating ? floating : component
        project=_native_tmm_resource_project(base,count;wall=kind===:conformal)
        saved=deepcopy(project)
        err=_native_tmm_resource_rejection(project,kind)
        @test err isa ArgumentError
        label=kind===:component ? "all native component geometry preflight" : "native TMM stack preflight"
        @test occursin(label,sprint(showerror,err))
        bytes=minimum(@allocated(_native_tmm_resource_rejection(project,kind)) for _ in 1:3)
        @test bytes<25000
        @test project.layers==saved.layers && project.metals==saved.metals && project.box==saved.box
        @test all(a.vertices==b.vertices for (a,b) in zip(project.polygons,saved.polygons))
        @test all(a.values==b.values for (a,b) in zip(project.ports,saved.ports))
    end
end
