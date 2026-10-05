using DiffMoM,Test

function _native_stack_budget_project(base,count)
    box=copy(base.box);box[1]=string(count-1)
    return SonnetProject(base.source,base.units,base.length_scale,base.frequency_scale,
        box,[copy(base.layers[1]) for _ in 1:count],base.metals,base.top,base.bottom,
        base.polygons,base.ports,Dict{String,String}(),base.components,base.sweeps,SonnetRecord[])
end

function _native_stack_budget_rejection(project,kind)
    try
        if kind===:floating
            sonnet_floating_model(project,1e9;max_bytes=2048)
        else
            sonnet_conformal_layout(project;freq=1e9,edge_size=.00025,
                interior_size=.0005,max_bytes=1)
        end
        nothing
    catch err
        err isa ArgumentError || rethrow()
        err
    end
end

@testset "Native stack budget rejects before layer construction" begin
    floating=_native_floating_pad_project(;cup=true)
    wall=read_sonnet_project(joinpath(@__DIR__,"fixtures","native_geovar_whole_polygon",
        "variants","nscd_ydir_positive_expand_parameter.son"))
    small=_native_stack_budget_project(wall,2)
    control=sonnet_conformal_layout(small;freq=1e9,edge_size=.00025,interior_size=.0005)
    @test control.labels==[1,2]
    @test sonnet_floating_model(floating,1e9).external_labels==[1]
    native=DiffMoM._sonnet_stack_geometry(small,1e9,nothing,Dict{String,Float64}();max_bytes=384)
    @test length(native.stack.layers)==2
    @test all(layer->layer.epsr==1 && layer.mur==1 && real(layer.thickness)>0,native.stack.layers)
    @test_throws ArgumentError DiffMoM._sonnet_stack_geometry(small,1e9,nothing,
        Dict{String,Float64}();max_bytes=383)
    for count in (128,1024),kind in (:floating,:conformal)
        project=_native_stack_budget_project(kind===:floating ? floating : wall,count)
        saved=deepcopy(project)
        err=_native_stack_budget_rejection(project,kind)
        @test err isa ArgumentError
        @test occursin("native stack layer workspace",sprint(showerror,err))
        allocated=minimum(@allocated(_native_stack_budget_rejection(project,kind)) for _ in 1:3)
        @test allocated<25000
        @test project.layers==saved.layers && project.box==saved.box
        @test all(a.vertices==b.vertices for (a,b) in zip(project.polygons,saved.polygons))
        @test all(a.values==b.values for (a,b) in zip(project.ports,saved.ports))
    end
end
