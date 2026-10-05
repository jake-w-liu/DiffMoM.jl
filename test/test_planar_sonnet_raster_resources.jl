using DiffMoM, Test, LinearAlgebra

function _native_raster_budget_error(f)
    try
        f()
    catch err
        err isa ArgumentError || rethrow()
        return sprint(showerror,err)
    end
    error("expected native raster resource rejection")
end

function _native_raster_numerical_budget(model)
    budget=1
    for _ in 1:4
        try
            raw=solve_planar(model.problem,1e9;max_bytes=budget,mx=8,my=8,
                surface_zs=model.sheet_zs,via_sigma=model.via_sigma)
            return raw,budget
        catch err
            err isa ArgumentError || rethrow()
            required=match(r"requires (\d+) raw bytes",sprint(showerror,err))
            required===nothing && rethrow()
            next=parse(Int,required.captures[1])
            next>budget || rethrow()
            budget=next
        end
    end
    error("numerical resource preflight did not converge")
end

@testset "Native raster budgets precede grid-sized storage" begin
    p=read_sonnet_project(joinpath(@__DIR__,"fixtures","native_box_port_attachment",
        "cases","attachment__baseline","project.son"))
    saved=deepcopy(p)
    before=DiffMoM._content_fingerprint(p,zero(UInt))
    for lower in (false,true)
        bytes=Int[]
        for n in (64,256,512)
            f=lower ? ()->sonnet_planar_problem(p;grid=(n,n),max_bytes=1) :
                ()->solve_sonnet_project(p,1e9;raw=true,grid=(n,n),max_bytes=1,mx=8,my=8)
            message=_native_raster_budget_error(f)
            @test occursin("native raster source workspace",message)
            @test occursin("max_bytes=1",message)
            allocated=minimum(@allocated(_native_raster_budget_error(f)) for _ in 1:3)
            @test allocated<32768
            push!(bytes,allocated)
        end
        @test maximum(bytes)-minimum(bytes)<4096
    end
    # The source fits this budget; a 256-square material map alone does not.
    # Reject before creating that map, rather than after materializing it.
    f=()->sonnet_planar_problem(p;grid=(256,256),max_bytes=65536)
    @test occursin("native raster geometry workspace",_native_raster_budget_error(f))
    @test (@allocated _native_raster_budget_error(f))<100000
    for limit in (0,-1,BigInt(typemax(Int))+1,typemax(UInt128))
        @test_throws ArgumentError sonnet_planar_problem(p;max_bytes=limit)
        @test_throws ArgumentError solve_sonnet_project(p,1e9;raw=true,max_bytes=limit)
    end
    for grid in ((0,32),(true,32),(1.5,32),(32,),(:invalid,32),
            (BigInt(typemax(Int))+1,32))
        @test_throws ArgumentError sonnet_planar_problem(p;grid)
    end
    @test_throws ArgumentError sonnet_planar_problem(p;grid=(typemax(Int),typemax(Int)))
    @test DiffMoM._content_fingerprint(p,zero(UInt))==before
    @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved.polygons))
    @test all(a.values==b.values for (a,b) in zip(p.ports,saved.ports))
end

@testset "Native raster solve reserves geometry and external response" begin
    p=read_sonnet_project(joinpath(@__DIR__,"fixtures","native_box_port_attachment",
        "cases","attachment__baseline","project.son"))
    model=sonnet_planar_problem(p;freq=1e9,_materials=true,_details=true)
    # Independent lower bound from the storage visible through the public model.
    basis=model.problem.basis
    basis_bytes=sum(sizeof,getfield.(Ref(basis),fieldnames(typeof(basis))))
    material_bytes=sum(sizeof,model.sheet_zs)
    @test model.payload>=basis_bytes+material_bytes+sizeof(model.contraction)+sizeof(model.z0)
    raw,required=_native_raster_numerical_budget(model)
    @test all(isfinite,raw.s)
    # That exact numerical-only budget must not also fund the wrapper's model.
    @test_throws ArgumentError solve_sonnet_project(p,1e9;raw=true,max_bytes=required,mx=8,my=8)
    generous=solve_sonnet_project(p,1e9;raw=true,max_bytes=2required,mx=8,my=8)
    @test generous.s≈raw.s rtol=1e-12
    @test generous.raw.currents≈raw.currents rtol=1e-12
    @test_throws ArgumentError sonnet_planar_problem(p;freq=1e9,_materials=true,
        _details=true,max_bytes=model.payload-1)
    # Wall normalization copies only the affected vertex array. The original
    # records, flags and other arrays remain borrowed and are counted once.
    projected=deepcopy(p)
    vertices=only(projected.polygons).vertices
    vertices[1,vertices[1,:].==0.].=-model.problem.grid.dx/8
    saved=copy(vertices)
    owned=sonnet_planar_problem(projected;freq=1e9,_materials=true,_details=true)
    @test owned.geometry_project!==projected
    @test vertices==saved
    @test owned.problem.basis.kind==basis.kind
    @test owned.problem.basis.x0==basis.x0
    @test sizeof(vertices)<=owned.payload-model.payload<4096
    reused=solve_sonnet_project(projected,1e9;raw=true,max_bytes=2required,mx=8,my=8)
    @test reused.s≈generous.s rtol=1e-12
end
