using DiffMoM, Test, LinearAlgebra

function _native_pec_same_problem(a,b)
    @test a.grid==b.grid && a.stack.layers==b.stack.layers
    @test a.ports==b.ports && length(a.sheets)==length(b.sheets)
    for (left,right) in zip(a.sheets,b.sheets),field in fieldnames(SheetLevel)
        @test getfield(left,field)==getfield(right,field)
    end
    for field in fieldnames(PlanarBasisSet)
        @test getfield(a.basis,field)==getfield(b.basis,field)
    end
    @test length(a.vias)==length(b.vias)
    for (left,right) in zip(a.vias,b.vias),field in fieldnames(ViaLevel)
        @test getfield(left,field)==getfield(right,field)
    end
end

@testset "PEC raster lowering avoids unused material maps" begin
    p=read_sonnet_project(joinpath(@__DIR__,"fixtures","native_box_port_attachment",
        "cases","attachment__baseline","project.son"))
    before=DiffMoM._content_fingerprint(p,zero(UInt))
    for n in (64,128)
        plain()=sonnet_planar_problem(p;grid=(n,n))
        detailed()=sonnet_planar_problem(p;grid=(n,n),_details=true)
        problem=plain();model=detailed()
        _native_pec_same_problem(problem,model.problem)
        @test length(model.sheet_zs)==1
        @test size(only(model.sheet_zs))==(n,n)
        @test all(iszero,only(model.sheet_zs))
        plain_bytes=minimum(@allocated(plain()) for _ in 1:3)
        detailed_bytes=minimum(@allocated(detailed()) for _ in 1:3)
        # A detailed zero map alone requires 16 bytes per cell. Default PEC
        # lowering must avoid it while preserving the same geometry/basis.
        @test detailed_bytes-plain_bytes>=16n^2-4096
    end
    @test DiffMoM._content_fingerprint(p,zero(UInt))==before
    # PEC via cover pads need masks and current bases, without a material map.
    via=deepcopy(p)
    vertices=[.000375 .0005 .0005 .000375 .000375;
              .0004375 .0004375 .0005625 .0005625 .0004375]
    push!(via.polygons,SonnetPolygon(:via,0,-1,2,vertices,"GND","",["SOLID","COVERS"]))
    bare=sonnet_planar_problem(via;grid=(16,16))
    model=sonnet_planar_problem(via;grid=(16,16),_details=true)
    _native_pec_same_problem(bare,model.problem)
    @test all(iszero,only(model.sheet_zs))
    a=solve_planar(bare,1e9;mx=8,my=8)
    b=solve_planar(model.problem,1e9;mx=8,my=8,
        surface_zs=model.sheet_zs,via_sigma=model.via_sigma)
    @test a.s≈b.s rtol=1e-12
    @test a.currents≈b.currents rtol=1e-12
    # Material-enabled lowering still needs cell maps to reject mixed overlap.
    lossy=deepcopy(p)
    push!(lossy.metals,["first","0","SUP","1","0","0","0"])
    push!(lossy.metals,["second","1","SUP","2","0","0","0"])
    source=only(lossy.polygons)
    empty!(lossy.polygons)
    for material in (0,1)
        push!(lossy.polygons,SonnetPolygon(:sheet,source.level,material,material+1,
            copy(source.vertices),source.target,source.technology,copy(source.flags)))
    end
    err=try sonnet_planar_problem(lossy;_materials=true);nothing catch err;err end
    @test err isa ArgumentError && occursin("different sheet materials overlap",sprint(showerror,err))
    @test_throws ArgumentError sonnet_planar_problem(lossy)
end
