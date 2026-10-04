using Test
using DiffMoM
using LinearAlgebra

# Independent stream writer used only for small known-coordinate fixtures.
function layout_test_u16(x)
    v=UInt16(x); UInt8[v>>8,v&0xff]
end
function layout_test_i32(x)
    v=reinterpret(UInt32,Int32(x)); UInt8[v>>24,(v>>16)&0xff,(v>>8)&0xff,v&0xff]
end
function layout_test_real(x)
    x==0 && return zeros(UInt8,8)
    power=floor(Int,log(abs(x))/log(16))+1
    mantissa=round(UInt64,abs(x)/16.0^power*2.0^56)
    UInt8[(x<0 ? 0x80 : 0x00)|(power+64),
        [(mantissa>>(8j))&0xff for j in 6:-1:0]...]
end
layout_test_record(kind,dtype,payload=UInt8[])=vcat(layout_test_u16(4+length(payload)),UInt8[kind,dtype],payload)
layout_test_string(x)=iseven(ncodeunits(x)) ? collect(codeunits(x)) : vcat(collect(codeunits(x)),0x00)
layout_test_xy(points)=reduce(vcat,layout_test_i32.(vec(points)))
function layout_test_gds(path;hierarchy=false)
    records=Vector{UInt8}[]
    put(k,d,p=UInt8[])=push!(records,layout_test_record(k,d,p))
    put(0,2,layout_test_u16(600)); put(1,2,zeros(UInt8,24)); put(2,6,layout_test_string("TEST"))
    put(3,5,vcat(layout_test_real(.001),layout_test_real(1e-6)))
    put(5,2,zeros(UInt8,24)); put(6,6,layout_test_string("RECT"))
    put(8,0); put(13,2,layout_test_u16(1)); put(14,2,layout_test_u16(0))
    vertices=hierarchy ? [0 4 4 0 0;0 0 2 2 0] : [0 1000 1000 0 0;375 375 625 625 375]
    put(16,3,layout_test_xy(vertices)); put(17,0); put(7,0)
    if hierarchy
        put(5,2,zeros(UInt8,24)); put(6,6,layout_test_string("TOP"))
        put(10,0); put(18,6,layout_test_string("RECT")); put(26,1,layout_test_u16(0x8000))
        put(27,5,layout_test_real(2.)); put(28,5,layout_test_real(90.)); put(16,3,layout_test_xy(reshape([10,20],2,1))); put(17,0)
        put(11,0); put(18,6,layout_test_string("RECT")); put(19,2,vcat(layout_test_u16(2),layout_test_u16(3)))
        put(16,3,layout_test_xy([0 8 0;0 0 12])); put(17,0); put(7,0)
    end
    put(4,0); write(path,reduce(vcat,records)); return path
end

@testset "GDSII/DXF physical layout import" begin
    mktempdir() do dir
        file=layout_test_gds(joinpath(dir,"line.gds"))
        doc=read_gdsii(file)
        @test doc isa PlanarLayoutDocument
        @test doc.database_unit_m≈1e-6 rtol=1e-14
        @test length(doc.polygons)==1
        @test doc.polygons[1].layer==(1,0)
        @test doc.polygons[1].vertices≈[0 .001 .001 0 0;.000375 .000375 .000625 .000625 .000375]
        @test_throws ArgumentError read_gdsii(file;max_bytes=10)
        @test_throws ArgumentError read_gdsii(file;topcell="absent")
        file2=layout_test_gds(joinpath(dir,"hierarchy.gds");hierarchy=true)
        hierarchy=read_gdsii(file2)
        @test length(hierarchy.polygons)==7
        @test minimum(hierarchy.polygons[1].vertices;dims=2)≈[10e-6;20e-6;;]
        @test maximum(hierarchy.polygons[1].vertices;dims=2)≈[14e-6;28e-6;;]
        @test minimum(hierarchy.polygons[7].vertices;dims=2)≈[4e-6;8e-6;;]
        @test maximum(hierarchy.polygons[7].vertices;dims=2)≈[8e-6;10e-6;;]
        @test_throws ArgumentError read_gdsii(file2;max_elements=3)
        damaged=joinpath(dir,"truncated.gds"); write(damaged,read(file)[1:end-1])
        @test_throws ArgumentError read_gdsii(damaged)

        dxf=joinpath(dir,"line.dxf")
        text="0\nSECTION\n2\nHEADER\n9\n\$INSUNITS\n70\n4\n0\nENDSEC\n0\nSECTION\n2\nENTITIES\n0\nLWPOLYLINE\n8\nCu\n90\n4\n70\n1\n10\n0\n20\n.375\n10\n1\n20\n.375\n10\n1\n20\n.625\n10\n0\n20\n.625\n0\nENDSEC\n0\nEOF\n"
        write(dxf,text); drawing=read_dxf(dxf)
        @test drawing.database_unit_m==.001
        @test drawing.polygons[1].layer=="Cu"
        @test drawing.polygons[1].vertices≈doc.polygons[1].vertices[:,1:4]
        write(dxf,replace(text,"70\n4\n"=>"70\n0\n";count=1))
        @test_throws ArgumentError read_dxf(dxf)
        @test read_dxf(dxf;unit_m=.001).polygons[1].vertices≈drawing.polygons[1].vertices
        write(dxf,replace(text,"LWPOLYLINE"=>"HATCH"))
        @test_throws ArgumentError read_dxf(dxf)
        write(dxf,replace(text,"90\n4"=>"90\n5"))
        @test_throws ArgumentError read_dxf(dxf)

        # Independent hand-built geometry and EM solve, not a parser roundtrip.
        stack=PlanarStackup([PlanarLayer(1.,1.,.0005),PlanarLayer(1.,1.,.0005)],TERM_GND,TERM_GND,.001,.001)
        grid=CellGrid(.001,.001,16,16)
        ports=[PlanarPort(1,:west,7:10,50.),PlanarPort(1,:east,7:10,50.)]
        imported=layout_planar_layout(doc,stack,grid,Dict((1,0)=>1),ports)
        direct=sheet_level(1,16,16); rasterize_rect!(direct,grid,0.,.001,.000375,.000625;connected=true)
        reference=build_planar_problem(stack,grid,[direct],ports)
        @test imported.problem.sheets[1].mask==direct.mask
        @test imported.problem.sheets[1].connect_west==direct.connect_west
        @test solve_planar(imported,1e9;mx=32,my=32).s≈solve_planar(reference,1e9;mx=32,my=32).s atol=1e-12
        copper=layout_planar_layout(doc,stack,grid,Dict((1,0)=>(interface=1,metal="film")),ports;metals=Dict("film"=>(f->2+0im)))
        @test solve_planar(copper,1e9;mx=32,my=32).s≈solve_planar(reference,1e9;mx=32,my=32,surface_zs=2.).s atol=1e-12
        @test_throws ArgumentError layout_planar_layout(doc,stack,grid,Dict(),ports)
        @test_throws ArgumentError layout_planar_layout(doc,stack,grid,Dict((1,0)=>3),ports)
        @test_throws ArgumentError layout_planar_layout(doc,stack,grid,Dict((1,0)=>1),ports;max_bytes=1)
        @test_throws ArgumentError layout_planar_layout(doc,stack,grid,Dict((1,0)=>1),ports;linear_transform=zeros(2,2))
    end
end
