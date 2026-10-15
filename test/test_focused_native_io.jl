module FocusedNativeIOTests
using DiffMoM,Test,LinearAlgebra

@testset "Native parser, physical ports and small calibrated solve" begin
    mktempdir() do directory
        path=joinpath(directory,"line.son")
        write(path,"""FTYP SONPROJ 19
VER "18.53"
DIM
FREQ GHZ
LNG MM
END DIM
CONTROL
OPTIONS
END CONTROL
GEO
TMET "PEC" 0 SUP 0 0 0 0
BMET "PEC" 0 SUP 0 0 0 0
BOX 1 2 2 8 8 100 0
0.5 1 1 0 0 0 0 "Air_top"
0.5 1 1 0 0 0 0 "Air_bottom"
POR1 BOX
POLY 9 1
3
1 50 0 0 0 0 1
POR1 BOX
POLY 9 1
1
2 75 0 0 0 2 1
NUM 1
0 5 -1 N 9 1 1 100 100 0 0 0 Y
0 0.5
2 0.5
2 1.5
0 1.5
0 0.5
END
END GEO
""")
        project=read_sonnet_project(path);saved=deepcopy(project)
        model=sonnet_planar_problem(project;freq=10e9,_materials=true,_details=true)
        @test project.length_scale==.001 && project.frequency_scale==1e9
        @test (model.problem.grid.nx,model.problem.grid.ny)==(4,4)
        @test model.labels==[1,2] && model.z0==[50.,75.]
        @test [p.wall for p in model.problem.ports]==[:west,:east]
        @test planar_basis_count(model.problem.basis)==14
        @test all(iszero,only(model.sheet_zs))

        # Build the same physical conductor without the native importer.
        grid=CellGrid(.002,.002,4,4);sheet=sheet_level(1,4,4)
        sheet.mask[:,2:3].=true
        sheet.connect_west[2:3].=true;sheet.connect_east[2:3].=true
        stack=PlanarStackup([PlanarLayer(1,1,.0005),PlanarLayer(1,1,.0005)],
            TERM_GND,TERM_GND,grid.a,grid.b)
        hand=build_planar_problem(stack,grid,[sheet],
            [PlanarPort(1,:west,2:3,50.),PlanarPort(1,:east,2:3,75.)])
        reference=solve_planar(hand,10e9;mx=8,my=8,max_bytes=4*1024^2)
        raw=solve_sonnet_project(project,10e9;raw=true,mx=8,my=8,max_bytes=4*1024^2)
        @test only(raw.raw.problem.sheets).mask==sheet.mask
        @test raw.raw.z_mom≈reference.z_mom rtol=1e-13
        @test raw.s≈reference.s rtol=1e-12
        rhs=zeros(ComplexF64,14,2)
        for b in eachindex(hand.basis.port)
            port=hand.basis.port[b]
            port==0 || (rhs[b,port]=(port==1 ? -1 : 1)*hand.basis.width[b])
        end
        @test norm(raw.raw.z_mom*raw.raw.currents-rhs)/norm(rhs)<1e-10
        @test raw.s≈transpose(raw.s) rtol=1e-12
        @test raw.s'*raw.s≈Matrix{ComplexF64}(I,2,2) atol=1e-12

        # A shunt launch adds its admittance directly, giving an expected
        # result independent of the general ABCD embedding implementation.
        shunts=ComplexF64[.001im,.002im]
        launches=[ComplexF64[1 0;y 1] for y in shunts]
        calibrated=solve_sonnet_project(project,10e9;calibration=launches,
            mx=8,my=8,max_bytes=4*1024^2)
        @test calibrated.y≈raw.y-Diagonal(shunts) rtol=1e-12
        @test calibrated.voltage_transfer≈Matrix{ComplexF64}(I,2,2) atol=1e-12
        @test [q.vertices for q in project.polygons]==[q.vertices for q in saved.polygons]
        @test getproperty.(project.ports,:values)==getproperty.(saved.ports,:values) &&
            project.layers==saved.layers && project.box==saved.box
        @test_throws ArgumentError solve_sonnet_project(project,10e9;raw=true,surface_zs=1.)
        @test_throws ArgumentError sonnet_planar_problem(project;max_bytes=1)
    end
end

@testset "Captured native material order and resistor reference" begin
    fixtures=joinpath(@__DIR__,"..","validation","sonnet_stripline")
    project=read_sonnet_project(joinpath(fixtures,"native_material_reference","planar_order_probe.son"))
    frequency=1e9;skin=(1+im)*2sqrt(frequency)
    @test project.metals[1]==["Planar","1","SUP","1","2","3","4"]
    @test sonnet_metal_zs(project,project.metals[1],frequency)≈
        skin/tanh(skin)+im*(3+2pi*frequency*4e-12)
    expected=ComplexF64[1/3 2/3;2/3 1/3] # 50 Ω series element between 50 Ω ports.
    directory=joinpath(fixtures,"native_circuit_units_reference")
    for unit in ("OH","KOH")
        native=planar_read_touchstone(joinpath(directory,unit*".s2p"))
        project=read_sonnet_project(joinpath(directory,unit*".son"))
        @test maximum(abs,native.s[2]-expected)<=4e-14
        @test solve_sonnet_project(project,native.frequencies[2]).s≈expected atol=4e-14
    end
end
end
