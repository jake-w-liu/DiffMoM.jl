using DiffMoM,Test,LinearAlgebra
if !isdefined(DiffMoM,:read_gerber)
    Base.include(DiffMoM,joinpath(@__DIR__,"..","src","planar","PlanarArtworkIO.jl"))
    Base.include(DiffMoM,joinpath(@__DIR__,"..","src","planar","PlanarGerberIO.jl"))
end
function _test_gerber_text(body;header="%FSLAX36Y36*%\n%MOMM*%\n",kw...)
    mktempdir() do directory
        path=joinpath(directory,"metal.gbr");write(path,header*body)
        return read_gerber(path;kw...)
    end
end
_gerber_member(doc,x,y)=begin
    occupied=false
    for object in doc.objects
        DiffMoM._artwork_contains(object.shape,x*1e-3,y*1e-3) && (occupied=object.dark)
    end
    occupied
end

@testset "Gerber analytic polarity, standard apertures and metadata" begin
    body=raw"""
%TF.FileFunction,Copper,L1,Top*%
%TA.AperFunction,Conductor*%
%ADD10C,2X1*%
%ADD11R,0.2X0.2*%
%ADD12O,3X1*%
%ADD13P,2X4X45*%
%TO.N,signal*%
D11*X0Y0D03*
D10*X0Y0D03*
%LPC*%D11*X800000Y0D03*
%LPD*%D12*X5000000Y0D03*
D13*X10000000Y0D03*
M02*
"""
    doc=_test_gerber_text(body)
    @test length(doc.objects)==5
    @test _gerber_member(doc,0,0) # aperture hole is transparent to old copper
    @test !_gerber_member(doc,.3,0)
    @test _gerber_member(doc,.6,0)
    @test !_gerber_member(doc,.8,0) # LPC clears the preceding circular object
    @test _gerber_member(doc,6.3,0)
    @test !_gerber_member(doc,6.3,.45)
    @test _gerber_member(doc,10.6,.6)
    @test !_gerber_member(doc,10.8,0)
    @test doc.attributes[".FileFunction"]==["Copper","L1","Top"]
    @test doc.objects[1].attributes[".AperFunction"]==["Conductor"]
    @test doc.objects[1].attributes[".N"]==["signal"]
    doc=_test_gerber_text("%ADD10C,1*%\nD10*X01Y02D03*M02*";header="%FSTAX24Y24*%%MOMM*%")
    @test collect(doc.objects[1].bounds)≈[.0005,.0015,.0015,.0025]
    doc=_test_gerber_text("%ADD10C,1*%D10*X1000000Y0D03*X1000000D03*M02*";header="%FSLIX36Y36*%%MOMM*%")
    @test _gerber_member(doc,1,0) && _gerber_member(doc,2,0)
end

@testset "Gerber macro arithmetic, butt ends and transformed holes" begin
    body=raw"""
%AMHole*$3=$1/2+0*21,1,$1,$1,0,0,0*1,0,$3,0,0*%
%AMVector*20,1,1,0,0,2,0,0*%
%AMThermal*7,0,0,4,2,0.4,0*%
%AMTriangle*4,1,3,0,0,2,0,0,2,0,0,0*%
%ADD10Hole,4*%%ADD11Vector*%%ADD12Thermal*%%ADD13Triangle*%
D10*X0Y0D03*
D11*X10000000Y0D03*
D12*X20000000Y0D03*
%LMX*%%LR90*%D13*X30000000Y0D03*
M02*
"""
    doc=_test_gerber_text(body)
    @test !_gerber_member(doc,0,0)
    @test _gerber_member(doc,1.5,0)
    @test !_gerber_member(doc,9.9,0) # macro vector lines have butt ends
    @test _gerber_member(doc,10.1,.4)
    @test !_gerber_member(doc,20,1.5)
    @test _gerber_member(doc,21.1,1.1)
    @test !_gerber_member(doc,20.5,.5)
    @test _gerber_member(doc,29.5,-.5)
    @test !_gerber_member(doc,30.5,.5)
    @test DiffMoM._gerber_expression(raw"$99+1+2+3",Dict{Int,Float64}())==6
    @test DiffMoM._gerber_expression("2x3x4",Dict{Int,Float64}())==24
    @test_throws ArgumentError DiffMoM._gerber_expression("run(`calc`)",Dict{Int,Float64}())
    @test_throws ArgumentError DiffMoM._gerber_expression("1/0",Dict{Int,Float64}())
end

@testset "Gerber circular drawings, curved regions and cut-in hole" begin
    body="%ADD10C,0.2*%D10*G75*X2000000Y0D02*G03*X0Y2000000I-2000000J0D01*M02*"
    doc=_test_gerber_text(body)
    @test _gerber_member(doc,sqrt(2),sqrt(2))
    @test !_gerber_member(doc,-sqrt(2),sqrt(2))
    @test _gerber_member(doc,2,-.05) # circular aperture endpoint
    body="G75*G36*X1000000Y0D02*G03*X1000000Y0I-1000000J0D01*G37*M02*"
    doc=_test_gerber_text(body)
    for x in range(-.95,.95;length=17),y in range(-.93,.93;length=16)
        @test _gerber_member(doc,x,y)==(hypot(x,y)<1)
    end
    body="G36*X0Y0D02*X4000000Y0D01*X4000000Y4000000D01*X0Y4000000D01*X0Y2000000D01*X1000000Y2000000D01*X1000000Y3000000D01*X3000000Y3000000D01*X3000000Y1000000D01*X1000000Y1000000D01*X1000000Y2000000D01*X0Y2000000D01*X0Y0D01*G37*M02*"
    doc=_test_gerber_text(body)
    @test _gerber_member(doc,.5,.5)
    @test !_gerber_member(doc,2,2)
    @test _gerber_member(doc,3.5,2)
    @test_throws ArgumentError _test_gerber_text("G36*X0Y0D02*X1000000Y0D01*G37*M02*")
end

@testset "Gerber block/repeat image order and solver lowering" begin
    body="%ADD10C,2*%%ADD11C,1*%%ABD20*%D10*X0Y0D03*%LPC*%D11*X0Y0D03*%AB*%%LPD*%%SRX2Y2I3J3*%D20*X2000000Y2000000D03*%SR*%M02*"
    doc=_test_gerber_text(body)
    @test length(doc.objects)==8
    @test !_gerber_member(doc,2,2)
    @test _gerber_member(doc,2.7,2)
    @test _gerber_member(doc,5.7,5)
    grid=CellGrid(.008,.008,16,16);masks=artwork_cell_masks(doc,grid)
    @test count(masks["metal"])==32
    stack=PlanarStackup([PlanarLayer(1.,1.,.001)],TERM_GND,TERM_GND,grid.a,grid.b)
    prob=artwork_planar_problem(doc,stack,grid,Dict("metal"=>(;kind=:sheet,interface=0)),PlanarPort[])
    @test prob.sheets[1].mask==masks["metal"]
    @test_throws ArgumentError artwork_cell_masks(doc,grid;max_bytes=1)
    @test_throws ArgumentError artwork_planar_problem(doc,stack,grid,Dict(),PlanarPort[])
    @test_throws ArgumentError _test_gerber_text(body;max_objects=3)
    @test_throws ArgumentError _test_gerber_text(body;max_bytes=1)
    @test_throws ArgumentError _test_gerber_text("%ADD10C,1*%D10*X0Y0D03*")
    @test_throws ArgumentError _test_gerber_text("%KO*%M02*")
    negative=_test_gerber_text("%IPNEG*%M02*")
    @test all(artwork_cell_masks(negative,grid)["metal"])
    prob=artwork_planar_problem(negative,stack,grid,Dict("metal"=>(;kind=:sheet,interface=0)),PlanarPort[])
    @test all(prob.sheets[1].mask)
    block="%ADD10C,2*%%ADD11C,1*%%ABD20*%D10*X0Y0D03*%LPC*%D11*X0Y0D03*%AB*%%LPD*%D11*X0Y0D03*D20*X0Y0D03*"
    cleared=_test_gerber_text(block*"M02*")
    @test !_gerber_member(cleared,0,0) # block clear erases earlier pad
    @test _gerber_member(cleared,.7,0)
    toggled=_test_gerber_text(block*"%LPC*%D20*X0Y0D03*M02*")
    @test _gerber_member(toggled,0,0) # LPC toggles all block object polarities
    @test !_gerber_member(toggled,.7,0)
    @test_throws ArgumentError _test_gerber_text("%ADD10C,1*%%ABD20*%D10*X0Y0D03*%AB*%D20*X0D03*M02*")
end
