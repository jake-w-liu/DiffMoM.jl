module LibraryStoredDomainTests
using DiffMoM,Test,LinearAlgebra,SHA,TOML
@testset "Library proof preserves original domain failures" begin
    directory=joinpath(@__DIR__,"fixtures","library_stored_domain")
    hashes=TOML.parsefile(joinpath(directory,"sha256.toml"))["sha256"]
    @test length(hashes)==10
    for (file,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(directory,file))))==digest
    end
    before=TOML.parsefile(joinpath(directory,"before.toml"))
    after=TOML.parsefile(joinpath(directory,"after.toml"))
    @test before["source_sha256"]==hashes["source_before.jl"]
    @test after["source_sha256"]==hashes["source_after.jl"]
    @test length(before["rows"])==length(after["rows"])==16
    for index in (1,2,4,5,7,8,14,15,16)
        @test before["rows"][index]["status"]=="accepted"
        @test after["rows"][index]["status"]=="rejected"
    end
    for index in (3,6,9)
        @test after["rows"][index]["status"]=="accepted"
        @test parse(Float64,after["rows"][index]["value"])≈parse(Float64,before["rows"][index]["value"]) rtol=2e-15
    end
    for index in (10,11,12)
        @test after["rows"][index]["status"]=="accepted"
        @test after["rows"][index]["finite"]
        @test parse(Float64,after["rows"][index]["value"])>0
    end
    @test before["rows"][13]["status"]=="rejected"
    @test occursin("MethodError",before["rows"][13]["diagnostic"])
    @test after["rows"][13]["status"]=="accepted"
end

@testset "Library stored positive and placement domains" begin
    stack=PlanarStackup([PlanarLayer(4.,1.,1e-6)],TERM_GND,TERM_GND,.001,.001)
    for value in (big"1e-500",big"1e500",BigFloat(Inf),BigFloat(NaN))
        for method in (:wheeler,:current_sheet)
            @test_throws ArgumentError planar_spiral_inductance(shape=:rectangular,
                turns=value,d_out=.001,d_in=.0005,method=method)
        end
        @test_throws ArgumentError planar_parallel_plate_capacitance(stack,1,0,value)
        @test_throws ArgumentError planar_line(length=value,width=.0001,level=1,metal="pec")
        @test_throws ArgumentError planar_interdigital_capacitor(fingers=2,
            finger_length=.001,width=.0001,gap=value,level=1,metal="pec")
    end
    shape=planar_line(length=.001,width=.0001,level=1,metal="pec")
    for value in (big"1e-500",big"1e500",BigFloat(Inf),BigFloat(NaN))
        @test_throws ArgumentError planar_transform(shape;offset=(value,0))
        @test_throws ArgumentError planar_transform(shape;angle=value)
    end
    @test_throws ArgumentError planar_transform(shape;offset=(1e100,1e100))
    @test_throws ArgumentError planar_transform(shape;offset=(1+im,0))
    for mirror in (false,true),angle in (BigFloat(0),big"0.1",BigFloat(pi/2))
        wide=planar_transform(shape;angle=angle,mirror=mirror,offset=(BigFloat(.002),BigFloat(.003)))
        control=planar_transform(shape;angle=Float64(angle),mirror=mirror,offset=(.002,.003))
        @test wide.polygons[1].vertices==control.polygons[1].vertices
        @test wide.pins==control.pins
        @test length(wide.polygons[1].vertices)==length(shape.polygons[1].vertices)
        for pin in wide.pins
            vertices=wide.polygons[1].vertices;edge=pin.edge
            @test pin.point≈(vertices[edge]+vertices[edge==length(vertices) ? 1 : edge+1])/2 rtol=2e-15
            @test norm(pin.direction)≈1 rtol=2e-15
        end
    end
end

@testset "Library broadside offsets and emitted placement pin contracts" begin
    for offset in (big"1e-1000",-big"1e-1000",big"1e1000",-big"1e1000",BigFloat(Inf),BigFloat(NaN))
        @test_throws ArgumentError planar_broadside_coupled_lines(length=.001,width=.0001,
            upper_level=2,lower_level=1,offset=offset,metal="pec")
    end
    for offset in (0.,.0001,-.0001,BigFloat(.0001),-BigFloat(.0001))
        shape=planar_broadside_coupled_lines(length=.001,width=.0001,
            upper_level=2,lower_level=1,offset=offset,metal="pec")
        @test shape.meta["offset"]==Float64(offset)
    end
    original=planar_line(length=.001,width=.0001,level=1,metal="pec")
    for shift in (0.,.003,1e9,1e10,1e11),mirror in (false,true),angle in (0.,.1,pi/2)
        placed=planar_transform(original;offset=(shift,shift),angle=angle,mirror=mirror,name="placed")
        polygon=only(placed.polygons)
        for pin in placed.pins
            a=polygon.vertices[pin.edge];b=polygon.vertices[mod1(pin.edge+1,length(polygon.vertices))]
            delta=b-a
            @test pin.polygon==polygon.name
            @test pin.level==polygon.level
            @test pin.width==hypot(delta[1],delta[2])
            expected_point=setprecision(BigFloat,256) do
                Float64.((BigFloat.(a)+BigFloat.(b))/2)
            end
            @test pin.point==expected_point
            @test norm(pin.direction)≈1 rtol=2e-15
            @test abs(sum(pin.direction.*delta))<=2eps(Float64)*pin.width
            @test sum(pin.direction.*(pin.point-sum(polygon.vertices)/length(polygon.vertices)))>0
        end
    end
    for value in (nextfloat(0.),-nextfloat(0.))
        pin=planar_pin(planar_transform(original;offset=(value,0)),"p1")
        @test pin.point[1]==value
    end
    for (a,b) in ((nextfloat(0.),2nextfloat(0.)),(floatmax(Float64),floatmax(Float64)),
            (-floatmax(Float64),-floatmax(Float64)),(floatmax(Float64),-floatmax(Float64)))
        expected=setprecision(BigFloat,256) do
            Float64((BigFloat(a)+BigFloat(b))/2)
        end
        @test DiffMoM._lib_midpoint_coordinate(a,b)==expected
    end
end

function reference_inductance(shape,n,outside,inside,method)
    setprecision(BigFloat,256) do
        turns=BigFloat(n);outer=BigFloat(outside);inner=BigFloat(inside)
        average=(outer+inner)/2;fill=(outer-inner)/(outer+inner)
        if method===:wheeler
            coefficients=Dict(:rectangular=>(2.34,2.75),:hexagonal=>(2.33,3.82),:octagonal=>(2.25,3.55))
            k1,k2=coefficients[shape]
            return Float64(BigFloat(k1)*BigFloat(DiffMoM._MU0)*turns^2*average/(1+BigFloat(k2)*fill))
        end
        coefficients=Dict(:rectangular=>(1.27,2.07,.18,.13),:hexagonal=>(1.09,2.23,0.,.17),
            :octagonal=>(1.07,2.29,0.,.19),:circular=>(1.,2.46,0.,.2))
        c1,c2,c3,c4=BigFloat.(coefficients[shape])
        return Float64(BigFloat(DiffMoM._MU0)*turns^2*average*c1/2*
            (log(c2/fill)+c3*fill+c4*fill^2))
    end
end

@testset "Library spiral reference avoids finite intermediate range loss" begin
    for method in (:wheeler,:current_sheet),shape in (:rectangular,:hexagonal,:octagonal,:circular)
        method===:wheeler && shape===:circular && continue
        for turns in (1e-200,1e-158,1e-150,1e-100,1.,1e100,1e150,1e200),
            outside in (1e-300,1e-200,1e-100,.001,1e100,1e200,1e300,1e308),ratio in (.5,.9)
            inside=ratio*outside
            expected=reference_inductance(shape,turns,outside,inside,method)
            if isfinite(expected) && expected>0
                actual=planar_spiral_inductance(;shape,turns,d_out=outside,d_in=inside,method)
                @test actual≈expected rtol=2e-15 atol=nextfloat(0.)
            else
                @test_throws ArgumentError planar_spiral_inductance(;shape,turns,d_out=outside,d_in=inside,method)
            end
        end
        outside=2nextfloat(0.);inside=nextfloat(0.);turns=1e200
        @test planar_spiral_inductance(;shape,turns,d_out=outside,d_in=inside,method)≈
            reference_inductance(shape,turns,outside,inside,method) rtol=2e-15
    end
    # Warmed range-safe products stay fixed-size and allocation free.
    spiral()=planar_spiral_inductance(shape=:rectangular,turns=1e200,d_out=1e-300,d_in=5e-301)
    spiral();@test @allocated(spiral())==0
end

@testset "Library capacitance reference retains representable products" begin
    for area in (nextfloat(0.),1e-300,1e-200,1e-8,1e100,1e200,1e300),thickness in (1e-20,1e-6,1.,1e100)
        stack=PlanarStackup([PlanarLayer(4.,1.,thickness)],TERM_GND,TERM_GND,.001,.001)
        expected=setprecision(BigFloat,256) do
            Float64(BigFloat(DiffMoM._EPS0)*BigFloat(area)/(BigFloat(thickness)/4))
        end
        if isfinite(expected) && expected>0
            @test planar_parallel_plate_capacitance(stack,1,0,area)≈expected rtol=2e-15 atol=nextfloat(0.)
        else
            @test_throws ArgumentError planar_parallel_plate_capacitance(stack,1,0,area)
        end
    end
end

@testset "Library thin bridge vias and scaled dielectric ratios" begin
    for span in (.001,1.)
        @test_throws ArgumentError planar_air_bridge(;span,width=.0001,landing=.0001,
            bridge_level=2,base_level=1,via_type="v",metal="pec",via_margin=prevfloat(.00005))
        shape=planar_air_bridge(;span,width=.0001,landing=.0001,
            bridge_level=2,base_level=1,via_type="v",metal="pec",via_margin=.000025)
        @test length(shape.vias)==2
        for via in shape.vias
            @test length(unique(via.vertices))==4
            x=first.(via.vertices);y=last.(via.vertices)
            @test (maximum(x)-minimum(x))*(maximum(y)-minimum(y))>0
        end
    end
    for height in (1e-300,1e-100,1e-6,1.,1e100,1e300),
            epsilon in (1e-100,1.,1e100),area in (1e-300,1e-100,1e-8,1e100,1e300)
        stack=PlanarStackup([PlanarLayer(epsilon,1.,height)],TERM_GND,TERM_GND,.001,.001)
        expected=setprecision(BigFloat,256) do
            Float64(BigFloat(DiffMoM._EPS0)*BigFloat(area)*BigFloat(epsilon)/BigFloat(height))
        end
        if isfinite(expected) && expected>0
            @test planar_parallel_plate_capacitance(stack,1,0,area)≈expected rtol=2e-15 atol=nextfloat(0.)
        else
            @test_throws ArgumentError planar_parallel_plate_capacitance(stack,1,0,area)
        end
    end
    for ((h1,e1),(h2,e2)) in (((1e-300,1e100),(1e-200,1e100)),
            ((1e300,1e-100),(1e200,1e-100)),((1e-100,1e100),(1e100,1e-100)))
        for area in (1e-300,1e-8,1e300)
            stack=PlanarStackup([PlanarLayer(e1,1.,h1),PlanarLayer(e2,1.,h2)],TERM_GND,TERM_GND,.001,.001)
            expected=setprecision(BigFloat,256) do
                Float64(BigFloat(DiffMoM._EPS0)*BigFloat(area)/(BigFloat(h1)/BigFloat(e1)+BigFloat(h2)/BigFloat(e2)))
            end
            if isfinite(expected) && expected>0
                @test planar_parallel_plate_capacitance(stack,2,0,area)≈expected rtol=2e-15 atol=nextfloat(0.)
            else
                @test_throws ArgumentError planar_parallel_plate_capacitance(stack,2,0,area)
            end
        end
    end
end
end
