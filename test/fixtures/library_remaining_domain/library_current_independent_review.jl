using DiffMoM,Test,LinearAlgebra,SHA,TOML
const rows=Dict{String,Any}[]
const source_hash=bytes2hex(sha256(read(joinpath(@__DIR__,"..","..","src","planar","PlanarLibrary.jl"))))
function check_capacitance(ratios,area)
    stack=PlanarStackup([PlanarLayer(epsr,1.,height) for (height,epsr) in ratios],
        TERM_GND,TERM_GND,.001,.001)
    expected=setprecision(BigFloat,4096) do
        Float64(BigFloat(DiffMoM._EPS0)*BigFloat(area)/
            sum(BigFloat(height)/BigFloat(epsr) for (height,epsr) in ratios))
    end
    if isfinite(expected)&&expected>0
        actual=planar_parallel_plate_capacitance(stack,length(ratios),0,area)
        @test actual≈expected rtol=3e-15 atol=nextfloat(0.)
        push!(rows,Dict("layer_count"=>length(ratios),"area"=>area,"expected"=>expected,"actual"=>actual))
    else
        @test_throws ArgumentError planar_parallel_plate_capacitance(stack,length(ratios),0,area)
    end
end
@testset "Current broadside offsets reject nonzero storage loss" begin
    for value in (big"1e-1000",-big"1e-1000",big"1e1000",-big"1e1000")
        @test_throws ArgumentError planar_broadside_coupled_lines(length=.001,width=.0001,
            upper_level=2,lower_level=1,offset=value,metal="pec")
    end
    for value in (BigFloat(0),BigFloat(.0002),BigFloat(nextfloat(0.)))
        shape=planar_broadside_coupled_lines(length=.001,width=.0001,
            upper_level=2,lower_level=1,offset=value,metal="pec")
        @test shape.meta["offset"]==Float64(value)
    end
end
@testset "Current transformed pins match actual geometry over original falsifiers" begin
    shape=planar_line(length=.001,width=.0001,level=1,metal="pec")
    for mirror in (false,true),angle in (0.,.1,.3,pi/4,1.),offset in
            ((0.,1e9),(0.,1e10),(0.,1e11),(1e11,0.),(1e11,1e11),
                (nextfloat(0.),0.),(-nextfloat(0.),0.),(.002,.003))
        placed=planar_transform(shape;angle,offset,mirror)
        poly=only(placed.polygons);vertices=poly.vertices
        # Exact stored-coordinate area sign independently identifies outward.
        area=sum(Rational{BigInt}(vertices[i][1])*Rational{BigInt}(vertices[mod1(i+1,length(vertices))][2])-
            Rational{BigInt}(vertices[i][2])*Rational{BigInt}(vertices[mod1(i+1,length(vertices))][1]) for i in eachindex(vertices))
        for pin in placed.pins
            a=vertices[pin.edge];b=vertices[mod1(pin.edge+1,length(vertices))]
            delta=b-a;width=hypot(delta...)
            expected_midpoint=Float64.((Rational{BigInt}.(a)+Rational{BigInt}.(b))/2)
            expected_normal=sign(area)*DiffMoM._P2(delta[2]/width,-delta[1]/width)
            @test pin.width==width
            @test pin.point==expected_midpoint
            @test pin.direction==expected_normal
            @test norm(pin.direction)≈1 rtol=3e-16
            @test abs(dot(pin.direction,delta/width))<=2e-16
        end
    end
end
@testset "Current air-bridge via geometry is noncollapsed or rejects" begin
    for span in (.001,1.)
        @test_throws ArgumentError planar_air_bridge(;span,width=.0001,landing=.0001,
            bridge_level=2,base_level=1,via_type="v",metal="pec",via_margin=prevfloat(.00005))
    end
    for margin in (0.,.00001,.00004)
        shape=planar_air_bridge(span=.001,width=.0001,landing=.0001,
            bridge_level=2,base_level=1,via_type="v",metal="pec",via_margin=margin)
        for via in shape.vias
            @test length(unique(via.vertices))==4
            normalized,bbox=planar_normalize_polygon(via.vertices)
            @test length(normalized)==4
        end
    end
end
@testset "Current capacitance series ratios vs 4096-bit independent oracle" begin
    combinations=[(1e-300,1e100),(1e300,1e-100),(1e-300,1e-100),
        (1e300,1e100),(1e-100,1e300),(1e100,1e-300),(1e-6,4.),
        (nextfloat(0.),1e100),(1e100,nextfloat(0.))]
    for ratios in ([v] for v in combinations),area in (1e-300,1e-200,1e-8,1e200,1e300)
        check_capacitance(ratios,area)
    end
    for left in combinations,right in combinations,area in (1e-300,1e-8,1e300)
        check_capacitance([left,right],area)
    end
end
@test source_hash==bytes2hex(sha256(read(joinpath(@__DIR__,"..","..","src","planar","PlanarLibrary.jl"))))
open(joinpath(@__DIR__,"library_current_independent_review.toml"),"w") do io
    TOML.print(io,Dict("source_sha256"=>source_hash,"rows"=>rows))
end
