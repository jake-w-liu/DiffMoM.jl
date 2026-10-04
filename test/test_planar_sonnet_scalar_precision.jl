module NativeScalarPrecisionTests
using Test,DiffMoM
const DM=DiffMoM

@testset "Native table interpolation preserves closely spaced stored coordinates" begin
    for origin in (0.,1e-300,1e9,1e16,1e100,-1e100),offset in 0:7
        lo=nextfloat(origin,offset);hi=nextfloat(lo,3);key=nextfloat(lo)
        expected=setprecision(BigFloat,256) do
            Float64((BigFloat(key)-BigFloat(lo))/(BigFloat(hi)-BigFloat(lo)))
        end
        @test DM._sonnet_scalar_axis([lo,hi],key,:reject)[3]≈expected rtol=2eps(Float64)
    end
    # Retain finite interpolation for wide opposite-sign ranges, subnormal
    # intervals, literal endpoints and the explicit hold/reject contracts.
    axis=[-floatmax(Float64),floatmax(Float64)]
    for key in (-floatmax(Float64),-1e308,0.,1e308,floatmax(Float64))
        expected=setprecision(BigFloat,256) do
            Float64((BigFloat(key)-BigFloat(axis[1]))/(BigFloat(axis[2])-BigFloat(axis[1])))
        end
        @test DM._sonnet_scalar_axis(axis,key,:reject)[3]≈expected rtol=2eps(Float64)
    end
    @test DM._sonnet_scalar_axis([1.,2.],0.,:hold)[3]==0.
    @test DM._sonnet_scalar_axis([1.,2.],3.,:hold)[3]==1.
    @test_throws ArgumentError DM._sonnet_scalar_axis([1.,2.],0.,:reject)
    @test_throws ArgumentError DM._sonnet_scalar_axis([1.,2.],3.,:reject)
    @test DM._sonnet_scalar_lerp(50.,75.,.5)==62.5
    for (x,y) in ((1e16,nextfloat(1e16,3)),(-1e16,nextfloat(-1e16,3)),
            (-floatmax(Float64),floatmax(Float64)),(1e308,floatmax(Float64))),a in (.1,.5,.9)
        expected=setprecision(BigFloat,256) do
            Float64((1-BigFloat(a))*BigFloat(x)+BigFloat(a)*BigFloat(y))
        end
        @test DM._sonnet_scalar_lerp(x,y,a)≈expected rtol=2eps(Float64)
    end
    mktempdir() do directory
        source=joinpath(directory,"coupon.son")
        cp(joinpath(@__DIR__,"fixtures/native_sonnet_scalar_logarithms/ln_positive/ln_positive.son"),source)
        p=read_sonnet_project(source)
        lo=1e16;key=nextfloat(lo);hi=nextfloat(lo,3)
        write(joinpath(directory,"close.csv"),"$lo,0\n$hi,3\n")
        write(joinpath(directory,"close2.csv"),",$lo,$hi\n$lo,0,3\n$hi,3,6\n")
        p.variables["Loss"]="table1(\"close.csv\",$key)"
        @test sonnet_variable_value(p,"Loss")==1.
        p.variables["Loss"]="table2(\"close2.csv\",$key,$key)"
        @test sonnet_variable_value(p,"Loss")==2.
    end
end
end
