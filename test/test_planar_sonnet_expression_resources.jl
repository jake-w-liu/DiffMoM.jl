module NativeScalarExpressionResourcesTests
using Test,DiffMoM
const DM=DiffMoM

prepared(p,ex,variables,active)=DM._sonnet_expr(ex,p,variables,1e9,active)
function allocation(p,text)
    ex=DM._sonnet_parse_scalar(text)
    variables=Dict{String,Float64}();active=Set{String}()
    prepared(p,ex,variables,active)
    @allocated prepared(p,ex,variables,active)
end

@testset "Native scalar arithmetic avoids per-node function dictionaries" begin
    p=read_sonnet_project(joinpath(@__DIR__,"fixtures/native_sonnet_scalar_logarithms/ln_positive/ln_positive.son"))
    text="sin(0.25)+cos(0.5)*sqrt(4)+ln(exp(2))+p2h(2)/FREQ"
    expected=sin(.25)+cos(.5)*2+2+2
    @test sonnet_variable_value(p,text)≈expected rtol=1e-15
    @test allocation(p,text)<=4096
    for (expression,expected) in (("1+2+3+4",10.),("2*3*4",24.),
            ("-(2)",-2.),("5-3",2.),("2^3",8.),("8/4",2.),
            ("h2p(FREQ)",1.),("p2h(1)",1e9),("m2p(.001)",1.),("p2m(1)",.001),
            ("ln(-exp(2))",2.),("log10(-100)",2.))
        @test sonnet_variable_value(p,expression)≈expected rtol=1e-15
    end
    @test signbit(sonnet_variable_value(p,"+(-0.0)"))
    p.variables["Leaf"]="3";p.variables["Branch"]="Leaf*Leaf"
    @test sonnet_variable_value(p,"Branch+Leaf")==12.
    @test sonnet_variable_value(p,"Branch+Leaf";variables=Dict("Leaf"=>4.))==20.
    p.variables["Leaf"]="5"
    @test sonnet_variable_value(p,"Branch+Leaf")==30.
    for expression in ("log(exp(2))","sin(1,2)","ln(2,3)","p2h(1,2)","run(1)")
        @test_throws ArgumentError sonnet_variable_value(p,expression)
    end
    p.variables["Leaf"]="Branch"
    @test_throws ArgumentError sonnet_variable_value(p,"Branch")
end
end
