module PayloadSumTests
using Test,DiffMoM,Random

function oracle(values)
    any(value->value<0,values) && return "payloads must be nonnegative"
    total=sum(BigInt.(values);init=BigInt(0))
    total>typemax(Int) && return "workspace estimate overflows Int"
    return Int(total)
end
function check(values)
    expected=oracle(values)
    if expected isa Int
        @test DiffMoM._checked_payload_sum("check",values...)===expected
    else
        caught=try DiffMoM._checked_payload_sum("check",values...);nothing catch err;err end
        @test caught isa ArgumentError && occursin(expected,sprint(showerror,caught))
    end
end
function ordinary_allocations(values::NTuple{3,Int})
    for _ in 1:100;DiffMoM._checked_payload_sum("allocation check",values...);end
    return @allocated DiffMoM._checked_payload_sum("allocation check",values...)
end
@testset "Payload sum exact integer domains, overflow priority and allocation" begin
    limit=typemax(Int)
    for values in ((),(0,),(false,true),(limit,0),(limit-1,1),(limit,1),
            (UInt128(limit),Int8(0)),(UInt128(limit),UInt8(1)),
            (BigInt(limit)-1,UInt8(1)),(BigInt(limit)+1,),
            (BigInt(1)<<10000,),(-(BigInt(1)<<10000),),
            (typemax(UInt128),),(typemax(Int128),),
            (typemax(UInt128),-1),(-1,typemax(UInt128)),
            (Int128(limit),UInt64(0)),(BigInt(0),UInt128(0),Int8(0)))
        check(values)
    end
    rng=MersenneTwister(0xc80192)
    types=(Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64,Int128,UInt128,Bool)
    for _ in 1:256
        values=Tuple(rand(rng,rand(rng,types)) for _ in 1:rand(rng,0:12))
        check(values)
        check((values...,BigInt(rand(rng,Int128))))
    end
    for values in ((16,32,64),(0,0,0),(limit-2,1,1))
        @test ordinary_allocations(values)==0
    end
end
end
