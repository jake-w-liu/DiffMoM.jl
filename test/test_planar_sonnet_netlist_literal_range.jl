module NativeSonnetNetlistLiteralRangeTests
using Test, DiffMoM, LinearAlgebra

function project(kind,literal,unit)
    units=Dict("RES"=>"OH","CAP"=>"PF","IND"=>"NH")
    units[kind]=unit
    parameter=kind=="RES" ? "R" : kind=="CAP" ? "C" : "L"
    records=[SonnetRecord(1,[kind,"1","2",parameter*"="*literal]),
        SonnetRecord(2,["DEF2P","1","2","Device","R","50"])]
    return SonnetNetlistProject("literal_range.son",units,1e9,records,records)
end

@testset "Native CKT literals preserve nonzero SI values before storage" begin
    for (kind,unit,literal) in (("CAP","PF","1e-320"),("CAP","FF","1e-310"),
            ("IND","NH","1e-320"),("IND","PH","1e-313"),
            ("RES","OH","1e-1000"),("RES","KOH","1e308"),
            ("CAP","PF","Inf"),("IND","NH","NaN"),
            ("CAP","PF","-1"),("IND","NH","-1"))
        @test_throws ArgumentError sonnet_planar_circuit(project(kind,literal,unit))
    end
    for (kind,unit,scale) in (("RES","OH",1.),("CAP","PF",1e-12),("IND","NH",1e-9))
        field=kind=="RES" ? :r : kind=="CAP" ? :c : :l
        zero=sonnet_planar_circuit(project(kind,"0",unit))
        @test getproperty(only(zero.elements),field)===0.
        # Exact native decimal 2.6e-312 PF rounds to the smallest positive F.
        literal=kind=="CAP" ? "2.6e-312" : kind=="IND" ? "2.6e-315" : "2.6e-324"
        tiny=sonnet_planar_circuit(project(kind,literal,unit))
        @test getproperty(only(tiny.elements),field)==
            Float64(parse(BigFloat,literal)*BigFloat(scale))==nextfloat(0.)
    end
    # Unit conversion remains physically equivalent through the public CKT
    # solve, including continuous-frequency reactive branches.
    for (kind,units,values) in (("RES",("OH","KOH","MOH"),("1000","1",".001")),
            ("CAP",("PF","NF","F"),("1000","1","1e-9")),
            ("IND",("NH","UH","H"),("1000","1","1e-6")))
        responses=[solve_planar_circuit(sonnet_planar_circuit(project(kind,value,unit)),250e6).s
            for (unit,value) in zip(units,values)]
        @test all(response->response≈first(responses),responses)
    end
    @test_throws ArgumentError sonnet_planar_circuit(project("CAP",repeat("0",16385),"PF"))
    for exponent in ("-1000000000000000000","-10000000000000000000")
        @test_throws ArgumentError sonnet_planar_circuit(project("RES","1e"*exponent,"OH"))
        @test_throws ArgumentError sonnet_planar_circuit(project("CAP","1.01e"*exponent,"PF"))
        for literal in ("0e"*exponent,"0.000e"*exponent,"-0.0e"*exponent)
            @test iszero(only(sonnet_planar_circuit(project("RES",literal,"OH")).elements).r)
        end
    end
    # A file stores the same Float64 value regardless of caller arithmetic
    # configuration, and parsing must restore the caller's scoped settings.
    function configured(bits,mode)
        setprecision(BigFloat,bits) do
            setrounding(BigFloat,mode) do
                values=[only(sonnet_planar_circuit(project("RES","1.00001","OH")).elements).r,
                    only(sonnet_planar_circuit(project("CAP","2.6e-312","PF")).elements).c]
                (;values,bits=precision(BigFloat),mode=rounding(BigFloat))
            end
        end
    end
    configurations=[(bits,mode) for bits in (16,32,256,1024),mode in (RoundNearest,RoundDown,RoundUp,RoundToZero)]
    for (bits,mode) in configurations
        outcome=configured(bits,mode)
        @test outcome.values==[1.00001,nextfloat(0.)]
        @test outcome.bits==bits && outcome.mode==mode
    end
    tasks=[Threads.@spawn configured(bits,mode) for (bits,mode) in configurations]
    for ((bits,mode),task) in zip(configurations,tasks)
        outcome=fetch(task)
        @test outcome.values==[1.00001,nextfloat(0.)]
        @test outcome.bits==bits && outcome.mode==mode
    end
    # Exact integer arithmetic constructs decimal literals immediately on
    # either side of the 1.0/nextfloat(1.0) midpoint. Fixed 256-bit parsing
    # would round both long decimals to that midpoint before Float64 storage.
    digits=2000;decimal_scale=big(10)^digits
    midpoint=decimal_scale+div(decimal_scale,big(2)^53)
    decimal(n)=string(div(n,decimal_scale))*"."*lpad(string(rem(n,decimal_scale)),digits,'0')
    for (integer,expected) in ((midpoint-1,1.),(midpoint,1.),(midpoint+1,nextfloat(1.)))
        @test only(sonnet_planar_circuit(project("RES",decimal(integer),"OH")).elements).r==expected
    end
    mktempdir() do directory
        source=joinpath(directory,"coupon.son")
        write(source,"FTYP SONNETPRJ 19\nDIM\nLNG MIL\nFREQ GHZ\nRES OH\nCAP PF\nIND NH\nEND DIM\nCKT\nPRJ 1 2 child.son 2 0\nCAP 1 2 C=1e-320\nDEF2P 1 2 Device R 50\nEND CKT\n")
        parsed=read_sonnet_project(source)
        calls=Ref(0)
        provider=(path,f)->begin;calls[]+=1;Matrix{ComplexF64}(I,2,2);end
        @test_throws ArgumentError sonnet_planar_circuit(parsed;project_response=provider)
        @test calls[]==0
    end
end
end
