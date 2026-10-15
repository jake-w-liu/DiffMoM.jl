module MeepReferenceTableContractTests
using Test,CSV,DataFrames
include(joinpath(@__DIR__,"..","validation","meep","regen_mom_reference_table.jl"))

@testset "Meep table preserves finite measured evidence before solving" begin
    mktempdir() do dir
        path=joinpath(dir,"reference.csv")
        names=[c.name for c in _MEEP_REFERENCE_CASES]
        measured=[.9,.8,.7,.6,.5]
        calls=Ref(0)
        solve(;kw...)=(calls[]+=1;.25)
        invalid=(DataFrame(case=names[1:1],R_MEEP=measured[1:1]),
            DataFrame(case=vcat(names,names[1]),R_MEEP=vcat(measured,measured[1])),
            DataFrame(case=names,R_MEEP=[NaN;measured[2:end]]),
            DataFrame(case=names,R_MEEP=[Inf;measured[2:end]]),
            DataFrame(case=names,R_MEEP=[missing;measured[2:end]]),
            DataFrame(case=names,R_MEEP=fill("unmeasured",length(names))))
        for original in invalid
            CSV.write(path,original);saved=read(path)
            @test_throws ArgumentError regenerate_mom_reference_table(path;reflectance=solve)
            @test calls[]==0 && read(path)==saved
        end
        CSV.write(path,DataFrame(case=names,R_MEEP=measured))
        saved=read(path)
        @test_throws ArgumentError regenerate_mom_reference_table(path;reflectance=(;kw...)->NaN)
        @test read(path)==saved
        rows=regenerate_mom_reference_table(path;reflectance=solve)
        persisted=CSV.read(path,DataFrame)
        @test calls[]==length(names)
        @test rows.R_MEEP==persisted.R_MEEP==measured
        @test rows.metal_fill_pct[4]≈100*184/196
        @test all(==(.25),rows.R_MoM) && rows.T_MEEP≈1 .-measured
    end
end
end
