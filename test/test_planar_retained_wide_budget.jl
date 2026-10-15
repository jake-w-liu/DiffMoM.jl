module PlanarRetainedWideBudgetTests
using DiffMoM,Test,LinearAlgebra,SparseArrays

# Independent storage oracle for the explicitly retained numerical arrays.
# BigFloat's own summary includes its MPFR allocation at stored precision.
function array_payload(arrays)
    seen_arrays=Base.IdSet{Any}();scalars=Base.IdSet{Any}();bytes=0
    for array in arrays
        array===nothing && continue
        array in seen_arrays && continue
        push!(seen_arrays,array)
        bytes+=Base.elsize(array)*length(array)
        if eltype(array)<:Union{BigFloat,Complex{BigFloat}}
            for value in array
                for scalar in (value isa BigFloat ? (value,) : (real(value),imag(value)))
                    push!(scalars,scalar)
                end
            end
        end
    end
    return bytes+sum(Base.summarysize,scalars;init=0)
end

raw_arrays(raw::PlanarResult)=(raw.z_mom,raw.lu_fact.factors,raw.lu_fact.ipiv,
    raw.currents,raw.y,raw.s,raw.z0,raw.relative_residuals)

@testset "Retained wide arrays own backing slots and unique MPFR scalars" begin
    a=BigFloat(1.;precision=128);b=BigFloat(2.;precision=192)
    c=BigFloat(3.;precision=256)
    real_values=BigFloat[a,b,a]
    complex_values=Complex{BigFloat}[complex(a,b),complex(c,a)]
    expected=array_payload((real_values,complex_values))
    values=Any[real_values,complex_values,real_values]
    expected+=Base.elsize(values)*length(values)
    @test DiffMoM._subdivision_retained_payload(values)==expected
    setprecision(BigFloat,64) do
        @test DiffMoM._subdivision_retained_payload(values)==expected
        @test DiffMoM._sonnet_retained_solve_payload(values)==expected
    end
    sparse_values=sparse([1,2],[1,2],complex_values,2,2)
    expected_sparse=array_payload((sparse_values.colptr,sparse_values.rowval,sparse_values.nzval))
    @test DiffMoM._subdivision_retained_payload(sparse_values)==expected_sparse
end

@testset "Subdivision reserves actual owned wide results before the next solve" begin
    fixture=joinpath(@__DIR__,"fixtures","native_box_port_attachment","cases",
        "attachment__baseline","project.son")
    model=sonnet_planar_problem(read_sonnet_project(fixture);freq=1e9,grid=(8,8),
        _materials=true,_details=true)
    @test planar_basis_count(model.problem.basis)==26
    options=(mx=8,my=8,surface_zs=model.sheet_zs,via_sigma=model.via_sigma)
    raw=solve_planar(model.problem,1e9;options...,retain_matrix=false,max_bytes=2*1024^2)
    @test eltype(raw.currents)===Complex{BigFloat}
    @test eltype(raw.lu_fact.factors)===Complex{BigFloat}
    expected=array_payload(raw_arrays(raw))
    @test expected>100_000
    @test DiffMoM._subdivision_retained_payload(raw)==expected
    @test DiffMoM._sonnet_retained_solve_payload(raw)==expected
    @test DiffMoM._subdivision_retained_payload(raw.problem)==0
    aliased=Any[raw,raw]
    @test DiffMoM._subdivision_retained_payload(aliased)==expected+sizeof(aliased)

    count=25
    plan=PlanarSubdivision(PlanarCircuit(2,[1,2]),fill(model.problem,count),
        [[1,2] for _ in 1:count])
    part_keywords=fill(options,count)
    # Before the fix, this accepted 2 MiB while retaining over 2.5 MiB of
    # wide factors/currents alone. All numerical systems have only 26 rows.
    @test_throws ArgumentError solve_planar_subdivision(plan,1e9;
        retain_results=true,part_keywords,max_bytes=2*1024^2)
    result=solve_planar_subdivision(plan,1e9;retain_results=true,part_keywords,
        max_bytes=4*1024^2)
    @test length(result.results)==count
    @test all(r->eltype(r.currents)===Complex{BigFloat},result.results)
    retained=array_payload(Iterators.flatten(raw_arrays(r) for r in result.results))+
        sizeof(result.results)
    @test 2*1024^2<retained<4*1024^2
    @test DiffMoM._subdivision_retained_payload(result.results)==retained
    # Retaining coefficients must change only resource use. This coarse
    # wide-recovery fixture has nearly singular Y, so compare the same
    # circuit path rather than round-tripping its admittance through S.
    control=solve_planar_subdivision(plan,1e9;part_keywords,max_bytes=2*1024^2)
    @test result.y≈control.y rtol=1e-12
    @test result.s≈control.s rtol=1e-12
end

function balanced_project()
    record(tokens)=SonnetRecord(1,String.(tokens))
    left=SonnetPolygon(:sheet,0,-1,1,
        [0.0 0.0 .0025 .0025;.0015 .0025 .0025 .0015],"","","",String[])
    right=SonnetPolygon(:sheet,0,-1,2,
        [.0035 .006 .006 .0035;.0015 .0015 .0025 .0025],"","","",String[])
    ports=[SonnetPortSpec(:box,1,0,1,["1","50","0","0","0","0","2"],SonnetRecord[]),
        SonnetPortSpec(:box,2,1,-1,["-1","50","0","0","0","6","2"],SonnetRecord[])]
    component=[record(["TYPE","IDEAL","RES","10"]),record(["GNDREF","FLOAT"]),
        record(["TERMW","FEED"]),record(["SMDP","0","2.5","2","L","3","1"]),
        record(["SMDP","0","3.5","2","R","-3","2"])]
    return SonnetProject("retained_balanced.son",Dict("LNG"=>"MM","CAP"=>"PF","IND"=>"NH"),.001,1e9,
        ["1","6","4","24","16","20","0"],[[".5","1","1","0","0","0"],
        [".5","1","1","0","0","0"]],Vector{String}[],["PEC","0","RES","0"],
        ["PEC","0","RES","0"],[left,right],ports,Dict{String,String}(),[component],SonnetRecord[],SonnetRecord[])
end

@testset "Balanced native floating sources retain the complete source solve" begin
    result=solve_sonnet_project(balanced_project(),1e8;raw=true,mx=36,my=24,
        max_bytes=4*1024^2)
    em=result.em;contracted=em.raw;source=contracted.raw
    @test em isa DiffMoM._SonnetBalancedSourceResult
    @test planar_basis_count(em.problem.basis)==38
    # Enumerate the three result layers explicitly; borrowed geometry is
    # intentionally absent and aliased contraction/response arrays dedupe.
    arrays=(em.z0,em.currents,em.y,em.s,em.voltage_transfer,
        contracted.contraction,contracted.z0,contracted.currents,contracted.y,contracted.s,
        source.contraction,source.z0,source.z_mom,source.lu_fact.factors,source.lu_fact.ipiv,
        source.basis_scale,source.currents,source.y,source.s,source.iterations,
        source.relative_residuals,source.galerkin_relative_residuals)
    expected=array_payload(arrays)
    @test expected>DiffMoM._subdivision_retained_payload(contracted)
    @test DiffMoM._subdivision_retained_payload(em)==expected
    @test DiffMoM._sonnet_retained_solve_payload(em)==expected
    seen=Base.IdSet{Any}()
    @test DiffMoM._subdivision_retained_payload(contracted,seen)>0
    @test DiffMoM._subdivision_retained_payload(em,seen)==
        expected-array_payload((contracted.contraction,contracted.z0,contracted.currents,
            contracted.y,contracted.s,source.contraction,source.z0,source.z_mom,
            source.lu_fact.factors,source.lu_fact.ipiv,source.basis_scale,source.currents,
            source.y,source.s,source.iterations,source.relative_residuals,
            source.galerkin_relative_residuals))
    @test all(isfinite,result.s)
    @test opnorm(result.s)<=1+1e-10
end
end
