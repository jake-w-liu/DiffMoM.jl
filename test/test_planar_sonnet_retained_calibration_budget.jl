module PlanarSonnetRetainedCalibrationBudgetTests
using DiffMoM, Test, LinearAlgebra

function numerical_budget(problem,frequency;kw...)
    budget=1
    for _ in 1:6
        try
            return solve_planar(problem,frequency;max_bytes=budget,kw...),budget
        catch error
            error isa ArgumentError || rethrow()
            required=match(r"requires (\d+) raw bytes",sprint(showerror,error))
            required===nothing && rethrow()
            next=parse(Int,required[1])
            budget<next<=8*1024^2 || rethrow()
            budget=next
        end
    end
    error("tiny numerical budget preflight did not converge")
end

@testset "Native calibration reserves the raw result while child solves run" begin
    mktempdir() do directory
        source=joinpath(directory,"line.son")
        write(source,"""FTYP SONPROJ 19
VER "18.53"
DIM
FREQ GHZ
LNG MM
END DIM
CONTROL
OPTIONS -d
END CONTROL
GEO
TMET "Lossless" 0 SUP 0 0 0 0
BMET "Lossless" 0 SUP 0 0 0 0
MET "Lossless" 0 SUP 0 0 0 0
BOX 1 2 2 8 8 100 0
0.5 1 1 0 0 0 0 "Air_bottom"
0.5 1 1 0 0 0 0 "Air_top"
LORGN 0 1 U
POR1 BOX
POLY 9 1
3
1 50 0 0 0 0 1
POR1 BOX
POLY 9 1
1
2 50 0 0 0 2 1
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
        project=read_sonnet_project(source);frequency=10e9
        model=sonnet_planar_problem(project;freq=frequency,_materials=true,_details=true)
        @test planar_basis_count(model.problem.basis)<32
        port=first(model.problem.ports)
        _,first_budget=numerical_budget(DiffMoM._sonnet_feed_standard(model.problem,port,8),frequency;
            mx=16,my=8,block=8)
        _,second_budget=numerical_budget(DiffMoM._sonnet_feed_standard(model.problem,port,16),frequency;
            mx=32,my=8,block=8)
        numerical=max(first_budget,second_budget)
        raw=solve_planar(model.problem,frequency;max_bytes=numerical,mx=8,my=8,block=8,
            surface_zs=model.sheet_zs,via_sigma=model.via_sigma)
        reserve=model.payload+DiffMoM._sonnet_raster_response_workspace(model)
        arrays=(raw.z_mom,raw.lu_fact.factors,raw.lu_fact.ipiv,raw.currents,
            raw.y,raw.s,raw.z0,raw.relative_residuals)
        retained=sum(sizeof,arrays)
        @test DiffMoM._sonnet_retained_solve_payload(raw)==retained
        @test DiffMoM._sonnet_child_solve_budget(raw,reserve,reserve+retained+1)==1
        @test_throws ArgumentError DiffMoM._sonnet_child_solve_budget(raw,reserve,reserve+retained-1)
        # The old flow gave the full numerical budget to each standard
        # while retaining raw. Reconstruct that control without another
        # full native solve, then verify the aggregate wrapper rejects it.
        chains=DiffMoM._sonnet_wall_feed_chains(model,frequency;
            max_bytes=numerical,mx=8,my=8,block=8)
        expected_y=deembed_ports(raw.y,chains)
        expected_s=planar_y_to_s(expected_y,model.z0)
        error=try
            solve_sonnet_project(project,frequency;max_bytes=reserve+numerical,
                mx=8,my=8,block=8)
            nothing
        catch error
            error
        end
        @test error isa ArgumentError
        @test occursin("max_bytes=$(numerical-retained)",sprint(showerror,error))
        generous=solve_sonnet_project(project,frequency;max_bytes=reserve+retained+numerical,
            mx=8,my=8,block=8)
        @test generous.s≈expected_s rtol=1e-12
        @test generous.y≈expected_y rtol=1e-12
        compact_solve=solve_sonnet_project(project,frequency;max_bytes=reserve+retained+numerical,
            retain_matrix=false,mx=8,my=8,block=8)
        @test compact_solve.raw.z_mom===nothing
        @test compact_solve.s≈expected_s rtol=1e-12
        # The iterative path retains FFT kernels/workspaces instead of
        # dense factors and must leave room for the same child solves.
        fft=solve_sonnet_project(project,frequency;method=:ufft,max_bytes=8*1024^2,
            mx=8,my=8,block=8,rtol=1e-10)
        @test fft.s≈expected_s rtol=1e-8
        fft_retained=DiffMoM._sonnet_retained_solve_payload(fft.raw)
        @test fft_retained>sizeof(fft.raw.currents)+sizeof(fft.raw.y)+sizeof(fft.raw.s)
        @test DiffMoM._sonnet_child_solve_budget(fft.raw,reserve,reserve+fft_retained+1)==1
        # Shared dense buffers are charged once, including retain_matrix=false.
        aliased=PlanarResult(raw.problem,raw.omega,raw.freq,raw.lu_fact.factors,
            raw.lu_fact,raw.currents,raw.y,raw.s,raw.z0,raw.relative_residuals)
        @test DiffMoM._sonnet_retained_solve_payload(aliased)==retained-sizeof(raw.z_mom)
        compact=PlanarResult(raw.problem,raw.omega,raw.freq,nothing,
            raw.lu_fact,raw.currents,raw.y,raw.s,raw.z0,raw.relative_residuals)
        @test DiffMoM._sonnet_retained_solve_payload(compact)==retained-sizeof(raw.z_mom)
        # Wide storage uses its own precision, independently of caller
        # precision, and shared MPFR components are counted only once.
        wide=BigFloat(1.;precision=192);widezero=BigFloat(0.;precision=192)
        currents=fill(complex(wide,widezero),size(raw.currents))
        factor=lu(Matrix{Complex{BigFloat}}(I,size(raw.lu_fact.factors)...))
        wide_result=PlanarResult(raw.problem,raw.omega,raw.freq,raw.z_mom,
            factor,currents,raw.y,raw.s,raw.z0,raw.relative_residuals)
        scalars=Base.IdSet{Any}()
        for array in (factor.factors,currents),value in array,component in (real(value),imag(value))
            push!(scalars,component)
        end
        measured_wide=sizeof(Complex{BigFloat})*(length(factor.factors)+length(currents))+
            sum(Base.summarysize,scalars)
        expected=retained-sizeof(raw.lu_fact.factors)-sizeof(raw.currents)+measured_wide
        @test DiffMoM._sonnet_retained_solve_payload(wide_result)==expected
    end
end
end
