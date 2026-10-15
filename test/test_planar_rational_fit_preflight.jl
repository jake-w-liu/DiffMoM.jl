module PlanarRationalFitPreflightTests
using DiffMoM,Test

function reject_fit(series,frequencies)
    try
        planar_fit_rational(series,frequencies;order=1,max_bytes=1)
    catch exception
        exception isa ArgumentError || rethrow()
        return exception
    end
    error("fit accepted insufficient workspace")
end

@testset "Rational fit budget rejects before sample-sized frequency storage" begin
    allocations=Int[]
    for count in (64,4096)
        series=fill(fill(.02+0im,1,1),count)
        frequencies=collect(range(1e8,1e10;length=count))
        error=reject_fit(series,frequencies)
        @test occursin("vector fit requires",sprint(showerror,error))
        bytes=minimum(@allocated(reject_fit(series,frequencies)) for _ in 1:3)
        @test bytes<16_384
        push!(allocations,bytes)
    end
    @test abs(allocations[2]-allocations[1])<4096
end

@testset "Fit passivity samples share the numerical budget" begin
    frequencies=[1e8,2e8,3e8];series=fill(fill(.02+0im,1,1),3)
    for points in (8192,typemax(Int))
        @test_throws ArgumentError planar_fit_rational(series,frequencies;
            order=1,n_passivity=points,max_bytes=65536)
    end
    # This previously accepted 64 KiB while two 64 KiB grids coexisted.
    # A sufficient cap must retain the original constant passive response.
    for tolerance in (1e-9,0.)
        model=planar_fit_rational(series,frequencies;order=1,n_passivity=8192,
            passivity_tol=tolerance,max_bytes=256*1024)
        @test model.globally_passive
        @test all(f->isapprox(only(planar_rational_eval(model,f)),.02;rtol=1e-12),frequencies)
    end
end
end
