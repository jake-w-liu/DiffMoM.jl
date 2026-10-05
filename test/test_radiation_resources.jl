# An out-of-line call must retain allocation-free immutable reductions even
# when the compiler cannot inline the shared cancellation classifier.
@noinline function _radiation_reduction_barrier(value, real_bound, imag_bound)
    DiffMoM._farfield_vector_reduction_requires_exact(
        value, real_bound, imag_bound, 6)
end

function _radiation_reduction_barrier_allocation(value, real_bound, imag_bound)
    _radiation_reduction_barrier(value, real_bound, imag_bound)
    @allocated for _ in 1:1024
        _radiation_reduction_barrier(value, real_bound, imag_bound)
    end
end

@testset "Radiation reductions across a compiler call boundary" begin
    real_bound = Vec3(1.0, 2.0, 3.0)
    imag_bound = Vec3(3.0, 2.0, 1.0)
    ordinary = CVec3(1 + 2im, 2 + 1im, 3 + 1im)
    cancelled = CVec3(1.0e-16, 0, 0)
    for (value, expected) in ((ordinary, false), (cancelled, true))
        @test _radiation_reduction_barrier(value, real_bound, imag_bound) == expected
        @test DiffMoM._farfield_vector_reduction_requires_exact(
            value, MVector(real_bound), MVector(imag_bound), 6) == expected
        @test _radiation_reduction_barrier_allocation(
            value, real_bound, imag_bound) == 0
    end
    @test !_radiation_reduction_barrier(zero(CVec3), zero(Vec3), zero(Vec3))
    @test _radiation_reduction_barrier(ordinary, Vec3(Inf, 0, 0), imag_bound)
end
