using Test
using DiffMoM

@testset "Gradient verification preserves usable perturbations" begin
    calls = Ref(0)
    objective = theta -> begin
        calls[] += 1
        theta[1]^3
    end
    @test complex_step_grad(objective, [0.0], 1) ≈ -1e-60 rtol=4eps(Float64)
    @test calls[] == 2

    # Respect the requested complex-step formula once its imaginary response
    # is normal, including a user-selected step with nonzero truncation error.
    step = 1e-3
    @test complex_step_grad(theta -> sin(theta[1]), [0.0], 1; eps=step) ≈
          sinh(step) / step rtol=4eps(Float64)
    @test complex_step_grad(theta -> 1e-294 * theta[1], [1.0], 1) == 1e-294
    @test_throws ArgumentError complex_step_grad(_ -> 1.0, [1.0], 1)

    row = only(verify_gradient(theta -> sin(theta[1]), [1.0], [0.0]))
    @test row.cs == 1.0
    @test row.rel_err_fd < 1e-12
end

@testset "Finite differences retain representable quotients" begin
    # Both objective samples are finite; only their unscaled difference
    # overflows. The complete quotient remains representable.
    @test fd_grad(theta -> 1e308 * theta[1], [0.0], 1; h=1.0) == 1e308
    @test fd_grad(theta -> complex(1e308, -1e308) * theta[1],
                  [0.0], 1; h=1.0) == complex(1e308, -1e308)
    @test fd_grad(theta -> 1e308 * theta[1], [-1.0], 1;
                  h=2.0, scheme=:forward) == 1e308

    # The separation between two valid perturbations may itself overflow.
    @test fd_grad(theta -> theta[1], [0.0], 1; h=1e308) == 1.0
    @test fd_grad(theta -> theta[1] * 1e-308, [0.0], 1; h=1e308) == 1e-308

    @test_throws OverflowError fd_grad(theta -> 1e308 * (theta[1] * 2),
                                       [0.0], 1; h=0.25)
    @test_throws ArgumentError fd_grad(theta -> theta[1], [1e16], 1; h=1e-6)
end
