using Test
if isdefined(Main, :DiffMoM)
using .DiffMoM
else
    using DiffMoM
end
using LinearAlgebra

function _complex_vector_allocation_2d(n::Int)
    zeros(ComplexF64, n)
    return @allocated zeros(ComplexF64, n)
end

function _complex_matrix_allocation_2d(m::Int, n::Int)
    Matrix{ComplexF64}(undef, m, n)
    return @allocated Matrix{ComplexF64}(undef, m, n)
end

function _real_vector_allocation_2d(n::Int)
    zeros(Float64, n)
    return @allocated zeros(Float64, n)
end

@testset "2D TM MoM" begin

    @testset "Mesh2D construction" begin
        mesh = Mesh2D((-1.0, 1.0), (-0.5, 0.5), 4, 2)
        @test mesh.ncells == 8
        @test mesh.nx == 4
        @test mesh.ny == 2
        @test mesh.dx ≈ 0.5
        @test mesh.dy ≈ 0.5
        @test mesh.cell_area ≈ 0.25

        # Centers should be at cell midpoints
        @test mesh.centers[1] ≈ Vec2(-0.75, -0.25)
        @test mesh.centers[8] ≈ Vec2(0.75, 0.25)

        widest = floatmax(Float64)
        wide_mesh = Mesh2D((-widest, widest), (0.0, 1.0), 2, 1)
        @test wide_mesh.dx == widest
        @test wide_mesh.centers ==
              [Vec2(-widest / 2, 0.5), Vec2(widest / 2, 0.5)]

        # Equivalent radius: πa² = cell_area
        @test equivalent_radius(mesh)^2 * π ≈ mesh.cell_area atol=1e-14

        # Preserve positive subnormal cell areas instead of underflowing the
        # division by π before the square root.
        subnormal_mesh = Mesh2D(
            (0.0, nextfloat(0.0)), (0.0, 1.0), 1, 1)
        @test equivalent_radius(subnormal_mesh) > 0.0
        Z_subnormal, D_subnormal = assemble_vie_2d(
            subnormal_mesh, 1.0, [0.0])
        @test Z_subnormal == ones(ComplexF64, 1, 1)
        @test all(isfinite, D_subnormal)
        @test !iszero(D_subnormal[1, 1])

        # Invalid inputs
        @test_throws ArgumentError Mesh2D((1.0, -1.0), (-0.5, 0.5), 4, 2)
        @test_throws ArgumentError Mesh2D((-1.0, 1.0), (-0.5, 0.5), 0, 2)
        @test_throws ArgumentError Mesh2D((0.0, Inf), (-0.5, 0.5), 4, 2)
        @test_throws ArgumentError Mesh2D((-1.0, 1.0), (NaN, 0.5), 4, 2)
        @test_throws ArgumentError Mesh2D(
            (1.0, nextfloat(1.0)), (0.0, 1.0), 3, 1)
        @test_throws ArgumentError Mesh2D(
            (0.0, 1.0), (0.0, 1.0), 3, 3; max_cells=8)
        @test_throws ArgumentError Mesh2D(
            (0.0, 1.0), (0.0, 1.0), 3, 3;
            max_raw_bytes=9sizeof(Vec2) - 1)
        @test Mesh2D(
            (0.0, 1.0), (0.0, 1.0), 3, 3;
            max_cells=9, max_raw_bytes=9sizeof(Vec2)).ncells == 9
        large_origin = 1.0e16
        duplicate_center = Vec2(large_origin + 2.0, 0.5)
        @test_throws ArgumentError Mesh2D(
            [duplicate_center, duplicate_center], 4.0,
            2, 2, 1, 4.0, 1.0, large_origin, 0.0)
        @test_throws ArgumentError Mesh2D(
            (-1.0, 1.0), (-0.5, 0.5), typemax(Int), 2)
    end

    @testset "2D Green's function" begin
        k = 2π
        r1 = Vec2(0.0, 0.0)
        r2 = Vec2(1.0, 0.0)

        G = greens_2d(r1, r2, k)

        # G₂D = (-i/4) H₀⁽²⁾(kR), verify against direct computation
        using SpecialFunctions
        R = 1.0
        G_ref = (-im / 4) * besselh(0, 2, k * R)
        @test G ≈ G_ref atol=1e-14

        # Symmetry: G(r1,r2) = G(r2,r1)
        @test greens_2d(r1, r2, k) ≈ greens_2d(r2, r1, k) atol=1e-14

        # Self-term: returns zero (handled by self_cell_integral)
        @test greens_2d(r1, r1, k) == 0.0 + 0.0im
        @test_throws ArgumentError greens_2d(r1, r2, 0.0)
        @test_throws ArgumentError greens_2d(r1, r2, Inf)
        @test_throws ArgumentError greens_2d(Vec2(NaN, 0.0), r2, k)

        # Distinct subnormal-scale separations remain finite; squaring the
        # displacement or applying an absolute coincidence cutoff loses them.
        tiny_separation = 1e-200
        tiny_point = Vec2(tiny_separation, 0.0)
        tiny_green = greens_2d(r1, tiny_point, 1.0)
        tiny_reference = (-im / 4) * besselh(0, 2, tiny_separation)
        @test tiny_green ≈ tiny_reference rtol=2e-15

        underflow_k = 1e-200
        underflow_distance = 1e-200
        underflow_point = Vec2(underflow_distance, 0.0)
        underflow_expected = ComplexF64(
            -(log(underflow_k) + log(underflow_distance) - log(2.0) +
              Float64(Base.MathConstants.eulergamma)) / (2π),
            -0.25,
        )
        underflow_green = greens_2d(r1, underflow_point, underflow_k)
        @test underflow_green ≈ underflow_expected rtol=2e-15

        subnormal_k = 1e-300
        subnormal_distance = 1e-20
        subnormal_expected = ComplexF64(
            -(log(subnormal_k * subnormal_distance) - log(2.0) +
              Float64(Base.MathConstants.eulergamma)) / (2π),
            -0.25,
        )
        @test greens_2d(r1, Vec2(subnormal_distance, 0.0), subnormal_k) ≈
              subnormal_expected rtol=2e-15
        @test @allocated(greens_2d(r1, underflow_point, underflow_k)) == 0

        # Preserve the exact Float64 wavenumber-distance product through the
        # oscillatory Hankel phase.  Rounding kR first selects an unrelated
        # phase even though the final Green value is finite.
        large_green_k = 1.1
        large_green_distance = 1.0e20
        large_green_reference = setprecision(BigFloat, 2304) do
            argument = BigFloat(large_green_k) *
                       BigFloat(large_green_distance)
            term = one(Complex{BigFloat})
            series = term
            for order in 1:24
                odd = BigFloat(2order - 1)
                term *= Complex{BigFloat}(0, -1) * (-odd^2) /
                        (BigFloat(order) * 8 * argument)
                series += term
            end
            phase = exp(Complex{BigFloat}(
                0, -(argument - BigFloat(pi) / 4)))
            ComplexF64(
                (-Complex{BigFloat}(0, 1) / 4) *
                sqrt(2 / (BigFloat(pi) * argument)) * phase * series)
        end
        @test greens_2d(
            Vec2(large_green_distance, 0.0), r1, large_green_k) ==
              large_green_reference

        overflow_green_reference = setprecision(BigFloat, 2304) do
            argument = BigFloat(1.0e200)^2
            phase = exp(Complex{BigFloat}(
                0, -(argument - BigFloat(pi) / 4)))
            ComplexF64(
                (-Complex{BigFloat}(0, 1) / 4) *
                sqrt(2 / (BigFloat(pi) * argument)) * phase)
        end
        @test greens_2d(
            Vec2(1.0e200, 0.0), r1, 1.0e200) ==
              overflow_green_reference

        # Decay with distance
        G_near = abs(greens_2d(r1, Vec2(0.5, 0.0), k))
        G_far = abs(greens_2d(r1, Vec2(5.0, 0.0), k))
        @test G_near > G_far
    end

    @testset "Self-cell integral" begin
        k = 2π
        a_eq = 0.01  # small cell

        D_self = self_cell_integral_2d(k, a_eq)
        @test !isnan(D_self)
        @test !isinf(D_self)

        # For very small ka, the self-cell integral should be dominated by
        # the logarithmic singularity of H₀⁽²⁾: G ~ (-i/4)(1 - 2i/π ln(kR/2) - ...)
        # Numerical check: should have nonzero real and imaginary parts
        @test abs(real(D_self)) > 0
        @test abs(imag(D_self)) > 0

        # The direct Hankel antiderivative cancels its logarithmic real part
        # when (k*a)^2 approaches machine precision.  Check the stable small-
        # argument branch against the leading analytic expansion, whose next
        # correction is O((k*a)^2 log(k*a)).
        tiny_k = 1e-6
        tiny_a = 1e-6
        tiny_ka = tiny_k * tiny_a
        tiny_expected = ComplexF64(
            -(tiny_a^2 / 2) *
            (log(tiny_ka / 2) + Float64(Base.MathConstants.eulergamma) - 0.5),
            -(π * tiny_a^2 / 4),
        )
        tiny_integral = self_cell_integral_2d(tiny_k, tiny_a)
        @test tiny_integral ≈ tiny_expected rtol=2e-15
        @test @allocated(self_cell_integral_2d(tiny_k, tiny_a)) == 0
        self_cell_integral_2d(k, 0.2)
        @test @allocated(self_cell_integral_2d(k, 0.2)) == 0

        # k² overflows, but the analytic integral remains a representable
        # subnormal value after the 1/k² scaling.
        extreme_k = 1e160
        extreme_a = 1e-160
        extreme_expected = setprecision(BigFloat, 256) do
            k_big = BigFloat(extreme_k)
            a_big = BigFloat(extreme_a)
            ka_big = k_big * a_big
            H1_big = Complex{BigFloat}(
                besselj(1, ka_big), -bessely(1, ka_big))
            imaginary_unit =
                Complex{BigFloat}(zero(BigFloat), one(BigFloat))
            return ComplexF64(
                (-imaginary_unit * BigFloat(π) / (2 * k_big^2)) *
                (ka_big * H1_big -
                 2 * imaginary_unit / BigFloat(π)))
        end
        @test self_cell_integral_2d(extreme_k, extreme_a) == extreme_expected

        function large_self_reference(k_value, a_value)
            return setprecision(BigFloat, 2304) do
                k_big = BigFloat(k_value)
                argument = k_big * BigFloat(a_value)
                term = one(Complex{BigFloat})
                series = term
                for order in 1:24
                    odd = BigFloat(2order - 1)
                    term *= Complex{BigFloat}(0, -1) *
                            (BigFloat(4) - odd^2) /
                            (BigFloat(order) * 8 * argument)
                    series += term
                end
                hankel = sqrt(2 / (BigFloat(pi) * argument)) *
                         exp(Complex{BigFloat}(
                             0, -(argument - 3BigFloat(pi) / 4))) *
                         series
                ComplexF64(
                    (-Complex{BigFloat}(0, 1) * BigFloat(pi) /
                     (2 * k_big^2)) *
                    (argument * hankel -
                     2Complex{BigFloat}(0, 1) / BigFloat(pi)))
            end
        end
        @test self_cell_integral_2d(1.1, 1.0e20) ==
              large_self_reference(1.1, 1.0e20)
        @test self_cell_integral_2d(1.0e200, 1.0e200) ==
              large_self_reference(1.0e200, 1.0e200)

        # Positive equivalent radius required
        @test_throws ArgumentError self_cell_integral_2d(k, 0.0)
        @test_throws ArgumentError self_cell_integral_2d(0.0, a_eq)
        @test_throws ArgumentError self_cell_integral_2d(Inf, a_eq)
        @test_throws ArgumentError self_cell_integral_2d(k, Inf)
    end

    @testset "VIE assembly and solve" begin
        k0 = 2π
        mesh = Mesh2D((-0.5, 0.5), (-0.5, 0.5), 5, 5)
        chi = fill(1.0, mesh.ncells)  # εᵣ = 2

        Z, D = assemble_vie_2d(mesh, k0, chi)
        @test size(Z) == (25, 25)
        @test size(D) == (25, 25)
        @test D == DiffMoM.assemble_D_matrix(mesh, k0)
        matrix_bytes = sizeof(ComplexF64) * mesh.ncells^2
        @test_throws ArgumentError DiffMoM.assemble_D_matrix(
            mesh, k0; max_output_bytes=matrix_bytes - 1)
        @test DiffMoM.assemble_D_matrix(
            mesh, k0; max_output_bytes=matrix_bytes) == D
        @test_throws ArgumentError assemble_vie_2d(
            mesh, k0, chi; max_output_bytes=2matrix_bytes - 1)
        @test assemble_vie_2d(
            mesh, k0, chi; max_output_bytes=2matrix_bytes) == (Z, D)
        Z_reference = Matrix{ComplexF64}(undef, mesh.ncells, mesh.ncells)
        @inbounds for n in 1:mesh.ncells
            for m in 1:mesh.ncells
                Z_reference[m, n] = -k0^2 * chi[n] * D[m, n]
            end
            Z_reference[n, n] += 1.0
        end
        @test Z == Z_reference
        assemble_vie_2d(mesh, k0, chi)
        assembly_alloc = @allocated assemble_vie_2d(mesh, k0, chi)
        matrix_alloc = _complex_matrix_allocation_2d(
            mesh.ncells, mesh.ncells)
        @test assembly_alloc <= 2 * matrix_alloc + 512

        # Z should be invertible
        @test cond(Z) < 1e10

        # With chi = 0 (free space), Z = I
        Z0, _ = assemble_vie_2d(mesh, k0, zeros(mesh.ncells))
        @test Z0 ≈ I(mesh.ncells) atol=1e-14
        @test_throws DimensionMismatch assemble_vie_2d(
            mesh, k0, zeros(mesh.ncells - 1))
        @test_throws ArgumentError assemble_vie_2d(
            mesh, k0, fill(NaN, mesh.ncells))
        @test_throws ArgumentError assemble_vie_2d(
            mesh, Inf, zeros(mesh.ncells))

        # Solve with plane wave
        E_inc = planewave_2d(mesh, k0, 0.0)
        @test all(abs.(E_inc) .≈ 1.0)  # unit amplitude

        vr = solve_vie_2d(mesh, k0, chi, E_inc)
        @test length(vr.E_total) == 25
        @test !any(isnan, vr.E_total)
        @test !any(isinf, vr.E_total)
        E_inc_bad = copy(E_inc)
        E_inc_bad[1] = NaN + 0im
        @test_throws ArgumentError solve_vie_2d(
            mesh, k0, chi, E_inc_bad)
        @test_throws DimensionMismatch solve_vie_2d(
            mesh, k0, chi, E_inc[1:end-1])
        @test_throws ArgumentError solve_vie_2d(
            mesh, k0, chi, E_inc;
            max_output_bytes=2matrix_bytes - 1)

        # In free space (chi=0), total field = incident field
        vr_free = solve_vie_2d(mesh, k0, zeros(mesh.ncells), E_inc)
        @test vr_free.E_total ≈ E_inc atol=1e-12
    end

    @testset "Extreme VIE product scaling" begin
        # Every physical input and final result is representable, but k₀²χ
        # overflows and Z⁻¹D underflows if those intermediates are formed.
        mesh = Mesh2D((0.0, 1e-100), (0.0, 1e-100), 1, 1)
        k0 = 1e100
        chi = [1e200]
        E_inc = ComplexF64[1.0 + 0.0im]
        r_obs = [Vec2(2e-100, 5e-101)]

        Z, D = assemble_vie_2d(mesh, k0, chi)
        vr = solve_vie_2d(mesh, k0, chi, E_inc)
        G_obs = green_obs_matrix(r_obs, mesh, k0)
        E_scat = scattered_field_2d(vr, r_obs)
        J, _ = jacobian_scattered_field_2d(vr, r_obs)

        Z_ref, E_ref, E_scat_ref, J_ref =
            setprecision(BigFloat, 256) do
                k_big = BigFloat(k0)
                chi_big = BigFloat(chi[1])
                area_big = BigFloat(mesh.cell_area)
                D_big = Complex{BigFloat}(D[1, 1])
                G_big = Complex{BigFloat}(G_obs[1, 1])
                Z_big = one(Complex{BigFloat}) -
                        k_big^2 * chi_big * D_big
                E_big = inv(Z_big)
                scattered_big =
                    k_big^2 * area_big * chi_big * E_big * G_big
                jacobian_big = k_big^2 * area_big * G_big / Z_big^2
                return ComplexF64(Z_big), ComplexF64(E_big),
                       ComplexF64(scattered_big), ComplexF64(jacobian_big)
            end

        @test Z[1, 1] ≈ Z_ref rtol=2e-15
        @test vr.E_total[1] ≈ E_ref rtol=2e-15
        @test E_scat[1] ≈ E_scat_ref rtol=2e-15
        @test J[1, 1] == J_ref
        @test all(isfinite, Z)
        @test all(isfinite, vr.E_total)
        @test all(isfinite, E_scat)
        @test all(isfinite, J)

        # A still larger finite wavenumber has an overflowing square while
        # k₀²D and k₀²A remain order one. The stored D self term is subnormal,
        # so assembly must evaluate the combined coefficient before rounding D.
        square_overflow_mesh = Mesh2D(
            (0.0, 1e-160), (0.0, Float64(π) * 1e-160), 1, 1)
        square_overflow_k0 = 1e160
        square_overflow_chi = [1.0]
        square_overflow_incident = ComplexF64[1.0 + 0.0im]
        square_overflow_observation = [Vec2(
            2e-160, square_overflow_mesh.centers[1][2])]

        Z_overflow, D_overflow = assemble_vie_2d(
            square_overflow_mesh, square_overflow_k0,
            square_overflow_chi)
        vr_overflow = solve_vie_2d(
            square_overflow_mesh, square_overflow_k0,
            square_overflow_chi, square_overflow_incident)
        G_overflow = green_obs_matrix(
            square_overflow_observation, square_overflow_mesh,
            square_overflow_k0)
        scattered_overflow = scattered_field_2d(
            vr_overflow, square_overflow_observation)
        jacobian_overflow, _ = jacobian_scattered_field_2d(
            vr_overflow, square_overflow_observation)

        D_overflow_ref, Z_overflow_ref, E_overflow_ref,
        G_overflow_ref, scattered_overflow_ref, jacobian_overflow_ref =
            setprecision(BigFloat, 256) do
                k_big = BigFloat(square_overflow_k0)
                area_big = BigFloat(square_overflow_mesh.cell_area)
                a_big = sqrt(area_big / BigFloat(π))
                ka_big = k_big * a_big
                imaginary_unit =
                    Complex{BigFloat}(zero(BigFloat), one(BigFloat))
                H1_big = Complex{BigFloat}(
                    besselj(1, ka_big), -bessely(1, ka_big))
                D_big = (-imaginary_unit * BigFloat(π) /
                         (2 * k_big^2)) *
                        (ka_big * H1_big -
                         2 * imaginary_unit / BigFloat(π))
                dx_big = BigFloat(square_overflow_observation[1][1]) -
                         BigFloat(square_overflow_mesh.centers[1][1])
                dy_big = BigFloat(square_overflow_observation[1][2]) -
                         BigFloat(square_overflow_mesh.centers[1][2])
                phase_big = k_big * sqrt(dx_big^2 + dy_big^2)
                G_big = (-imaginary_unit / 4) * Complex{BigFloat}(
                    besselj(0, phase_big), -bessely(0, phase_big))
                Z_big = one(Complex{BigFloat}) - k_big^2 * D_big
                E_big = inv(Z_big)
                scattered_big = k_big^2 * area_big * E_big * G_big
                jacobian_big = k_big^2 * area_big * G_big / Z_big^2
                return ComplexF64(D_big), ComplexF64(Z_big),
                       ComplexF64(E_big), ComplexF64(G_big),
                       ComplexF64(scattered_big), ComplexF64(jacobian_big)
            end

        @test D_overflow[1, 1] == D_overflow_ref
        @test Z_overflow[1, 1] ≈ Z_overflow_ref rtol=2e-15
        @test vr_overflow.E_total[1] ≈ E_overflow_ref rtol=2e-15
        @test G_overflow[1, 1] ≈ G_overflow_ref rtol=2e-15
        @test scattered_overflow[1] ≈ scattered_overflow_ref rtol=2e-15
        @test jacobian_overflow[1, 1] ≈ jacobian_overflow_ref rtol=2e-15
        @test all(isfinite, Z_overflow)
        @test all(isfinite, vr_overflow.E_total)
        @test all(isfinite, scattered_overflow)
        @test all(isfinite, jacobian_overflow)
    end

    @testset "Extreme VIE right-hand-side scaling" begin
        mesh = Mesh2D((0.0, 1.0), (0.0, 0.25), 4, 1)
        k0 = 0.1
        chi = fill(1e4, mesh.ncells)
        observation = [Vec2(1.25, 0.125)]
        Z, _ = assemble_vie_2d(mesh, k0, chi)
        G_obs = green_obs_matrix(observation, mesh, k0)

        target_pattern = ComplexF64[1.0, 1.0, -1.0, -1.0]
        source_scales = ComplexF64[
            k0^2 * chi[n] * G_obs[1, n] * mesh.cell_area
            for n in 1:mesh.ncells
        ]
        unit_field = target_pattern ./ source_scales
        unit_rhs = Z * unit_field
        rhs_multiplier = 0.8 * floatmax(Float64)
        E_inc = rhs_multiplier .* unit_rhs
        @test all(isfinite, E_inc)

        vr = solve_vie_2d(mesh, k0, chi, E_inc)
        reference = setprecision(BigFloat, 256) do
            Z_big = Matrix{Complex{BigFloat}}(Z)
            rhs_big = Complex{BigFloat}.(E_inc)
            ComplexF64.(Z_big \ rhs_big)
        end

        @test all(isfinite, vr.E_total)
        @test all(
            isapprox(vr.E_total[index], reference[index]; rtol=2e-14)
            for index in eachindex(reference))

        # Each source contribution is representable, but the first two
        # positive terms overflow when accumulated before the cancelling
        # negative terms. The complete field remains representable.
        source_terms, scattered_reference =
            setprecision(BigFloat, 256) do
                terms = ComplexF64[]
                field = zero(Complex{BigFloat})
                for n in 1:mesh.ncells
                    contribution =
                        BigFloat(k0)^2 * BigFloat(chi[n]) *
                        Complex{BigFloat}(vr.E_total[n]) *
                        Complex{BigFloat}(G_obs[1, n]) *
                        BigFloat(mesh.cell_area)
                    push!(terms, ComplexF64(contribution))
                    field += contribution
                end
                terms, ComplexF64(field)
            end
        @test all(isfinite, source_terms)
        @test !isfinite(source_terms[1] + source_terms[2])

        scattered = scattered_field_2d(vr, observation)[1]
        @test isfinite(scattered)
        @test isapprox(
            real(scattered), real(scattered_reference); rtol=2e-15)
        @test isapprox(
            imag(scattered), imag(scattered_reference); rtol=2e-15)
    end

    @testset "Ill-conditioned VIE solve accuracy" begin
        mesh = Mesh2D((0.0, 1.0), (0.0, 0.5), 2, 1)
        k0 = 1e150
        chi = [-1e-240, 1e60]
        Z, _ = assemble_vie_2d(mesh, k0, chi)

        # The right-hand side is exactly the second stored matrix column, so
        # the exact solution of the stored system is the second unit vector.
        # Working-precision LU loses the first component by about 267 orders
        # of magnitude because this matrix has reciprocal condition ~1e-300.
        E_inc = copy(@view Z[:, 2])
        vr = solve_vie_2d(mesh, k0, chi, E_inc)
        reference = setprecision(BigFloat, 4096) do
            Z_big = Complex{BigFloat}.(Z)
            rhs_big = Complex{BigFloat}.(E_inc)
            ComplexF64.(Z_big \ rhs_big)
        end

        @test reference == ComplexF64[0.0, 1.0]
        @test vr.E_total == reference
        @test all(isfinite, vr.E_total)

        observation = [Vec2(2.0, 0.25)]
        J, G_obs = jacobian_scattered_field_2d(vr, observation)
        jacobian_reference = setprecision(BigFloat, 4096) do
            Z_big = Complex{BigFloat}.(Z)
            G_big = Complex{BigFloat}.(G_obs)
            E_big = Complex{BigFloat}.(vr.E_total)
            sensitivity_transpose = Z_big \ transpose(G_big)
            ComplexF64.(
                BigFloat(k0)^2 * BigFloat(mesh.cell_area) .*
                transpose(sensitivity_transpose) .* transpose(E_big))
        end

        @test !iszero(jacobian_reference[1, 2])
        @test J == jacobian_reference
        @test all(isfinite, J)
    end

    @testset "Plane wave excitation" begin
        mesh = Mesh2D((-1.0, 1.0), (-1.0, 1.0), 4, 4)
        k0 = 2π

        # Different incident angles
        for phi_inc in [0.0, π/4, π/2, π]
            E_inc = planewave_2d(mesh, k0, phi_inc)
            @test all(abs.(E_inc) .≈ 1.0)  # unit amplitude for plane wave
        end
        @test_throws ArgumentError planewave_2d(mesh, Inf, 0.0)
        @test_throws ArgumentError planewave_2d(mesh, k0, Inf)
        @test_throws ArgumentError planewave_2d(mesh, k0, 0.0; E0=Inf)
        planewave_2d(mesh, k0, 0.0)
        planewave_alloc = @allocated planewave_2d(mesh, k0, 0.0)
        @test planewave_alloc <=
              _complex_vector_allocation_2d(mesh.ncells) + 128

        # The phase k₀ k̂⋅r overflows Float64 although its exponential is a
        # finite unit phasor. Check the exceptional path independently.
        extreme_x = 1e200
        extreme_mesh = Mesh2D(
            (extreme_x, nextfloat(extreme_x)), (0.0, 1.0), 1, 1)
        extreme_k0 = 1e200
        extreme_incident = planewave_2d(extreme_mesh, extreme_k0, 0.0)
        extreme_reference = setprecision(BigFloat, 256) do
            phase = BigFloat(extreme_k0) *
                    BigFloat(extreme_mesh.centers[1][1])
            ComplexF64(exp(Complex{BigFloat}(zero(BigFloat), -phase)))
        end
        @test extreme_incident == [extreme_reference]

        # A finite dot product can lose an observable phase through
        # cancellation, and the public incidence angle must be evaluated at
        # sufficient precision rather than by promoting rounded Float64 trig.
        cancellation_angle = 0.2
        cancellation_x = 1.0e12
        cancellation_y = -Float64(
            (cos(cancellation_angle) / sin(cancellation_angle)) *
            cancellation_x)
        cancellation_mesh = Mesh2D(
            (cancellation_x - 0.5, cancellation_x + 0.5),
            (cancellation_y - 0.5, cancellation_y + 0.5),
            1, 1)
        cancellation_reference = setprecision(BigFloat, 2048) do
            sin_angle, cos_angle = sincos(BigFloat(cancellation_angle))
            center = cancellation_mesh.centers[1]
            phase = cos_angle * BigFloat(center[1]) +
                    sin_angle * BigFloat(center[2])
            ComplexF64(exp(Complex{BigFloat}(0, -phase)))
        end
        cancellation_precision =
            DiffMoM._planewave_phase_precision_2d(
                1.0,
                Vec2(cos(cancellation_angle), sin(cancellation_angle)),
                cancellation_mesh.centers[1])
        @test planewave_2d(
            cancellation_mesh, 1.0, cancellation_angle) ==
              [cancellation_reference]
        @test_throws ArgumentError planewave_2d(
            cancellation_mesh, 1.0, cancellation_angle;
            max_exact_phase_work=cancellation_precision - 1)

        # Phase consistency: E(r) = exp(-ik₀ k̂·r)
        E_inc = planewave_2d(mesh, k0, 0.0)
        for i in 1:mesh.ncells
            expected = exp(-im * k0 * mesh.centers[i][1])
            @test E_inc[i] ≈ expected atol=1e-14
        end
    end

    @testset "Mie series - PEC cylinder" begin
        k0 = 2π
        a = 0.5

        c, N = mie_coefficients_2d(k0, a, 1.0; pec=true)
        @test length(c) == 2N + 1
        @test !any(isnan, c)

        # PEC: cₙ = -Jₙ(k₀a) / Hₙ⁽²⁾(k₀a)
        using SpecialFunctions
        k0a = k0 * a
        c0_ref = -besselj(0, k0a) / besselh(0, 2, k0a)
        @test c[N + 1] ≈ c0_ref atol=1e-14  # n=0 coefficient
        @test_throws ArgumentError mie_coefficients_2d(Inf, a, 1.0; pec=true)
        @test_throws ArgumentError mie_coefficients_2d(k0, a, 1.0; nmax=-1, pec=true)
        @test_throws ArgumentError mie_coefficients_2d(
            k0, a, 1.0; nmax=DiffMoM._MAX_MIE2D_ORDER + 1, pec=true)
        @test_throws ArgumentError DiffMoM._validated_mie2d_order(
            1.0e8, nothing)
        auto_large_order = DiffMoM._validated_mie2d_order(10_000.0, nothing)[1]
        @test auto_large_order == 10_089
        @test mie_coefficients_2d(
            1.0, 10_000.0, 1.0;
            nmax=auto_large_order)[2] == auto_large_order
        high_order_pec, high_order_pec_N = mie_coefficients_2d(
            1.0, 1.0, 1.0; nmax=152, pec=true)
        @test high_order_pec_N == 152
        @test all(isfinite, high_order_pec)
        @test iszero(high_order_pec[high_order_pec_N + 1 + 152])

        # Miller-ratio normalization must remain finite at a valid zero of J0.
        bessel_zero_size = 2.404825557695773
        bessel_zero_pec, bessel_zero_N = mie_coefficients_2d(
            bessel_zero_size, 1.0, 1.0; nmax=65, pec=true)
        @test bessel_zero_N == 65
        @test all(isfinite, bessel_zero_pec)
        @test bessel_zero_pec[bessel_zero_N + 2] ≈
              -0.9623064131054789 - 0.19045414251610876im rtol=8eps(Float64)
        @test all(isfinite, mie_scattered_field_2d(
            bessel_zero_size, 1.0, 1.0, [Vec2(2.0, 0.0)];
            nmax=65, pec=true))

        # Total field on cylinder surface should be near zero for PEC
        # Tolerance limited by Mie series truncation at finite nmax
        r_surf = [Vec2(a * cos(phi), a * sin(phi)) for phi in range(0, 2π, length=37)[1:36]]
        E_total = mie_total_field_2d(k0, a, 1.0, r_surf; pec=true)
        @test maximum(abs.(E_total)) < 1e-6

        # Coefficients that individually underflow may still multiply the
        # surface Hankel field to a representable term.  The coupled fallback
        # preserves the exact PEC boundary cancellation.
        underflow_surface = [Vec2(1e-200, 0.0)]
        @test mie_scattered_field_2d(
            1.0, 1e-200, 1.0, underflow_surface;
            nmax=10, pec=true) == ComplexF64[-1.0 + 1e-200im]
        @test mie_total_field_2d(
            1.0, 1e-200, 1.0, underflow_surface;
            nmax=10, pec=true) == zeros(ComplexF64, 1)
    end

    @testset "Mie series - dielectric cylinder" begin
        k0 = 2π
        a = 0.3
        eps_r = 4.0

        c, N = mie_coefficients_2d(k0, a, eps_r)
        @test !any(isnan, c)

        matched_coefficients, matched_order =
            mie_coefficients_2d(1.0, 1e-200, 1.0)
        @test matched_order == 10
        @test all(iszero, matched_coefficients)

        tiny_size = 1e-160
        tiny_coefficients, tiny_order =
            mie_coefficients_2d(1.0, tiny_size, 4.0)
        @test tiny_order == 10
        @test all(isfinite, tiny_coefficients)
        @test tiny_coefficients[tiny_order + 1] ≈
              ComplexF64(0.0, -(3π / 4) * tiny_size^2) rtol=5e-4

        tiny_enz, tiny_enz_order =
            mie_coefficients_2d(1.0, tiny_size, 0.0)
        @test all(isfinite, tiny_enz)
        @test tiny_enz[tiny_enz_order + 1] ≈
              ComplexF64(0.0, (π / 4) * tiny_size^2) rtol=5e-4

        underflow_coefficients, underflow_order =
            mie_coefficients_2d(1e-200, 1e-200, 4.0)
        @test underflow_order == 10
        @test all(iszero, underflow_coefficients)

        mie_coefficients_2d(1.0, tiny_size, 4.0)
        @test @allocated(mie_coefficients_2d(1.0, tiny_size, 4.0)) < 1_200_000

        tiny_surface = [Vec2(tiny_size, 0.0)]
        tiny_pec_scattered = mie_scattered_field_2d(
            1.0, tiny_size, 1.0, tiny_surface; pec=true)
        @test all(isfinite, tiny_pec_scattered)
        @test tiny_pec_scattered[1] ≈ -1.0 + 0.0im rtol=1e-15
        @test all(isfinite, mie_total_field_2d(
            1.0, tiny_size, 1.0, tiny_surface; pec=true))
        @test all(isfinite, mie_scattered_field_2d(
            1.0, tiny_size, 4.0, tiny_surface))
        @test all(isfinite, mie_scattered_field_2d(
            1.0, tiny_size, 0.0, tiny_surface))

        underflow_size = 1e-200
        underflow_surface = [Vec2(underflow_size, 0.0)]
        @test mie_scattered_field_2d(
            underflow_size, underflow_size, 1.0, underflow_surface) ==
              zeros(ComplexF64, 1)
        @test mie_total_field_2d(
            underflow_size, underflow_size, 1.0, underflow_surface) ==
              ones(ComplexF64, 1)

        # In a matched medium the scattered field vanishes, but the incident
        # phase product can overflow even though its complex exponential is
        # finite and representable.
        extreme_k = 1.0e200
        extreme_radius = 1.0e-200
        extreme_observation = [Vec2(1.0e200, 0.0)]
        extreme_reference = setprecision(BigFloat, 256) do
            phase = BigFloat(extreme_k) *
                    BigFloat(extreme_observation[1][1])
            ComplexF64(exp(
                Complex{BigFloat}(zero(BigFloat), -phase)))
        end
        @test mie_total_field_2d(
            extreme_k, extreme_radius, 1.0,
            extreme_observation) == [extreme_reference]

        exact_angle_observation = [Vec2(1.0e100, 1.0e100)]
        exact_angle_reference = setprecision(BigFloat, 512) do
            angle = BigFloat(0.2)
            phase = BigFloat(1.0) *
                (cos(angle) * BigFloat(exact_angle_observation[1][1]) +
                 sin(angle) * BigFloat(exact_angle_observation[1][2]))
            ComplexF64(exp(Complex{BigFloat}(0, -phase)))
        end
        @test mie_total_field_2d(
            1.0, 1.0, 1.0, exact_angle_observation;
            phi_inc=0.2, nmax=0) == [exact_angle_reference]

        # The Float64 direction/coordinate product can underflow before a
        # finite k0 restores a representable incident phase.
        underflow_phase_angle = reinterpret(Float64, 0x3ff51700e0c14b25)
        underflow_phase_k = reinterpret(Float64, 0x427fffffffffffff)
        underflow_phase_radius = nextfloat(0.0)
        underflow_phase_point = [Vec2(underflow_phase_radius, 0.0)]
        underflow_phase_reference = setprecision(BigFloat, 512) do
            angle = BigFloat(underflow_phase_angle)
            phase = BigFloat(underflow_phase_k) * cos(angle) *
                    BigFloat(underflow_phase_radius)
            ComplexF64(exp(Complex{BigFloat}(0, -phase)))
        end
        @test mie_total_field_2d(
            underflow_phase_k, underflow_phase_radius, 1.0,
            underflow_phase_point;
            phi_inc=underflow_phase_angle, nmax=0) ==
              [underflow_phase_reference]

        # Symmetry: c_{-n} should satisfy specific relations
        # For symmetric incidence (phi_inc=0), c_{-n} = c_n
        for n in 1:min(N, 5)
            @test c[-n + N + 1] ≈ c[n + N + 1] atol=1e-12
        end

        # The exact epsilon-near-zero limit is finite and agrees with a
        # sufficiently small positive permittivity.
        c_enz, N_enz = mie_coefficients_2d(k0, a, 0.0; nmax=6)
        c_near_enz, _ = mie_coefficients_2d(k0, a, 1e-14; nmax=6)
        @test N_enz == 6
        @test all(isfinite, c_enz)
        @test norm(c_enz - c_near_enz) / max(norm(c_near_enz), eps()) < 1e-10
        @test_throws ArgumentError mie_coefficients_2d(
            k0, a, Inf; nmax=3)

        # Dimensionless formulas remain valid when k0 and a individually sit
        # at opposite Float64 extremes.
        balanced_size = 1e308 * 1e-308
        for balanced_eps in (4.0, 0.0)
            balanced, _ = mie_coefficients_2d(
                1e308, 1e-308, balanced_eps; nmax=3)
            reference, _ = mie_coefficients_2d(
                1.0, balanced_size, balanced_eps; nmax=3)
            @test balanced == reference
        end

        tiny_material, tiny_material_order = mie_coefficients_2d(
            1.0, 1.0, 1e-300; nmax=10)
        @test all(isfinite, tiny_material)
        @test tiny_material[tiny_material_order + 4] ≈
              -5.538599011872469e-9 + 7.44217641634246e-5im rtol=2e-15
        negative_material, _ = mie_coefficients_2d(
            1.0, 1.0, -1e6; nmax=3)
        @test all(isfinite, negative_material)
        large_internal, large_internal_order = mie_coefficients_2d(
            1.0, 1e200, 1e308; nmax=0)
        @test large_internal_order == 0
        @test large_internal[1] ==
              -0.007330558908273147 + 0.0853042895425868im
        finite_amos_large, _ = mie_coefficients_2d(
            1.0, 1e200, 4.0; nmax=0)
        @test finite_amos_large[1] ==
              -0.16966452998983803 + 0.37533781750999373im
        high_order_material, high_order_material_N = mie_coefficients_2d(
            1.0, 1.0, 4.0; nmax=217)
        @test all(isfinite, high_order_material)
        @test iszero(high_order_material[high_order_material_N + 1 + 153])
        for extreme_material in (0.36, 1e12, -1e12)
            high_order_extreme, high_order_extreme_N =
                mie_coefficients_2d(
                    1.0, 1.0, extreme_material; nmax=160)
            @test all(isfinite, high_order_extreme)
            @test iszero(
                high_order_extreme[high_order_extreme_N + 1 + 153])
        end
        high_order_enz, high_order_enz_N = mie_coefficients_2d(
            1.0, 1.0, 0.0; nmax=160)
        @test all(isfinite, high_order_enz)
        @test iszero(high_order_enz[high_order_enz_N + 1 + 153])
        high_order_size = 100.0
        high_order_low_index = (0.51 / high_order_size)^2
        high_order_tail, high_order_tail_N = mie_coefficients_2d(
            1.0, high_order_size, high_order_low_index; nmax=220)
        @test high_order_tail[high_order_tail_N + 1 + 193] ≈
              -1.862141133606551e-144 +
              1.3646029215880168e-72im rtol=4eps(Float64)

        tiny_dielectric, tiny_dielectric_order = mie_coefficients_2d(
            1.0, 1e-40, 4.0; nmax=1)
        @test tiny_dielectric[tiny_dielectric_order + 2] ==
              ComplexF64(-8.7e-322, -2.9452431127404304e-161)
        subnormal_dielectric, subnormal_dielectric_order =
            mie_coefficients_2d(1.0, 1e-80, 4.0; nmax=1)
        @test subnormal_dielectric[subnormal_dielectric_order + 2] ==
              ComplexF64(-0.0, -2.945e-321)
        @test_throws ArgumentError mie_coefficients_2d(
            1.0, 1e-30, 4.0; nmax=100_000)
        @test_throws ArgumentError mie_coefficients_2d(
            1.0, 1e-30, 4.0;
            nmax=DiffMoM._MAX_MIE2D_SMALL_SERIES_ORDER + 1)
        @test_throws ArgumentError mie_coefficients_2d(
            1.0, 1.0, 1e-66;
            nmax=DiffMoM._MAX_MIE2D_INTERNAL_PAIR_ORDER + 1)
        @test_throws ArgumentError mie_coefficients_2d(
            1.0, 1.0, 0.16; nmax=100_000)
        try
            mie_coefficients_2d(1.0, 1.0, 0.16; nmax=100_000)
        catch
        end
        @test @allocated(try
            mie_coefficients_2d(1.0, 1.0, 0.16; nmax=100_000)
        catch
        end) < 4_096

        exceptional_point = [Vec2(1e308, 1e308)]
        exceptional_pec = mie_scattered_field_2d(
            2.0, 0.5, 1.0, exceptional_point;
            phi_inc=0.2, nmax=1, pec=true)
        @test exceptional_pec == ComplexF64[
            -5.9087432546189295e-155 - 4.068155253276633e-155im]
        exceptional_dielectric = mie_scattered_field_2d(
            2.0, 0.5, 4.0, exceptional_point;
            phi_inc=0.2, nmax=1)
        @test exceptional_dielectric == ComplexF64[
            8.560495527615164e-156 - 8.048017980335233e-155im]
        @test_throws ArgumentError mie_scattered_field_2d(
            1.0, 1.0, 1.0,
            fill(Vec2(1e100, 0.0), 1_000);
            nmax=10, pec=true)
        @test_throws ArgumentError mie_total_field_2d(
            1.0, 1.0, 1.0,
            fill(Vec2(1e100, 1e100), 2_000);
            phi_inc=0.2, nmax=0)
        @test_throws ArgumentError mie_scattered_field_2d(
            1.0, 1e-30, 4.0, [Vec2(1e100, 0.0)];
            nmax=40)
    end

    @testset "Mie series - validation and allocation" begin
        k0 = 2π
        a = 0.5
        eps_r = 2.5
        observations = [Vec2(2a, 0.0), Vec2(3a, a)]

        @test_throws DomainError mie_scattered_field_2d(
            k0, a, eps_r, [Vec2(0.5a, 0.0)]; nmax=3)
        @test_throws DomainError mie_scattered_field_2d(
            1.0, nextfloat(0.0), 4.0, [Vec2(0.0, 0.0)];
            nmax=0, pec=true)
        @test_throws ArgumentError mie_scattered_field_2d(
            k0, a, eps_r, [Vec2(NaN, 2a)]; nmax=3)
        @test_throws ArgumentError mie_scattered_field_2d(
            k0, a, eps_r, observations; phi_inc=Inf, nmax=3)
        @test_throws ArgumentError mie_scattered_field_2d(
            1.0, 1.0, 1.0, [Vec2(2.0, 0.0)];
            nmax=1, pec=true, max_field_terms=2)
        mie_output_bytes = sizeof(ComplexF64) * length(observations)
        @test_throws ArgumentError mie_scattered_field_2d(
            k0, a, eps_r, observations;
            nmax=3, max_output_bytes=mie_output_bytes - 1)
        @test mie_scattered_field_2d(
            k0, a, eps_r, observations;
            nmax=3, max_output_bytes=mie_output_bytes) ==
              mie_scattered_field_2d(k0, a, eps_r, observations; nmax=3)
        @test_throws ArgumentError mie_total_field_2d(
            k0, a, eps_r, observations;
            nmax=3, max_output_bytes=mie_output_bytes - 1)
        @test mie_total_field_2d(
            k0, a, eps_r, observations;
            nmax=3, max_output_bytes=mie_output_bytes) ==
              mie_total_field_2d(k0, a, eps_r, observations; nmax=3)

        mie_coefficients_2d(k0, a, eps_r; nmax=10)
        mie_scattered_field_2d(k0, a, eps_r, observations; nmax=10)
        mie_total_field_2d(k0, a, eps_r, observations; nmax=10)
        coeff_alloc = @allocated mie_coefficients_2d(
            k0, a, eps_r; nmax=10)
        scattered_alloc = @allocated mie_scattered_field_2d(
            k0, a, eps_r, observations; nmax=10)
        total_alloc = @allocated mie_total_field_2d(
            k0, a, eps_r, observations; nmax=10)
        @test coeff_alloc <= 512
        @test scattered_alloc <= 640
        @test total_alloc <= scattered_alloc
        low_index_coefficients, _ = mie_coefficients_2d(
            1.0, 1.0, 0.16; nmax=10)
        @test all(isfinite, low_index_coefficients)
        @test @allocated(mie_coefficients_2d(
            1.0, 1.0, 0.16; nmax=10)) < 2_048
    end

    @testset "MoM vs Mie convergence" begin
        # Circular dielectric cylinder
        # Note: rectangular-grid VIE has non-monotonic convergence for curved
        # boundaries due to staircase approximation. We test overall accuracy.
        freq = 1e9; c0 = 3e8; lambda = c0 / freq; k0 = 2π / lambda
        a = 0.1 * lambda; eps_r = 4.0; chi_val = eps_r - 1.0

        r_obs = [Vec2(3a * cos(phi), 3a * sin(phi))
                 for phi in range(0, 2π, length=37)[1:36]]
        E_scat_mie = mie_scattered_field_2d(k0, a, eps_r, r_obs; phi_inc=0.0)

        errors = Float64[]
        for n in [10, 20, 40]
            mesh = Mesh2D((-a, a), (-a, a), n, n)
            chi = zeros(mesh.ncells)
            for i in 1:mesh.ncells
                r = sqrt(mesh.centers[i][1]^2 + mesh.centers[i][2]^2)
                if r <= a; chi[i] = chi_val; end
            end
            vr = solve_vie_2d(mesh, k0, chi, planewave_2d(mesh, k0, 0.0))
            E_scat_mom = scattered_field_2d(vr, r_obs)
            push!(errors, norm(E_scat_mom - E_scat_mie) / norm(E_scat_mie))
        end

        # Coarsest mesh should be reasonable (< 5%)
        @test errors[1] < 0.05

        # Finest mesh should be significantly better than coarsest
        @test errors[3] < errors[1]

        # Finest mesh should be within 1%
        @test errors[3] < 0.01
    end

    @testset "Jacobian accuracy" begin
        freq = 1e9; c0 = 3e8; lambda = c0 / freq; k0 = 2π / lambda
        a = 0.1 * lambda; eps_r = 2.5; chi_val = eps_r - 1.0

        mesh = Mesh2D((-a, a), (-a, a), 8, 8)
        chi = zeros(mesh.ncells)
        for i in 1:mesh.ncells
            r = sqrt(mesh.centers[i][1]^2 + mesh.centers[i][2]^2)
            if r <= a; chi[i] = chi_val; end
        end
        E_inc = planewave_2d(mesh, k0, 0.0)
        vr = solve_vie_2d(mesh, k0, chi, E_inc)

        r_obs = [Vec2(3a * cos(phi), 3a * sin(phi))
                 for phi in range(0, 2π, length=13)[1:12]]
        E_scat_ref = scattered_field_2d(vr, r_obs)
        J, _ = jacobian_scattered_field_2d(vr, r_obs)

        @test size(J) == (12, mesh.ncells)
        @test !any(isnan, J)
        @test !any(isinf, J)
        @test_throws ArgumentError green_obs_matrix(
            [Vec2(NaN, 0.0)], mesh, k0)
        @test_throws DomainError green_obs_matrix(
            [mesh.centers[1]], mesh, k0)
        @test_throws ArgumentError scattered_field_2d(
            vr, [Vec2(NaN, 0.0)])
        @test_throws DomainError scattered_field_2d(
            vr, [mesh.centers[1]])
        @test_throws ArgumentError jacobian_scattered_field_2d(
            vr, [Vec2(NaN, 0.0)])
        observation_matrix_bytes =
            sizeof(ComplexF64) * length(r_obs) * mesh.ncells
        @test_throws ArgumentError green_obs_matrix(
            r_obs, mesh, k0;
            max_output_bytes=observation_matrix_bytes - 1)
        @test green_obs_matrix(
            r_obs, mesh, k0;
            max_output_bytes=observation_matrix_bytes) ==
              green_obs_matrix(r_obs, mesh, k0)
        scattered_output_bytes = sizeof(ComplexF64) * length(r_obs)
        @test_throws ArgumentError scattered_field_2d(
            vr, r_obs; max_output_bytes=scattered_output_bytes - 1)
        @test scattered_field_2d(
            vr, r_obs; max_output_bytes=scattered_output_bytes) == E_scat_ref
        @test_throws ArgumentError jacobian_scattered_field_2d(
            vr, r_obs;
            max_work_bytes=3observation_matrix_bytes - 1)
        @test jacobian_scattered_field_2d(
            vr, r_obs;
            max_work_bytes=3observation_matrix_bytes)[1] == J

        # Scattered-field evaluation streams Green-function values and should
        # allocate only its returned vector. The Jacobian needs its returned
        # G_obs/J matrices, one N×M transposed-sensitivity workspace, and the
        # O(N) LAPACK reciprocal-condition work vectors.
        scattered_field_2d(vr, r_obs)
        jacobian_scattered_field_2d(vr, r_obs)
        scattered_alloc = @allocated scattered_field_2d(vr, r_obs)
        jacobian_alloc = @allocated jacobian_scattered_field_2d(vr, r_obs)
        vector_output_alloc = _complex_vector_allocation_2d(length(r_obs))
        rectangular_output_alloc =
            _complex_matrix_allocation_2d(length(r_obs), mesh.ncells)
        condition_workspace_alloc =
            _complex_vector_allocation_2d(2 * mesh.ncells) +
            _real_vector_allocation_2d(2 * mesh.ncells)
        @test scattered_alloc <= vector_output_alloc + 128
        @test jacobian_alloc <=
              3 * rectangular_output_alloc + condition_workspace_alloc + 512

        # Verify 5 random cells against finite differences
        delta = 1e-7
        cells_to_test = [findfirst(x -> x > 0, chi)]  # inside cell
        push!(cells_to_test, findfirst(x -> x == 0, chi))  # outside cell
        for _ in 1:3
            p = rand(1:mesh.ncells)
            p ∉ cells_to_test && push!(cells_to_test, p)
        end

        for p in cells_to_test
            chi_pert = copy(chi); chi_pert[p] += delta
            vr_pert = solve_vie_2d(mesh, k0, chi_pert, E_inc)
            E_scat_pert = scattered_field_2d(vr_pert, r_obs)
            J_fd = (E_scat_pert - E_scat_ref) / delta

            if norm(J[:, p]) > 1e-15
                rel_err = norm(J[:, p] - J_fd) / norm(J[:, p])
                @test rel_err < 1e-4  # FD accuracy limited by step size
            end
        end
    end

    @testset "Reciprocity check" begin
        # For a reciprocal medium: G(r1,r2) = G(r2,r1)
        # This means the D matrix should be symmetric
        k0 = 2π
        mesh = Mesh2D((-0.5, 0.5), (-0.5, 0.5), 6, 6)
        D = DiffMoM.assemble_D_matrix(mesh, k0)
        @test D ≈ transpose(D) atol=1e-13
    end

    @testset "Line source excitation" begin
        k0 = 2π
        mesh = Mesh2D((-0.5, 0.5), (-0.5, 0.5), 6, 6)
        r_src = Vec2(3.0, 0.0)

        E_inc = linesource_2d(mesh, k0, r_src)
        @test length(E_inc) == mesh.ncells
        @test !any(isnan, E_inc)
        @test !any(isinf, E_inc)
        @test_throws ArgumentError linesource_2d(mesh, Inf, r_src)
        @test_throws ArgumentError linesource_2d(
            mesh, k0, Vec2(NaN, 0.0))
        @test_throws DomainError linesource_2d(
            mesh, k0, mesh.centers[1])

        tiny_mesh = Mesh2D((-1.0, 1.0), (-1.0, 1.0), 1, 1)
        tiny_source = Vec2(1e-200, 0.0)
        tiny_incident = linesource_2d(tiny_mesh, 1.0, tiny_source)
        @test tiny_incident[1] ≈
              greens_2d(tiny_mesh.centers[1], tiny_source, 1.0) rtol=2e-15

        # Amplitude should decrease with distance from source
        # Find nearest and farthest cells
        dists = [norm(mesh.centers[i] - r_src) for i in 1:mesh.ncells]
        @test abs(E_inc[argmin(dists)]) > abs(E_inc[argmax(dists)])
    end

end
