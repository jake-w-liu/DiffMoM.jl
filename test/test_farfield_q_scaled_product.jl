using Test, DiffMoM, LinearAlgebra, SparseArrays

struct _QScaleFiniteOnly <: AbstractMatrix{ComplexF64} end
Base.size(::_QScaleFiniteOnly) = (2,2)
LinearAlgebra.mul!(result,::_QScaleFiniteOnly,input) = copyto!(result,input)

function _scaled_q_mul_allocation(result, operator, input)
    mul!(result, operator, input, .5+.25im, .1-.2im)
    return @allocated mul!(result, operator, input, .5+.25im, .1-.2im)
end

@testset "Generic Q children retain finite-product Number scaling" begin
    input,initial = ones(ComplexF64,2),ComplexF64[.4,.5]
    for child in (Diagonal(ComplexF64[1,2]),Hermitian(ComplexF64[1 0;0 2]),
                  _QScaleFiniteOnly())
        operator = SumQMatrix(child,zeros(ComplexF64,2,2))
        product = child isa _QScaleFiniteOnly ? input : child*input
        for (alpha,beta) in ((BigFloat("0.1"),BigFloat("0.3")),(1//3,2//7))
            expected = setprecision(BigFloat,16000) do
                ComplexF64.(alpha .* Complex{BigFloat}.(product) .+
                            beta .* Complex{BigFloat}.(initial))
            end
            result = copy(initial)
            mul!(result,operator,input,alpha,beta)
            @test result ≈ expected rtol=4eps(Float64)
        end
    end
end

@testset "Q constructors retain default field conversions" begin
    shared = ComplexF32[1,0,0]
    radiation = ComplexF32[1 2 3;0 0 0;0 0 0]
    converted = FarFieldQMatrix(radiation,Float32[1],reshape(shared,3,1),
        Bool[true],Int32(3),shared,zeros(ComplexF32,3),ReentrantLock(),0)
    @test converted.N === 3 && converted.mask == trues(1)
    @test converted.G_mat isa Matrix{ComplexF64}
    @test converted.weights isa Vector{Float64}
    @test converted.work isa Vector{ComplexF64}
    @test !Base.mightalias(converted.work,converted.pol)
    @test converted*ones(ComplexF64,3) == ComplexF64[6,12,18]
    @test shared == ComplexF32[1,0,0]

    shared_matrix = ones(ComplexF32,1,1)
    converted_sum = SumQMatrix{Matrix{ComplexF64},Matrix{ComplexF64}}(
        shared_matrix,zeros(ComplexF32,1,1),vec(shared_matrix),
        zeros(ComplexF32,1),ReentrantLock())
    @test converted_sum.A isa Matrix{ComplexF64}
    @test converted_sum.work_a isa Vector{ComplexF64}
    @test !Base.mightalias(converted_sum.A,converted_sum.work_a)
    @test converted_sum*ComplexF64[2] == ComplexF64[2]
    @test shared_matrix == ones(ComplexF32,1,1)
end

@testset "Q workspace graph stays disjoint from borrowed coefficients" begin
    pol = reshape(ComplexF64[1,0,0],3,1)
    shared = ComplexF64[1,0,0]
    radiation = ComplexF64[1 2 3;0 0 0;0 0 0]
    @test_throws ArgumentError FarFieldQMatrix(
        radiation,[1.],reshape(shared,3,1),nothing,3,shared,ReentrantLock())
    for make_matrix in (
            work -> reshape(work,1,1),
            work -> view(reshape(work,1,1),[1],[1]),
            work -> Diagonal(work),
            work -> SparseMatrixCSC(1,1,[1,2],[1],work),
            work -> Hermitian(reshape(work,1,1)))
        work = ComplexF64[1]
        matrix = make_matrix(work)
        @test_throws ArgumentError SumQMatrix(
            matrix,zeros(ComplexF64,1,1),work,zeros(ComplexF64,1),ReentrantLock())
        @test work == ComplexF64[1]
    end
    first = FarFieldQMatrix(radiation,[1.],pol,nothing,3,shared,ReentrantLock())
    second = FarFieldQMatrix(radiation,[1.],reshape(shared,3,1),nothing,3)
    @test_throws ArgumentError SumQMatrix(first,second) # sibling write/read alias
    child = FarFieldQMatrix(reshape(ComplexF64[1,0,0],3,1),[1.],pol,nothing,1)
    @test_throws ArgumentError SumQMatrix(
        child,zeros(ComplexF64,1,1),child.work,zeros(ComplexF64,1),ReentrantLock())
    @test SumQMatrix(child,child)*ComplexF64[1] == ComplexF64[2]
end

@testset "Far-field Q scales before exceptional product conversion" begin
    grid = make_sph_grid(1, 1)
    polarization = reshape(ComplexF64[1, 0, 0], 3, 1)
    for (amplitude, input, alpha) in ((2.5e153, 4., .1), (2.0^-539, .25, 4.))
        radiation = reshape(ComplexF64[amplitude, 0, 0], 3, 1)
        operator = build_Q_operator(radiation, grid, polarization)
        expected = setprecision(BigFloat, 16000) do
            ComplexF64(BigFloat(alpha)*BigFloat(grid.w[1])*
                       BigFloat(amplitude)^2*BigFloat(input))
        end
        result = zeros(ComplexF64, 1)
        mul!(result, operator, ComplexF64[input], alpha, 0.)
        @test only(result) == expected
    end

    # Both factorized-work branches (N <= NΩ and N > NΩ), complex scales,
    # aliased input/output, and mutation of borrowed radiation storage.
    for count in (1, 2)
        radiation = ones(ComplexF64, 3, count)
        operator = build_Q_operator(radiation, grid, polarization)
        input = ComplexF64[.2+.3im for _ in 1:count]
        initial = ComplexF64[-.4+.1im for _ in 1:count]
        alpha, beta = .25+.5im, -.4+.2im
        expected = alpha*(Matrix(operator)*input) + beta*initial
        result = copy(initial)
        mul!(result, operator, input, alpha, beta)
        @test result ≈ expected rtol=4eps(Float64)
        result = copy(input)
        expected_alias = alpha*(Matrix(operator)*input) + beta*input
        mul!(result, operator, result, alpha, beta)
        @test result ≈ expected_alias rtol=4eps(Float64)

        radiation[1, :] .= 2.5e153
        expected = setprecision(BigFloat, 16000) do
            weight = BigFloat(grid.w[1])*BigFloat(2.5e153)^2
            term = Complex{BigFloat}(alpha)*weight*sum(Complex{BigFloat}.(input))
            ComplexF64[term + Complex{BigFloat}(beta)*value for value in initial]
        end
        result = copy(initial)
        mul!(result, operator, input, alpha, beta)
        @test result == expected
    end

    operator = FarFieldQMatrix(reshape(ComplexF64[1, 0, 0], 3, 1),
                              [1.], polarization, nothing, 1)
    alpha, beta, rounding_beta = setprecision(BigFloat, 24000) do
        power = BigFloat(2)^20000
        power+1, -power, 1+BigFloat(2)^-53+BigFloat(2)^-20000
    end
    result = ones(ComplexF64, 1)
    mul!(result, operator, result, alpha, beta)
    @test only(result) == 1
    mul!(result, operator, ComplexF64[NaN], 0, rounding_beta)
    @test only(result) == nextfloat(1.)
    mul!(result, operator, ComplexF64[NaN], 0, 0)
    @test only(result) == 0

    sum_operator = SumQMatrix(ones(ComplexF64,1,1), zeros(ComplexF64,1,1))
    result .= 1
    mul!(result, sum_operator, result, alpha, beta)
    @test only(result) == 1
    mul!(result, sum_operator, ComplexF64[NaN], 0, rounding_beta)
    @test only(result) == nextfloat(1.)
    huge_child = build_Q_operator(
        reshape(ComplexF64[2.5e154,0,0],3,1),grid,polarization)
    nested = SumQMatrix(SumQMatrix(huge_child,zeros(ComplexF64,1,1)),
                        zeros(ComplexF64,1,1))
    expected = setprecision(BigFloat,16000) do
        Float64(BigFloat(.001)*BigFloat(grid.w[1])*BigFloat(2.5e154)^2)
    end
    mul!(result,nested,ComplexF64[1],.001,0.)
    @test only(result) == expected
    dense_sum = SumQMatrix(ComplexF64[1e16 1;1 1e16],zeros(ComplexF64,2,2))
    result2 = fill(ComplexF64(-1e16),2)
    mul!(result2,dense_sum,ones(ComplexF64,2),1,1)
    @test result2 == ones(ComplexF64,2)
    diagonal_sum = SumQMatrix(Diagonal(ComplexF64[1,2]),zeros(ComplexF64,2,2))
    for scale in (BigFloat(1), BigFloat(.25))
        mul!(result2,diagonal_sum,ones(ComplexF64,2),scale,0)
        @test result2 == ComplexF64[scale,2scale]
    end

    work = ComplexF64[1]
    supplied = FarFieldQMatrix(reshape(ComplexF64[1,0,0],3,1),
        [1.],polarization,nothing,1,work,ReentrantLock())
    @test_throws ArgumentError mul!(zeros(ComplexF64,1),supplied,work)
    @test_throws ArgumentError mul!(work,supplied,ComplexF64[1],2,4)
    supplied_sum = SumQMatrix(SumQMatrix(supplied,zeros(ComplexF64,1,1)),
                             zeros(ComplexF64,1,1))
    @test_throws ArgumentError mul!(work,supplied_sum,ComplexF64[1],2,4)
    @test_throws ArgumentError mul!(zeros(ComplexF64,1),supplied_sum,work)
    @test work == ComplexF64[1]

    for alpha in (2., Inf)
        result = ComplexF64[.3+.7im]
        before = copy(result)
        @test_throws OverflowError mul!(result, operator, ComplexF64[floatmax(Float64)], alpha, 0)
        @test result == before
        mul!(result, operator, ComplexF64[1], 1, 0)
        @test only(result) == 1 # the failed operation released its lock
    end
    normal_grid = make_sph_grid(3, 4)
    radiation = ComplexF64[complex(sin(i+j), cos(i-j))
        for i in 1:3length(normal_grid.w), j in 1:5]
    normal = build_Q_operator(radiation, normal_grid, pol_linear_x(normal_grid))
    input = ComplexF64[sin(j)+im*cos(j) for j in 1:5]
    @test _scaled_q_mul_allocation(zeros(ComplexF64, 5), normal, input) == 0
end
