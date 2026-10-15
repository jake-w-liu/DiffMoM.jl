using Test, LinearAlgebra, DiffMoM

@testset "conformal FFT rejects owned output aliases before mutation" begin
    a, b = .004, .003
    mesh = PlanarConformalMesh([0. a 0. a; b/3 b/3 2b/3 2b/3],
        [1 2; 2 4; 3 3])
    stack = PlanarStackup([PlanarLayer(2., 1., .0003),
        PlanarLayer(1., 1., .0007)], TERM_GND, TERM_GND, a, b)
    problem = PlanarConformalProblem(stack, mesh,
        [PlanarConformalPort(1, :west, (b/3, 2b/3)),
         PlanarConformalPort(1, :east, (b/3, 2b/3))])
    operator = planar_conformal_ufft_operator(problem, 2e9;
        nx=1, ny=3, mx=5, my=7)
    x = ComplexF64[1 + .2im, 2 - .3im, -1 + .7im]
    product = operator * x
    workspaces = (operator.output, operator.lattice, operator.source_te,
        operator.source_tm, operator.field_te, operator.field_tm)
    originals = map(copy, workspaces)
    for workspace in workspaces
        owned_view = view(vec(workspace), 1:operator.n)
        @test_throws ArgumentError mul!(owned_view, operator, x, 1., 1.)
        @test_throws ArgumentError mul!(zeros(ComplexF64, operator.n),
            operator, owned_view)
        @test all(isequal(workspace, original)
            for (workspace, original) in zip(workspaces, originals))
    end

    # Caller-owned source/output overlap is still supported, including
    # beta's dependence on the original output values.
    storage = vcat(x, 3. - .5im)
    source, output = view(storage, 1:operator.n), view(storage, 2:operator.n+1)
    old_output = copy(output)
    mul!(output, operator, source, .7 + .2im, -.3)
    @test output ≈ (.7 + .2im) .* product .- .3 .* old_output rtol=1e-13
    y = similar(x)
    mul!(y, operator, x)
    @test (@allocated mul!(y, operator, x)) == 0
    @test y ≈ product rtol=1e-13
end
