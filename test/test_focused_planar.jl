module FocusedPlanarTests
using DiffMoM, Test, LinearAlgebra

@testset "Layered Green function agrees with two terminated transmission lines" begin
    frequency = 8e9
    omega = 2pi * frequency
    mu0 = 4pi * 1e-7
    eps0 = 8.854187817e-12
    layers = [PlanarLayer(2.2 - .02im, 1., .0004),
              PlanarLayer(1.5 - .01im, 1., .0007)]
    stack = PlanarStackup(layers, TERM_GND, TERM_GND, .008, .006)
    kc2 = (pi / stack.a)^2 + (pi / stack.b)^2
    for pol in (TE_POL, TM_POL)
        impedances = map(layers) do layer
            gamma = sqrt(complex(kc2 - (omega / 299792458.)^2 * layer.epsr))
            characteristic = pol == TE_POL ? im * omega * mu0 / gamma :
                gamma / (im * omega * eps0 * layer.epsr)
            characteristic * tanh(gamma * layer.thickness)
        end
        expected = inv(inv(impedances[1]) + inv(impedances[2]))
        cascade = planar_mode_cascade(stack, omega, kc2, pol)
        @test planar_modal_voltage(cascade, 1, 1) ≈ expected rtol=1e-12
        @test planar_modal_voltage(cascade, 0, 1) == 0
        @test planar_modal_voltage(cascade, 2, 1) == 0
    end
    @test_throws ArgumentError planar_mode_cascade(stack, 0., kc2, TE_POL)
    @test_throws ArgumentError planar_mode_cascade(stack, omega, -1., TE_POL)

    # The homogeneous TE cutoff has a finite series-inductance limit.
    cutoff = (omega / 299792458.)^2
    h = .0005
    vacuum = PlanarStackup(fill(PlanarLayer(1., 1., h), 2),
        TERM_GND, TERM_GND, stack.a, stack.b)
    @test planar_modal_voltage(planar_mode_cascade(vacuum, omega, cutoff, TE_POL),
        1, 1) ≈ im * omega * mu0 * h / 2 rtol=1e-12
    sigma = 5.8e7
    @test planar_surface_zs(frequency, sigma) ≈
        (1 + im) * sqrt(pi * frequency * mu0 / sigma) rtol=1e-12
end

function stripline(walls)
    grid = CellGrid(.004, .003, 3, 2; walls)
    stack = PlanarStackup([PlanarLayer(2.2 - .02im, 1., .0004),
        PlanarLayer(1., 1., .0006)], TERM_GND, TERM_GND, grid.a, grid.b)
    sheet = sheet_level(1, grid.nx, grid.ny)
    sheet.mask .= true
    sheet.connect_west .= true; sheet.connect_east .= true
    sheet.connect_south .= true; sheet.connect_north .= true
    build_planar_problem(stack, grid, [sheet],
        [PlanarPort(1, :west, 1:2, 50.), PlanarPort(1, :east, 1:2, 75.)])
end

@testset "Small dense, FFT assembly and UFFT solves preserve the same equations" begin
    for walls in (WALL_PEC, WALL_PMC)
        problem = stripline(walls)
        # Above the grid Nyquist counts, so this also exercises mode folding.
        options = (; mx=7, my=6, surface_zs=.2 + .05im)
        dense = solve_planar(problem, 8e9; options...)
        fft = solve_planar(problem, 8e9; method=:dense_fft, options...)
        iterative = solve_planar_ufft(problem, 8e9;
            options..., rtol=1e-10, memory=40)
        @test fft.z_mom ≈ dense.z_mom rtol=2e-12
        @test dense.z_mom ≈ transpose(dense.z_mom) rtol=2e-12
        for result in (fft, iterative)
            @test result.y ≈ dense.y rtol=1e-8
            @test result.s ≈ dense.s rtol=1e-8
            @test result.currents ≈ dense.currents rtol=1e-8
        end
        @test maximum(iterative.relative_residuals) < 1e-10
        @test opnorm(dense.s) <= 1 + 1e-10

        operator = iterative.operator
        x = ComplexF64[sin(j) + im*cos(j) for j in 1:size(operator, 1)]
        expected = dense.z_mom * x
        @test operator * x ≈ expected rtol=2e-12
        @test DiffMoM._planar_ufft_diagonal(operator) ≈ diag(dense.z_mom) rtol=2e-12
        aliased = copy(x)
        mul!(aliased, operator, aliased, .7 + .2im, -.3)
        @test aliased ≈ (.7 + .2im) .* expected .- .3 .* x rtol=2e-12
        out = similar(x)
        mul!(out, operator, x)
        @test (@allocated mul!(out, operator, x)) == 0
        @test_throws ArgumentError solve_planar(problem, 8e9; options..., max_bytes=1)
        @test_throws DimensionMismatch mul!(zeros(ComplexF64, length(x)+1), operator, x)
    end
end

@testset "Two conformal triangles preserve area and dense/FFT reactions" begin
    a, b = .004, .003
    mesh = PlanarConformalMesh([0. a 0. a; b/3 b/3 2b/3 2b/3],
        [1 2; 2 4; 3 3])
    @test sum(mesh.areas) ≈ a * b / 3 rtol=2e-15
    stack = PlanarStackup([PlanarLayer(2., 1., .0003),
        PlanarLayer(1., 1., .0007)], TERM_GND, TERM_GND, a, b)
    problem = PlanarConformalProblem(stack, mesh,
        [PlanarConformalPort(1, :west, (b/3, 2b/3)),
         PlanarConformalPort(1, :east, (b/3, 2b/3))])
    options = (; mx=5, my=7, surface_zs=.1)
    dense = solve_planar_conformal(problem, 2e9; options...)
    fast = solve_planar_conformal(problem, 2e9; options...,
        method=:ufft, nx=1, ny=3, rtol=1e-10, memory=10)
    @test fast.y ≈ dense.y rtol=1e-9
    @test fast.currents ≈ dense.currents rtol=1e-9
    @test maximum(fast.relative_residuals) < 1e-10
    @test opnorm(dense.s) <= 1 + 1e-10
    @test fast.y ≈ transpose(fast.y) rtol=1e-10
end

@testset "Network conversion and a parallel RLC retain analytic port responses" begin
    admittance = ComplexF64[.02+.01im -.003im; -.003im .03+.02im]
    references = [50., 75.]
    d, r = Diagonal(references), Diagonal(sqrt.(references))
    expected = r \ ((I - d * admittance) / (I + d * admittance)) * r
    scattering = planar_y_to_s(admittance, references)
    @test scattering ≈ expected rtol=2e-14
    @test opnorm(scattering) <= 1 + 1e-12
    mktempdir() do directory
        path = joinpath(directory, "small.s2p")
        planar_write_touchstone(path, [1e9, 2e9], [scattering, scattering];
            z0=references)
        loaded = planar_read_touchstone(path)
        @test loaded.frequencies == [1e9, 2e9]
        @test loaded.z0 == references
        @test all(s -> isapprox(s, scattering; atol=2e-14), loaded.s)
    end
    circuit = PlanarCircuit(1, [1])
    circuit_add_rlc!(circuit, 1, 0; r=100., l=1e-9, c=.1e-12,
        topology=:parallel)
    omega = 2pi * 3e9
    y = 1/100 + inv(im * omega * 1e-9) + im * omega * .1e-12
    result = solve_planar_circuit(circuit, 3e9)
    @test result.y[1, 1] ≈ y rtol=1e-13
    @test result.s[1, 1] ≈ (1 - 50y) / (1 + 50y) rtol=1e-13
    @test solve_planar_circuit(circuit, 0.).s[1, 1] ≈ -1
    @test_throws ArgumentError solve_planar_circuit(circuit, 3e9; max_bytes=1)
end
end
