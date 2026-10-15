using Test, LinearAlgebra, DiffMoM

@testset "coherent source weights precede field and RHS conversion" begin
    frequency = 299792458.0 / (2pi)
    origin, axis, point = Vec3(.3, .7, -10), Vec3(0, 0, 1), Vec3(1, 0, 0)
    dipole(kind, amplitude) = make_dipole(origin, CVec3(0, 0, amplitude), axis, kind, frequency)
    pattern(amplitude) = make_pattern_feed([0., pi], [0., pi],
        fill(complex(amplitude, amplitude), 2, 2), zeros(ComplexF64, 2, 2),
        frequency; phase_center=Vec3(pi/4, 0, 0))
    builders = (a -> dipole(:magnetic, a), a -> dipole(:electric, a),
        a -> make_loop(origin, axis, 1., a, frequency),
        a -> make_monopole(origin, axis, pi, a, frequency), pattern)
    weights = (2.0^-10, 2.0^-40, 2.0^-12, 2.0^-8, 2.0^-10)
    amplitudes = (1e308, 1e308, 1e308, 1e308, 1.7e308)
    mesh = make_rect_plate(.1, .1, 1, 1)
    rwg = build_rwg(mesh)
    evaluators = (source -> incident_farfield(source, point, 1.),
        source -> DiffMoM._incident_electric_field(source, point, 1.),
        source -> assemble_excitation(mesh, rwg, source; quad_order=1))
    for (build, weight, amplitude) in zip(builders, weights, amplitudes)
        large, reference = build(amplitude), build(amplitude * weight)
        combined = make_multi_excitation([large], [complex(weight, weight/2)])
        nested = make_multi_excitation(
            [make_multi_excitation([large], [2. + 0im])], [weight/2 + 0im])
        cancelled = make_multi_excitation([large, large, reference], [1., -1., 1.])
        zeroed = make_multi_excitation([large], [0. + 0im])
        for evaluate in evaluators
            expected = evaluate(reference)
            @test evaluate(combined) ≈ (1 + .5im) * expected rtol=1e-12
            @test evaluate(nested) ≈ expected rtol=1e-12
            @test evaluate(cancelled) ≈ expected rtol=1e-12
            @test all(iszero, evaluate(zeroed))
        end
    end

    # A one-edge voltage source also applies its coherent weight before
    # dividing by a small physical gap; duplicate port edges retain assignment semantics.
    gap = make_delta_gap(1, 1e308, .001)
    @test assemble_excitation(mesh, rwg,
        make_multi_excitation([gap], [2.0^-20])) ≈ [1e308 * 2.0^-20 / .001]
    port = PortExcitation([1, 1], 1e308, 50.)
    @test assemble_excitation(mesh, rwg,
        make_multi_excitation([port], [2.0^-20])) ≈ [1e308 * 2.0^-20 / rwg.len[1]]
    @test_throws ArgumentError assemble_excitation(mesh, rwg,
        make_multi_excitation([gap], [2.0^-20]); max_exact_bytes=1)

    density = ImportedExcitation(_ -> CVec3(1e308, 0, 0);
        kind=:surface_current_density, eta_equiv=1e3, min_quad_order=1)
    density_reference = ImportedExcitation(_ -> CVec3(1e308, 0, 0);
        kind=:surface_current_density, eta_equiv=.001, min_quad_order=1)
    @test assemble_excitation(mesh,rwg,make_multi_excitation([density],[1e-6]);quad_order=1) ≈
          assemble_excitation(mesh,rwg,density_reference;quad_order=1) rtol=1e-12
    large_mesh=make_rect_plate(1e150,1e150,1,1);large_rwg=build_rwg(large_mesh)
    field=ImportedExcitation(_ -> CVec3(1e308,0,0);min_quad_order=1)
    unitfield=ImportedExcitation(_ -> CVec3(1,0,0);min_quad_order=1)
    @test assemble_excitation(large_mesh,large_rwg,make_multi_excitation([field],[1e-308]);quad_order=1) ≈
          assemble_excitation(large_mesh,large_rwg,unitfield;quad_order=1) rtol=1e-12

    # Zero weights preserve validation and callback failures.
    calls = Ref(0)
    invalid = ImportedExcitation(_ -> (calls[] += 1; error("source failed")); kind=:electric_field)
    @test_throws ErrorException DiffMoM._incident_electric_field(
        make_multi_excitation([invalid], [0.]), point, 1.)
    @test calls[] == 1
    calls[]=0
    invalid_rhs=ImportedExcitation(_ -> (calls[]+=1;throw(OverflowError("callback failed")));min_quad_order=1)
    @test_throws OverflowError assemble_excitation(mesh,rwg,make_multi_excitation([invalid_rhs],[0.]);quad_order=1)
    @test calls[] == 1
    @test_throws ErrorException incident_farfield(make_multi_excitation([gap], [0.]), point, 1.)
    @test_throws OverflowError incident_farfield(
        make_multi_excitation([dipole(:magnetic, 1e308)], [1.]), point, 1.)

    ordinary = dipole(:magnetic, 1.)
    incident_farfield(ordinary, point, 1.)
    DiffMoM._incident_electric_field(ordinary, point, 1.)
    @test (@allocated incident_farfield(ordinary, point, 1.)) == 0
    @test (@allocated DiffMoM._incident_electric_field(ordinary, point, 1.)) == 0

    # The loop moment can underflow while k^2 restores its field. Interleave
    # that finite field between equal overflowing loops to exercise exact
    # accumulator growth rather than a fixed Float64-product precision.
    extreme_k=2.0^900;extreme_f=extreme_k*(299792458.0/(2pi))
    small=make_loop(Vec3(0,0,0),axis,2.0^-900,2.0^-900,extreme_f)
    large=make_loop(Vec3(0,0,0),axis,2.0^900,2.0^900,extreme_f)
    expected=setprecision(BigFloat,10000) do
        Float64(BigFloat(DiffMoM._ETA0)/4*BigFloat(extreme_k)^2*BigFloat(2.0^-900)^3)
    end
    @test incident_farfield(small,point,extreme_k) == CVec3(0,expected,0)
    @test incident_farfield(make_multi_excitation([large,small,large],[1.,1.,-1.]),point,extreme_k) == CVec3(0,expected,0)
    near_small=make_loop(origin,axis,2.0^-900,2.0^-900,extreme_f)
    near_large=make_loop(origin,axis,2.0^900,2.0^900,extreme_f)
    rescued=make_multi_excitation([near_large,near_small,near_large],[1.,1.,-1.])
    @test DiffMoM._incident_electric_field(rescued,point,extreme_k) ≈
          DiffMoM._incident_electric_field(near_small,point,extreme_k) rtol=1e-12
    @test assemble_excitation(mesh,rwg,rescued;quad_order=1,max_exact_bytes=20_000) ≈
          assemble_excitation(mesh,rwg,near_small;quad_order=1) rtol=1e-12
    @test_throws ArgumentError assemble_excitation(mesh,rwg,rescued;quad_order=1,
        max_exact_bytes=DiffMoM._multi_exact_accumulator_bytes(rwg.nedges))
end
