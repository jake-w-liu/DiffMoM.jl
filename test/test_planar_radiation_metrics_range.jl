using DiffMoM, LinearAlgebra, Test

@testset "Radiation polarization and power are invariant to finite field scale" begin
    # Orthogonal quadrature components trace an ellipse with known axes.
    # A common phase and real rotation preserve those axes.
    phase=cis(.37);c,s=cos(.61),sin(.61)
    for scale in (1e-200,1.,1e155),ratio in (2.,1e10)
        et=scale*phase;ep=im*(scale/ratio)*phase
        for (x,y) in ((et,ep),(c*et-s*ep,s*et+c*ep))
            p=PlanarRadiationPattern([0.],[0.],fill(x,1,1),fill(y,1,1),zeros(1,1),nothing,1e9)
            metrics=planar_radiation_metrics(p)
            @test metrics.axial_ratio[1] ≈ ratio rtol=(ratio==2 ? 8eps(Float64) : 2e-6)
        end
    end
    p=PlanarRadiationPattern([0.],[0.],fill(1e308+0im,1,1),fill(-1e308im,1,1),fill(1e308,1,1),1e308,1e9)
    metrics=planar_radiation_metrics(p;radiated_power=1e308)
    @test metrics.circular_plus[1] ≈ sqrt(2)*1e308
    @test metrics.circular_minus[1] == 0
    @test metrics.gain[1] ≈ 4pi
    @test metrics.directivity[1] ≈ 4pi
    mktemp() do path,io
        close(io)
        write_planar_radiation_csv(path,p)
        @test parse(Float64,last(split(last(readlines(path)),','))) ≈ 4pi
    end
end

@testset "Radiated-power quadrature preserves finite sums and relative error" begin
    grid=CellGrid(.002,.002,4,4)
    stack=PlanarStackup([PlanarLayer(1.,1.,.001),PlanarLayer(1.,1.,.001)],
        TERM_GND,TERM_SPACE,grid.a,grid.b)
    sheet=sheet_level(1,4,4);rasterize_rect!(sheet,grid,0,grid.a,0,grid.b)
    prob=build_planar_problem(stack,grid,[sheet],[PlanarPort(1,:x,2,2:2,50.)])
    coefficients=ones(ComplexF64,planar_basis_count(prob.basis))
    options=(;ntheta=2,nphi=48,refine=false)
    base=planar_radiated_power(prob,coefficients,1e9;options...).power
    amplitude=sqrt(floatmax(Float64))*sqrt(.5)/sqrt(base)
    large=planar_radiated_power(prob,coefficients.*amplitude,1e9;options...)
    @test large.power ≈ .5floatmax(Float64) rtol=1e-13
    ordinary=planar_radiated_power(prob,coefficients,1e9;ntheta=2,nphi=4)
    small=planar_radiated_power(prob,coefficients.*1e-6,1e9;ntheta=2,nphi=4)
    @test small.power ≈ ordinary.power*1e-12 rtol=1e-13
    @test small.relative_change ≈ ordinary.relative_change rtol=1e-8 atol=1e-15

    # Caller-owned solutions satisfy the retained source equation exactly.
    # Their finite power must retain its sign under voltage scaling.
    nb=planar_basis_count(prob.basis);pbs=DiffMoM._port_basis_indices(prob.basis,1)
    signs=[DiffMoM._planar_port_sign(prob.ports[1])]
    rhs=DiffMoM._planar_dense_source_rhs!(zeros(ComplexF64,nb,1),prob)
    for polarity in (-1,1)
        Z=Matrix{ComplexF64}(I,nb,nb)
        for p in pbs[1]
            desired=polarity*1e-200/(length(pbs[1])*signs[1]*DiffMoM._planar_port_weight(prob.basis,p))
            Z[p,p]=rhs[p,1]/desired
        end
        F=lu(Z);X=F\rhs;Y=DiffMoM._planar_dense_port_y(prob,pbs,signs,X)
        result=PlanarResult(prob,complex(2pi*1e9),complex(1e9),Z,F,X,Y,planar_y_to_s(Y,[50.]))
        @test Z*X ≈ rhs rtol=eps(Float64)
        options=(;voltages=ComplexF64[1e200],theta=[.4],phi=[.2])
        if polarity<0
            @test_throws ArgumentError planar_farfield(result;options...)
        else
            pattern=planar_farfield(result;options...)
            @test pattern.accepted_power ≈ 5e199
            @test all(isfinite,pattern.intensity)
        end
        @test_throws ArgumentError planar_farfield(result;options...,max_bytes=1)
    end
end
