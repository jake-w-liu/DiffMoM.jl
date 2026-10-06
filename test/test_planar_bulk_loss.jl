using DiffMoM, Test, LinearAlgebra

@testset "bulk constitutive values must fit finite solver arithmetic" begin
    for label in ("via_sigma","volume_sigma"), value in
            (big"1e-500",big"1e500",1e-320)
        @test_throws ArgumentError DiffMoM._planar_bulk_resistivities(value,1,label)
    end
    @test DiffMoM._planar_bulk_resistivities(Inf,2,"via_sigma")==zeros(ComplexF64,2)
    @test_throws ArgumentError DiffMoM._planar_bulk_resistivities(Inf+im,1,"via_sigma")
    @test DiffMoM._planar_bulk_resistivities(1+2im,1,"via_sigma")==[.2-.4im]
    @test DiffMoM._planar_bulk_resistivities([1+2im,Inf],2,"volume_sigma")==[.2-.4im,0.]
    @test DiffMoM._planar_bulk_resistivities(-2im,1,"via_sigma")==[.5im]
    @test DiffMoM._planar_bulk_resistivities(2im,1,"volume_sigma")==[-.5im]
    for value in (0.,-1.,-1+2im,complex(big"0",big"1e-500"),complex(big"0",big"1e500"),1e-320im)
        @test_throws ArgumentError DiffMoM._planar_bulk_resistivities(value,1,"via_sigma")
    end
end

function _planar_reactive_bulk_fixture(kind)
    if kind===:via
        grid=CellGrid(2e-3,2e-3,2,2)
        stack=PlanarStackup([PlanarLayer(1.,1.,.5e-3)],TERM_GND,TERM_GND,grid.a,grid.b)
        via=via_level(1,2,2);via.uni.=true
        return build_planar_problem(stack,grid,SheetLevel[],[PlanarPort(1,:via,1:4,50.)];vias=[via])
    end
    grid=CellGrid(3e-3,2e-3,6,4);h=20e-6
    stack=PlanarStackup([PlanarLayer(1.,1.,.4e-3),PlanarLayer(1.,1.,h),
        PlanarLayer(1.,1.,.6e-3)],TERM_GND,TERM_GND,grid.a,grid.b)
    vol=vol_level(2,6,4);rasterize_rect!(vol,grid,0,grid.a,.5e-3,1.5e-3)
    vol.connect_west[2:3].=true;vol.connect_east[2:3].=true
    build_planar_problem(stack,grid,SheetLevel[],[PlanarPort(1,:volume_west,2:3,50.),
        PlanarPort(1,:volume_east,2:3,50.)];vols=[vol])
end

function _planar_reactive_dense_residual(result)
    G=DiffMoM;prob=result.problem;n=size(result.currents,1);np=length(prob.ports)
    weights=[G._planar_port_weight(prob.basis,b) for b in 1:n]
    rhs=zeros(ComplexF64,n,np)
    for b in 1:n
        p=prob.basis.port[b];p==0 && continue
        rhs[b,p]=-G._planar_port_sign(prob.ports[p])*weights[b]
    end
    residual=(result.z_mom*result.currents-rhs)./weights
    maximum(norm(view(residual,:,p))/norm(view(rhs,:,p)./weights) for p in 1:np)
end

@testset "purely reactive bulk impedance physical power and solver parity" begin
    G=DiffMoM
    for kind in (:via,:volume)
        prob=_planar_reactive_bulk_fixture(kind);sigma=kind===:via ? -1000im : -10000im
        material=s->kind===:via ? (;via_sigma=s) : (;volume_sigma=s)
        mx,my=kind===:via ? (12,12) : (24,20)
        impedance=r->kind===:via ? inv(r.y[1,1]) : 2/(r.y[1,1]-r.y[1,2])
        # Independent rho*h/A and rho*length/(width*h) geometric laws.
        # Opposite reactances separate the material term from the magnetic
        # background. At1MHz the volume's nonuniformity error is negligible.
        plus=solve_planar(prob,1e6;mx,my,material(sigma)...)
        minus=solve_planar(prob,1e6;mx,my,material(-sigma)...)
        expected=kind===:via ? .5e-3/(sigma*prob.grid.a*prob.grid.b) :
            prob.grid.a/(sigma*1e-3*20e-6)
        @test (impedance(plus)-impedance(minus))/2≈expected rtol=2e-8
        @test iszero(real(impedance(plus))) && iszero(real(impedance(minus)))
        for sign in (1,-1)
            kw=material(sign*sigma)
            dense=solve_planar(prob,1e9;mx,my,kw...)
            folded=solve_planar(prob,1e9;method=:dense_fft,mx,my,kw...)
            fft=solve_planar(prob,1e9;method=:ufft,mx,my,kw...,memory=100,rtol=1e-10)
            @test _planar_reactive_dense_residual(dense)<=1e-10
            @test _planar_reactive_dense_residual(folded)<=1e-10
            @test maximum(fft.relative_residuals)<=1e-10
            @test folded.y≈dense.y rtol=2e-12
            @test fft.y≈dense.y rtol=2e-9
            @test norm(adjoint(dense.s)*dense.s-I)<=1e-11
            @test norm(adjoint(fft.s)*fft.s-I)<=1e-9
            Z0=assemble_planar_z(prob.stack,prob.grid,prob.sheets,prob.basis,2pi*1e9;
                vias=prob.vias,vols=prob.vols,mx,my)
            reaction=dense.z_mom-Z0
            @test reaction≈transpose(reaction) rtol=1e-12
            @test norm(reaction+adjoint(reaction))<=1e-12*norm(reaction)
            volts=ComplexF64[1+.3im for _ in prob.ports];current=dense.y*volts
            @test abs(real(dot(volts,current)))<=1e-12*norm(volts)*norm(current)
            refs=[50+10p*im for p in eachindex(prob.ports)]
            S=planar_y_to_s(dense.y,refs)
            @test norm(adjoint(S)*S-I)<=1e-11
            waves=ComplexF64[.2+.1p*im for p in eachindex(prob.ports)]
            V=planar_wave_voltages(S,waves;z0=refs);Iport=dense.y*V
            power=(norm(waves)^2-norm(S*waves)^2)/2
            @test power≈real(dot(V,Iport))/2 atol=1e-13
            @test abs(power)<=1e-13
        end
        # Thickness derivatives include the purely reactive Gram term;
        # a sensitive imaginary-Y objective avoids norm² cancellation.
        layer=kind===:via ? 1 : 2;params=[PlanarParam(layer,:thickness,:re)]
        objective=Y->imag(Y[1,1]);gY=Y->begin a=zeros(ComplexF64,size(Y));a[1,1]=-im;a;end
        _,g=planar_objective_gradient(prob,1e9,objective;params,gY,mx,my,material(sigma)...)
        theta=planar_param_values(prob.stack,params);step=theta[1]*1e-4
        tp,tm=copy(theta),copy(theta);tp[1]+=step;tm[1]-=step
        pp=PlanarProblem(planar_with_params(prob.stack,params,tp),prob.grid,prob.sheets,prob.ports,prob.vias,prob.basis,prob.vols)
        pm=PlanarProblem(planar_with_params(prob.stack,params,tm),prob.grid,prob.sheets,prob.ports,prob.vias,prob.basis,prob.vols)
        fd=(objective(solve_planar(pp,1e9;mx,my,material(sigma)...).y)-
            objective(solve_planar(pm,1e9;mx,my,material(sigma)...).y))/(2step)
        @test g[1]≈fd rtol=3e-5 atol=1e-8
        _,gf=planar_objective_gradient(prob,1e9,objective;params,gY,mx,my,method=:ufft,
            memory=100,rtol=1e-10,material(sigma)...)
        @test gf≈g rtol=2e-8
    end
end

function _planar_invalid_bulk_operator(prob)
    try
        planar_ufft_operator(prob,7e9;mx=128,my=128,via_sigma=0.)
    catch error
        return error
    end
end

@testset "planar: bulk conductor dissipation" begin
    grid = CellGrid(2e-3, 2e-3, 2, 2)
    stack = PlanarStackup([PlanarLayer(4.0, 1.0, 5e-6),
        PlanarLayer(2.0, 1.0, 0.5e-3)], TERM_GND, TERM_SPACE, grid.a, grid.b)
    sheet = sheet_level(1, 2, 2)
    sheet.mask .= true
    sheet.connect_west .= true
    sheet.connect_east .= true
    vol = vol_level(1, 2, 2)
    vol.mask .= true
    via = via_level(2, 2, 2)
    via.uni[1, 1] = true
    via.tap[1, 1] = true
    prob = build_planar_problem(stack, grid, [sheet],
        [PlanarPort(1, :west, 1:2, 50.0),
         PlanarPort(1, :east, 1:2, 50.0)]; vols=[vol], vias=[via])
    sigma = 5.8e7
    Z0 = assemble_planar_z(stack, grid, [sheet], prob.basis, 2pi * 7e9;
        vias=[via], vols=[vol])
    Z1 = assemble_planar_z(stack, grid, [sheet], prob.basis, 2pi * 7e9;
        vias=[via], vols=[vol], via_sigma=sigma, volume_sigma=[sigma])
    loss = Z1 - Z0
    @test loss ≈ transpose(loss) atol=1e-15
    @test all(iszero, imag.(loss))
    @test maximum(eigvals(Hermitian(real.(loss)))) <= 1e-14
    nb = planar_basis_count(prob.basis)
    c = zeros(ComplexF64, nb)
    for p in 1:nb
        k = prob.basis.kind[p]
        if k == DiffMoM._BASIS_VIA_U
            c[p] = 2
        elseif k == DiffMoM._BASIS_VIA_T
            c[p] = 3
        end
    end
    # Integral of the physical via profile |2+3*z/h|² is 13*h.
    expected_via = 13 * grid.dx * grid.dy * 0.5e-3 / sigma
    @test -real(dot(c, loss * c)) ≈ expected_via rtol=1e-5
    fill!(c, 0)
    for p in 1:nb
        k = prob.basis.kind[p]
        if k == DiffMoM._BASIS_VX_FULL
            c[p] = 2 + 3im
        elseif k == DiffMoM._BASIS_VY_FULL
            c[p] = 4 - 1im
        end
    end
    # Each full triangle has integral(T² dA)=2*dx*dy/3; four
    # independent x/y rooftops give no orthogonal cross dissipation.
    expected_volume = (2abs2(2+3im) + 2abs2(4-1im)) *
        (2grid.dx * grid.dy / 3) / (sigma * 5e-6)
    @test -real(dot(c, loss * c)) ≈ expected_volume rtol=1e-9
    @test_throws ArgumentError assemble_planar_z(stack, grid, [sheet],
        prob.basis, 2pi*7e9; vias=[via], vols=[vol], via_sigma=0.0)
    @test_throws DimensionMismatch assemble_planar_z(stack, grid, [sheet],
        prob.basis, 2pi*7e9; vias=[via], vols=[vol], volume_sigma=[sigma,sigma])
    @test_throws ArgumentError assemble_planar_z(stack, grid, [sheet],
        prob.basis,2pi*7e9;vias=[via],vols=[vol],via_sigma=big"1e-500")
    @test_throws ArgumentError planar_ufft_operator(prob,7e9;via_sigma=big"1e500")
    @test _planar_invalid_bulk_operator(prob) isa ArgumentError
    # The old late material validation allocated 8.5 MB of modal kernels
    # before rejecting this input. Warm rejection now needs metadata only.
    _planar_invalid_bulk_operator(prob)
    @test (@allocated _planar_invalid_bulk_operator(prob)) < 200_000
    # Complex local constitutive impedance is already supported through
    # complex conductivity; it keeps positive real dissipation and matches
    # the exact FFT reaction without changing source/terminal conventions.
    complex_sigma=1e3/(1+2im)
    dense_complex=solve_planar(prob,7e9;mx=12,my=12,
        via_sigma=complex_sigma,volume_sigma=complex_sigma)
    fft_complex=solve_planar(prob,7e9;method=:ufft,mx=12,my=12,
        via_sigma=complex_sigma,volume_sigma=complex_sigma,memory=50,rtol=1e-10)
    @test opnorm(dense_complex.s)<=1+1e-10
    @test fft_complex.y≈dense_complex.y rtol=2e-9
    @test fft_complex.s≈dense_complex.s rtol=2e-9
    @test maximum(fft_complex.relative_residuals)<=1e-10
    # Thickness affects both modal physics and normalized ohmic profiles.
    params = [PlanarParam(1, :thickness, :re), PlanarParam(2, :thickness, :re)]
    objective = Y -> real(Y[1, 1] + Y[2, 2])
    _, gradient = planar_objective_gradient(prob, 7e9, objective;
        params=params, gY=Y -> Matrix{ComplexF64}(I, 2, 2),
        via_sigma=1e3, volume_sigma=1e3, mx=12, my=12)
    theta = planar_param_values(stack, params)
    for j in eachindex(params)
        step = theta[j] * 1e-4
        tp, tm = copy(theta), copy(theta)
        tp[j] += step
        tm[j] -= step
        pp = PlanarProblem(planar_with_params(stack, params, tp), grid,
            prob.sheets, prob.ports, prob.vias, prob.basis, prob.vols)
        pm = PlanarProblem(planar_with_params(stack, params, tm), grid,
            prob.sheets, prob.ports, prob.vias, prob.basis, prob.vols)
        yp = solve_planar(pp, 7e9; via_sigma=1e3, volume_sigma=1e3, mx=12, my=12).y
        ym = solve_planar(pm, 7e9; via_sigma=1e3, volume_sigma=1e3, mx=12, my=12).y
        fd = (objective(yp) - objective(ym)) / (2step)
        @test gradient[j] ≈ fd rtol=2e-4 atol=1e-7
    end
end

@testset "bulk conductivity preserves nonzero stored components" begin
    setprecision(BigFloat,256) do
        tiny,small=big"1e-400",big"1e-300"
        rational_tiny=big(1)//big(10)^400
        rational_small=big(1)//big(10)^300
        invalid=(complex(tiny,small),complex(small,tiny),complex(small,-tiny),
            complex(rational_tiny,rational_small),complex(rational_small,rational_tiny),
            complex(rational_small,-rational_tiny))
        for label in ("via_sigma","volume_sigma"),value in invalid,vector in (false,true)
            input=vector ? [value] : value
            error=try
                DiffMoM._planar_bulk_resistivities(input,1,label)
                nothing
            catch caught
                caught
            end
            @test error isa ArgumentError
            @test occursin("$label[1]",sprint(showerror,error))
        end
        for kind in (:via,:volume)
            prob=_planar_reactive_bulk_fixture(kind)
            material(s)=kind===:via ? (;via_sigma=s) : (;volume_sigma=s)
            for method in (:dense,:dense_fft,:ufft),value in invalid[1:3]
                @test_throws ArgumentError solve_planar(prob,1e9;method,
                    mx=12,my=12,material(value)...)
            end
        end
    end
end
