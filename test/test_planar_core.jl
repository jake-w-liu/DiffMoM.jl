using Test, LinearAlgebra, DiffMoM

@testset "planar core: cutoff limits and derivatives" begin
    G = DiffMoM
    w = 2pi * 10e9
    h = 1e-3
    kc2 = Float64(real(G.planar_k2_layer(w, 1.0, 1.0)))
    lay = PlanarLayer(1.0, 1.0, h)
    stack = PlanarStackup([lay, lay], TERM_GND, TERM_GND, 0.01, 0.01)
    cte = planar_mode_cascade(stack, w, kc2, TE_POL)
    @test planar_modal_voltage(cte, 1, 1) ≈ 1im * w * G._MU0 * h / 2
    @test all(isfinite, cte.zdn)
    @test_throws ArgumentError planar_mode_cascade(stack, 0.0, kc2, TE_POL)
    @test_throws ArgumentError planar_mode_cascade(stack, w, -1.0, TE_POL)
    @test_throws ArgumentError planar_mode_cascade(stack, w, Inf, TE_POL)
    @test planar_modal_voltage(cte, 0, 0) == 0
    ctm = planar_mode_cascade(stack, w, kc2, TM_POL)
    @test planar_modal_voltage(ctm, 1, 1) == 0
    pmc = PlanarTerminator(TERM_PMC)
    stm = PlanarStackup([lay, lay], pmc, pmc, 0.01, 0.01)
    ctm_open = planar_mode_cascade(stm, w, kc2, TM_POL)
    @test planar_modal_voltage(ctm_open, 1, 1) ≈ inv(1im * w * G._EPS0 * 2h)
    for (st, pol) in ((stack, TE_POL), (stm, TM_POL))
        exact = planar_modal_voltage(planar_mode_cascade(st, w, kc2, pol), 1, 1)
        for dw in (-1e-8, 1e-8)
            nearby = planar_modal_voltage(planar_mode_cascade(st, w * (1 + dw), kc2, pol), 1, 1)
            @test nearby ≈ exact rtol=1e-7
        end
    end
    # Volume currents see the finite layer impedance at cutoff, including
    # a nonzero TE self response between two PEC faces.
    volstack = PlanarStackup([lay], TERM_GND, TERM_GND, 0.01, 0.01)
    vte = G._vol_layer_state(volstack,
        planar_mode_cascade(volstack, w, kc2, TE_POL), w, kc2, 1, TE_POL)
    @test vte.vt == vte.vb == 0
    @test vte.mself ≈ 1im * w * G._MU0 * h / 12
    volopen = PlanarStackup([lay], pmc, pmc, 0.01, 0.01)
    vtm = G._vol_layer_state(volopen,
        planar_mode_cascade(volopen, w, kc2, TM_POL), w, kc2, 1, TM_POL)
    @test vtm.vt ≈ inv(1im * w * G._EPS0 * h)
    @test vtm.vt == vtm.vb == vtm.mself
    # Constructors must preserve derivative-carrying thickness values.
    @test PlanarLayer(2.0, 1.0, h + 1e-20im).thickness == h + 1e-20im
    dlay = PlanarLayer(2.0, 1.0, G._PlanarDual(h + 0im, 1.0 + 0im))
    @test dlay.thickness.d == 1
    # The cascade remains analytic in layer parameters at exact cutoff.
    for pol in (TE_POL, TM_POL)
        param = PlanarParam(1, :epsr, :re)
        dualst = G._planar_dual_stackup(stack, param)
        cv = planar_modal_voltage(planar_mode_cascade(dualst, w, kc2, pol), 1, 1)
        @test isfinite(cv.v) && isfinite(cv.d)
        delta = 1e-5
        plus = planar_with_params(stack, [param], [1.0 + delta])
        minus = planar_with_params(stack, [param], [1.0 - delta])
        fd = (planar_modal_voltage(planar_mode_cascade(plus, w, kc2, pol), 1, 1) -
            planar_modal_voltage(planar_mode_cascade(minus, w, kc2, pol), 1, 1)) / (2delta)
        @test cv.d ≈ fd rtol=1e-7 atol=1e-10
    end
end

@testset "planar core: geometry and complex validation" begin
    G = DiffMoM
    grid = CellGrid(1.0, 1.0, 2, 2)
    s = sheet_level(1, 2, 2)
    s.mask .= true
    s.connect_west .= true
    @test_throws ArgumentError build_planar_basis(grid, [s],
        [PlanarPort(2, :west, 1:2, 50.0)])
    @test_throws ArgumentError build_planar_basis(grid, [s],
        [PlanarPort(1, :west, 1:0, 50.0)])
    s.connect_west[2] = false
    @test_throws ArgumentError build_planar_basis(grid, [s],
        [PlanarPort(1, :west, 1:2, 50.0)])
    bad = SheetLevel(1, falses(2, 2), falses(1), falses(2), falses(2), falses(2))
    @test_throws DimensionMismatch build_planar_basis(grid, [bad], PlanarPort[])
    @test_throws DimensionMismatch rasterize_rect!(bad, grid, 0.0, 1.0, 0.0, 1.0)
    @test_throws ArgumentError rasterize_poly!(s, grid, [0.0, 1.0, NaN], [0.0, 0.0, 1.0])
    for field in (:epsr, :mur, :epsr_z, :mur_z, :thickness)
        vals = Dict(:epsr => 1.0 + 0im, :mur => 1.0 + 0im,
            :epsr_z => 1.0 + 0im, :mur_z => 1.0 + 0im, :thickness => 1.0 + 0im)
        vals[field] = complex(1.0, Inf)
        lay = PlanarLayer(vals[:epsr], vals[:mur], vals[:thickness],
            vals[:epsr_z], vals[:mur_z])
        @test_throws ArgumentError planar_validate(PlanarStackup([lay],
            TERM_GND, TERM_GND, 1.0, 1.0))
    end
    # The former fixed id split at layer 8192 aliased vias with volumes.
    for layer in (1, 8192, 8193, 100000)
        ev = G._vol_elem(layer)
        @test G._is_vol_elem(ev) && !G._is_via_elem(ev)
        @test G._vol_elem_layer(ev) == layer
        for kind in (G._BASIS_VIA_U, G._BASIS_VIA_T)
            e = G._via_elem(layer, kind)
            @test G._is_via_elem(e) && !G._is_vol_elem(e)
            @test G._via_elem_layer(e) == layer
            @test G._via_elem_kind(e) == kind
        end
    end
end

@testset "planar core: assembly preflight and workspace reuse" begin
    grid = CellGrid(0.01, 0.01, 2, 2)
    s = sheet_level(1, 2, 2)
    s.mask .= true
    basis = build_planar_basis(grid, [s], PlanarPort[])
    stack = PlanarStackup([PlanarLayer(1.0, 1.0, 1e-3),
        PlanarLayer(2.0, 1.0, 1e-3)], TERM_GND, TERM_GND, 0.01, 0.01)
    rejected(mx) = try
        assemble_planar_z(stack, grid, [s], basis, 1e9; mx, my=1, max_bytes=1)
    catch err
        err
    end
    @test rejected(typemax(Int)) isa ArgumentError
    rejected(10)
    rejected(100000)
    @test @allocated(rejected(100000)) < 100000
    a(block) = assemble_planar_z(stack, grid, [s], basis, 1e9; mx=48, my=48, block)
    z1, z512 = a(1), a(512)
    @test z1 ≈ z512 rtol=1e-12
    # Workspace cost must depend on the chosen block, rather than one
    # cascade allocation per block repeated across all 2304 modes.
    @test @allocated(a(1)) < 100000
end

# Independent Maxwell first-order BVP using RK4 and a 2x2 boundary solve.
# This reference neither constructs characteristic impedances nor uses
# the via-drive moments.  The source is the unit normalized line forcing.
function _core_via_hmoment_reference(stack,omega,kc2,jb,kb,ja,ka; n=800)
    G = DiffMoM
    function integrate(v0,h0,forced)
        V,H = v0,h0
        moment = 0.0im
        for (l,layer) in enumerate(stack.layers)
            height = real(layer.thickness)
            step = height/n
            Y = 1im*omega*layer.epsr*G._EPS0
            g2 = G._planar_gamma2_layer(TM_POL,kc2,omega,layer)
            function derivative(z,v,h)
                w = forced && l==jb ? (kb==G._BASIS_VIA_U ? 1.0 : z/height) : 0.0
                return -g2*h/Y+w,-Y*v
            end
            for t in 1:n
                z = (t-1)*step
                a,b = derivative(z,V,H)
                av,bv = derivative(z+step/2,V+step*a/2,H+step*b/2)
                cv,dv = derivative(z+step/2,V+step*av/2,H+step*bv/2)
                ev,fv = derivative(z+step,V+step*cv,H+step*dv)
                if l==ja
                    weight = ka==G._BASIS_VIA_U ? 1.0 : (z+step/2)/height
                    moment += weight*(H+step*(b+bv)/4)*step
                end
                V += step*(a+2av+2cv+ev)/6
                H += step*(b+2bv+2dv+fv)/6
            end
        end
        return V,H,moment
    end
    one_v = integrate(1.0+0im,0.0im,false)
    one_h = integrate(0.0im,1.0+0im,false)
    forced = integrate(0.0im,0.0im,true)
    zd,zu = stack.bottom.zs,stack.top.zs
    # These references use PEC or finite surface-impedance covers.
    M = ComplexF64[1 zd;
        one_v[1]-zu*one_v[2] one_h[1]-zu*one_h[2]]
    initial = M\ComplexF64[0,-forced[1]+zu*forced[2]]
    return initial[1]*one_v[3]+initial[2]*one_h[3]+forced[3]
end

@testset "planar core: loaded via axial cutoff" begin
    G=DiffMoM
    omega=2pi*10e9
    kc2=real(G.planar_k2_layer(omega,1.0,1.0))
    stack=PlanarStackup([PlanarLayer(2.0,1.0,.4e-3),
        PlanarLayer(1.0,1.0,.2e-3),PlanarLayer(3.0,1.0,.3e-3)],
        TERM_GND,TERM_GND,.01,.01)
    casc=planar_mode_cascade(stack,omega,kc2,TM_POL)
    states=[G._via_layer_state(stack,casc,omega,kc2,j) for j in 1:3]
    @test iszero(states[2].coef) # finite H representation at exact cutoff
    n3=stack.a*stack.b/4
    for ja in 1:3,jb in 1:3,ka in (G._BASIS_VIA_U,G._BASIS_VIA_T),
            kb in (G._BASIS_VIA_U,G._BASIS_VIA_T)
        Hmoment=_core_via_hmoment_reference(stack,omega,kc2,jb,kb,ja,ka)
        overlap=ja==jb ? G._via_overlap(ka,kb,stack.layers[jb].thickness) : 0.0im
        reference=-1im*(states[ja].amp_h*Hmoment+overlap)/(states[jb].om_eps_z*n3)
        actual=G._via_via_kern(ja,ka,jb,kb,casc,states[ja],states[jb],n3)
        @test actual ≈ reference rtol=2e-6 atol=1e-12
        swapped=G._via_via_kern(jb,kb,ja,ka,casc,states[jb],states[ja],n3)
        @test actual ≈ swapped rtol=1e-10 atol=1e-12
    end
    # Convergence through cutoff has no spurious jump, including field
    # cutoff/source noncutoff and the reciprocal pair.
    for delta in (-1e-8,1e-8)
        c=planar_mode_cascade(stack,omega*(1+delta),kc2,TM_POL)
        st=[G._via_layer_state(stack,c,omega*(1+delta),kc2,j) for j in 1:3]
        for j in 1:3,kind in (G._BASIS_VIA_U,G._BASIS_VIA_T)
            @test G._via_via_kern(2,G._BASIS_VIA_U,j,kind,c,st[2],st[j],n3) ≈
                G._via_via_kern(2,G._BASIS_VIA_U,j,kind,casc,states[2],states[j],n3) rtol=1e-6
        end
    end
    # The all-air PEC-PEC stack has a real TM_mn0 cavity pole at this
    # frequency.  Preserve a loud resonance error instead of fabricating
    # finite reactions for that distinct case.
    resonance=PlanarStackup([PlanarLayer(1.0,1.0,.2e-3)],
        TERM_GND,TERM_GND,.01,.01)
    @test_throws DomainError G._via_layer_state(resonance,
        planar_mode_cascade(resonance,omega,kc2,TM_POL),omega,kc2,1)
    # Gradient across the now regular loaded cutoff must agree with a
    # separate central perturbation of the whole port problem.
    grid=CellGrid(stack.a,stack.b,4,4)
    sh1,sh2=sheet_level(2,4,4),sheet_level(1,4,4)
    sh1.mask.=true; sh2.mask.=true
    sh1.connect_west.=true; sh2.connect_east.=true
    via=via_level(2,4,4)
    via.uni[2,2]=via.tap[2,2]=true
    prob=build_planar_problem(stack,grid,[sh1,sh2],
        [PlanarPort(1,:west,1:4,50.),PlanarPort(2,:east,1:4,50.)];vias=[via])
    freq=G._C0/2*sqrt((1/stack.a)^2+(1/stack.b)^2)
    params=[PlanarParam(2,:epsr,:re),PlanarParam(2,:thickness,:re)]
    objective=Y->abs2(planar_y_to_s(Y,[50.,50.])[2,1])
    values=planar_param_values(stack,params)
    J,gradient=planar_objective_gradient(prob,freq,objective;params,
        surface_zs=.1,mx=8,my=8)
    @test isfinite(J) && all(isfinite,gradient)
    for p in eachindex(params)
        delta=1e-6*abs(values[p])
        plus,minus=copy(values),copy(values)
        plus[p]+=delta; minus[p]-=delta
        ps=PlanarProblem(planar_with_params(stack,params,plus),grid,prob.sheets,
            prob.ports,prob.vias,prob.basis)
        ms=PlanarProblem(planar_with_params(stack,params,minus),grid,prob.sheets,
            prob.ports,prob.vias,prob.basis)
        fd=(objective(solve_planar(ps,freq;surface_zs=.1,mx=8,my=8).y)-
            objective(solve_planar(ms,freq;surface_zs=.1,mx=8,my=8).y))/(2delta)
        @test gradient[p] ≈ fd rtol=1e-4 atol=1e-9
    end
end

@testset "planar core: spatial conductor impedance" begin
    G=DiffMoM
    grid=CellGrid(1.,1.,2,2)
    s=sheet_level(1,2,2)
    s.mask.=true
    s.connect_west.=true; s.connect_east.=true
    s.connect_south.=true; s.connect_north.=true
    basis=build_planar_basis(grid,[s],PlanarPort[])
    nb=planar_basis_count(basis)
    spatial=ComplexF64[1+im 2+im;3+2im 4+3im]
    Z=zeros(ComplexF64,nb,nb)
    G._add_gram!(Z,basis,grid,[spatial])
    # Independent midpoint vector-dot integral with the actual cell map.
    for p in 1:nb,q in 1:nb
        want=0.0im
        if G._is_xdir(basis.kind[p])==G._is_xdir(basis.kind[q])
            for j in 1:2,i in 1:2
                # On a shared cell each rooftop is affine in its flow
                # coordinate, so two midpoint samples plus a correction
                # can be avoided with a 2-point Gauss rule, exact here.
                for gy in (-inv(sqrt(3.)),inv(sqrt(3.))),
                        gx in (-inv(sqrt(3.)),inv(sqrt(3.)))
                    x=(i-.5)*grid.dx+gx*grid.dx/2
                    y=(j-.5)*grid.dy+gy*grid.dy/2
                    # Use the independent scalar-shape helper defined in
                    # test_planar.jl when the full suite is running.
                    function shape(p,x,y)
                        k=basis.kind[p];xd=G._is_xdir(k)
                        t=xd ? x/grid.dx : y/grid.dy
                        edge=xd ? basis.ei[p] : basis.ej[p]
                        col=xd ? basis.ej[p] : basis.ei[p]
                        trans=xd ? y/grid.dy : x/grid.dx
                        col-1<=trans<col || return 0.
                        if k in (G._BASIS_X_LO,G._BASIS_Y_LO)
                            return 0<=t<1 ? 1-t : 0.
                        elseif k in (G._BASIS_X_HI,G._BASIS_Y_HI)
                            return 1<=t<2 ? t-1 : 0.
                        end
                        return max(1-abs(t-edge),0.)
                    end
                    want-=spatial[i,j]*shape(p,x,y)*shape(q,x,y)*grid.dx*grid.dy/4
                end
            end
        end
        @test Z[p,q] ≈ want atol=1e-15 rtol=1e-13
    end
    @test minimum(eigvals(Hermitian(-real.(Z)))) > 0
    uniform=zeros(ComplexF64,nb,nb);mapped=similar(uniform);fill!(mapped,0)
    G._add_gram!(uniform,basis,grid,2+.4im)
    G._add_gram!(mapped,basis,grid,[fill(2+.4im,2,2)])
    @test uniform==mapped
    @test_throws DimensionMismatch G._add_gram!(copy(Z),basis,grid,[zeros(2,3)])
    @test_throws ArgumentError G._add_gram!(copy(Z),basis,grid,[fill(Inf,2,2)])
end
