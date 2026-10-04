using DiffMoM, Test, LinearAlgebra, StaticArrays

@testset "planar library: geometry and independent references" begin
    area(s) = abs(DiffMoM._p2_signed_area(s.polygons[1].vertices))
    w, l = 1e-4, 1e-3
    line = planar_line(length=l, width=w, level=1, metal="pec")
    @test area(line) ≈ w*l rtol=1e-13
    @test planar_pin(line,"p1").point == SVector(0.,0.)
    @test planar_pin(line,"p2").point == SVector(l,0.)
    taper = planar_taper(length=l, width1=w, width2=2w, level=1, metal="pec")
    @test area(taper) ≈ 1.5w*l rtol=1e-13
    bend = planar_bend(width=w, arm=l, miter=.5, level=1, metal="pec")
    @test area(bend) ≈ 2w*l-w^2/2 rtol=1e-13
    curved = planar_curved_bend(width=w,radius=l,segments=64,level=1,metal="pec")
    @test area(curved) ≈ w*l*pi/2 rtol=1e-4
    tee = planar_tee(width_main=w,width_branch=w/2,length_main=l,
        length_branch=l/2,level=1,metal="pec")
    @test area(tee) ≈ 1.25w*l rtol=1e-13
    cross = planar_cross(width_x=w,width_y=w/2,arm_x=l/2,
        arm_y=l/2,level=1,metal="pec")
    @test area(cross) ≈ 1.5w*l-w^2/2 rtol=1e-13
    step = planar_step(width1=w,width2=2w,length1=l,length2=l,level=1,metal="pec")
    @test area(step) ≈ 3w*l rtol=1e-13
    radial = planar_radial_stub(radius=l,angle=pi/3,feed_width=w,
        segments=64,level=1,metal="pec")
    @test area(radial) ≈ radial.meta["ideal_area"] rtol=1e-4
    for shape in (:rectangular,:hexagonal,:octagonal,:circular)
        spiral = planar_spiral(shape=shape,turns=2.,width=10e-6,
            spacing=5e-6,d_out=160e-6,segments_per_turn=96,level=1,metal="pec")
        @test area(spiral) > 0
        @test length(spiral.pins) == 2
        coeff = shape == :rectangular ? (1.27,2.07,.18,.13) :
            shape == :hexagonal ? (1.09,2.23,0.,.17) :
            shape == :octagonal ? (1.07,2.29,0.,.19) : (1.,2.46,0.,.20)
        outer, inner = 160e-6, spiral.meta["d_in"]
        avg, rho = (outer+inner)/2, (outer-inner)/(outer+inner)
        c1,c2,c3,c4 = coeff
        expected = (4pi*1e-7)*4avg*c1/2*(log(c2/rho)+c3*rho+c4*rho^2)
        @test planar_spiral_inductance(shape=shape,turns=2.,d_out=outer,
            d_in=inner,method=:current_sheet) ≈ expected rtol=1e-13
    end
    attached = planar_attach(line,"p1",planar_pin(line,"p2");name="second")
    @test planar_pin(attached,"p1").point ≈ planar_pin(line,"p2").point
    @test planar_pin(attached,"p1").direction ≈ -planar_pin(line,"p2").direction
    mirrored = planar_transform(line; angle=pi/2, offset=(1e-3,2e-3),mirror=true)
    @test planar_pin(mirrored,"p2").point ≈ SVector(1e-3,3e-3)
    @test planar_layout_dict(line; unit="um")["polygons"][1]["vertices"][2][1] == 1000.
    @test_throws ArgumentError planar_layout_dict(line; unit="parsec")
    @test_throws ArgumentError planar_normalize_polygon(Tuple{Float64,Float64}[])
    @test_throws ArgumentError planar_normalize_polygon([(0.,0.),(1.,0.),(NaN,1.)])
    @test_throws ArgumentError planar_normalize_polygon([(0.,0.),(1.,1.),(1.,0.),(0.,1.)])
    # Area remains stable under translation; the shoelace sum is relative
    # to the first vertex, rather than subtracting large global products.
    pts, _ = planar_normalize_polygon([(1e3,1e3),(1e3+.01,1e3),
        (1e3+.01,1e3+.01),(1e3,1e3+.01)])
    @test DiffMoM._p2_signed_area(pts) ≈ 1e-4 rtol=1e-9

    stack = PlanarStackup([PlanarLayer(2.,1.,1e-6;epsr_z=4.),
        PlanarLayer(3.,1.,2e-6)],TERM_GND,TERM_GND,1e-3,1e-3)
    @test planar_parallel_plate_capacitance(stack,2,0,1e-8) ≈
        DiffMoM._EPS0*1e-8/(1e-6/4+2e-6/3)
    cap = planar_mim_capacitor(width=w,length=l,upper_level=2,
        lower_level=1,upper_metal="pec",overhang=w/10,
        lead_width=w/2,lead_length=w)
    @test [p.level for p in cap.polygons] == [2,1]
    broad = planar_broadside_coupled_lines(length=l,width=w,
        upper_level=2,lower_level=1,metal="pec")
    @test [p.level for p in broad.polygons] == [2,1]
    vias = planar_via_array(nx=2,ny=2,pitch_x=w,pitch_y=w,via_size=w/2,
        from_level=2,to_level=0,via_type="uniform",pad_metal="pec")
    @test length(vias.vias) == 4
    bridge = planar_air_bridge(span=l,width=w,landing=w,bridge_level=2,
        base_level=1,via_type="uniform",metal="pec")
    @test length(bridge.vias) == 2
    idc = planar_interdigital_capacitor(fingers=3,finger_length=l,width=w,
        gap=w/2,level=1,metal="pec",net1="left",net2="right")
    @test length(idc.polygons) == 8
    @test idc.meta["height"] ≈ 6w+5w/2
    @test length(filter(p -> p.net == "left",idc.polygons)) == 4
    @test planar_pin(idc,"p2").point[1] ≈ 2w+l+w/2
end

function _interior_library_layout(stack=nothing)
    grid=CellGrid(4e-3,3e-3,8,6)
    stack===nothing && (stack=PlanarStackup([PlanarLayer(1.,1.,.2e-3),
        PlanarLayer(2.,1.,.4e-3),PlanarLayer(1.,1.,1e-3)],
        TERM_GND,TERM_GND,grid.a,grid.b))
    shape=planar_transform(planar_line(length=2e-3,width=1e-3,
        level=2,metal="resistive",net="wire");offset=(1e-3,1.5e-3))
    return build_planar_layout(stack,grid,[shape],
        [planar_pin(shape,"p2"),planar_pin(shape,"p1")];metals=Dict("resistive"=>.1))
end

@testset "planar library: physical interior pin workflow" begin
    layout=_interior_library_layout()
    @test layout.contraction!==nothing
    @test [p.wall for p in layout.problem.ports]==[:terminal_x_hi,:terminal_x_lo]
    @test all(p -> p.layers==1:2,layout.terminal_paths)
    @test all(p -> p.direction==:below,layout.terminal_paths)
    @test isempty(planar_connectivity(layout).grounded_components)
    @test length(planar_connectivity(layout).port_components)==2
    result=solve_planar(layout,1e9;mx=20,my=18)
    @test result isa PlanarContractedResult
    raw=solve_planar(layout.source_problem,1e9;mx=20,my=18,surface_zs=.1)
    C=layout.contraction
    # The unscaled independently driven slice matrix has condition 1.5e11;
    # direct voltage-normalized physical sources avoid its cancellation.
    @test result.y ≈ transpose(C)*raw.y*C rtol=1e-11
    @test opnorm(result.s)<=1+1e-10
    fft=solve_planar(layout,1e9;method=:ufft,mx=20,my=18,rtol=1e-10,memory=100)
    @test fft.s ≈ result.s rtol=1e-8
    maps=planar_current_maps(result;incident_waves=ComplexF64[1,0])
    @test count(m -> m.kind==:via,maps)==4
    @test all(m -> all(isfinite,m.jx) && all(isfinite,m.jz),maps)
    params=[PlanarParam(1,:epsr,:re),PlanarParam(1,:thickness,:re),PlanarParam(2,:thickness,:re)]
    # A common-mode capacitive objective resolves epsr sensitivity without
    # subtracting nearly identical O(0.35) norm-squared objective values.
    # The near-insensitive norm objective has a separate BigFloat oracle.
    objective(Y)=imag(sum(Y));gY(Y)=fill(-1im,size(Y))
    J,gradient=planar_objective_gradient(layout,1e9,objective;params,gY,mx=20,my=18)
    @test J ≈ objective(result.y) rtol=1e-12
    values=planar_param_values(layout.problem.stack,params)
    for k in eachindex(params)
        h=1e-3abs(values[k]);vp=copy(values);vm=copy(values);vp[k]+=h;vm[k]-=h
        lp=_interior_library_layout(planar_with_params(layout.problem.stack,params,vp))
        lm=_interior_library_layout(planar_with_params(layout.problem.stack,params,vm))
        fd=(objective(solve_planar(lp,1e9;mx=20,my=18).y)-objective(solve_planar(lm,1e9;mx=20,my=18).y))/(2h)
        @test gradient[k] ≈ fd rtol=3e-5
    end
    Jfd,gfd=planar_objective_gradient(layout,1e9,objective;params,mx=20,my=18)
    @test Jfd ≈ J rtol=1e-12
    @test gfd ≈ gradient rtol=1e-5
    sweep=planar_sweep_abs(layout,.9e9,1.1e9;mx=20,my=18,n_eval=9,max_points=4)
    @test all(S -> opnorm(S)<=1+1e-8,sweep.dense_s)
    @test_throws ArgumentError solve_planar(layout,1e9;max_bytes=1)
    calls=Ref(0)
    @test_throws ArgumentError planar_objective_gradient(layout,1e9,Y->(calls[]+=1;sum(abs2,Y));params,max_bytes=1)
    @test calls[]==0
end

function _library_layout(; metal="pec", metals=Dict("pec"=>0.))
    grid = CellGrid(4e-3,3e-3,8,6)
    stack = PlanarStackup([PlanarLayer(2.,1.,.4e-3),PlanarLayer(1.,1.,.4e-3)],
        TERM_GND,TERM_GND,grid.a,grid.b)
    shape = planar_transform(planar_line(length=grid.a,width=1e-3,
        level=1,metal=metal,net="signal");offset=(0.,grid.b/2))
    return build_planar_layout(stack,grid,[shape],
        [planar_pin(shape,"p1"), (planar_pin(shape,"p2"),75.)]; metals)
end

@testset "planar library: physical layout solve and connectivity" begin
    layout = _library_layout()
    prob = planar_layout_problem(layout)
    @test size(prob.sheets[1].mask) == (8,6)
    @test prob.ports[2].z0 == 75.
    result = solve_planar(layout,5e9;mx=20,my=18)
    @test result.s ≈ solve_planar(prob,5e9;mx=20,my=18).s rtol=1e-12
    @test planar_sparams(layout,[5e9];mx=20,my=18)[1] ≈ result.s rtol=1e-12
    net = planar_connectivity(layout)
    @test net.component_count == 1
    @test net.port_components == [[1],[1]]
    @test isempty(net.grounded_components)
    @test isempty(net.open_nets) && isempty(net.shorted_nets)
    narrow = sheet_level(1,8,6)
    narrow.mask[:,3] .= true
    narrow.connect_west[3] = narrow.connect_east[3] = true
    narrowprob = build_planar_problem(prob.stack,prob.grid,[narrow],
        [PlanarPort(1,:west,3:3,50.),PlanarPort(1,:east,3:3,50.)])
    @test planar_connectivity(narrowprob).port_components == [[1],[1]]
    @test_throws ArgumentError planar_connectivity(layout;max_bytes=1)
    @test_throws ArgumentError _library_layout(metal="undefined")
    @test_throws ArgumentError build_planar_layout(prob.stack,prob.grid,
        layout.shapes,prob.ports;max_bytes=1)

    # An intentional open and then an intentional short, declared by nets.
    function changed_shapes(right_start, right_net)
        left = planar_transform(planar_line(length=1.5e-3,width=1e-3,
            level=1,metal="pec",name="left",net="wire");offset=(0.,1.5e-3))
        right = planar_transform(planar_line(length=prob.grid.a-right_start,
            width=1e-3,level=1,metal="pec",name="right",net=right_net);
            offset=(right_start,1.5e-3))
        return build_planar_layout(prob.stack,prob.grid,[left,right],
            [planar_pin(left,"p1"),planar_pin(right,"p2")])
    end
    open = planar_connectivity(changed_shapes(2.5e-3,"wire"))
    @test open.component_count == 2
    @test open.open_nets == ["wire"]
    short = planar_connectivity(changed_shapes(1.5e-3,"other"))
    @test short.component_count == 1
    @test short.shorted_nets == [["other","wire"]]

    # Named lossy metals retain per-frequency values and pass through the
    # same analytic cellwise Gram term on both solver paths.
    calls = Ref(0)
    loss(f) = (calls[] += 1; .1+.02im*(f/5e9))
    lossy = _library_layout(metal="copper",metals=Dict("copper"=>loss))
    rloss = solve_planar(lossy,5e9;mx=20,my=18)
    @test calls[] == 1
    @test rloss.s ≈ solve_planar(lossy.problem,5e9;mx=20,my=18,surface_zs=.1+.02im).s rtol=1e-12
    @test maximum(svdvals(rloss.s)) <= 1+1e-10
    rfft = solve_planar(lossy,5e9;mx=20,my=18,method=:ufft,memory=100,rtol=1e-10)
    @test rfft.s ≈ rloss.s rtol=1e-8
    J, grad = planar_objective_gradient(lossy,5e9,Y -> sum(abs2,Y);
        params=[PlanarParam(1,:epsr,:re)],gY=Y -> 2conj.(Y),mx=20,my=18)
    Jraw, gradraw = planar_objective_gradient(lossy.problem,5e9,Y -> sum(abs2,Y);
        params=[PlanarParam(1,:epsr,:re)],gY=Y -> 2conj.(Y),mx=20,my=18,
        surface_zs=.1+.02im)
    @test J ≈ Jraw rtol=1e-12
    @test grad ≈ gradraw rtol=1e-12
    sweep = planar_sweep_abs(lossy,4.9e9,5.1e9;mx=20,my=18,n_eval=9,max_points=4)
    @test all(isfinite,reduce(vcat,vec.(sweep.dense_s)))
end
