using DiffMoM,Test,LinearAlgebra

function _floating_standard_fixture(;rotated=false,coupled=false)
    nx=12;ny=coupled ? 12 : 8
    grid=rotated ? CellGrid(ny*.15e-3,nx*.2e-3,ny,nx) : CellGrid(nx*.2e-3,ny*.15e-3,nx,ny)
    stack=PlanarStackup([PlanarLayer(2.,1.,.4e-3),PlanarLayer(2.,1.,.4e-3)],
        TERM_GND,TERM_GND,grid.a,grid.b)
    sheet=sheet_level(1,grid.nx,grid.ny);ports=PlanarPort[];bridges=Any[]
    rows=coupled ? ((2,5),(8,11)) : ((2,6),)
    axis=rotated ? :x : :y
    for (ground,signal) in rows
        for along in 2:11,t in (ground,signal)
            i,j=rotated ? (t,along) : (along,t);sheet.mask[i,j]=true
        end
        for along in (2,11)
            cells=Tuple{Int,Int}[]
            for t in ground+1:signal-1
                i,j=rotated ? (t,along) : (along,t)
                sheet.mask[i,j]=true;push!(cells,(i,j))
            end
            cut=ground+(signal-ground)÷2
            p=PlanarPort(1,axis,cut,along:along,50.)
            sig=PlanarPort(1,rotated ? :terminal_x : :terminal_y,signal-1,along:along,50.;metal_side=:positive)
            ref=PlanarPort(1,rotated ? :terminal_x : :terminal_y,ground,along:along,50.;metal_side=:negative)
            push!(ports,p)
            push!(bridges,(port=length(ports),cells=cells,signal=sig,reference=ref,axis=axis,
                sheet=1,interface=1,source_edge=cut,span=along:along,
                width=.2e-3,length=(signal-ground-1)*.15e-3,polarity=1))
        end
    end
    return build_planar_problem(stack,grid,[sheet],ports),bridges
end

@testset "floating local-ground sister geometry retains physical leads" begin
    prob,bridges=_floating_standard_fixture()
    original=copy(prob.sheets[1].mask)
    st=planar_floating_line_standards(prob,bridges;left=[1],right=[2])
    @test st.ordering==[1,2]
    @test st.uniform_cells==3:10
    @test st.double_uniform_cells==3:18
    @test st.length≈8prob.grid.dx
    @test st.double_line.grid.nx==20
    @test st.double_line.grid.dx≈prob.grid.dx
    @test prob.sheets[1].mask==original==st.line.sheets[1].mask
    @test st.double_line.ports[2].cells==19:19
    @test st.double_line_bridges[2].span==19:19
    @test st.double_line_bridges[2].signal.cells==19:19
    @test Set(st.double_line_bridges[2].cells)==Set([(19,t) for t in 3:5])
    @test all(st.reflect.sheets[1].mask[2:11,2])
    @test !any(st.reflect.sheets[1].mask[3:10,6])
    @test all(st.reflect.sheets[1].mask[2,2:6])
    @test all(st.reflect.sheets[1].mask[11,2:6])
    @test isempty(st.line.vias) && isempty(st.double_line.vias)
    @test all(!any(x) for x in (st.double_line.sheets[1].connect_west,
        st.double_line.sheets[1].connect_east,st.double_line.sheets[1].connect_south,st.double_line.sheets[1].connect_north))
    rotated,rb=_floating_standard_fixture(;rotated=true)
    yr=planar_floating_line_standards(rotated,rb;left=[1],right=[2])
    @test yr.double_line.sheets[1].mask==transpose(st.double_line.sheets[1].mask)
    @test yr.double_line.ports[2].cells==st.double_line.ports[2].cells
    @test_throws ArgumentError planar_floating_line_standards(prob,bridges;left=[1],right=[2],max_bytes=1)
    @test_throws ArgumentError planar_floating_line_standards(prob,bridges;left=[1],right=[1])
    @test_throws ArgumentError planar_floating_line_standards(prob,bridges[1:1];left=[1],right=[2])
    @test_throws ArgumentError planar_floating_line_standards(prob,bridges;left=[1],right=[2],uniform_cells=4:10)
    wrong=[merge(bridges[1],(source_edge=2,)),bridges[2]]
    @test_throws ArgumentError planar_floating_line_standards(prob,wrong;left=[1],right=[2])
    # A uniform sheet's added differential series resistance is the
    # physical signal-plus-reference rho*L/w; launch turns cancel.
    a=solve_planar(st.line,1e6;mx=48,my=48,surface_zs=.05)
    b=solve_planar(st.double_line,1e6;mx=48,my=48,surface_zs=.05)
    za=2/(a.y[1,1]-a.y[1,2]);zb=2/(b.y[1,1]-b.y[1,2])
    @test real(zb-za)≈2*.05*st.length/prob.grid.dy rtol=1e-7
    @test imag(zb)>imag(za)>0
    prob.sheets[1].mask[4,6]=false
    @test_throws ArgumentError planar_floating_line_standards(prob,bridges;left=[1],right=[2])
end

@testset "coupled calibrated results preserve actual physical fields" begin
    prob,bridges=_floating_standard_fixture(;coupled=true)
    raw=solve_planar(prob,1e9;mx=32,my=32,surface_zs=.05)
    angle=.31;Q=[cos(angle) -sin(angle);sin(angle) cos(angle)]
    E=[Q zeros(2,2);zeros(2,2) Q]
    groups=[[1,3],[2,4]]
    calibrated=planar_cocalibrate(raw,groups,[E,E])
    globalQ=zeros(4,4);globalQ[groups[1],groups[1]].=Q;globalQ[groups[2],groups[2]].=Q
    @test calibrated.raw===raw
    @test calibrated.groups==groups
    @test calibrated.y≈transpose(globalQ)*raw.y*globalQ rtol=1e-12
    @test calibrated.gap_voltage_transfer≈globalQ rtol=1e-12
    @test calibrated.freq==raw.freq && calibrated.z0==fill(50.,4)
    voltage=ComplexF64[1,.2im,.3,-.1]
    maps=planar_current_maps(calibrated;voltages=voltage)
    oracle=planar_current_maps(raw;voltages=globalQ*voltage)
    @test maps[1].jx≈oracle[1].jx rtol=1e-12
    @test maps[1].jy≈oracle[1].jy rtol=1e-12
    space=PlanarStackup(prob.stack.layers,TERM_SPACE,TERM_SPACE,prob.grid.a,prob.grid.b)
    far=planar_farfield(calibrated;voltages=voltage,theta=[.3,1.1],phi=[.2,2.],radiation_stack=space)
    reference=planar_farfield(raw;voltages=globalQ*voltage,theta=[.3,1.1],phi=[.2,2.],radiation_stack=space)
    @test far.etheta≈reference.etheta rtol=1e-12
    @test far.ephi≈reference.ephi rtol=1e-12
    # The former public reference-plane wrapper lacked z0/freq metadata,
    # so even an identity launch threw in the generic radiation adapter.
    planes=planar_reference_planes(raw)
    identityfield=planar_farfield(planes;voltages=voltage,theta=[.3],phi=[.2],radiation_stack=space)
    rawfield=planar_farfield(raw;voltages=voltage,theta=[.3],phi=[.2],radiation_stack=space)
    @test identityfield.etheta≈rawfield.etheta rtol=1e-12
    composed=planar_cocalibrate(planes,groups,[E,E])
    @test composed.raw===raw
    @test composed.groups==[collect(1:4)]
    @test composed.y≈calibrated.y rtol=1e-12
    @test composed.gap_voltage_transfer≈calibrated.gap_voltage_transfer rtol=1e-12
    @test planar_current_maps(composed;voltages=voltage)[1].jy≈oracle[1].jy rtol=1e-12
    firstplane=planar_cocalibrate(raw,[[1,3]],[E])
    secondplane=planar_cocalibrate(firstplane,[[2,4]],[E])
    @test secondplane.raw===raw
    @test secondplane.y≈calibrated.y rtol=1e-12
    @test secondplane.gap_voltage_transfer≈calibrated.gap_voltage_transfer rtol=1e-12
    @test_throws ArgumentError planar_cocalibrate(raw,groups,[E,E];max_bytes=1)
    @test_throws ArgumentError planar_current_maps(calibrated;max_bytes=1)
    @test_throws ArgumentError planar_cocalibrate(raw,[[1,2],[2,3]],[E,E])
    @test_throws ArgumentError planar_cocalibrate(raw,[[1,5]],[E])
    @test_throws DimensionMismatch planar_cocalibrate(raw,groups,[zeros(3,3),E])
end

@testset "floating coupled standards physical response and calibration" begin
    prob,bridges=_floating_standard_fixture(;coupled=true)
    st=planar_floating_line_standards(prob,bridges;left=[1,3],right=[2,4])
    responses=[solve_planar(p,1e9;mx=48,my=48,surface_zs=.05) for p in
        (st.line,st.double_line,st.reflect)]
    for r in responses
        @test r.y≈transpose(r.y) rtol=1e-11
        @test opnorm(r.s)<=1+1e-10
    end
    a,b=responses[1:2];order=st.ordering
    cal=planar_group_double_delay_calibrate(a.y[order,order],b.y[order,order];length=st.length,tol=1e-9)
    y=deembed_cocal_group(a.y[order,order],[1,2],cal.launch)
    y=deembed_cocal_group(y,[3,4],cal.launch)
    @test y≈planar_abcd_to_y(cal.line) rtol=1e-9
    @test cal.residual<1e-9
    # Mirror fixtures plus reciprocal coupled transfer give an exact
    # reconstruction test, not an independent native-calibrated reference.
    T1=planar_y_to_abcd(a.y[order,order]);T2=planar_y_to_abcd(b.y[order,order])
    @test cal.launch*cal.line*cal.launch≈T1 rtol=1e-9
    @test cal.launch*cal.line^2*cal.launch≈T2 rtol=1e-9
    yr,rb=_floating_standard_fixture(;rotated=true,coupled=true)
    ys=planar_floating_line_standards(yr,rb;left=[1,3],right=[2,4])
    rr=solve_planar(ys.line,1e9;mx=48,my=48,surface_zs=.05)
    @test rr.y≈a.y rtol=1e-10
    fft=solve_planar(st.line,1e9;method=:ufft,mx=48,my=48,surface_zs=.05,memory=80,rtol=1e-10)
    @test fft.y≈a.y rtol=1e-9
end
