using DiffMoM,Test,LinearAlgebra

function _native_floating_pad_project(;cup=false)
    r(t)=SonnetRecord(1,String.(t))
    left=SonnetPolygon(:sheet,0,-1,1,
        [.00025 .0004375 .0004375 .00025 .00025;.0004375 .0004375 .0005625 .0005625 .0004375],"","","",String[])
    right=SonnetPolygon(:sheet,0,-1,2,
        [.0005625 .00075 .00075 .0005625 .0005625;.0004375 .0004375 .0005625 .0005625 .0004375],"","","",String[])
    pins=[r(["TYPE","NONE"]),r(["GNDREF","FLOAT"]),r(["TERMW","FEED"]),
        r(["SMDP","0",".4375",".5","L","1","1"]),r(["SMDP","0",".5625",".5","R","-1","2"])]
    ports=SonnetPortSpec[]
    if cup
        for (poly,edge,num,x) in ((1,1,1,".4375"),(2,3,-1,".5625"))
            records=[r(["POR1","CUP","Auto"]),r(["GNDREF","FLOAT"]),r(["TERMW","FEED"])]
            push!(ports,SonnetPortSpec(:cup,poly,edge,num,[string(num),"50","0","0","0",x,".5"],records))
        end
    end
    return SonnetProject("floating_contract.son",Dict("LNG"=>"MM","CAP"=>"PF","IND"=>"NH"),.001,1e9,
        ["1","1","1","32","32","20","0"],[[".5","1","1","0","0","0"],[".5","1","1","0","0","0"]],
        Vector{String}[],["PEC","0","RES","0"],["PEC","0","RES","0"],[left,right],ports,
        Dict{String,String}(),cup ? Vector{SonnetRecord}[] : [pins],SonnetRecord[],SonnetRecord[])
end

@testset "native paired FLOAT preserves physical differential source" begin
    smd=_native_floating_pad_project();cup=_native_floating_pad_project(;cup=true)
    a=solve_sonnet_project(smd,1e9;raw=true,mx=32,my=32)
    b=solve_sonnet_project(cup,1e9;raw=true,mx=32,my=32)
    @test a isa SonnetFloatingResult && b isa SonnetFloatingResult
    @test a.port_numbers==b.port_numbers==[1]
    @test a.y≈b.y rtol=1e-12
    @test a.s≈b.s rtol=1e-12
    @test a.source_incidence==[1.;-1.;;]
    @test a.circuit.gauge_nodes==[1]
    @test isempty(a.em.problem.vias) && isempty(b.em.problem.vias)
    @test only(a.bridges).polarity==-1
    @test Set(only(a.bridges).cells)==Set([(i,j) for i in 8:9,j in 8:9])
    @test opnorm(a.s)<=1+1e-10
    @test a.em.y≈transpose(a.em.y) rtol=1e-12
    # Independent physical geometry/current source, with no native parser
    # or component adapter in its construction.
    g=CellGrid(.001,.001,16,16);sh=sheet_level(1,16,16)
    sh.mask[5:7,8:9].=true;sh.mask[10:12,8:9].=true
    stack=PlanarStackup([PlanarLayer(1,1,.0005),PlanarLayer(1,1,.0005)],TERM_GND,TERM_GND,g.a,g.b)
    bare=PlanarProblem(stack,g,[sh],PlanarPort[],build_planar_basis(g,[sh],PlanarPort[]))
    physical=planar_floating_bridge(bare,PlanarPort(1,:terminal_x,7,8:9,50;metal_side=:negative),
        PlanarPort(1,:terminal_x,9,8:9,50;metal_side=:positive))
    oracle=solve_planar(physical.problem,1e9;mx=32,my=32)
    @test a.y≈oracle.y rtol=1e-12
    @test planar_current_maps(a;incident_waves=[1.])[1].jx≈
        planar_current_maps(a.em;voltages=transpose(a.source_incidence)*a.circuit.voltages[:,1])[1].jx rtol=1e-12
    u=solve_sonnet_project(cup,1e9;raw=true,method=:ufft,mx=32,my=32,memory=80,rtol=1e-10)
    @test u.y≈a.y rtol=1e-9
    model=sonnet_floating_model(cup,1e9)
    @test model.source_terminals==[(1,2)]
    @test model.external_labels==[1] && model.labels==[1]
    @test all(iszero,model.sheet_zs[1])
    @test_throws ArgumentError sonnet_floating_model(cup,1e9;max_bytes=1)
    @test_throws ArgumentError solve_sonnet_project(smd,1e9;raw=true,max_bytes=1)
    @test_throws ArgumentError planar_current_maps(a;max_bytes=1)
    bad=_native_floating_pad_project();bad.components[1][5]=SonnetRecord(1,["SMDP","0",".5625",".5","R","-2","2"])
    @test_throws ArgumentError sonnet_floating_model(bad,1e9)
    width=_native_floating_pad_project();width.components[1][3]=SonnetRecord(1,["TERMW","CELL"])
    cellmodel=sonnet_floating_model(width,1e9)
    @test only(cellmodel.bridges).signal.cells==9:9
    badmode=_native_floating_pad_project();badmode.components[1][3]=SonnetRecord(1,["TERMW","BOGUS"])
    @test_throws ArgumentError sonnet_floating_model(badmode,1e9)
    badcup=_native_floating_pad_project(;cup=true);badcup.ports[1].records[3]=SonnetRecord(1,["TERMW","ODD"])
    @test_throws ArgumentError sonnet_floating_model(badcup,1e9)
    @test_throws ArgumentError sonnet_floating_model(cup,1e9;bridge_zs=-1)
    @test_throws ArgumentError sonnet_floating_model(cup,1e9;bridge_zs=big"1e400")
    unknown=_native_floating_pad_project();push!(unknown.components[1],SonnetRecord(1,["UNKNOWN_PHYSICS","1"]))
    @test_throws ArgumentError sonnet_floating_model(unknown,1e9)
    group=Matrix{ComplexF64}(I,2,2)
    calibrated=solve_sonnet_project(cup,1e9;raw=true,pin_calibration=group,mx=32,my=32)
    @test calibrated.y≈a.y rtol=1e-12
    @test planar_current_maps(calibrated)[1].jx≈planar_current_maps(a)[1].jx rtol=1e-12
end

@testset "native FLOAT loading and mixed reference modes" begin
    p=_synthetic_smd_project()
    p.components[1][2]=SonnetRecord(1,["GNDREF","FLOAT"])
    p.components[1][4]=SonnetRecord(1,["SMDP","0","2.5","2","L","3","1"])
    p.components[1][5]=SonnetRecord(1,["SMDP","0","3.5","2","R","-3","2"])
    model=sonnet_component_model(p,1e8)
    @test model.labels==[1,2,3]
    @test model.external_labels==[1,2]
    @test model.source_terminals==[(1,0),(2,0),(3,4)]
    @test isempty(model.problem.vias)
    loaded=solve_sonnet_project(p,1e8;raw=true,mx=36,my=24)
    @test loaded.circuit.gauge_nodes==[3]
    @test loaded.s≈transpose(loaded.s) rtol=1e-10
    @test opnorm(loaded.s)<=1+1e-10
    # A negative-labeled terminal marks the reference side of its
    # differential port: the ideal load lands between the positive pad
    # and the component's isolated floating common, so the external pair
    # stays open. Verified against installed Sonnet's `float_pair` probe
    # (S21 ~ 1e-3), which also leaves the signed pair unloaded.
    @test abs(loaded.s[2,1])<.01
    # The reference node is the component's own floating common (node 5),
    # not the negative pad's circuit terminal.
    direct=PlanarCircuit(5,[(1,0),(2,0)])
    circuit_add_em!(direct,model.source_terminals,loaded.em)
    circuit_add_rlc!(direct,3,5;r=10.)
    oracle=solve_planar_circuit(direct,1e8;floating_gauge=:auto)
    @test oracle.s≈loaded.s rtol=1e-12
    @test planar_current_maps(loaded;incident_waves=[1.,0.])[1].jx≈
        planar_current_maps(loaded.em;voltages=transpose(loaded.source_incidence)*loaded.circuit.voltages[:,1])[1].jx rtol=1e-12
    # Asymmetric native negative wall sources require their common-mode
    # potential to be eliminated jointly with the floating component source.
    p.ports[2]=SonnetPortSpec(:box,2,1,-1,["-1","50","0","0","0","6","2"],SonnetRecord[])
    balanced=solve_sonnet_project(p,1e8;raw=true,mx=36,my=24)
    @test balanced.port_numbers==[1]
    @test size(balanced.em.voltage_transfer)==(3,2)
    @test all(isfinite,balanced.s)
    @test opnorm(balanced.s)<=1+1e-10
end

@testset "native cover components retain balanced external current constraint" begin
    p=_synthetic_smd_project()
    p.ports[2]=SonnetPortSpec(:box,2,1,-1,["-1","50","0","0","0","6","2"],SonnetRecord[])
    model=sonnet_component_model(p,1e8)
    @test size(model.floating_common)==(6,1)
    r=solve_sonnet_project(p,1e8;raw=true,mx=36,my=24)
    @test r.port_numbers==[1]
    v=r.contraction*(r.gap_transfer*r.circuit.voltages[:,1])
    @test norm(transpose(model.floating_common)*r.raw.y*v)<1e-10norm(r.raw.y*v)
    @test planar_current_maps(r.raw;voltages=v)[1].jx≈sonnet_component_current_maps(r)[1].jx rtol=1e-12
    @test opnorm(r.s)<=1+1e-10
end

@testset "native GNDREF resolution and cover-selection contracts" begin
    # Installed Lite probes: an IDEAL/NONE component under component-level
    # AUTO resolves to the same floating local ground as explicit FLOAT
    # (identical geometry and response), while BOX keeps via returns whose
    # per-pin direction is resolved independently.
    auto=_synthetic_smd_project()
    auto.components[1][2]=SonnetRecord(1,["GNDREF","AUTO"])
    resolved=sonnet_floating_model(auto,1e8)
    @test resolved.labels==[1,2,3]
    @test resolved.external_labels==[1,2]
    @test isempty(resolved.problem.vias)
    @test only(resolved.bridges).port>0
    explicit=_synthetic_smd_project()
    explicit.components[1][1]=SonnetRecord(1,["TYPE","NONE"])
    explicit.components[1][2]=SonnetRecord(1,["GNDREF","FLOAT"])
    twin=sonnet_floating_model(explicit,1e8)
    @test twin.labels==resolved.labels
    @test isempty(twin.circuit.elements)
    @test twin.external_labels==[1,2,3]
    # An IDEAL load on all-positive FLOAT pins lands between the two pad
    # nodes; no isolated common node is allocated for it (native: the
    # component common only matters when a terminal maps to it).
    loaded=_synthetic_smd_project()
    loaded.components[1][2]=SonnetRecord(1,["GNDREF","FLOAT"])
    loadedmodel=sonnet_floating_model(loaded,1e8)
    @test length(loadedmodel.circuit.elements)==1
    @test loadedmodel.source_terminals==[(1,0),(2,0),(3,4)]
    @test loadedmodel.circuit.nnodes==4
    # A file-model component under AUTO keeps via returns (it cannot resolve
    # to a floating ground without an adapter), verified with a callback.
    vendor=_synthetic_smd_project()
    vendor.components[1][1]=SonnetRecord(1,["TYPE","SPARAM","1"])
    vendor.components[1][2]=SonnetRecord(1,["GNDREF","AUTO"])
    staged=sonnet_component_model(vendor,1e8;
        component_response=(p,c,f)->(response=ComplexF64[0 1;1 0],format=:s,z0=50))
    @test !isempty(staged.problem.vias)
    # Per-pin forced directions resolve independently of the component's
    # BOX token; pins landing on different covers reject like native.
    pin_gndref(dir)=[SonnetRecord(1,["EXSMDP"]),SonnetRecord(1,["GNDREF","BOX",dir]),
        SonnetRecord(1,["END","EXSMDP"])]
    mixed=_synthetic_smd_project()
    splice!(mixed.components[1],5:4,pin_gndref("ABOVE"))
    append!(mixed.components[1],pin_gndref("BELOW"))
    @test_throws ArgumentError sonnet_component_model(mixed,1e8)
    try
        sonnet_component_model(mixed,1e8)
    catch error
        @test occursin("mixed combination of upper and lower ground references",
            sprint(showerror,error))
    end
    # Forcing a pin at a non-conductive (free space) cover rejects like
    # native; AUTO skips the non-conductive cover for the far conductive one.
    free_top(p)=SonnetProject(p.source,p.units,p.length_scale,p.frequency_scale,
        p.box,p.layers,p.metals,["\"FreeSpace\"","0","FREESPACE","376.7303136","0","0","0"],
        p.bottom,p.polygons,p.ports,p.variables,p.components,p.sweeps,p.records)
    forced=free_top(_synthetic_smd_project())
    append!(forced.components[1],pin_gndref("ABOVE"))
    @test_throws ArgumentError sonnet_component_model(forced,1e8)
    try
        sonnet_component_model(forced,1e8)
    catch error
        @test occursin("no unobstructed physical return to its requested cover",
            sprint(showerror,error))
    end
    skipmodel=sonnet_component_model(free_top(_synthetic_smd_project()),1e8)
    @test !isempty(skipmodel.problem.vias)
    @test all(path->path.direction===:below,skipmodel.reference_paths)
end

@testset "installed-Lite GNDREF probe corpus replay" begin
    # Every archived native response in test/fixtures/native_gndref_probes
    # was produced by the installed Sonnet Lite 18.53 engine. The probe
    # matrix covers AUTO/FLOAT resolution, per-pin forced directions,
    # lossy and free-space covers, and signed FLOAT pairs. Open-boundary
    # cases carry a documented residual gate; all others stay below 0.02.
    fixture=joinpath(@__DIR__,"fixtures","native_gndref_probes")
    # Free-space/open-boundary covers carry the documented half-space
    # residual (~0.05 complex, verified identical cover selection via the
    # forced-direction native control); all other gates stay below 0.02.
    openresidual=Dict("boxauto_freetop"=>.06,"forced_below_freetop"=>.06)
    for name in sort([splitext(f)[1] for f in readdir(fixture) if endswith(f,".son")])
        project=read_sonnet_project(joinpath(fixture,name*".son"))
        if isfile(joinpath(fixture,name*".native.s2p"))
            native=only(planar_read_touchstone(joinpath(fixture,name*".native.s2p");nports=2).s)
            actual=solve_sonnet_project(project,1e9;raw=true)
            @test abs(actual.s[2,1]-native[2,1])<get(openresidual,name,.02)
            @test abs(actual.s[1,1]-native[1,1])<get(openresidual,name,.02)
        else
            @test_throws ArgumentError solve_sonnet_project(project,1e9;raw=true)
        end
    end
end

@testset "native FLOAT grounded pins require the floating-common adapter" begin
    # A zero terminal on a FLOAT component is not box ground: its local
    # common floats, so the pin must route through the floating-common
    # adapter rather than silently shorting to the box.
    p=_synthetic_smd_project()
    p.components[1][2]=SonnetRecord(1,["GNDREF","FLOAT"])
    p.components[1][4]=SonnetRecord(1,["SMDP","0","2.5","2","L","0","1"])
    @test_throws ArgumentError sonnet_component_model(p,1e8)
    @test_throws ArgumentError solve_sonnet_project(p,1e8;raw=true,mx=36,my=24)
end

"""Read `em_s.txt`/`dev_s.txt` block fixtures: a frequency line followed by
`i j re im` rows per n×n matrix."""
function _float_fixture_blocks(path::AbstractString,n::Integer)
    lines=readlines(path);freqs=Float64[];blocks=Matrix{ComplexF64}[];k=1
    while k<=length(lines)
        push!(freqs,parse(Float64,strip(lines[k])));k+=1
        S=zeros(ComplexF64,n,n)
        for _ in 1:n*n
            i,j,a,b=parse.(Float64,split(lines[k]));S[Int(i),Int(j)]=a+im*b;k+=1
        end
        push!(blocks,S)
    end
    return freqs,blocks
end

"""Read a Nexxim `sdf2csv` export into per-frequency complex S matrices.
Columns are `r(Sij)`/`i(Sij)` in row-major order (same layout as
`_nexxim_csv_s` in test_planar_calibration.jl; this file is included
earlier so it keeps a local copy)."""
function _float_nexxim_csv(path::AbstractString,n::Integer)
    rows=readlines(path);header=split(rows[1],",")
    col(name)=findfirst(==(name),header)
    return map(rows[2:end]) do row
        fields=split(row,",")
        S=zeros(ComplexF64,n,n)
        for i in 1:n,j in 1:n
            S[i,j]=parse(Float64,fields[col("r(S$i$j)")])+
                im*parse(Float64,fields[col("i(S$i$j)")])
        end
        S
    end
end

@testset "FLOAT file-model attach law: Nexxim oracle" begin
    # Sonnet Lite cannot solve a FLOAT SPARAM component, so the archived
    # test/fixtures/nexxim_float_attach netlist replays the exact
    # multi-reference wiring through Ansys Nexxim: the 3-port EM block on
    # [(1,0),(2,0),(3,4)] plus the 2-port file model on [(3,5),(4,5)]
    # sharing the floating common node 5 (see the fixture README).
    fixture=joinpath(@__DIR__,"fixtures","nexxim_float_attach")
    freqs,emblocks=_float_fixture_blocks(joinpath(fixture,"em_s.txt"),3)
    _,devblocks=_float_fixture_blocks(joinpath(fixture,"dev_s.txt"),2)
    lines=readlines(joinpath(fixture,"expected.txt"))
    expected=Matrix{ComplexF64}[]
    k=1
    while k<=length(lines)
        @test parse(Float64,strip(lines[k]))≈freqs[length(expected)+1]
        push!(expected,reshape(parse.(ComplexF64,strip.(lines[k+1:k+4])),2,2))
        k+=5
    end
    nexxim=_float_nexxim_csv(joinpath(fixture,"attach.csv"),2)
    # The netlist's embedded .MODEL DATA rows must be the stored blocks.
    data=Float64[]
    for line in readlines(joinpath(fixture,"attach.sp"))
        row=strip(line)
        startswith(row,"+ ") || continue
        row=strip(row[3:end])
        (isempty(row) || row=="]" || startswith(row,"DATA") ) && continue
        append!(data,parse.(Float64,split(row)))
    end
    q=0
    for (block,n) in Iterators.flatten((((b,3) for b in emblocks),((b,2) for b in devblocks)))
        for i in 1:n,j in 1:n
            @test block[i,j]≈data[q+1]+im*data[q+2]
            q+=2
        end
    end
    @test q==length(data)
    resp=(p,c,f)->begin
        s=1im*f/1e8
        (response=ComplexF64[0.1+0.02im*s 0.85-0.05im*s;0.80-0.06im*s 0.15+0.03im*s],
            format=:s,z0=fill(50.0+0im,2))
    end
    p=_synthetic_smd_project()
    p.components[1][1]=SonnetRecord(1,["TYPE","SPARAM","2"])
    p.components[1][2]=SonnetRecord(1,["GNDREF","FLOAT"])
    model=sonnet_floating_model(p,1e8;component_response=resp)
    # Wiring asserted in the netlist header must match the lowered model.
    @test model.circuit.nnodes==5
    @test model.source_terminals==[(1,0),(2,0),(3,4)]
    @test only(model.circuit.elements).terminals==[(3,5),(4,5)]
    @test model.circuit.ports==[(1,0),(2,0)]
    @test model.external_labels==[1,2]
    for (f,Sexp,Snex,em_s,dev_s) in zip(freqs,expected,nexxim,emblocks,devblocks)
        r=solve_sonnet_project(p,f;raw=true,component_response=resp,mx=36,my=24)
        @test r.em.s≈em_s rtol=1e-8
        @test resp(nothing,nothing,f).response≈dev_s
        @test r.s≈Sexp atol=1e-10
        # Nexxim's independent MNA on the same wiring (gate 1e-3 for the
        # ~6-digit CSV; observed agreement is ~1e-7).
        @test r.s≈Snex atol=1e-3
        # Replaying the archived blocks through PlanarCircuit with the same
        # floating gauge reproduces the composite, matching the netlist.
        circuit=PlanarCircuit(5,[(1,0),(2,0)];z0=model.circuit.z0)
        circuit_add_em!(circuit,[(1,0),(2,0),(3,4)],(s=em_s,z0=r.em.z0))
        circuit_add_network!(circuit,[(3,5),(4,5)],dev_s;format=:s,z0=fill(50.0,2))
        replay=solve_planar_circuit(circuit,f;floating_gauge=:auto)
        @test replay.s≈Sexp atol=1e-10
        @test replay.s≈Snex atol=1e-3
        # The symmetric EM geometry symmetrizes the composite even though
        # the file model itself is nonreciprocal.
        @test Sexp[1,2]≈Sexp[2,1] rtol=1e-10
    end
end
