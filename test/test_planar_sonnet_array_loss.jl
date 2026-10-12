module SonnetArrayLossTests
using DiffMoM, Test, LinearAlgebra, SHA, JSON

const FIXTURE=joinpath(@__DIR__,"fixtures","native_array_via_loss")
cross2(a,b)=a[1]*b[2]-a[2]*b[1]

function metal_project(original,record)
    return SonnetProject(original.source,original.units,original.length_scale,original.frequency_scale,
        original.box,original.layers,[record],original.top,original.bottom,original.polygons,
        original.ports,original.variables,original.components,original.sweeps,original.records)
end

# Independent reference: the volume skin law evaluated on the fill-factored
# cross-section, computed in wide precision from the file's own fields.
function reference_array_sigma(p,model,frequency,loss,percent;selector="CDVY")
    setprecision(BigFloat,2400) do
        polygon=only(filter(q->q.kind===:via,p.polygons));v=BigFloat.(polygon.vertices)
        n=size(v,2);v[:,1]==v[:,end] && (n-=1)
        area=abs(sum(cross2(v[:,k],v[:,mod1(k+1,n)]) for k in 1:n))/2
        width=maximum(v[1,:])-minimum(v[1,:]);breadth=maximum(v[2,:])-minimum(v[2,:])
        conductivity=selector=="RSVY" ? 100/BigFloat(parse(Float64,loss)) :
            BigFloat(parse(Float64,loss))
        fill=BigFloat(parse(Float64,percent))/100
        fraction=min(area*fill/(width*breadth),one(BigFloat))
        shorter,longer=minmax(width,breadth);aspect=shorter/longer
        depth=shorter*(fraction/(1+aspect+sqrt((1-aspect)^2+4aspect*(1-fraction))))
        via=only(model.problem.vias)
        mesh=BigFloat(count(via.uni))*BigFloat(model.problem.grid.dx)*BigFloat(model.problem.grid.dy)
        q=depth*sqrt(BigFloat(pi)*BigFloat(DiffMoM._MU0)*BigFloat(frequency)*conductivity)
        u=complex(q,q);transition=q>100 ? u : u/tanh(u)
        return conductivity*fill*area/(mesh*transition)
    end
end

function component_errors(actual,reference)
    setprecision(BigFloat,2400) do
        return abs((BigFloat(real(actual))-real(reference))/real(reference)),
            abs((BigFloat(imag(actual))-imag(reference))/imag(reference))
    end
end

@testset "array loss captured native provenance" begin
    manifest=JSON.parsefile(joinpath(FIXTURE,"manifest.json"))
    @test manifest["engine_version"]=="18.53-Lite (64-bit Windows)"
    for file in manifest["files"]
        @test bytes2hex(sha256(read(joinpath(FIXTURE,file["path"]))))==file["sha256"]
    end
    for name in manifest["cases"]
        capture=JSON.parsefile(joinpath(FIXTURE,name,"capture.json"))
        @test capture["exit_code"]==0
        @test isfile(joinpath(FIXTURE,name,"native_raw.s2p"))
    end
end

@testset "array loss selector, fill law and range" begin
    cases=(("conductivity_fill","10000",".01","CDVY"),
        ("conductivity_fill2x","10000",".02","CDVY"),
        ("conductivity_solid","58000000","100","CDVY"),
        ("conductivity_onepct","58000000","1","CDVY"),
        ("resistivity","1e-4",".01","RSVY"))
    for (name,loss,percent,selector) in cases
        original=read_sonnet_project(joinpath(FIXTURE,name,"project.son"))
        record=only(original.metals)
        for frequency in (1e9,1e10,1e300)
            p=metal_project(original,copy(record))
            coordinates=copy(only(filter(q->q.kind===:via,p.polygons)).vertices)
            metals=deepcopy(p.metals);bits=precision(BigFloat)
            model=sonnet_planar_problem(p;freq=frequency,grid=(20,20),_materials=true,_details=true)
            expected=reference_array_sigma(p,model,frequency,loss,percent;selector)
            er,ei=component_errors(only(model.via_sigma),expected)
            @test er<=2e-12
            @test ei<=2e-12
            @test p.metals==metals && precision(BigFloat)==bits &&
                only(filter(q->q.kind===:via,p.polygons)).vertices==coordinates
            @test eltype(model.via_sigma)==ComplexF64
        end
    end
end

@testset "array loss polymorphic sixth field" begin
    original=read_sonnet_project(joinpath(FIXTURE,"conductivity_fill","project.son"))
    base=only(original.metals)
    # A selector token is honored only in the sixth field; a selector in the
    # density slot is a failed density entry that Lite warns on and defaults.
    for (record,selector) in (([base[1],base[2],"ARR",".5",".01","RSVY"],"RSVY"),
            ([base[1],base[2],"ARR",".5",".01","2","RSVY"],"CDVY"),
            ([base[1],base[2],"ARR",".5",".01","2",".5","RSVY"],"CDVY"),
            ([base[1],base[2],"ARR",".5",".01","RSVY","2"],"RSVY"),
            ([base[1],base[2],"ARR",".5",".01","CDVY"],"CDVY"))
        p=metal_project(original,record)
        model=sonnet_planar_problem(p;freq=1e9,grid=(20,20),_materials=true,_details=true)
        expected=reference_array_sigma(p,model,1e9,".5",".01";selector)
        er,ei=component_errors(only(model.via_sigma),expected)
        @test er<=2e-12
        @test ei<=2e-12
    end
    # Spacing, density and later tail fields are inert for non-RPV records.
    for record in ([base[1],base[2],"ARR","10000",".01"],
            [base[1],base[2],"ARR","10000",".01","CDVY"],
            [base[1],base[2],"ARR","10000",".01","1","4"],
            [base[1],base[2],"ARR","10000",".01","1","4","extra","tokens"])
        p=metal_project(original,record)
        model=sonnet_planar_problem(p;freq=1e9,grid=(20,20),_materials=true,_details=true)
        expected=reference_array_sigma(p,model,1e9,"10000",".01")
        er,ei=component_errors(only(model.via_sigma),expected)
        @test er<=2e-12 && ei<=2e-12
    end
    # RPV is honored only in the sixth field; a later RPV token is a failed
    # density entry and the record resolves as the fill-factor model.
    literal=metal_project(original,[base[1],base[2],"ARR","10000",".01","2","RPV"])
    lmodel=sonnet_planar_problem(literal;freq=1e9,grid=(20,20),_materials=true,_details=true)
    expected=reference_array_sigma(literal,lmodel,1e9,"10000",".01")
    er,ei=component_errors(only(lmodel.via_sigma),expected)
    @test er<=2e-12 && ei<=2e-12
end

@testset "array loss domain validation" begin
    original=read_sonnet_project(joinpath(FIXTURE,"conductivity_fill","project.son"))
    base=only(original.metals)
    # Lite converts an array sheet-resistance record to a fixed 3.72e7 S/m
    # conductivity independent of the entered loss, so an SRVY record must
    # produce exactly the CDVY record at that conductivity.
    smodel=sonnet_planar_problem(
        metal_project(original,[base[1],base[2],"ARR","10000",".01","SRVY"]);
        freq=1e9,grid=(20,20),_materials=true,_details=true)
    sexpected=reference_array_sigma(original,smodel,1e9,"3.72e7",".01")
    er,ei=component_errors(only(smodel.via_sigma),sexpected)
    @test er<=2e-12 && ei<=2e-12
    # A selector token in the density slot is a failed entry and stays inert.
    buried=metal_project(original,[base[1],base[2],"ARR","10000",".01","2",".5","SRVY"])
    bmodel=sonnet_planar_problem(buried;freq=1e9,grid=(20,20),_materials=true,_details=true)
    bexpected=reference_array_sigma(buried,bmodel,1e9,"10000",".01")
    er,ei=component_errors(only(bmodel.via_sigma),bexpected)
    @test er<=2e-12 && ei<=2e-12
    for record in ([base[1],base[2],"ARR","10000","0"],[base[1],base[2],"ARR","10000","-.01"],
            [base[1],base[2],"ARR","10000","150"],[base[1],base[2],"ARR","-100",".01"],
            [base[1],base[2],"ARR","10000"],[base[1],base[2],"ARR"])
        @test_throws ArgumentError sonnet_planar_problem(metal_project(original,record);
            freq=1e9,grid=(20,20),_materials=true)
    end
    # Zero conductivity is a perfect insulator; Lite runs the polygon as a
    # fully inert record — no axial element and no endpoint pads.
    open_model=sonnet_planar_problem(
        metal_project(original,[base[1],base[2],"ARR","0",".01"]);
        freq=1e9,grid=(20,20),_materials=true,_details=true)
    @test isempty(open_model.problem.vias)
    # Zero resistivity and infinite conductivity are perfect conductors.
    for record in ([base[1],base[2],"ARR","0",".01","RSVY"],
            [base[1],base[2],"ARR","INF",".01"])
        @test only(sonnet_planar_problem(metal_project(original,record);
            freq=1e9,grid=(20,20),_materials=true,_details=true).via_sigma)==Inf
    end
end

function physical_residuals(raw)
    basis=raw.problem.basis
    weights=[basis.width[b]*(basis.kind[b]==DiffMoM._BASIS_VIA_T ? .5 : 1.) for b in axes(raw.currents,1)]
    rhs=zeros(ComplexF64,size(raw.currents))
    for b in axes(rhs,1)
        q=basis.port[b];q==0 && continue
        port=raw.problem.ports[q]
        sign=port.polarity*(port.wall in (:east,:north,:volume_east,:volume_north,:terminal_x_hi,:terminal_y_hi) ? -1 : 1)
        rhs[b,q]=-sign*weights[b]
    end
    return [norm((raw.z_mom*raw.currents[:,q]-rhs[:,q])./weights)/norm(rhs[:,q]./weights) for q in axes(rhs,2)]
end

@testset "array loss native S and physical equation" begin
    for name in ("conductivity_fill","conductivity_fill2x","conductivity_solid",
            "conductivity_onepct","resistivity")
        directory=joinpath(FIXTURE,name);p=read_sonnet_project(joinpath(directory,"project.son"))
        reference=planar_read_touchstone(joinpath(directory,"native_raw.s2p"))
        for frequency in (1e9,1e10)
            native=reference.s[only(findall(==(frequency),reference.frequencies))]
            actual=solve_sonnet_project(p,frequency;raw=true,grid=(20,20),mx=160,my=160,method=:dense_fft)
            @test maximum(abs,actual.s-native)<=.06
            # The loss-resistance ratio magnifies milliohm parasitics on
            # near-PEC fills; it is only meaningful where the via is lossy.
            lossy=abs(real(100native[1,1]/native[2,1]))
            lossy>=1 && @test abs(real(100actual.s[1,1]/actual.s[2,1])/lossy-1)<=.05
            @test maximum(physical_residuals(actual.raw))<=1e-9
            @test maximum(svdvals(actual.s))<=1+1e-6
            @test maximum(abs,actual.s-transpose(actual.s))<=1e-10
        end
    end
end

@testset "array loss precision and import workspace" begin
    limit=1_000_000;outside=precision(BigFloat)
    original=read_sonnet_project(joinpath(FIXTURE,"conductivity_fill","project.son"))
    reference=sonnet_planar_problem(original;freq=1e9,grid=(20,20),_materials=true,
        _details=true,max_bytes=limit)
    expected=reference_array_sigma(original,reference,1e9,"10000",".01")
    for method in (:raster,:conformal),bits in (32,8192,1_048_576)
        setprecision(BigFloat,bits) do
            import_model()=method===:raster ?
                sonnet_planar_problem(original;freq=1e9,grid=(20,20),_materials=true,_details=true,max_bytes=limit) :
                sonnet_conformal_layout(original;freq=1e9,grid=(20,20),edge_size=1e-4,interior_size=2e-4,max_bytes=limit)
            if bits==1_048_576
                @test 8*8cld(bits,64)>limit
                @test_throws ArgumentError import_model()
            else
                er,ei=component_errors(only(import_model().via_sigma),expected)
                @test er<=2e-12
                @test ei<=2e-12
            end
            @test precision(BigFloat)==bits
        end
    end
    @test precision(BigFloat)==outside
end

@testset "array conductivity pads, open column and RSVY conversion" begin
    # Lite 18.53: ARR endpoint pads are generated films of the converted
    # conductivity at the via height — a zero-sigma record is completely
    # inert (bit-identical to the absent-via control), RSVY converts with
    # sigma=100/rho, and INF leaves perfect-conductor pads and column.
    fixture=joinpath(@__DIR__,"fixtures","native_arr_open_via")
    names=("cdvy_loss0","cdvy_1e-6","novia","probe_s0.01","probe_s1",
        "probe_s1e4","probe_s100","probe_s1000000","probe_inf",
        "probe_f6_1_RSVY","probe_f6_0.01_RSVY","probe_f6_0.0001_RSVY",
        "probe_f6_1_CDVY","probe_f6_1_none","probe_r0")
    results=Dict{String,Any}()
    for name in names
        p=read_sonnet_project(joinpath(fixture,name*".son"))
        reference=planar_read_touchstone(joinpath(fixture,name*".native.s2p"))
        for frequency in (1e9,1e10)
            native=reference.s[only(findall(==(frequency),reference.frequencies))]
            actual=solve_sonnet_project(p,frequency;raw=true,grid=(20,20),mx=160,my=160,method=:dense_fft)
            @test maximum(abs,actual.s-native)<=.06
            @test maximum(svdvals(actual.s))<=1+1e-6
            frequency==1e10 && (results[name]=actual)
        end
    end
    # The zero-sigma column must produce no axial elements and match the
    # via-absent solve, not merely a small-sigma one.
    @test isempty(results["cdvy_loss0"].raw.problem.vias)
    @test maximum(abs,results["cdvy_loss0"].s-results["novia"].s)<=1e-6
    # RSVY in the selector slot converts sigma=100/rho; the same token in
    # the density slot is ignored (CDVY default), so probe_r0 runs open.
    @test isempty(results["probe_r0"].raw.problem.vias)
    @test maximum(abs,results["probe_f6_1_RSVY"].s-results["probe_s100"].s)<=1e-6
    @test maximum(abs,results["probe_f6_0.01_RSVY"].s-results["probe_s1e4"].s)<=1e-6
    @test maximum(abs,results["probe_f6_0.0001_RSVY"].s-results["probe_s1000000"].s)<=1e-6
    @test maximum(abs,results["probe_f6_1_CDVY"].s-results["probe_f6_1_none"].s)<=1e-9
end

@testset "native ARR sheet-resistance conversion" begin
    # Lite 18.53 converts an ARR SRVY record to a fixed 3.72e7 S/m
    # conductivity independent of the entered loss, fill factor, and via
    # height (its log reports "Array via metal type converted from Sheet
    # Resistance at DC to Conductivity"); INF still selects a perfect
    # conductor. Every finite value — zero and negatives included — takes
    # the same conversion, so SRVY records replay bit-identically against
    # a CDVY record at sigma=3.72e7.
    fixture=joinpath(@__DIR__,"fixtures","native_arr_srvy")
    names=("as_1","as_100","ad_1_1_SRVY","an__5__01_SRVY","ad_INF__01_SRVY",
        "ah_srvy","at372","ac_INF","ap_3.72e7")
    results=Dict{String,Any}()
    for name in names
        p=read_sonnet_project(joinpath(fixture,name*".son"))
        reference=planar_read_touchstone(joinpath(fixture,name*".native.s2p"))
        for frequency in (1e9,1e10)
            native=reference.s[only(findall(==(frequency),reference.frequencies))]
            actual=solve_sonnet_project(p,frequency;raw=true,grid=(20,20),mx=160,my=160,method=:dense_fft)
            @test maximum(abs,actual.s-native)<=.06
            @test maximum(svdvals(actual.s))<=1+1e-6
            frequency==1e10 && (results[name]=actual)
        end
    end
    # Finite losses of any value — including zero and negatives — all take
    # the fixed conversion, so the records solve identically to the CDVY
    # sigma=3.72e7 control and to each other.
    @test maximum(abs,results["as_1"].s-results["ap_3.72e7"].s)<=1e-6
    @test maximum(abs,results["as_100"].s-results["as_1"].s)<=1e-6
    @test maximum(abs,results["an__5__01_SRVY"].s-results["as_1"].s)<=1e-6
    # Halving the layer thicknesses leaves the converted conductivity
    # unchanged: the thin-stack SRVY record is bit-equivalent to the
    # thin-stack CDVY sigma=3.72e7 control.
    @test maximum(abs,results["ah_srvy"].s-results["at372"].s)<=1e-6
    # INF still selects a perfect conductor under SRVY: the record is
    # bit-equivalent to a CDVY INF record.
    @test maximum(abs,results["ad_INF__01_SRVY"].s-results["ac_INF"].s)<=1e-6
end

@testset "native metal selector case and fallback" begin
    # Lite 18.53 matches loss selectors case-insensitively on ARR records
    # (lowercase rsvy/srvy convert exactly like the uppercase tokens), and
    # an unrecognized token in the selector slot resolves plain
    # conductivity rather than rejecting the record.
    fixture=joinpath(@__DIR__,"fixtures","native_metal_selector_case")
    names=("acase_1__01_Srvy","acase_1__01_rsvy","actl_1__01_CDVY",
        "actl_1__01_RSVY","actl_100__01_CDVY")
    results=Dict{String,Any}()
    for name in names
        p=read_sonnet_project(joinpath(fixture,name*".son"))
        reference=planar_read_touchstone(joinpath(fixture,name*".native.s2p"))
        for frequency in (1e9,1e10)
            native=reference.s[only(findall(==(frequency),reference.frequencies))]
            actual=solve_sonnet_project(p,frequency;raw=true,grid=(20,20),mx=160,my=160,method=:dense_fft)
            @test maximum(abs,actual.s-native)<=.06
            @test maximum(svdvals(actual.s))<=1+1e-6
            frequency==1e10 && (results[name]=actual)
        end
    end
    @test maximum(abs,results["acase_1__01_rsvy"].s-results["actl_1__01_RSVY"].s)<=1e-9
    @test maximum(abs,results["actl_1__01_RSVY"].s-results["actl_100__01_CDVY"].s)<=1e-9
end
end
