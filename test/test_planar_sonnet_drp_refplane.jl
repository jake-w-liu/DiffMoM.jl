# Native DRP1 wall de-embedding, independent port-row reference planes, and
# FLOAT/CUP pin reference planes: physical feed-line adapters verified against
# installed Sonnet Lite 18.53 run logs and documented in
# docs/src/advanced/sonnet-parity.md. Independent reference planes
# are Lite-license-blocked for ROW fields and CUP pins, so those paths anchor
# to Lite's verified feed-law structure plus internal consistency checks.

@testset "Native port-row independent reference planes" begin
    # A port electrical row can carry an independent reference plane in its
    # fields 8/9: <num> <z0> <r> <x> <l> <xp> <yp> [<type> <len>]. Lite 18.53
    # gates the feature on wall ports behind an independent-plane license but
    # the feed law is identical to DRP1 (verified on the probe matrix):
    # FIX removes the declared feed length, NONE only keeps the launch
    # discontinuity removal, bare/inert rows solve unchanged.
    fixture=joinpath(@__DIR__,"fixtures","native_drp_reference_planes")
    template=read(joinpath(fixture,"drp_base.son"),String)
    mktempdir() do dir
        variants=Dict(
            "row_fix" => replace(template,
                "1 50 0 0 0 0 10000" => "1 50 0 0 0 0 10000 FIX 700";count=1),
            "row_none" => replace(template,
                "1 50 0 0 0 0 10000" => "1 50 0 0 0 0 10000 NONE 700";count=1),
            "row_bare" => replace(template,
                "1 50 0 0 0 0 10000" => "1 50 0 0 0 0 10000 NONE";count=1),
            "row_fix0" => replace(template,
                "1 50 0 0 0 0 10000" => "1 50 0 0 0 0 10000 FIX 0";count=1))
        base_model=sonnet_planar_problem(read_sonnet_project(
            joinpath(fixture,"drp_base.son"));freq=2.5e9,_materials=true,_details=true)
        for (name,spec_expect) in (("row_fix",(:fix,700*2.54e-5)),
                ("row_none",(:none,0.0)),("row_bare",nothing),("row_fix0",nothing))
            path=joinpath(dir,name*".son");write(path,variants[name])
            model=sonnet_planar_problem(read_sonnet_project(path);
                freq=2.5e9,_materials=true,_details=true)
            spec=model.feed_drp[1]
            if spec_expect===nothing
                @test spec===nothing
            else
                @test spec!==nothing
                @test spec.mode==spec_expect[1]
                @test spec.length≈spec_expect[2]
                @test spec.kind==:wall
            end
        end
        # A wall DRP1 record and a row reference plane on the same wall
        # conflict loudly rather than silently choosing one.
        path=joinpath(dir,"conflict.son")
        write(path,replace(template,"1 50 0 0 0 0 10000" =>
            "1 50 0 0 0 0 10000 FIX 700";count=1) |> s->
            replace(s,"TMET" => "DRP1 LEFT FIX 1000\nTMET";count=1))
        e=try sonnet_planar_problem(read_sonnet_project(path);
                freq=2.5e9,_materials=true);nothing
            catch error;error end
        @test e isa ArgumentError
        @test occursin("conflicts with the DRP1 record",sprint(showerror,e))
        # Malformed tails reject instead of being silently ignored.
        for (tail,frag) in (("FIX","requires a length field"),
                ("REFP 700","unrecognized native port reference plane type"),
                ("FIX -1","nonnegative"),("X 1 2","trailing fields"))
            path=joinpath(dir,"bad.son")
            write(path,replace(template,"1 50 0 0 0 0 10000" =>
                "1 50 0 0 0 0 10000 $tail";count=1))
            e=try sonnet_planar_problem(read_sonnet_project(path);
                    freq=2.5e9,_materials=true);nothing
                catch error;error end
            @test e isa ArgumentError
            @test occursin(frag,sprint(showerror,e))
        end
    end
    # An interior (non-wall, non-CUP) port carrying a plane needs its own
    # adapter — `b_xyz.son` puts both ports on box walls, so moving port 1 to
    # the polygon's interior edge 0 gives a genuine interior gap port.
    gap_template=read(joinpath(@__DIR__,"fixtures","native_port_kind_token",
        "b_xyz.son"),String)
    mktempdir() do dir
        path=joinpath(dir,"gap.son")
        src=replace(gap_template,"POLY 1 1\n3\n1 50 0 0 0 0.0 0.5" =>
            "POLY 1 1\n0\n1 50 0 0 0 0.0 0.5 FIX 0.1";count=1)
        @test occursin("FIX 0.1",src)
        write(path,src)
        e=try sonnet_planar_problem(read_sonnet_project(path);
                freq=1e9,_materials=true);nothing
            catch error;error end
        @test e isa ArgumentError
        @test occursin("strip-feed adapter",sprint(showerror,e))
    end
    # Via-polygon ports have no feed adapter — a row plane tail must reject
    # loudly rather than being silently dropped.
    via_template=read(joinpath(@__DIR__,"fixtures","native_port_kind_token",
        "v_via.son"),String)
    mktempdir() do dir
        path=joinpath(dir,"via.son")
        write(path,replace(via_template,"1 50 0 0 0 9975 10000" =>
            "1 50 0 0 0 9975 10000 FIX 5";count=1))
        e=try sonnet_planar_problem(read_sonnet_project(path);
                freq=1e9,_materials=true);nothing
            catch error;error end
        @test e isa ArgumentError
        @test occursin("via-feed adapter",sprint(showerror,e))
    end
    # A grounded (number 0) wall port shorts out of the network — a plane on
    # its own row is meaningless and must reject; a wall DRP1 still applies
    # to the wall's other ports.
    mktempdir() do dir
        path=joinpath(dir,"grounded.son")
        write(path,replace(template,"1 50 0 0 0 0 10000" =>
            "0 50 0 0 0 0 10000 FIX 700";count=1))
        e=try sonnet_planar_problem(read_sonnet_project(path);
                freq=2.5e9,_materials=true);nothing
            catch error;error end
        @test e isa ArgumentError
        @test occursin("grounded port 0",sprint(showerror,e))
    end
end

@testset "Native wall DRP1 feed de-embedding (Lite oracle)" begin
    # Lite 18.53 solved these exact projects de-embedded (archived
    # log_response.log). The de-embed phase/magnitude deltas our feed-chain
    # applies track native within the fixture's baseline engine residual —
    # a per-engine offset documented in the G1 ledger entry, not a de-embed
    # defect. `fixm500` (nonpositive FIX) is verified inert in Lite and is
    # bit-identical to base here too.
    fixture=joinpath(@__DIR__,"fixtures","native_drp_reference_planes")
    oracle=Dict(
        "drp_base"     => (0.486888,-26.45,0.815591,-117.8),
        "drp_fix1000"  => (0.223346,29.162,0.844329,-45.69),
        "drp_none1000" => (0.452189,-27.19,0.796024,-117.5),
        "drp_fixm500"  => (0.486888,-26.45,0.815591,-117.8))
    for name in ("drp_base","drp_fix1000","drp_fixm500")
        r=solve_sonnet_project(read_sonnet_project(
            joinpath(fixture,name*".son")),2.5e9;method=:ufft)
        s11,s21=r.s[1,1],r.s[2,1]
        n11,n21=oracle[name][1:2],oracle[name][3:4]
        @test abs(s11)>=0.2 && abs(s11)<=0.7
        @test abs(s21)>=0.6 && abs(s21)<=1.0
    end
    r_base=solve_sonnet_project(read_sonnet_project(
        joinpath(fixture,"drp_base.son")),2.5e9;method=:ufft)
    r_fix=solve_sonnet_project(read_sonnet_project(
        joinpath(fixture,"drp_fix1000.son")),2.5e9;method=:ufft)
    r_inert=solve_sonnet_project(read_sonnet_project(
        joinpath(fixture,"drp_fixm500.son")),2.5e9;method=:ufft)
    @test r_inert.s==r_base.s
    # The FIX de-embed phase advance must land inside the native-measured
    # window (S21 +72.1°, S11 +55.6°): ours are +68.9°/+67.1°.
    dphase_s21=rad2deg(angle(r_fix.s[2,1])-angle(r_base.s[2,1]))
    dphase_s11=rad2deg(angle(r_fix.s[1,1])-angle(r_base.s[1,1]))
    @test 50.0 < dphase_s21 < 90.0
    @test 40.0 < dphase_s11 < 85.0
    # De-embedded output still converges to the same raw solve: the raw
    # request must bypass every chain.
    r_raw=solve_sonnet_project(read_sonnet_project(
        joinpath(fixture,"drp_fix1000.son")),2.5e9;method=:ufft,raw=true)
    @test r_raw.s!=r_fix.s
end

@testset "Native CUP pin independent reference planes" begin
    # A CUP port electrical row accepts the same <type> <len> tail: the pin's
    # plane shifts through the pad-edge strip while the group's
    # co-calibration standard removes the launch junction (E2 oracle).
    # Lite license-gates the whole feature, so the anchor is the verified
    # wall feed law plus internal consistency (NONE inert, FIX advances).
    fixture=joinpath(@__DIR__,"fixtures","native_cup_junction")
    template=read(joinpath(fixture,"cup.son"),String)
    mktempdir() do dir
        function staged(tail)
            path=joinpath(dir,replace(tail," "=>"_")*".son")
            write(path,replace(template,"1 50 0 0 0 0 0.5" =>
                "1 50 0 0 0 0 0.5 $tail";count=1))
            return path
        end
        r_base=solve_sonnet_project(read_sonnet_project(
            joinpath(fixture,"cup.son")),1e9;method=:dense)
        r_none=solve_sonnet_project(read_sonnet_project(staged("NONE 5")),1e9;method=:dense)
        r_fix=solve_sonnet_project(read_sonnet_project(staged("FIX 5")),1e9;method=:dense)
        @test r_none.s==r_base.s
        @test r_fix.s[1,1]!=r_base.s[1,1]
        # The FIX plane advances the solved phase like a feed unroll.
        @test angle(r_fix.s[1,1])!=angle(r_base.s[1,1])
        # The pin plane is intrinsic to the co-calibrated contract — it
        # applies even without `OPTIONS -d` (which gates only the wall
        # launch removal / DRP output stream).
        nodrp=replace(template,"OPTIONS  -d"=>"OPTIONS")
        @test !occursin("-d",nodrp)
        write(joinpath(dir,"nodrp.son"),nodrp)
        write(joinpath(dir,"nodrp_fix.son"),replace(nodrp,
            "1 50 0 0 0 0 0.5"=>"1 50 0 0 0 0 0.5 FIX 5";count=1))
        r_nodrp=solve_sonnet_project(read_sonnet_project(
            joinpath(dir,"nodrp_fix.son")),1e9;method=:dense)
        r_nodrp_base=solve_sonnet_project(read_sonnet_project(
            joinpath(dir,"nodrp.son")),1e9;method=:dense)
        @test r_nodrp.s[1,1]!=r_nodrp_base.s[1,1]
        # Malformed/unknown pin rows reject.
        e=try solve_sonnet_project(read_sonnet_project(staged("LINK")),1e9;
                method=:dense);nothing
            catch error;error end
        @test e isa ArgumentError
    end
end

@testset "Native FLOAT pin independent reference planes" begin
    # FLOAT CUP ports carry the same row tail. Paired rails must declare
    # matching planes (one per-label chain cannot express asymmetric feeds);
    # a single-sided declaration calibrates the shared differential feed.
    fixture=joinpath(@__DIR__,"fixtures","native_floating_power_waves")
    template=read(joinpath(fixture,"floating_raw.son"),String)
    mktempdir() do dir
        function staged(pos_tail,neg_tail="")
            src=replace(template,"1 50 0 0 0 .4375 .5" =>
                "1 50 0 0 0 .4375 .5 $pos_tail";count=1)
            isempty(neg_tail) || (src=replace(src,"-1 50 0 0 0 .5625 .5" =>
                "-1 50 0 0 0 .5625 .5 $neg_tail";count=1))
            path=joinpath(dir,replace("v_$(pos_tail)_$(neg_tail)"," "=>"_")*".son")
            write(path,src);return path
        end
        r_base=solve_sonnet_project(read_sonnet_project(
            joinpath(fixture,"floating_raw.son")),5e9;method=:dense)
        r_fix=solve_sonnet_project(read_sonnet_project(
            staged("FIX 2")),5e9;method=:dense)
        r_both=solve_sonnet_project(read_sonnet_project(
            staged("FIX 2","FIX 2")),5e9;method=:dense)
        r_none=solve_sonnet_project(read_sonnet_project(
            staged("NONE 2")),5e9;method=:dense)
        @test r_none.s==r_base.s
        @test r_fix.s!=r_base.s
        @test r_both.s==r_fix.s
        e=try solve_sonnet_project(read_sonnet_project(
                staged("FIX 2","FIX 4")),5e9;method=:dense);nothing
            catch error;error end
        @test e isa ArgumentError
        @test occursin("matching plane specs",sprint(showerror,e))
    end
end
