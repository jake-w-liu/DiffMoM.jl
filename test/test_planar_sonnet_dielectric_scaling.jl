module NativeSonnetDielectricScalingTests
using DiffMoM,Test,LinearAlgebra

function oracle_loss(sigma,f)
    setprecision(BigFloat,256) do
        Float64(BigFloat(sigma)/(2BigFloat(pi)*BigFloat(f)*BigFloat(DiffMoM._EPS0)))
    end
end

@testset "Native and STF dielectric conductive loss preserves finite ratios" begin
    native=read_sonnet_project(joinpath(@__DIR__,"fixtures/native_dielectric_resistivity_gui/rsvy_1GHz/rsvy_1GHz.son"))
    technology=read_sonnet_technology(joinpath(@__DIR__,"fixtures/native_stf_unit_gui/input_resistivity.stf"))
    for f in (1e9,1e308,floatmax(Float64))
        expected=oracle_loss(100/7,f)
        @test isfinite(expected) && expected>0
        static=sonnet_technology_stack(technology,f,.001,.001)
        lowered=DiffMoM._sonnet_stack_geometry(native,f,nothing,Dict())
        @test -imag(static.layers[1].epsr)≈expected rtol=4e-15
        @test -imag(lowered.stack.layers[1].epsr)≈expected rtol=4e-15
    end
    # Exercise admitted tiny frequencies and conductivities through the
    # public static STF provider, using declared quantity overrides.
    mktempdir() do dir
        xml=read(technology.source,String)
        xml=replace(xml,"<variables>"=>"<variables><var name=\"Conductivity\" value=\"1\"/>",
            "cond_res=\"rsvy\""=>"cond_res=\"cond\"", "rsvy=\"Rho\""=>"cond=\"Conductivity\"")
        source=joinpath(dir,"conductivity.stf");write(source,xml)
        t=read_sonnet_technology(source)
        tiny=nextfloat(0.)
        for (sigma,f) in ((tiny,tiny),(0.,tiny),(1e-300,1e-310),
                (1e-310,1e-300),(1e300,1e300),(tiny,1.),(tiny,1e9),(tiny,floatmax(Float64)))
            expected=oracle_loss(sigma,f)
            actual=sonnet_technology_stack(t,f,.001,.001;variables=Dict("Conductivity"=>sigma))
            @test isfinite(actual.layers[1].epsr)
            @test -imag(actual.layers[1].epsr)≈expected rtol=4e-15
            @test actual.layers[2].epsr==1.
        end
        @test_throws ArgumentError sonnet_technology_stack(t,1e-300,.001,.001;
            variables=Dict("Conductivity"=>floatmax(Float64)))
    end
end

@testset "native dielectric SRVY law and selector semantics" begin
    # Lite 18.53 probes (fixtures/native_dielectric_srvy): the first token
    # after the layer name is the loss selector — RSVY reads ohm-cm
    # (sigma=100/rho), SRVY reads ohms-per-square through the layer
    # thickness (sigma=1/(Rs*t)), every other token including CDVY selects
    # plain conductivity, and later tokens are inert. The selector is
    # case-insensitive. Field 7 and any further numerics are parsed but
    # inert, so an "anisotropic" row leaves the selector position at a
    # numeric token and the loss stays plain conductivity either way.
    fixture=joinpath(@__DIR__,"fixtures","native_dielectric_srvy")
    loss_of(name,f)=begin
        p=read_sonnet_project(joinpath(fixture,name*".son"))
        -imag(DiffMoM._sonnet_stack_geometry(p,f,nothing,Dict()).stack.layers[2].epsr)
    end
    # sigma = 1/(Rs*t) with t = 0.1 mm, verified bit-identical against
    # matched CDVY conductivity controls in Lite.
    for (name,sigma) in (("srvy_1e5",.1),("srvy_2e5",.05),("srvy_5e5",.02),
            ("srvy_1e4",1.),("srvy_1",1e4),("srvy_lower",.1),("srvy_mixed",.1),
            ("srvy_erz",.1))
        @test loss_of(name,1e9)≈oracle_loss(sigma,1e9) rtol=1e-10
        @test loss_of(name,1e10)≈oracle_loss(sigma,1e10) rtol=1e-10
    end
    # First-token selector: X and CDVY win the slot and leave the field as
    # plain conductivity; SRVY wins over a trailing CDVY. An anisotropic
    # record keeps the extra numerics inert, so its trailing SRVY token
    # never reaches the selector slot.
    for (name,sigma) in (("first_x_rsvy",1.),("first_cdvy_srvy",1.),
            ("first_srvy_cdvy",1e4),("aniso_srvy_inert",.01),
            ("aniso_fields",.01),("sigma_0p1",.1),("sigma_1e4",1e4))
        @test loss_of(name,1e9)≈oracle_loss(sigma,1e9) rtol=1e-10
    end
    # Negative conductivity rejects natively for every selector; a zero
    # resistivity or sheet resistance is a degenerate perfect-conductor
    # layer that the lowered model cannot represent, so it also rejects.
    # The reciprocal selectors map a literal INF field to sigma=0, but a
    # plain-conductivity INF is that same degenerate conductor layer.
    for name in ("srvy_inf","rsvy_inf")
        @test loss_of(name,1e9)==0
        @test loss_of(name,1e10)==0
    end
    for name in ("srvy_neg","sigma_neg","srvy_zero","rsvy_zero","cdvy_inf")
        p=read_sonnet_project(joinpath(fixture,name*".son"))
        @test_throws ArgumentError DiffMoM._sonnet_stack_geometry(p,1e9,nothing,Dict())
    end
    # The record is positional: the loss slot doubles as the name position
    # in a five-field short record (six tokens with a name token there, or
    # six bare numerics), field 7 is inert but must stay numeric, and flags
    # only exist after a name in the long form. Each boundary probe below
    # is a native parse reject.
    @test loss_of("short_name",1e9)==0
    @test loss_of("bare_seven",1e9)==0
    @test loss_of("six_numeric",1e9)≈oracle_loss(1e5,1e9) rtol=1e-10
    for name in ("name_at_f7","bad_f7","flag_at_f7","f7_flag_token",
            "short_name_flag","six_n_flag")
        p=read_sonnet_project(joinpath(fixture,name*".son"))
        @test_throws ArgumentError DiffMoM._sonnet_stack_geometry(p,1e9,nothing,Dict())
    end
    # Four-field short records are also legal natively but below the
    # reader's six-token floor, so they reject at read rather than stack.
    @test_throws ArgumentError read_sonnet_project(joinpath(fixture,"four_n.son"))
    p=read_sonnet_project(joinpath(fixture,"unquoted_srvy.son"))
    @test_throws ArgumentError DiffMoM._sonnet_stack_geometry(p,1e9,nothing,Dict())
    # End-to-end replay agreement in the regime where the lossy layer still
    # passes measurable transmission; the same solves also check that SRVY
    # and plain-conductivity records of the converted sigma are identical
    # inside the engine.
    results=Dict{String,Any}()
    for name in ("base","srvy_1e5","srvy_2e5","srvy_5e5","srvy_1e4","sigma_0p1",
            "sigma_0p05","sigma_0p02","sigma_1","srvy_erz","srvy_lower",
            "first_x_rsvy","first_cdvy_srvy","aniso_fields","aniso_srvy_inert",
            "srvy_inf","rsvy_inf")
        p=read_sonnet_project(joinpath(fixture,name*".son"))
        reference=planar_read_touchstone(joinpath(fixture,name*".native.s2p"))
        for frequency in (1e9,1e10)
            native=reference.s[only(findall(==(frequency),reference.frequencies))]
            actual=solve_sonnet_project(p,frequency;raw=true,grid=(20,20),mx=160,my=160,method=:dense_fft)
            @test maximum(abs,actual.s-native)<=.06
            @test maximum(svdvals(actual.s))<=1+1e-6
            frequency==1e10 && (results[name]=actual.s)
        end
    end
    for (a,b) in (("srvy_1e5","sigma_0p1"),("srvy_2e5","sigma_0p05"),
            ("srvy_5e5","sigma_0p02"),("srvy_1e4","sigma_1"),
            ("srvy_lower","srvy_1e5"),("srvy_erz","srvy_1e5"),
            ("aniso_srvy_inert","aniso_fields"))
        @test maximum(abs,results[a]-results[b])<=1e-12
    end
end
end
