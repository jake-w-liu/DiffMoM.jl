module NativeSonnetDielectricScalingTests
using DiffMoM,Test

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
end
