module PlanarSpectreExpressionDialectTests
using DiffMoM, Test, LinearAlgebra

@testset "Spectre expressions retain SI factor and parameter case" begin
    mktempdir() do directory
        source=joinpath(directory,"expression.scs")
        write(source,"""simulator lang=spectre
subckt resistors a b ref
R1 a ref resistor r={1M}
R2 b ref resistor r={1m}
ends
""")
        model=planar_spice_model(planar_read_spectre(source),"resistors")
        @test [element.value for element in model.elements]==[1e6,1e-3]
        result=planar_spice_sparams(model,1e9;pin_pairs=[("a","ref"),("b","ref")])
        @test result.s≈Diagonal(ComplexF64[(1e6-50)/(1e6+50),(1e-3-50)/(1e-3+50)]) atol=1e-14
        write(source,"""simulator lang=spectre
parameters R=10 r=20
subckt Child (a b) U=3 u=4
R1 a b resistor r={R+U}
R2 a b resistor r={r+u}
ends
subckt Top (a b) P=7 p=11
instance a b Child U={P+1} u={p+2}
ends
""")
        library=planar_read_spectre(source)
        @test library.parameters==["R"=>"10.0","r"=>"20.0"]
        @test first.(library.definitions["Top"].defaults)==["P","p"]
        @test [element.value for element in planar_spice_model(library,"Top").elements]==[18.,33.]
        overridden=planar_spice_model(library,"Top";parameters=Dict("P"=>17.,"p"=>21.))
        @test [element.value for element in overridden.elements]==[28.,43.]
        @test overridden.parameters==Dict("P"=>17.,"p"=>21.)
        @test_throws ArgumentError planar_spice_model(library,"Top";parameters=Dict("U"=>1.))
        write(source,"""simulator lang=spectre
parameters Delay=2n delay=8n Zref=75 zref=100
subckt line a ar b br
line1 (a ar b br) tline z0={Zref} td={Delay+1n}
ends
""")
        line=planar_spice_model(planar_read_spectre(source),"line")
        @test line.lines[1].z0==75.
        @test line.lines[1].td≈3e-9
        result=planar_spice_sparams(line,1e8;z0=75.,pin_pairs=[("a","ar"),("b","br")])
        @test result.s≈ComplexF64[0 exp(-2pi*im*1e8*3e-9);exp(-2pi*im*1e8*3e-9) 0] atol=2e-14
        # Undefined case variants reject instead of selecting another name.
        write(source,"simulator lang=spectre\nparameters R=10\nsubckt bad a b\nR1 a b resistor r={r}\nends\n")
        @test_throws ArgumentError planar_spice_model(planar_read_spectre(source),"bad")
        # SPICE keeps its case-insensitive parameters and milli-M rule.
        spice=joinpath(directory,"control.lib")
        write(spice,".param R=10\n.subckt control a b P=7\nR1 a b {r+p+1M}\n.ends\n")
        @test only(planar_spice_model(planar_read_spice(spice),"CONTROL").elements).value≈17.001
    end
end
end
