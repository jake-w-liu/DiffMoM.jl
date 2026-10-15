module PlanarSonnetSharedBindingsTests
using DiffMoM, Test, LinearAlgebra

function geometry_child(path)
    write(path,"""FTYP SONPROJ 19
VER "18.53"
DIM
FREQ MHZ
CAP PF
IND NH
LNG MIL
RES OH
CON /OH
ANG DEG
END DIM
GEO
TMET "Lossless" 0 SUP 0 0 0 0
BMET "Lossless" 0 SUP 0 0 0 0
MET "Copper" 1 NOR 58000000 .5 .01
BOX 1 80 40 40 20 100 0
1 1 1 0 0 0 0 "Air_bottom"
1 1 1 0 0 0 0 "Air_top"
VALVAR G LNG 4 "gap"
GEOVAR G ANC XDIR 1 NSCD
POS 0 0
NOM 4
REF1 POLY 9 1
2
REF2 POLY 10 1
3
PS1 0
END
PS2 1
POLY 10 1
0
END
END
LORGN 0 1 U
POR1 BOX
POLY 9 1
3
1 50 0 0 0 0 21
POR1 BOX
POLY 10 1
1
2 50 0 0 0 80 24
NUM 2
0 5 -1 N 9 1 1 100 100 0 0 0 Y
0 18
20 18
20 24
0 24
0 18
END
0 5 -1 N 10 1 1 100 100 0 0 0 Y
24 18
80 18
80 30
24 30
24 18
END
END GEO
""")
end

@testset "Shared SPROJ responses depend on each instance's variables" begin
    fixture=joinpath(@__DIR__,"fixtures","native_sparam_model_files","native","sparameter.son")
    mktempdir() do directory
        parent=joinpath(directory,"parent.son");child=joinpath(directory,"child.son")
        write(parent,replace(read(fixture,String),"TYPE SPARAM 1"=>
            "TYPE SPROJ child.son G=8\nINHSWP N"))
        geometry_child(child)
        project=read_sonnet_project(parent)
        second=deepcopy(only(project.components))
        only(filter(r->first(r.tokens)=="ID",second)).tokens[2]="2"
        secondtype=only(filter(r->first(r.tokens)=="TYPE",second)).tokens
        secondtype[4]="G=16"
        push!(project.components,second)
        staged=sonnet_component_files(project,10e9)
        isolated=deepcopy(project);deleteat!(isolated.components,1)
        expected=only(sonnet_component_files(isolated,10e9).bindings)
        @test length(staged.sources)==2
        @test length(staged.bindings)==2
        @test staged.bindings[1].source==staged.bindings[2].source==realpath(child)
        @test staged.bindings[1].response!=staged.bindings[2].response
        @test staged.bindings[2].response≈expected.response rtol=1e-13
        @test maximum(abs,staged.bindings[1].response-expected.response)>.1
        # A cache hit for a previous instance must not bypass validation.
        secondtype[4]="UNKNOWN=8"
        @test_throws ArgumentError sonnet_component_files(project,10e9)
        # Empty and explicit bindings remain distinct even when the empty
        # instance is visited after an overridden instance of the same file.
        resize!(secondtype,3)
        default=sonnet_component_files(project,10e9)
        isolated=deepcopy(project);deleteat!(isolated.components,1)
        @test default.bindings[2].response≈only(sonnet_component_files(isolated,10e9).bindings).response rtol=1e-13
        @test default.bindings[1].response!=default.bindings[2].response
        # Equal effective assignments share a canonical key regardless of
        # the dictionary iteration order.
        a=Dict("G"=>8.,"W"=>3.);b=Dict("W"=>3.,"G"=>8.)
        @test DiffMoM._sonnet_project_response_key(realpath(child),a)==
            DiffMoM._sonnet_project_response_key(realpath(child),b)
    end
end
end
