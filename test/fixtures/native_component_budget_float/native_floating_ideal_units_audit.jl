using DiffMoM,SHA,TOML
text=read(joinpath(@__DIR__,"../../test/test_planar_component_returns.jl"),String)
definition="function _synthetic_smd_project"*first(split(last(split(text,"function _synthetic_smd_project";limit=2)),"@testset";limit=2))
include_string(Main,definition)
function floating_project(value,unit)
    p=_synthetic_smd_project("RES",value);p.units["RES"]=unit
    p.components[1][2]=SonnetRecord(1,["GNDREF","FLOAT"])
    p.components[1][5]=SonnetRecord(1,["SMDP","0","3.5","2","R","-3","2"])
    p
end
report=Dict{String,Any}("production_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"../../src/planar/PlanarSonnetFloating.jl")))))
for (value,unit,expected) in (("10","OH",10.),("10","KOH",10000.),(".01","KOH",10.))
    p=floating_project(value,unit)
    direct=sonnet_floating_model(p,1e8);wrapper=sonnet_component_model(p,1e8)
    stored=only(direct.circuit.elements).r
    wrapped=only(wrapper.circuit.elements).r
    println(unit," ",value," expected=",expected," direct_stored=",stored," wrapper_stored=",wrapped)
    report[unit*" "*value]=Dict("expected"=>expected,"direct_stored"=>stored,"wrapper_stored"=>wrapped)
end
p=floating_project("RVALUE","OH");p.variables["RVALUE"]="10"
try
    result=sonnet_floating_model(p,1e8;variables=Dict("RVALUE"=>BigFloat("1e-1000")))
    report["BigFloat_override_underflow"]=only(result.circuit.elements).r
    println("direct floating nonzero BigFloat override stored=",only(result.circuit.elements).r)
catch e
    println("BigFloat override rejected: ",sprint(showerror,e))
end
open(joinpath(@__DIR__,"native_floating_ideal_units_audit.toml"),"w") do io
    TOML.print(io,report)
end
