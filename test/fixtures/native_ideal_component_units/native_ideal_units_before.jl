using DiffMoM,LinearAlgebra,TOML
root=joinpath(@__DIR__,"..","..","data","sonnet_validation","native_ideal_component_units_qY6wcp")
for name in ("ohm","kohm_same_literal","kohm_same_physical")
    p=read_sonnet_project(joinpath(root,name*".son"))
    model=sonnet_component_model(p,2e8)
    native=planar_read_touchstone(joinpath(root,name,"native_device.s2p"))
    result=solve_sonnet_project(p,2e8;raw=true,mx=64,my=80)
    delta=maximum(abs,result.s-planar_network_response(native,2e8))
    println(name," DIM RES=",p.units["RES"]," storedR=",only(model.circuit.elements).r,
        " fullS_native_error=",delta," resultS=",result.s)
end
