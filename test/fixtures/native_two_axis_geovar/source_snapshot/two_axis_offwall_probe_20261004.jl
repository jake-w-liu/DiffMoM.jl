using DiffMoM,SHA,TOML
const repo=normpath(joinpath(@__DIR__,".."))
include(joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
source0=joinpath(repo,"test/fixtures/native_sonnet_geometry_variables/accepted/single_ydir_anc_positive_parameter.son")
directory=mktempdir(joinpath(repo,"data/sonnet_validation");prefix="two_axis_offwall_",cleanup=false)
source=joinpath(directory,"offwall.son")
write(source,replace(read(source0,String)," NSCD"=>" SCXY"))
try
    native=reference_run(find_em(),source;output_dir=joinpath(directory,"native"),deembedded=false)
    checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false)
    println("Native accepted off-wall candidate")
catch exception
    println(sprint(showerror,exception));println(read(joinpath(directory,"native/engine_stderr.log"),String))
end
println("Retained evidence: ",directory)
