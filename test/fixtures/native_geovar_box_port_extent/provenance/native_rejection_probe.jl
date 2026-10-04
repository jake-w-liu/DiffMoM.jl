using DiffMoM,SHA,TOML
const repo=normpath(joinpath(@__DIR__,".."))
include(joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
function main()
    original=joinpath(repo,"test/fixtures/native_rad_reference_headers/rad_ydir_positive_expand_scxy.son")
    text=replace(read(original,String),"PS2 1\nPOLY 2 0"=>"PS2 2\nPOLY 2 0\nPOLY 2 0")
    output=mktempdir(joinpath(repo,"data/planar_audit");prefix="radial_three_pass_whole_selector_",cleanup=false);cp(@__FILE__,joinpath(output,"producer.jl"))
    source=joinpath(output,"repeated_whole_parameter.son");write(source,text)
    native=reference_run(find_em(),source;output_dir=joinpath(output,"parameter"),deembedded=false)
    expected=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
    command=joinpath(dirname(find_em()),"soncmd.exe");current=source;stages=Any[]
    for pass in 1:3
        dest=joinpath(output,"stage$(pass).son")
        run(pipeline(`$command -ReadWrite $current $dest`;stdout=joinpath(output,"stage$(pass).stdout.log"),stderr=joinpath(output,"stage$(pass).stderr.log")))
        p=read_sonnet_project(dest);push!(stages,[collect(q.vertices) for q in p.polygons]);current=dest
    end
    # Native normalizer geometry is independent of production. Its retained
    # variable records are removed for a fresh literal engine simulation.
    literaltext=read(current,String);lines=split(literaltext,'\n');start=findfirst(x->startswith(x,"VALVAR Move "),lines);stop=start
    while !startswith(lines[stop],"LORGN");stop+=1;end
    literaltext=join(vcat(lines[1:start-1],lines[stop:end]),'\n')
    literaltext=replace(literaltext,"TOUCH D Y"=>"TOUCH ND Y")
    literal=joinpath(output,"literal.son");write(literal,literaltext)
    native=reference_run(find_em(),literal;output_dir=joinpath(output,"literal"),deembedded=false)
    actual=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
    p=read_sonnet_project(source);q=read_sonnet_project(literal);effective=DiffMoM._sonnet_geometry_project(p,1e9)
    report=Dict("native_full_s_error"=>maximum(abs,actual-expected),"bit_identical"=>actual==expected,"geometry_error"=>maximum(maximum(abs,a.vertices-b.vertices) for (a,b) in zip(effective.polygons,q.polygons)),"stages"=>stages,"source_sha256"=>bytes2hex(sha256(read(source))))
    open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
    println(report);println("Retained whole-selector probe: ",output)
    @assert report["native_full_s_error"]<=1e-12 && report["geometry_error"]<2e-15
end
main()
