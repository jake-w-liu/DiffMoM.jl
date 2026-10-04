using DiffMoM,SHA,TOML
const repo=normpath(joinpath(@__DIR__,".."))
include(joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
function project_text(r,target;movingx=.875,parameter=true)
    selected=r==0 ? "PS2 0\nEND" : "PS2 1\nPOLY 2 $(r)\n$(join(fill(1,r),'\n'))\nEND"
    block=parameter ? """
VALVAR Radius LNG $(target)
GEOVAR Radius RAD XDIR 1 NSCD
POS .5 .5
NOM .25
REF1 POLY 3 1
0
REF2 POLY 2 1
1
PS1 0
END
$(selected)
END
""" : ""
    """
FTYP SONPROJ 19
VER "18.53"
DIM
ANG DEG
CAP PF
CON /OH
FREQ GHZ
IND NH
LNG MM
RES OH
END DIM
CONTROL
VARSWP
OPTIONS -j
SPEED 0
SUBSPLAM N 100
END CONTROL
GEO
TMET "PEC" 0 SUP 0 0 0 0
BMET "PEC" 0 SUP 0 0 0 0
BOX 1 1 1 64 64 100 0
.1 1 1 0 0 0 2 "Air"
.1 4 1 0 0 0 2 "Substrate"
$(block)POR1 BOX
POLY 1 1
3
1 50 0 0 0 0 .21875
POR1 BOX
POLY 1 1
1
2 50 0 0 0 1 .21875
NUM 3
0 5 -1 N 1 1 1 100 100 0 0 0 Y
0 .125
1 .125
1 .3125
0 .3125
0 .125
END
0 4 -1 N 2 1 1 100 100 0 0 0 Y
.625 .875
$(movingx) .625
.9375 .875
.625 .875
END
0 5 -1 N 3 1 1 100 100 0 0 0 Y
.625 .625
.65625 .625
.65625 .65625
.625 .65625
.625 .625
END
END GEO
VARSWP
ENABLED Y
FREQ Y AN SWEEP 1
END
END VARSWP
FILEOUT
TOUCH ND Y native_raw.s2p IC 15 S RI R 50
FOLDER .
END FILEOUT
"""
end
function predicted_coordinate(r,target)
    offset=.25;nominal=.25;stages=Float64[]
    for _ in 1:3
        step=target-nominal
        for _ in 1:r+1
            iszero(step) && continue
            iszero(offset) && throw(ArgumentError("anchor encountered before next movement"))
            offset+=sign(offset)*step
        end
        push!(stages,.625+offset);nominal=abs(offset)
    end
    last(stages),stages
end
function main()
    original=joinpath(repo,"data/planar_audit/radial_reference_anchor_topology_UAU79k")
    prior=TOML.parsefile(joinpath(original,"comparison.toml"))["cases"]
    cases=vcat([(c["r"],c["target"]) for c in prior],[(r,t) for r in (0,1,5) for t in (.21875,.15625,.109375,.078125,.28125)])
    output=mktempdir(joinpath(repo,"data/planar_audit");prefix="radial_reference_three_pass_",cleanup=false)
    cp(@__FILE__,joinpath(output,"producer.jl"))
    paths=vcat([@__FILE__,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")],
        [joinpath(d,f) for folder in (joinpath(repo,"src"),original) for (d,_,fs) in walkdir(folder) for f in fs])
    hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
    report=Dict{String,Any}("scope"=>"Native RAD three successive geometry adjustments, recomputing reference radius after each pass; original10 strong contraction cases plus explicit counts0/1/5 under5 independent targets. Independent literal geometry, unchanged native fullS1e-12; source stability; no production edits.","source_before"=>hashes(),"cases"=>Any[])
    try
        for (r,target) in cases
            tag="r$(r)_target$(target)";source=joinpath(output,tag*"_parameter.son");write(source,project_text(r,target))
            native=try
                reference_run(find_em(),source;output_dir=joinpath(output,tag*"_parameter"),deembedded=false)
            catch err
                r==5 && target==.078125 || rethrow()
                row=Dict{String,Any}("case"=>tag,"r"=>r,"target"=>target,"native_error"=>sprint(showerror,err),"native_full_s_status"=>"UNVERIFIED: native memory allocation failure")
                try
                    predicted_coordinate(r,target);row["prediction_rejected"]=false
                catch prediction
                    row["prediction_rejected"]=true;row["prediction_error"]=sprint(showerror,prediction)
                end
                push!(report["cases"],row);println(row);flush(stdout);continue
            end
            expected=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
            coordinate,stages=predicted_coordinate(r,target)
            literal=joinpath(output,tag*"_literal.son");write(literal,project_text(r,target;movingx=coordinate,parameter=false))
            native=reference_run(find_em(),literal;output_dir=joinpath(output,tag*"_literal"),deembedded=false)
            actual=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
            row=Dict{String,Any}("case"=>tag,"r"=>r,"target"=>target,"moving_x"=>coordinate,"stages"=>stages,"full_s_error"=>maximum(abs,actual-expected),"bit_identical"=>actual==expected)
            p=read_sonnet_project(source);effective=DiffMoM._sonnet_geometry_project(p,1e9)
            row["production_moving_x"]=effective.polygons[2].vertices[1,2]/p.length_scale
            push!(report["cases"],row);println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained three-pass control: ",output)
    end
    @assert report["source_unchanged"] && length(report["cases"])==25
    @assert count(c->get(c,"full_s_error",Inf)<=1e-12,report["cases"])==24
    @assert count(c->haskey(c,"native_full_s_status") && get(c,"prediction_rejected",false),report["cases"])==1
end
main()
