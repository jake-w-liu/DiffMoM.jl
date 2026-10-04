using DiffMoM,SHA,TOML
const repo=normpath(joinpath(@__DIR__,".."))
include(joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
function project_text(r,target;movingx=.775,movingy=.825,ordinaryx=.5,ordinary_count=2,reversed=false,parameter=true)
    selected=r==0 ? "PS2 0\nEND" : "PS2 1\nPOLY 2 $(r)\n$(join(fill(1,r),'\n'))\nEND"
    ordinary="POLY 4 $(ordinary_count)\n$(join(fill(1,ordinary_count),'\n'))"
    references="POLY 2 $(r)\n$(join(fill(1,r),'\n'))"
    selected="PS2 2\n"*(reversed ? ordinary*"\n"*references : references*"\n"*ordinary)*"\nEND"
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
NUM 4
0 5 -1 N 1 1 1 100 100 0 0 0 Y
0 .125
1 .125
1 .3125
0 .3125
0 .125
END
0 4 -1 N 2 1 1 100 100 0 0 0 Y
.625 .96875
$(movingx) $(movingy)
.9375 .96875
.625 .96875
END
0 5 -1 N 3 1 1 100 100 0 0 0 Y
.625 .625
.65625 .625
.65625 .65625
.625 .65625
.625 .625
END
0 4 -1 N 4 1 1 100 100 0 0 0 Y
.4375 .875
$(ordinaryx) .625
.5625 .875
.4375 .875
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
function expected_points(r,target,ordinary_count)
    x,y=.775,.825;ox,oy=.5,.625;nominal=.25;stages=Any[]
    for _ in 1:3
        step=target-nominal
        for _ in 1:r+1
            iszero(step) && continue
            dx,dy=x-.625,y-.625;radius=hypot(dx,dy)
            x+=step*dx/radius;y+=step*dy/radius
        end
        for _ in 1:ordinary_count
            iszero(step) && continue
            dx,dy=ox-.625,oy-.625;radius=hypot(dx,dy)
            ox+=step*dx/radius;oy+=step*dy/radius
        end
        nominal=hypot(x-.625,y-.625);push!(stages,[x,y,ox,oy])
    end
    x,y,ox,stages
end
function main()
    output=mktempdir(joinpath(repo,"data/planar_audit");prefix="radial_three_pass_diagonal_",cleanup=false)
    cp(@__FILE__,joinpath(output,"producer.jl"))
    paths=vcat([@__FILE__,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")],[joinpath(d,f) for (d,_,fs) in walkdir(joinpath(repo,"src")) for f in fs])
    hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
    report=Dict{String,Any}("scope"=>"Independent diagonal RAD references with repeated ordinary point on opposite ray, point-group order reversed, counts2/3/4, two strong contractions. Native fullS1e-12 gate and production mask/coordinate comparisons; no shared production movement helpers in expected literal.","source_before"=>hashes(),"cases"=>Any[])
    try
        for r in (2,3,4),target in (.21875,.109375),reversed in (false,true)
            tag="r$(r)_target$(target)_reverse$(reversed)";source=joinpath(output,tag*"_parameter.son");text=project_text(r,target;reversed)
            reversed && (text=replace(text,"RAD XDIR 1 NSCD"=>"RAD YDIR -1 SCXY"))
            write(source,text)
            native=reference_run(find_em(),source;output_dir=joinpath(output,tag*"_parameter"),deembedded=false)
            expected=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
            x,y,ordinaryx,stages=expected_points(r,target,2)
            literal=joinpath(output,tag*"_literal.son");write(literal,project_text(r,target;movingx=x,movingy=y,ordinaryx,parameter=false))
            native=reference_run(find_em(),literal;output_dir=joinpath(output,tag*"_literal"),deembedded=false)
            actual=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
            p=read_sonnet_project(source);q=read_sonnet_project(literal);effective=DiffMoM._sonnet_geometry_project(p,1e9)
            masks_equal=all(a.mask==b.mask for (a,b) in zip(sonnet_planar_problem(p;freq=1e9).sheets,sonnet_planar_problem(q;freq=1e9).sheets))
            row=Dict{String,Any}("case"=>tag,"r"=>r,"target"=>target,"reversed"=>reversed,"ordinary_count"=>2,"stages"=>stages,"moving_x"=>x,"moving_y"=>y,"ordinary_x"=>ordinaryx,"full_s_error"=>maximum(abs,actual-expected),"bit_identical"=>actual==expected,"geometry_error"=>maximum(maximum(abs,a.vertices-b.vertices) for (a,b) in zip(effective.polygons,q.polygons)),"masks_equal"=>masks_equal)
            push!(report["cases"],row);println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained diagonal controls: ",output)
    end
    @assert report["source_unchanged"] && length(report["cases"])==12 && all(c->c["full_s_error"]<=1e-12 && c["masks_equal"] && c["geometry_error"]<2e-15,report["cases"])
end
main()
