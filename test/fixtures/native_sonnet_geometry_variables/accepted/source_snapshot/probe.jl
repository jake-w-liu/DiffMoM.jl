# Read-only production audit: independently declared native GEOVAR/literal pairs.
using DiffMoM, SHA, TOML, Dates
include(joinpath(@__DIR__, "../../sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference

function geovar_source(kind, sign, width; literal=false,axis="YDIR",split=false)
    nominal=.25
    low,high=literal ? (kind=="SYM" ? (.5-width/2,.5+width/2) :
        sign==1 ? (.375,.375+width) : (.625-width,.625)) : (.375,.625)
    block=if literal
        ""
    else
        firstref,secondref=sign==1 ? (0,3) : (3,0)
        firstextra,secondextra=sign==1 ? (1,2) : (2,1)
        firstset=kind=="SYM" ? "PS1 1\nPOLY 1 1\n$(firstextra)\nEND\n" : "PS1 0\nEND\n"
        """
VALVAR Width LNG $(width) "independent width control"
GEOVAR Width $(kind) $(axis) $(sign) NSCD
POS 0.5 0.5
NOM $(nominal)
REF1 POLY 1 1
$(firstref)
REF2 POLY 1 1
$(secondref)
$(firstset)PS2 1
POLY 1 1
$(secondextra)
END
END
"""
    end
    text="""
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
OPTIONS
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
1 50 0 0 0 0 .5
POR1 BOX
POLY 1 1
1
2 50 0 0 0 1 .5
NUM 1
0 5 -1 N 1 1 1 100 100 0 0 0 Y
0 $(low)
1 $(low)
1 $(high)
0 $(high)
0 $(low)
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
    if axis=="XDIR"
        text=replace(text,"0 $(low)\n1 $(low)\n1 $(high)\n0 $(high)\n0 $(low)"=>
            "$(low) 0\n$(low) 1\n$(high) 1\n$(high) 0\n$(low) 0",
            "1 50 0 0 0 0 .5"=>"1 50 0 0 0 .5 0",
            "2 50 0 0 0 1 .5"=>"2 50 0 0 0 .5 1")
    end
    if split
        firstidx,secondidx=sign==1 ? (0,2) : (3,1)
        upper="POLY 11 2\n2\n3\nPOLY 91 1\n$(sign==1 ? 3 : 2)\n"
        lower="POLY 11 2\n0\n1\nPOLY 91 1\n0\n"
        firstset=kind=="SYM" ? (sign==1 ?
            "PS1 2\nPOLY 11 1\n1\nPOLY 91 2\n0\n1\nEND\n" :
            "PS1 2\nPOLY 11 1\n2\nPOLY 91 2\n2\n3\nEND\n") : "PS1 0\nEND\n"
        block2="""
VALVAR Width LNG $(width) "independent width control"
GEOVAR Width $(kind) $(axis) $(sign) NSCD
POS 0.5 0.5
NOM $(nominal)
REF1 POLY 11 1
$(firstidx)
REF2 POLY 91 1
$(secondidx)
$(firstset)PS2 2
$(sign==1 ? upper : lower)END
END
"""
        literal || (text=replace(text,block=>block2))
        firstrect=axis=="YDIR" ? "0 $(low)\n.5 $(low)\n.5 $(high)\n0 $(high)\n0 $(low)" :
            "$(low) 0\n$(low) .5\n$(high) .5\n$(high) 0\n$(low) 0"
        secondrect=axis=="YDIR" ? ".5 $(low)\n1 $(low)\n1 $(high)\n.5 $(high)\n.5 $(low)" :
            "$(low) .5\n$(low) 1\n$(high) 1\n$(high) .5\n$(low) .5"
        beginpolygon=findfirst("NUM 1\n",text);endpolygon=findfirst("END GEO",text)
        polygontext="NUM 2\n0 5 -1 N 11 1 1 100 100 0 0 0 Y\n$(firstrect)\nEND\n0 5 -1 N 91 1 1 100 100 0 0 0 Y\n$(secondrect)\nEND\n"
        text=text[1:first(beginpolygon)-1]*polygontext*text[first(endpolygon):end]
        text=replace(text,"POR1 BOX\nPOLY 1 1\n3"=>"POR1 BOX\nPOLY 11 1\n3",
            "POR1 BOX\nPOLY 1 1\n1"=>"POR1 BOX\nPOLY 91 1\n1")
    end
    return text
end

function main()
    directory=joinpath(@__DIR__,"native_"*string(time_ns()));mkpath(directory)
    em=find_em();em===nothing && error("installed native engine unavailable")
    sources=[joinpath(@__DIR__,"../../../src/planar/PlanarSonnetIO.jl"),
        joinpath(@__DIR__,"../../../src/planar/PlanarSonnetScalarFiles.jl"),
        joinpath(@__DIR__,"../../sonnet_stripline/sonnet_reference.jl"),@__FILE__]
    hashes()=Dict(abspath(path)=>bytes2hex(sha256(read(path))) for path in sources)
    report=Dict{String,Any}("scope"=>"actual native simple ANC/SYM NSCD width changes versus independently declared literal rectangles; no production geometry adapter or broader dependent/scaled/radial claims",
        "started_utc"=>string(now(UTC)),"source_before"=>hashes(),"cases"=>Dict{String,Any}[])
    try
        for split in (false,true),axis in ("YDIR","XDIR"),(kind,sign,width) in (("ANC",1,.375),("ANC",-1,.375),("SYM",1,.375),("SYM",-1,.375))
            tag="$(split ? "split" : "single")_$(lowercase(axis))_$(lowercase(kind))_$(sign==1 ? "positive" : "negative")";matrices=Matrix{ComplexF64}[]
            record=Dict{String,Any}("kind"=>kind,"axis"=>axis,"direction"=>sign,"split_polygons"=>split,"width_mm"=>width,"status"=>"UNVERIFIED")
            push!(report["cases"],record)
            for literal in (false,true)
                name=tag*(literal ? "_literal" : "_parameter")
                source=joinpath(directory,name*".son");write(source,geovar_source(kind,sign,width;literal,axis,split))
                reference=reference_run(em,source;output_dir=joinpath(directory,name),deembedded=false)
                data=checked_native_touchstone(reference,joinpath(reference.output_dir,"native_raw.s2p");deembedded=false)
                matrix=only(data.s);push!(matrices,matrix)
                record[name*"_sha256"]=bytes2hex(sha256(read(source)))
                record[name*"_s_real"]=vec(real(matrix));record[name*"_s_imag"]=vec(imag(matrix))
                record[name*"_stdout"]=read(joinpath(reference.output_dir,"engine_stdout.log"),String)
            end
            record["full_s_error"]=maximum(abs,matrices[1]-matrices[2])
            record["bit_identical_s"]=matrices[1]==matrices[2]
            record["status"]=record["full_s_error"]<=1e-12 ? "PASS" : "FAIL"
            println(tag," full-S error=",record["full_s_error"]," bit-identical=",record["bit_identical_s"])
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Evidence directory: ",directory)
    end
end
main()
