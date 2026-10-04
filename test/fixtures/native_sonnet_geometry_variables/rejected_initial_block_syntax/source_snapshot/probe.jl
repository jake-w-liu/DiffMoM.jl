# Read-only production audit: independently declared native GEOVAR/literal pairs.
using DiffMoM, SHA, TOML, Dates
include(joinpath(@__DIR__, "../../sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference

function geovar_source(kind, sign, width; literal=false)
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
GEOVAR Width $(kind) YDIR $(sign) NSCD
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
END
"""
    end
    return """
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
        for (kind,sign,width) in (("ANC",1,.375),("ANC",-1,.375),("SYM",1,.375),("SYM",-1,.375))
            tag="$(lowercase(kind))_$(sign==1 ? "positive" : "negative")";matrices=Matrix{ComplexF64}[]
            record=Dict{String,Any}("kind"=>kind,"direction"=>sign,"width_mm"=>width,"status"=>"UNVERIFIED")
            push!(report["cases"],record)
            for literal in (false,true)
                name=tag*(literal ? "_literal" : "_parameter")
                source=joinpath(directory,name*".son");write(source,geovar_source(kind,sign,width;literal))
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
