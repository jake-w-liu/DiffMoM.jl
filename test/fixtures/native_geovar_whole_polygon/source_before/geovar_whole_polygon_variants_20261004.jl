using DiffMoM,SHA,TOML,LinearAlgebra
const repo=normpath(joinpath(@__DIR__,".."))
const producer=joinpath(@__DIR__,"geovar_whole_polygon_probe_20261004.jl")
const loaded_source=replace(replace(replace(read(producer,String),r"\nmain\(\)\s*$"=>"\n"),
    "function whole_source(mode,hypothesis=\"parameter\")"=>"function whole_source(mode,hypothesis=\"parameter\";target=.1875)"),
    "nominal=.125;target=.1875;anchor"=>"nominal=.125;anchor")
include_string(Main,loaded_source,producer)
function orient(text,axis,direction)
    rows=split(text,'\n');num=findfirst(==("NUM 3"),rows)
    transform(x,y)=axis=="XDIR" ? (direction<0 ? 1-y : y,x) : (x,direction<0 ? 1-y : y)
    i=num+1
    for _ in 1:3
        count=parse(Int,split(rows[i])[2]);i+=1
        for _ in 1:count
            x,y=parse.(Float64,split(rows[i]));a,b=transform(x,y);rows[i]="$a $b";i+=1
        end
        @assert rows[i]=="END";i+=1
    end
    for i in eachindex(rows)
        rows[i]=="POR1 BOX" || continue
        fields=split(rows[i+3]);a,b=transform(parse(Float64,fields[6]),parse(Float64,fields[7]))
        fields[6]=string(a);fields[7]=string(b);rows[i+3]=join(fields,' ')
    end
    return replace(join(rows,'\n'),"YDIR 1"=>"$axis $direction")
end
function variants_main()
    directory=mktempdir(joinpath(repo,"data/sonnet_validation");prefix="geovar_whole_polygon_variants_",cleanup=false)
    write(joinpath(directory,"loaded_source.jl"),loaded_source)
    paths=[@__FILE__,producer,template,joinpath(repo,"src/planar/PlanarSonnetGeometryVariables.jl"),joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")]
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in paths)
    report=Dict{String,Any}("scope"=>"whole-polygon zero count and explicit logical-vertex selectors versus independent sequential literals; ANC NSCD/RAD, axes, directions, contraction/expansion; native full-S1e-12, public full-S .005 and voltage residual1e-9 at mx=my128", "source_before"=>hashes(),"cases"=>Any[])
    try
        for mode in ("NSCD","RAD"),axis in ("XDIR","YDIR"),direction in (-1,1),target in (.09375,.1875)
            tag=lowercase(mode)*"_"*lowercase(axis)*"_"*(direction<0 ? "negative" : "positive")*"_"*(target<.125 ? "contract" : "expand")
            matrices=Dict{String,Matrix{ComplexF64}}();projects=Dict{String,SonnetProject}()
            row=Dict{String,Any}("tag"=>tag,"mode"=>mode,"axis"=>axis,"direction"=>direction,"target"=>target);push!(report["cases"],row)
            for suffix in ("parameter","explicit","literal")
                text=whole_source(mode,suffix=="literal" ? "double_reference" : "parameter";target)
                suffix=="explicit" && (text=replace(text,"POLY 2 0\n"=>"POLY 2 4\n0\n1\n2\n3\n"))
                source=joinpath(directory,tag*"_"*suffix*".son");write(source,orient(text,axis,direction))
                native=reference_run(find_em(),source;output_dir=joinpath(directory,tag*"_"*suffix),deembedded=false)
                matrices[suffix]=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
                projects[suffix]=read_sonnet_project(source)
            end
            row["native_literal_full_s_error"]=maximum(abs,matrices["parameter"]-matrices["literal"])
            row["native_explicit_full_s_error"]=maximum(abs,matrices["parameter"]-matrices["explicit"])
            row["all_bit_identical"]=matrices["parameter"]==matrices["literal"]==matrices["explicit"]
            for suffix in ("parameter","literal")
                result=solve_sonnet_project(projects[suffix],1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
                raw=result.raw;rhs=zeros(ComplexF64,size(raw.currents))
                for b in eachindex(raw.problem.basis.port)
                    port=raw.problem.basis.port[b];iszero(port) && continue
                    rhs[b,port]=(port==1 ? -1. : 1.)*raw.problem.basis.width[b]
                end
                row[suffix*"_full_s_error"]=maximum(abs,result.s-matrices["parameter"])
                row[suffix*"_voltage_residual"]=norm(raw.z_mom*raw.currents-rhs)/norm(rhs)
            end
            row["status"]=row["native_literal_full_s_error"]<=1e-12 && row["native_explicit_full_s_error"]<=1e-12 &&
                all(row[s*"_full_s_error"]<=.005 && row[s*"_voltage_residual"]<=1e-9 for s in ("parameter","literal")) ? "PASS" : "FAIL"
            println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained whole-polygon variants: ",directory)
    end
    @assert report["source_unchanged"] && length(report["cases"])==16 && all(r->r["status"]=="PASS",report["cases"])
end
variants_main()
