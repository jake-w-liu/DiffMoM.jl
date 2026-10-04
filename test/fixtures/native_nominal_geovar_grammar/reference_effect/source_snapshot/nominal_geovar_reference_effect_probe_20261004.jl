using DiffMoM,LinearAlgebra,SHA,TOML
const repo=normpath(joinpath(@__DIR__,".."))
include(joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference

function main()
    inputs=joinpath(repo,"test/fixtures/native_nominal_geovar_grammar")
    directory=mktempdir(joinpath(repo,"data/sonnet_validation");prefix="nominal_reference_effect_",cleanup=false)
    sources=[joinpath(dir,file) for (dir,_,files) in walkdir(joinpath(repo,"src")) for file in files if endswith(file,".jl")]
    push!(sources,@__FILE__)
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in sources)
    report=Dict{String,Any}("scope"=>"fresh native repeat and independent saved-coordinate literal control for out-of-range nominal point; declared current full-S gate .005 and original voltage residual 1e-9","source_before"=>hashes(),"cases"=>Any[])
    try
        for name in ("valid","unknown_polygon","unknown_point","literal")
            text=read(joinpath(inputs,(name=="literal" ? "valid" : name)*".son"),String)
            if name=="literal"
                first=findfirst("VALVAR Width",text);last=findfirst("POR1 BOX",text)
                text=text[1:first.start-1]*text[last.start:end]
            end
            source=joinpath(directory,name*".son");write(source,text)
            native=reference_run(find_em(),source;output_dir=joinpath(directory,name),deembedded=false)
            reference=checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false)
            p=read_sonnet_project(source)
            result=solve_sonnet_project(p,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
            raw=result.raw;rhs=zeros(ComplexF64,size(raw.currents))
            for b in eachindex(raw.problem.basis.port)
                port=raw.problem.basis.port[b];iszero(port) && continue
                rhs[b,port]=(port==1 ? -1. : 1.)*raw.problem.basis.width[b]
            end
            row=Dict{String,Any}("case"=>name,"current_full_s_error"=>maximum(abs,result.s-only(reference.s)),
                "original_voltage_residual"=>norm(raw.z_mom*raw.currents-rhs)/norm(rhs),
                "native_s_real"=>vec(real(only(reference.s))),"native_s_imag"=>vec(imag(only(reference.s))),
                "current_s_real"=>vec(real(result.s)),"current_s_imag"=>vec(imag(result.s)))
            push!(report["cases"],row);println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained evidence: ",directory)
    end
    @assert report["source_unchanged"]
end
main()
