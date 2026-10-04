using DiffMoM,Test,TOML,SHA,Dates,LinearAlgebra
include("sonnet_reference.jl")
using .SonnetReference

function independent_ladder(f,centers,resistance,delay)
    om=2pi*f;chain=Matrix{ComplexF64}(I,2,2)
    for center in centers
        L=1e-9;C=1/((2pi*center)^2*L)
        Y=1/(resistance+im*(om*L-1/(om*C)))
        theta=om*delay
        chain=chain*ComplexF64[cos(theta) 50im*sin(theta);im*sin(theta)/50 cos(theta)]*
            ComplexF64[1 0;Y 1]
    end
    A,B,C,D=chain[1,1],chain[1,2],chain[2,1],chain[2,2]
    den=A+B/50+C*50+D
    return ComplexF64[(A+B/50-C*50-D)/den 2*(A*D-B*C)/den;
        2/den (-A+B/50-C*50+D)/den]
end

function write_native_ladder(path,centers,resistance,delay)
    open(path,"w") do io
        println(io,"FTYP SONNETPRJ 19\nVER \"18.53\"\nDIM\nANG DEG\nCAP PF\nCON /OH\nFREQ GHZ\nIND NH\nLNG MM\nRES OH\nEND DIM\nCKT")
        if iszero(delay)
            # A zero-delay through path has a literal zero resistance.
            println(io,"RES 1 2 R=0")
        end
        for (i,center) in enumerate(centers)
            positive=iszero(delay) ? 2 : i+1
            !iszero(delay) && println(io,"S2P ",i," ",i+1," delay.s2p")
            internal1=length(centers)+2i+1;internal2=internal1+1
            println(io,"RES ",positive," ",internal1," R=",resistance)
            println(io,"IND ",internal1," ",internal2," L=1")
            println(io,"CAP ",internal2," 0 C=",1e12/((2pi*center)^2*1e-9))
        end
        output=iszero(delay) ? 2 : length(centers)+1
        println(io,"DEF2P 1 ",output," RESONANT R 50\nEND CKT\nCONTROL\nVARSWP\nOPTIONS\nSPEED 0\nEND CONTROL\nVARSWP\nENABLED Y\nFREQ Y AN SWEEP 1 6 0.05\nEND\nEND VARSWP")
        println(io,"FILEOUT\nTOUCH ND Y native_resonant.s2p IC 15 S RI R 50\nFOLDER .\nEND FILEOUT")
    end
end

function main()
    em=find_em();em===nothing && error("actual Sonnet engine required")
    directory=evidence_directory("abs_resonance");println("Retained native evidence: ",directory)
    fs=collect(range(1e9,6e9;length=1001));rows=Dict{String,Any}()
    root=normpath(joinpath(@__DIR__,"..",".."))
    source_paths=("src/planar/PlanarSweep.jl","src/planar/PlanarSonnetIO.jl","src/planar/PlanarCircuit.jl",
        "src/planar/PlanarNetworkIO.jl","validation/sonnet_stripline/sonnet_reference.jl",
        "validation/sonnet_stripline/validate_sonnet_abs_resonance.jl")
    source_hashes()=Dict(p=>open(io->bytes2hex(sha256(io)),joinpath(root,p)) for p in source_paths)
    rows["source_sha256_before"]=source_hashes()
    try
    @testset "Actual Sonnet resonant circuit and adaptive sweep" begin
        for (name,centers,resistance,delay) in (("three_with_delay",[2.2e9,3.7e9,5.1e9],.2,2e-11),
                ("deep_notch_with_delay",[3.7e9],1e-4,2e-11))
            path=joinpath(directory,name);mkpath(path)
            source=joinpath(path,"source.son");write_native_ladder(source,centers,resistance,delay)
            dependencies=String[]
            if !iszero(delay)
                network=joinpath(path,"delay.s2p")
                matrices=[ComplexF64[0 exp(-2pi*im*f*delay);exp(-2pi*im*f*delay) 0] for f in fs]
                planar_write_touchstone(network,PlanarNetworkData(fs,matrices;z0=50.);version="1.0")
                push!(dependencies,network)
            end
            ref=reference_run(em,source;output_dir=joinpath(path,"actual"),deembedded=false,dependencies)
            project=read_sonnet_project(source);circuit=sonnet_planar_circuit(project)
            sweep=planar_sweep_abs(f->solve_planar_circuit(circuit,f).s,1e9,6e9;
                nports=2,n_eval=1001,max_points=64,rel_tol=1e-5,max_bytes=8*1024^2)
            @test sweep.converged
            maximum_round_error=0.;maximum_abs_error=0.;maximum_sweep_error=0.
            for (tokens,native) in zip(ref.row_tokens,ref.rows)
                f=native[1]*1e9;expected=twoport_matrix(native)
                truth=independent_ladder(f,centers,resistance,delay)
                # Bound comes from printed fields, before inspecting error.
                bound=twoport_quantization_bound(tokens).+1e-9
                error=abs.(expected-truth)
                @test all(error.<=bound)
                maximum_round_error=max(maximum_round_error,maximum(error./bound))
                maximum_abs_error=max(maximum_abs_error,maximum(error))
                k=argmin(abs.(sweep.dense_freqs.-f))
                @test abs(sweep.dense_freqs[k]-f)<1e-5
                sweep_error=maximum(abs,sweep.dense_s[k]-truth)
                @test sweep_error<1e-4
                maximum_sweep_error=max(maximum_sweep_error,sweep_error)
            end
            # Preserve native high-precision exports if the installed
            # circuit engine supplies them; log rounding remains separate.
            exports=String[]
            for (folder,_,files) in walkdir(ref.output_dir),file in files
                lowercase(file)=="native_resonant.s2p" && push!(exports,joinpath(folder,file))
            end
            export_errors=Float64[]
            for file in exports
                data=checked_native_touchstone(ref,file;deembedded=false)
                push!(export_errors,maximum(maximum(abs,data.s[k]-independent_ladder(data.frequencies[k],centers,resistance,delay)) for k in eachindex(data.frequencies)))
                @test last(export_errors)<1e-9
                for k in eachindex(data.frequencies)
                    candidate=argmin(abs.(sweep.dense_freqs.-data.frequencies[k]))
                    @test abs(sweep.dense_freqs[candidate]-data.frequencies[k])<1e-4
                    @test maximum(abs,sweep.dense_s[candidate]-data.s[k])<1e-4
                end
                if resistance==1e-4
                    k=argmin(abs.(data.frequencies.-3.7e9))
                    depth=20log10(abs(data.s[k][2,1]))
                    truth_depth=20log10(abs(independent_ladder(data.frequencies[k],centers,resistance,delay)[2,1]))
                    @test depth< -100
                    @test abs(depth-truth_depth)<1e-5
                end
            end
            rows[name]=Dict("native_points"=>length(ref.rows),"adaptive_points"=>length(sweep.freqs),
                "converged"=>sweep.converged,"max_native_log_full_s"=>maximum_abs_error,
                "max_native_error_to_printed_bound"=>maximum_round_error,"max_adaptive_full_s"=>maximum_sweep_error,
                "native_high_precision_exports"=>[relpath(p,directory) for p in exports],
                "max_native_high_precision_full_s"=>export_errors)
            println(name," => ",rows[name])
        end
    end
    finally
    rows["scope"]="actual native circuit engine with supplied RLC and matched Touchstone-delay fixtures; declared 1-6GHz points, excludes physical EM/measured RFIC resonance acceptance and hidden-resonance guarantee"
    rows["utc"]=string(now(UTC))
    rows["source_sha256_after"]=source_hashes()
    rows["source_unchanged"]=rows["source_sha256_before"]==rows["source_sha256_after"]
    open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,rows);end
    end
    @test rows["source_unchanged"]
end
main()
