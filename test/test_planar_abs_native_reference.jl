using DiffMoM,Test,LinearAlgebra,SHA,TOML

function _native_abs_analytic(f,centers,resistance,delay)
    omega=2pi*f;chain=Matrix{ComplexF64}(I,2,2)
    for center in centers
        inductance=1e-9;capacitance=1/((2pi*center)^2*inductance)
        admittance=1/(resistance+im*(omega*inductance-1/(omega*capacitance)))
        theta=omega*delay
        chain=chain*ComplexF64[cos(theta) 50im*sin(theta);im*sin(theta)/50 cos(theta)]*
            ComplexF64[1 0;admittance 1]
    end
    A,B,C,D=chain[1,1],chain[1,2],chain[2,1],chain[2,2]
    denominator=A+B/50+C*50+D
    ComplexF64[(A+B/50-C*50-D)/denominator 2*(A*D-B*C)/denominator;
        2/denominator (-A+B/50-C*50+D)/denominator]
end

@testset "adaptive sweep vs archived actual Sonnet full matrices" begin
    root=normpath(joinpath(@__DIR__,"..","validation","sonnet_stripline","abs_reference"))
    hashes=TOML.parsefile(joinpath(root,"sha256.toml"))["sha256"]
    @test length(hashes)==30
    for (path,wanted) in hashes
        actual=open(io->bytes2hex(sha256(io)),joinpath(root,path))
        @test actual==wanted
    end
    evidence=TOML.parsefile(joinpath(root,"comparison.toml"))
    @test evidence["source_unchanged"]
    @test evidence["source_sha256_before"]==evidence["source_sha256_after"]
    for (name,centers,resistance,delay) in (("three_with_delay",[2.2e9,3.7e9,5.1e9],.2,2e-11),
            ("deep_notch_with_delay",[3.7e9],1e-4,2e-11))
        folder=joinpath(root,name);source=joinpath(folder,"source.son")
        metadata=TOML.parsefile(joinpath(folder,"actual","metadata.toml"))
        @test startswith(metadata["engine_version"],"18.53-Lite")
        @test metadata["process_success"]
        @test metadata["deembedded"]==false
        @test metadata["response_nports"]==2
        @test metadata["source_sha256"]==open(io->bytes2hex(sha256(io)),source)
        for (file,wanted) in metadata["dependency_sha256"]
            @test open(io->bytes2hex(sha256(io)),joinpath(folder,file))==wanted
        end
        native=planar_read_touchstone(joinpath(folder,"actual","native_resonant.s2p"))
        @test length(native.frequencies)==101
        @test native.z0==[50.,50.]
        circuit=sonnet_planar_circuit(read_sonnet_project(source))
        @test circuit.nnodes==3length(centers)+1
        sweep=planar_sweep_abs(f->solve_planar_circuit(circuit,f).s,1e9,6e9;
            nports=2,n_eval=1001,max_points=64,rel_tol=1e-5,max_bytes=8*1024^2)
        @test sweep.converged
        @test length(sweep.freqs)==evidence[name]["adaptive_points"]
        for k in eachindex(native.frequencies)
            f=native.frequencies[k];truth=_native_abs_analytic(f,centers,resistance,delay)
            @test maximum(abs,native.s[k]-truth)<1e-9
            solved=solve_planar_circuit(circuit,f)
            @test maximum(abs,solved.s-truth)<1e-9
            @test maximum(abs,solved.s-native.s[k])<1e-9
            candidate=argmin(abs.(sweep.dense_freqs.-f))
            @test abs(sweep.dense_freqs[candidate]-f)<1e-4
            @test maximum(abs,sweep.dense_s[candidate]-native.s[k])<1e-4
        end
        if resistance==1e-4
            k=argmin(abs.(native.frequencies.-3.7e9))
            depth=20log10(abs(native.s[k][2,1]))
            truth_depth=20log10(abs(_native_abs_analytic(native.frequencies[k],centers,resistance,delay)[2,1]))
            @test depth< -100
            @test abs(depth-truth_depth)<1e-5
        end
    end
end
