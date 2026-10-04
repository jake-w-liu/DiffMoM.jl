using DiffMoM,Test,LinearAlgebra

# Independent terminal constitutive equations plus physical voltage sources.
# This oracle never calls production wave transforms or Y/Z conversion.
function network_complex_source_oracle(E,F,refs)
    n=length(refs);roots=sqrt.(real.(refs))
    source=vcat(hcat(E,F),hcat(Matrix{ComplexF64}(I,n,n),Diagonal(refs)))
    rhs=vcat(zeros(ComplexF64,n,n),Matrix(Diagonal(2roots)))
    vi=source\rhs;v=vi[1:n,:];current=vi[n+1:2n,:]
    return Diagonal(1 ./ (2roots))*(v-Diagonal(conj.(refs))*current)
end

@testset "network complex references: independent constitutive source oracle" begin
    for n in 1:5
        E=Matrix{ComplexF64}(I,n,n)
        Z=ComplexF64[(p==q ? 25+3p+im*(7p-4) : .5+im*.7/(p+q)) for p in 1:n,q in 1:n]
        old=ComplexF64[35+3p+im*(11p-7) for p in 1:n]
        new=ComplexF64[70-2p-im*(9p+4) for p in 1:n]
        S=network_complex_source_oracle(E,-Z,old)
        wanted=network_complex_source_oracle(E,-Z,new)
        @test planar_renormalize_s(S,old,new)≈wanted atol=2e-15 rtol=2e-14
        @test planar_renormalize_s(wanted,new,old)≈S atol=2e-15 rtol=2e-14
        @test opnorm(S)<=1+2e-14
        @test S≈transpose(S) atol=2e-14
        @test planar_renormalize_s(S,old,old)==S
        for (e,f) in ((zeros(ComplexF64,n,n),E),(E,zeros(ComplexF64,n,n)))
            exact=network_complex_source_oracle(e,f,old)
            @test planar_renormalize_s(exact,old,new)≈network_complex_source_oracle(e,f,new) atol=2e-15
        end
    end
    E=ComplexF64[1 -1;0 0];F=ComplexF64[0 0;1 1]
    old=ComplexF64[50+20im,75-10im];new=ComplexF64[32-17im,91+30im]
    S=network_complex_source_oracle(E,F,old)
    @test planar_renormalize_s(S,old,new)≈network_complex_source_oracle(E,F,new) atol=2e-15
    @test S'*S≈Matrix{ComplexF64}(I,2,2) atol=2e-15
    @test_throws ArgumentError planar_renormalize_s(S,[0+20im,50],new)
end

@testset "frequency-dependent databanks and fixed-reference file export" begin
    freqs=[1e9,2e9,4e9];Z=ComplexF64[42+8im 4-2im;4-2im 70-12im]
    E=Matrix{ComplexF64}(I,2,2);calls=Float64[]
    refs(f)=ComplexF64[35+f/1e9+im*(3+f/1e9),80-im*f/1e9]
    provider=f->(push!(calls,f);refs(f))
    samples=[network_complex_source_oracle(E,-Z,refs(f)) for f in freqs]
    @test_throws ArgumentError PlanarNetworkData(freqs,samples;z0=provider,max_bytes=1)
    @test isempty(calls)
    @test_throws ArgumentError PlanarNetworkData([big"1e1000"],[samples[1]];z0=provider)
    @test_throws ArgumentError PlanarNetworkData([1.],[fill(big"1e1000",2,2)];z0=provider)
    @test isempty(calls)
    data=PlanarNetworkData(freqs,samples;z0=provider)
    @test data.z0==refs(first(freqs))
    @test data.reference_series==refs.(freqs)
    @test eltype(data.z0)==ComplexF64
    @test calls==freqs
    for f in (freqs...,1.5e9,3e9)
        @test planar_network_response(data,f;z0=[50.,65.])≈network_complex_source_oracle(E,-Z,[50.,65.]) atol=3e-15
        @test planar_network_response(data,f)≈network_complex_source_oracle(E,-Z,refs(f)) atol=3e-15
    end
    knot=planar_network_response(data,first(freqs));knot[1,1]=99
    @test data.s[1][1,1]!=99
    stored=refs.(freqs);copydata=PlanarNetworkData(freqs,samples;reference_series=stored)
    stored[1][1]=999
    @test copydata.z0==refs(first(freqs)) && copydata.reference_series[1]==refs(first(freqs))
    @test_throws DimensionMismatch PlanarNetworkData(freqs,samples;reference_series=[refs(first(freqs))])
    @test_throws ArgumentError PlanarNetworkData(freqs,samples;z0=50.,reference_series=refs.(freqs))
    realdata=PlanarNetworkData(freqs,[zeros(2,2) for f in freqs])
    @test eltype(realdata.z0)==Float64 && realdata.reference_series===nothing
    wanted=PlanarNetworkData(freqs,[network_complex_source_oracle(E,-Z,[50.,65.]) for f in freqs];z0=[50.,65.])
    @test_throws ArgumentError planar_compare_sweeps(data,wanted)
    @test planar_compare_sweeps(data,wanted;renormalize=true,atol=1e-14).passed
    divergent=PlanarNetworkData(freqs,samples;reference_series=[refs(freqs[1]),refs(freqs[2]).+[0,1im],refs(freqs[3])])
    @test_throws ArgumentError planar_compare_sweeps(data,divergent)
    @test_throws ArgumentError plot_planar_sparams(data;reference=divergent)
    overlay=plot_planar_sparams(data;reference=wanted,renormalize=true,quantity=:real)
    @test overlay.data[1][:y]≈overlay.data[2][:y] atol=2e-15
    mktempdir() do dir
        for (writer,reader,name) in ((planar_write_touchstone,planar_read_touchstone,"network.s2p"),
                (planar_write_databank_csv,planar_read_sparam_csv,"network.csv"))
            path=joinpath(dir,name);writer(path,data)
            result=reader(path);fixed=real.(data.z0)
            @test result.z0==fixed && result.reference_series===nothing
            for S in result.s
                @test S≈network_complex_source_oracle(E,-Z,fixed) atol=3e-15
            end
            writer(path,data;z0=[50.,65.]);result=reader(path)
            @test result.z0==[50.,65.]
            @test planar_compare_sweeps(result,wanted;atol=1e-14).passed
            write(path,"sentinel")
            @test_throws ArgumentError writer(path,data;z0=[50+20im,65])
            @test read(path,String)=="sentinel"
            @test_throws ArgumentError writer(path,data;max_bytes=1)
            @test read(path,String)=="sentinel"
        end
        path=joinpath(dir,"array.s2p")
        planar_write_touchstone(path,freqs,samples;z0=provider,output_z0=[50.,65.])
        @test planar_compare_sweeps(planar_read_touchstone(path),wanted;atol=1e-14).passed
        csv=joinpath(dir,"array.csv")
        planar_write_databank_csv(csv,freqs,samples;z0=provider,output_z0=[50.,65.])
        @test planar_compare_sweeps(planar_read_sparam_csv(csv),wanted;atol=1e-14).passed
        empty!(calls)
        write(csv,"sentinel")
        @test_throws ArgumentError planar_write_databank_csv(csv,freqs,samples;z0=provider,max_bytes=1)
        @test isempty(calls) && read(csv,String)=="sentinel"
        report=joinpath(dir,"report.txt");planar_write_report(report,data)
        text=read(report,String)
        @test occursin("frequency-dependent Kurokawa",text)
        @test all(f->occursin(string(f)*" Hz",text),freqs)
    end
end

@testset "Touchstone export versions and legacy matrix lines" begin
    mktempdir() do dir
        for n in 1:9,version in ("1.0","1.1","2.0","2.1")
            freqs=[1e9,2e9]
            samples=[ComplexF64[(p+2q)/100+im*(3p-q+k)/100 for p in 1:n,q in 1:n] for k in 1:2]
            refs=version=="1.0" ? fill(75.,n) : [35.0+5p for p in 1:n]
            data=PlanarNetworkData(freqs,samples;z0=refs)
            path=joinpath(dir,"network.s$(n)p")
            planar_write_touchstone(path,data;version)
            parsed=planar_read_touchstone(path)
            @test parsed.frequencies==freqs
            @test parsed.z0==refs
            @test parsed.s==samples
            lines=filter(line->!isempty(line)&&!startswith(line,'!'),readlines(path))
            legacy=startswith(version,"1.")
            @test any(line->startswith(line,"[Version]"),lines)==!legacy
            @test any(line->line=="[End]",lines)==!legacy
            if legacy
                numeric=lines[2:end]
                @test all(line->length(split(line))<=9,numeric)
                if n==2
                    fields=parse.(Float64,split(first(numeric)))
                    @test fields[2:end]==reduce(vcat,([real(samples[1][p,q]),imag(samples[1][p,q])] for q in 1:2 for p in 1:2))
                elseif n>=3
                    chunks=cld(n,4)
                    @test length(numeric)==2n*chunks
                    values=Float64[]
                    for (i,line) in enumerate(numeric)
                        fields=parse.(Float64,split(line))
                        i in (1,n*chunks+1) && popfirst!(fields)
                        append!(values,fields)
                    end
                    expected=reduce(vcat,([real(samples[k][p,q]),imag(samples[k][p,q])] for k in 1:2 for p in 1:n for q in 1:n))
                    @test values==expected
                end
            end
        end
        Z=ComplexF64[40+3im 7-2im;5+im 80-4im]
        E=Matrix{ComplexF64}(I,2,2);old=ComplexF64[35+8im,85-7im]
        S=network_complex_source_oracle(E,-Z,old)
        data=PlanarNetworkData([1e9],[S];z0=old)
        path=joinpath(dir,"complex.s2p")
        for version in ("1.0","1.1","2.0","2.1")
            planar_write_touchstone(path,data;version,z0=60.)
            parsed=planar_read_touchstone(path)
            @test parsed.z0==[60.,60.]
            @test parsed.s[1]≈network_complex_source_oracle(E,-Z,[60.,60.]) atol=3e-15
        end
        write(path,"sentinel")
        @test_throws ArgumentError planar_write_touchstone(path,data;version="1.0")
        @test read(path,String)=="sentinel"
        @test_throws ArgumentError planar_write_touchstone(path,data;version="1.2")
        @test read(path,String)=="sentinel"
        @test_throws ArgumentError planar_write_touchstone(path,data;version="1.0",z0=60.,max_bytes=1)
        @test read(path,String)=="sentinel"
        calls=Ref(0)
        provider=f->begin calls[]+=1;old end
        @test_throws ArgumentError planar_write_touchstone(path,[1e9],[S];z0=provider,version="3.0")
        @test calls[]==0 && read(path,String)=="sentinel"
        planar_write_touchstone(path,[1e9],[S];z0=provider,output_z0=60.,version="1.0")
        @test planar_read_touchstone(path).s[1]≈network_complex_source_oracle(E,-Z,[60.,60.]) atol=3e-15
    end
end

@testset "actual Sonnet graph TERM/FTERM files retain their wave bases" begin
    dir=joinpath(@__DIR__,"..","validation","sonnet_stripline","native_reference")
    baseline=planar_read_touchstone(joinpath(dir,"baseline.s2p"))
    definitions=Dict("reactance"=>([50.,75.],[20.,-10.],[0.,0.],[0.,0.]),
        "inductance"=>([50.,75.],[0.,0.],[1e-9,.5e-9],[0.,0.]),
        "capacitance"=>([50.,75.],[0.,0.],[0.,0.],[2e-12,4e-12]),
        "combined"=>([50.,75.],[10.,-10.],[.25e-9,.5e-9],[2e-12,4e-12]))
    for (kind,(r,x,l,c)) in definitions
        native=planar_read_touchstone(joinpath(dir,"native_graph_$kind.s2p"))
        @test native.frequencies==baseline.frequencies
        @test (native.reference_series===nothing)==(kind=="reactance")
        for (k,f) in enumerate(native.frequencies)
            # Native parallel-C impedance, evaluated independently in SI units.
            z=1 ./ (1 ./ (r+im*x+im*2pi*f*l)+im*2pi*f*c)
            actual=native.reference_series===nothing ? native.z0 : native.reference_series[k]
            @test actual≈z rtol=3e-15
            @test maximum(abs,native.s[k]-planar_renormalize_s(baseline.s[k],baseline.z0,z))<1e-9
            @test planar_network_response(native,f;z0=50.)≈baseline.s[k] atol=1e-9
        end
        mktempdir() do outdir
            path=joinpath(outdir,"fixed.s2p");planar_write_touchstone(path,native;z0=50.)
            @test planar_compare_sweeps(planar_read_touchstone(path),baseline;atol=1e-9).passed
        end
    end
    mktempdir() do dir
        file=joinpath(dir,"malformed.s2p")
        header="# Hz S RI R 50\n";row="1 0 0 1 0 1 0 0 0\n"
        for bad in ("! TERM 50 20 75\n","! TERM 0 20 75 -10\n","! TERM NaN 20 75 -10\n",
                "! FTERM 50 0 1e-9 0 &\n","! FTERM 50 0 & 1e-9 0\n",
                "! FTERM 50 0 1e-9 0 &\n# Hz S RI\n",
                "! TERM 50 20 75 -10\n! TERM 50 0 50 0\n",
                "! TERM 50 20 75 -10\n! FTERM 50 0 0 0 75 0 0 0\n")
            write(file,header*bad*row)
            @test_throws ArgumentError planar_read_touchstone(file)
        end
        write(file,header*row*"! TERM 50 20 75 -10\n")
        @test_throws ArgumentError planar_read_touchstone(file)
        write(file,replace(header," S "=>" Y ")*"! TERM 50 20 75 -10\n"*row)
        @test_throws ArgumentError planar_read_touchstone(file)
        write(file,header*"! FTERM 50 10 1e-9 2e-12 &\n75 -10 .5e-9 4e-12\n"*replace(row,"1 "=>"0 ";count=1))
        dc=planar_read_touchstone(file)
        @test dc.reference_series==[ComplexF64[50+10im,75-10im]]
        @test_throws ArgumentError planar_read_touchstone(file;max_bytes=1)
        for n in (1,3,10)
            path=joinpath(dir,"wrapped.s$(n)p")
            rows=["! FTERM 50 10 1e-9 2e-12"* (n==1 ? "" : " &")]
            for p in 2:n
                push!(rows,"50 10 1e-9 2e-12"*(p==n ? "" : " &"))
            end
            values=join(fill("0 0",n*n)," ")
            write(path,header*join(rows,"\n")*"\n0 "*values*"\n2000000000 "*values*"\n")
            parsed=planar_read_touchstone(path)
            @test parsed.reference_series[1]==fill(50+10im,n)
            wanted=inv(inv(50+im*(10+2pi*2e9*1e-9))+im*2pi*2e9*2e-12)
            @test parsed.reference_series[2]≈fill(wanted,n) rtol=3e-15
            @test all(S->S==zeros(ComplexF64,n,n),parsed.s)
        end
    end
end

@testset "complex terminated impedance, Smith chart and voltage SWR" begin
    refs=ComplexF64[50+20im,70-15im];loads=conj.(refs)
    S=zeros(ComplexF64,2,2);data=PlanarNetworkData([1e9],[S];z0=refs)
    curves=planar_equation_curves(data)
    @test vec(curves.zin_ohm)≈loads atol=2e-15
    @test all(isinf,curves.reflection_db)
    expected=(loads-real.(refs))./(loads+real.(refs))
    @test vec(curves.swr)≈(1 .+abs.(expected))./(1 .-abs.(expected)) atol=2e-15
    chart=plot_planar_smith(data)
    for (p,trace) in enumerate(chart.data[end-1:end])
        @test trace[:x]≈[real(expected[p])] atol=2e-15
        @test trace[:y]≈[imag(expected[p])] atol=2e-15
    end
    @test_throws ArgumentError plot_planar_smith(data;z0=50+20im)
    shorts=PlanarNetworkData([1e9],[Matrix(Diagonal(-conj.(refs)./refs))];z0=refs)
    @test norm(planar_equation_curves(shorts).zin_ohm)<1e-14
    @test plot_planar_smith(shorts).data[end][:x]≈[-1.] atol=2e-15
    opens=PlanarNetworkData([1e9],[Matrix{ComplexF64}(I,2,2)];z0=refs)
    @test plot_planar_smith(opens).data[end][:x]==[1.]
end

@testset "layout ABS and radiation preserve physical complex-reference excitations" begin
    grid=CellGrid(.002,.002,4,4)
    stack=PlanarStackup([PlanarLayer(1.,1.,.001),PlanarLayer(1.,1.,.001)],TERM_GND,TERM_SPACE,grid.a,grid.b)
    shape=planar_transform(planar_line(length=grid.a,width=grid.b/2,level=1,metal="test_metal");offset=(0.,grid.b/2))
    calls=Float64[];law=f->(push!(calls,f);50+im*2pi*f*1e-9)
    layout=build_planar_layout(stack,grid,[shape],[(shape.pins[1],law),(shape.pins[2],75-10im)];metals=Dict("test_metal"=>.1))
    @test_throws ArgumentError planar_sweep_abs(layout,1e9,1.2e9;n_eval=8,max_points=3,max_bytes=1)
    @test isempty(calls)
    physical=build_planar_problem(stack,grid,layout.problem.sheets,
        [PlanarPort(p.level,p.wall,p.cells,law) for p in layout.problem.ports])
    for object in (layout,physical)
        for (lo,hi) in ((big"1e500",big"2e500"),(big"1e-500",big"2e-500"),
                (1e9,nextfloat(1e9)),(1e9,nextfloat(nextfloat(1e9))))
            @test_throws ArgumentError planar_sweep_abs(object,lo,hi;n_eval=17,max_points=3)
            @test isempty(calls)
        end
        @test_throws ArgumentError planar_sweep_abs(object,1e9,1.2e9;n_eval=17,max_points=3,max_bytes=1)
        @test isempty(calls)
    end
    sweep=planar_sweep_abs(layout,1e9,1.2e9;n_eval=8,max_points=3,rel_tol=1.,mx=9,my=11)
    @test sweep.z0==ComplexF64[50+im*2pi*1e9*1e-9,75-10im]
    for (f,S) in zip(sweep.freqs,sweep.s)
        solved=solve_planar(layout,f;mx=9,my=11)
        @test S≈planar_renormalize_s(solved.s,solved.z0,sweep.z0) atol=2e-13
    end
    solved=solve_planar(layout,1e9;mx=9,my=11)
    a=ComplexF64[.3+.2im,-.1im];b=solved.s*a
    v=(conj.(solved.z0).*a+solved.z0.*b)./sqrt.(real.(solved.z0))
    count=length(calls)
    pa=planar_farfield(solved;incident_waves=a,theta=[.3,.7],phi=[.2])
    pv=planar_farfield(solved;voltages=v,theta=[.3,.7],phi=[.2])
    @test pa.etheta≈pv.etheta atol=2e-13
    @test pa.ephi≈pv.ephi atol=2e-13
    @test pa.accepted_power≈.5*(sum(abs2,a)-sum(abs2,b)) atol=2e-13
    @test length(calls)==count
end
