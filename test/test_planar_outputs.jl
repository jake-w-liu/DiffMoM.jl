using DiffMoM,Test,LinearAlgebra

@testset "planar databanks: power-wave references and standards" begin
    Y=ComplexF64[.03+.01im -.007+.002im;-.007+.002im .02+.03im]
    old,new=[50.,75.],[25.,100.]
    S=planar_y_to_s(Y,old)
    @test planar_renormalize_s(S,old,new) ≈ planar_y_to_s(Y,new) rtol=1e-13
    @test planar_renormalize_s(planar_renormalize_s(S,old,new),new,old) ≈ S rtol=1e-13
    for A in (Matrix{ComplexF64}(I,2,2),-Matrix{ComplexF64}(I,2,2),ComplexF64[0 1;1 0])
        @test planar_renormalize_s(planar_renormalize_s(A,old,new),new,old) ≈ A atol=1e-14
    end
    @test_throws ArgumentError planar_renormalize_s(S,old,[0.,50.])
    @test_throws ArgumentError PlanarNetworkData([2.,1.],[S,S])
    @test_throws DimensionMismatch PlanarNetworkData([1.],[S,S])
    @test_throws ArgumentError PlanarNetworkData([1.],[S];port_names=["p","p"])
    @test_throws ArgumentError PlanarNetworkData([1.],[S];max_bytes=1)
    @test_throws ArgumentError PlanarNetworkData([big"1e1000"],[ones(ComplexF64,1,1)])
    @test_throws ArgumentError PlanarNetworkData([1.],[fill(big"1e1000",1,1)])
    d=PlanarNetworkData([1.,3.],[S,2S];z0=old)
    @test planar_network_response(d,2.) ≈ 1.5S
    knot=planar_network_response(d,1.);knot[1,1]=999
    @test d.s[1][1,1]==S[1,1]
    @test_throws ArgumentError planar_network_response(d,0.)
    @test_throws ArgumentError planar_compare_sweeps(d,PlanarNetworkData([1.,3.],[S,2S]);atol=.1)

    mktempdir() do dir
        for n in (1,2,3,10)
            matrix=ComplexF64[.001p+.002q+im*(.003p-.004q) for p in 1:n,q in 1:n]
            source=PlanarNetworkData([0.,1e9,2e9],[matrix,2matrix,3matrix];z0=collect(50.0:(50.0+n-1)))
            path=joinpath(dir,"roundtrip.s$(n)p")
            planar_write_touchstone(path,source)
            readback=planar_read_touchstone(path)
            @test readback.frequencies==source.frequencies
            @test readback.s==source.s
            @test readback.z0==source.z0
            @test_throws ArgumentError planar_read_touchstone(path;max_bytes=1)
            csv=joinpath(dir,"roundtrip.csv")
            planar_write_databank_csv(csv,source)
            fromcsv=planar_read_sparam_csv(csv)
            @test fromcsv.s==source.s
            @test fromcsv.z0==source.z0
            @test fromcsv.port_names==source.port_names
            @test_throws ArgumentError planar_read_sparam_csv(csv;max_bytes=1)
        end
        file=joinpath(dir,"test.s2p")
        function checkfile(text;np=2)
            write(file,text);return planar_read_touchstone(file;nports=np)
        end
        @test checkfile("# S R 50 MHz RI\n1 0 0 0.1 0.2 0.3 0.4 0.5 0.6\n").s[1]==
            ComplexF64[0 .3+.4im;.1+.2im .5+.6im]
        repeated=checkfile("# Hz S RI R 50\n# MHz S MA R 75\n1 0 0 1 0 1 0 0 0\n# GHz S DB R 100\n2 0 0 1 0 1 0 0 0\n")
        @test repeated.frequencies==[1.,2.] && repeated.z0==[50.,50.]
        @test checkfile("#\n1 0 0 0.5 90 0.25 -90 0 0\n").s[1] ≈
            ComplexF64[0 -.25im;.5im 0] atol=1e-15
        @test checkfile("# Hz S DB R 50\n1 -20 0 -6.020599913279624 90 -12.041199826559248 -90 -20 180\n").s[1] ≈
            ComplexF64[.1 -.25im;.5im -.1] atol=1e-15
        # Version 1 non-S numbers normalized to R; version 2 are SI.
        for (version,normalization) in ((1.,50.),(2.,1.))
            Yp=ComplexF64[.03+.01im -.004;-.002 .02+.03im]
            Zp=inv(Yp)
            for kind in ("Y","Z","H","G")
                raw=kind=="Y" ? Yp*normalization : kind=="Z" ? Zp/normalization :
                    kind=="H" ? ComplexF64[(Zp[1,1]-Zp[1,2]*Zp[2,1]/Zp[2,2])/normalization Zp[1,2]/Zp[2,2];
                        -Zp[2,1]/Zp[2,2] normalization/Zp[2,2]] :
                    ComplexF64[normalization/Zp[1,1] -Zp[1,2]/Zp[1,1];
                        Zp[2,1]/Zp[1,1] (Zp[2,2]-Zp[2,1]*Zp[1,2]/Zp[1,1])/normalization]
                header=version==1 ? "# Hz $kind RI R 50\n" :
                    "[Version] 2.1\n# Hz $kind RI R 50\n[Number of Ports] 2\n[Two-Port Data Order] 21_12\n[Number of Frequencies] 1\n[Network Data]\n"
                text=header*"1 "*join([v for q in 1:2 for p in 1:2 for v in (real(raw[p,q]),imag(raw[p,q]))]," ")*"\n"*
                    (version==1 ? "" : "[End]\n")
                @test checkfile(text).s[1] ≈ planar_y_to_s(Yp,[50.,50.]) rtol=2e-13
            end
        end
        for format in ("Upper","Lower")
            text="[Version] 2.0\n# Hz S RI R 50\n[Number of Ports] 2\n[Two-Port Data Order] 12_21\n[Number of Frequencies] 1\n[Matrix Format] $format\n[Reference]\n50\n75\n[Network Data]\n1 .1 .2 .3 .4 .5 .6\n[End]\n"
            @test checkfile(text).s[1]==ComplexF64[.1+.2im .3+.4im;.3+.4im .5+.6im]
        end
        # Mixed-mode transforms are checked against independent wave matrices.
        U=[1/sqrt(2) -1/sqrt(2) 0;0 0 1;1/sqrt(2) 1/sqrt(2) 0]
        P=[1. -1. 0;0 0 1;.5 .5 0];Q=inv(transpose(P))
        refs=[50.,50.,75.];Y3=ComplexF64[.02 -.004 .001;-.004 .03 .002;.001 .002 .01]
        Z3=inv(Y3);S3=planar_y_to_s(Y3,refs)
        for kind in ("S","Y","Z")
            M=kind=="S" ? U*S3*transpose(U) : kind=="Y" ? Q*Y3/P : P*Z3/Q
            path=joinpath(dir,"mixed.s3p")
            text="[Version] 2.1\n# Hz $kind RI R 50\n[Number of Ports] 3\n[Number of Frequencies] 1\n[Reference] 50 50 75\n[Mixed-Mode Order] D1,2 S3 C1,2\n[Network Data]\n1 "*
                join([v for p in 1:3 for q in 1:3 for v in (real(M[p,q]),imag(M[p,q]))]," ")*"\n[End]\n"
            write(path,text)
            @test planar_read_touchstone(path).s[1] ≈ S3 rtol=1e-13
        end
        text="[Version] 2.1\n# Hz S RI\n[Number of Ports] 2\n[Two-Port Data Order] 21_12\n[Number of Frequencies] 1\n[Number of Noise Frequencies] 1\n[Begin Information]\n[arbitrary user metadata]\n[End Information]\n[Network Data]\n1 0 0 1 0 1 0 0 0\n[Noise Data]\n1 0 0 0 0\n[End]\n"
        @test checkfile(text).s[1]==ComplexF64[0 1;1 0]
        for bad in ("# Hz S RI\n1 0 0\n","# Hz S RI\n1 0 0 1 0 1 0 0 0\n1 0 0 1 0 1 0 0 0\n",
                replace(text,"[End]"=>"[Unsupported]"),replace(text,"[Number of Frequencies] 1"=>"[Number of Frequencies] 2"),
                replace(text,"[Reference]"=>"[BadReference]"))
            # Last replacement leaves text unchanged, and is skipped.
            bad==text && continue
            write(file,bad);@test_throws ArgumentError planar_read_touchstone(file)
        end
        csv=joinpath(dir,"bad.csv")
        damaged=PlanarNetworkData([1.],[S]);pop!(damaged.port_names)
        write(csv,"sentinel")
        @test_throws ArgumentError planar_write_databank_csv(csv,damaged)
        @test read(csv,String)=="sentinel"
        named=PlanarNetworkData([1.],[S];port_names=["signal, a","return \"b\""])
        planar_write_databank_csv(csv,named)
        @test planar_read_sparam_csv(csv).port_names==named.port_names
        write(csv,"frequency_hz,output_port,input_port,s_real,s_imag\n1,p1,p1,0,0\n1,p2,p2,0,0\n")
        @test_throws ArgumentError planar_read_sparam_csv(csv)
        write(csv,"frequency_hz,output_port,input_port,s_real,s_imag\n1,p1,p1,0,0\n1,p1,p1,0,0\n")
        @test_throws ArgumentError planar_read_sparam_csv(csv)
        # Measured S block follows the same circuit wave equations as raw EM.
        thru=PlanarNetworkData([1.,2.],[ComplexF64[0 1;1 0],ComplexF64[0 1;1 0]])
        circuit=PlanarCircuit(2,[1,2]);circuit_add_network!(circuit,[1,2],f->planar_network_response(thru,f))
        @test solve_planar_circuit(circuit,1.5).s ≈ thru.s[1] atol=1e-14
    end
end

@testset "planar response curves and verified report scope" begin
    freqs=[1.,2.,4.,7.];tau=.01
    mats=[ComplexF64[0 cis(-2pi*f*tau);cis(-2pi*f*tau) 0] for f in freqs]
    data=PlanarNetworkData(freqs,mats)
    curves=planar_equation_curves(data)
    @test curves.group_delay_s ≈ fill(tau,4) atol=1e-15
    @test curves.zin_ohm==fill(50.0+0im,4,2)
    @test all(isinf,curves.reflection_db)
    @test all(==(1.),curves.swr)
    cubic=[ComplexF64[0 cis(-.001*f^2);cis(-.001*f^2) 0] for f in freqs]
    @test planar_equation_curves(PlanarNetworkData(freqs,cubic)).group_delay_s ≈ 0.001 .*freqs./pi atol=1e-15
    # Exact open/short, active SWR and the notch's undefined phase.
    open=planar_equation_curves(PlanarNetworkData([0.],[ones(ComplexF64,1,1)]))
    @test real(open.zin_ohm[1])==Inf && imag(open.zin_ohm[1])==0
    @test isnan(open.l_eff_h[1]) && isinf(open.swr[1])
    active=planar_equation_curves(PlanarNetworkData([1.],[fill(2.0+0im,1,1)]))
    @test isnan(active.swr[1])
    notch=deepcopy(mats);notch[2][2,1]=0
    @test all(isnan,planar_equation_curves(PlanarNetworkData(freqs,notch)).group_delay_s)
    altered=PlanarNetworkData(freqs,[S.+.01 for S in mats])
    comparison=planar_compare_sweeps(altered,data;atol=.011)
    @test comparison.passed && comparison.max_absolute_error≈.01
    @test !planar_compare_sweeps(altered,data;atol=.009).passed
    coarse=PlanarNetworkData(freqs,[S.+.1 for S in mats])
    @test planar_convergence_certificate([coarse,altered,data];atol=.011).verified
    @test !planar_convergence_certificate([coarse,data];atol=.2).verified
    @test !planar_convergence_certificate([altered,data,coarse];atol=.2).verified
    # Independent analytic lossless air transmission-line reference.
    len=.001;f=10e9;theta=2pi*f*len/299792458.;zc=50.
    Y=ComplexF64[-im*cot(theta)/zc im*csc(theta)/zc;im*csc(theta)/zc -im*cot(theta)/zc]
    benchmark=planar_stripline_benchmark(Y,len,f)
    @test benchmark.passed && benchmark.absolute_error<1e-13
    @test_throws ArgumentError planar_stripline_benchmark(Y,0.,f)
    model=PlanarRationalModel(ComplexF64[],Matrix{ComplexF64}[],reshape([.02],1,1),
        zeros(1,1),[1e6,2e6],0.,0.,true,true,:positive_residues,.02,0.,0.)
    @test abs(only(planar_dc_sparams(model)))<1e-15
    bad=PlanarRationalModel(ComplexF64[],Matrix{ComplexF64}[],reshape([-.02],1,1),
        zeros(1,1),[1e6,2e6],0.,0.,false,false,:none,-.02,0.,0.)
    @test_throws ArgumentError planar_dc_sparams(bad)
    mktempdir() do dir
        curvespath=joinpath(dir,"curves.csv");reportpath=joinpath(dir,"report.txt")
        planar_write_equation_curves(curvespath,data)
        @test length(readlines(curvespath))==5
        planar_write_report(reportpath,data;name="analytic line",accuracy=benchmark)
        @test occursin("finite response samples",read(reportpath,String))
        @test occursin("Absolute accuracy",read(reportpath,String))
    end
end
