module ExactStorageTests
using DiffMoM,LinearAlgebra,Test,TOML

function determinant_reference(A)
    n=size(A,1)
    n==0 && return Rational{BigInt}(1)
    n==1 && return only(A)
    result=Rational{BigInt}(0)
    for column in 1:n
        rows=collect(2:n);columns=[j for j in 1:n if j!=column]
        result+=(isodd(column) ? 1 : -1)*A[1,column]*determinant_reference(A[rows,columns])
    end
    result
end

function psd_reference(A)
    values=Rational{BigInt}.(A);n=size(A,1)
    # Every principal minor is nonnegative iff a real symmetric matrix is
    # PSD. Recursive Laplace expansion is independent of Bareiss elimination.
    for mask in 1:(2^n-1)
        indices=[i for i in 1:n if !iszero(mask & (1<<(i-1)))]
        determinant_reference(values[indices,indices])>=0 || return false
    end
    true
end

function model(d,e=zeros(size(d));residue=nothing)
    poles=residue===nothing ? ComplexF64[] : ComplexF64[-1.]
    residues=residue===nothing ? Matrix{ComplexF64}[] : [ComplexF64.(residue)]
    PlanarRationalModel(poles,residues,copy(d),copy(e),[0.,1.],
        0.,0.,false,false,:none,NaN,0.,0.)
end

function exercise(root,output)
    @assert realpath(dirname(dirname(pathof(DiffMoM))))==realpath(root)
    @assert output===nothing || !isfile(output)
    matrices=Matrix{Float64}[]
    for n in 1:4
        B=[Float64(i==j || i+1==j) for i in 1:n,j in 1:n]
        append!(matrices,[Matrix{Float64}(I,n,n),ones(n,n),B'*B,
            Float64.([(-1)^(i+j) for i in 1:n,j in 1:n])])
    end
    append!(matrices,[[1. prevfloat(1.);prevfloat(1.) 1.],
        [1. nextfloat(1.);nextfloat(1.) 1.],
        [1. 1. 1.;1. 1. -1.;1. -1. 1.],
        [0. 0. 0.;0. 1. 1.;0. 1. 1.]])
    for large in (1.,floatmax(Float64)),small in (nextfloat(0.),floatmin(Float64)),off in (small,2small)
        push!(matrices,[small off 0.;off small 0.;0. 0. large])
    end
    result=@testset "Exact rational certificates retain multiscale and singular stored energy" begin
        tiny=nextfloat(0.)
        for Y in ([0. tiny;0. 1.],[0. tiny*im;0. 1.],
                ComplexF64[1. 1im;-1im 1.],ComplexF64[1. 1im;-1im prevfloat(1.)],
                [1. floatmax(Float64);-floatmax(Float64) 1.])
            saved=copy(Y);Q=Complex{Rational{BigInt}}.(Y);H=Q+Q'
            reference=[real.(H) -imag.(H);imag.(H) real.(H)]
            @test DiffMoM._vf_psd(Y,0.;hermitian=true)==psd_reference(reference)
            @test Y==saved
        end
        for A in ([1. .5;.5 1.],ones(2,2),[1. tiny;tiny 1.]),shift in (0.,.5,1.,nextfloat(1.))
            Q=Rational{BigInt}.(A)-Rational{BigInt}(shift)*Matrix{Rational{BigInt}}(I,2,2)
            @test DiffMoM._vf_psd(A,0.;diagonal_shift=shift)==psd_reference(Q)
            @test DiffMoM._vf_psd(A,0.;hermitian=true,diagonal_shift=shift)==psd_reference(Q)
        end
        for residue in (nothing,zeros(2,2))
            active=model([0. tiny;0. 1.];residue)
            @test !planar_rational_certificate(active).certified
            @test !planar_rational_passivity(active,[0.];tol=0.).passive
        end
        hidden=model(Matrix{Float64}(I,2,2);residue=[-1. -tiny;-tiny -1.])
        @test !planar_rational_certificate(hidden).certified
        @test !planar_rational_passivity(hidden,[0.];tol=0.).passive
        @test DiffMoM._vf_frobenius_bound(hidden;max_bytes=DiffMoM._default_max_dense_payload_bytes())>1.
        @test sum(Rational{BigInt}(real(x)) for x in planar_rational_eval(hidden,0.))<0
        for scale in (floatmin(Float64),1.,floatmax(Float64)/4)
            valid=PlanarRationalModel(ComplexF64[-1.,-2.],
                [fill(ComplexF64(scale),1,1),fill(ComplexF64(-2scale),1,1)],
                fill(scale/2,1,1),zeros(1,1),[0.,1.],0.,0.,false,false,:none,NaN,0.,0.)
            c=planar_rational_certificate(valid)
            @test c.certified && c.method==:hamiltonian
        end
        # Build the positive-real storage inequality independently with
        # exact rationals and matrix products. This oracle uses no integer
        # lattice builder, shared GMP scratch, or fraction-free pivoting.
        X=Rational{BigInt}
        for poles in (ComplexF64[-1.],ComplexF64[-1.0+2im,-1.0-2im]),
                source in (.5+0im,-.5+.25im),feedthrough in (.25,1.,4.)
            residues=length(poles)==1 ? [fill(ComplexF64(real(source)),1,1)] :
                [fill(ComplexF64(source),1,1),fill(ComplexF64(conj(source)),1,1)]
            value=PlanarRationalModel(poles,residues,fill(feedthrough,1,1),zeros(1,1),
                [0.,1.],0.,0.,false,false,:none,NaN,0.,0.)
            ci=DiffMoM._vf_pair_indices(poles);s=length(poles)
            A=s==1 ? fill(X(-1),1,1) : X.([-1. 2.;-2. -1.])
            B=s==1 ? ones(X,1,1) : reshape(X[1,0],2,1)
            C=s==1 ? fill(X(real(source)),1,1) : reshape(X[2X(real(source)),2X(imag(source))],1,2)
            for P in (Matrix{Float64}(I,s,s),2Matrix{Float64}(I,s,s),ones(s,s)),a in (.5,2.),f in (.5,2.)
                exactP=X(a)*X(f)*X.(P);cross=C'-exactP*B
                inequality=[-(A'*exactP+exactP*A) cross;cross' fill(2X(feedthrough),1,1)]
                expected=psd_reference(inequality)
                saved=copy(P)
                @test DiffMoM._vf_storage_exact(value,ci,P,a,f;max_bytes=DiffMoM._default_max_dense_payload_bytes())==expected
                @test P==saved
            end
        end
        # Caller MPFR state is irrelevant to directed Frobenius bounds
        # and exact-integer storage checks, including concurrent callers.
        for bits in (precision(Float64),2precision(Float64)),mode in (RoundNearest,RoundDown,RoundUp)
            setprecision(BigFloat,bits) do
                setrounding(BigFloat,mode) do
                    @test !planar_rational_certificate(hidden).certified
                    @test DiffMoM._vf_frobenius_bound(hidden;max_bytes=DiffMoM._default_max_dense_payload_bytes())>1.
                    @test precision(BigFloat)==bits && rounding(BigFloat)===mode
                end
            end
        end
        @test !fetch(Threads.@spawn planar_rational_certificate(hidden)).certified
        @test fetch(Threads.@spawn planar_rational_certificate(model(ones(1,1)))).certified
        # Exact norm lattice remains an upper bound on a known scalar
        # two-residue sum, under maximum/subnormal stored coefficients.
        for magnitude in (tiny,floatmin(Float64),1.,floatmax(Float64)/4)
            norm_model=PlanarRationalModel(ComplexF64[-1.,-2.],
                [fill(ComplexF64(magnitude),1,1),fill(ComplexF64(-magnitude),1,1)],
                ones(1,1),zeros(1,1),[0.,1.],0.,0.,false,false,:none,NaN,0.,0.)
            bound=DiffMoM._vf_frobenius_bound(norm_model;max_bytes=DiffMoM._default_max_dense_payload_bytes())
            @test isfinite(bound) && Rational{BigInt}(bound)>=Rational{BigInt}(magnitude)*3/2
        end
        storage_model=model(ones(1,1);residue=ones(1,1))
        ci=DiffMoM._vf_pair_indices(storage_model.poles);P=ones(1,1)
        denbits,bits,payload=DiffMoM._vf_storage_integer_workspace(storage_model,P,1.,1.)
        @test DiffMoM._vf_storage_exact(storage_model,ci,P,1.,1.;max_bytes=payload)
        @test_throws ArgumentError DiffMoM._vf_storage_exact(storage_model,ci,P,1.,1.;max_bytes=payload-1)
        @test_throws ArgumentError DiffMoM._vf_storage_exact(storage_model,ci,P,1.,1.;max_bytes=payload,retained_bytes=sizeof(Float64))
        @test DiffMoM._vf_storage_exact(storage_model,ci,P,1.,1.;max_bytes=payload+sizeof(Float64),retained_bytes=sizeof(Float64))
        for A in matrices
            expected=psd_reference(A);saved=copy(A)
            @test (@inferred DiffMoM._vf_psd(A,1e-8))==expected
            @test A==saved
            certificate=@inferred planar_rational_certificate(model(A))
            @test certificate.certified==expected
        end
        indefinite=[nextfloat(0.) 2nextfloat(0.) 0.;2nextfloat(0.) nextfloat(0.) 0.;0. 0. floatmax(Float64)]
        @test !planar_rational_certificate(model(zeros(3,3),indefinite)).certified
        @test !planar_rational_certificate(model(zeros(3,3);residue=indefinite)).certified
        for A in (ones(2,2),[1. .5;.5 1.])
            denbits,bits,payload=DiffMoM._vf_psd_integer_workspace(A)
            @test bits%Base.GMP.BITS_PER_LIMB==0 && denbits>=0
            @test DiffMoM._vf_psd(A,1e-8;max_bytes=payload)
            @test_throws ArgumentError DiffMoM._vf_psd(A,1e-8;max_bytes=payload-1)
            @test DiffMoM._vf_psd(A,1e-8;max_bytes=payload+sizeof(Float64),retained_bytes=sizeof(Float64))
            @test_throws ArgumentError DiffMoM._vf_psd(A,1e-8;max_bytes=payload,retained_bytes=sizeof(Float64))
            @test planar_rational_certificate(model(A);max_bytes=payload).certified
            @test_throws ArgumentError planar_rational_certificate(model(A);max_bytes=payload-1)
        end
        value=model([1. 1.;-1. 1.])
        @test planar_rational_certificate(value;max_bytes=sizeof(UInt8)).certified # minimum positive budget, no owned workspace for diagonal Hermitian energy
        diagonal=model(ones(1,1);residue=ones(1,1))
        @test_throws ArgumentError planar_rational_certificate(diagonal;max_bytes=sizeof(Int)-1)
        @test planar_rational_certificate(diagonal;max_bytes=sizeof(Int)).certified
        for bits in (precision(Float64),2precision(Float64)),mode in (RoundNearest,RoundDown,RoundUp)
            setprecision(BigFloat,bits) do
                setrounding(BigFloat,mode) do
                    @test !DiffMoM._vf_psd(indefinite,1e-8)
                    @test precision(BigFloat)==bits && rounding(BigFloat)===mode
                end
            end
        end
        @test fetch(Threads.@spawn DiffMoM._vf_psd(ones(3,3),1e-8))
        @test !fetch(Threads.@spawn DiffMoM._vf_psd(indefinite,1e-8))
    end
    counts=Test.get_test_counts(result)
    output===nothing || open(output,"w") do io
        TOML.print(io,Dict("version"=>string(VERSION),"threads"=>Threads.nthreads(),
            "passes"=>counts.passes+counts.cumulative_passes,"fails"=>counts.fails+counts.cumulative_fails,
            "errors"=>counts.errors+counts.cumulative_errors))
    end
end
length(ARGS)==2 ? exercise(ARGS...) : exercise(normpath(joinpath(@__DIR__,"..")),nothing)
end
