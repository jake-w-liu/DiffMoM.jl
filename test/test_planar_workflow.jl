using Test, LinearAlgebra, DiffMoM

@testset "planar workflow: rational sweep complete matrices" begin
    # A matched reciprocal line has exactly zero reflections. That first
    # entry must not prevent the other entries from being populated.
    response = f -> ComplexF64[0 exp(-1im * f / 3e9); exp(-1im * f / 3e9) 0]
    sweep = planar_sweep_abs(response, 1e9, 5e9; nports=2,
        n_eval=65, max_points=24, rel_tol=1e-4)
    @test sweep.converged
    @test all(isfinite, reduce(vcat, vec.(sweep.dense_s)))
    @test maximum(maximum(abs, sweep.dense_s[k] - response(sweep.dense_freqs[k]))
        for k in eachindex(sweep.dense_s)) < 2e-4
    @test all(sweep.est_err .<= 1e-4)
    # Linear and constant data are valid complete lower-degree models.
    linear = f -> ComplexF64[0.1 0.2 + f / 1e11; 0.2 + f / 1e11 -0.1]
    sl = planar_sweep_abs(linear, 1e9, 5e9; nports=2, n_eval=17)
    @test sl.converged
    @test length(sl.freqs) == 3
    @test all(sl.dense_s[k] ≈ linear(sl.dense_freqs[k]) for k in eachindex(sl.dense_s))
    # Equal endpoint samples on a nonconstant curve need rational fallback.
    symmetric = f -> fill(0.2 + 0.1 * ((f - 3e9) / 2e9)^2, 1, 1)
    sp = planar_sweep_abs(symmetric, 1e9, 5e9; nports=1,
        n_eval=33, rel_tol=1e-6)
    @test sp.converged
    @test all(sp.dense_s[k] ≈ symmetric(sp.dense_freqs[k]) for k in eachindex(sp.dense_s))
    @test_throws ArgumentError planar_sweep_abs(linear, 1e9, 5e9; nports=2, rel_tol=Inf)
    @test_throws ArgumentError planar_sweep_abs(f -> zeros(3,3), 1e9, 5e9; nports=2)
    @test_throws ArgumentError planar_sweep_abs(f -> fill(NaN,2,2), 1e9, 5e9; nports=2)
end

@testset "adaptive sweep stored frequency and response domains" begin
    calls=Ref(0);types=DataType[]
    response=f->begin calls[]+=1;push!(types,typeof(f));fill(.2+0im,1,1) end
    sweep=planar_sweep_abs(response,big"1e9",big"2e9";nports=1,n_eval=17)
    @test sweep.converged
    @test calls[]==3
    @test all(==(Float64),types)
    @test sweep.dense_freqs==collect(range(1e9,2e9;length=17))
    for (lo,hi) in ((big"1e500",big"2e500"),(big"1e-500",big"2e-500"),
            (1e9,nextfloat(1e9)),(1e9,nextfloat(nextfloat(1e9))))
        calls[]=0
        @test_throws ArgumentError planar_sweep_abs(response,lo,hi;nports=1,n_eval=17)
        @test calls[]==0
    end
    calls[]=0
    bad_response=f->begin calls[]+=1;fill(Complex{BigFloat}(big"1e500",0),1,1) end
    @test_throws ArgumentError planar_sweep_abs(bad_response,1e9,2e9;nports=1,n_eval=17)
    @test calls[]==1
    # Sufficient Float64 resolution at either extreme stays supported.
    for (lo,hi) in ((1e-300,2e-300),(1e308,1.5e308))
        result=planar_sweep_abs(f->fill(.2+0im,1,1),lo,hi;nports=1,n_eval=17)
        @test result.converged
        @test all(isfinite,result.dense_freqs)
        @test length(unique(result.dense_freqs))==17
    end
end

@testset "adaptive sweep independent resonant ABCD ladder" begin
    # Truth comes from analytic series-RLC admittances and cascaded line
    # sections, without production circuit stamps or rational fitting.
    function response(f,centers,resistance,delay)
        om=2pi*f;T=Matrix{ComplexF64}(I,2,2)
        for center in centers
            inductance=1e-9;capacitance=1/((2pi*center)^2*inductance)
            admittance=1/(resistance+1im*(om*inductance-1/(om*capacitance)))
            theta=om*delay
            T=T*ComplexF64[cos(theta) 50im*sin(theta);im*sin(theta)/50 cos(theta)]*
                ComplexF64[1 0;admittance 1]
        end
        A,B,C,D=T[1,1],T[1,2],T[2,1],T[2,2]
        denominator=A+B/50+C*50+D
        return ComplexF64[(A+B/50-C*50-D)/denominator 2*(A*D-B*C)/denominator;
            2/denominator (-A+B/50-C*50+D)/denominator]
    end
    for (centers,resistance,delay) in (([3.7e9],1.,0.),
            ([2.2e9,3.7e9,5.1e9],.2,2e-11),([3.7e9],1e-4,0.))
        calls=Float64[]
        sample=f->begin push!(calls,f);response(f,centers,resistance,delay) end
        sweep=planar_sweep_abs(sample,1e9,6e9;nports=2,n_eval=1001,
            max_points=64,rel_tol=1e-5,max_bytes=8*1024^2)
        @test sweep.converged
        @test length(calls)==length(unique(calls))==length(sweep.freqs)
        @test length(calls)<32
        @test all(sweep.est_err.<=1e-5)
        for k in eachindex(sweep.dense_freqs)
            truth=response(sweep.dense_freqs[k],centers,resistance,delay)
            @test maximum(abs,sweep.dense_s[k]-truth)<1e-4
            @test all(isfinite,sweep.dense_s[k])
            @test maximum(abs,sweep.dense_s[k]-transpose(sweep.dense_s[k]))<1e-10
            @test opnorm(sweep.dense_s[k])<=1+1e-5
        end
        # Additional frequencies between the adaptive candidate grid points
        # use the final analyzed data; analytic truth is evaluated afresh.
        xs=2 .* ((sweep.freqs.-1e9)./5e9).-1
        models=[DiffMoM._sweep_model(xs,[S[p,q] for S in sweep.s]) for p in 1:2,q in 1:2]
        for f in 1e9.+((0:20:999).+.37).*5e6
            predicted=ComplexF64[DiffMoM._sweep_eval(models[p,q],2*((f-1e9)/5e9)-1) for p in 1:2,q in 1:2]
            @test maximum(abs,predicted-response(f,centers,resistance,delay))<1e-4
        end
        if resistance==1e-4
            k=findfirst(==(3.7e9),sweep.dense_freqs)
            actual_db=20log10(abs(sweep.dense_s[k][2,1]))
            truth_db=20log10(abs(response(3.7e9,centers,resistance,delay)[2,1]))
            @test actual_db< -100
            @test abs(actual_db-truth_db)<1e-5
        end
    end
    # Interpolation workspace rejection precedes the first callback.
    calls=Ref(0)
    bytes=DiffMoM._planar_sweep_storage_bytes(2,17,16)+2*2*sizeof(ComplexF64)
    @test_throws ArgumentError planar_sweep_abs(f->begin calls[]+=1;zeros(2,2) end,
        1e9,6e9;nports=2,n_eval=17,max_points=16,max_bytes=bytes-1)
    @test calls[]==0
    x=collect(range(-1.,1.;length=17));y=ComplexF64[1/(2+1im*z) for z in x]
    model=DiffMoM._sweep_aaa_model(x,y)::DiffMoM._SweepAAA
    evaluate=()->DiffMoM._sweep_eval(model,.173)
    evaluate()
    @test @allocated(evaluate())==0
    @test abs(evaluate()-1/(2+.173im))<1e-13
end

@testset "planar workflow: coupled-line RLGC and phase continuation" begin
    # A matrix-exponential telegrapher oracle exercises mutual coupling,
    # noncommuting Z'/Y', losses, and repeated modal eigenvalues.
    len = 0.002
    R = [2.0 0.3; 0.3 3.0]
    L = [200e-9 40e-9; 40e-9 250e-9]
    G = [0.01 -0.003; -0.003 0.02]
    C = [90e-12 -10e-12; -10e-12 110e-12]
    function coupled_y(z, y, len)
        n = size(z,1)
        T = exp([zeros(ComplexF64,n,n) z; y zeros(ComplexF64,n,n)] * len)
        A,B = T[1:n,1:n], T[1:n,n+1:2n]
        Cb,D = T[n+1:2n,1:n], T[n+1:2n,n+1:2n]
        return [D / B Cb - (D / B) * A; -(B \ Matrix{ComplexF64}(I,n,n)) B \ A]
    end
    for f in (1e9,5e9,20e9)
        z,y = R + 2pi*f*1im*L, G + 2pi*f*1im*C
        Y = coupled_y(z,y,len)
        r = planar_rlgc(Y,len,f)
        @test r isa MulticonductorRLGC
        @test r.r ≈ R rtol=2e-6
        @test r.l ≈ L rtol=1e-10
        @test r.g ≈ G rtol=2e-6
        @test r.c ≈ C rtol=1e-10
        @test length(r.gammas) == 2
    end
    f = 3e9
    repeated = coupled_y(2pi*f*1im*200e-9 * Matrix{Float64}(I,2,2),
        2pi*f*1im*100e-12 * Matrix{Float64}(I,2,2),len)
    rr = planar_rlgc(repeated,len,f)
    @test rr.l ≈ 200e-9 * Matrix{Float64}(I,2,2)
    @test rr.c ≈ 100e-12 * Matrix{Float64}(I,2,2)
    @test_throws ArgumentError planar_rlgc(repeated,len,f;max_bytes=1)
    bad = copy(repeated); bad[1,3] *= 0.9
    @test_throws ArgumentError planar_rlgc(bad,len,f)
    # Scalar long-line sweep crosses several 2pi phase branches.
    fs = collect(range(1.1e9,20.3e9;length=41))
    length_long = 0.02
    ys = [DiffMoM._y_of_abcd(planar_line_abcd(50,
        0.005 + 2pi*1im*f*length_long/2e8)) for f in fs]
    rs = planar_rlgc_sweep(ys,length_long,fs)
    @test all(isapprox(r.l,50/2e8;rtol=1e-10) for r in rs)
    @test all(isapprox(r.c,1/(50*2e8);rtol=1e-10) for r in rs)
    # A high-frequency first point supplies its independent branch hint.
    hinted = planar_rlgc(ys[end],length_long,fs[end];
        phase_hint=2pi*fs[end]*length_long/2e8)
    @test hinted.l ≈ 50/2e8
    @test_throws ArgumentError planar_rlgc_sweep(ys,length_long,reverse(fs))
end

@testset "planar workflow: extraction invariants" begin
    section = planar_line_abcd(50, 0.3 + 0.7im)
    asymmetric = copy(section)
    asymmetric[1,1] += 0.2
    asymmetric[2,2] -= 0.2
    @test_throws ArgumentError planar_line_params(DiffMoM._y_of_abcd(asymmetric))
    @test_throws ArgumentError planar_pi_model([2.0 -1.0; -2.0 3.0])
    @test_throws ArgumentError planar_inductor([2.0 -1.0; -2.0 3.0], 1e9)
    @test_throws ArgumentError planar_line_abcd(0, 0.3)
    @test_throws ArgumentError planar_line_abcd(50, 1000)
    @test_throws DimensionMismatch DiffMoM._y_of_abcd(ones(1,1))
    # A nonzero rank-deficient de-embedding system reaches LU's check.
    Y = ones(2,2)
    chain = ComplexF64[1 1; 0 0]
    @test_throws ArgumentError deembed_ports(Y, [chain,chain])
end

@testset "planar workflow: magnetic cover resonances" begin
    a, b, h = 0.02, 0.03, 0.01
    stack = PlanarStackup([PlanarLayer(1,1,h)], TERM_GND,
        PlanarTerminator(TERM_PMC), a, b)
    exact = DiffMoM._C0 / 2 * sqrt((1/a)^2 + (1/b)^2 + (1/(2h))^2)
    modes = planar_box_resonances(stack, 0.9exact, 1.1exact;
        mmax=1, nmax=1, nsamp=129)
    @test any(r -> r.m == r.n == 1 && r.pol == TM_POL &&
        isapprox(r.freq, exact; rtol=1e-8), modes)
    # The scan's answer must be independent of the requested interface.
    lower = planar_box_resonances(stack, 0.9exact, 1.1exact;
        mmax=1, nmax=1, nsamp=129, iface=0)
    @test [(r.freq,r.m,r.n,r.pol) for r in modes] ==
          [(r.freq,r.m,r.n,r.pol) for r in lower]
    @test_throws ArgumentError planar_box_resonances(stack,1e9,2e9;rtol=NaN)
end
