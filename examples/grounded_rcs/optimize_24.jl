include(joinpath(@__DIR__, "framework_energy_honest.jl"))
using Serialization, Printf, CSV, DataFrames
const PKG_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const PROJECT_ROOT = normpath(joinpath(PKG_ROOT, ".."))
const ARTIFACT_DIR = get(ENV, "GROUND_ARTIFACT_DIR",
    joinpath(PROJECT_ROOT, "paper", "data", "grounded_artifacts"))
const DATA_DIR = joinpath(PROJECT_ROOT, "paper", "data")
const DESIGN_PATH = joinpath(ARTIFACT_DIR, "honest_design.jls")
mkpath(ARTIFACT_DIR)
mkpath(DATA_DIR)

ground_path = joinpath(ARTIFACT_DIR, "grounded_24.jls")
isfile(ground_path) || error("missing grounded artifact: $(ground_path)")
D = deserialize(ground_path)
freq=D.freq; lam=C0/freq; k=2π/lam; dxc=D.dxl*lam; h=D.hfrac*lam; NMESH=D.NMESH
mesh=make_rect_plate(dxc,dxc,NMESH,NMESH); lat=PeriodicLattice(dxc,dxc,0.0,0.0,k)
rwg=build_rwg_periodic(mesh,lat;precheck=true,allow_boundary=true,require_closed=false)
Nt=ntriangles(mesh); cfg=DensityConfig(;p=3.0,Z_max_factor=100.0,reactive=true)
P = make_hproblem(mesh, lat, rwg, k, h, D.Zg, D.v, cfg)
@printf("loaded Zg %s, %d propagating modes\n", size(D.Zg), length(P.Ws)); flush(stdout)

# (rigor 1) svec_mode(0,0) must equal svec_fast
let s00 = P.Ws[P.i00] ./ (1 - exp(-2im*k*h))
    @printf("svec_mode(0,0) vs svec_fast: maxdiff=%.2e\n", maximum(abs.(s00 .- svec_fast(mesh,rwg,k,lat)))); flush(stdout)
end

function build_design(Npix, mult, rmin_pix)
    E=zeros(Float64,Nt,Npix*Npix)
    for t in 1:Nt
        c=(t+1)÷2; jx=(c-1)%NMESH+1; jy=(c-1)÷NMESH+1
        px=(jx-1)÷mult+1; py=(jy-1)÷mult+1; E[t,(py-1)*Npix+px]=1.0
    end
    return E, conic_filter_matrix(Npix, dxc/Npix, rmin_pix*dxc/Npix)
end

# (rigor 2) FD check the energy-honest combined adjoint
let
    Npix=8; E,Hf=build_design(Npix,3,2.0)
    Random.seed!(2); rho=0.3 .+ 0.4*rand(Npix*Npix); beta=8.0
    J,g,=objgrad_honest(P,E,Hf,rho,beta)
    Random.seed!(5); idxs=rand(1:Npix*Npix,5); hfd=1e-5; mx=0.0
    for t in idxs
        rp=copy(rho);rp[t]+=hfd; rm=copy(rho);rm[t]-=hfd
        Jp,=objgrad_honest(P,E,Hf,rp,beta); Jm,=objgrad_honest(P,E,Hf,rm,beta)
        mx=max(mx, abs(g[t]-(Jp-Jm)/(2hfd))/max(abs((Jp-Jm)/(2hfd)),1e-12))
    end
    pass = mx < 1e-3
    @printf("honest-gradient FD check: max rel err=%.2e  %s\n", mx, pass ? "PASS" : "FAIL"); flush(stdout)
    pass || error("grounded combined-adjoint FD check failed")
end

function opt_honest(E,Hf,Npix; seed=11)
    Random.seed!(seed); rho=rand(Npix*Npix)
    history = DataFrame(beta=Float64[], R00_copol_dB=Float64[],
        full_reflected_budget=Float64[], binary_pct=Float64[])
    rho_by_beta = Dict{Float64, Vector{Float64}}()
    for beta in [1.0,2.0,4.0,8.0,16.0,32.0,64.0]
        step=0.2
        for it in 1:40
            J,g,=objgrad_honest(P,E,Hf,rho,beta); ng=norm(g); ng<1e-14 && break
            d=-g./ng; acc=false
            for _ in 1:6
                rt=clamp.(rho.+step.*d,0.0,1.0); Jt,=objgrad_honest(P,E,Hf,rt,beta)
                Jt<J ? (rho=rt;step*=1.3;acc=true;break) : (step*=0.5)
            end
            acc || (step*=0.5); step<1e-7 && break
        end
        R00,budget,bf=eval_honest(P,E,Hf,rho,beta)
        push!(history, (beta, 20log10(R00+1e-15), budget, bf))
        rho_by_beta[beta] = copy(rho)
        @printf("  β=%2d | |R00|=%.4f (%.1f dB) | full vector budget=%.4f | binary=%.0f%%\n",Int(beta),R00,20log10(R00+1e-15),budget,bf); flush(stdout)
    end
    return rho, history, rho_by_beta
end

@printf("\n=== ENERGY-HONEST (J=|R00|², vector budget reported): Npix=12, %dx%d analysis ===\n",NMESH,NMESH); flush(stdout)
let
    E,Hf=build_design(12,2,2.0); rho, history, rho_by_beta=opt_honest(E,Hf,12)
    R00,budget,bf=eval_honest(P,E,Hf,rho,64.0)
    @printf("FINAL: |R00|=%.4f (%.1f dB) full vector budget=%.4f binary=%.0f%%\n",R00,20log10(R00+1e-15),budget,bf); flush(stdout)
    CSV.write(joinpath(DATA_DIR, "grounded_beta_trace.csv"), history)
    serialize(DESIGN_PATH, (rho=rho, Npix=12, mult=2, rho_by_beta=rho_by_beta))
    println("saved $(DESIGN_PATH)")
end
