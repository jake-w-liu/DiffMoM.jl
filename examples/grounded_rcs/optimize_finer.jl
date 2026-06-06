include(joinpath(@__DIR__, "framework_energy_honest.jl"))
using Serialization, Printf
const JLS = get(ENV, "GJLS", "/tmp/grounded_36.jls")
D = deserialize(JLS)
freq=D.freq; lam=C0/freq; k=2π/lam; dxc=D.dxl*lam; h=D.hfrac*lam; NMESH=D.NMESH
mesh=make_rect_plate(dxc,dxc,NMESH,NMESH); lat=PeriodicLattice(dxc,dxc,0.0,0.0,k)
rwg=build_rwg_periodic(mesh,lat;precheck=true,allow_boundary=true,require_closed=false)
Nt=ntriangles(mesh); cfg=DensityConfig(;p=3.0,Z_max_factor=100.0,reactive=true)
P = make_hproblem(mesh, lat, rwg, k, h, D.Zg, D.v, cfg)
@printf("loaded %s (N=%d), %d propagating modes\n", JLS, size(D.Zg,1), length(P.Ws)); flush(stdout)

function build_design(Npix, rmin_pix)
    mult = NMESH ÷ Npix
    E=zeros(Float64,Nt,Npix*Npix)
    for t in 1:Nt
        c=(t+1)÷2; jx=(c-1)%NMESH+1; jy=(c-1)÷NMESH+1
        px=(jx-1)÷mult+1; py=(jy-1)÷mult+1; E[t,(py-1)*Npix+px]=1.0
    end
    return E, conic_filter_matrix(Npix, dxc/Npix, rmin_pix*dxc/Npix)
end

# TEST 1 (decisive): re-evaluate the 24x24-optimized design at this finer mesh.
if isfile("/tmp/honest_design.jls")
    Dd = deserialize("/tmp/honest_design.jls")
    E12,Hf12 = build_design(Dd.Npix, 2.0)
    R00,budget,bf = eval_honest(P, E12, Hf12, Dd.rho, 64.0)
    @printf("\n[convergence] 24x24-design re-evaluated at %dx%d: |R00|=%.4f (%.1f dB) | full vector budget=%.4f | binary=%.0f%%\n",
            NMESH,NMESH,R00,20log10(R00+1e-15),budget,bf); flush(stdout)
end

# TEST 2: fresh energy-honest optimization at this mesh.
function opt_honest(E,Hf,Npix; seed=11)
    Random.seed!(seed); rho=rand(Npix*Npix)
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
        @printf("  β=%2d | |R00|=%.4f (%.1f dB) | full vector budget=%.4f | binary=%.0f%%\n",Int(beta),R00,20log10(R00+1e-15),budget,bf); flush(stdout)
    end
    return rho
end
@printf("\n[optimize] energy-honest at %dx%d, Npix=12:\n", NMESH, NMESH); flush(stdout)
E,Hf = build_design(12, 2.0); rho = opt_honest(E,Hf,12)
R00,budget,bf = eval_honest(P,E,Hf,rho,64.0)
@printf("FINAL %dx%d: |R00|=%.4f (%.1f dB) full vector budget=%.4f binary=%.0f%%\n", NMESH,NMESH,R00,20log10(R00+1e-15),budget,bf)
serialize("/tmp/honest_design_$(NMESH).jls", (rho=rho, Npix=12, NMESH=NMESH))
