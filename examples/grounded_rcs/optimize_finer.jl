include(joinpath(@__DIR__, "framework_energy_honest.jl"))
using Serialization, Printf
const PKG_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const ARTIFACT_DIR = abspath(get(ENV, "GROUND_ARTIFACT_DIR",
    joinpath(PKG_ROOT, "data", "grounded_artifacts")))
const JLS = abspath(get(ENV, "GJLS", joinpath(ARTIFACT_DIR, "grounded_36.jls")))
isfile(JLS) || error(
    "Grounded operator artifact not found at $JLS. Run " *
    "examples/grounded_rcs/assemble_save.jl with NMESH=36 and the same " *
    "GROUND_ARTIFACT_DIR, or set GJLS to a compatible artifact path.")
D = deserialize(JLS)
required_fields = (:Zg, :v, :NMESH, :dxl, :hfrac, :freq)
missing_fields = filter(name -> !hasproperty(D, name), required_fields)
isempty(missing_fields) || error(
    "Grounded operator artifact $JLS is incompatible; missing fields: " *
    join(string.(missing_fields), ", ") * ". Regenerate it with assemble_save.jl.")
(D.NMESH isa Integer && D.NMESH > 0) || error(
    "Grounded operator artifact $JLS must record a positive integer NMESH; " *
    "got $(D.NMESH). Regenerate it with assemble_save.jl.")
D.NMESH % 12 == 0 || error(
    "Grounded operator artifact $JLS records NMESH=$(D.NMESH), which is not " *
    "divisible by the fixed 12×12 design grid. Use a compatible artifact.")
all(value -> value isa Real && isfinite(value) && value > 0,
    (D.freq, D.dxl, D.hfrac)) || error(
    "Grounded operator artifact $JLS must contain positive finite freq, dxl, " *
    "and hfrac values. Regenerate it with assemble_save.jl.")
(D.Zg isa AbstractMatrix && D.v isa AbstractVector) || error(
    "Grounded operator artifact $JLS must contain matrix Zg and vector v fields. " *
    "Regenerate it with assemble_save.jl.")
freq=D.freq; lam=C0/freq; k=2π/lam; dxc=D.dxl*lam; h=D.hfrac*lam; NMESH=D.NMESH
mesh=make_rect_plate(dxc,dxc,NMESH,NMESH); lat=PeriodicLattice(dxc,dxc,0.0,0.0,k)
rwg=build_rwg_periodic(mesh,lat;precheck=true,allow_boundary=true,require_closed=false)
size(D.Zg) == (rwg.nedges, rwg.nedges) || error(
    "Grounded operator artifact $JLS has Zg size $(size(D.Zg)); expected " *
    "($(rwg.nedges), $(rwg.nedges)) for NMESH=$NMESH. Regenerate it.")
length(D.v) == rwg.nedges || error(
    "Grounded operator artifact $JLS has excitation length $(length(D.v)); " *
    "expected $(rwg.nedges). Regenerate it.")
Nt=ntriangles(mesh); cfg=DensityConfig(;p=3.0,Z_max_factor=100.0,reactive=true)
P = make_hproblem(mesh, lat, rwg, k, h, D.Zg, D.v, cfg)
@printf("Loaded %s (N=%d), %d propagating modes\n", JLS, size(D.Zg,1), length(P.Ws)); flush(stdout)

function build_design(Npix, rmin_pix)
    Npix > 0 || error("Npix must be greater than zero; got $Npix")
    NMESH % Npix == 0 || error(
        "Analysis mesh NMESH=$NMESH must be divisible by Npix=$Npix. " *
        "Use a compatible grounded artifact or design grid.")
    rmin_pix > 0 || error("rmin_pix must be greater than zero; got $rmin_pix")
    mult = NMESH ÷ Npix
    E=zeros(Float64,Nt,Npix*Npix)
    for t in 1:Nt
        c=(t+1)÷2; jx=(c-1)%NMESH+1; jy=(c-1)÷NMESH+1
        px=(jx-1)÷mult+1; py=(jy-1)÷mult+1; E[t,(py-1)*Npix+px]=1.0
    end
    return E, conic_filter_matrix(Npix, dxc/Npix, rmin_pix*dxc/Npix)
end

# Cross-mesh check: re-evaluate the 24×24 design on this finer mesh.
design_path = joinpath(ARTIFACT_DIR, "honest_design.jls")
if isfile(design_path)
    Dd = deserialize(design_path)
    (hasproperty(Dd, :Npix) && hasproperty(Dd, :rho)) || error(
        "Grounded design artifact $design_path is incompatible; expected Npix " *
        "and rho fields. Regenerate it with optimize_24.jl.")
    (Dd.Npix isa Integer && Dd.Npix > 0 && Dd.rho isa AbstractVector) || error(
        "Grounded design artifact $design_path must contain a positive integer " *
        "Npix and a density vector. Regenerate it with optimize_24.jl.")
    length(Dd.rho) == Dd.Npix^2 || error(
        "Grounded design artifact $design_path has $(length(Dd.rho)) density " *
        "values; expected $(Dd.Npix^2) for Npix=$(Dd.Npix). Regenerate it.")
    E12,Hf12 = build_design(Dd.Npix, 2.0)
    R00,budget,bf = eval_honest(P, E12, Hf12, Dd.rho, 64.0)
    @printf("\n[cross-mesh check] 24x24-design re-evaluated at %dx%d: |R00|=%.4f (%.1f dB) | full vector budget=%.4f | binary=%.0f%%\n",
            NMESH,NMESH,R00,20log10(R00+1e-15),budget,bf); flush(stdout)
end

# Fresh optimization on the finer analysis mesh.
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
@printf("\n[optimization] specular objective with full-vector power diagnostic at %dx%d, Npix=12:\n", NMESH, NMESH); flush(stdout)
E,Hf = build_design(12, 2.0); rho = opt_honest(E,Hf,12)
R00,budget,bf = eval_honest(P,E,Hf,rho,64.0)
@printf("FINAL %dx%d: |R00|=%.4f (%.1f dB) full vector budget=%.4f binary=%.0f%%\n", NMESH,NMESH,R00,20log10(R00+1e-15),budget,bf)
output_path = joinpath(ARTIFACT_DIR, "honest_design_$(NMESH).jls")
serialize(output_path, (rho=rho, Npix=12, NMESH=NMESH))
println("Saved: $output_path")
