using DiffMoM, LinearAlgebra, StaticArrays, Serialization
const C0=2.99792458e8
const PKG_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const PROJECT_ROOT = normpath(joinpath(PKG_ROOT, ".."))
const ARTIFACT_DIR = get(ENV, "GROUND_ARTIFACT_DIR",
    joinpath(PROJECT_ROOT, "paper", "data", "grounded_artifacts"))
mkpath(ARTIFACT_DIR)
freq=10e9; lam=C0/freq; k=2π/lam; dxl=1.2; hfrac=0.25
NMESH=parse(Int, get(ENV, "NMESH", "24"))
dxc=dxl*lam; h=hfrac*lam
mesh=make_rect_plate(dxc,dxc,NMESH,NMESH); lat=PeriodicLattice(dxc,dxc,0.0,0.0,k)
rwg=build_rwg_periodic(mesh,lat;precheck=true,allow_boundary=true,require_closed=false)
println("assembling grounded operator $(NMESH)x$(NMESH) (N=$(rwg.nedges))..."); flush(stdout)
@time Zg=assemble_Z_efie_grounded(mesh,rwg,k,lat;height=h)
pw=make_plane_wave(Vec3(0.0,0.0,-k),1.0,Vec3(1.0,0.0,0.0))
v=Vector{ComplexF64}(assemble_excitation_grounded(mesh,rwg,pw,k,lat;height=h))
out = joinpath(ARTIFACT_DIR, "grounded_$(NMESH).jls")
serialize(out, (Zg=Zg, v=v, NMESH=NMESH, dxl=dxl, hfrac=hfrac, freq=freq))
println("saved $(out)  Zg=$(size(Zg))")
