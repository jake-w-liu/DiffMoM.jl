# Assemble and serialize a grounded periodic operator for the optimization drivers.
# Run: NMESH=24 julia --project=. examples/grounded_rcs/assemble_save.jl

using DiffMoM, LinearAlgebra, StaticArrays, Serialization
const C0=2.99792458e8
const PKG_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const ARTIFACT_DIR = abspath(get(ENV, "GROUND_ARTIFACT_DIR",
    joinpath(PKG_ROOT, "data", "grounded_artifacts")))
freq=10e9; lam=C0/freq; k=2π/lam; dxl=1.2; hfrac=0.25
nmesh_text = get(ENV, "NMESH", "24")
NMESH = tryparse(Int, nmesh_text)
NMESH === nothing && error("NMESH must be an integer; got '$nmesh_text'")
NMESH > 0 || error("NMESH must be greater than zero; got $NMESH")
mkpath(ARTIFACT_DIR)
dxc=dxl*lam; h=hfrac*lam
mesh=make_rect_plate(dxc,dxc,NMESH,NMESH); lat=PeriodicLattice(dxc,dxc,0.0,0.0,k)
rwg=build_rwg_periodic(mesh,lat;precheck=true,allow_boundary=true,require_closed=false)
println("Assembling grounded operator $(NMESH)×$(NMESH) (N=$(rwg.nedges))..."); flush(stdout)
assembly_seconds = @elapsed Zg=assemble_Z_efie_grounded(mesh,rwg,k,lat;height=h)
println("Assembly elapsed time: $(round(assembly_seconds, digits=3)) s " *
        "(includes first-call compilation)")
pw=make_plane_wave(Vec3(0.0,0.0,-k),1.0,Vec3(1.0,0.0,0.0))
v=Vector{ComplexF64}(assemble_excitation_grounded(mesh,rwg,pw,k,lat;height=h))
out = joinpath(ARTIFACT_DIR, "grounded_$(NMESH).jls")
serialize(out, (Zg=Zg, v=v, NMESH=NMESH, dxl=dxl, hfrac=hfrac, freq=freq))
println("Saved: $out (Zg size $(size(Zg)))")
