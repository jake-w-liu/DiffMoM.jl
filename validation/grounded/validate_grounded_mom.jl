using DiffMoM
using LinearAlgebra, StaticArrays, Printf

const C0 = 299792458.0
const freq = 10e9
const lam = C0 / freq
const k = 2π / lam
const eta0 = 376.730313668

# Build a periodic cell, optionally with a centered rectangular void slot.
function build_cell(dx_lam, Nx; slot_wx=0.0, slot_wy=0.0)
    dxc = dx_lam * lam; dyc = dx_lam * lam
    full = make_rect_plate(dxc, dyc, Nx, Nx)
    keep = Int[]
    for jy in 1:Nx, jx in 1:Nx
        x = -0.5dxc + (jx - 0.5) * dxc / Nx
        y = -0.5dyc + (jy - 0.5) * dyc / Nx
        in_slot = (abs(x) <= 0.5 * slot_wx * dxc) && (abs(y) <= 0.5 * slot_wy * dyc)
        if !in_slot
            base = 2 * ((jy - 1) * Nx + (jx - 1)) + 1
            push!(keep, base); push!(keep, base + 1)
        end
    end
    mesh = TriMesh(full.xyz, full.tri[:, keep])
    lat = PeriodicLattice(dxc, dyc, 0.0, 0.0, k)
    rwg = build_rwg_periodic(mesh, lat; precheck=true, allow_boundary=true, require_closed=false)
    return mesh, rwg, lat, dxc, dyc
end

function solve_grounded(dx_lam, Nx, h; slot_wx=0.0, slot_wy=0.0, reactive=true, Zfac=0.0)
    mesh, rwg, lat, dxc, dyc = build_cell(dx_lam, Nx; slot_wx=slot_wx, slot_wy=slot_wy)
    Zg = assemble_Z_efie_grounded(mesh, rwg, k, lat; height=h)
    Zpen = zeros(ComplexF64, rwg.nedges, rwg.nedges)
    if Zfac > 0
        # uniform penalty (acts on the metal sheet) for an absorption/energy check
        Mt = precompute_triangle_mass(mesh, rwg)
        cfg = DensityConfig(; p=1.0, Z_max_factor=Zfac, reactive=reactive)
        rho = fill(0.5, ntriangles(mesh))     # half-penalized everywhere
        Zpen = assemble_Z_penalty(Mt, rho, cfg)
    end
    Ztot = Zg + Zpen
    pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))
    v = assemble_excitation_grounded(mesh, rwg, pw, k, lat; height=h)
    I = Ztot \ Vector{ComplexF64}(v)
    modes, Rg = reflection_coefficients_grounded(mesh, rwg, I, k, lat;
                                                 height=h, quad_order=3, N_orders=3,
                                                 E0=1.0, pol=SVector(1.0, 0.0, 0.0))
    modesv, Rv = reflection_coefficient_vectors_grounded(mesh, rwg, I, k, lat;
                                                         height=h, quad_order=3, N_orders=3,
                                                         E0=1.0, pol=SVector(1.0, 0.0, 0.0))
    refl = sum(reflected_power_fractions(modesv, Rv, k))
    R00 = 0.0 + 0im
    for (i, m) in enumerate(modes)
        (m.m == 0 && m.n == 0) && (R00 = Rg[i])
    end
    A = dxc * dyc
    Pinc = A / (2eta0)
    Pabs = 0.5 * real(dot(I, Zpen * I))
    return (R00=R00, refl=refl, abs=Pabs / Pinc)
end

println("="^70)
println("  Grounded periodic MoM validation")
println("="^70)

println("\n[1] Full PEC sheet at height h (expect R00 = -1, full vector budget = 1):")
for h in [lam/8, lam/4, 3lam/8, lam/2]
    r = solve_grounded(0.5, 8, h)
    @printf("  h=%.3fλ | R00=%+.5f%+.5fi  |R00|=%.5f | vector_budget=%.6f\n",
            h/lam, real(r.R00), imag(r.R00), abs(r.R00), r.refl)
end

println("\n[2] Lossless reactive sheet, 1.2λ cell (full vector budget → 1):")
for h in [lam/8, lam/4]
    r = solve_grounded(1.2, 12, h; reactive=true, Zfac=8.0)
    @printf("  h=%.3fλ | |R00|=%.5f | vector_budget=%.6f | abs_frac=%.2e\n",
            h/lam, abs(r.R00), r.refl, r.abs)
end

println("\n[3] Resistive density penalties are not used as a grounded absorption certificate here;")
println("    this validation certifies the lossless image-theory reflection budget.")
println("="^70)
