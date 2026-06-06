using DiffMoM
using StaticArrays
using SpecialFunctions
using Printf

const c0 = 299792458.0
const freq = 10e9
const lam = c0 / freq
const k = 2π / lam
const dx = 0.5 * lam
const dy = 0.5 * lam
const A = dx * dy
const Eopt = sqrt(π / A)

# Pure Floquet spectral representation of the FULL periodic Green's function.
# Converges for Δz != 0 because evanescent orders decay as exp(-|kz||Δz|).
function gper_spectral(drho, dz; kx=0.0, ky=0.0, P=60)
    val = 0.0 + 0.0im
    for p in -P:P, q in -P:P
        kxp = kx + 2π * p / dx
        kyq = ky + 2π * q / dy
        kz = sqrt(complex(k^2 - kxp^2 - kyq^2))
        imag(kz) > 0 && (kz = -kz)              # Im(kz) <= 0 (outgoing/decaying)
        abs(kz) < 1e-9 * k && continue
        phase = exp(-im * (kxp * drho[1] + kyq * drho[2]))
        val += phase * exp(-im * kz * abs(dz)) / (2im * kz) / A
    end
    return val
end

g0(R) = exp(-im * k * R) / (4π * R)

lat(E; Ns=8, Nf=18) = DiffMoM.PeriodicLattice(dx, dy, 0.0, 0.0, k, E, Ns, Nf)

println("="^72)
println("  Ewald periodic Green's function — vertical separation validation")
println("  λ=$(lam*1e3)mm  cell=0.5λ  normal incidence  Eopt=$(round(Eopt,digits=2))")
println("="^72)

testpts = [
    (SVector(0.0, 0.0, 0.0), SVector(0.0, 0.0, 0.10lam)),   # Δρ=0,  Δz=0.10λ
    (SVector(0.0, 0.0, 0.0), SVector(0.07lam, 0.04lam, 0.10lam)),
    (SVector(0.0, 0.0, 0.0), SVector(0.07lam, 0.04lam, 0.25lam)),
    (SVector(0.0, 0.0, 0.0), SVector(0.12lam, -0.03lam, 0.50lam)),
    (SVector(0.0, 0.0, 0.0), SVector(0.07lam, 0.04lam, 0.0)),   # coplanar regression
]

maxE = 0.0
maxS = 0.0
for (r, rp) in testpts
    drho = SVector(r[1]-rp[1], r[2]-rp[2])
    dz = r[3] - rp[3]
    R = sqrt(drho[1]^2 + drho[2]^2 + dz^2)

    # (1) E-independence
    g_lo = greens_periodic_correction(r, rp, k, lat(0.5Eopt))
    g_md = greens_periodic_correction(r, rp, k, lat(1.0Eopt))
    g_hi = greens_periodic_correction(r, rp, k, lat(2.0Eopt))
    eind = max(abs(g_lo - g_md), abs(g_hi - g_md)) / max(abs(g_md), 1e-30)
    global maxE = max(maxE, eind)

    # (2) Ewald (G0 + ΔG) vs pure spectral reference (skip pure check at Δz=0: G0 singular cancels)
    if abs(dz) > 1e-9
        gper_ewald = g0(R) + g_md
        gper_ref = gper_spectral(drho, dz)
        srel = abs(gper_ewald - gper_ref) / max(abs(gper_ref), 1e-30)
        global maxS = max(maxS, srel)
        @printf("  Δρ=(%.3f,%.3f)λ Δz=%.2fλ | E-indep=%.2e | Ewald-vs-spectral=%.2e\n",
                drho[1]/lam, drho[2]/lam, dz/lam, eind, srel)
    else
        @printf("  Δρ=(%.3f,%.3f)λ Δz=%.2fλ | E-indep=%.2e | (coplanar, spectral ref skipped)\n",
                drho[1]/lam, drho[2]/lam, dz/lam, eind)
    end
end

println("="^72)
@printf("  MAX E-independence rel error:        %.3e  %s\n", maxE, maxE < 1e-9 ? "PASS" : "FAIL")
@printf("  MAX Ewald-vs-spectral rel error:     %.3e  %s\n", maxS, maxS < 1e-7 ? "PASS" : "FAIL")
println("="^72)
