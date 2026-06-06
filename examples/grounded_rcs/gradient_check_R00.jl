using DiffMoM, LinearAlgebra, StaticArrays, Random, Printf
const C0 = 2.99792458e8; lam = C0/10e9; k = 2π/lam; eta0 = 376.730313668
dxc = 1.2*lam; Nx = 10; h = lam/4
mesh = make_rect_plate(dxc, dxc, Nx, Nx); lat = PeriodicLattice(dxc, dxc, 0.0, 0.0, k)
rwg = build_rwg_periodic(mesh, lat; precheck=true, allow_boundary=true, require_closed=false)
N = rwg.nedges; Nt = ntriangles(mesh); Mt = precompute_triangle_mass(mesh, rwg)
W, wsum = build_filter_weights(mesh, 2.5*dxc/Nx)
Zg = assemble_Z_efie_grounded(mesh, rwg, k, lat; height=h)
pw = make_plane_wave(Vec3(0.0,0.0,-k), 1.0, Vec3(1.0,0.0,0.0))
v = Vector{ComplexF64}(assemble_excitation_grounded(mesh, rwg, pw, k, lat; height=h))
cfg = DensityConfig(; p=3.0, Z_max_factor=100.0, reactive=true)

# (0,0) reflection vector s: R_cur00 = sᵀ I  (extract column-by-column via unit currents)
modes, _ = reflection_coefficients(mesh, rwg, zeros(ComplexF64, N), k, lat; N_orders=3, E0=1.0, pol=SVector(1.0,0.0,0.0))
i00 = findfirst(m -> m.m==0 && m.n==0, modes)
s = ComplexF64[ reflection_coefficients(mesh, rwg, ComplexF64.((1:N) .== n), k, lat; N_orders=3, E0=1.0, pol=SVector(1.0,0.0,0.0))[2][i00] for n in 1:N ]
phf = 1 - exp(-2im*k*h); w = phf .* s; b = -exp(-2im*k*h)
R00g(I) = sum(w .* I) + b
beta = 8.0

function objgrad(rho)
    rt, rb = filter_and_project(W, wsum, rho, beta)
    Ztot = Zg + assemble_Z_penalty(Mt, rb, cfg); F = lu(Ztot)
    I = F \ v; R = R00g(I); J = abs2(R)
    _, Rg = reflection_coefficients_grounded(mesh, rwg, I, k, lat; height=h, N_orders=3, E0=1.0, pol=SVector(1.0,0.0,0.0))
    lam_adj = F' \ (R * conj(w))
    g_rb = gradient_density(Mt, Vector{ComplexF64}(I), Vector{ComplexF64}(lam_adj), rb, cfg)
    g = gradient_chain_rule(g_rb, rt, W, wsum, beta)
    return J, g, R, Rg[i00]
end

Random.seed!(7); rho = 0.3 .+ 0.4*rand(Nt)
J0, g, R, Rgg = objgrad(rho)
@printf("R00 my-formula=%+.5f%+.5fi | grounded-fn=%+.5f%+.5fi | match=%.2e\n", real(R), imag(R), real(Rgg), imag(Rgg), abs(R-Rgg))
Random.seed!(3); idxs = rand(1:Nt, 6); hfd = 1e-5; maxerr = 0.0
for t in idxs
    rp = copy(rho); rp[t] += hfd; rm = copy(rho); rm[t] -= hfd
    Jp, = objgrad(rp); Jm, = objgrad(rm); gfd = (Jp - Jm)/(2hfd)
    rel = abs(g[t] - gfd)/max(abs(gfd), 1e-12); global maxerr = max(maxerr, rel)
    @printf("  t=%3d | adj=%+.4e fd=%+.4e relerr=%.2e\n", t, g[t], gfd, rel)
end
@printf("MAX gradient rel error=%.3e  %s\n", maxerr, maxerr < 1e-4 ? "PASS" : "FAIL")
@printf("initial |R00_grounded|=%.4f (%.2f dB vs bare ground)\n", abs(R), 20log10(abs(R)+1e-15))
