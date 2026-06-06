using DiffMoM, LinearAlgebra, StaticArrays, Random, Printf, Statistics, CSV, DataFrames
const PKG_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const PROJECT_ROOT = normpath(joinpath(PKG_ROOT, ".."))
const DATA_DIR = joinpath(PROJECT_ROOT, "paper", "data")
mkpath(DATA_DIR)
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
hfd = 1e-5
rows = DataFrame(triangle=Int[], g_adjoint=Float64[], g_fd=Float64[],
    abs_error=Float64[], rel_error=Float64[])
maxerr = 0.0
for t in 1:Nt
    rp = copy(rho); rp[t] += hfd; rm = copy(rho); rm[t] -= hfd
    Jp, = objgrad(rp); Jm, = objgrad(rm); gfd = (Jp - Jm)/(2hfd)
    abs_error = abs(g[t] - gfd)
    rel = abs_error/max(abs(gfd), 1e-12); global maxerr = max(maxerr, rel)
    push!(rows, (t, g[t], gfd, abs_error, rel))
end
max_fd = maximum(abs.(rows.g_fd))
scale_floor = 1e-4 * max_fd
rows.scaled_error = rows.abs_error ./ max.(abs.(rows.g_fd), scale_floor)
max_abs_error = maximum(rows.abs_error)
max_scaled_error = maximum(rows.scaled_error)
CSV.write(joinpath(DATA_DIR, "grounded_gradient_check.csv"), rows)
CSV.write(joinpath(DATA_DIR, "grounded_gradient_summary.csv"),
    DataFrame(max_rel_error=[maxerr], mean_rel_error=[mean(rows.rel_error)],
        max_abs_error=[max_abs_error], max_scaled_error=[max_scaled_error],
        fd_step=[hfd], variables_checked=[Nt], beta=[beta],
        R_formula_match=[abs(R-Rgg)]))
pass = max_abs_error < 1e-10 && max_scaled_error < 1e-2
@printf("MAX gradient abs error=%.3e, max scaled error=%.3e  %s\n",
    max_abs_error, max_scaled_error, pass ? "PASS" : "FAIL")
@printf("initial |R00_grounded|=%.4f (%.2f dB vs bare ground)\n", abs(R), 20log10(abs(R)+1e-15))
abs(R - Rgg) < 1e-10 || error("grounded R00 formula mismatch")
pass || error("grounded R00 gradient finite-difference check failed")
