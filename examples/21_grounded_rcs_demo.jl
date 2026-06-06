using DiffMoM, LinearAlgebra, StaticArrays, Random, Printf
const C0 = 2.99792458e8; lam = C0/10e9; k = 2π/lam; eta0 = 376.730313668
dxc = 1.2*lam; Nx = 14; h = lam/4
mesh = make_rect_plate(dxc, dxc, Nx, Nx); lat = PeriodicLattice(dxc, dxc, 0.0, 0.0, k)
rwg = build_rwg_periodic(mesh, lat; precheck=true, allow_boundary=true, require_closed=false)
N = rwg.nedges; Nt = ntriangles(mesh); Mt = precompute_triangle_mass(mesh, rwg)
W, wsum = build_filter_weights(mesh, 0.4*lam)
Zg = assemble_Z_efie_grounded(mesh, rwg, k, lat; height=h)
pw = make_plane_wave(Vec3(0.0,0.0,-k), 1.0, Vec3(1.0,0.0,0.0))
v = Vector{ComplexF64}(assemble_excitation_grounded(mesh, rwg, pw, k, lat; height=h))
cfg = DensityConfig(; p=3.0, Z_max_factor=100.0, reactive=true)
modes, _ = reflection_coefficients(mesh, rwg, zeros(ComplexF64, N), k, lat; N_orders=3, E0=1.0, pol=SVector(1.0,0.0,0.0))
i00 = findfirst(m -> m.m==0 && m.n==0, modes)
s = ComplexF64[ reflection_coefficients(mesh, rwg, ComplexF64.((1:N) .== n), k, lat; N_orders=3, E0=1.0, pol=SVector(1.0,0.0,0.0))[2][i00] for n in 1:N ]
phf = 1 - exp(-2im*k*h); w = phf .* s; b = -exp(-2im*k*h)

function objgrad(rho, beta)
    rt, rb = filter_and_project(W, wsum, rho, beta)
    Ztot = Zg + assemble_Z_penalty(Mt, rb, cfg); F = lu(Ztot)
    I = F \ v; R = sum(w .* I) + b; J = abs2(R)
    lam_adj = F' \ (R * conj(w))
    g_rb = gradient_density(Mt, Vector{ComplexF64}(I), Vector{ComplexF64}(lam_adj), rb, cfg)
    g = gradient_chain_rule(g_rb, rt, W, wsum, beta)
    return J, g, I, rb
end

# Normalized projected-gradient descent with backtracking + beta-continuation.
# Random init breaks the uniform-sheet stationary symmetry.
function run_opt()
    Random.seed!(11); rho = rand(Nt)
    betas = [1.0,2.0,4.0,8.0,16.0,32.0,64.0]
    for beta in betas
        step = 0.2
        for it in 1:50
            J, g, = objgrad(rho, beta)
            ng = norm(g)
            ng < 1e-14 && break
            d = -g ./ ng
            accepted = false
            for _ in 1:18
                rt = clamp.(rho .+ step .* d, 0.0, 1.0)
                Jt, = objgrad(rt, beta)
                if Jt < J
                    rho = rt; step *= 1.2; accepted = true; break
                end
                step *= 0.5
            end
            accepted || (step *= 0.5)
            step < 1e-6 && break
        end
        Jb, _, _, rbb = objgrad(rho, beta)
        @printf("  β=%2d | |R00|=%.4f (%.2f dB) | binary=%.0f%%\n", Int(beta), sqrt(Jb),
                20log10(sqrt(Jb)+1e-15), 100*count(x -> x<0.05 || x>0.95, rbb)/Nt)
    end
    return rho
end

function analyze(rho)
    Jf, _, If, rbf = objgrad(rho, 64.0)
    modesf, Rgf = reflection_coefficients_grounded(mesh, rwg, If, k, lat; height=h, N_orders=3, E0=1.0, pol=SVector(1.0,0.0,0.0))
    modesv, Rv = reflection_coefficient_vectors_grounded(mesh, rwg, If, k, lat; height=h, N_orders=3, E0=1.0, pol=SVector(1.0,0.0,0.0))
    vector_budget = sum(reflected_power_fractions(modesv, Rv, k))
    copol_sum = 0.0; pm = Tuple{Int,Int,Float64}[]
    for (i,m) in enumerate(modesf)
        if m.propagating
            p = abs2(Rgf[i])*real(m.kz)/k; copol_sum += p; push!(pm, (m.m, m.n, p))
        end
    end
    println("\n=== FINAL ground-backed optimized metasurface (1.2λ cell, h=λ/4) ===")
    @printf("specular |R00| = %.4f  (%.2f dB below bare ground)\n", sqrt(Jf), 20log10(sqrt(Jf)+1e-15))
    @printf("full vector reflected budget (lossless check) = %.4f\n", vector_budget)
    @printf("co-polar Floquet sum = %.4f\n", copol_sum)
    println("co-polar propagating Floquet power fractions:")
    for (mm,nn,p) in sort(pm, by=x->-x[3]); @printf("  (%2d,%2d): %.4f\n", mm, nn, p); end
    @printf("binary fraction = %.1f%%\n", 100*count(x -> x<0.05 || x>0.95, rbf)/Nt)
end

@printf("init |R00| (random) = %.4f\n", sqrt(objgrad(rand(MersenneTwister(11), Nt), 1.0)[1]))
rho = run_opt()
analyze(rho)
