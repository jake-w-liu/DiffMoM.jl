# 21_grounded_rcs_demo.jl — Ground-backed specular-RCS density optimization
#
# Runs a fixed, lossless 10 GHz case and checks the specular objective against
# the public grounded-Floquet postprocessor and the full reflected-power budget.
#
# Run: julia --project=. examples/21_grounded_rcs_demo.jl

include(joinpath(@__DIR__, "grounded_rcs", "framework_pixel_design.jl"))

const MAX_SPECULAR_AMPLITUDE = 0.30
const MIN_BINARY_FRACTION = 0.85
const MAX_POWER_BUDGET_ERROR = 0.02
const MAX_OBJECTIVE_CROSSCHECK_ERROR = 1e-10

lam = C0/10e9; k = 2π/lam
dxc = 1.2*lam; Nx = 14; h = lam/4
mesh = make_rect_plate(dxc, dxc, Nx, Nx); lat = PeriodicLattice(dxc, dxc, 0.0, 0.0, k)
rwg = build_rwg_periodic(mesh, lat; precheck=true, allow_boundary=true, require_closed=false)
Nt = ntriangles(mesh); Mt = precompute_triangle_mass(mesh, rwg)
W, wsum = build_filter_weights(mesh, 0.4*lam)
Zg = assemble_Z_efie_grounded(mesh, rwg, k, lat; height=h)
pw = make_plane_wave(Vec3(0.0,0.0,-k), 1.0, Vec3(1.0,0.0,0.0))
v = Vector{ComplexF64}(assemble_excitation_grounded(mesh, rwg, pw, k, lat; height=h))
cfg = DensityConfig(; p=3.0, Z_max_factor=100.0, reactive=true)
# Assemble the linear (0,0) reflection map in one mesh pass. This is equivalent
# to evaluating every unit-current vector separately, without repeating the
# Floquet quadrature for all N basis functions.
s = svec_fast(mesh, rwg, k, lat)
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
    rho_by_beta = Dict{Float64, Vector{Float64}}()
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
        rho_by_beta[beta] = copy(rho)
        @printf("  β=%2d | |R00|=%.4f (%.2f dB) | binary=%.0f%%\n", Int(beta), sqrt(Jb),
                20log10(sqrt(Jb)+1e-15), 100*count(x -> x<0.05 || x>0.95, rbb)/Nt)
    end
    return rho_by_beta
end

function analyze(rho)
    Jf, _, If, rbf = objgrad(rho, 64.0)
    modesf, Rgf = reflection_coefficients_grounded(mesh, rwg, If, k, lat; height=h, N_orders=3, E0=1.0, pol=SVector(1.0,0.0,0.0))
    modesv, Rv = reflection_coefficient_vectors_grounded(mesh, rwg, If, k, lat; height=h, N_orders=3, E0=1.0, pol=SVector(1.0,0.0,0.0))
    i00f = findfirst(m -> m.m == 0 && m.n == 0, modesf)
    i00f === nothing && error("Specular (0,0) mode is missing from the grounded result")
    objective_R00 = sqrt(Jf)
    postprocessed_R00 = abs(Rgf[i00f])
    objective_error = abs(objective_R00 - postprocessed_R00)
    vector_budget = sum(reflected_power_fractions(modesv, Rv, k))
    copol_sum = 0.0; pm = Tuple{Int,Int,Float64}[]
    for (i,m) in enumerate(modesf)
        if m.propagating
            p = abs2(Rgf[i])*real(m.kz)/k; copol_sum += p; push!(pm, (m.m, m.n, p))
        end
    end
    binary_fraction = count(x -> x<0.05 || x>0.95, rbf) / Nt
    println("\n=== FINAL ground-backed density-TO result (1.2λ cell, h=λ/4) ===")
    @printf("specular |R00| = %.4f  (%.2f dB relative to bare ground)\n", objective_R00, 20log10(objective_R00+1e-15))
    @printf("objective/postprocessor |R00| difference = %.3e\n", objective_error)
    @printf("full vector reflected budget (lossless check) = %.4f\n", vector_budget)
    @printf("co-polar Floquet sum = %.4f\n", copol_sum)
    println("co-polar propagating Floquet power fractions:")
    for (mm,nn,p) in sort(pm, by=x->-x[3]); @printf("  (%2d,%2d): %.4f\n", mm, nn, p); end
    @printf("near-binary fraction = %.1f%%\n", 100*binary_fraction)
    return (; R00=objective_R00, objective_error, vector_budget,
            copol_sum, binary_fraction)
end

initial_R00 = sqrt(objgrad(rand(MersenneTwister(11), Nt), 1.0)[1])
@printf("initial |R00| (random density, β=1) = %.4f\n", initial_R00)
println("stage metrics below use each stage's own β value:")
rho_by_beta = run_opt()

# A continuation stage can look excellent at its current β and degrade when the
# sharper final projection is applied. Compare every saved raw design at β=64,
# then choose the lowest-|R00| candidate that meets the final binarity gate.
final_candidates = NamedTuple[]
println("final-projection candidates (all evaluated at β=64):")
for beta in sort(collect(keys(rho_by_beta)))
    rho_candidate = rho_by_beta[beta]
    J_candidate, _, _, rho_bar_candidate = objgrad(rho_candidate, 64.0)
    R00_candidate = sqrt(J_candidate)
    binary_candidate = count(x -> x < 0.05 || x > 0.95, rho_bar_candidate) / Nt
    push!(final_candidates, (; source_beta=beta, rho=rho_candidate,
                             R00=R00_candidate, binary=binary_candidate))
    @printf("  source β=%2d | |R00|=%.4f | binary=%.1f%%\n",
            Int(beta), R00_candidate, 100 * binary_candidate)
end
eligible_candidates = filter(c -> c.binary >= MIN_BINARY_FRACTION, final_candidates)
isempty(eligible_candidates) && error(
    "No β-stage design meets the $(100 * MIN_BINARY_FRACTION)% final binarity gate")
selected = eligible_candidates[argmin([c.R00 for c in eligible_candidates])]
rho = selected.rho
@printf("selected source β=%d: |R00|=%.4f, binary=%.1f%%\n",
        Int(selected.source_beta), selected.R00, 100 * selected.binary)
result = analyze(rho)

all(isfinite, (initial_R00, result.R00, result.objective_error,
               result.vector_budget, result.copol_sum,
               result.binary_fraction)) || error(
    "Ground-backed optimization produced a non-finite acceptance metric")
result.R00 <= MAX_SPECULAR_AMPLITUDE || error(
    "Final |R00|=$(round(result.R00, digits=4)) exceeds $MAX_SPECULAR_AMPLITUDE")
result.objective_error <= MAX_OBJECTIVE_CROSSCHECK_ERROR || error(
    "Objective/postprocessor |R00| difference $(result.objective_error) exceeds " *
    "$MAX_OBJECTIVE_CROSSCHECK_ERROR")
abs(result.vector_budget - 1) <= MAX_POWER_BUDGET_ERROR || error(
    "Full reflected-power budget $(round(result.vector_budget, digits=5)) differs " *
    "from one by more than $MAX_POWER_BUDGET_ERROR")
result.copol_sum <= result.vector_budget + 1e-10 || error(
    "Co-polar reflected power exceeds the full vector power budget")
result.binary_fraction >= MIN_BINARY_FRACTION || error(
    "Final density is only $(round(100 * result.binary_fraction, digits=1))% near-binary; " *
    "expected at least $(100 * MIN_BINARY_FRACTION)%")
println("PASS: specular, cross-check, reflected-power, and binarity gates")
