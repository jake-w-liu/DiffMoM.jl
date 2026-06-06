# Ground-backed RCS-reduction optimization, T-AP quality:
# DESIGN resolution (coarse, well-resolved square pixels) is decoupled from
# ANALYSIS resolution (fine mesh) so the optimizer cannot exploit sub-mesh
# structure and the energy balance is accurate.
using DiffMoM, LinearAlgebra, StaticArrays, Random, Printf
const C0 = 2.99792458e8

struct Problem
    lam; k; dxc; h; mesh; lat; rwg; Nt; Mt; W; wsum; Zg; v; w; b; i00
    Npix; Hf; E   # design grid + conic-filter matrix + pixel→triangle map
    cfg
end

# Fast O(N) (0,0)-mode reflection vector s (R_cur00 = sᵀ I), normal incidence, x-pol.
# Equivalent to N unit-current calls to reflection_coefficients but one pass.
function svec_fast(mesh, rwg, k, lat; quad_order=3, E0=1.0, eta0=376.730313668)
    xi, wq = tri_quad_rule(quad_order); Nq = length(wq); Nt = ntriangles(mesh); N = rwg.nedges
    A_cell = lat.dx * lat.dy
    tri_to_basis = [Int[] for _ in 1:Nt]
    for n in 1:N
        push!(tri_to_basis[rwg.tplus[n]], n); push!(tri_to_basis[rwg.tminus[n]], n)
    end
    Fx = zeros(Float64, N)                       # x-component of ∫ f_n
    for t in 1:Nt
        At = triangle_area(mesh, t); qp = tri_quad_points(mesh, t, xi)
        for q in 1:Nq, n in tri_to_basis[t]
            fn = eval_rwg(rwg, n, qp[q], t)
            Fx[n] += real(fn[1]) * wq[q] * (2 * At)
        end
    end
    return ComplexF64.(-(eta0) / (2 * E0 * A_cell) .* Fx)
end

# Conic (cone) filter weight matrix on the Npix×Npix pixel grid.
function conic_filter_matrix(Npix, dpix, rmin)
    Hf = zeros(Float64, Npix*Npix, Npix*Npix)
    ctr(i) = ( (i-1) % Npix, (i-1) ÷ Npix )  # (col,row)
    for i in 1:Npix*Npix
        ci, ri = ctr(i); ssum = 0.0
        for j in 1:Npix*Npix
            cj, rj = ctr(j)
            d = hypot((ci-cj)*dpix, (ri-rj)*dpix)
            wij = max(0.0, 1 - d/rmin)
            Hf[i,j] = wij; ssum += wij
        end
        Hf[i,:] ./= ssum
    end
    return Hf
end

function make_problem(; freq=10e9, dxl=1.2, Npix=12, mult=3, hfrac=0.25, rmin_pix=2.0, Zmax=100.0, p=3.0)
    lam = C0/freq; k = 2π/lam; dxc = dxl*lam; h = hfrac*lam
    Nmesh = Npix*mult
    mesh = make_rect_plate(dxc, dxc, Nmesh, Nmesh); lat = PeriodicLattice(dxc, dxc, 0.0, 0.0, k)
    rwg = build_rwg_periodic(mesh, lat; precheck=true, allow_boundary=true, require_closed=false)
    N = rwg.nedges; Nt = ntriangles(mesh); Mt = precompute_triangle_mass(mesh, rwg)
    W, wsum = build_filter_weights(mesh, 1e-9)  # identity (design filter is on the pixel grid)
    Zg = assemble_Z_efie_grounded(mesh, rwg, k, lat; height=h)
    pw = make_plane_wave(Vec3(0.0,0.0,-k), 1.0, Vec3(1.0,0.0,0.0))
    v = Vector{ComplexF64}(assemble_excitation_grounded(mesh, rwg, pw, k, lat; height=h))
    modes, _ = reflection_coefficients(mesh, rwg, zeros(ComplexF64,N), k, lat; N_orders=4, E0=1.0, pol=SVector(1.0,0.0,0.0))
    i00 = findfirst(m -> m.m==0 && m.n==0, modes)
    s = svec_fast(mesh, rwg, k, lat)
    phf = 1 - exp(-2im*k*h); w = phf .* s; b = -exp(-2im*k*h)
    # pixel→triangle map E (Nt × Npix^2): triangle t in cell c → pixel of that cell
    E = zeros(Float64, Nt, Npix*Npix)
    for t in 1:Nt
        c = (t + 1) ÷ 2                      # cell index (2 triangles per cell)
        jx = (c - 1) % Nmesh + 1; jy = (c - 1) ÷ Nmesh + 1
        px = (jx - 1) ÷ mult + 1; py = (jy - 1) ÷ mult + 1
        E[t, (py-1)*Npix + px] = 1.0
    end
    Hf = conic_filter_matrix(Npix, dxc/Npix, rmin_pix*dxc/Npix)
    cfg = DensityConfig(; p=p, Z_max_factor=Zmax, reactive=true)
    return Problem(lam,k,dxc,h,mesh,lat,rwg,Nt,Mt,W,wsum,Zg,v,w,b,i00,Npix,Hf,E,cfg)
end

# forward + objective + adjoint gradient w.r.t. raw pixel design
function objgrad(P::Problem, rho_pix, beta)
    rt_pix = P.Hf * rho_pix                                  # conic filter (pixels)
    rb_pix = heaviside_project(rt_pix, beta)                 # projection (pixels)
    rb_tri = P.E * rb_pix                                    # expand to analysis triangles
    F = lu(P.Zg + assemble_Z_penalty(P.Mt, rb_tri, P.cfg))
    I = F \ P.v; R = sum(P.w .* I) + P.b
    lam_adj = F' \ (R * conj(P.w))
    g_tri = gradient_density(P.Mt, Vector{ComplexF64}(I), Vector{ComplexF64}(lam_adj), rb_tri, P.cfg)
    g_pix_bar = P.E' * g_tri                                 # accumulate to pixels
    g_pix_tilde = g_pix_bar .* heaviside_derivative(rt_pix, beta)
    g_pix = P.Hf' * g_pix_tilde
    return abs2(R), g_pix, rb_pix
end

function evaluate(P::Problem, rho_pix, beta)
    rt_pix = P.Hf * rho_pix; rb_pix = heaviside_project(rt_pix, beta); rb_tri = P.E * rb_pix
    I = (P.Zg + assemble_Z_penalty(P.Mt, rb_tri, P.cfg)) \ P.v
    md, Rg = reflection_coefficients_grounded(P.mesh, P.rwg, I, P.k, P.lat; height=P.h, N_orders=4, E0=1.0, pol=SVector(1.0,0.0,0.0))
    copol_sum = 0.0; pm = Tuple{Int,Int,Float64}[]
    for (i,m) in enumerate(md)
        if m.propagating
            pw_ = abs2(Rg[i])*real(m.kz)/P.k; copol_sum += pw_; push!(pm, (m.m,m.n,pw_))
        end
    end
    i00 = findfirst(m -> m.m==0 && m.n==0, md)
    binf = 100*count(x->x<0.05||x>0.95, rb_pix)/length(rb_pix)
    return abs(Rg[i00]), copol_sum, binf, sort(pm, by=x->-x[3]), rb_pix
end

function optimize(P::Problem; betas=[1.0,2.0,4.0,8.0,16.0,32.0,64.0], iters=60, seed=11, verbose=true)
    Random.seed!(seed); rho = rand(P.Npix*P.Npix)
    for beta in betas
        step = 0.2
        for it in 1:iters
            J, g, = objgrad(P, rho, beta); ng = norm(g); ng < 1e-14 && break
            d = -g ./ ng; acc = false
            for _ in 1:20
                rt = clamp.(rho .+ step .* d, 0.0, 1.0); Jt, = objgrad(P, rt, beta)
                Jt < J ? (rho = rt; step *= 1.2; acc = true; break) : (step *= 0.5)
            end
            acc || (step *= 0.5); step < 1e-7 && break
        end
        if verbose
            R00, copol_sum, binf, = evaluate(P, rho, beta)
            @printf("  β=%2d | |R00|=%.4f (%6.1f dB) | co-pol Floquet sum=%.4f | binary=%.0f%%\n",
                    Int(beta), R00, 20log10(R00+1e-15), copol_sum, binf); flush(stdout)
        end
    end
    return rho
end
