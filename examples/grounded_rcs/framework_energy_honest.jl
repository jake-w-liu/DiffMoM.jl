include(joinpath(@__DIR__, "framework_pixel_design.jl"))
using Printf

# Radiation vector for an arbitrary Floquet mode: R_cur_mode = s_mode ᵀ I.
function svec_mode(mesh, rwg, k, lat, mode; quad_order=3, E0=1.0, eta0=376.730313668, pol=SVector(1.0,0.0,0.0))
    xi, wq = tri_quad_rule(quad_order); Nq=length(wq); Nt=ntriangles(mesh); N=rwg.nedges
    A_cell = lat.dx*lat.dy
    t2b = [Int[] for _ in 1:Nt]
    for n in 1:N; push!(t2b[rwg.tplus[n]],n); push!(t2b[rwg.tminus[n]],n); end
    kz = real(mode.kz); khat = SVector(mode.kx/k, mode.ky/k, kz/k)
    pmr = pol - dot(pol,khat)*khat; pm = pmr/norm(pmr)
    G = zeros(ComplexF64, N)
    for t in 1:Nt
        At=triangle_area(mesh,t); qp=tri_quad_points(mesh,t,xi)
        for q in 1:Nq
            rq=qp[q]; ph=exp(im*(mode.kx*rq[1]+mode.ky*rq[2]))
            for n in t2b[t]
                fn=eval_rwg(rwg,n,rq,t); G[n]+= dot(pm,fn)*ph*wq[q]*(2*At)
            end
        end
    end
    return -(eta0*k)/(2*kz*E0) .* (G ./ A_cell)
end

# Energy-honest problem: precompute all propagating modes' grounded reflection maps.
struct HProblem
    k; h; mesh; lat; rwg; Nt; Mt; Zg; v; cfg
    i00            # index (in modelist) of (0,0)
    Ws             # vector of w_mode (length nprop) : R_grounded_mode = Wsᵀ I + bs
    bs             # backgrounds
    wts            # kz/k weights
end
function make_hproblem(mesh, lat, rwg, k, h, Zg, v, cfg)
    Nt=ntriangles(mesh); Mt=precompute_triangle_mass(mesh,rwg)
    modes = floquet_modes(k, lat; N_orders=4)
    prop = [m for m in modes if m.propagating]
    kzi = real(floquet_modes(k,lat;N_orders=0)[1].kz)  # (0,0) kz = k at normal
    Ws=Vector{Vector{ComplexF64}}(); bs=ComplexF64[]; wts=Float64[]; i00=0
    for (i,m) in enumerate(prop)
        s = svec_mode(mesh,rwg,k,lat,m)
        push!(Ws, (1 - exp(-2im*real(m.kz)*h)) .* s)
        is00 = (m.m==0 && m.n==0)
        push!(bs, is00 ? -exp(-2im*kzi*h) : 0.0+0im)
        push!(wts, real(m.kz)/k)
        is00 && (i00=i)
    end
    return HProblem(k,h,mesh,lat,rwg,Nt,Mt,Zg,v,cfg,i00,Ws,bs,wts)
end

# Objective J = |R00|² for the co-polar specular return. The full vector
# reflected-power budget is reported separately by `eval_honest`; using only a
# scalar co-polar Floquet sum as an energy denominator misses cross-polarized
# power and can create a false conservation residual.
function objgrad_honest(P::HProblem, E, Hf, rho, beta)
    rt=Hf*rho; rb=heaviside_project(rt,beta); rbt=E*rb
    F=lu(P.Zg + assemble_Z_penalty(P.Mt, rbt, P.cfg)); I=F\P.v
    Rs = [ sum(P.Ws[i].*I)+P.bs[i] for i in eachindex(P.Ws) ]
    A = abs2(Rs[P.i00])
    J = A
    # dA/dI* = R00 conj(w00)
    dA = Rs[P.i00].*conj(P.Ws[P.i00])
    dJ = dA
    lam = F' \ dJ
    g_tri = gradient_density(P.Mt, Vector{ComplexF64}(I), Vector{ComplexF64}(lam), rbt, P.cfg)
    g = Hf' * ((E' * g_tri) .* heaviside_derivative(rt, beta))
    return J, g, A
end

function eval_honest(P::HProblem, E, Hf, rho, beta)
    rt=Hf*rho; rb=heaviside_project(rt,beta); rbt=E*rb
    I=(P.Zg+assemble_Z_penalty(P.Mt,rbt,P.cfg))\P.v
    Rs=[ sum(P.Ws[i].*I)+P.bs[i] for i in eachindex(P.Ws) ]
    A=abs2(Rs[P.i00])
    modes_vec, R_vecs = reflection_coefficient_vectors_grounded(P.mesh, P.rwg, I, P.k, P.lat;
        height=P.h, N_orders=4, E0=1.0, pol=SVector(1.0,0.0,0.0))
    B=sum(reflected_power_fractions(modes_vec, R_vecs, P.k))
    return sqrt(A), B, 100*count(x->x<0.05||x>0.95,rb)/length(rb)
end
