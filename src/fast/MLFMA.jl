# MLFMA.jl — Multi-Level Fast Multipole Algorithm operator
#
# Provides O(N log N) matvec for the EFIE system matrix via octree-based
# plane-wave decomposition of the Green's function. Integrates with the
# existing GMRES/preconditioner infrastructure.
#
# Reference: Chew, Jin, Michielssen, Song, "Fast and Efficient Algorithms
# in Computational Electromagnetics", Artech House, 2001.

export MLFMAOperator, MLFMAAdjointOperator, build_mlfma_operator
export SphereSampling, assemble_mlfma_nearfield

# ─── Spherical sampling ─────────────────────────────────────────

struct SphereSampling
    L::Int                    # truncation order
    ntheta::Int
    nphi::Int
    npts::Int                 # = ntheta * nphi
    theta::Vector{Float64}
    phi::Vector{Float64}
    weights::Vector{Float64}
    khat::Matrix{Float64}     # (3, npts) unit direction vectors
end

"""
    gauss_legendre(n)

Compute Gauss-Legendre nodes on [-1,1] and weights using the
symmetric tridiagonal eigenvalue method.
"""
function gauss_legendre(n::Int)
    n >= 1 || error("gauss_legendre: n must be >= 1")
    if n == 1
        return [0.0], [2.0]
    end
    β = [i / sqrt(4.0 * i^2 - 1.0) for i in 1:n-1]
    J = SymTridiagonal(zeros(n), β)
    eig = eigen(J)
    nodes = eig.values
    weights = 2.0 .* eig.vectors[1, :].^2
    # Sort by node value
    p = sortperm(nodes)
    return nodes[p], weights[p]
end

"""
    truncation_order(box_edge, k; precision=3)

Compute the MLFMA truncation order L for a box with given edge length.
Uses the empirical formula: L = floor(k*d + 2.16 * p^(2/3) * (k*e)^(1/3))
where d = √3 * e (box diagonal), e = box edge length.
"""
function truncation_order(box_edge::Float64, k::Float64; precision::Int=3)
    d = sqrt(3.0) * box_edge    # box diagonal
    kd = k * d                   # main truncation term uses diagonal
    ke = k * box_edge            # excess term uses edge (standard convention)
    L = floor(Int, kd + 2.16 * precision^(2/3) * ke^(1/3))
    return max(L, 3)
end

"""
    make_sphere_sampling(L)

Create a spherical sampling grid for truncation order L.
θ: Gauss-Legendre with Nθ = L+1 points
φ: Uniform with Nφ = 2L+2 points on [0, 2π)
"""
function make_sphere_sampling(L::Int)
    ntheta = L + 1
    nphi = 2 * L + 2
    npts = ntheta * nphi

    # Gauss-Legendre nodes on [-1,1] → θ via acos
    gl_nodes, gl_weights = gauss_legendre(ntheta)

    # Uniform φ
    dphi = 2π / nphi
    phi_vals = [(j - 0.5) * dphi for j in 1:nphi]

    theta = Vector{Float64}(undef, npts)
    phi = Vector{Float64}(undef, npts)
    weights = Vector{Float64}(undef, npts)
    khat = Matrix{Float64}(undef, 3, npts)

    idx = 0
    for it in 1:ntheta
        θ = acos(clamp(gl_nodes[it], -1.0, 1.0))
        wθ = gl_weights[it]
        sθ = sin(θ)
        cθ = cos(θ)
        for ip in 1:nphi
            idx += 1
            φ = phi_vals[ip]
            theta[idx] = θ
            phi[idx] = φ
            weights[idx] = wθ * dphi  # GL weight × uniform φ spacing
            khat[1, idx] = sθ * cos(φ)
            khat[2, idx] = sθ * sin(φ)
            khat[3, idx] = cθ
        end
    end

    return SphereSampling(L, ntheta, nphi, npts, theta, phi, weights, khat)
end

# ─── Special functions ──────────────────────────────────────────

"""
    spherical_hankel2_all(l_max, x)

Compute h_l^(2)(x) = j_l(x) - i*y_l(x) for l = 0, 1, ..., l_max.
Uses forward recurrence (stable for moderate l_max ≤ 50).
"""
function spherical_hankel2_all(l_max::Int, x::Float64)
    h = Vector{ComplexF64}(undef, l_max + 1)

    if abs(x) < 1e-30
        fill!(h, zero(ComplexF64))
        return h
    end

    sx, cx = sincos(x)

    # y_l forward recurrence (always stable — y_l diverges as l → ∞)
    y0 = -cx / x
    y1 = -cx / x^2 - sx / x
    yl = Vector{Float64}(undef, l_max + 1)
    yl[1] = y0
    if l_max >= 1
        yl[2] = y1
    end
    for l in 1:l_max-1
        yl[l + 2] = (2l + 1) / x * yl[l + 1] - yl[l]
    end

    # j_l via backward recurrence (Miller's algorithm) — stable for all l.
    # Forward recurrence is unstable for l > x.
    l_start = l_max + max(15, ceil(Int, sqrt(40.0 * l_max)))
    jl_unnorm = Vector{Float64}(undef, l_start + 2)
    jl_unnorm[l_start + 2] = 0.0
    jl_unnorm[l_start + 1] = 1.0
    for l in (l_start - 1):-1:0
        jl_unnorm[l + 1] = (2l + 3) / x * jl_unnorm[l + 2] - jl_unnorm[l + 3]
    end

    # Normalize using the Wronskian (DLMF 10.51.2):
    #   j_{n-1}(x)*y_n(x) - j_n(x)*y_{n-1}(x) = 1/x²
    # At n=1: j_0*y_1 - j_1*y_0 = -1/x²
    # With unnormalized j values: scale * (j0u*y1 - j1u*y0) = -1/x²
    wronskian_unnorm = jl_unnorm[1] * y1 - jl_unnorm[2] * y0
    scale = (-1.0 / x^2) / wronskian_unnorm

    jl = Vector{Float64}(undef, l_max + 1)
    for l in 0:l_max
        jl[l + 1] = jl_unnorm[l + 1] * scale
    end

    for l in 0:l_max
        h[l + 1] = jl[l + 1] - im * yl[l + 1]
    end
    return h
end

"""
    legendre_all(l_max, x)

Compute P_l(x) for l = 0, 1, ..., l_max using 3-term recurrence.
"""
function legendre_all(l_max::Int, x::Float64)
    P = Vector{Float64}(undef, l_max + 1)
    P[1] = 1.0
    if l_max >= 1
        P[2] = x
    end
    for l in 1:l_max-1
        P[l + 2] = ((2l + 1) * x * P[l + 1] - l * P[l]) / (l + 1)
    end
    return P
end

# ─── Translation operators ──────────────────────────────────────

"""
    compute_translation_factor(d_vec, k, sampling)

Compute the translation operator T_L(k̂; d) at all sample directions k̂_q.

T_L(k̂; d) = Σ_{l=0}^{L} (-i)^l (2l+1) h_l^(2)(k|d|) P_l(k̂·d̂)

Returns a vector of length npts.
"""
function compute_translation_factor(d_vec::Vec3, k::Float64, sampling::SphereSampling)
    d = norm(d_vec)
    d_hat = d_vec / d
    L = sampling.L

    h = spherical_hankel2_all(L, k * d)

    # Precompute (-i)^l * (2l+1) * h_l
    coeffs = Vector{ComplexF64}(undef, L + 1)
    neg_i_pow = ComplexF64(1.0)  # (-i)^0 = 1
    for l in 0:L
        coeffs[l + 1] = neg_i_pow * (2l + 1) * h[l + 1]
        neg_i_pow *= -im
    end

    T = Vector{ComplexF64}(undef, sampling.npts)
    for q in 1:sampling.npts
        cos_alpha = dot(Vec3(sampling.khat[1, q], sampling.khat[2, q], sampling.khat[3, q]), d_hat)
        cos_alpha = clamp(cos_alpha, -1.0, 1.0)
        Pl = legendre_all(L, cos_alpha)
        val = zero(ComplexF64)
        for l in 0:L
            val += coeffs[l + 1] * Pl[l + 1]
        end
        T[q] = val
    end
    return T
end

"""
    precompute_translation_factors(level, k, sampling)

Precompute translation factors for all unique relative positions
in the interaction lists at the given level.
"""
function precompute_translation_factors(level::OctreeLevel, k::Float64, sampling::SphereSampling)
    edge = level.edge_length
    factors = Dict{NTuple{3,Int}, Vector{ComplexF64}}()

    # Collect all unique relative positions from interaction lists
    for box in level.boxes
        for il_id in box.interaction_list
            il_box = level.boxes[il_id]
            dijk = (box.ijk[1] - il_box.ijk[1],
                    box.ijk[2] - il_box.ijk[2],
                    box.ijk[3] - il_box.ijk[3])
            if !haskey(factors, dijk)
                d_vec = Vec3(dijk[1] * edge, dijk[2] * edge, dijk[3] * edge)
                factors[dijk] = compute_translation_factor(d_vec, k, sampling)
            end
        end
    end
    return factors
end

# ─── Lagrange interpolation ─────────────────────────────────────

"""
    build_lagrange_interp_1d(target_pts, source_pts; order=6, cyclic=false, period=0.0, polar_theta=false)

Build a sparse 1D Lagrange interpolation matrix from source_pts to target_pts.
Each row has at most `order` nonzero entries.

If `cyclic=true`, wrapping at boundaries is handled with given `period`.
If `polar_theta=true`, θ-direction polar reflection is used at θ=0 and θ=π
boundaries: stencils extending past the pole are reflected with sign-flipped
weights (matching the standard MLFMA anterpolation convention).
"""
function build_lagrange_interp_1d(target_pts::Vector{Float64}, source_pts::Vector{Float64};
                                   order::Int=6, cyclic::Bool=false, period::Float64=0.0,
                                   polar_theta::Bool=false)
    nt = length(target_pts)
    ns = length(source_pts)
    order = min(order, ns)

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for i in 1:nt
        t = target_pts[i]

        if cyclic && period > 0
            # Find nearest source points considering periodicity
            best_start = _find_nearest_start_cyclic(t, source_pts, order, period)
            indices = _cyclic_indices(best_start, order, ns)
            pts = [_cyclic_dist(t, source_pts[idx], period) + t for idx in indices]
            signs = ones(Float64, length(indices))
        elseif polar_theta
            # θ-direction with polar reflection at north (θ=0) and south (θ=π) poles
            best_start = _find_nearest_start(t, source_pts, order; allow_oob=true)
            raw_indices = best_start:(best_start + order - 1)
            indices = Int[]
            pts = Float64[]
            signs = Float64[]
            for idx in raw_indices
                col, pt, sgn = _pick_theta(idx, source_pts)
                push!(indices, col)
                push!(pts, pt)
                push!(signs, sgn)
            end
        else
            # Find nearest source points (standard clamped)
            best_start = _find_nearest_start(t, source_pts, order)
            indices = collect(best_start:min(best_start + order - 1, ns))
            pts = source_pts[indices]
            signs = ones(Float64, length(indices))
        end

        # Lagrange weights
        for (ji, j) in enumerate(indices)
            w = 1.0
            xj = pts[ji]
            for (ki, _) in enumerate(indices)
                ki == ji && continue
                xk = pts[ki]
                denom = xj - xk
                if abs(denom) < 1e-15
                    w = 0.0
                    break
                end
                w *= (t - xk) / denom
            end
            w *= signs[ji]  # apply sign flip for polar reflection
            if abs(w) > 1e-15
                push!(rows, i)
                push!(cols, j)
                push!(vals, w)
            end
        end
    end

    return sparse(rows, cols, vals, nt, ns)
end

"""
Pick θ value with polar reflection for out-of-range stencil indices.
Returns (clamped_index, reflected_theta, sign_factor).
At the north pole (index < 1): θ → -θ, sign = -1
At the south pole (index > n): θ → 2π - θ, sign = -1
"""
function _pick_theta(index::Int, θs::Vector{Float64})
    n = length(θs)
    if index < 1
        reflected = 1 - index + 1  # map to positive side
        reflected = clamp(reflected, 1, n)
        return reflected, -θs[reflected], -1.0
    elseif index > n
        reflected = 2 * n + 1 - index
        reflected = clamp(reflected, 1, n)
        return reflected, 2π - θs[reflected], -1.0
    else
        return index, θs[index], 1.0
    end
end

function _find_nearest_start(t::Float64, pts::Vector{Float64}, order::Int;
                              allow_oob::Bool=false)
    ns = length(pts)
    # Binary search for nearest point
    idx = searchsortedfirst(pts, t)
    # Center the stencil
    start = idx - order ÷ 2
    if allow_oob
        # Allow out-of-bounds starts for polar reflection handling
        return clamp(start, 1 - order ÷ 2, max(1, ns - order ÷ 2))
    else
        return clamp(start, 1, max(1, ns - order + 1))
    end
end

function _find_nearest_start_cyclic(t::Float64, pts::Vector{Float64}, order::Int, period::Float64)
    ns = length(pts)
    # Find nearest point considering periodicity
    best_idx = 1
    best_dist = Inf
    for j in 1:ns
        d = abs(_cyclic_dist(t, pts[j], period))
        if d < best_dist
            best_dist = d
            best_idx = j
        end
    end
    start = best_idx - order ÷ 2
    return mod1(start, ns)
end

function _cyclic_indices(start::Int, order::Int, ns::Int)
    return [mod1(start + j - 1, ns) for j in 1:order]
end

function _cyclic_dist(a::Float64, b::Float64, period::Float64)
    d = a - b
    while d > period / 2
        d -= period
    end
    while d < -period / 2
        d += period
    end
    return d
end

"""
Build interpolation matrices for transitioning between two SphereSamplings.
Returns (I_theta, I_phi) sparse matrices for the aggregation direction (fine → coarse).
For disaggregation, use the TRANSPOSE of these matrices (anterpolation).
"""
function build_interp_matrices(target::SphereSampling, source::SphereSampling; order::Int=6)
    # θ interpolation: clamped stencil (no polar reflection for xyz components —
    # polar sign-flip is only correct for θ/φ component representation)
    #
    # IMPORTANT: GL nodes are stored ascending in x = cos(θ), so θ is DESCENDING
    # in the data arrays (it=1 → θ≈π, it=nθ → θ≈0). The Lagrange builder
    # requires sorted (ascending) θ values, so we build with sorted values
    # then reverse both dimensions to match the data (GL-node) ordering.
    src_theta_sorted = sort(unique(source.theta))
    tgt_theta_sorted = sort(unique(target.theta))
    I_theta_sorted = build_lagrange_interp_1d(tgt_theta_sorted, src_theta_sorted; order=order)
    # Reverse both dims: sorted ascending-θ → data order (descending-θ = ascending-x)
    I_theta = sparse(I_theta_sorted[end:-1:1, end:-1:1])

    # φ interpolation: uniform grids with cyclic boundary (already in correct order)
    src_phi = [(j - 0.5) * 2π / source.nphi for j in 1:source.nphi]
    tgt_phi = [(j - 0.5) * 2π / target.nphi for j in 1:target.nphi]
    I_phi = build_lagrange_interp_1d(tgt_phi, src_phi;
                                      order=order, cyclic=true, period=2π)

    return I_theta, I_phi
end

# ─── Radiation patterns ─────────────────────────────────────────

"""
    compute_bf_radiation_patterns(mesh, rwg, k, octree, sampling; quad_order=3)

Compute 4-component radiation patterns for each RWG basis function
at the leaf-level spherical sampling points.

Components 1:3 = vector pattern S_n(k̂), component 4 = scalar pattern D_n(k̂)/k.

Phase is relative to the leaf box center: exp(+jk k̂·(r' - r_c)).
"""
function compute_bf_radiation_patterns(mesh::TriMesh, rwg::RWGData, k::Float64,
                                        octree::Octree, sampling::SphereSampling;
                                        quad_order::Int=3)
    N = rwg.nedges
    npts = sampling.npts
    patterns = zeros(ComplexF64, 4, npts, N)

    xi, wq = tri_quad_rule(quad_order)
    Nq = length(wq)

    leaf_level = octree.levels[octree.nLevels]

    for box in leaf_level.boxes
        r_c = box.center
        for n_perm in box.bf_range
            n = octree.perm[n_perm]  # original BF index

            for t in (rwg.tplus[n], rwg.tminus[n])
                A = triangle_area(mesh, t)
                pts = tri_quad_points(mesh, t, xi)
                dv = div_rwg(rwg, n, t)

                for qq in 1:Nq
                    rp = pts[qq]
                    fn = eval_rwg(rwg, n, rp, t)
                    wt = wq[qq] * 2 * A  # quadrature weight × Jacobian

                    dr = rp - r_c  # r' - r_c

                    for q in 1:npts
                        kh = Vec3(sampling.khat[1, q], sampling.khat[2, q], sampling.khat[3, q])
                        phase = exp(im * k * dot(kh, dr)) * wt

                        # Vector components
                        patterns[1, q, n] += fn[1] * phase
                        patterns[2, q, n] += fn[2] * phase
                        patterns[3, q, n] += fn[3] * phase
                        # Scalar component (div/k)
                        patterns[4, q, n] += (dv / k) * phase
                    end
                end
            end
        end
    end

    return patterns
end

# ─── Near-field sparse matrix ───────────────────────────────────

"""
    assemble_mlfma_nearfield(octree, mesh, rwg, k; quad_order=3, eta0=376.730313668)

Assemble the near-field (neighbor interaction) sparse matrix for MLFMA.
Only computes entries for BF pairs in neighboring leaf boxes.
Returns a CSC sparse matrix in original BF ordering.
"""
function assemble_mlfma_nearfield(octree::Octree, mesh::TriMesh, rwg::RWGData, k::Float64;
                                   quad_order::Int=3, eta0::Float64=376.730313668)
    N = rwg.nedges
    cache = _build_efie_cache(mesh, rwg, k; quad_order=quad_order, eta0=eta0)
    leaf_level = octree.levels[octree.nLevels]

    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]

    for box in leaf_level.boxes
        for nbr_id in box.neighbors
            nbr_box = leaf_level.boxes[nbr_id]
            # Compute all entries Z[m, n] for m ∈ box, n ∈ nbr_box
            for m_perm in box.bf_range
                m = octree.perm[m_perm]
                for n_perm in nbr_box.bf_range
                    n = octree.perm[n_perm]
                    val = _efie_entry(cache, m, n)
                    push!(rows, m)
                    push!(cols, n)
                    push!(vals, val)
                end
            end
        end
    end

    return sparse(rows, cols, vals, N, N)
end

# ─── Spectral filter for disaggregation ────────────────────────

"""
    _build_spectral_theta(src_samp, tgt_samp, L_trunc)

Build a dense spectral transfer matrix for θ direction.
Performs Legendre analysis at source GL nodes then synthesis at target GL nodes,
retaining only modes 0:L_trunc.

    T[tgt_i, src_j] = w_src_j * Σ_{l=0}^{L_trunc} ((2l+1)/2) P_l(x_src_j) P_l(x_tgt_i)

Used for BOTH:
- Disaggregation (parent→child): src=parent, tgt=child, L_trunc=L_child
- Aggregation   (child→parent): src=child,  tgt=parent, L_trunc=L_child
"""
function _build_spectral_theta(src_samp::SphereSampling, tgt_samp::SphereSampling, L_trunc::Int)
    nθs = src_samp.ntheta
    nθt = tgt_samp.ntheta

    # Use GL nodes directly in data order (ascending x = descending θ).
    # This ensures F[i,j] matches the data layout where it=1 → first GL node.
    src_gl_nodes, src_gl_weights = gauss_legendre(nθs)
    tgt_gl_nodes, _ = gauss_legendre(nθt)

    F = zeros(Float64, nθt, nθs)
    for j in 1:nθs
        xj = src_gl_nodes[j]
        wj = src_gl_weights[j]
        Pj = legendre_all(L_trunc, xj)
        for i in 1:nθt
            xi = tgt_gl_nodes[i]
            Pi = legendre_all(L_trunc, xi)
            val = 0.0
            for l in 0:L_trunc
                val += (2l + 1) / 2.0 * Pi[l + 1] * Pj[l + 1]
            end
            F[i, j] = wj * val
        end
    end
    return F
end

"""
    _build_spectral_phi(src_nphi, tgt_nphi, src_phi, tgt_phi, M_trunc)

Build a dense spectral transfer matrix for φ direction.
Performs Fourier analysis on source uniform grid then synthesis on target uniform grid,
retaining only modes 0:M_trunc.

Used for BOTH disaggregation and aggregation.
"""
function _build_spectral_phi(src_nphi::Int, tgt_nphi::Int,
                              src_phi::Vector{Float64}, tgt_phi::Vector{Float64},
                              M_trunc::Int)
    F = zeros(Float64, tgt_nphi, src_nphi)
    for j in 1:src_nphi
        for i in 1:tgt_nphi
            delta = tgt_phi[i] - src_phi[j]
            val = 1.0
            for m in 1:M_trunc
                val += 2.0 * cos(m * delta)
            end
            F[i, j] = val / src_nphi
        end
    end
    return F
end

"""
Apply spectral filtering to a (4, npts_src) matrix using separated θ/φ filters.
Used for aggregation (Lagrange interpolation child→parent).
"""
function _filter_2step(data::Matrix{ComplexF64},
                        src::SphereSampling, tgt::SphereSampling,
                        F_theta::Matrix{Float64}, F_phi::Matrix{Float64})
    nθs, nφs = src.ntheta, src.nphi
    nθt, nφt = tgt.ntheta, tgt.nphi

    # Step 1: φ filtering (for each θ row)
    mid = zeros(ComplexF64, 4, nθs, nφt)
    for it in 1:nθs
        for c in 1:4
            for ip_tgt in 1:nφt
                val = zero(ComplexF64)
                for ip_src in 1:nφs
                    val += F_phi[ip_tgt, ip_src] * data[c, (it - 1) * nφs + ip_src]
                end
                mid[c, it, ip_tgt] = val
            end
        end
    end

    # Step 2: θ filtering (for each φ column)
    result = zeros(ComplexF64, 4, nθt * nφt)
    for ip in 1:nφt
        for c in 1:4
            for it_tgt in 1:nθt
                val = zero(ComplexF64)
                for it_src in 1:nθs
                    val += F_theta[it_tgt, it_src] * mid[c, it_src, ip]
                end
                result[c, (it_tgt - 1) * nφt + ip] = val
            end
        end
    end

    return result
end

"""
    associated_legendre_m_all(l_max, m, x)

Compute P_l^m(x) for l = m, m+1, ..., l_max using stable upward recurrence.
Uses unnormalized (Ferrer) associated Legendre functions without Condon-Shortley phase.
"""
function associated_legendre_m_all(l_max::Int, m::Int, x::Float64)
    m >= 0 || error("m must be non-negative")
    l_max >= m || error("l_max must be >= m")

    P = Vector{Float64}(undef, l_max - m + 1)

    # P_m^m(x) = (2m-1)!! * (1-x²)^{m/2}
    pmm = 1.0
    fact = 1.0
    omx2 = sqrt(max(1.0 - x * x, 0.0))
    for i in 1:m
        pmm *= fact * omx2
        fact += 2.0
    end
    P[1] = pmm

    if l_max == m
        return P
    end

    # P_{m+1}^m(x) = x * (2m+1) * P_m^m(x)
    pmm1 = x * (2m + 1) * pmm
    P[2] = pmm1

    # Upward recurrence: (l-m) P_l^m = (2l-1) x P_{l-1}^m - (l+m-1) P_{l-2}^m
    for l in (m + 2):l_max
        pll = ((2l - 1) * x * P[l - m] - (l + m - 1) * P[l - m - 1]) / (l - m)
        P[l - m + 1] = pll
    end

    return P
end

"""
    _build_theta_filter_m(src_samp, tgt_samp, L_trunc, m)

Build the θ spectral filter matrix for azimuthal mode m.
Uses associated Legendre functions P_l^m for proper band-limiting of mode m.

    Ft_m[i,j] = wθ_j * Σ_{l=m}^{L_trunc} (2l+1)/2 * [(l-m)!/(l+m)!] * P_l^m(x_i) * P_l^m(x_j)

For m=0, this reduces to the standard Legendre filter.
"""
function _build_theta_filter_m(src_samp::SphereSampling, tgt_samp::SphereSampling,
                                L_trunc::Int, m::Int)
    nθs = src_samp.ntheta
    nθt = tgt_samp.ntheta

    src_gl_nodes, src_gl_weights = gauss_legendre(nθs)
    tgt_gl_nodes, _ = gauss_legendre(nθt)

    # Precompute normalization factors: (l-m)!/(l+m)! for l = m, ..., L_trunc
    norm_fac = Vector{Float64}(undef, L_trunc - m + 1)
    for l in m:L_trunc
        # (l-m)!/(l+m)! = 1/[(l-m+1)(l-m+2)...(l+m)]
        fac = 1.0
        for j in (l - m + 1):(l + m)
            fac *= j
        end
        norm_fac[l - m + 1] = 1.0 / fac
    end

    F = zeros(Float64, nθt, nθs)
    for j in 1:nθs
        xj = src_gl_nodes[j]
        wj = src_gl_weights[j]
        Pj = associated_legendre_m_all(L_trunc, m, xj)
        for i in 1:nθt
            xi = tgt_gl_nodes[i]
            Pi = associated_legendre_m_all(L_trunc, m, xi)
            val = 0.0
            for l in m:L_trunc
                idx = l - m + 1
                val += (2l + 1) / 2.0 * norm_fac[idx] * Pi[idx] * Pj[idx]
            end
            F[i, j] = wj * val
        end
    end
    return F
end

"""
    _build_disagg_filters_all_m(parent_samp, child_samp, L_child)

Build per-m θ filter matrices for disaggregation.
Returns a Vector{Matrix{Float64}} indexed by m+1 (m = 0, ..., M_child),
each of size (nθ_child × nθ_parent).
"""
function _build_disagg_filters_all_m(parent_samp::SphereSampling,
                                      child_samp::SphereSampling,
                                      L_child::Int)
    M_child = div(child_samp.nphi, 2)
    M_eff = min(M_child, L_child)  # can't have m > L_child

    filters = Vector{Matrix{Float64}}(undef, M_eff + 1)
    for m in 0:M_eff
        filters[m + 1] = _build_theta_filter_m(parent_samp, child_samp, L_child, m)
    end
    return filters
end

"""
Apply disaggregation spectral filter using per-m θ filters.
Decomposes data into φ Fourier modes, applies P_l^m band-limiting for each m,
then reconstructs at child φ points. Correctly handles all m modes.

data: (4, npts_parent) → result: (4, npts_child)
"""
function _apply_disagg_filter(data::Matrix{ComplexF64},
                               filters_m::Vector{Matrix{Float64}},
                               parent_samp::SphereSampling,
                               child_samp::SphereSampling)
    nθp, nφp = parent_samp.ntheta, parent_samp.nphi
    nθc, nφc = child_samp.ntheta, child_samp.nphi
    M_eff = length(filters_m) - 1  # max m index

    # φ values (uniform grid with half-sample offset)
    parent_phi = [(j - 0.5) * 2π / nφp for j in 1:nφp]
    child_phi  = [(j - 0.5) * 2π / nφc for j in 1:nφc]

    # Step 1: DFT analysis in φ at parent sampling
    # For each θ_s and component c, compute Fourier coefficients a_m, b_m
    # a_0 = (1/nφ) Σ_j f(φ_j)
    # a_m = (2/nφ) Σ_j f(φ_j) cos(mφ_j)  for m > 0
    # b_m = (2/nφ) Σ_j f(φ_j) sin(mφ_j)  for m > 0
    a_coeff = zeros(ComplexF64, 4, nθp, M_eff + 1)  # a_m[c, θ, m+1]
    b_coeff = zeros(ComplexF64, 4, nθp, M_eff + 1)  # b_m[c, θ, m+1]

    for it in 1:nθp
        for ip in 1:nφp
            q = (it - 1) * nφp + ip
            φ = parent_phi[ip]
            @inbounds for c in 1:4
                dval = data[c, q]
                a_coeff[c, it, 1] += dval  # m=0
                for m in 1:M_eff
                    a_coeff[c, it, m + 1] += dval * cos(m * φ)
                    b_coeff[c, it, m + 1] += dval * sin(m * φ)
                end
            end
        end
        # Normalize
        @inbounds for c in 1:4
            a_coeff[c, it, 1] /= nφp
            for m in 1:M_eff
                a_coeff[c, it, m + 1] *= 2.0 / nφp
                b_coeff[c, it, m + 1] *= 2.0 / nφp
            end
        end
    end

    # Step 2: Apply per-m θ filters
    a_out = zeros(ComplexF64, 4, nθc, M_eff + 1)
    b_out = zeros(ComplexF64, 4, nθc, M_eff + 1)

    for m in 0:M_eff
        Ft = filters_m[m + 1]  # (nθc, nθp)
        for it_t in 1:nθc
            @inbounds for c in 1:4
                va = zero(ComplexF64)
                vb = zero(ComplexF64)
                for it_s in 1:nθp
                    w = Ft[it_t, it_s]
                    va += w * a_coeff[c, it_s, m + 1]
                    if m > 0
                        vb += w * b_coeff[c, it_s, m + 1]
                    end
                end
                a_out[c, it_t, m + 1] = va
                b_out[c, it_t, m + 1] = vb
            end
        end
    end

    # Step 3: IDFT synthesis at child φ points
    result = zeros(ComplexF64, 4, nθc * nφc)
    for it in 1:nθc
        for ip in 1:nφc
            q = (it - 1) * nφc + ip
            φ = child_phi[ip]
            @inbounds for c in 1:4
                val = a_out[c, it, 1]  # m=0
                for m in 1:M_eff
                    val += a_out[c, it, m + 1] * cos(m * φ) +
                           b_out[c, it, m + 1] * sin(m * φ)
                end
                result[c, q] = val
            end
        end
    end

    return result
end

"""
Pre-allocated scratch arrays for `_apply_disagg_filter!`, sized for a
specific parent↔child level transition.
"""
struct DisaggFilterScratch
    a_coeff::Array{ComplexF64,3}   # (4, nθ_parent, M_eff+1)
    b_coeff::Array{ComplexF64,3}   # (4, nθ_parent, M_eff+1)
    a_out::Array{ComplexF64,3}     # (4, nθ_child,  M_eff+1)
    b_out::Array{ComplexF64,3}     # (4, nθ_child,  M_eff+1)
    parent_phi::Vector{Float64}    # uniform φ grid at parent sampling
    child_phi::Vector{Float64}     # uniform φ grid at child sampling
end

function DisaggFilterScratch(parent_samp::SphereSampling,
                              child_samp::SphereSampling,
                              M_eff::Int)
    nθp, nφp = parent_samp.ntheta, parent_samp.nphi
    nθc, nφc = child_samp.ntheta, child_samp.nphi
    return DisaggFilterScratch(
        zeros(ComplexF64, 4, nθp, M_eff + 1),
        zeros(ComplexF64, 4, nθp, M_eff + 1),
        zeros(ComplexF64, 4, nθc, M_eff + 1),
        zeros(ComplexF64, 4, nθc, M_eff + 1),
        [(j - 0.5) * 2π / nφp for j in 1:nφp],
        [(j - 0.5) * 2π / nφc for j in 1:nφc],
    )
end

"""
In-place variant of `_apply_disagg_filter` that writes into pre-allocated
`result` (4, nθc*nφc) and uses `scratch` buffers to avoid heap allocations.
"""
function _apply_disagg_filter!(result::Matrix{ComplexF64},
                                data::Matrix{ComplexF64},
                                filters_m::Vector{Matrix{Float64}},
                                parent_samp::SphereSampling,
                                child_samp::SphereSampling,
                                scratch::DisaggFilterScratch)
    nθp, nφp = parent_samp.ntheta, parent_samp.nphi
    nθc, nφc = child_samp.ntheta, child_samp.nphi
    M_eff = length(filters_m) - 1

    a_coeff = scratch.a_coeff
    b_coeff = scratch.b_coeff
    a_out   = scratch.a_out
    b_out   = scratch.b_out

    fill!(a_coeff, zero(ComplexF64))
    fill!(b_coeff, zero(ComplexF64))
    fill!(a_out, zero(ComplexF64))
    fill!(b_out, zero(ComplexF64))
    fill!(result, zero(ComplexF64))

    # Step 1: DFT analysis in φ at parent sampling
    for it in 1:nθp
        for ip in 1:nφp
            q = (it - 1) * nφp + ip
            φ = scratch.parent_phi[ip]
            @inbounds for c in 1:4
                dval = data[c, q]
                a_coeff[c, it, 1] += dval
                for m in 1:M_eff
                    a_coeff[c, it, m + 1] += dval * cos(m * φ)
                    b_coeff[c, it, m + 1] += dval * sin(m * φ)
                end
            end
        end
        @inbounds for c in 1:4
            a_coeff[c, it, 1] /= nφp
            for m in 1:M_eff
                a_coeff[c, it, m + 1] *= 2.0 / nφp
                b_coeff[c, it, m + 1] *= 2.0 / nφp
            end
        end
    end

    # Step 2: Apply per-m θ filters
    for m in 0:M_eff
        Ft = filters_m[m + 1]
        for it_t in 1:nθc
            @inbounds for c in 1:4
                va = zero(ComplexF64)
                vb = zero(ComplexF64)
                for it_s in 1:nθp
                    w = Ft[it_t, it_s]
                    va += w * a_coeff[c, it_s, m + 1]
                    if m > 0
                        vb += w * b_coeff[c, it_s, m + 1]
                    end
                end
                a_out[c, it_t, m + 1] = va
                b_out[c, it_t, m + 1] = vb
            end
        end
    end

    # Step 3: IDFT synthesis at child φ points
    for it in 1:nθc
        for ip in 1:nφc
            q = (it - 1) * nφc + ip
            φ = scratch.child_phi[ip]
            @inbounds for c in 1:4
                val = a_out[c, it, 1]
                for m in 1:M_eff
                    val += a_out[c, it, m + 1] * cos(m * φ) +
                           b_out[c, it, m + 1] * sin(m * φ)
                end
                result[c, q] = val
            end
        end
    end

    return result
end

# ─── Workspace structs for allocation-free mul! ────────────────

"""
Pre-allocated workspace for MLFMA `mul!`, eliminating per-call heap
allocations. Created once by `build_mlfma_operator` and reused in every
matvec.
"""
mutable struct MLFMAWorkspace
    # agg[idx][box] = (4, npts) buffer, idx = 1..nL-1 → octree level idx+1
    agg::Vector{Vector{Matrix{ComplexF64}}}
    # incoming[idx][box] = (4, npts) buffer, same indexing
    incoming::Vector{Vector{Matrix{ComplexF64}}}
    # Per-level-transition scratch for _apply_disagg_filter! (aggregation direction)
    agg_disagg_scratch::Vector{DisaggFilterScratch}
    # Per-level-transition scratch for _apply_disagg_filter! (disaggregation direction)
    disagg_disagg_scratch::Vector{DisaggFilterScratch}
    # Per-level-transition scratch: interpolation result (4, parent_npts) for aggregation
    interp_result::Vector{Matrix{ComplexF64}}
    # Per-level-transition scratch: shifted copy (4, parent_npts) for disaggregation
    shifted_buf::Vector{Matrix{ComplexF64}}
    # Per-level-transition scratch: filter result (4, child_npts) for disaggregation
    filter_result::Vector{Matrix{ComplexF64}}
end

function _build_mlfma_workspace(octree::Octree,
                                 samplings::Vector{SphereSampling},
                                 agg_filters::Vector{Vector{Matrix{Float64}}},
                                 disagg_filters::Vector{Vector{Matrix{Float64}}})
    nL = octree.nLevels

    # Pre-allocate agg and incoming with correct sizes per level
    agg = Vector{Vector{Matrix{ComplexF64}}}(undef, nL - 1)
    incoming = Vector{Vector{Matrix{ComplexF64}}}(undef, nL - 1)
    for idx in 1:(nL - 1)
        level = idx + 1  # octree level
        nboxes = length(octree.levels[level].boxes)
        npts = samplings[idx].npts
        agg[idx] = [zeros(ComplexF64, 4, npts) for _ in 1:nboxes]
        incoming[idx] = [zeros(ComplexF64, 4, npts) for _ in 1:nboxes]
    end

    # Per-level-transition scratch for disagg filters
    n_transitions = max(nL - 2, 0)  # number of level transitions (levels 2..nL-1)
    agg_disagg_scratch = Vector{DisaggFilterScratch}(undef, n_transitions)
    disagg_disagg_scratch = Vector{DisaggFilterScratch}(undef, n_transitions)
    interp_result_vec = Vector{Matrix{ComplexF64}}(undef, n_transitions)
    shifted_buf_vec = Vector{Matrix{ComplexF64}}(undef, n_transitions)
    filter_result_vec = Vector{Matrix{ComplexF64}}(undef, n_transitions)

    for i in 1:n_transitions
        # interp_idx = i corresponds to l-1 where l goes from 2 to nL-1
        # child_samp = samplings[i+1], parent_samp = samplings[i]
        child_samp = samplings[i + 1]
        parent_samp = samplings[i]

        # Aggregation filters: child→parent
        if i <= length(agg_filters)
            M_eff_agg = length(agg_filters[i]) - 1
        else
            M_eff_agg = 0
        end
        agg_disagg_scratch[i] = DisaggFilterScratch(child_samp, parent_samp, M_eff_agg)

        # Disaggregation filters: parent→child
        if i <= length(disagg_filters)
            M_eff_disagg = length(disagg_filters[i]) - 1
        else
            M_eff_disagg = 0
        end
        disagg_disagg_scratch[i] = DisaggFilterScratch(parent_samp, child_samp, M_eff_disagg)

        # Interpolation result: (4, parent_npts) for aggregation
        interp_result_vec[i] = zeros(ComplexF64, 4, parent_samp.npts)
        # Shifted buffer: (4, parent_npts) for disaggregation
        shifted_buf_vec[i] = zeros(ComplexF64, 4, parent_samp.npts)
        # Filter result: (4, child_npts) for disaggregation
        filter_result_vec[i] = zeros(ComplexF64, 4, child_samp.npts)
    end

    return MLFMAWorkspace(agg, incoming,
                           agg_disagg_scratch, disagg_disagg_scratch,
                           interp_result_vec, shifted_buf_vec, filter_result_vec)
end

# ─── MLFMAOperator ──────────────────────────────────────────────

struct MLFMAOperator <: AbstractMatrix{ComplexF64}
    octree::Octree
    Z_near::SparseMatrixCSC{ComplexF64,Int}
    k::Float64
    eta0::Float64
    prefactor::ComplexF64
    samplings::Vector{SphereSampling}        # indexed by level (2:nLevels), so samplings[l-1]
    trans_factors::Vector{Dict{NTuple{3,Int}, Vector{ComplexF64}}}  # indexed same
    bf_patterns::Array{ComplexF64,3}         # (4, npts_leaf, N)
    interp_theta::Vector{Matrix{Float64}}    # aggregation: Lagrange interp child→parent (θ) [unused, kept for compat]
    interp_phi::Vector{Matrix{Float64}}      # aggregation: Lagrange interp child→parent (φ) [unused, kept for compat]
    agg_filters::Vector{Vector{Matrix{Float64}}}    # aggregation: per-m θ filters [level][m+1]
    disagg_filters::Vector{Vector{Matrix{Float64}}}  # disaggregation: per-m θ filters [level][m+1]
    N::Int
    workspace::MLFMAWorkspace               # pre-allocated buffers for mul!
end

struct MLFMAAdjointOperator <: AbstractMatrix{ComplexF64}
    op::MLFMAOperator
end

Base.size(A::MLFMAOperator) = (A.N, A.N)
Base.eltype(::MLFMAOperator) = ComplexF64
Base.size(A::MLFMAAdjointOperator) = size(A.op)
Base.eltype(::MLFMAAdjointOperator) = ComplexF64
LinearAlgebra.adjoint(A::MLFMAOperator) = MLFMAAdjointOperator(A)

# Fallback getindex via near-field (for preconditioner construction)
function Base.getindex(A::MLFMAOperator, i::Int, j::Int)
    # Return near-field entry if available, else 0
    return A.Z_near[i, j]
end

function Base.getindex(A::MLFMAAdjointOperator, i::Int, j::Int)
    return conj(A.op.Z_near[j, i])
end

"""
    build_mlfma_operator(mesh, rwg, k; kwargs...)

Build an MLFMA operator for the EFIE system.

# Arguments
- `leaf_lambda=0.25`: leaf box edge length in wavelengths
- `quad_order=3`: surface quadrature order
- `precision=3`: translation truncation precision parameter
- `eta0=376.730313668`: free-space impedance
- `verbose=false`: print progress
"""
function build_mlfma_operator(mesh::TriMesh, rwg::RWGData, k::Float64;
                               leaf_lambda::Float64=0.25,
                               quad_order::Int=3,
                               precision::Int=3,
                               eta0::Float64=376.730313668,
                               verbose::Bool=false)
    N = rwg.nedges
    centers = rwg_centers(mesh, rwg)

    # 1. Build octree
    verbose && print("  MLFMA: Building octree... ")
    t0 = time()
    octree = build_octree(centers, k; leaf_lambda=leaf_lambda)
    verbose && println("$(round(time()-t0, digits=2))s, $(octree.nLevels) levels, " *
                       "$(length(octree.levels[octree.nLevels].boxes)) leaf boxes")

    # 2. Near-field matrix
    verbose && print("  MLFMA: Assembling near-field... ")
    t0 = time()
    Z_near = assemble_mlfma_nearfield(octree, mesh, rwg, k;
                                       quad_order=quad_order, eta0=eta0)
    nnz_ratio = nnz(Z_near) / N^2
    verbose && println("$(round(time()-t0, digits=2))s, nnz=$(round(nnz_ratio*100, digits=1))%")

    # 3. Spherical sampling at each level (for levels 2:nLevels)
    nL = octree.nLevels
    samplings = Vector{SphereSampling}(undef, nL - 1)
    for l in 2:nL
        edge_l = octree.levels[l].edge_length
        L_l = truncation_order(edge_l, k; precision=precision)
        samplings[l - 1] = make_sphere_sampling(L_l)
    end
    verbose && println("  MLFMA: Sampling — leaf L=$(samplings[end].L), " *
                       "npts=$(samplings[end].npts)")

    # 4. BF radiation patterns at leaf level
    verbose && print("  MLFMA: Computing radiation patterns... ")
    t0 = time()
    leaf_sampling = samplings[nL - 1]
    bf_patterns = compute_bf_radiation_patterns(mesh, rwg, k, octree, leaf_sampling;
                                                 quad_order=quad_order)
    verbose && println("$(round(time()-t0, digits=2))s")

    # 5. Translation factors at each level
    verbose && print("  MLFMA: Computing translation factors... ")
    t0 = time()
    trans_factors = Vector{Dict{NTuple{3,Int}, Vector{ComplexF64}}}(undef, nL - 1)
    for l in 2:nL
        trans_factors[l - 1] = precompute_translation_factors(
            octree.levels[l], k, samplings[l - 1])
    end
    verbose && println("$(round(time()-t0, digits=2))s")

    # 6. Interpolation (aggregation) and spectral filter (disaggregation) matrices
    #    Aggregation uses Lagrange interpolation (child→parent, upsampling).
    #    Disaggregation uses addition-theorem spectral filter (parent→child, band-limiting).
    interp_theta = Vector{Matrix{Float64}}()
    interp_phi = Vector{Matrix{Float64}}()
    agg_filters = Vector{Vector{Matrix{Float64}}}()
    disagg_filters = Vector{Vector{Matrix{Float64}}}()

    if nL > 2
        verbose && print("  MLFMA: Building interpolation/filter matrices... ")
        t0 = time()
        for l in 2:nL-1
            child_samp = samplings[l]      # level l+1 (finer)
            parent_samp = samplings[l - 1]  # level l (coarser)

            # Aggregation: Lagrange interp (kept as fallback, not used)
            It_sp, Ip_sp = build_interp_matrices(parent_samp, child_samp; order=6)
            push!(interp_theta, Matrix(It_sp))
            push!(interp_phi, Matrix(Ip_sp))

            # Per-m spectral filters using associated Legendre P_l^m
            L_child = child_samp.L
            # Aggregation: child → parent (spectral interp, L_trunc = L_child)
            push!(agg_filters, _build_disagg_filters_all_m(child_samp, parent_samp, L_child))
            # Disaggregation: parent → child (spectral filter, L_trunc = L_child)
            push!(disagg_filters, _build_disagg_filters_all_m(parent_samp, child_samp, L_child))
        end
        verbose && println("$(round(time()-t0, digits=2))s")
    end

    prefactor = -k^2 * eta0 / (16π^2)

    workspace = _build_mlfma_workspace(octree, samplings, agg_filters, disagg_filters)

    return MLFMAOperator(octree, Z_near, k, eta0, prefactor,
                          samplings, trans_factors, bf_patterns,
                          interp_theta, interp_phi, agg_filters, disagg_filters, N,
                          workspace)
end

# ─── Phase shift helper ─────────────────────────────────────────

"""
Apply phase shift exp(±jk k̂·d) to a 4×npts aggregation/incoming array.
`sign` should be +1.0 for aggregation (sending) or -1.0 for disaggregation.
"""
function _apply_phase_shift!(data::Matrix{ComplexF64}, sampling::SphereSampling,
                              k::Float64, d::Vec3, sign::Float64)
    for q in 1:sampling.npts
        kh = Vec3(sampling.khat[1, q], sampling.khat[2, q], sampling.khat[3, q])
        phase = exp(im * sign * k * dot(kh, d))
        @inbounds for c in 1:4
            data[c, q] *= phase
        end
    end
end

# ─── Two-step interpolation / anterpolation helpers ────────────

"""
Interpolate a (4, npts_src) matrix from source to target sampling.
Applies φ interpolation first, then θ (matching MoM_Kernels convention).
Data layout: (4, ntheta * nphi) with θ as the outer index.
"""
function _interp_2step(data::Matrix{ComplexF64},
                        src::SphereSampling, tgt::SphereSampling,
                        I_theta::SparseMatrixCSC{Float64,Int},
                        I_phi::SparseMatrixCSC{Float64,Int})
    nθs, nφs = src.ntheta, src.nphi
    nθt, nφt = tgt.ntheta, tgt.nphi

    # Step 1: φ interpolation (for each θ row)
    mid = zeros(ComplexF64, 4, nθs, nφt)
    for it in 1:nθs
        for c in 1:4
            src_row = [data[c, (it - 1) * nφs + ip] for ip in 1:nφs]
            tgt_row = I_phi * src_row
            for ip in 1:nφt
                mid[c, it, ip] = tgt_row[ip]
            end
        end
    end

    # Step 2: θ interpolation (for each φ column)
    result = zeros(ComplexF64, 4, nθt * nφt)
    for ip in 1:nφt
        for c in 1:4
            src_col = [mid[c, it, ip] for it in 1:nθs]
            tgt_col = I_theta * src_col
            for it in 1:nθt
                result[c, (it - 1) * nφt + ip] = tgt_col[it]
            end
        end
    end

    return result
end

"""
Anterpolate (transpose interpolation) a (4, npts_src) matrix.
Applies θ^T first, then φ^T (reverse order of interpolation, matching
the standard MLFMA convention for disaggregation).
"""
function _anterp_2step(data::Matrix{ComplexF64},
                        src::SphereSampling, tgt::SphereSampling,
                        I_theta::SparseMatrixCSC{Float64,Int},
                        I_phi::SparseMatrixCSC{Float64,Int})
    nθs, nφs = src.ntheta, src.nphi
    nθt, nφt = tgt.ntheta, tgt.nphi

    # Step 1: θ^T anterpolation (for each φ column)
    # I_theta is (nθ_coarse × nθ_fine), so I_theta' is (nθ_fine × nθ_coarse)
    mid = zeros(ComplexF64, 4, nθt, nφs)
    for ip in 1:nφs
        for c in 1:4
            src_col = [data[c, (it - 1) * nφs + ip] for it in 1:nθs]
            tgt_col = I_theta' * src_col
            for it in 1:nθt
                mid[c, it, ip] = tgt_col[it]
            end
        end
    end

    # Step 2: φ^T anterpolation (for each θ row)
    # I_phi is (nφ_coarse × nφ_fine), so I_phi' is (nφ_fine × nφ_coarse)
    result = zeros(ComplexF64, 4, nθt * nφt)
    for it in 1:nθt
        for c in 1:4
            src_row = [mid[c, it, ip] for ip in 1:nφs]
            tgt_row = I_phi' * src_row
            for ip in 1:nφt
                result[c, (it - 1) * nφt + ip] = tgt_row[ip]
            end
        end
    end

    return result
end

# ─── Forward matvec ─────────────────────────────────────────────

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64}, A::MLFMAOperator,
                             x::AbstractVector)
    N = A.N
    nL = A.octree.nLevels
    length(x) == N || throw(DimensionMismatch("x length $(length(x)) != $N"))
    length(y) == N || throw(DimensionMismatch("y length $(length(y)) != $N"))

    ws = A.workspace
    agg = ws.agg
    incoming = ws.incoming

    # 1. Near-field
    mul!(y, A.Z_near, x)

    # 2. Aggregation at leaf level
    leaf_level = A.octree.levels[nL]
    leaf_samp = A.samplings[nL - 1]

    # Zero leaf agg buffers and aggregate
    for (bi, box) in enumerate(leaf_level.boxes)
        a = agg[nL - 1][bi]
        fill!(a, zero(ComplexF64))
        for n_perm in box.bf_range
            n = A.octree.perm[n_perm]
            xn = x[n]
            if abs(xn) > 0
                @inbounds for q in 1:leaf_samp.npts
                    for c in 1:4
                        a[c, q] += xn * A.bf_patterns[c, q, n]
                    end
                end
            end
        end
    end

    # Bottom-up aggregation: spectral filter, then phase shift
    for l in (nL - 1):-1:2
        parent_level = A.octree.levels[l]
        child_level = A.octree.levels[l + 1]
        parent_samp = A.samplings[l - 1]
        child_samp = A.samplings[l]
        nboxes_p = length(parent_level.boxes)
        interp_idx = l - 1

        # Zero parent agg buffers
        for bi in 1:nboxes_p
            fill!(agg[l - 1][bi], zero(ComplexF64))
        end

        for (ci, cbox) in enumerate(child_level.boxes)
            pid = cbox.parent
            pid > 0 || continue
            pbox = parent_level.boxes[pid]
            d = cbox.center - pbox.center

            # Step 1: Spectral interpolation from child to parent sampling
            interp_buf = ws.interp_result[interp_idx]
            if !isempty(A.agg_filters) && interp_idx <= length(A.agg_filters)
                _apply_disagg_filter!(interp_buf, agg[l][ci],
                                       A.agg_filters[interp_idx],
                                       child_samp, parent_samp,
                                       ws.agg_disagg_scratch[interp_idx])
            else
                copyto!(interp_buf, agg[l][ci])
            end

            # Step 2: Phase shift child → parent center (at parent sampling)
            _apply_phase_shift!(interp_buf, parent_samp, A.k, d, 1.0)

            agg[l - 1][pid] .+= interp_buf
        end
    end

    # 3. Translation at each level
    for l in 2:nL
        level = A.octree.levels[l]
        samp = A.samplings[l - 1]
        nboxes = length(level.boxes)

        # Zero incoming buffers
        for bi in 1:nboxes
            fill!(incoming[l - 1][bi], zero(ComplexF64))
        end

        tf = A.trans_factors[l - 1]
        for (bi, box) in enumerate(level.boxes)
            dst = incoming[l - 1][bi]
            for il_id in box.interaction_list
                il_box = level.boxes[il_id]
                dijk = (box.ijk[1] - il_box.ijk[1],
                        box.ijk[2] - il_box.ijk[2],
                        box.ijk[3] - il_box.ijk[3])
                T = tf[dijk]
                src = agg[l - 1][il_id]
                @inbounds for q in 1:samp.npts
                    Tq = T[q]
                    for c in 1:4
                        dst[c, q] += Tq * src[c, q]
                    end
                end
            end
        end
    end

    # 4. Disaggregation (top-down): phase shift first, then spectral filter
    for l in 2:nL-1
        parent_level = A.octree.levels[l]
        child_level = A.octree.levels[l + 1]
        parent_samp = A.samplings[l - 1]
        interp_idx = l - 1

        for (ci, cbox) in enumerate(child_level.boxes)
            pid = cbox.parent
            pid > 0 || continue
            pbox = parent_level.boxes[pid]
            d = cbox.center - pbox.center

            # Step 1: Phase shift parent → child center (at parent sampling)
            shifted = ws.shifted_buf[interp_idx]
            copyto!(shifted, incoming[l - 1][pid])
            _apply_phase_shift!(shifted, parent_samp, A.k, d, -1.0)

            # Step 2: Spectral filter from parent to child sampling (band-limiting)
            if !isempty(A.disagg_filters) && interp_idx <= length(A.disagg_filters)
                child_samp = A.samplings[l]
                filtered = ws.filter_result[interp_idx]
                _apply_disagg_filter!(filtered, shifted,
                                       A.disagg_filters[interp_idx],
                                       parent_samp, child_samp,
                                       ws.disagg_disagg_scratch[interp_idx])
            else
                filtered = shifted
            end

            # Child level l+1 → incoming index l (since incoming[j] ↔ level j+1)
            incoming[l][ci] .+= filtered
        end
    end

    # 5. Disaggregate to BFs at leaf level
    for (bi, box) in enumerate(leaf_level.boxes)
        for n_perm in box.bf_range
            n = A.octree.perm[n_perm]
            val = zero(ComplexF64)
            inc = incoming[nL - 1][bi]
            @inbounds for q in 1:leaf_samp.npts
                dot4 = zero(ComplexF64)
                for c in 1:3
                    dot4 += conj(A.bf_patterns[c, q, n]) * inc[c, q]
                end
                # Scalar part with minus sign (Z = vec - scl)
                dot4 -= conj(A.bf_patterns[4, q, n]) * inc[4, q]
                val += leaf_samp.weights[q] * dot4
            end
            y[n] += A.prefactor * val
        end
    end

    return y
end

function Base.:*(A::MLFMAOperator, x::AbstractVector)
    y = zeros(ComplexF64, size(A, 1))
    mul!(y, A, Vector{ComplexF64}(x))
    return y
end

# ─── Adjoint matvec ─────────────────────────────────────────────

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64}, A::MLFMAAdjointOperator,
                             x::AbstractVector)
    N = A.op.N
    nL = A.op.octree.nLevels
    length(x) == N || throw(DimensionMismatch("x length $(length(x)) != $N"))
    length(y) == N || throw(DimensionMismatch("y length $(length(y)) != $N"))

    ws = A.op.workspace
    agg = ws.agg
    incoming = ws.incoming

    # Near-field adjoint
    mul!(y, adjoint(A.op.Z_near), x)

    # For the adjoint MLFMA, we swap the roles of aggregation and disaggregation:
    # - Aggregate using the "receiving" patterns (conj of sending)
    # - Translate with conjugated translation factors
    # - Disaggregate using the "sending" patterns

    leaf_level = A.op.octree.levels[nL]
    leaf_samp = A.op.samplings[nL - 1]

    # Aggregation at leaf: using conjugated patterns (receiving role)
    for (bi, box) in enumerate(leaf_level.boxes)
        a = agg[nL - 1][bi]
        fill!(a, zero(ComplexF64))
        for n_perm in box.bf_range
            n = A.op.octree.perm[n_perm]
            xn = x[n]
            if abs(xn) > 0
                @inbounds for q in 1:leaf_samp.npts
                    # Adjoint: aggregate with conj(pattern) for vec, -conj for scalar
                    for c in 1:3
                        a[c, q] += xn * conj(A.op.bf_patterns[c, q, n])
                    end
                    a[4, q] -= xn * conj(A.op.bf_patterns[4, q, n])
                end
            end
        end
    end

    # Bottom-up aggregation (same structure as forward)
    for l in (nL - 1):-1:2
        parent_level = A.op.octree.levels[l]
        child_level = A.op.octree.levels[l + 1]
        parent_samp = A.op.samplings[l - 1]
        child_samp = A.op.samplings[l]
        interp_idx = l - 1
        nboxes_p = length(parent_level.boxes)

        # Zero parent agg buffers
        for bi in 1:nboxes_p
            fill!(agg[l - 1][bi], zero(ComplexF64))
        end

        for (ci, cbox) in enumerate(child_level.boxes)
            pid = cbox.parent
            pid > 0 || continue
            pbox = parent_level.boxes[pid]
            d = cbox.center - pbox.center

            # Step 1: Spectral interpolation from child to parent sampling
            interp_buf = ws.interp_result[interp_idx]
            if !isempty(A.op.agg_filters) && interp_idx <= length(A.op.agg_filters)
                _apply_disagg_filter!(interp_buf, agg[l][ci],
                                       A.op.agg_filters[interp_idx],
                                       child_samp, parent_samp,
                                       ws.agg_disagg_scratch[interp_idx])
            else
                copyto!(interp_buf, agg[l][ci])
            end

            # Step 2: Phase shift child → parent center (at parent sampling)
            _apply_phase_shift!(interp_buf, parent_samp, A.op.k, d, 1.0)

            agg[l - 1][pid] .+= interp_buf
        end
    end

    # Translation with conjugated factors
    for l in 2:nL
        level = A.op.octree.levels[l]
        samp = A.op.samplings[l - 1]
        nboxes = length(level.boxes)

        # Zero incoming buffers
        for bi in 1:nboxes
            fill!(incoming[l - 1][bi], zero(ComplexF64))
        end

        tf = A.op.trans_factors[l - 1]
        for (bi, box) in enumerate(level.boxes)
            dst = incoming[l - 1][bi]
            for il_id in box.interaction_list
                il_box = level.boxes[il_id]
                dijk = (box.ijk[1] - il_box.ijk[1],
                        box.ijk[2] - il_box.ijk[2],
                        box.ijk[3] - il_box.ijk[3])
                T = tf[dijk]
                src = agg[l - 1][il_id]
                @inbounds for q in 1:samp.npts
                    Tq = conj(T[q])  # conjugate for adjoint
                    for c in 1:4
                        dst[c, q] += Tq * src[c, q]
                    end
                end
            end
        end
    end

    # Disaggregation (top-down): phase shift first, then spectral filter
    for l in 2:nL-1
        parent_level = A.op.octree.levels[l]
        child_level = A.op.octree.levels[l + 1]
        parent_samp = A.op.samplings[l - 1]
        interp_idx = l - 1

        for (ci, cbox) in enumerate(child_level.boxes)
            pid = cbox.parent
            pid > 0 || continue
            pbox = parent_level.boxes[pid]
            d = cbox.center - pbox.center

            # Step 1: Phase shift parent → child center (at parent sampling)
            shifted = ws.shifted_buf[interp_idx]
            copyto!(shifted, incoming[l - 1][pid])
            _apply_phase_shift!(shifted, parent_samp, A.op.k, d, -1.0)

            # Step 2: Spectral filter from parent to child sampling (band-limiting)
            if !isempty(A.op.disagg_filters) && interp_idx <= length(A.op.disagg_filters)
                child_samp = A.op.samplings[l]
                filtered = ws.filter_result[interp_idx]
                _apply_disagg_filter!(filtered, shifted,
                                       A.op.disagg_filters[interp_idx],
                                       parent_samp, child_samp,
                                       ws.disagg_disagg_scratch[interp_idx])
            else
                filtered = shifted
            end

            # Child level l+1 → incoming index l (since incoming[j] ↔ level j+1)
            incoming[l][ci] .+= filtered
        end
    end

    # Disaggregate to BFs: using the (non-conjugated) patterns
    for (bi, box) in enumerate(leaf_level.boxes)
        for n_perm in box.bf_range
            n = A.op.octree.perm[n_perm]
            val = zero(ComplexF64)
            inc = incoming[nL - 1][bi]
            @inbounds for q in 1:leaf_samp.npts
                dot4 = zero(ComplexF64)
                for c in 1:4
                    dot4 += A.op.bf_patterns[c, q, n] * inc[c, q]
                end
                val += leaf_samp.weights[q] * dot4
            end
            y[n] += conj(A.op.prefactor) * val
        end
    end

    return y
end

function Base.:*(A::MLFMAAdjointOperator, x::AbstractVector)
    y = zeros(ComplexF64, size(A, 1))
    mul!(y, A, Vector{ComplexF64}(x))
    return y
end
