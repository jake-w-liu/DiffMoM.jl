# PlanarGreens.jl — Box-mode weights and Galerkin impedance assembly
#
# For box mode (m,n):  kx = m*pi/a, ky = n*pi/b, kc2 = kx^2 + ky^2.
# Normalized modal vectors for PEC sidewalls, with
#   g1 = cos(kx x) sin(ky y),  g2 = sin(kx x) cos(ky y):
#     e_TE = ( ky g1, -kx g2 ) / N_TE,  N_TE^2 = ky^2 Ic_m Js_n + kx^2 Is_m Jc_n
#     e_TM = ( kx g1,  ky g2 ) / N_TM,  N_TM^2 = kx^2 Ic_m Js_n + ky^2 Is_m Jc_n
# where Ic_m = a (m=0) or a/2, Is_m = 0 (m=0) or a/2 — Neumann-type factors
# that handle the m=0/n=0 degenerate modes without special-casing (the
# norm formula gives N_TM = 0 exactly on the non-existing modes).
#
# PMC sidewalls swap the parity on every axis (E_x uses sin(kx x)cos(ky y),
# E_y uses cos(kx x)sin(ky y)) and swap the TE/TM norm formulas:
#   N_TE^2(PMC) = ky^2 Is_m Jc_n + kx^2 Ic_m Js_n   (m,n >= 1 only)
#   N_TM^2(PMC) = kx^2 Is_m Jc_n + ky^2 Ic_m Js_n   (m,n >= 0, incl. axes)
# so the PMC TE fields carry the PEC TM parity and vice versa.  Wall-port
# half-rooftops transform against sin(kx x) on PMC walls via Hs instead of
# the PEC Hc.
#
# An x-directed rooftop couples only through e_x, a y-directed one through
# e_y, so per-basis modal weights factor:
#   <f_p, e_pol> = w_pol(dir_p, m, n) * fx_p[m] * fy_p[n]
#     x-directed:  w_TE = ky/N_TE,  w_TM = kx/N_TM
#     y-directed:  w_TE = -kx/N_TE, w_TM = ky/N_TM
# A via column (z-directed) couples only through the TM g3 = Ez parity:
#     via: w_TE = 0,  w_TM = fx*fy = Ta (raw transform, norm folded into
#     the per-pair kernel in PlanarVias.jl).
# and the Galerkin impedance between rooftop p (interface f) and q
# (interface s) is the modal sum
#   Z[p,q] = -sum_mn V_pol(f,s;mn) * <f_p,e_pol> * <e_pol,f_q>
# evaluated per level pair through BLAS dgemm on mode-blocked weight
# matrices stored transposed (mode-major) so each basis column is
# contiguous.  The norm factors 1/N_pol^2 are folded into the per-mode
# voltages, so weight blocks contain only kx/ky * pulse transforms.

export PlanarModeGrid, planar_mode_grid
export assemble_planar_z

# ---------------- mode grid ----------------

"""Precomputed box-mode data on a `CellGrid`: wavenumbers `kx/ky`,
cos^2/sin^2 wall integrals, and the analytic pulse transforms used by the
rooftop modal inner products.  `walls` selects the sidewall parity
(`WALL_PEC` or `WALL_PMC`, taken from `grid.walls`)."""
struct PlanarModeGrid
    mx::Int              # modes m = 0..mx-1
    my::Int              # modes n = 0..my-1
    walls::SidewallKind
    ic::Vector{Float64}  # integral cos^2(m pi x/a) over the box
    is::Vector{Float64}  # integral sin^2(m pi x/a)
    jc::Vector{Float64}
    js::Vector{Float64}
    kx::Vector{Float64}
    ky::Vector{Float64}
    tx_tri_cos::Vector{Float64}  # triangle (support 2dx) kernel sinc^2(k dx/2)
    ty_tri_cos::Vector{Float64}  # (same kernel multiplies the sin transform too)
    tx_rect_sin::Vector{Float64} # rectangle (width dx) kernel sinc(k dx/2)
    ty_rect_sin::Vector{Float64}
    hx_cos::Vector{Float64}      # wall half-ramp (1-x/dx) vs cos(kx x)
    hy_cos::Vector{Float64}
    hx_sin::Vector{Float64}      # wall half-ramp vs sin(kx x) (PMC parity)
    hy_sin::Vector{Float64}
end

@inline _sinc0(t::Float64) = abs2(t) < 1e-6 ? 1.0 - t*t/6.0 : sin(t) / t
@inline _sinc0sq(t::Float64) = abs2(t) < 1e-6 ? 1.0 - t*t/3.0 : (sin(t)/t)^2

# half-ramp cos transform: (1-cos(kh))/(k^2 h), limit h/2 at k = 0
@inline function _ramp_cos(k::Float64, h::Float64)
    kh = k * h
    abs2(kh) < 1e-6 && return h * (0.5 - kh * kh / 24.0)
    return (1.0 - cos(kh)) / (k * k * h)
end

# half-ramp sin transform: (kh - sin(kh))/(k^2 h), ~ k h^2/6 at k -> 0
@inline function _ramp_sin(k::Float64, h::Float64)
    kh = k * h
    abs2(kh) < 1e-6 && return k * h * h * (1 / 6 - kh * kh / 120)
    return (kh - sin(kh)) / (k * k * h)
end

"""Build the mode grid for `mx` x-modes and `my` y-modes (indices
m = 0..mx-1, n = 0..my-1) on the box `grid` (sidewall kind from
`grid.walls`)."""
function planar_mode_grid(grid::CellGrid, mx::Integer, my::Integer)
    mx >= 1 && my >= 1 ||
        throw(ArgumentError("mode counts must be >= 1"))
    a, b, dx, dy = grid.a, grid.b, grid.dx, grid.dy
    kx = [m * pi / a for m in 0:mx-1]
    ky = [n * pi / b for n in 0:my-1]
    ic = [m == 0 ? a : a / 2 for m in 0:mx-1]
    isv = [m == 0 ? 0.0 : a / 2 for m in 0:mx-1]
    jc = [n == 0 ? b : b / 2 for n in 0:my-1]
    jsv = [n == 0 ? 0.0 : b / 2 for n in 0:my-1]
    tx_tri_cos = [dx * _sinc0sq(k * dx / 2) for k in kx]
    ty_tri_cos = [dy * _sinc0sq(k * dy / 2) for k in ky]
    tx_rect_sin = [dx * _sinc0(k * dx / 2) for k in kx]
    ty_rect_sin = [dy * _sinc0(k * dy / 2) for k in ky]
    hx_cos = [_ramp_cos(k, dx) for k in kx]
    hy_cos = [_ramp_cos(k, dy) for k in ky]
    hx_sin = [_ramp_sin(k, dx) for k in kx]
    hy_sin = [_ramp_sin(k, dy) for k in ky]
    return PlanarModeGrid(mx, my, grid.walls, ic, isv, jc, jsv, kx, ky,
        tx_tri_cos, ty_tri_cos, tx_rect_sin, ty_rect_sin, hx_cos, hy_cos,
        hx_sin, hy_sin)
end

# ---------------- per-basis separable pulse vectors ----------------

function _basis_fx!(fx::AbstractVector{Float64}, basis::PlanarBasisSet,
        p::Int, mg::PlanarModeGrid, grid::CellGrid)
    kind = _sheet_kind(basis.kind[p])
    nx = grid.nx
    pec = mg.walls == WALL_PEC
    @inbounds if kind == _BASIS_X_FULL
        e = basis.ei[p]
        for m in 1:mg.mx
            # x-flow rooftop transforms vs cos(kx x) on PEC, sin on PMC
            fx[m] = mg.tx_tri_cos[m] *
                (pec ? cospi((m - 1) * e / nx) : sinpi((m - 1) * e / nx))
        end
    elseif kind == _BASIS_X_LO
        copyto!(fx, pec ? mg.hx_cos : mg.hx_sin)
    elseif kind == _BASIS_X_HI
        for m in 1:mg.mx
            # cos: (-1)^(m-1) Hc;  sin: -(-1)^(m-1) Hs
            s = isodd(m - 1) ? -1.0 : 1.0
            fx[m] = pec ? s * mg.hx_cos[m] : -s * mg.hx_sin[m]
        end
    else # y-directed: rectangle vs sin(kx x) on PEC / cos on PMC, at centre
        i = basis.ei[p]
        for m in 1:mg.mx
            fx[m] = mg.tx_rect_sin[m] *
                (pec ? sinpi((m - 1) * (i - 0.5) / nx) :
                       cospi((m - 1) * (i - 0.5) / nx))
        end
    end
    return fx
end

function _basis_fy!(fy::AbstractVector{Float64}, basis::PlanarBasisSet,
        p::Int, mg::PlanarModeGrid, grid::CellGrid)
    kind = _sheet_kind(basis.kind[p])
    ny = grid.ny
    pec = mg.walls == WALL_PEC
    @inbounds if kind == _BASIS_Y_FULL
        f = basis.ej[p]
        for n in 1:mg.my
            fy[n] = mg.ty_tri_cos[n] *
                (pec ? cospi((n - 1) * f / ny) : sinpi((n - 1) * f / ny))
        end
    elseif kind == _BASIS_Y_LO
        copyto!(fy, pec ? mg.hy_cos : mg.hy_sin)
    elseif kind == _BASIS_Y_HI
        for n in 1:mg.my
            s = isodd(n - 1) ? -1.0 : 1.0
            fy[n] = pec ? s * mg.hy_cos[n] : -s * mg.hy_sin[n]
        end
    else # x-directed: rectangle vs sin(ky y) on PEC / cos on PMC, at centre
        j = basis.ej[p]
        for n in 1:mg.my
            fy[n] = mg.ty_rect_sin[n] *
                (pec ? sinpi((n - 1) * (j - 0.5) / ny) :
                       cospi((n - 1) * (j - 0.5) / ny))
        end
    end
    return fy
end

@inline _is_xdir(kind::UInt8) =
    kind == _BASIS_X_FULL || kind == _BASIS_X_LO || kind == _BASIS_X_HI ||
    kind == _BASIS_VX_FULL || kind == _BASIS_VX_LO || kind == _BASIS_VX_HI

# ---------------- impedance assembly ----------------

function _level_pairs(iface::Vector{Int})
    uniq = unique(iface)
    pairs = Tuple{Int,Int}[]
    sizehint!(pairs, length(uniq)^2)
    for f in uniq, s in uniq
        push!(pairs, (f, s))
    end
    return pairs
end

"""
    assemble_planar_z(stack, grid, sheets, basis, omega; kw...) -> Matrix{ComplexF64}

Galerkin impedance matrix for the shielded multilayer problem.  Modes are
processed in column blocks; combined operation-owned workspace (the two real
accumulator matrices, the returned complex matrix, modal weight blocks, and
per-mode cascade state) is bounded by `max_bytes`.  `surface_zs` (scalar or
per-sheet vector) adds the analytic surface-impedance Gram term for
conductor loss.  `vias` lists the `ViaLevel` column levels the basis was
built with (via basis functions index into it through `basis.level`).
"""
function assemble_planar_z(stack::PlanarStackup, grid::CellGrid,
        sheets::Vector{SheetLevel}, basis::PlanarBasisSet, omega::Number;
        vias::Vector{ViaLevel}=ViaLevel[],
        vols::Vector{VolLevel}=VolLevel[],
        mx::Integer=2 * grid.nx, my::Integer=2 * grid.ny,
        block::Integer=512,
        surface_zs=zero(ComplexF64),
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    planar_validate(stack)
    # real(omega) > 0 admits complex-step perturbation omega = w0 + i*eps
    isfinite(omega) && real(omega) > 0 ||
        throw(ArgumentError(
            "omega must be finite with Re > 0, got $omega"))
    surface_zs isa AbstractVector &&
        length(surface_zs) != length(sheets) &&
        throw(ArgumentError(
            "surface_zs length $(length(surface_zs)) != " *
            "sheet count $(length(sheets))"))
    block >= 1 || throw(ArgumentError("block must be >= 1, got $block"))
    nb = planar_basis_count(basis)
    nb == 0 && return zeros(ComplexF64, 0, 0)
    mg = planar_mode_grid(grid, mx, my)
    nmode = mg.mx * mg.my
    L = length(stack.layers)

    # element axis: sheet bases key on their interface, via/volume bases
    # on negative ids from PlanarVias.jl / PlanarVolumes.jl
    iface = Vector{Int}(undef, nb)
    vialayers = Set{Int}()
    vollayers = Set{Int}()
    @inbounds for p in 1:nb
        lv = basis.level[p]
        if _is_vol_kind(basis.kind[p])
            1 <= lv <= length(vols) ||
                throw(ArgumentError("basis $p references missing vol $lv"))
            lay = vols[lv].layer
            1 <= lay <= L || throw(ArgumentError(
                "vol $lv layer $lay outside 1:$L"))
            iface[p] = _vol_elem(lay)
            push!(vollayers, lay)
        elseif _is_via_kind(basis.kind[p])
            1 <= lv <= length(vias) ||
                throw(ArgumentError("basis $p references missing via $lv"))
            lay = vias[lv].layer
            1 <= lay <= L || throw(ArgumentError(
                "via $lv layer $lay outside 1:$L"))
            iface[p] = _via_elem(lay, basis.kind[p])
            push!(vialayers, lay)
        else
            1 <= lv <= length(sheets) ||
                throw(ArgumentError("basis $p references missing sheet $lv"))
            iface[p] = sheets[lv].interface
            0 <= iface[p] <= L ||
                throw(ArgumentError(
                    "sheet $lv interface $(iface[p]) outside 0:$L"))
        end
    end
    vlay = sort!(collect(vialayers))
    volay = sort!(collect(vollayers))
    pairs = _level_pairs(iface)
    npair = length(pairs)
    blk = clamp(Int(block), 1, nmode)

    est = _checked_payload_sum("planar Z",
        _checked_array_payload_bytes(ComplexF64, nb, nb),
        _checked_array_payload_bytes(Float64, nb, nb),
        _checked_array_payload_bytes(Float64, nb, nb),
        _checked_array_payload_bytes(Float64, blk, nb),
        _checked_array_payload_bytes(Float64, blk, nb),
        _checked_array_payload_bytes(ComplexF64, blk, npair),
        _checked_array_payload_bytes(ComplexF64, blk, npair),
        _checked_array_payload_bytes(Float64, mg.mx, nb),
        _checked_array_payload_bytes(Float64, mg.my, nb),
        # per-block cascade workspaces: 2 pols x (2*(L+1) + 2*L) + 3*L
        _checked_array_payload_bytes(ComplexF64, 11 * (L + 1)),
        _checked_array_payload_bytes(Float64, blk, nb),
        _checked_array_payload_bytes(Float64, blk, nb),
        _checked_array_payload_bytes(Int, blk),
        _checked_array_payload_bytes(Int, blk),
        # per-layer via modal state + volume states + element id vector
        _checked_array_payload_bytes(ComplexF64, 10 * max(L, 1)),
        _checked_array_payload_bytes(ComplexF64, 8 * max(L, 1)),
        _checked_array_payload_bytes(Int, nb),
        # per-pair row/col index vectors, each <= nb entries
        _checked_array_payload_bytes(Int, 2 * npair, nb))
    _enforce_payload_limit(est, max_bytes, "planar Z", "max_bytes")

    Zr = zeros(Float64, nb, nb)
    Zi = zeros(Float64, nb, nb)

    # per-basis separable pulse vectors (one pass, reused for every block)
    fxb = Matrix{Float64}(undef, mg.mx, nb)
    fyb = Matrix{Float64}(undef, mg.my, nb)
    @inbounds for p in 1:nb
        _basis_fx!(view(fxb, :, p), basis, p, mg, grid)
        _basis_fy!(view(fyb, :, p), basis, p, mg, grid)
    end

    mlist = fill(-99999, blk)
    nlist = fill(-99999, blk)
    Wte = fill(NaN, blk, nb)
    Wtm = fill(NaN, blk, nb)
    vte = fill(ComplexF64(NaN), blk, npair)
    vtm = fill(ComplexF64(NaN), blk, npair)

    pair_rows = Dict{Tuple{Int,Int},Vector{Int}}()
    pair_cols = Dict{Tuple{Int,Int},Vector{Int}}()
    for (f, s) in pairs
        pair_rows[(f, s)] = findall(==(f), iface)
        pair_cols[(f, s)] = findall(==(s), iface)
    end
    maxc = maximum((p -> length(pair_cols[p])), pairs)
    sdr = fill(NaN, blk, maxc)
    sdi = fill(NaN, blk, maxc)

    c = 0
    for n in 1:mg.my, m in 1:mg.mx
        c += 1
        mlist[c] = m
        nlist[c] = n
        if c == blk || (m == mg.mx && n == mg.my)
            _planar_z_block!(Zr, Zi, stack, omega, mg, grid,
                view(mlist, 1:c), view(nlist, 1:c),
                Wte, Wtm, view(vte, 1:c, :), view(vtm, 1:c, :),
                fxb, fyb, basis, pairs, pair_rows, pair_cols, sdr, sdi,
                vlay, volay)
            c = 0
        end
    end

    Z = complex.(Zr, Zi)
    _add_gram!(Z, basis, grid, surface_zs)
    return Z
end



# process one block of modes: modal voltages, weight blocks, dgemm accumulate
function _planar_z_block!(Zr::Matrix{Float64}, Zi::Matrix{Float64},
        stack::PlanarStackup, omega::Number,
        mg::PlanarModeGrid, grid::CellGrid,
        mlist::AbstractVector{Int}, nlist::AbstractVector{Int},
        Wte::Matrix{Float64}, Wtm::Matrix{Float64},
        vte::AbstractMatrix{ComplexF64}, vtm::AbstractMatrix{ComplexF64},
        fxb::Matrix{Float64}, fyb::Matrix{Float64},
        basis::PlanarBasisSet, pairs::Vector{Tuple{Int,Int}},
        pair_rows::Dict{Tuple{Int,Int},Vector{Int}},
        pair_cols::Dict{Tuple{Int,Int},Vector{Int}},
        sdr::Matrix{Float64}, sdi::Matrix{Float64},
        vlay::Vector{Int}=Int[], volay::Vector{Int}=Int[])
    cblk = length(mlist)
    nb = planar_basis_count(basis)

    # cascade workspaces are reused across all modes of the block; the
    # four cascade vectors + three scratch vectors are allocated once per
    # block and per polarization (accounted for in the caller's estimate)
    L = length(stack.layers)
    casc_te = PlanarCascade(Vector{ComplexF64}(undef, L + 1),
        Vector{ComplexF64}(undef, L + 1), Vector{ComplexF64}(undef, L),
        Vector{ComplexF64}(undef, L))
    casc_tm = PlanarCascade(Vector{ComplexF64}(undef, L + 1),
        Vector{ComplexF64}(undef, L + 1), Vector{ComplexF64}(undef, L),
        Vector{ComplexF64}(undef, L))
    scratch = (Vector{ComplexF64}(undef, L), Vector{ComplexF64}(undef, L),
               Vector{ComplexF64}(undef, L))
    vsts = isempty(vlay) ? _ViaLayerState{ComplexF64}[] :
        Vector{_ViaLayerState{ComplexF64}}(undef, L)
    volsts = isempty(volay) ? (_VolLayerState{ComplexF64}[],
        _VolLayerState{ComplexF64}[]) :
        (Vector{_VolLayerState{ComplexF64}}(undef, L),
         Vector{_VolLayerState{ComplexF64}}(undef, L))

    _planar_mode_voltages!(vte, vtm, casc_te, casc_tm, scratch, stack,
        omega, mg, mlist, nlist, pairs, vsts, vlay, volsts, volay)
    _planar_weight_block!(Wte, Wtm, mlist, nlist, mg, fxb, fyb, basis)

    # Z[rows,cols] -= W_f' * (V .* W_s) via real dgemm on each component
    @inbounds for pi_ in eachindex(pairs)
        rows = pair_rows[pairs[pi_]]
        cols = pair_cols[pairs[pi_]]
        ns = length(cols)
        sd_r = view(sdr, 1:cblk, 1:ns)
        sd_i = view(sdi, 1:cblk, 1:ns)
        for (W, vv) in ((Wte, view(vte, :, pi_)),
                        (Wtm, view(vtm, :, pi_)))
            for (jj, q) in enumerate(cols)
                for c in 1:cblk
                    w = W[c, q]
                    v = vv[c]
                    sdr[c, jj] = real(v) * w
                    sdi[c, jj] = imag(v) * w
                end
            end
            wf = view(W, 1:cblk, rows)
            mul!(view(Zr, rows, cols), transpose(wf), sd_r, -1.0, 1.0)
            mul!(view(Zi, rows, cols), transpose(wf), sd_i, -1.0, 1.0)
        end
    end
    return nothing
end

# TE/TM modal voltages V(f,s)/N_pol^2 for one mode block.  `vte`/`vtm` are
# (cblk x npair); the cascade workspaces must carry the scalar type that
# differentiates the call (ComplexF64 analysis, _PlanarDual for gradients).
# Pair element ids: >= 0 sheet interface, < 0 via element (layer, kind)
# encoded by PlanarVias.jl — via pairs couple through TM only and their
# kernels are evaluated against the per-layer `vsts` state.
function _planar_mode_voltages!(vte::AbstractMatrix,
        vtm::AbstractMatrix, casc_te::PlanarCascade,
        casc_tm::PlanarCascade, scratch, stack::PlanarStackup,
        omega::Number, mg::PlanarModeGrid,
        mlist::AbstractVector{Int}, nlist::AbstractVector{Int},
        pairs::Vector{Tuple{Int,Int}},
        vsts::Vector{<:_ViaLayerState}=_ViaLayerState{ComplexF64}[],
        vlay::AbstractVector{Int}=Int[],
        volsts::Tuple=(_VolLayerState{ComplexF64}[],
            _VolLayerState{ComplexF64}[]),
        volay::AbstractVector{Int}=Int[])
    cblk = length(mlist)
    pec = mg.walls == WALL_PEC
    nvia = length(vlay)
    nvol = length(volay)
    volsts_te, volsts_tm = volsts
    for c in 1:cblk
        m, n = mlist[c], nlist[c]
        kx, ky = mg.kx[m], mg.ky[n]
        kc2 = kx * kx + ky * ky
        # norm^2 factors: PMC swaps the TE/TM sin/cos parities.  A zero
        # norm marks a non-existent mode (PEC TM needs m,n >= 1; PMC TE
        # needs m,n >= 1 and PMC TM drops only the uniform (0,0) mode).
        nte2, ntm2 = if pec
            (ky * ky * mg.ic[m] * mg.js[n] +
             kx * kx * mg.is[m] * mg.jc[n],
             kx * kx * mg.ic[m] * mg.js[n] +
             ky * ky * mg.is[m] * mg.jc[n])
        else
            (ky * ky * mg.is[m] * mg.jc[n] +
             kx * kx * mg.ic[m] * mg.js[n],
             kx * kx * mg.is[m] * mg.jc[n] +
             ky * ky * mg.ic[m] * mg.js[n])
        end
        n3_2 = pec ? mg.is[m] * mg.js[n] : mg.ic[m] * mg.jc[n]
        te_ok = kc2 != 0.0 && nte2 != 0.0
        tm_ok = ntm2 != 0.0
        te_ok && planar_mode_cascade!(casc_te, stack, omega, kc2, TE_POL,
                                      scratch)
        tm_ok && planar_mode_cascade!(casc_tm, stack, omega, kc2, TM_POL,
                                      scratch)
        if nvol != 0
            te_ok && @inbounds for j in volay
                volsts_te[j] = _vol_layer_state(stack, casc_te, omega,
                    kc2, j, TE_POL)
            end
            tm_ok && @inbounds for j in volay
                volsts_tm[j] = _vol_layer_state(stack, casc_tm, omega,
                    kc2, j, TM_POL)
            end
        end
        if tm_ok && nvia != 0
            @inbounds for j in vlay
                vsts[j] = _via_layer_state(stack, casc_tm, omega, kc2, j)
            end
        end
        @inbounds for pi_ in eachindex(pairs)
            f, s = pairs[pi_]
            if f >= 0 && s >= 0
                vte[c, pi_] = te_ok ?
                    planar_modal_voltage(casc_te, f, s) / nte2 : 0.0im
                vtm[c, pi_] = tm_ok ?
                    planar_modal_voltage(casc_tm, f, s) / ntm2 : 0.0im
            else
                # via elements couple through TM only; volume elements
                # (transverse current) couple through both pols
                vte[c, pi_] = te_ok && (_is_vol_elem(f) || f >= 0) &&
                    (_is_vol_elem(s) || s >= 0) ?
                    _elem_pair_pol(f, s, casc_te, vsts,
                        volsts_te, nte2, 0.0) : 0.0im
                vtm[c, pi_] = tm_ok ?
                    _elem_pair_pol(f, s, casc_tm, vsts,
                        volsts_tm, ntm2, n3_2) : 0.0im
            end
        end
    end
    return nothing
end

# weight blocks: W[c,p] = modal factor * fx_p[m] * fy_p[n]
function _planar_weight_block!(Wte::AbstractMatrix{Float64},
        Wtm::AbstractMatrix{Float64},
        mlist::AbstractVector{Int}, nlist::AbstractVector{Int},
        mg::PlanarModeGrid, fxb::Matrix{Float64}, fyb::Matrix{Float64},
        basis::PlanarBasisSet)
    nb = planar_basis_count(basis)
    cblk = length(mlist)
    for p in 1:nb
        xdir = _is_xdir(basis.kind[p])
        via = _is_via_kind(basis.kind[p])
        fx = view(fxb, :, p)
        fy = view(fyb, :, p)
        @inbounds for c in 1:cblk
            m, n = mlist[c], nlist[c]
            if via
                # z-directed: couples through TM g3 only (raw transform)
                Wte[c, p] = 0.0
                Wtm[c, p] = fx[m] * fy[n]
            else
                Wte[c, p] = (xdir ? mg.ky[n] : -mg.kx[m]) * fx[m] * fy[n]
                Wtm[c, p] = (xdir ? mg.kx[m] : mg.ky[n]) * fx[m] * fy[n]
            end
        end
    end
    return nothing
end

# ---------------- surface-impedance Gram term ----------------

@inline function _zs_for_level(surface_zs, lv::Int)
    return surface_zs isa AbstractVector ? ComplexF64(surface_zs[lv]) :
                                           ComplexF64(surface_zs)
end

function _gram_self(kind::UInt8, grid::CellGrid)
    dx, dy = grid.dx, grid.dy
    if kind == _BASIS_X_FULL
        return (2dx / 3) * dy
    elseif kind == _BASIS_X_LO || kind == _BASIS_X_HI
        return (dx / 3) * dy
    elseif kind == _BASIS_Y_FULL
        return (2dy / 3) * dx
    else
        return (dy / 3) * dx
    end
end

# overlap of two same-direction rooftops sharing one cell along the flow
# direction: int_0^1 s(1-s) ds * cell area = dx*dy/6
@inline _gram_adj(grid::CellGrid) = grid.dx * grid.dy / 6

# overlap of a transverse pair (x- and y-directed rooftops sharing one
# cell): each linear profile integrates to half the cell side, so the
# corner overlap is (dx/2)*(dy/2)
@inline _gram_cross(grid::CellGrid) = grid.dx * grid.dy / 4

# cell indices spanned by the flow-direction support of one basis
@inline _xcells(kind::UInt8, ei::Int, nx::Int) =
    kind == _BASIS_X_LO ? (1, 0) :
    kind == _BASIS_X_HI ? (nx, 0) : (ei, ei + 1)
@inline _ycells(kind::UInt8, ej::Int, ny::Int) =
    kind == _BASIS_Y_LO ? (1, 0) :
    kind == _BASIS_Y_HI ? (ny, 0) : (ej, ej + 1)

"""
Add the conductor-loss term Zs * G where G[p,q] = int f_p . f_q dA is the
analytic rooftop Gram overlap on the same sheet level: the basis self
term, the colinear neighbour term `dx*dy/6`, and the transverse corner
term `dx*dy/4` where an x- and a y-directed rooftop share one cell.
"""
function _add_gram!(Z::Matrix{ComplexF64}, basis::PlanarBasisSet,
        grid::CellGrid, surface_zs)
    iszero(surface_zs) && return Z
    nb = planar_basis_count(basis)
    nx, ny = grid.nx, grid.ny
    lookup = Dict{NTuple{4,Int},Int}()
    sizehint!(lookup, nb)
    @inbounds for p in 1:nb
        lookup[(basis.level[p], Int(basis.kind[p]),
                basis.ei[p], basis.ej[p])] = p
    end
    @inbounds for p in 1:nb
        kp = basis.kind[p]
        kp >= _BASIS_VIA_U && continue   # vias/volumes carry no sheet Gram term
        zs = _zs_for_level(surface_zs, basis.level[p])
        iszero(zs) && continue
        lv = basis.level[p]
        Z[p, p] += zs * _gram_self(kp, grid)
        if _is_xdir(kp)
            # colinear neighbour along the flow direction: FULL bases reach
            # ei+1; wall halves reach the interior edge (LO:1 / HI:nx-1)
            e2 = kp == _BASIS_X_HI ? basis.ei[p] - 1 : basis.ei[p] + 1
            q = get(lookup, (lv, Int(_BASIS_X_FULL), e2, basis.ej[p]), 0)
            # nx == 1 box: LO and HI halves share the single cell column
            if q == 0 && kp == _BASIS_X_LO && nx == 1
                q = get(lookup, (lv, Int(_BASIS_X_HI), nx, basis.ej[p]), 0)
            end
            if q != 0
                g = zs * _gram_adj(grid)
                Z[p, q] += g
                Z[q, p] += g
            end
            # transverse overlaps: y-bases on a column spanned by this
            # x-basis whose y-support contains row ej[p]
            i1, i2 = _xcells(kp, basis.ei[p], nx)
            for i in (i1, i2)
                i == 0 && continue
                j = basis.ej[p]
                for (ky, f) in ((Int(_BASIS_Y_FULL), j - 1),
                                (Int(_BASIS_Y_FULL), j),
                                (Int(_BASIS_Y_LO), 0),
                                (Int(_BASIS_Y_HI), ny))
                    # y-coverage: FULL f spans rows {f, f+1}, LO row 1,
                    # HI row ny -- select candidates covering row j
                    (ky == Int(_BASIS_Y_LO) && j != 1 ||
                     ky == Int(_BASIS_Y_HI) && j != ny) && continue
                    q = get(lookup, (lv, ky, i, f), 0)
                    q == 0 && continue
                    g = zs * _gram_cross(grid)
                    Z[p, q] += g
                    Z[q, p] += g
                end
            end
        else
            f2 = kp == _BASIS_Y_HI ? basis.ej[p] - 1 : basis.ej[p] + 1
            q = get(lookup, (lv, Int(_BASIS_Y_FULL), basis.ei[p], f2), 0)
            if q == 0 && kp == _BASIS_Y_LO && ny == 1
                q = get(lookup, (lv, Int(_BASIS_Y_HI), basis.ei[p], ny), 0)
            end
            if q != 0
                g = zs * _gram_adj(grid)
                Z[p, q] += g
                Z[q, p] += g
            end
        end
    end
    return Z
end
