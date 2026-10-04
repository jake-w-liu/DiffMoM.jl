# Exact uniform-grid FFT application of the finite shielded-box modal sum.
# Each sin/cos product has four Fourier terms.  Modes beyond Nyquist are
# folded only after multiplication by their analytic pulse transforms and
# full modal kernels, preserving near-cell and wall interactions exactly.
# No quadrature or sampled Green function is used.

export PlanarUFFTOperator, PlanarUFFTResult
export planar_ufft_operator, solve_planar_ufft

struct _PlanarUFFTFamily
    element::Int
    kind::UInt8
    indices::Vector{Int}
    lattice::Vector{Int}
    xplus::Vector{ComplexF64}
    xminus::Vector{ComplexF64}
    yplus::Vector{ComplexF64}
    yminus::Vector{ComplexF64}
end

struct _PlanarUFFTFoldedWorkspace
    spectra::Array{ComplexF64,3}
    fields::Matrix{ComplexF64}
end

"""FFT operator for the same analytic Galerkin modal sum as
`assemble_planar_z`.  Supports sheets, wall half rooftops, uniform/tapered
vias, volume rooftops, multilayers, PEC/PMC sidewalls and metal loss.
Its mutable FFT workspace belongs to this operator; simultaneous calls
require separate operators.  Storage has no dense basis-by-basis matrix."""
struct PlanarUFFTOperator{PF,PB} <: AbstractMatrix{ComplexF64}
    n::Int
    grid::CellGrid
    modes::PlanarModeGrid
    families::Vector{_PlanarUFFTFamily}
    k_te::Matrix{ComplexF64}
    k_tm::Matrix{ComplexF64}
    source_te::Matrix{ComplexF64}
    source_tm::Matrix{ComplexF64}
    field_te::Matrix{ComplexF64}
    field_tm::Matrix{ComplexF64}
    lattice::Matrix{ComplexF64}
    forward::PF
    backward::PB
    local_loss::SparseMatrixCSC{ComplexF64,Int}
    output::Vector{ComplexF64}
    folded::Union{Nothing,_PlanarUFFTFoldedWorkspace}
end

# Preserve construction with the former retained-workspace field list.
PlanarUFFTOperator(n,grid,modes,families,k_te,k_tm,source_te,source_tm,
    field_te,field_tm,lattice,forward,backward,local_loss,output)=
    PlanarUFFTOperator(n,grid,modes,families,k_te,k_tm,source_te,source_tm,
        field_te,field_tm,lattice,forward,backward,local_loss,output,nothing)

# Retained dense assembly uses the same finite modal kernels, without the
# four mode-by-element matvec buffers or the per-basis output vector.
struct _PlanarFFTAssemblyWorkspace{PB}
    n::Int
    ne::Int
    modes::PlanarModeGrid
    families::Vector{_PlanarUFFTFamily}
    k_te::Matrix{ComplexF64}
    k_tm::Matrix{ComplexF64}
    lattice::Matrix{ComplexF64}
    backward::PB
    local_loss::SparseMatrixCSC{ComplexF64,Int}
end

# High mode counts can be folded into the small family-pair lattices before
# allocating the dense matrix, retaining only a bounded modal kernel block.
struct _PlanarFFTBlockAssemblyWorkspace{PB}
    n::Int
    ne::Int
    modes::PlanarModeGrid
    families::Vector{_PlanarUFFTFamily}
    spectra::Array{ComplexF64,3}
    lattice::Matrix{ComplexF64}
    backward::PB
    local_loss::SparseMatrixCSC{ComplexF64,Int}
end

Base.size(A::PlanarUFFTOperator) = (A.n, A.n)
Base.size(A::PlanarUFFTOperator, d::Integer) = d < 1 ?
    throw(ArgumentError("dimension must be positive")) : d <= 2 ? A.n : 1
Base.eltype(::Type{<:PlanarUFFTOperator}) = ComplexF64

function _ufft_family(grid, mg, basis, idx, element)
    kind = basis.kind[first(idx)]
    k = _sheet_kind(kind)
    pec = grid.walls === WALL_PEC
    via = _is_via_kind(kind)
    xdir = _is_xdir(kind)
    lo_x = k == _BASIS_X_LO
    hi_x = k == _BASIS_X_HI
    lo_y = k == _BASIS_Y_LO
    hi_y = k == _BASIS_Y_HI
    half_x = lo_x || hi_x
    half_y = lo_y || hi_y
    xf = via || !xdir ? mg.tx_rect_sin : half_x ? mg.hx_cos : mg.tx_tri_cos
    yf = via || xdir ? mg.ty_rect_sin : half_y ? mg.hy_cos : mg.ty_tri_cos
    sinx = via || !xdir ? pec : !pec
    siny = via || xdir ? pec : !pec
    dxoff = via || !xdir ? 0.5 : 0.0
    dyoff = via || xdir ? 0.5 : 0.0
    # A translated half ramp is Hc*cos -/+ Hs*sin (PEC), or
    # Hc*sin +/- Hs*cos (PMC). Store the two Fourier coefficients so
    # its mixed sine/cosine component and diagonal cross term are exact.
    xp=Vector{ComplexF64}(undef,mg.mx);xm=similar(xp)
    yp=Vector{ComplexF64}(undef,mg.my);ym=similar(yp)
    for m in 0:mg.mx-1
        phase=cispi(m*dxoff/grid.nx)
        mix=half_x ? (pec ? (lo_x ? -1 : 1) : (lo_x ? 1 : -1))*mg.hx_sin[m+1] : 0.0
        xp[m+1]=phase*(xf[m+1]*_ufft_trig_coefficient(sinx,true)+mix*_ufft_trig_coefficient(!sinx,true))
        xm[m+1]=conj(phase)*(xf[m+1]*_ufft_trig_coefficient(sinx,false)+mix*_ufft_trig_coefficient(!sinx,false))
    end
    for n in 0:mg.my-1
        phase=cispi(n*dyoff/grid.ny)
        mix=half_y ? (pec ? (lo_y ? -1 : 1) : (lo_y ? 1 : -1))*mg.hy_sin[n+1] : 0.0
        yp[n+1]=phase*(yf[n+1]*_ufft_trig_coefficient(siny,true)+mix*_ufft_trig_coefficient(!siny,true))
        ym[n+1]=conj(phase)*(yf[n+1]*_ufft_trig_coefficient(siny,false)+mix*_ufft_trig_coefficient(!siny,false))
    end
    lattice = Vector{Int}(undef, length(idx))
    px = 2grid.nx
    for (q, p) in enumerate(idx)
        xi = via || !xdir ? basis.ei[p] - 1 : basis.ei[p]
        yi = via || xdir ? basis.ej[p] - 1 : basis.ej[p]
        lattice[q] = xi + 1 + px * yi
    end
    return _PlanarUFFTFamily(element, kind, idx, lattice,xp,xm,yp,ym)
end

"""`planar_ufft_operator(problem, freq; mx=2nx, my=2ny, ...)`
constructs the exact modal FFT operator.  The mode counts can exceed
Nyquist; every analytic high-mode contribution is retained by alias folding.
When smaller, bounded modal blocks are folded into family-pair spectra;
matvecs then use lattice convolutions without mode-by-element work arrays.
`max_bytes` bounds owned array payloads before FFT/kernel allocation."""
planar_ufft_operator(prob::PlanarProblem, freq::Number;kw...) =
    _planar_fft_workspace(prob,freq,Val(false);kw...)

function _planar_fft_workspace(prob::PlanarProblem, freq::Number,::Val{dense};
        mx::Integer=2prob.grid.nx, my::Integer=2prob.grid.ny,
        surface_zs=zero(ComplexF64), via_sigma=Inf, volume_sigma=Inf,
        sheet_coupling_zs=nothing,
        block::Integer=512,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
        _fold_dense::Bool=false, _fold_iterative::Bool=true) where {dense}
    planar_validate(prob.stack)
    omega = 2pi * ComplexF64(freq)
    isfinite(omega) && real(omega) > 0 ||
        throw(ArgumentError("freq must be finite with Re > 0"))
    mx >= 1 && my >= 1 || throw(ArgumentError("mode counts must be >= 1"))
    block >= 1 || throw(ArgumentError("block must be >= 1"))
    nb = planar_basis_count(prob.basis)
    nb > 0 || throw(ArgumentError("FFT operator requires basis functions"))
    # Discovery alone owns linear element/group/index arrays. Reject an
    # insufficient budget before constructing those arrays; the complete
    # estimate below still includes this same payload exactly once.
    _enforce_payload_limit(_checked_array_payload_bytes(Int,8,nb),max_bytes,
        "planar FFT discovery","max_bytes")
    _validate_planar_surface_zs(surface_zs,length(prob.sheets),prob.grid)
    _validate_planar_sheet_coupling(sheet_coupling_zs,length(prob.sheets),prob.grid)
    nmode = _checked_array_payload_bytes(UInt8, mx, my; label="FFT mode count")
    elements = [_basis_elem(prob.basis, p, prob.sheets, prob.vias, prob.vols)
        for p in 1:nb]
    uniq = unique(elements)
    ne = length(uniq)
    group_keys = unique([(elements[p], prob.basis.kind[p]) for p in 1:nb])
    nf = length(group_keys)
    loss_links=sheet_coupling_zs===nothing ? 0 : maximum(
        sum(!iszero(sheet_coupling_zs[i,j]) for j in axes(sheet_coupling_zs,2))
        for i in axes(sheet_coupling_zs,1);init=0)
    L = length(prob.stack.layers)
    kernel_block=Int(min(BigInt(nmode),BigInt(block)))
    full_kernel_bytes=32BigInt(nmode)*ne*ne
    folded_kernel_bytes=32BigInt(kernel_block)*ne*ne+
        16BigInt(4)*nf*nf*(2BigInt(prob.grid.nx))*(2BigInt(prob.grid.ny))
    fold_dense=dense && _fold_dense && folded_kernel_bytes<full_kernel_bytes
    modal_work_bytes=64BigInt(nmode)*ne
    folded_work_bytes=16BigInt(nf)*(2BigInt(prob.grid.nx))*(2BigInt(prob.grid.ny))
    fold_iterative=!dense && _fold_iterative &&
        folded_kernel_bytes+folded_work_bytes<full_kernel_bytes+modal_work_bytes
    fold_kernels=fold_dense || fold_iterative
    est = _checked_payload_sum("planar FFT",
        fold_kernels ? folded_kernel_bytes : full_kernel_bytes,
        dense ? 0 : fold_iterative ? folded_work_bytes : modal_work_bytes,
        _checked_array_payload_bytes(ComplexF64, 2prob.grid.nx, 2prob.grid.ny),
        _checked_array_payload_bytes(ComplexF64, 2, nf, mx + my),
        _checked_array_payload_bytes(Float64, 7, mx + my),
        # Element/group discovery, retained family indices and lattice
        # locations, element-pair records and bounded mode-index blocks.
        _checked_array_payload_bytes(Int, 8, nb),
        _checked_array_payload_bytes(Int, 2, ne, ne),
        _checked_array_payload_bytes(Int, 2, kernel_block),
        dense ? 0 : _checked_array_payload_bytes(ComplexF64, nb),
        _checked_array_payload_bytes(ComplexF64, length(prob.vias) + length(prob.vols)),
        _checked_array_payload_bytes(ComplexF64, 50, L + 1),
        # local loss sparse triplets and CSC; each basis has <=3 entries
        _checked_array_payload_bytes(ComplexF64, 8, nb, 1+loss_links),
        _checked_array_payload_bytes(Int, 16, nb, 1+loss_links))
    _enforce_payload_limit(est, max_bytes, "planar FFT", "max_bytes")
    # Material conversion must reject invalid or nonrepresentable values
    # before allocating and filling the modal kernel arrays.
    vr = _planar_bulk_resistivities(via_sigma, length(prob.vias), "via_sigma")
    mr = _planar_bulk_resistivities(volume_sigma, length(prob.vols), "volume_sigma")
    mg = planar_mode_grid(prob.grid, mx, my)
    elem_index = Dict(e => j for (j, e) in enumerate(uniq))
    families = [_ufft_family(prob.grid, mg, prob.basis,
        findall(p -> elements[p] == e && prob.basis.kind[p] == k, 1:nb),
        elem_index[e]) for (e, k) in group_keys]
    pairs = [(f, s) for f in uniq for s in uniq]
    k_te = Matrix{ComplexF64}(undef, fold_kernels ? kernel_block : nmode, ne * ne)
    k_tm = similar(k_te)
    spectra=fold_kernels ? zeros(ComplexF64,2prob.grid.nx,2prob.grid.ny,4nf*nf) : nothing
    vlay = sort!(unique([v.layer for v in prob.vias]))
    volay = sort!(unique([v.layer for v in prob.vols]))
    cte, ctm, scratch, vsts, volsts = _planar_mode_workspace(L, !isempty(vlay), !isempty(volay))
    # Fill in modest blocks so mode-index vectors stay bounded.
    mb = Vector{Int}(undef, kernel_block)
    nbmode = similar(mb)
    firstmode = 1
    while firstmode <= nmode
        count = min(length(mb), nmode - firstmode + 1)
        for q in 1:count
            t = firstmode + q - 2
            mb[q] = rem(t, Int(mx)) + 1
            nbmode[q] = t ÷ Int(mx) + 1
        end
        range=fold_kernels ? (1:count) : (firstmode:firstmode+count-1)
        te=view(k_te,range,:);tm=view(k_tm,range,:)
        _planar_mode_voltages!(te,tm, cte, ctm, scratch,
            prob.stack, omega, mg, view(mb, 1:count), view(nbmode, 1:count),
            pairs, vsts, vlay, volsts, volay)
        if fold_kernels
            all(isfinite,te) && all(isfinite,tm) ||
                throw(ArgumentError("FFT modal kernel is non-finite at a box resonance"))
            _planar_fft_fold_dense_block!(spectra,families,ne,mg,te,tm,mb,nbmode,count)
        end
        firstmode += count
    end
    fold_kernels || (all(isfinite, k_te) && all(isfinite, k_tm)) ||
        throw(ArgumentError("FFT modal kernel is non-finite at a box resonance"))
    rows, cols, values = Int[], Int[], ComplexF64[]
    emit(p, q, v) = (push!(rows, p); push!(cols, q); push!(values, v); nothing)
    _planar_sheet_loss_entries(emit, prob.basis, prob.grid, surface_zs,sheet_coupling_zs)
    _planar_bulk_loss_entries(prob.basis, prob.grid, prob.stack,
        prob.vias, prob.vols, vr, mr) do p, q, value, layer, power
        emit(p, q, value)
    end
    loss = sparse(rows, cols, values, nb, nb)
    lattice = zeros(ComplexF64, 2prob.grid.nx, 2prob.grid.ny)
    # FFTW's supported plan API defaults to its inexpensive estimate.
    backward = FFTW.plan_bfft!(lattice)
    fold_dense && return _PlanarFFTBlockAssemblyWorkspace(nb,ne,mg,families,
        spectra,lattice,backward,loss)
    dense && return _PlanarFFTAssemblyWorkspace(nb,ne,mg,families,k_te,k_tm,
        lattice,backward,loss)
    forward = FFTW.plan_fft!(lattice)
    if fold_iterative
        folded=_PlanarUFFTFoldedWorkspace(spectra,zeros(ComplexF64,length(lattice),nf))
        empty_kernel=zeros(ComplexF64,0,ne*ne)
        empty_modes=zeros(ComplexF64,0,ne)
        return PlanarUFFTOperator(nb,prob.grid,mg,families,empty_kernel,empty_kernel,
            empty_modes,empty_modes,empty_modes,empty_modes,lattice,forward,backward,
            loss,zeros(ComplexF64,nb),folded)
    end
    src_te = zeros(ComplexF64, nmode, ne)
    src_tm = similar(src_te)
    dst_te, dst_tm = similar(src_te), similar(src_te)
    return PlanarUFFTOperator(nb, prob.grid, mg, families, k_te, k_tm,
        src_te, src_tm, dst_te, dst_tm, lattice, forward, backward,
        loss, zeros(ComplexF64, nb))
end

@inline _ufft_trig_coefficient(issin::Bool, positive::Bool) =
    issin ? (positive ? -0.5im : 0.5im) : (0.5 + 0im)

@inline function _ufft_family_sum(F, m, n, family, px, py)
    value = 0.0im
    for positive_y in (true, false), positive_x in (true, false)
        sx = positive_x ? 1 : -1
        sy = positive_y ? 1 : -1
        xp = positive_x ? family.xplus[m + 1] : family.xminus[m + 1]
        yp = positive_y ? family.yplus[n + 1] : family.yminus[n + 1]
        factor = xp * yp
        value += factor * F[mod(-sx * m, px) + 1, mod(-sy * n, py) + 1]
    end
    return value
end

@inline function _ufft_family_fold!(F, value, m, n, family, px, py)
    for positive_y in (true, false), positive_x in (true, false)
        sx = positive_x ? 1 : -1
        sy = positive_y ? 1 : -1
        xp = positive_x ? family.xplus[m + 1] : family.xminus[m + 1]
        yp = positive_y ? family.yplus[n + 1] : family.yminus[n + 1]
        factor = xp * yp
        F[mod(sx * m, px) + 1, mod(sy * n, py) + 1] += factor * value
    end
    return nothing
end

function LinearAlgebra.mul!(y::AbstractVector, A::PlanarUFFTOperator,
        x::AbstractVector, alpha::Number, beta::Number)
    length(y) == length(x) == A.n || throw(DimensionMismatch("FFT matvec vector size mismatch"))
    if A.folded!==nothing
        _planar_ufft_folded_apply!(A,x,A.folded)
        for q in eachindex(y)
            y[q]=iszero(beta) ? alpha*A.output[q] : alpha*A.output[q]+beta*y[q]
        end
        return y
    end
    st, sm, ft, fm = A.source_te, A.source_tm, A.field_te, A.field_tm
    fill!(st, 0); fill!(sm, 0)
    F = A.lattice
    px, py = size(F)
    mg = A.modes
    for family in A.families
        fill!(F, 0)
        for q in eachindex(family.indices)
            F[family.lattice[q]] += x[family.indices[q]]
        end
        A.forward * F
        via = _is_via_kind(family.kind)
        xd = _is_xdir(family.kind)
        e = family.element
        for n in 0:mg.my-1, m in 0:mg.mx-1
            t = m + 1 + mg.mx * n
            f = _ufft_family_sum(F, m, n, family, px, py)
            st[t, e] += via ? 0.0im : (xd ? mg.ky[n+1] : -mg.kx[m+1]) * f
            sm[t, e] += (via ? 1.0 : xd ? mg.kx[m+1] : mg.ky[n+1]) * f
        end
    end
    fill!(ft, 0); fill!(fm, 0)
    ne = size(st, 2)
    for f in 1:ne, s in 1:ne
        pair = (f - 1) * ne + s
        for t in axes(st, 1)
            ft[t, f] -= A.k_te[t, pair] * st[t, s]
            fm[t, f] -= A.k_tm[t, pair] * sm[t, s]
        end
    end
    fill!(A.output, 0)
    for family in A.families
        fill!(F, 0)
        via = _is_via_kind(family.kind)
        xd = _is_xdir(family.kind)
        e = family.element
        for n in 0:mg.my-1, m in 0:mg.mx-1
            t = m + 1 + mg.mx * n
            wt = via ? 0.0 : xd ? mg.ky[n+1] : -mg.kx[m+1]
            wm = via ? 1.0 : xd ? mg.kx[m+1] : mg.ky[n+1]
            value = wt * ft[t, e] + wm * fm[t, e]
            _ufft_family_fold!(F, value, m, n, family, px, py)
        end
        A.backward * F
        for q in eachindex(family.indices)
            A.output[family.indices[q]] += F[family.lattice[q]]
        end
    end
    mul!(A.output, A.local_loss, x, 1.0, 1.0)
    for q in eachindex(y)
        y[q] = iszero(beta) ? alpha * A.output[q] : alpha * A.output[q] + beta * y[q]
    end
    return y
end

# The signed spatial kernel is h(field + r*source). Reflecting the source
# lattice by -r makes its application an ordinary circular convolution.
function _planar_ufft_folded_apply!(A,x,folded::_PlanarUFFTFoldedWorkspace)
    F=A.lattice;px,py=size(F);nf=length(A.families);fields=folded.fields
    fill!(fields,0)
    for (si,source) in enumerate(A.families)
        for (xi,rx) in enumerate((-1,1)),(yi,ry) in enumerate((-1,1))
            fill!(F,0)
            for q in eachindex(source.indices)
                z=source.lattice[q]-1;xs,ys=rem(z,px),z÷px
                F[mod(-rx*xs,px)+1,mod(-ry*ys,py)+1]+=x[source.indices[q]]
            end
            A.forward*F
            for fi in 1:nf
                K=view(folded.spectra,:,:,4*((fi-1)*nf+si-1)+2*(xi-1)+yi)
                for t in eachindex(F)
                    fields[t,fi]+=K[t]*F[t]
                end
            end
        end
    end
    fill!(A.output,0)
    for (fi,field) in enumerate(A.families)
        copyto!(F,view(fields,:,fi));A.backward*F
        for q in eachindex(field.indices)
            A.output[field.indices[q]]+=F[field.lattice[q]]
        end
    end
    mul!(A.output,A.local_loss,x,1.,1.)
    return nothing
end

LinearAlgebra.mul!(y::AbstractVector, A::PlanarUFFTOperator, x::AbstractVector) =
    mul!(y, A, x, 1.0, 0.0)

function Base.:*(A::PlanarUFFTOperator, x::AbstractVector)
    return mul!(zeros(ComplexF64, A.n), A, x)
end

# Exact operator diagonal by the nine Fourier terms of sin^2/cos^2.
# This remains O(number-of-families * modes + families * grid log grid)
# and scales the different sheet/via units without any dense probing.
function _planar_ufft_diagonal(A::PlanarUFFTOperator)
    A.folded===nothing || return _planar_ufft_folded_diagonal(A,A.folded)
    d = zeros(ComplexF64, A.n)
    F, mg = A.lattice, A.modes
    px, py = size(F)
    ne = size(A.source_te, 2)
    for family in A.families
        fill!(F, 0)
        via, xd = _is_via_kind(family.kind), _is_xdir(family.kind)
        e = family.element
        pair = (e - 1) * ne + e
        for n in 0:mg.my-1, m in 0:mg.mx-1
            t = m + 1 + mg.mx * n
            wt = via ? 0.0 : xd ? mg.ky[n+1] : -mg.kx[m+1]
            wm = via ? 1.0 : xd ? mg.kx[m+1] : mg.ky[n+1]
            value = -(wt^2 * A.k_te[t, pair] + wm^2 * A.k_tm[t, pair])
            for sy in (-2, 0, 2), sx in (-2, 0, 2)
                xp = sx == 0 ? 2*family.xplus[m+1]*family.xminus[m+1] :
                    sx > 0 ? family.xplus[m+1]^2 : family.xminus[m+1]^2
                yp = sy == 0 ? 2*family.yplus[n+1]*family.yminus[n+1] :
                    sy > 0 ? family.yplus[n+1]^2 : family.yminus[n+1]^2
                F[mod(sx*m,px)+1, mod(sy*n,py)+1] += value * xp * yp
            end
        end
        A.backward * F
        for q in eachindex(family.indices)
            p = family.indices[q]
            d[p] = F[family.lattice[q]] + A.local_loss[p, p]
        end
    end
    return d
end

function _planar_ufft_folded_diagonal(A,folded::_PlanarUFFTFoldedWorkspace)
    F=A.lattice;px,py=size(F);nf=length(A.families)
    d=zeros(ComplexF64,A.n)
    for (fi,family) in enumerate(A.families)
        for (xi,rx) in enumerate((-1,1)),(yi,ry) in enumerate((-1,1))
            copyto!(F,view(folded.spectra,:,:,4*((fi-1)*nf+fi-1)+2*(xi-1)+yi))
            A.backward*F
            for q in eachindex(family.indices)
                z=family.lattice[q]-1;xs,ys=rem(z,px),z÷px
                d[family.indices[q]]+=F[mod((1+rx)*xs,px)+1,mod((1+ry)*ys,py)+1]
            end
        end
    end
    for p in eachindex(d)
        d[p]+=A.local_loss[p,p]
    end
    return d
end

"""Result of `solve_planar_ufft`, retaining the bounded FFT operator and
currents instead of a dense matrix/factorization.  Residuals are recomputed
from the full operator; unconverged ports raise an error."""
struct PlanarUFFTResult
    problem::PlanarProblem
    freq::ComplexF64
    omega::ComplexF64
    operator::PlanarUFFTOperator
    currents::Matrix{ComplexF64}
    y::Matrix{ComplexF64}
    s::Matrix{ComplexF64}
    iterations::Vector{Int}
    relative_residuals::Vector{Float64}
    z0::Vector{ComplexF64}
end
PlanarUFFTResult(prob::PlanarProblem,freq,omega,A,X,Y,S,iterations,residuals)=
    PlanarUFFTResult(prob,freq,omega,A,X,Y,S,iterations,residuals,
        _planar_reference_values([p.z0 for p in prob.ports],length(prob.ports);freq=real(freq)))

"""Solve the planar port problem by GMRES with the exact FFT operator.
The complex-symmetric Galerkin matrix is generally indefinite, so the
iteration uses GMRES rather than positive-definite CG.  `rtol` gates the
independently recomputed full residual for every port."""
function solve_planar_ufft(prob::PlanarProblem, freq::Number;
        rtol::Real=1e-9, maxiter::Integer=0, memory::Integer=50,
        restart::Bool=true, precondition::Bool=true,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES, kw...)
    omega=2pi*ComplexF64(freq)
    isfinite(omega) && real(omega)>0 || throw(ArgumentError("FFT solve frequency must be finite with positive real part"))
    isfinite(rtol) && rtol > 0 || throw(ArgumentError("rtol must be finite and positive"))
    maxiter >= 0 && memory >= 1 || throw(ArgumentError("invalid Krylov iteration limits"))
    nb, np = planar_basis_count(prob.basis), length(prob.ports)
    memory <= typemax(Int) && maxiter <= typemax(Int) ||
        throw(ArgumentError("Krylov iteration limit overflows Int"))
    mem = restart ? min(BigInt(memory), BigInt(nb)) :
        max(BigInt(memory), iszero(maxiter) ? 2BigInt(nb) : BigInt(maxiter))
    workbytes = _checked_payload_sum("planar FFT solve",
        _checked_array_payload_bytes(ComplexF64, nb, np),
        _checked_array_payload_bytes(ComplexF64, nb, mem + 10),
        _checked_array_payload_bytes(ComplexF64, mem + 1, mem),
        _checked_array_payload_bytes(ComplexF64, 4, np, np),
        _checked_array_payload_bytes(Float64, 2, np),
        _checked_array_payload_bytes(ComplexF64, 2, np),
        _checked_array_payload_bytes(Int, nb + np))
    _enforce_payload_limit(workbytes, max_bytes, "planar FFT solve", "max_bytes")
    refs = _planar_reference_values([p.z0 for p in prob.ports],np;freq=real(freq))
    # Reserve the Krylov/results payload while constructing the operator.
    remaining = Int(BigInt(max_bytes) - workbytes)
    A = planar_ufft_operator(prob, freq; max_bytes=remaining, kw...)
    inverse_diagonal = precondition ? _planar_ufft_diagonal(A) : ComplexF64[]
    if precondition
        for p in eachindex(inverse_diagonal)
            v = inverse_diagonal[p]
            isfinite(v) && !iszero(v) || throw(ArgumentError("FFT preconditioner has invalid diagonal at $p"))
            inverse_diagonal[p] = inv(v)
        end
    end
    N = precondition ? Diagonal(inverse_diagonal) : I
    np = length(prob.ports)
    X = Matrix{ComplexF64}(undef, A.n, np)
    Y = zeros(ComplexF64, np, np)
    its, residuals = zeros(Int, np), zeros(Float64, np)
    pbs = _port_basis_indices(prob.basis, np)
    rhs, residual = zeros(ComplexF64, A.n), zeros(ComplexF64, A.n)
    sgn = [_planar_port_sign(p) for p in prob.ports]
    for q in 1:np
        fill!(rhs, 0)
        for b in pbs[q]
            rhs[b] = -sgn[q] * _planar_port_weight(prob.basis, b)
        end
        norm(rhs) > 0 || throw(ArgumentError("port $q has no excitation"))
        xq, stats = Krylov.gmres(A, rhs; rtol=Float64(rtol)/10, atol=0.0,
            itmax=Int(maxiter), memory=min(Int(memory), A.n),
            restart, reorthogonalization=true, N)
        mul!(residual, A, xq)
        residual .-= rhs
        rr = norm(residual) / norm(rhs)
        isfinite(rr) && rr <= rtol || throw(ErrorException(
            "FFT solve port $q failed residual gate: $rr > $rtol after $(stats.niter) iterations"))
        X[:, q] .= xq
        its[q], residuals[q] = stats.niter, rr
        for p in 1:np
            acc = 0.0im
            for b in pbs[p]
                acc += sgn[p] * _planar_port_weight(prob.basis, b) * xq[b]
            end
            Y[p, q] = acc
        end
    end
    S = planar_y_to_s(Y, refs)
    return PlanarUFFTResult(prob, ComplexF64(freq), 2pi * ComplexF64(freq),
        A, X, Y, S, its, residuals, refs)
end
