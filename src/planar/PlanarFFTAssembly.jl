export assemble_planar_z_ufft

Base.@propagate_inbounds _planar_family_coefficient(f,m,n,sx,sy)=
    (sx==1 ? f.xplus[m+1] : f.xminus[m+1])*
    (sy==1 ? f.yplus[n+1] : f.yminus[n+1])

_planar_fft_element_count(A::PlanarUFFTOperator)=size(A.source_te,2)
_planar_fft_element_count(A::_PlanarFFTAssemblyWorkspace)=A.ne

@inline function _planar_fft_family_kernel(te,tm,fv,sv,fx,sxdir,kx,ky)
    ft=fv ? 0. : fx ? ky : -kx
    st=sv ? 0. : sxdir ? ky : -kx
    fm=fv ? 1. : fx ? kx : ky
    sm=sv ? 1. : sxdir ? kx : ky
    return -te*ft*st-tm*fm*sm
end

Base.@propagate_inbounds function _planar_fft_fold_family_mode!(F,kernel,field,source,m,n,rx,ry)
    px,py=size(F)
    for sigx in (-1,1),sigy in (-1,1)
        value=kernel*_planar_family_coefficient(field,m,n,sigx,sigy)*
            _planar_family_coefficient(source,m,n,rx*sigx,ry*sigy)
        F[mod(sigx*m,px)+1,mod(sigy*n,py)+1]+=value
    end
    return nothing
end

function _planar_fft_dense_update!(Z,F,field,source,rx,ry)
    px,py=size(F)
    for q in eachindex(source.indices)
        sq=source.lattice[q]-1;xs,ys=rem(sq,px),sq÷px
        for p in eachindex(field.indices)
            fp=field.lattice[p]-1;xf,yf=rem(fp,px),fp÷px
            Z[field.indices[p],source.indices[q]]+=F[mod(xf+rx*xs,px)+1,mod(yf+ry*ys,py)+1]
        end
    end
    return nothing
end

function _planar_fft_fold_dense_block!(spectra::Array{ComplexF64,3},
        families,ne,mg,k_te,k_tm,mlist,nlist,count)
    nf=length(families)
    for (fi,field) in enumerate(families),(si,source) in enumerate(families)
        pair=(field.element-1)*ne+source.element
        fv,sv=_is_via_kind(field.kind),_is_via_kind(source.kind)
        fx,sxdir=_is_xdir(field.kind),_is_xdir(source.kind)
        for (xi,rx) in enumerate((-1,1)),(yi,ry) in enumerate((-1,1))
            F=view(spectra,:,:,4*((fi-1)*nf+si-1)+2*(xi-1)+yi)
            # Each lattice sees the same mode and sign order as the retained
            # kernel route; block boundaries introduce no reassociation.
            # Constructor-owned coefficients, block rows and pair indices
            # cover these ranges; folded indices are reduced to the lattice.
            @inbounds for q in 1:count
                m=mlist[q]-1;n=nlist[q]-1
                kernel=_planar_fft_family_kernel(k_te[q,pair],k_tm[q,pair],fv,sv,fx,sxdir,mg.kx[m+1],mg.ky[n+1])
                _planar_fft_fold_family_mode!(F,kernel,field,source,m,n,rx,ry)
            end
        end
    end
    return nothing
end

function _planar_fft_folded_dense_kernels!(Z,A,spectra)
    F=A.lattice;nf=length(A.families)
    for (fi,field) in enumerate(A.families),(si,source) in enumerate(A.families)
        for (xi,rx) in enumerate((-1,1)),(yi,ry) in enumerate((-1,1))
            copyto!(F,view(spectra,:,:,4*((fi-1)*nf+si-1)+2*(xi-1)+yi))
            A.backward*F
            _planar_fft_dense_update!(Z,F,field,source,rx,ry)
        end
    end
    return Z
end

_planar_fft_dense_kernels!(Z,A::_PlanarFFTBlockAssemblyWorkspace)=
    _planar_fft_folded_dense_kernels!(Z,A,A.spectra)

function _planar_fft_dense_kernels!(Z,
        A::Union{PlanarUFFTOperator,_PlanarFFTAssemblyWorkspace})
    if A isa PlanarUFFTOperator && A.folded!==nothing
        return _planar_fft_folded_dense_kernels!(Z,A,A.folded.spectra)
    end
    F=A.lattice;px,py=size(F);mg=A.modes;ne=_planar_fft_element_count(A)
    # A product of real sin/cos subsection transforms is a sum of four
    # kernels evaluated at signed sums/differences of lattice locations.
    # Reuse one Fourier buffer, retaining every high mode before folding.
    # There is no per-column FFT and no sampled/near-cell approximation.
    for field in A.families,source in A.families
        pair=(field.element-1)*ne+source.element
        fv,sv=_is_via_kind(field.kind),_is_via_kind(source.kind)
        fx,sxdir=_is_xdir(field.kind),_is_xdir(source.kind)
        for rx in (-1,1),ry in (-1,1)
            fill!(F,0)
            for n in 0:mg.my-1,m in 0:mg.mx-1
                t=m+1+mg.mx*n
                kernel=_planar_fft_family_kernel(A.k_te[t,pair],A.k_tm[t,pair],fv,sv,fx,sxdir,mg.kx[m+1],mg.ky[n+1])
                _planar_fft_fold_family_mode!(F,kernel,field,source,m,n,rx,ry)
            end
            A.backward*F
            _planar_fft_dense_update!(Z,F,field,source,rx,ry)
        end
    end
    return Z
end

function _planar_fft_dense_fill!(Z::Matrix{ComplexF64},
        A::Union{PlanarUFFTOperator,_PlanarFFTAssemblyWorkspace,_PlanarFFTBlockAssemblyWorkspace})
    size(Z)==(A.n,A.n) || throw(DimensionMismatch("FFT dense matrix and workspace dimensions differ"))
    fill!(Z,0)
    _planar_fft_dense_kernels!(Z,A)
    loss=A.local_loss
    for q in axes(loss,2),r in nzrange(loss,q)
        Z[rowvals(loss)[r],q]+=nonzeros(loss)[r]
    end
    return Z
end

"""Assemble the retained dense analytic Galerkin matrix with FFT lattice
kernels. Four signed sum/difference kernels per basis-family pair are
inverse transformed, then evaluated at actual subsection locations. This
includes all supplied modes, translated half rooftops, layers, vias,
volume currents, PEC/PMC walls and local/coupled conductor loss exactly as
[`assemble_planar_z`](@ref). No quadrature, projection or near-cell
correction is used. `max_bytes` reserves the dense output before constructing
the modal/FFT workspace. Dense assembly omits the four mode-by-element
matvec arrays needed by the iterative operator. At high mode counts it folds
bounded modal blocks into family-pair lattices when that uses less storage;
every modal contribution and its summation order are preserved. For factorization use
`solve_planar(...;method=:dense_fft)`;
for storage without an Nb² matrix use `method=:ufft`."""
function assemble_planar_z_ufft(prob::PlanarProblem,freq::Number;
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,kw...)
    nb=planar_basis_count(prob.basis)
    matrix_bytes=_checked_array_payload_bytes(ComplexF64,nb,nb)
    _enforce_payload_limit(matrix_bytes,max_bytes,"FFT dense assembly","max_bytes")
    remaining=Int(BigInt(_validated_resource_limit("max_bytes",max_bytes))-matrix_bytes)
    A=_planar_fft_workspace(prob,freq,Val(true);max_bytes=remaining,_fold_dense=true,kw...)
    Z=Matrix{ComplexF64}(undef,nb,nb)
    _planar_fft_dense_fill!(Z,A)
    all(isfinite,Z) || throw(ArgumentError("FFT dense matrix is nonfinite"))
    return Z
end
