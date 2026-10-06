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

# A purely imaginary modal factor multiplies a real spatial sin/cos kernel.
# Rotate that known factor out before the FFT, then restore it after gathering.
# Local conductor loss is added separately, including every nonzero real part.
function _planar_fft_kernel_transform!(F,backward,imaginary_modes)
    if imaginary_modes
        for index in eachindex(F)
            value=F[index];F[index]=complex(imag(value),-real(value))
        end
    end
    backward*F
    return F
end

@inline _planar_fft_kernel_value(value,imaginary_modes)=
    imaginary_modes ? complex(0.,real(value)) : value

function _planar_fft_dense_update!(Z,F,field,source,rx,ry,imaginary_modes=false)
    px,py=size(F)
    for q in eachindex(source.indices)
        sq=source.lattice[q]-1;xs,ys=rem(sq,px),sq÷px
        for p in eachindex(field.indices)
            fp=field.lattice[p]-1;xf,yf=rem(fp,px),fp÷px
            value=F[mod(xf+rx*xs,px)+1,mod(yf+ry*ys,py)+1]
            Z[field.indices[p],source.indices[q]]+=_planar_fft_kernel_value(value,imaginary_modes)
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

# At a box wall both reflected half-rooftop coordinates coincide. Interior
# terminal halves instead have separate symmetric/antisymmetric channels.
# Each channel uses its own reflection parity and the original transforms.
Base.@propagate_inbounds function _ufft_image_coefficient(plus,minus,mode,sign,half,projection)
    projection==1 && return (plus[mode]+minus[mode])/2
    projection==2 && return sign*(plus[mode]-minus[mode])/2
    return half ? plus[mode]+minus[mode] : sign==1 ? plus[mode] : minus[mode]
end

function _planar_fft_fold_images_block!(spectra::Array{ComplexF64,3},
        families,images,ne,mg,k_te,k_tm,mlist,nlist,count)
    nc=length(images)
    for (fi,field) in enumerate(families),(ci,image) in enumerate(images)
        source=families[image.source];pair=(field.element-1)*ne+source.element
        fv,sv=_is_via_kind(field.kind),_is_via_kind(source.kind)
        fx,sxdir=_is_xdir(field.kind),_is_xdir(source.kind)
        F=view(spectra,:,:,(fi-1)*nc+ci);px,py=size(F)
        # Constructor-owned coefficient ranges and bounded mode/pair
        # indices are valid; every spectral index is reduced to the grid.
        @inbounds for q in 1:count
            m=mlist[q]-1;n=nlist[q]-1
            kernel=_planar_fft_family_kernel(k_te[q,pair],k_tm[q,pair],fv,sv,fx,sxdir,mg.kx[m+1],mg.ky[n+1])
            for sx in (-1,1),sy in (-1,1)
                cx=_ufft_image_coefficient(source.xplus,source.xminus,m+1,sx,image.halfx,
                    image.projection<=2 ? image.projection : 0)
                cy=_ufft_image_coefficient(source.yplus,source.yminus,n+1,sy,image.halfy,
                    image.projection>=3 ? image.projection-2 : 0)
                F[mod(sx*m,px)+1,mod(sy*n,py)+1]+=
                    kernel*_planar_family_coefficient(field,m,n,sx,sy)*cx*cy
            end
        end
    end
    return nothing
end

function _planar_fft_images_dense_kernels!(Z,A,folded)
    F=A.lattice;px,py=size(F);nc=length(folded.images)
    for (fi,field) in enumerate(A.families),(ci,image) in enumerate(folded.images)
        source=A.families[image.source]
        copyto!(F,view(folded.spectra,:,:,(fi-1)*nc+ci))
        _planar_fft_kernel_transform!(F,A.backward,folded.imaginary_modes)
        for q in eachindex(source.indices)
            sq=source.lattice[q]-1;xs,ys=rem(sq,px),sq÷px
            for rx in _ufft_image_signs(image.halfx),ry in _ufft_image_signs(image.halfy)
                ix,iy,sign=_ufft_source_image(image,xs,ys,rx,ry)
                for p in eachindex(field.indices)
                    fp=field.lattice[p]-1;xf,yf=rem(fp,px),fp÷px
                    value=F[mod(xf-ix,px)+1,mod(yf-iy,py)+1]
                    Z[field.indices[p],source.indices[q]]+=sign*_planar_fft_kernel_value(value,folded.imaginary_modes)
                end
            end
        end
    end
    return Z
end

function _planar_fft_folded_dense_kernels!(Z,A,spectra)
    F=A.lattice;nf=length(A.families)
    for (fi,field) in enumerate(A.families),(si,source) in enumerate(A.families)
        for (xi,rx) in enumerate((-1,1)),(yi,ry) in enumerate((-1,1))
            copyto!(F,view(spectra,:,:,4*((fi-1)*nf+si-1)+2*(xi-1)+yi))
            _planar_fft_kernel_transform!(F,A.backward,A.imaginary_modes)
            _planar_fft_dense_update!(Z,F,field,source,rx,ry,A.imaginary_modes)
        end
    end
    return Z
end

_planar_fft_dense_kernels!(Z,A::_PlanarFFTBlockAssemblyWorkspace)=
    _planar_fft_folded_dense_kernels!(Z,A,A.spectra)

function _planar_fft_dense_kernels!(Z,
        A::Union{PlanarUFFTOperator,_PlanarFFTAssemblyWorkspace})
    if A isa PlanarUFFTOperator && A.folded!==nothing
        return _planar_fft_images_dense_kernels!(Z,A,A.folded)
    end
    F=A.lattice;px,py=size(F);mg=A.modes;ne=_planar_fft_element_count(A)
    imaginary_modes=all(z->iszero(real(z)),A.k_te) && all(z->iszero(real(z)),A.k_tm)
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
            _planar_fft_kernel_transform!(F,A.backward,imaginary_modes)
            _planar_fft_dense_update!(Z,F,field,source,rx,ry,imaginary_modes)
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
