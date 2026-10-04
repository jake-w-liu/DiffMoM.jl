export assemble_planar_z_ufft

@inline _planar_family_coefficient(f,m,n,sx,sy)=
    (sx==1 ? f.xplus[m+1] : f.xminus[m+1])*
    (sy==1 ? f.yplus[n+1] : f.yminus[n+1])

function _planar_fft_dense_fill!(Z::Matrix{ComplexF64},A::PlanarUFFTOperator)
    size(Z)==size(A) || throw(DimensionMismatch("FFT dense matrix and operator dimensions differ"))
    fill!(Z,0)
    F=A.lattice;px,py=size(F);mg=A.modes;ne=size(A.source_te,2)
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
                ft=fv ? 0. : fx ? mg.ky[n+1] : -mg.kx[m+1]
                st=sv ? 0. : sxdir ? mg.ky[n+1] : -mg.kx[m+1]
                fm=fv ? 1. : fx ? mg.kx[m+1] : mg.ky[n+1]
                sm=sv ? 1. : sxdir ? mg.kx[m+1] : mg.ky[n+1]
                kernel=-A.k_te[t,pair]*ft*st-A.k_tm[t,pair]*fm*sm
                for sigx in (-1,1),sigy in (-1,1)
                    value=kernel*_planar_family_coefficient(field,m,n,sigx,sigy)*
                        _planar_family_coefficient(source,m,n,rx*sigx,ry*sigy)
                    F[mod(sigx*m,px)+1,mod(sigy*n,py)+1]+=value
                end
            end
            A.backward*F
            for q in eachindex(source.indices)
                sq=source.lattice[q]-1;xs,ys=rem(sq,px),sq÷px
                for p in eachindex(field.indices)
                    fp=field.lattice[p]-1;xf,yf=rem(fp,px),fp÷px
                    Z[field.indices[p],source.indices[q]]+=F[mod(xf+rx*xs,px)+1,mod(yf+ry*ys,py)+1]
                end
            end
        end
    end
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
the FFT operator. For factorization use `solve_planar(...;method=:dense_fft)`;
for storage without an Nb² matrix use `method=:ufft`."""
function assemble_planar_z_ufft(prob::PlanarProblem,freq::Number;
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,kw...)
    nb=planar_basis_count(prob.basis)
    matrix_bytes=_checked_array_payload_bytes(ComplexF64,nb,nb)
    _enforce_payload_limit(matrix_bytes,max_bytes,"FFT dense assembly","max_bytes")
    remaining=Int(BigInt(_validated_resource_limit("max_bytes",max_bytes))-matrix_bytes)
    A=planar_ufft_operator(prob,freq;max_bytes=remaining,kw...)
    Z=Matrix{ComplexF64}(undef,nb,nb)
    _planar_fft_dense_fill!(Z,A)
    all(isfinite,Z) || throw(ArgumentError("FFT dense matrix is nonfinite"))
    return Z
end
