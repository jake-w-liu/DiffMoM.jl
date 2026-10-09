# Real shared-pole vector fitting. Private numerator coefficients are
# eliminated with one QR factorization before fitting the shared poles;
# workspace grows with nports^2, rather than nports^4.

export PlanarRationalModel, planar_fit_rational, planar_rational_eval
export planar_write_spice, planar_rational_passivity
export planar_rational_certificate

"""Stable real admittance model `Y(s)=d+s*e+sum(residues[k]/(s-poles[k]))`.
Complex poles/residues occur in conjugate pairs. `sampled_passive` and
`passivity_margin` describe the checked frequency grid. `globally_passive`
records the positive-real certificate and `passivity_method` identifies
it. `rms_error` and `relative_rms_error` include conductance loading and
the recorded `capacitance_adjustment` used in passivity repair."""
struct PlanarRationalModel
    poles::Vector{ComplexF64}
    residues::Vector{Matrix{ComplexF64}}
    d::Matrix{Float64}
    e::Matrix{Float64}
    frequencies::Vector{Float64}
    rms_error::Float64
    relative_rms_error::Float64
    sampled_passive::Bool
    globally_passive::Bool
    passivity_method::Symbol
    passivity_margin::Float64
    passivity_shift::Float64
    capacitance_adjustment::Float64
end

"""Evaluate a real, stable N-port admittance model at frequency [Hz]."""
function planar_rational_eval(model::PlanarRationalModel,f::Real)
    isfinite(f) || throw(ArgumentError("model frequency must be finite"))
    s = 2pi*1im*f
    if !isfinite(s)
        # A finite IEEE frequency can overflow angular frequency even when
        # its final response fits. Promote only this intermediate range.
        # Two Float64 significands retain the stored frequency/product;
        # restore the caller's MPFR precision and rounding on return.
        return setprecision(BigFloat,2precision(Float64)) do
            setrounding(BigFloat,RoundNearest) do
                wide_s=2pi*1im*BigFloat(f)
                isfinite(wide_s) || throw(ArgumentError(
                    "rational angular frequency exceeds the working exponent range"))
                value=model.d .+ wide_s .* model.e
                for k in eachindex(model.poles)
                    value .+= model.residues[k] ./ (wide_s-model.poles[k])
                end
                ComplexF64.(value)
            end
        end
    end
    Y = model.d .+ s .* model.e
    for k in eachindex(model.poles)
        Y .+= model.residues[k] ./ (s-model.poles[k])
    end
    return Y
end

function _vf_pair_indices(poles)
    ci = zeros(Int,length(poles))
    k = 1
    while k <= length(poles)
        if iszero(imag(poles[k]))
            k += 1
        else
            k < length(poles) && isapprox(poles[k+1],conj(poles[k]);rtol=1e-7) ||
                throw(ArgumentError("vector-fit poles must contain adjacent conjugate pairs"))
            ci[k],ci[k+1] = 1,2
            k += 2
        end
    end
    return ci
end

# Interleave equal conjugate modes in place by rotating the middle blocks.
# Divide at a pair boundary; recursion depth follows the number of modes.
function _vf_interleave_conjugates!(poles,left,pairs)
    pairs<=1 && return poles
    first_pairs=pairs÷2;remaining=pairs-first_pairs
    middle=left+first_pairs
    reverse!(poles,middle,middle+remaining-1)
    reverse!(poles,middle+remaining,middle+remaining+first_pairs-1)
    reverse!(poles,middle,middle+remaining+first_pairs-1)
    _vf_interleave_conjugates!(poles,left,first_pairs)
    _vf_interleave_conjugates!(poles,left+2first_pairs,remaining)
    return poles
end

function _vf_sort(poles)
    # Exact real/imaginary coordinates keep distinct decay rates separate.
    values=sort(ComplexF64.(poles);by=p->(real(p),abs(imag(p)),-imag(p)))
    left=1
    while left<=length(values)
        p=values[left]
        if iszero(imag(p));left+=1;continue;end
        stop=left+1
        while stop<=length(values) && real(values[stop])==real(p) && abs(imag(values[stop]))==abs(imag(p))
            stop+=1
        end
        pairs=(stop-left)÷2
        stop-left==2pairs || throw(ArgumentError("vector-fit poles must contain exact conjugate pairs"))
        for k in 1:pairs
            values[left+k-1]==conj(values[left+pairs+k-1]) ||
                throw(ArgumentError("vector-fit poles must contain exact conjugate pairs"))
        end
        _vf_interleave_conjugates!(values,left,pairs)
        left=stop
    end
    return values
end

function _vf_basis(s,poles)
    ci = _vf_pair_indices(poles)
    A = Matrix{ComplexF64}(undef,length(s),length(poles)+2)
    for r in eachindex(s)
        k = 1
        while k <= length(poles)
            if ci[k] == 0
                A[r,k] = inv(s[r]-poles[k])
                k += 1
            else
                p = poles[k]
                A[r,k] = inv(s[r]-p)+inv(s[r]-conj(p))
                A[r,k+1] = 1im*(inv(s[r]-p)-inv(s[r]-conj(p)))
                k += 2
            end
        end
        A[r,end-1],A[r,end] = 1.0,s[r]
    end
    return A
end

# Online scaled sum of squares, retaining finite magnitudes before squaring.
# The state represents scale^2*squares; no selected range threshold is needed.
@inline function _vf_scaled_square(state,value::Real)
    scale,squares=state
    magnitude=abs(value)
    iszero(magnitude) && return state
    return magnitude>scale ? (magnitude,1+squares*(scale/magnitude)^2) :
        (scale,squares+(magnitude/scale)^2)
end

function _vf_fit_errors(model,fs,Ys)
    errors=(0.0,0.0);data=(0.0,0.0)
    for k in eachindex(fs)
        predicted=planar_rational_eval(model,fs[k])
        for index in eachindex(predicted,Ys[k])
            difference=predicted[index]-Ys[k][index]
            errors=_vf_scaled_square(_vf_scaled_square(errors,real(difference)),imag(difference))
            value=Ys[k][index]
            data=_vf_scaled_square(_vf_scaled_square(data,real(value)),imag(value))
        end
    end
    rms=errors[1]*sqrt(errors[2]/(length(fs)*length(Ys[1])))
    relative=if iszero(data[1])
        rms
    elseif iszero(errors[1])
        0.0
    else
        # Form the scale ratio with binary exponents so its intermediate
        # division cannot overflow or erase a representable final ratio.
        em,ee=frexp(errors[1]);dm,de=frexp(data[1])
        ldexp((em/dm)*sqrt(errors[2]/data[2]),ee-de)
    end
    return rms,relative
end

function _vf_shared_relocate(s,Ys,poles)
    order,m,nel = length(poles),length(s),length(Ys[1])
    # Uniform response scaling cancels from C*x=b. Normalize before QR
    # projection so representable small responses retain the same pole fit.
    # Component maxima remain finite even when a complex magnitude overflows.
    amplitude=maximum(Y->maximum(y->max(abs(real(y)),abs(imag(y))),Y;init=0.0),Ys;init=0.0)
    iszero(amplitude) && return poles
    basis = _vf_basis(s,poles)
    Ar = vcat(real.(basis),imag.(basis))
    factor = LinearAlgebra.qr(Ar)
    omitted = (order+3):(2m)
    C = zeros(Float64,length(omitted)*nel,order)
    b = zeros(Float64,size(C,1))
    for e in 1:nel
        fe = ComplexF64[Y[e]/amplitude for Y in Ys]
        sigma = -fe .* view(basis,:,1:order)
        reduced = transpose(factor.Q)*hcat(vcat(real.(sigma),imag.(sigma)),
            vcat(real.(fe),imag.(fe)))
        rows = ((e-1)*length(omitted)+1):(e*length(omitted))
        C[rows,:] .= view(reduced,omitted,1:order)
        b[rows] .= view(reduced,omitted,order+1)
    end
    all(iszero,b) && return poles
    # Rank-revealing SVD prevents a constant/affine response or inactive
    # matrix entry from creating spurious poles through a singular fit.
    F = LinearAlgebra.svd(C;full=false)
    # Match the standard LinearAlgebra numerical-rank convention: the
    # smaller matrix dimension times working epsilon and largest singular value.
    cutoff = maximum(F.S;init=0.0)*min(size(C)...)*eps(eltype(F.S))
    x = F.V * [F.S[k] > cutoff ? dot(view(F.U,:,k),b)/F.S[k] : 0.0
        for k in eachindex(F.S)]
    H = zeros(Float64,order,order)
    drive = zeros(Float64,order)
    ci = _vf_pair_indices(poles)
    k = 1
    while k <= order
        if ci[k] == 0
            H[k,k],drive[k] = real(poles[k]),1.0
            k += 1
        else
            a,beta = real(poles[k]),imag(poles[k])
            H[k,k]=H[k+1,k+1]=a
            H[k,k+1],H[k+1,k] = beta,-beta
            drive[k] = 2.0
            k += 2
        end
    end
    H .-= drive*transpose(x)
    relocated=eigvals(H)
    all(isfinite,relocated) || throw(ArgumentError("vector-fit relocation produced nonfinite poles"))
    # Reflect unstable poles without imposing an unrelated decay rate.
    # A transient zero real part remains available to subsequent relocation;
    # strict stability is checked on the final physical pole set.
    return _vf_sort([complex(-abs(real(p)),imag(p)) for p in relocated])
end

# A Hermitian pair owns one output entry. Preserve ordinary sum-then-half
# arithmetic; finite overflow alone needs the equivalent halved operands.
function _vf_hermitian_part(Y)
    H=similar(Y)
    for column in axes(Y,2),row in axes(Y,1)
        left,right=Y[row,column],conj(Y[column,row])
        value=(left+right)/2
        if !isfinite(value) && isfinite(left) && isfinite(right)
            value=if value isa Complex
                complex(isfinite(real(value)) ? real(value) : real(left)/2+real(right)/2,
                    isfinite(imag(value)) ? imag(value) : imag(left)/2+imag(right)/2)
            else
                left/2+right/2
            end
        end
        H[row,column]=value
    end
    return H
end

"""Check the Hermitian part of the admittance on an explicit nonempty
frequency grid. Returns `(passive,margin,frequencies)`; this is a sampled
check, and stable poles establish causality of the fitted rational model.

At zero tolerance, passivity also checks exact stored Hermitian energy;
the reported floating margin can round a subnormal direction to zero."""
function planar_rational_passivity(model::PlanarRationalModel,frequencies;
        tol::Real=1e-9)
    isfinite(tol) && tol >= 0 || throw(ArgumentError("passivity tolerance must be finite and nonnegative"))
    fs = Float64.(collect(frequencies))
    !isempty(fs) && all(f -> isfinite(f) && f >= 0,fs) ||
        throw(ArgumentError("passivity grid must be nonempty, finite, and nonnegative"))
    margin = Inf
    exact_energy=true
    for f in fs
        Y = planar_rational_eval(model,f)
        iszero(tol) && (exact_energy &= _vf_psd(Y,tol;hermitian=true))
        H=_vf_hermitian_part(Y)
        margin = min(margin,minimum(eigvals(Hermitian(H))))
    end
    return (passive=margin >= -tol && exact_energy,margin=margin,frequencies=fs)
end

using GMP_jll: libgmp
const _vf_gmp_library=libgmp
# The loaded GMP ABI exposes its usable limb width. Its byte payload is
# derived from the C unsigned-byte width, never from a chosen capacity.
const _vf_gmp_limb_bits=Int(unsafe_load(cglobal((:__gmp_bits_per_limb,_vf_gmp_library),Cint)))
const _vf_gmp_limb_bytes=cld(_vf_gmp_limb_bits,ndigits(typemax(UInt8);base=2))
# MPFR_RNDU is the public C enumeration value 2 (mpfr.h). Dependency
# tests check mpfr_print_rnd_mode against the loaded public library.
const _vf_mpfr_up=Cint(2)

@inline function _vf_mpz_set_d!(out,x)
    ccall((:__gmpz_set_d,_vf_gmp_library),Cvoid,(Ref{BigInt},Cdouble),out,x)
    out
end
@inline function _vf_mpz_set_ui!(out,x)
    ccall((:__gmpz_set_ui,_vf_gmp_library),Cvoid,(Ref{BigInt},Culong),out,x)
    out
end
@inline function _vf_mpz_set!(out,x)
    ccall((:__gmpz_set,_vf_gmp_library),Cvoid,(Ref{BigInt},Ref{BigInt}),out,x)
    out
end
@inline function _vf_mpz_mul_2exp!(out,bits)
    ccall((:__gmpz_mul_2exp,_vf_gmp_library),Cvoid,(Ref{BigInt},Ref{BigInt},Culong),out,out,bits)
    out
end
@inline function _vf_mpz_fdiv_q_2exp!(out,bits)
    ccall((:__gmpz_fdiv_q_2exp,_vf_gmp_library),Cvoid,(Ref{BigInt},Ref{BigInt},Culong),out,out,bits)
    out
end
@inline function _vf_mpz_add!(out,a,b)
    ccall((:__gmpz_add,_vf_gmp_library),Cvoid,(Ref{BigInt},Ref{BigInt},Ref{BigInt}),out,a,b)
    out
end
@inline function _vf_mpz_sub!(out,a,b)
    ccall((:__gmpz_sub,_vf_gmp_library),Cvoid,(Ref{BigInt},Ref{BigInt},Ref{BigInt}),out,a,b)
    out
end
@inline function _vf_mpz_mul!(out,a,b)
    ccall((:__gmpz_mul,_vf_gmp_library),Cvoid,(Ref{BigInt},Ref{BigInt},Ref{BigInt}),out,a,b)
    out
end
@inline function _vf_mpz_neg!(out,a)
    ccall((:__gmpz_neg,_vf_gmp_library),Cvoid,(Ref{BigInt},Ref{BigInt}),out,a)
    out
end
@inline function _vf_mpz_tdiv_qr!(quotient,remainder,numerator,denominator)
    ccall((:__gmpz_tdiv_qr,_vf_gmp_library),Cvoid,
        (Ref{BigInt},Ref{BigInt},Ref{BigInt},Ref{BigInt}),quotient,remainder,numerator,denominator)
    quotient
end

# Hermitian energy is preserved by realification: [[Re(H),-Im(H)];
# [Im(H),Re(H)]]. Return its two original stored operands, keeping sums
# exact even when halving a subnormal value would erase its sign.
@inline function _vf_psd_pair(A,i,j,hermitian)
    !hermitian && return Float64(real(A[i,j])),0.0
    n=size(A,1)
    if eltype(A)<:Complex
        row,column=mod1(i,n),mod1(j,n)
        if (i<=n)==(j<=n)
            return Float64(real(A[row,column])),Float64(real(A[column,row]))
        elseif i<=n
            return Float64(imag(A[column,row])),-Float64(imag(A[row,column]))
        else
            return Float64(imag(A[row,column])),-Float64(imag(A[column,row]))
        end
    end
    return Float64(A[i,j]),Float64(A[j,i])
end
@inline _vf_psd_dimension(A,hermitian)=hermitian && eltype(A)<:Complex ? 2size(A,1) : size(A,1)

# The common dyadic denominator turns stored IEEE components into integers.
# k! times the largest entry to power k bounds every k-order minor;
# n^n bounds k!, deriving capacity from dimensions and actual input range.
function _vf_psd_integer_workspace(A;hermitian::Bool=false,diagonal_shift::Float64=0.0)
    n=_vf_psd_dimension(A,hermitian)
    denbits=maximum(x->max(_spice_dyadic_denbits(Float64(real(x))),
        hermitian ? _spice_dyadic_denbits(Float64(imag(x))) : 0),A;init=0)
    denbits=max(denbits,_spice_dyadic_denbits(diagonal_shift))
    entrybits=maximum(x->max(iszero(real(x)) ? 0 : exponent(abs(Float64(real(x))))+1+denbits,
        hermitian && !iszero(imag(x)) ? exponent(abs(Float64(imag(x))))+1+denbits : 0),A;init=0)
    !iszero(diagonal_shift) && (entrybits=max(entrybits,exponent(abs(diagonal_shift))+1+denbits))
    terms=hermitian ? 2 : 1
    !iszero(diagonal_shift) && (terms+=hermitian ? 2 : 1)
    terms>1 && (entrybits+=ndigits(terms-1;base=2))
    minorbits=BigInt(n)*(entrybits+ndigits(n;base=2))
    minorlimbs=cld(minorbits,_vf_gmp_limb_bits)
    limbs=2minorlimbs+1 # two minor products and one subtraction carry
    bits=Int(limbs*_vf_gmp_limb_bits)
    entries=BigInt(n)*(n+1)÷2
    roles=(:prior_pivot,:left_product,:right_product,:numerator,:remainder)
    bytes=_checked_payload_sum("exact positive-semidefinite workspace",
        _checked_array_payload_bytes(Ptr{Cvoid},n,n),
        _checked_array_payload_bytes(UInt8,entries+length(roles),
            sizeof(BigInt)+Int(limbs)*_vf_gmp_limb_bytes))
    return denbits,bits,bytes
end
@inline function _vf_set_dyadic!(value,x,denbits)
    mantissa,power=frexp(x)
    _vf_mpz_set_d!(value,ldexp(mantissa,precision(Float64)))
    shift=power-precision(Float64)+denbits
    if shift>=0
        _vf_mpz_mul_2exp!(value,shift)
    else
        _vf_mpz_fdiv_q_2exp!(value,-shift)
    end
    return value
end

function _vf_psd(A,tol;hermitian::Bool=false,diagonal_shift::Float64=0.0,max_bytes=nothing,retained_bytes::Integer=0)
    all(isfinite,A) || return false
    !hermitian && (A!=transpose(A) || any(x->!iszero(imag(x)),A)) && return false
    isfinite(diagonal_shift) && all(i->real(A[i,i])>=diagonal_shift,axes(A,1)) || return false
    n=_vf_psd_dimension(A,hermitian);diagonal=true
    for i in 1:n,j in i+1:n
        x,y=_vf_psd_pair(A,i,j,hermitian)
        x==-y && continue
        (real(A[mod1(i,size(A,1)),mod1(i,size(A,1))])==diagonal_shift ||
            real(A[mod1(j,size(A,1)),mod1(j,size(A,1))])==diagonal_shift) && return false
        diagonal=false
    end
    diagonal && return true
    denbits,bits,bytes=_vf_psd_integer_workspace(A;hermitian,diagonal_shift)
    total=_checked_payload_sum("exact positive-semidefinite workspace",bytes,retained_bytes)
    limit=max_bytes===nothing ? _default_max_dense_payload_bytes() : max_bytes
    _enforce_payload_limit(total,limit,"exact positive-semidefinite workspace","max_bytes")
    previous=BigInt(;nbits=bits)
    left=BigInt(;nbits=bits)
    right=BigInt(;nbits=bits)
    numerator=BigInt(;nbits=bits)
    remainder=BigInt(;nbits=bits)
    integers=Matrix{BigInt}(undef,n,n)
    for i in 1:n,j in i:n
        x,y=_vf_psd_pair(A,i,j,hermitian)
        value=_vf_set_dyadic!(BigInt(;nbits=bits),x,denbits)
        if hermitian
            _vf_set_dyadic!(left,y,denbits)
            _vf_mpz_add!(value,value,left)
        end
        if i==j && !iszero(diagonal_shift)
            _vf_set_dyadic!(left,diagonal_shift,denbits)
            hermitian && _vf_mpz_mul_2exp!(left,1)
            _vf_mpz_sub!(value,value,left)
        end
        integers[i,j]=integers[j,i]=value
    end
    return _vf_psd_integer!(integers,previous,left,right,numerator,remainder)
end

# Frobenius norm bounds spectral norm. Every square and sum is exact in
# the IEEE-derived product lattice; sqrt/division/accumulation round upward.
# The returned Float64 also rounds upward, so it remains a true bound.
function _vf_frobenius_bound(model;max_bytes,retained_bytes::Integer=0)
    count=maximum(length,model.residues;init=0)
    bits=_planar_terminal_product_precision(count)
    roles=(:bound,:squares,:operand,:product,:decay)
    payload=_checked_payload_sum("certified residue norm bound",retained_bytes,
        _checked_array_payload_bytes(UInt8,length(roles),_planar_wide_scalar_payload(bits)))
    _enforce_payload_limit(payload,max_bytes,"certified residue norm bound","max_bytes")
    bound,squares,operand,product,decay=ntuple(_->BigFloat(0.;precision=bits),length(roles))
    up=_vf_mpfr_up
    for k in eachindex(model.residues)
        _planar_wide_set!(squares,0.)
        for value in model.residues[k],component in (real,imag)
            _planar_wide_set!(operand,component(value))
            _planar_wide_mul!(product,operand,operand)
            _planar_wide_add!(squares,squares,product)
        end
        ccall((:mpfr_sqrt,_planar_mpfr_library),Cint,
            (Ref{BigFloat},Ref{BigFloat},Cint),squares,squares,up)
        _planar_wide_set!(decay,-real(model.poles[k]))
        ccall((:mpfr_div,_planar_mpfr_library),Cint,
            (Ref{BigFloat},Ref{BigFloat},Ref{BigFloat},Cint),product,squares,decay,up)
        ccall((:mpfr_add,_planar_mpfr_library),Cint,
            (Ref{BigFloat},Ref{BigFloat},Ref{BigFloat},Cint),bound,bound,product,up)
    end
    return Float64(bound,RoundUp)
end

# Binary scaling keeps a finite x/(y*z) from overflowing or disappearing
# in its intermediate product/divisions. y and z are positive scales.
@inline function _vf_certificate_ratio(x::Float64,y::Float64,z::Float64)
    iszero(x) && return x
    xm,xe=frexp(x);ym,ye=frexp(y);zm,ze=frexp(z)
    ldexp(xm/(ym*zm),xe-ye-ze)
end
@inline function _vf_certificate_product_ratio(x::Float64,y::Float64,z::Float64)
    iszero(x) && return x
    xm,xe=frexp(x);ym,ye=frexp(y);zm,ze=frexp(z)
    ldexp((xm*ym)/zm,xe+ye-ze)
end

# Exact original-model KYP verification. All stored inputs share one
# dyadic denominator. The largest polynomial degree is four (a*f*A*P).
# Clearing its denominator multiplies the whole inequality by a positive
# power of two; it cannot alter positive semidefiniteness.
function _vf_storage_integer_workspace(model,P,a,f)
    scalars=Iterators.flatten((P,model.d,model.poles,Iterators.flatten(model.residues),(a,f)))
    denbits=maximum(x->max(_spice_dyadic_denbits(Float64(real(x))),
        _spice_dyadic_denbits(Float64(imag(x)))),scalars;init=0)
    inputbits=maximum(x->max(iszero(real(x)) ? 0 : exponent(abs(Float64(real(x))))+1+denbits,
        iszero(imag(x)) ? 0 : exponent(abs(Float64(imag(x))))+1+denbits),scalars;init=0)
    inputbits=max(inputbits,denbits)
    # Each A column has at most two entries, giving four top-block terms.
    # Cross-block sums have at most one driven state per pole and port;
    # conjugate residue representation introduces an exact factor of two.
    terms=max(4,length(model.poles)+1)
    entrybits=4BigInt(inputbits)+ndigits(terms-1;base=2)+1
    states=size(P,1);n=states+size(model.d,1)
    minorbits=BigInt(n)*(entrybits+ndigits(n;base=2))
    limbs=2cld(minorbits,_vf_gmp_limb_bits)+1
    bits=Int(limbs*_vf_gmp_limb_bits)
    entries=BigInt(n)*(n+1)÷2+BigInt(states)*(states+1)÷2
    roles=(:prior_pivot,:left_operand,:right_operand,:product,:accumulator,
        :admittance_scale,:frequency_scale)
    bytes=_checked_payload_sum("exact positive-real storage workspace",
        _checked_array_payload_bytes(Ptr{Cvoid},n,n),
        _checked_array_payload_bytes(Ptr{Cvoid},states,states),
        _checked_array_payload_bytes(UInt8,entries+length(roles),
            sizeof(BigInt)+Int(limbs)*_vf_gmp_limb_bytes))
    return denbits,bits,bytes
end

function _vf_psd_integer!(integers,previous,left,right,numerator,remainder)
    n=size(integers,1)
    _vf_mpz_set_ui!(previous,1)
    for k in 1:n
        pivot=integers[k,k]
        pivot>=0 || return false
        if iszero(pivot)
            all(i->iszero(integers[k,i]),k+1:n) || return false
            continue
        end
        for i in k+1:n,j in i:n
            _vf_mpz_mul!(left,pivot,integers[i,j])
            _vf_mpz_mul!(right,integers[i,k],integers[k,j])
            _vf_mpz_sub!(numerator,left,right)
            _vf_mpz_tdiv_qr!(integers[i,j],remainder,numerator,previous)
            iszero(remainder) || error("exact positive-semidefinite elimination lost divisibility")
        end
        _vf_mpz_set!(previous,pivot)
    end
    true
end

function _vf_storage_exact(model,ci,P,a,f;max_bytes,retained_bytes::Integer=0)
    _vf_psd(P,0.;max_bytes,retained_bytes) || return false
    denbits,bits,bytes=_vf_storage_integer_workspace(model,P,a,f)
    _enforce_payload_limit(_checked_payload_sum("exact positive-real storage workspace",bytes,retained_bytes),
        max_bytes,"exact positive-real storage workspace","max_bytes")
    roles=(:prior_pivot,:left_operand,:right_operand,:product,:accumulator,
        :admittance_scale,:frequency_scale)
    previous,left,right,product,accumulator,admittance,frequency=ntuple(_->BigInt(;nbits=bits),length(roles))
    _vf_set_dyadic!(admittance,a,denbits);_vf_set_dyadic!(frequency,f,denbits)
    states=size(P,1);ports=size(model.d,1);n=states+ports
    storage=Matrix{BigInt}(undef,states,states)
    for i in 1:states,j in i:states
        value=_vf_set_dyadic!(BigInt(;nbits=bits),P[i,j],denbits)
        storage[i,j]=storage[j,i]=value
    end
    integers=Matrix{BigInt}(undef,n,n)
    for i in 1:n,j in i:n
        value=BigInt(;nbits=bits)
        if j<=states
            # A'P+PA: each real pole contributes a diagonal, each
            # conjugate mode contributes one additional coupling.
            for (column,other) in ((i,j),(j,i))
                k=cld(column,ports)
                _vf_set_dyadic!(left,real(model.poles[k]),denbits)
                _vf_mpz_mul!(product,left,storage[column,other])
                _vf_mpz_add!(value,value,product)
                if ci[k]!=0
                    partner=column+(ci[k]==1 ? ports : -ports)
                    _vf_set_dyadic!(left,-imag(model.poles[k]),denbits)
                    _vf_mpz_mul!(product,left,storage[partner,other])
                    _vf_mpz_add!(value,value,product)
                end
            end
            _vf_mpz_mul!(value,value,admittance)
            _vf_mpz_mul!(value,value,frequency)
            _vf_mpz_neg!(value,value)
        elseif i<=states
            port=j-states;k=cld(i,ports);column=mod1(i,ports)
            residue=ci[k]==2 ? imag(model.residues[k-1][port,column]) : real(model.residues[k][port,column])
            _vf_set_dyadic!(value,residue,denbits)
            ci[k]!=0 && _vf_mpz_mul_2exp!(value,1)
            _vf_mpz_mul_2exp!(value,3denbits)
            _vf_mpz_set_ui!(accumulator,0)
            for pole in eachindex(ci)
                ci[pole]==2 && continue
                _vf_mpz_add!(accumulator,accumulator,storage[i,(pole-1)*ports+port])
            end
            _vf_mpz_mul!(product,admittance,frequency)
            _vf_mpz_mul!(product,product,accumulator)
            _vf_mpz_mul_2exp!(product,denbits)
            _vf_mpz_sub!(value,value,product)
        else
            row,column=i-states,j-states
            _vf_set_dyadic!(value,model.d[row,column],denbits)
            _vf_set_dyadic!(left,model.d[column,row],denbits)
            _vf_mpz_add!(value,value,left)
            _vf_mpz_mul_2exp!(value,3denbits)
        end
        integers[i,j]=integers[j,i]=value
    end
    _vf_psd_integer!(integers,previous,left,right,product,accumulator)
end


# Reserve the named dense arrays owned by the Hamiltonian and storage
# stages. Array shapes follow the realization; temporary results are
# included conservatively. This is raw payload, excluding Julia headers
# and opaque LAPACK workspace, as with the other dense resource guards.
function _vf_hamiltonian_payload(states,n,npoles)
    state_square=(:state_matrix,:shifted_state_matrix,:input_gram,:output_gram,
        :storage_solve,:storage_symmetry,:input_gram_copy,:closed_state,
        :closed_product,:closed_eigen_input,:lyapunov_storage,:lyapunov_identity,
        :lyapunov_transpose,:curvature_left_product,:curvature_product,
        :curvature_singular_input,:storage_increment,:storage_sum,
        :scaled_input_gram,:scaled_output_gram,:negated_state_transpose)
    hamiltonian_square=(:hamiltonian,:schur_form,:schur_vectors,
        :ordered_schur_form,:ordered_schur_vectors,:hamiltonian_eigen_input)
    port_square=(:normalized_feedthrough,:hermitian_feedthrough,:feedthrough_sum,
        :feedthrough_factorization,:feedthrough_spectral_input)
    state_port=(:input_matrix,:output_matrix,:feedthrough_output_solve,
        :feedthrough_input_solve,:output_gram_solve)
    total=0
    for _ in state_square
        total=_checked_payload_sum("positive-real dense workspace",total,
            _checked_array_payload_bytes(Float64,states,states))
    end
    for _ in hamiltonian_square
        total=_checked_payload_sum("positive-real dense workspace",total,
            _checked_array_payload_bytes(Float64,2,states,2,states))
    end
    for _ in port_square
        total=_checked_payload_sum("positive-real dense workspace",total,
            _checked_array_payload_bytes(Float64,n,n))
    end
    for _ in state_port
        total=_checked_payload_sum("positive-real dense workspace",total,
            _checked_array_payload_bytes(Float64,states,n))
    end
    return _checked_payload_sum("positive-real dense workspace",total,
        _checked_array_payload_bytes(ComplexF64,n,n),
        _checked_array_payload_bytes(ComplexF64,npoles),
        _checked_array_payload_bytes(Int,npoles),
        _checked_array_payload_bytes(Int,states),
        _checked_array_payload_bytes(Float64,states),
        _checked_array_payload_bytes(Bool,2,states),
        _checked_array_payload_bytes(ComplexF64,2,states),
        _checked_array_payload_bytes(ComplexF64,2,states),
        _checked_array_payload_bytes(ComplexF64,2,states),
        _checked_array_payload_bytes(ComplexF64,states))
end

function _vf_storage_witness(model,ci,H,balancing,admittance_scale,frequency_scale;max_bytes,retained_bytes::Integer=0)
    states=size(H,1)÷2;n=size(model.d,1)
    factor=LinearAlgebra.schur(H)
    all(value->!iszero(real(value)),factor.values) || return false
    count(value->real(value)<0,factor.values)==states || return false
    ordered=LinearAlgebra.ordschur(factor,[real(value)<0 for value in factor.values])
    top=@view ordered.Z[1:states,1:states]
    bottom=@view ordered.Z[states+1:2states,1:states]
    P=try
        -balancing*(bottom/top)
    catch error
        error isa LinearAlgebra.SingularException || rethrow()
        return false
    end
    all(isfinite,P) || return false
    P=_vf_hermitian_part(P)
    Abar=@view H[1:states,1:states]
    G=-(@view H[1:states,states+1:2states])/balancing
    closed=Abar+G*P
    all(value->real(value)<0,eigvals(closed)) || return false
    # Moving into the storage-inequality interior avoids certifying a
    # floating approximation of an equality. Lyapunov gives L>0 and the
    # increment delta=1/(2*||LGL||) maximizes delta-delta^2*||LGL||.
    L=LinearAlgebra.lyap(Matrix(transpose(closed)),Matrix{Float64}(I,states,states))
    curvature=opnorm(L*G*L,2)
    isfinite(curvature) && curvature>0 || return false
    P+=ldexp.(L/curvature,-1)
    for i in 1:states,j in i:states
        P[i,j]=P[j,i]=(P[i,j]/2+P[j,i]/2)
    end
    all(isfinite,P) || return false
    return _vf_storage_exact(model,ci,P,admittance_scale,frequency_scale;max_bytes,retained_bytes)
end

"""Numerical positive-real certificate over all frequencies. A symmetric
positive-semidefinite affine term is required. Sufficient positive-residue
and uniform norm bounds also cover semidefinite feedthrough. Otherwise a
balanced real Hamiltonian locates every potential imaginary-axis zero of
the Hermitian admittance, and an exact positive-real storage inequality must also verify the
original stored model before the proper response is certified. `certified=false` is not proof
of nonpassivity; it can indicate a zero/tolerance boundary. The certificate
reports its method and potential crossover frequencies, not a grid claim.

The Hamiltonian criterion follows the positive-real state-space test;
see Semlyen & Gustavsen, IEEE TPWRD 24(1), 2009, DOI 10.1109/TPWRD.2008.923406. The exact original-model storage
inequality follows the positive-real lemma in Boyd et al., *Linear Matrix
Inequalities in System and Control Theory*, section 2.7.2 (1994).
Numerical absence of an imaginary-axis crossing alone is not a proof."""
function planar_rational_certificate(model::PlanarRationalModel;
        tol::Real=1e-8,max_bytes::Integer=_default_max_dense_payload_bytes())
    isfinite(tol) && tol>0 || throw(ArgumentError("certificate tolerance must be finite and positive"))
    no = (certified=false,method=:none,crossings_hz=Float64[])
    n = size(model.d,1)
    n>0 && size(model.d)==size(model.e)==(n,n) &&
        length(model.residues)==length(model.poles) &&
        all(isfinite,model.d) && all(isfinite,model.e) &&
        all(R -> size(R)==(n,n) && all(isfinite,R),model.residues) || return no
    all(p -> isfinite(p) && real(p)<0,model.poles) || return no
    _enforce_payload_limit(_checked_array_payload_bytes(Int,length(model.poles)),
        max_bytes,"positive-real certificate pole indexing","max_bytes")
    ci = _vf_pair_indices(model.poles)
    for k in eachindex(ci)
        if ci[k]==0
            all(x->iszero(imag(x)),model.residues[k]) || return no
            iszero(imag(model.poles[k])) || return no
        elseif ci[k]==1
            model.poles[k+1]==conj(model.poles[k]) &&
                all(i->model.residues[k+1][i]==conj(model.residues[k][i]),eachindex(model.residues[k])) || return no
        end
    end
    _vf_psd(model.e,tol;max_bytes,retained_bytes=_checked_array_payload_bytes(Int,length(ci))) || return no
    # Feedthrough Hermitian PSD is necessary at infinite frequency.
    reciprocal=model.d==transpose(model.d)
    _vf_psd(model.d,tol;hermitian=!reciprocal,max_bytes,retained_bytes=_checked_array_payload_bytes(Int,length(ci))) || return no
    # A stable pole with real PSD residue has nonnegative Hermitian energy.
    # Exact conjugate partners keep the complete model a real transfer.
    if (reciprocal || !isempty(model.poles)) && all(k ->
            _vf_psd(model.residues[k],tol;max_bytes,retained_bytes=_checked_array_payload_bytes(Int,length(ci))),eachindex(model.poles))
        return (certified=true,method=:positive_residues,crossings_hz=Float64[])
    end
    # Constant real feedthrough can be nonreciprocal: only its Hermitian
    # part contributes real power. It needs no pole-frequency scale.
    isempty(model.poles) && return (certified=true,method=:uniform_bound,crossings_hz=Float64[])
    certified_bound=_vf_frobenius_bound(model;max_bytes,retained_bytes=_checked_array_payload_bytes(Int,length(ci)))
    if isfinite(certified_bound) && _vf_psd(model.d,tol;hermitian=true,diagonal_shift=certified_bound,
            max_bytes,retained_bytes=_checked_array_payload_bytes(Int,length(ci)))
        return (certified=true,method=:uniform_bound,crossings_hz=Float64[])
    end
    states = _checked_array_payload_bytes(Float64,length(model.poles),n)÷sizeof(Float64)
    dense_payload=_vf_hamiltonian_payload(states,n,length(model.poles))
    _enforce_payload_limit(dense_payload,max_bytes,"positive-real certificate","max_bytes")
    # Positive admittance and frequency scaling preserve positive-realness.
    # Keep the Hamiltonian in those dimensionless units, instead of mixing
    # reciprocal conductance and its square across the Float64 range.
    admittance_scale=maximum(abs,model.d;init=0.0)
    iszero(admittance_scale) && return no
    frequency_scale=maximum(p->max(abs(real(p)),abs(imag(p))),model.poles;init=0.0)
    iszero(frequency_scale) && return no
    d=model.d/admittance_scale
    # A lost feedthrough component can erase a negative energy direction;
    # such a normalized representation cannot certify the original model.
    all(i->iszero(model.d[i]) || !iszero(d[i]),eachindex(d)) || return no
    poles=ComplexF64[p/frequency_scale for p in model.poles]
    all(k->isfinite(poles[k]) && real(poles[k])<0 &&
        (iszero(imag(model.poles[k])) || !iszero(imag(poles[k]))),eachindex(poles)) || return no
    symmetric_d=(d+transpose(d))/2
    minimum_d=minimum(eigvals(LinearAlgebra.Symmetric(symmetric_d)))
    residue=Matrix{ComplexF64}(undef,n,n)
    for k in eachindex(poles)
        residue.=complex.(_vf_certificate_ratio.(real.(model.residues[k]),admittance_scale,frequency_scale),
            _vf_certificate_ratio.(imag.(model.residues[k]),admittance_scale,frequency_scale))
        all(isfinite,residue) || return no
        all(i->(iszero(real(model.residues[k][i])) || !iszero(real(residue[i]))) &&
            (iszero(imag(model.residues[k][i])) || !iszero(imag(residue[i]))),eachindex(residue)) || return no
        # Range validation above protects the Hamiltonian representation.
    end
    # Approximate eigenvalues/norms are not sufficient uniform proofs.
    minimum_d>tol*opnorm(symmetric_d,2) || return no
    A,B,C = zeros(Float64,states,states),zeros(Float64,states,n),zeros(Float64,n,states)
    k = 1
    while k<=length(model.poles)
        left = ((k-1)*n+1):(k*n)
        p = poles[k]
        if ci[k]==0
            A[left,left] .= real(p)*Matrix{Float64}(I,n,n)
            B[left,:] .= Matrix{Float64}(I,n,n)
            C[:,left] .= _vf_certificate_ratio.(real.(model.residues[k]),admittance_scale,frequency_scale)
            k+=1
        else
            right = (k*n+1):((k+1)*n)
            A[left,left] .= real(p)*Matrix{Float64}(I,n,n)
            A[right,right] .= real(p)*Matrix{Float64}(I,n,n)
            A[left,right] .= imag(p)*Matrix{Float64}(I,n,n)
            A[right,left] .= -imag(p)*Matrix{Float64}(I,n,n)
            B[left,:] .= Matrix{Float64}(I,n,n)
            C[:,left] .= 2 .* _vf_certificate_ratio.(real.(model.residues[k]),admittance_scale,frequency_scale)
            C[:,right] .= 2 .* _vf_certificate_ratio.(imag.(model.residues[k]),admittance_scale,frequency_scale)
            k+=2
        end
    end
    R = d+transpose(d)
    Abar = A-B*(R\C)
    G,Q = B*(R\transpose(B)),transpose(C)*(R\C)
    qnorm,gnorm=opnorm(Q,Inf),opnorm(G,Inf)
    isfinite(qnorm) && isfinite(gnorm) && qnorm>0 && gnorm>0 || return no
    qm,qe=frexp(qnorm);gm,ge=frexp(gnorm);difference=qe-ge
    balancing=ldexp(sqrt(ldexp(qm/gm,mod(difference,2))),fld(difference,2))
    isfinite(balancing) && balancing>0 || return no
    H = [Abar -G*balancing; Q/balancing -transpose(Abar)]
    all(isfinite,H) || return no
    values = eigvals(H)
    all(isfinite,values) || return no
    crossings = sort!(unique([_vf_certificate_product_ratio(abs(imag(value)),frequency_scale,2pi) for value in values
        if abs(real(value)) <= tol*abs(imag(value))]))
    verified=isempty(crossings) && _vf_storage_witness(model,ci,H,balancing,admittance_scale,frequency_scale;max_bytes,retained_bytes=dense_payload)
    return (certified=verified,method=:hamiltonian,crossings_hz=crossings)
end

"""Fit a real stable shared-pole N-port model from finite Y matrices
(`format=:y`, default) or S matrices (`format=:s`, reference `z0`). The
affine term models capacitance. `enforce_passivity=true` projects the
affine term to PSD, adds conductance when needed to obtain a global
positive-real certificate, records both adjustments, and recomputes fit
error. The global norm-bound fallback can reduce fit accuracy; the model
reports that cost. Native MNA S blocks are preferable for ideal
shorts/thrus whose Y conversion is singular. `max_bytes` bounds fit
workspace before converting input data or allocating fit matrices."""
function planar_fit_rational(series::AbstractVector{<:AbstractMatrix},
        frequencies::AbstractVector{<:Real};format::Symbol=:y,z0=50.0,
        order::Integer=4,iterations::Integer=8,enforce_passivity::Bool=true,
        passivity_tol::Real=1e-9,n_passivity::Integer=max(257,8length(frequencies)),
        max_bytes::Integer=_default_max_dense_payload_bytes())
    m = length(series)
    m == length(frequencies) && m >= 3 || throw(ArgumentError("fit requires at least three matching samples"))
    order >= 1 && iterations >= 0 || throw(ArgumentError("order must be positive and iterations nonnegative"))
    order+2 < 2m || throw(ArgumentError("fit order exceeds the sample capacity"))
    isfinite(passivity_tol) && passivity_tol >= 0 && n_passivity >= 2 ||
        throw(ArgumentError("invalid passivity tolerance/grid"))
    format in (:y,:s) || throw(ArgumentError("fit format must be :y or :s"))
    n = size(series[1],1)
    n > 0 && all(Y -> size(Y)==(n,n) && all(isfinite,Y),series) ||
        throw(ArgumentError("fit matrices must be finite, square, and have equal sizes"))
    all(f -> isfinite(f) && f > 0,frequencies) || throw(ArgumentError("fit frequencies must be finite and positive"))
    perm = sortperm(frequencies)
    fs = Float64.(frequencies[perm])
    all(diff(fs) .> 0) || throw(ArgumentError("fit frequencies must be distinct"))
    # QR elimination avoids the giant sparse-in-structure shared solve.
    _enforce_payload_limit(_checked_payload_sum("vector fit",
        _checked_array_payload_bytes(ComplexF64,m,n,n),
        _checked_array_payload_bytes(Float64,8,m,n,n,order+2),
        _checked_array_payload_bytes(Float64,8,m,order+2),
        _checked_array_payload_bytes(ComplexF64,2,order+2,n,n)),
        max_bytes,"vector fit","max_bytes")
    Ys = Matrix{ComplexF64}[]
    for index in perm
        # Preserve the same dense workspace while checking each stored
        # component before it can disappear or become nonfinite.
        H = Matrix{ComplexF64}(undef,n,n)
        for (entry,value) in enumerate(series[index])
            H[entry] = _planar_stored_phasor(value)
        end
        if format === :s
            references=_circuit_z0(z0,n;freq=frequencies[index])
            H = planar_s_to_y(H,references;max_bytes)
        end
        push!(Ys,H)
    end
    scale = 2pi*last(fs)
    s = ComplexF64[2pi*1im*f/scale for f in fs]
    poles = ComplexF64[]
    betas = order÷2 <= 1 ? [0.5*(first(fs)/last(fs)+1)] :
        collect(range(first(fs)/last(fs),1.0;length=order÷2))
    for beta in betas
        length(poles)+2 <= order || break
        push!(poles,complex(-beta/100,beta),complex(-beta/100,-beta))
    end
    isodd(order) && push!(poles,complex(-0.5,0.0))
    poles = _vf_sort(poles)
    for _ in 1:iterations
        poles = _vf_shared_relocate(s,Ys,poles)
    end
    all(p->isfinite(p) && real(p)<0,poles) || throw(ArgumentError(
        "vector fit does not determine finite strictly decaying poles"))
    basis = _vf_basis(s,poles)
    Ar = vcat(real.(basis),imag.(basis))
    rhs = Matrix{Float64}(undef,2m,n*n)
    for e in 1:n*n,r in 1:m
        rhs[r,e],rhs[m+r,e] = real(Ys[r][e]),imag(Ys[r][e])
    end
    fitted = Ar \ rhs
    residues = [zeros(ComplexF64,n,n) for _ in poles]
    ci = _vf_pair_indices(poles)
    k = 1
    while k <= order
        if ci[k] == 0
            residues[k] .= reshape(view(fitted,k,:),n,n)*scale
            k += 1
        else
            residues[k] .= reshape(view(fitted,k,:),n,n)*scale +
                1im*reshape(view(fitted,k+1,:),n,n)*scale
            residues[k+1] .= conj.(residues[k])
            k += 2
        end
    end
    d = Matrix(reshape(fitted[end-1,:],n,n))
    e = Matrix(reshape(fitted[end,:],n,n))/scale
    poles .*= scale
    capacitance_adjustment = 0.0
    if enforce_passivity
        decomposition = eigen(LinearAlgebra.Symmetric((e+transpose(e))/2))
        floor = 32eps(Float64)*max(maximum(abs,decomposition.values),0)
        corrected = decomposition.vectors*Diagonal(max.(decomposition.values,floor))*transpose(decomposition.vectors)
        corrected = (corrected+transpose(corrected))/2
        capacitance_adjustment = norm(corrected-e)
        e .= corrected
    end
    provisional = PlanarRationalModel(poles,residues,d,e,fs,NaN,NaN,false,false,:none,NaN,0,capacitance_adjustment)
    grid = collect(range(0.0,last(fs);length=n_passivity))
    check = planar_rational_passivity(provisional,grid;tol=passivity_tol)
    shift = enforce_passivity && check.margin < -passivity_tol ?
        -check.margin+passivity_tol : 0.0
    for p in 1:n
        d[p,p] += shift
    end
    certificate = planar_rational_certificate(provisional;max_bytes=max_bytes)
    if enforce_passivity && !certificate.certified
        # A global norm bound is conservative but exact: each Hermitian
        # residue contribution is bounded below by -||R||/|Re(p)|.
        bound = _vf_frobenius_bound(provisional;max_bytes)
        smallest_d = minimum(eigvals(LinearAlgebra.Symmetric((d+transpose(d))/2)))
        extra = max(bound-smallest_d,0)+max(passivity_tol,1e-12*max(bound,1.0))
        for p in 1:n
            d[p,p] += extra
        end
        shift += extra
        certificate = planar_rational_certificate(provisional;max_bytes=max_bytes)
    end
    rms,relative = _vf_fit_errors(provisional,fs,Ys)
    margin = check.margin+shift
    return PlanarRationalModel(poles,residues,d,e,fs,rms,
        relative,margin>=-passivity_tol,
        certificate.certified,certificate.method,margin,shift,capacitance_adjustment)
end

function _vf_spice_damping(io,node,alpha)
    resistance=inv(alpha)
    if isfinite(resistance)
        println(io,"R",node," ",node," 0 ",resistance)
    else
        # A finite decay rate can have an unrepresentable reciprocal.
        # The self-controlled current source is the same conductance,
        # retaining that stable pole with a finite SPICE coefficient.
        println(io,"Gdecay",node," ",node," 0 ",node," 0 ",alpha)
    end
    return nothing
end

"""Write an N-port SPICE `.subckt` realizing the fitted admittance
exactly with real R/C elements and controlled sources. Pole equations
and state voltages are scaled by their decay rate where representable;
the affine term uses copied input voltages and sensed
capacitor currents. The final `dm_ref` terminal is a floating reference.
Auxiliary state voltages use node `0` to avoid subtracting tiny state
voltages from a large reference voltage; all external current and voltage
couplings use the declared reference terminal.
Export preserves the
model's sampled passivity status and reported fit error in comments."""
function planar_write_spice(model::PlanarRationalModel,path::AbstractString;
        subckt_name::AbstractString="diffmom_nport",port_names=nothing)
    n = size(model.d,1)
    size(model.d)==size(model.e)==(n,n) && length(model.residues)==length(model.poles) ||
        throw(ArgumentError("invalid rational model dimensions"))
    ports = port_names === nothing ? ["p$p" for p in 1:n] : String.(port_names)
    length(ports)==n && length(unique(lowercase.(ports)))==n &&
        all(p -> occursin(r"^[A-Za-z][A-Za-z0-9_]*$",p) &&
            lowercase(p)!="gnd" && !startswith(lowercase(p),"dm_"),ports) ||
        throw(ArgumentError("SPICE port names must be distinct identifiers outside the dm_ namespace"))
    occursin(r"^[A-Za-z][A-Za-z0-9_]*$",subckt_name) ||
        throw(ArgumentError("invalid SPICE subcircuit name"))
    all(p -> isfinite(p) && real(p)<0,model.poles) ||
        throw(ArgumentError("SPICE export requires stable finite poles"))
    ci = _vf_pair_indices(model.poles)
    all(isfinite,model.d) && all(isfinite,model.e) &&
        all(R -> size(R)==(n,n) && all(isfinite,R),model.residues) ||
        throw(ArgumentError("rational model coefficients are invalid"))
    # Real realization requires a real residue at every real pole and
    # conjugate residues at each complex pair.
    for k in eachindex(ci)
        if ci[k]==0
            iszero(imag(model.poles[k])) && all(x->iszero(imag(x)),model.residues[k]) ||
                throw(ArgumentError("real pole has a complex residue"))
        elseif ci[k]==1
            model.poles[k+1]==conj(model.poles[k]) &&
                all(i->model.residues[k+1][i]==conj(model.residues[k][i]),eachindex(model.residues[k])) ||
                throw(ArgumentError("pole residues are not conjugate pairs"))
        end
    end
    open(path,"w") do io
        println(io,"* DiffMoM real stable N-port admittance model")
        println(io,"* relative_rms_error: ",model.relative_rms_error)
        println(io,"* sampled_passive: ",model.sampled_passive,
            "  checked_margin: ",model.passivity_margin,"  conductance_shift: ",model.passivity_shift)
        println(io,"* globally_passive: ",model.globally_passive,
            "  certificate: ",model.passivity_method,"  capacitance_adjustment: ",model.capacitance_adjustment)
        println(io,".subckt ",subckt_name," ",join(ports," ")," dm_ref")
        for i in 1:n,j in 1:n
            iszero(model.d[i,j]) || println(io,"Gd",i,"_",j," ",ports[i],
                " dm_ref ",ports[j]," dm_ref ",model.d[i,j])
        end
        for j in 1:n
            any(!iszero,view(model.e,:,j)) || continue
            println(io,"Edm_copy",j," dm_v",j," 0 ",ports[j]," dm_ref 1")
            println(io,"Vdm_sense",j," dm_v",j," dm_c",j," 0")
            println(io,"Cdm_affine",j," dm_c",j," 0 1")
            for i in 1:n
                iszero(model.e[i,j]) || println(io,"Fdm_affine",i,"_",j,
                    " ",ports[i]," dm_ref Vdm_sense",j," ",model.e[i,j])
            end
        end
        state = 0
        k = 1
        while k <= length(model.poles)
            alpha = -real(model.poles[k])
            scale=max(1.0,alpha)
            for j in 1:n
                # Keep DC state voltages and output gains balanced together.
                # When a residue/alpha ratio cannot be stored, retain the
                # finite original residue scale for that input column.
                state_scale=all(r->isfinite(real(r)/alpha) && isfinite(imag(r)/alpha),
                    view(model.residues[k],:,j)) ? alpha : scale
                state += 1
                u = "dm_u$state"
                println(io,"C",u," ",u," 0 ",inv(scale))
                _vf_spice_damping(io,u,alpha/scale)
                # Paired states use twice the input drive and undoubled
                # residues. This retains finite coefficients even when
                # doubling a valid real/imaginary residue would overflow.
                println(io,"Gin",u," ",u," 0 ",ports[j]," dm_ref ",
                    (ci[k]==0 ? -1 : -2)*(state_scale/scale))
                if ci[k]==0
                    for i in 1:n
                        gain = real(model.residues[k][i,j])/state_scale
                        iszero(gain) || println(io,"Gs",u,"_",i," ",ports[i],
                            " dm_ref ",u," 0 ",gain)
                    end
                else
                    state += 1
                    v = "dm_u$state"
                    beta = imag(model.poles[k])/scale
                    println(io,"C",v," ",v," 0 ",inv(scale))
                    _vf_spice_damping(io,v,alpha/scale)
                    println(io,"Gx",u,"_",v," ",u," 0 ",v," 0 ",beta)
                    println(io,"Gx",v,"_",u," ",v," 0 ",u," 0 ",-beta)
                    for i in 1:n
                        residue = model.residues[k][i,j]
                        for (node,gain) in ((u,real(residue)/state_scale),(v,-imag(residue)/state_scale))
                            iszero(gain) || println(io,"Gs",node,"_",i," ",ports[i],
                                " dm_ref ",node," 0 ",gain)
                        end
                    end
                end
            end
            k += ci[k]==0 ? 1 : 2
        end
        println(io,".ends ",subckt_name)
    end
    return path
end
