# PlanarSolve.jl — Port excitation, solve, and network-parameter extraction
#
# Port model (Rautio & Harrington 1987 / co-calibrated port theory): a port
# is an infinitesimal gap-voltage source placed between the metal edge and
# the adjacent sidewall (wall port) or inside the sheet (internal port).
# The Galerkin RHS for a gap voltage V on basis b of lateral width w_b is
#     rhs_b = -s_q * w_b * V        (induced E cancels applied E on metal)
# where s_q = +1 for :west/:south ports and -1 for :east/:north ports so
# that positive V drives current *into* the network on every wall.  The
# port current into the network is I_p = s_p * sum_b w_b * i_b, hence
#     Y[p,q] = -s_p s_q * sum_{b in p, b' in q} w_b Z^{-1}[b,b'] w_b'.
# S-parameters follow from S = (I - Z0*Y)(I + Z0*Y)^{-1}.

export PlanarProblem, PlanarResult
export build_planar_problem, solve_planar, planar_sparams
export planar_y_to_s, write_touchstone

"""Assembled-ready planar problem: stackup, grid, sheets, ports, via
levels, and the rooftop basis built from them."""
struct PlanarProblem
    stack::PlanarStackup
    grid::CellGrid
    sheets::Vector{SheetLevel}
    ports::Vector{PlanarPort}
    vias::Vector{ViaLevel}
    basis::PlanarBasisSet
    vols::Vector{VolLevel}
end

# convenience constructors: no via/volume levels, or via only
PlanarProblem(stack::PlanarStackup, grid::CellGrid,
    sheets::Vector{SheetLevel}, ports::Vector{PlanarPort},
    basis::PlanarBasisSet) =
    PlanarProblem(stack, grid, sheets, ports, ViaLevel[], basis,
        VolLevel[])
PlanarProblem(stack::PlanarStackup, grid::CellGrid,
    sheets::Vector{SheetLevel}, ports::Vector{PlanarPort},
    vias::Vector{ViaLevel}, basis::PlanarBasisSet) =
    PlanarProblem(stack, grid, sheets, ports, vias, basis, VolLevel[])

"""Validate the stackup/ports and build the rooftop basis; every port must
claim at least one wall-connected edge.  `vias` adds z-directed via
columns on `ViaLevel`s (uniform and up-tapered profiles per cell);
`vols` adds thick-metal volume rooftops on `VolLevel`s (x- and
y-directed volume current distributed uniformly through each level's
layer).  Every level's `layer` must lie inside the stackup."""
function build_planar_problem(stack::PlanarStackup, grid::CellGrid,
        sheets::Vector{SheetLevel}, ports::Vector{PlanarPort};
        vias::Vector{ViaLevel}=ViaLevel[],
        vols::Vector{VolLevel}=VolLevel[])
    planar_validate(stack)
    stack.a == grid.a && stack.b == grid.b || throw(ArgumentError(
        "stackup box dimensions must match the cell grid"))
    isempty(ports) &&
        throw(ArgumentError("at least one port is required"))
    L = length(stack.layers)
    for (vl, vlvl) in enumerate(vias)
        1 <= vlvl.layer <= L || throw(ArgumentError(
            "vias[$vl] layer $(vlvl.layer) outside 1:$L"))
    end
    for (vl, vlvl) in enumerate(vols)
        1 <= vlvl.layer <= L || throw(ArgumentError(
            "vols[$vl] layer $(vlvl.layer) outside 1:$L"))
    end
    basis = build_planar_basis(grid, sheets, ports; vias=vias, vols=vols)
    planar_basis_count(basis) == 0 &&
        throw(ArgumentError("no basis functions: check masks/connections"))
    # every port must claim at least one basis
    claimed = falses(length(ports))
    @inbounds for p in eachindex(basis.port)
        basis.port[p] > 0 && (claimed[basis.port[p]] = true)
    end
    for (i, c) in enumerate(claimed)
        c || throw(ArgumentError(
            "port $i claimed no basis functions; check cells/edge/mask"))
    end
    return PlanarProblem(stack, grid, sheets, ports, vias, basis, vols)
end

"""Result of `solve_planar`: retained problem, MoM matrix and LU
factorization, per-port coefficient columns, short-circuit admittance `y`,
and S-parameters `s` normalized to the port reference impedances."""
struct PlanarResult{T<:Complex,F<:LinearAlgebra.LU}
    problem::PlanarProblem
    omega::ComplexF64              # complex (supports complex-step omega)
    freq::ComplexF64               # Hz
    z_mom::Union{Nothing,Matrix{ComplexF64}} # omitted with retain_matrix=false
    lu_fact::F
    currents::Matrix{T}   # checked physical coefficients, F64 or owned wide
    y::Matrix{ComplexF64}          # short-circuit port admittance [S]
    s::Matrix{ComplexF64}          # S-parameters normalized to port z0
    z0::Vector{ComplexF64}         # references evaluated at this frequency
    relative_residuals::Vector{Float64}
end

function PlanarResult(prob::PlanarProblem,omega,freq,Z,F,X,Y,S,z0)
    # Preserve the legacy ComplexF64 conversion for ordinary numeric
    # inputs. Only the solver's owned BigFloat extension retains its wider
    # representation; integer/Float32 complex arrays must not enter MPFR
    # consumer paths with non-BigFloat scalar components.
    supported=eltype(X) in (ComplexF64,Complex{BigFloat})
    stored=supported ? (X isa Matrix ? X : Matrix(X)) : ComplexF64.(X)
    PlanarResult(prob,omega,freq,Z,F,stored,Y,S,z0,Float64[])
end

# Preserve the original result constructor for callers retaining their
# own solved coefficients. New solver paths pass the evaluated refs directly.
PlanarResult(prob::PlanarProblem,omega,freq,Z,F,X,Y,S)=PlanarResult(prob,omega,freq,Z,F,X,Y,S,
    _planar_reference_values([p.z0 for p in prob.ports],length(prob.ports);freq=real(freq)))

"""Port-edge index lists from a basis set (basis.port marks port index)."""
function _port_basis_indices(basis::PlanarBasisSet, nports::Int)
    lists = [Int[] for _ in 1:nports]
    @inbounds for b in 1:planar_basis_count(basis)
        p = basis.port[b]
        p > 0 && push!(lists[p], b)
    end
    return lists
end

@inline _planar_port_sign(p::PlanarPort) =
    p.polarity * (p.wall in (:east,:north,:volume_east,:volume_north,
        :terminal_x_hi,:terminal_y_hi) ? -1.0 : 1.0)

# A distributed axial voltage V/h reacts with the average via profile:
# uniform = 1, up-taper = 1/2. Horizontal gap ports use the edge width.
@inline _planar_port_weight(basis::PlanarBasisSet, b::Int) =
    basis.width[b] * (basis.kind[b] == _BASIS_VIA_T ? 0.5 : 1.0)

using MPFR_jll: libmpfr
const _planar_mpfr_library=libmpfr
# MPFR_RNDN is the C API's round-to-nearest, ties-to-even value.
# The dependency-boundary tests confirm the loaded library's enum name.
const _planar_mpfr_nearest=Cint(0)
@inline function _planar_wide_set!(out::BigFloat,x::Float64)
    ccall((:mpfr_set_d,_planar_mpfr_library),Cint,(Ref{BigFloat},Cdouble,Cint),out,x,_planar_mpfr_nearest)
    out
end
@inline function _planar_wide_add!(out::BigFloat,a::BigFloat,b::BigFloat)
    ccall((:mpfr_add,_planar_mpfr_library),Cint,(Ref{BigFloat},Ref{BigFloat},Ref{BigFloat},Cint),out,a,b,_planar_mpfr_nearest)
    out
end
@inline function _planar_wide_sub!(out::BigFloat,a::BigFloat,b::BigFloat)
    ccall((:mpfr_sub,_planar_mpfr_library),Cint,(Ref{BigFloat},Ref{BigFloat},Ref{BigFloat},Cint),out,a,b,_planar_mpfr_nearest)
    out
end
@inline function _planar_wide_mul!(out::BigFloat,a::BigFloat,b::BigFloat)
    ccall((:mpfr_mul,_planar_mpfr_library),Cint,(Ref{BigFloat},Ref{BigFloat},Ref{BigFloat},Cint),out,a,b,_planar_mpfr_nearest)
    out
end
@inline function _planar_wide_div!(out::BigFloat,a::BigFloat,b::BigFloat)
    ccall((:mpfr_div,_planar_mpfr_library),Cint,(Ref{BigFloat},Ref{BigFloat},Ref{BigFloat},Cint),out,a,b,_planar_mpfr_nearest)
    out
end
function _planar_wide_current_parts(x,bits=_planar_current_precision(x))
    re=Matrix{BigFloat}(undef,size(x));im=similar(re)
    for i in eachindex(x)
        re[i]=BigFloat(real(x[i]);precision=bits);im[i]=BigFloat(imag(x[i]);precision=bits)
    end
    (re=re,im=im)
end
_planar_wide_dot_scratch(bits)=ntuple(_->BigFloat(0.;precision=bits),4)
function _planar_wide_physical_residual!(out,Z,parts,rhs,weights,scratch)
    real_sum,imag_sum,coefficient,product=scratch
    maximum_relative=0.
    for q in axes(rhs,2)
        residual_norm=0.;source_norm=0.
        for i in axes(rhs,1)
            _planar_wide_set!(real_sum,-real(rhs[i,q]));_planar_wide_set!(imag_sum,-imag(rhs[i,q]))
            for j in axes(Z,2)
                z=Z[i,j]
                _planar_wide_set!(coefficient,real(z));_planar_wide_mul!(product,coefficient,parts.re[j,q]);_planar_wide_add!(real_sum,real_sum,product)
                _planar_wide_set!(coefficient,-imag(z));_planar_wide_mul!(product,coefficient,parts.im[j,q]);_planar_wide_add!(real_sum,real_sum,product)
                _planar_wide_set!(coefficient,imag(z));_planar_wide_mul!(product,coefficient,parts.re[j,q]);_planar_wide_add!(imag_sum,imag_sum,product)
                _planar_wide_set!(coefficient,real(z));_planar_wide_mul!(product,coefficient,parts.im[j,q]);_planar_wide_add!(imag_sum,imag_sum,product)
            end
            value=complex(Float64(real_sum),Float64(imag_sum));out[i,q]=value
            # Normalize before narrowing, preserving a subnormal physical reaction.
            _planar_wide_set!(coefficient,weights[i])
            _planar_wide_div!(product,real_sum,coefficient);weighted_real=Float64(product)
            _planar_wide_div!(product,imag_sum,coefficient);weighted_imag=Float64(product)
            residual_norm=hypot(residual_norm,hypot(weighted_real,weighted_imag))
            source_norm=hypot(source_norm,abs(rhs[i,q])/weights[i])
        end
        maximum_relative=max(maximum_relative,residual_norm/source_norm)
    end
    maximum_relative
end
function _planar_wide_current_update!(parts,correction,scratch)
    coefficient=scratch[3]
    for i in eachindex(correction)
        _planar_wide_set!(coefficient,real(correction[i]));_planar_wide_sub!(parts.re[i],parts.re[i],coefficient)
        _planar_wide_set!(coefficient,imag(correction[i]));_planar_wide_sub!(parts.im[i],parts.im[i],coefficient)
    end
    parts
end

# Two Float64 significands cover a product; summation adds at most the
# binary digit count of similarly scaled terms. Round to complete machine limbs.
# This is an initial working precision, with acceptance checked independently
# against the original physical equation rather than inferred from bit count.
function _planar_dense_current_precision(nterms::Integer)
    nterms>=0 || throw(ArgumentError("current term count must be nonnegative"))
    limb_bits=8sizeof(UInt)
    required=2precision(Float64)+ndigits(max(nterms,1);base=2)
    cld(required,limb_bits)*limb_bits
end
const _PLANAR_DENSE_CURRENT_BITS=_planar_dense_current_precision(0)
const _PLANAR_DENSE_VOLTAGE_RTOL=1e-9
# The MPFR custom interface reports significand storage for the loaded library.
# Derive the remaining object cost from Julia's public representation measure;
# no per-call BigFloat allocation is needed for a payload calculation.
# https://www.mpfr.org/mpfr-current/mpfr.html#Custom-Interface
function _planar_wide_significand_payload(bits::Integer)
    0<bits<=typemax(Clong) || throw(ArgumentError("wide precision must be positive and fit MPFR's precision type"))
    ccall((:mpfr_custom_get_size,_planar_mpfr_library),Csize_t,(Clong,),bits)
end
const _planar_wide_scalar_metadata_payload=let bits=precision(Float64)
    Base.summarysize(BigFloat(0.;precision=bits))-Int(_planar_wide_significand_payload(bits))
end
function _planar_wide_scalar_payload(bits::Integer)
    _checked_payload_sum("wide current scalar",
        _planar_wide_scalar_metadata_payload,_planar_wide_significand_payload(bits))
end
# Charge two scalar object bounds and the complex value's reference storage.
# This is a representation-derived upper bound, including boxed/inline cases.
function _planar_wide_complex_payload(bits)
    _checked_payload_sum("owned wide complex value",
        2*_planar_wide_scalar_payload(bits),sizeof(Complex{BigFloat}))
end
function _planar_dense_source_rhs!(rhs,prob)
    fill!(rhs,0)
    for b in axes(rhs,1)
        port=prob.basis.port[b];port==0 && continue
        rhs[b,port]=-_planar_port_sign(prob.ports[port])*_planar_port_weight(prob.basis,b)
    end
    rhs
end
function _planar_dense_checked_currents(X,Z,F,prob,reserved,max_bytes;
        contracts=nothing,basis_scale=nothing,rtol=_PLANAR_DENSE_VOLTAGE_RTOL)
    nb,np=size(X);bits=_planar_dense_current_precision(nb)
    ordinary_bytes=_checked_payload_sum("dense physical residual",
        _checked_array_payload_bytes(ComplexF64,nb,np),
        _checked_array_payload_bytes(ComplexF64,nb),_checked_array_payload_bytes(Float64,nb))
    _enforce_payload_limit(_checked_payload_sum("dense physical solve",reserved,ordinary_bytes),max_bytes,"dense physical solve","max_bytes")
    rhs=similar(X)
    if contracts===nothing
        _planar_dense_source_rhs!(rhs,prob)
    else
        for q in axes(rhs,2)
            _planar_contracted_rhs!(view(rhs,:,q),prob,contracts,q)
        end
    end
    weights=Float64[_planar_port_weight(prob.basis,b) for b in 1:nb]
    all(w->isfinite(w) && w>0,weights) || throw(ArgumentError("source basis has invalid physical trace measures"))
    work=Vector{ComplexF64}(undef,nb);errors=Vector{Float64}(undef,np)
    for q in 1:np
        _planar_source_compensated_residual!(work,view(rhs,:,q),Z,view(X,:,q),false)
        numerator=0.;denominator=0.
        for b in 1:nb
            numerator=hypot(numerator,abs(work[b])/weights[b])
            denominator=hypot(denominator,abs(rhs[b,q])/weights[b])
        end
        errors[q]=numerator/denominator
    end
    all(e->isfinite(e) && e<=rtol,errors) && return X,errors,F
    scalarbytes=_planar_wide_scalar_payload(bits)
    wide_bytes=_checked_payload_sum("dense wide current recovery",ordinary_bytes,
        _checked_array_payload_bytes(UInt8,scalarbytes,2,nb,np),
        _checked_array_payload_bytes(UInt8,scalarbytes,4),
        _checked_array_payload_bytes(ComplexF64,nb,np),
        # The separate real/imaginary reference arrays and returned complex
        # reference array can coexist while sharing the same scalar data.
        _checked_array_payload_bytes(UInt8,2sizeof(BigFloat)+sizeof(Complex{BigFloat}),nb,np))
    _enforce_payload_limit(_checked_payload_sum("dense wide current solve",reserved,wide_bytes),max_bytes,"dense wide current solve","max_bytes")
    parts=_planar_wide_current_parts(X,bits);scratch=_planar_wide_dot_scratch(bits)
    correction=similar(rhs)
    relative=Inf
    previous=Inf
    # Each correction costs O(nb^2). A dimension-derived cap keeps total
    # refinement work O(nb^3), the same order as the direct recovery below.
    # Stop earlier when the original physical residual ceases to decrease.
    for iteration in 0:nb
        relative=_planar_wide_physical_residual!(correction,Z,parts,rhs,weights,scratch)
        isfinite(relative) && relative<=rtol && break
        (iteration==nb || !isfinite(relative) || relative>=previous) && break
        previous=relative
        all(isfinite,correction) || throw(ArgumentError("dense physical correction is outside finite Float64 range"))
        basis_scale===nothing || (correction .*= basis_scale)
        ldiv!(F,correction)
        basis_scale===nothing || (correction .*= basis_scale)
        all(isfinite,correction) || throw(ArgumentError("dense physical current correction is outside finite Float64 range"))
        _planar_wide_current_update!(parts,correction,scratch)
    end
    if !(isfinite(relative) && relative<=rtol)
        return _planar_dense_wide_factor_currents(Z,rhs,weights,reserved,wide_bytes,max_bytes;rtol)
    end
    currents=complex.(parts.re,parts.im)
    # Stored wide coefficients have owned precision. No global precision
    # change occurs in the residual or update loops.
    fill!(errors,relative)
    currents,errors,F
end
# A reusable complex arithmetic workspace: two products, two multiplication
# temporaries, denominator, two division numerators and the pivot norm.
const _planar_wide_factor_scratch_roles=(
    :product_re,:product_im,:t1,:t2,:denominator,:numerator_re,:numerator_im,:pivot_norm)
_planar_wide_factor_scratch(bits)=ntuple(
    _->BigFloat(0.;precision=bits),length(_planar_wide_factor_scratch_roles))
@inline function _planar_wide_set!(out::BigFloat,x::BigFloat)
    ccall((:mpfr_set,_planar_mpfr_library),Cint,
        (Ref{BigFloat},Ref{BigFloat},Cint),out,x,_planar_mpfr_nearest)
    out
end
function _planar_wide_complex_product!(re,im,a,b,scratch)
    t1,t2=scratch[3],scratch[4]
    _planar_wide_mul!(t1,real(a),real(b));_planar_wide_mul!(t2,imag(a),imag(b))
    _planar_wide_sub!(re,t1,t2)
    _planar_wide_mul!(t1,real(a),imag(b));_planar_wide_mul!(t2,imag(a),real(b))
    _planar_wide_add!(im,t1,t2)
    nothing
end
function _planar_wide_complex_divide!(target,source,divisor,scratch)
    t1,t2=scratch[3],scratch[4]
    den,nre,nim=scratch[5],scratch[6],scratch[7]
    a,b=real(source),imag(source);c,d=real(divisor),imag(divisor)
    _planar_wide_mul!(t1,c,c);_planar_wide_mul!(t2,d,d);_planar_wide_add!(den,t1,t2)
    iszero(den) && throw(LinearAlgebra.SingularException(0))
    _planar_wide_mul!(t1,a,c);_planar_wide_mul!(t2,b,d);_planar_wide_add!(nre,t1,t2)
    _planar_wide_mul!(t1,b,c);_planar_wide_mul!(t2,a,d);_planar_wide_sub!(nim,t1,t2)
    _planar_wide_div!(real(target),nre,den);_planar_wide_div!(imag(target),nim,den)
    nothing
end
function _planar_wide_squared_norm!(output,value,scratch)
    t1,t2=scratch[3],scratch[4]
    _planar_wide_mul!(t1,real(value),real(value));_planar_wide_mul!(t2,imag(value),imag(value))
    _planar_wide_add!(output,t1,t2)
end
function _planar_owned_wide_lu!(matrix::Matrix{Complex{BigFloat}},scratch)
    n=size(matrix,1);size(matrix,2)==n || throw(DimensionMismatch("wide factor must be square"))
    pivots=Vector{Int}(undef,n)
    pr,pi=scratch[1],scratch[2];candidate,best=scratch[5],scratch[8]
    for k in 1:n
        pivot=k;_planar_wide_squared_norm!(best,matrix[k,k],scratch)
        for row in k+1:n
            _planar_wide_squared_norm!(candidate,matrix[row,k],scratch)
            if candidate>best
                pivot=row;_planar_wide_set!(best,candidate)
            end
        end
        pivots[k]=pivot
        iszero(matrix[pivot,k]) && throw(LinearAlgebra.SingularException(k))
        if pivot!=k
            for column in 1:n
                matrix[k,column],matrix[pivot,column]=matrix[pivot,column],matrix[k,column]
            end
        end
        for row in k+1:n
            _planar_wide_complex_divide!(matrix[row,k],matrix[row,k],matrix[k,k],scratch)
        end
        for column in k+1:n,row in k+1:n
            _planar_wide_complex_product!(pr,pi,matrix[row,k],matrix[k,column],scratch)
            target=matrix[row,column]
            _planar_wide_sub!(real(target),real(target),pr)
            _planar_wide_sub!(imag(target),imag(target),pi)
        end
    end
    LinearAlgebra.LU(matrix,pivots,0)
end
function _planar_owned_wide_factor_solve!(factor,currents::Matrix{Complex{BigFloat}},scratch;
        transposed::Bool=false)
    A=factor.factors;n=size(A,1)
    size(A,2)==n && size(currents,1)==n || throw(DimensionMismatch("wide factor and source dimensions do not match"))
    factor.info==0 || throw(LinearAlgebra.SingularException(factor.info))
    pr,pi=scratch[1],scratch[2]
    subtract!(target,a,b)=begin
        _planar_wide_complex_product!(pr,pi,a,b,scratch)
        _planar_wide_sub!(real(target),real(target),pr)
        _planar_wide_sub!(imag(target),imag(target),pi)
    end
    if transposed
        # A = P' L U, so A^T x = b solves U^T, L^T, then P' in order.
        for column in axes(currents,2),row in 1:n
            for j in 1:row-1;subtract!(currents[row,column],A[j,row],currents[j,column]);end
            _planar_wide_complex_divide!(currents[row,column],currents[row,column],A[row,row],scratch)
        end
        for column in axes(currents,2),row in n:-1:1
            for j in row+1:n;subtract!(currents[row,column],A[j,row],currents[j,column]);end
        end
        for k in n:-1:1
            pivot=factor.ipiv[k]
            if pivot!=k
                for column in axes(currents,2)
                    currents[k,column],currents[pivot,column]=currents[pivot,column],currents[k,column]
                end
            end
        end
    else
        for k in 1:n
            pivot=factor.ipiv[k]
            if pivot!=k
                for column in axes(currents,2)
                    currents[k,column],currents[pivot,column]=currents[pivot,column],currents[k,column]
                end
            end
        end
        for column in axes(currents,2),row in 1:n
            for j in 1:row-1;subtract!(currents[row,column],A[row,j],currents[j,column]);end
        end
        for column in axes(currents,2),row in n:-1:1
            for j in row+1:n;subtract!(currents[row,column],A[row,j],currents[j,column]);end
            _planar_wide_complex_divide!(currents[row,column],currents[row,column],A[row,row],scratch)
        end
    end
    currents
end

function _planar_dense_wide_factor_currents(Z,rhs,weights,reserved,wide_bytes,max_bytes;
        rtol=_PLANAR_DENSE_VOLTAGE_RTOL)
    nb,np=size(rhs);bits=_planar_dense_current_precision(nb)
    scalar=_planar_wide_scalar_payload(bits)
    payload=_checked_payload_sum("dense wide factor recovery",reserved,wide_bytes,
        _checked_array_payload_bytes(UInt8,_planar_wide_complex_payload(bits),nb,nb),
        _checked_array_payload_bytes(UInt8,_planar_wide_complex_payload(bits)+2sizeof(BigFloat),nb,np),
        _checked_array_payload_bytes(Int,nb),
        _checked_array_payload_bytes(UInt8,scalar,length(_planar_wide_factor_scratch_roles)))
    _enforce_payload_limit(payload,max_bytes,"dense wide factor recovery","max_bytes")
    setprecision(BigFloat,bits) do
        setrounding(BigFloat,RoundNearest) do
            matrix=Matrix{Complex{BigFloat}}(undef,nb,nb)
            for j in 1:nb,i in 1:nb
                matrix[i,j]=Complex{BigFloat}(Z[i,j])
            end
            factor_scratch=_planar_wide_factor_scratch(bits)
            factor=_planar_owned_wide_lu!(matrix,factor_scratch)
            currents=Complex{BigFloat}.(rhs)
            _planar_owned_wide_factor_solve!(factor,currents,factor_scratch)
            all(isfinite,currents) || throw(ArgumentError("dense wide factor currents are non-finite"))
            parts=_planar_wide_current_parts(currents,bits)
            work=similar(rhs);scratch=_planar_wide_dot_scratch(bits)
            relative=_planar_wide_physical_residual!(work,Z,parts,rhs,weights,scratch)
            isfinite(relative) && relative<=rtol ||
                throw(ErrorException("dense planar wide factor physical voltage residual failed: $relative > $rtol"))
            currents,fill(relative,np),factor
        end
    end
end
function _planar_owned_current_product(currents,contracts,bits)
    rows,inner=size(currents)
    ndims(contracts) in (1,2) && size(contracts,1)==inner ||
        throw(DimensionMismatch("current contraction dimensions do not match"))
    columns=size(contracts,2)
    output=Matrix{Complex{BigFloat}}(undef,rows,columns)
    for i in eachindex(output)
        output[i]=complex(BigFloat(0.;precision=bits),BigFloat(0.;precision=bits))
    end
    # One real and one imaginary contract coefficient, plus two products,
    # suffice for both components of a complex multiply/accumulate.
    contract_re=BigFloat(0.;precision=bits)
    contract_im=BigFloat(0.;precision=bits)
    product=BigFloat(0.;precision=bits)
    temporary=BigFloat(0.;precision=bits)
    for column in 1:columns,row in 1:rows
        re=real(output[row,column]);im=imag(output[row,column])
        for q in 1:inner
            value=contracts[q,column]
            _planar_wide_set!(contract_re,Float64(real(value)))
            _planar_wide_set!(contract_im,Float64(imag(value)))
            source=currents[row,q]
            _planar_wide_mul!(product,real(source),contract_re)
            _planar_wide_mul!(temporary,imag(source),contract_im)
            _planar_wide_sub!(product,product,temporary)
            _planar_wide_add!(re,re,product)
            _planar_wide_mul!(product,real(source),contract_im)
            _planar_wide_mul!(temporary,imag(source),contract_re)
            _planar_wide_add!(product,product,temporary)
            _planar_wide_add!(im,im,product)
        end
    end
    ndims(contracts)==1 ? vec(output) : output
end
function _planar_owned_current_product_payload(bits,dimensions...)
    scalar=_planar_wide_scalar_payload(bits)
    # Each output owns two real MPFR scalars and their complex references.
    # The four scalar temporaries above are the only arithmetic workspace.
    _checked_payload_sum("owned current product",
        _checked_array_payload_bytes(UInt8,_planar_wide_complex_payload(bits),dimensions...),
        _checked_array_payload_bytes(UInt8,scalar,4))
end

function _planar_current_product(currents,contracts)
    eltype(currents)===ComplexF64 && return currents*contracts
    bits=_planar_current_precision(currents)
    setprecision(BigFloat,bits) do
        setrounding(BigFloat,RoundNearest) do
            eltype(contracts) in (Float64,ComplexF64) ?
                _planar_owned_current_product(currents,contracts,bits) : currents*contracts
        end
    end
end
function _planar_dense_port_y(prob,pbs,sgn,X)
    np=length(prob.ports);Y=Matrix{ComplexF64}(undef,np,np)
    evaluate()=begin
        for q in 1:np,p in 1:np
            acc=zero(eltype(X))
            for b in pbs[p]
                acc+=sgn[p]*_planar_port_weight(prob.basis,b)*X[b,q]
            end
            Y[p,q]=_planar_stored_phasor(acc)
        end
        Y
    end
    eltype(X)===ComplexF64 && return evaluate()
    setprecision(BigFloat,_planar_current_precision(X)) do
        setrounding(evaluate,BigFloat,RoundNearest)
    end
end

"""
    solve_planar(problem, freq; kw...) -> PlanarResult

Assemble the shielded planar MoM system at `freq` Hz, drive each port with
a unit gap voltage, and extract the short-circuit admittance and
S-parameters. Physical voltage residuals use the stored original matrix.
Rare low-frequency recovery derives its initial precision from Float64
significands, the system dimension and machine limb size, and retains the
owned current precision while
reusing the Float64 factorization. `retain_matrix=false` returns
`z_mom=nothing`. The dense compact path factors in place and reuses the
assembly's real and imaginary accumulators for exact residual checks,
saving a matrix allocation. Checked solving needs two matrices at peak
and retains only the factorization afterward. `max_bytes` bounds operation-owned raw array
payloads and is checked before assembly.
"""
function solve_planar(prob::PlanarProblem, freq::Number;
        method::Symbol=:dense,
        retain_matrix::Bool=true,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES, kw...)
    method === :ufft && return solve_planar_ufft(prob, freq;
        max_bytes=max_bytes, kw...)
    method in (:dense,:dense_fft) || throw(ArgumentError("method must be :dense, :dense_fft or :ufft"))
    omega = 2pi * ComplexF64(freq)
    isfinite(omega) && real(omega) > 0 ||
        throw(ArgumentError(
            "freq must be finite with Re > 0, got $freq"))
    nb = planar_basis_count(prob.basis)
    nports = length(prob.ports)

    base_est = _checked_dense_lu_work_bytes(ComplexF64, nb,
        2,
        _checked_array_payload_bytes(ComplexF64, nb, nports),
        # Y and the two buffers used by power-wave conversion.
        _checked_array_payload_bytes(ComplexF64, 3, nports, nports),
        _checked_array_payload_bytes(Int, nb), # port index lists
        _checked_array_payload_bytes(Int, nports), # conversion LU pivots
        _checked_array_payload_bytes(Float64, 3, nports),
        _checked_array_payload_bytes(ComplexF64, 2, nports);
        label="planar solve")
    physical_work=_checked_payload_sum("dense physical workspace",
        _checked_array_payload_bytes(ComplexF64,nb,nports),
        _checked_array_payload_bytes(ComplexF64,nb),_checked_array_payload_bytes(Float64,nb))
    est=_checked_payload_sum("dense checked solve",base_est,physical_work)
    _enforce_payload_limit(est, max_bytes, "planar solve", "max_bytes")
    # Validate reference impedances before an expensive assembly/solve.
    z0 = _planar_reference_values([p.z0 for p in prob.ports], nports;freq=real(freq))

    compact_components=!retain_matrix && method===:dense
    assembled = if method===:dense_fft
        matrix_bytes=_checked_array_payload_bytes(ComplexF64,nb,nb)
        assemble_planar_z_ufft(prob,freq;max_bytes=max_bytes-(est-matrix_bytes),kw...)
    else
        assemble_planar_z(prob.stack,prob.grid,prob.sheets,prob.basis,omega;
            vias=prob.vias,vols=prob.vols,_retain_components=compact_components,max_bytes,kw...)
    end
    Z=compact_components ? assembled.matrix : assembled
    physical=assembled
    pbs = _port_basis_indices(prob.basis, nports)

    # a non-finite entry means an exact box resonance (infinite modal
    # voltage) -- LU would return silent garbage, so fail loudly
    all(isfinite, Z) || throw(ArgumentError(
        "planar impedance matrix is non-finite: frequency is at/near " *
        "a box resonance"))
    F = compact_components ? lu!(Z) : lu(Z)
    # s_p: +1 for :west/:south, -1 for :east/:north (into-network direction)
    sgn = [_planar_port_sign(p) for p in prob.ports]
    RHS = zeros(ComplexF64, nb, nports)
    @inbounds for q in 1:nports
        isempty(pbs[q]) && throw(ArgumentError(
            "port $q claimed no basis functions"))
        for b in pbs[q]
            RHS[b, q] = -sgn[q] * _planar_port_weight(prob.basis, b)
        end
    end
    X = ldiv!(F, RHS)   # solves in place over the RHS buffer
    X,relative_residuals,F=_planar_dense_checked_currents(X,physical,F,prob,base_est,max_bytes)

    Y = _planar_dense_port_y(prob,pbs,sgn,X)

    S = planar_y_to_s(Y, z0)
    return PlanarResult(prob, omega, ComplexF64(freq),
        retain_matrix ? Z : nothing, F, X, Y, S, z0, relative_residuals)
end

"""`planar_sparams(problem, freqs; kw...) -> (S vector of matrices)`.
Sweeps `freqs` (Hz) re-assembly per point."""
function planar_sparams(prob::PlanarProblem,
        freqs::AbstractVector{<:Number}; retain_matrix::Bool=false, kw...)
    return [solve_planar(prob, f; retain_matrix=retain_matrix, kw...).s
            for f in freqs]
end

function _planar_reference_roots(z0::AbstractVector)
    r = Vector{Float64}(undef, length(z0))
    for (p, z) in enumerate(z0)
        r[p] = sqrt(real(_planar_reference_number(z)))
    end
    return r
end

"""
    planar_y_to_s(Y, z0) -> S

Convert short-circuit admittance matrix to S-parameters against per-port
reference impedances `z0` (finite, positive real parts). Kurokawa power
waves use `a=(V+Z0*I)/(2sqrt(real(Z0)))` and
`b=(V-conj(Z0)*I)/(2sqrt(real(Z0)))`. For complex references, an exact
short reflects as `-conj(Z0)/Z0` and conjugate matching gives zero reflection.
"""
function planar_y_to_s(Y::AbstractMatrix{<:Number},
        z0::AbstractVector)
    n = size(Y, 1)
    size(Y, 2) == n && length(z0) == n ||
        throw(DimensionMismatch("Y must be n x n with n = length(z0)"))
    n > 0 || throw(ArgumentError("Y must have at least one port"))
    all(isfinite, Y) || throw(ArgumentError("Y must be finite"))
    r = _planar_reference_roots(z0)
    K = Matrix{ComplexF64}(undef, n, n)
    @inbounds for q in 1:n, p in 1:n
        K[p, q] = (Y[p,q] * ComplexF64(z0[q])) * (r[p]/r[q])
        p == q && (K[p, q] += 1)
    end
    all(isfinite,K) || throw(ArgumentError("power-wave admittance normalization exceeds finite ComplexF64 values"))
    # K = I + H*Q, H=√R Y√R, Q=Z0/ReZ0. Rewriting I-2K⁻¹H
    # as 2K⁻¹Q⁻¹-diag(conj(Z0)/Z0) preserves tiny transmission
    # entries near a short. The second buffer starts as the identity.
    S=Matrix{ComplexF64}(I,n,n)
    ldiv!(lu!(K), S)
    @inbounds for q in 1:n, p in 1:n
        ref=ComplexF64(z0[q])
        diagonal=p==q ? 1.0 : 0.0
        S[p, q] = diagonal+2*(real(ref)/ref)*(S[p,q]-diagonal)
    end
    all(isfinite,S) || throw(ArgumentError("power-wave scattering conversion is nonfinite"))
    return S
end

"""Write an `n`-port Touchstone (.sNp) file, S in real/imag pairs, Hz.
Two-port files list `S11 S21 S12 S22` on the frequency line; `n > 2`
files put the frequency on its own line followed by one matrix row per
line in row-major order."""
function write_touchstone(path::AbstractString,
        freqs::AbstractVector{<:Real},
        S::AbstractVector{<:AbstractMatrix},
        z0::Real=50.0)
    length(freqs) == length(S) ||
        throw(DimensionMismatch("freqs and S lengths differ"))
    isempty(S) && throw(ArgumentError("at least one frequency is required"))
    isfinite(z0) && z0 > 0 || throw(ArgumentError(
        "reference impedance must be finite and positive"))
    n = size(S[1], 1)
    n > 0 || throw(ArgumentError("S must have at least one port"))
    # Validate the complete dataset before opening an existing file.
    for k in eachindex(freqs)
        isfinite(freqs[k]) && freqs[k] >= 0 || throw(ArgumentError(
            "freqs[$k] must be finite and nonnegative"))
        size(S[k]) == (n, n) || throw(DimensionMismatch(
            "S[$k] is $(size(S[k])), expected $n x $n"))
        all(isfinite, S[k]) || throw(ArgumentError("S[$k] must be finite"))
    end
    open(path, "w") do io
        println(io, "! DiffMoM planar S-parameters")
        println(io, "# HZ S RI R ", Float64(z0))
        for k in eachindex(freqs)
            mat = S[k]
            if n <= 2
                print(io, Float64(freqs[k]))
                for q in 1:n, p in 1:n
                    print(io, "  ", real(mat[p, q]), "  ",
                        imag(mat[p, q]))
                end
                println(io)
            else
                println(io, Float64(freqs[k]))
                for p in 1:n
                    for q in 1:n
                        print(io, "  ", real(mat[p, q]), "  ",
                            imag(mat[p, q]))
                    end
                    println(io)
                end
            end
        end
    end
    return path
end
