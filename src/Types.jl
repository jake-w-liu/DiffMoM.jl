# Types.jl — Core data structures for DiffMoM

export Vec3, CVec3, TriMesh, RWGData, LocalMassMatrix, PatchPartition, SphGrid, ScatteringResult
export nvertices, ntriangles

const Vec3 = SVector{3,Float64}
const CVec3 = SVector{3,ComplexF64}

# Public routines that materialize dense arrays use this as their default raw
# payload ceiling.  The limit is deliberately configurable at each call site;
# it bounds the arrays owned by that operation, not Julia object headers or
# allocator bookkeeping.
const _DEFAULT_MAX_DENSE_PAYLOAD_BYTES = 2_000_000_000

function _validated_resource_limit(
        name::AbstractString, value::Integer)
    limit = try
        Int(value)
    catch err
        err isa InexactError || rethrow()
        throw(ArgumentError("$name is outside the Int range"))
    end
    limit > 0 ||
        throw(ArgumentError("$name must be positive, got $value"))
    return limit
end

function _validated_nonnegative_resource_limit(
        name::AbstractString, value::Integer)
    limit = try
        Int(value)
    catch err
        err isa InexactError || rethrow()
        throw(ArgumentError("$name is outside the Int range"))
    end
    limit >= 0 ||
        throw(ArgumentError("$name must be nonnegative, got $value"))
    return limit
end

function _checked_array_payload_bytes(
        ::Type{T}, dimensions::Integer...;
        label::AbstractString="array") where {T}
    entries = 1
    try
        for dimension in dimensions
            dimension >= 0 ||
                throw(ArgumentError(
                    "$label dimensions must be nonnegative, got $dimensions"))
            entries = Base.Checked.checked_mul(entries, Int(dimension))
        end
        return Base.Checked.checked_mul(entries, sizeof(T))
    catch err
        err isa OverflowError || err isa InexactError || rethrow()
        throw(ArgumentError("$label raw-payload estimate overflows Int"))
    end
end

function _enforce_payload_limit(
        payload_bytes::Int,
        max_bytes::Integer,
        label::AbstractString,
        limit_name::AbstractString)
    limit = _validated_resource_limit(limit_name, max_bytes)
    payload_bytes <= limit ||
        throw(ArgumentError(
            "$label requires $payload_bytes raw bytes, exceeding " *
            "$limit_name=$limit"))
    return payload_bytes
end

@inline _complex_vector_input(x::AbstractVector{ComplexF64}) = x
@inline _complex_vector_input(x::AbstractVector) = Vector{ComplexF64}(x)

"""
Triangle mesh: vertices and triangle connectivity.
"""
struct TriMesh
    xyz::Matrix{Float64}        # (3, Nv) vertex coordinates
    tri::Matrix{Int}            # (3, Nt) 1-based vertex indices per triangle
end

nvertices(m::TriMesh) = size(m.xyz, 2)
ntriangles(m::TriMesh) = size(m.tri, 2)

"""
RWG basis function data.

`T` is the coefficient type used to weight each side of a basis function.
For standard RWG, `T=Float64` with unit side coefficients.
For Bloch-periodic paired edges, `T=ComplexF64` allows phase factors on one side.
"""
struct RWGData{T<:Number}
    mesh::TriMesh
    nedges::Int
    tplus::Vector{Int}          # T+ triangle index
    tminus::Vector{Int}         # T- triangle index
    evert::Matrix{Int}          # (2, nedges) edge vertex indices
    vplus_opp::Vector{Int}      # opposite vertex in T+
    vminus_opp::Vector{Int}     # opposite vertex in T-
    len::Vector{Float64}        # edge length
    area_plus::Vector{Float64}  # area of T+
    area_minus::Vector{Float64} # area of T-
    coeff_plus::Vector{T}       # multiplicative factor on T+ side
    coeff_minus::Vector{T}      # multiplicative factor on T- side
    has_periodic_bloch::Bool    # true when built with periodic Bloch constraints
end

"""
Compact sparse matrix for local RWG mass blocks.

Stores only triplet entries `(rows[k], cols[k], vals[k])` of an `n x n`
matrix. Duplicate triplets are allowed and are summed by `getindex`, `mul!`,
and dense accumulation helpers. This avoids the `O(n)` column-pointer storage
cost of one `SparseMatrixCSC` per triangle/patch while preserving an
`AbstractMatrix` interface for existing code.
"""
struct LocalMassMatrix{T<:Number} <: AbstractMatrix{T}
    n::Int
    rows::Vector{Int}
    cols::Vector{Int}
    vals::Vector{T}
    col_order::Vector{Int}
    function LocalMassMatrix{T}(n::Int,
                                rows::Vector{Int},
                                cols::Vector{Int},
                                vals::Vector{T}) where {T<:Number}
        n >= 0 ||
            throw(ArgumentError("LocalMassMatrix size must be nonnegative, got $n"))
        length(rows) == length(cols) == length(vals) ||
            throw(DimensionMismatch("LocalMassMatrix triplet arrays must have equal lengths."))
        @inbounds for k in eachindex(rows)
            1 <= rows[k] <= n ||
                throw(ArgumentError("LocalMassMatrix row index $(rows[k]) at triplet $k is outside 1:$n"))
            1 <= cols[k] <= n ||
                throw(ArgumentError("LocalMassMatrix column index $(cols[k]) at triplet $k is outside 1:$n"))
            isfinite(vals[k]) ||
                throw(ArgumentError(
                    "LocalMassMatrix value at triplet $k must be finite, got $(vals[k])"))
        end
        canonical_rows, canonical_cols, canonical_vals, col_order =
            _canonical_local_mass_triplets(rows, cols, vals)
        return new{T}(
            n, canonical_rows, canonical_cols, canonical_vals, col_order)
    end
end

LocalMassMatrix(n::Int, rows::Vector{Int}, cols::Vector{Int}, vals::Vector{T}) where {T<:Number} =
    LocalMassMatrix{T}(n, rows, cols, vals)

Base.size(M::LocalMassMatrix) = (M.n, M.n)
Base.eltype(::Type{LocalMassMatrix{T}}) where {T<:Number} = T
Base.eltype(::LocalMassMatrix{T}) where {T<:Number} = T

# Each component of α * M[k] * x contains at most four real triple products.
# Three finite Float64 coefficients need at most 6295 bits, and summing every
# addressable triplet adds fewer than 64 bits. The remaining precision is a
# guard margin for the exceptional accumulation path.
const _LOCAL_MASS_FALLBACK_PRECISION = 6656

@inline _local_mass_real_type(::Type{Complex{T}}) where {T<:AbstractFloat} = T
@inline _local_mass_real_type(::Type{T}) where {T<:AbstractFloat} = T
@inline _local_mass_real_type(::Type) = Float64

@inline function _local_mass_component_scale(value::Number)
    return max(
        abs(Float64(real(value))),
        abs(Float64(imag(value))),
    )
end

@inline function _local_mass_finite(value::Number)
    return isfinite(real(value)) && isfinite(imag(value))
end

@inline function _local_mass_component_magnitudes(value::Number)
    return abs(Float64(real(value))), abs(Float64(imag(value)))
end

@inline _local_mass_ieee_real_type(::Type{Float64}) = Float64
@inline _local_mass_ieee_real_type(::Type{ComplexF64}) = Float64
@inline _local_mass_ieee_real_type(::Type{Float32}) = Float32
@inline _local_mass_ieee_real_type(::Type{ComplexF32}) = Float32
@inline _local_mass_ieee_real_type(::Type) = nothing

# Within these component-exponent bands, every staged two- or three-factor
# product in a LocalMassMatrix operation remains safely inside the IEEE range,
# including the fewer than 64 reduction bits available to an Int-indexed
# container.  Exceptional factors use the exact BigFloat accumulator so that
# a rounded-away component can still contribute to a representable result.
@inline _local_mass_safe_factor_exponent(::Type{Float64}) = 128
@inline _local_mass_safe_factor_exponent(::Type{Float32}) = 16

@inline function _local_mass_extreme_component(
        component::Real,
        ::Type{R}) where {R<:Union{Float32,Float64}}
    magnitude = abs(R(component))
    isfinite(magnitude) || return true
    iszero(magnitude) && return false
    bound = _local_mass_safe_factor_exponent(R)
    component_exponent = exponent(magnitude)
    return component_exponent < -bound || component_exponent > bound
end

@inline function _local_mass_extreme_factor(
        value::Number,
        ::Type{R}) where {R<:Union{Float32,Float64}}
    return _local_mass_extreme_component(real(value), R) ||
           _local_mass_extreme_component(imag(value), R)
end

function _local_mass_sum_group(
        vals::Vector{T}, order::Vector{Int}, first::Int, last::Int) where {T}
    first == last && return vals[order[first]]
    total = zero(T)
    real_magnitude = 0.0
    imag_magnitude = 0.0
    requires_exact = false
    @inbounds for position in first:last
        value = vals[order[position]]
        total += value
        real_scale, imag_scale = _local_mass_component_magnitudes(value)
        real_magnitude += real_scale
        imag_magnitude += imag_scale
        requires_exact |= !_local_mass_finite(total) ||
                          !isfinite(real_magnitude) ||
                          !isfinite(imag_magnitude)
    end

    real_type = _local_mass_real_type(T)
    if real_type <: Union{Float16,Float32,Float64}
        factor = 4 * eps(real_type) * (last - first + 1)
        requires_exact |=
            (!iszero(real_magnitude) &&
             abs(Float64(real(total))) <= factor * real_magnitude) ||
            (!iszero(imag_magnitude) &&
             abs(Float64(imag(total))) <= factor * imag_magnitude)
    end
    requires_exact || return total

    return setprecision(BigFloat, _LOCAL_MASS_FALLBACK_PRECISION) do
        exact = zero(Complex{BigFloat})
        @inbounds for position in first:last
            exact += Complex{BigFloat}(vals[order[position]])
        end
        return _local_mass_convert_bigfloat(
            T, exact, "LocalMassMatrix duplicate entry", first)
    end
end

function _canonical_local_mass_triplets(
        rows::Vector{Int}, cols::Vector{Int}, vals::Vector{T}) where {T}
    isempty(vals) && return rows, cols, vals, Int[]
    order = sortperm(eachindex(vals); by=index -> (rows[index], cols[index]))
    canonical_rows = Int[]
    canonical_cols = Int[]
    canonical_vals = T[]
    sizehint!(canonical_rows, length(vals))
    sizehint!(canonical_cols, length(vals))
    sizehint!(canonical_vals, length(vals))

    position = firstindex(order)
    @inbounds while position <= lastindex(order)
        first = position
        index = order[position]
        row = rows[index]
        col = cols[index]
        position += 1
        while position <= lastindex(order)
            candidate = order[position]
            rows[candidate] == row && cols[candidate] == col || break
            position += 1
        end
        push!(canonical_rows, row)
        push!(canonical_cols, col)
        push!(canonical_vals,
              _local_mass_sum_group(vals, order, first, position - 1))
    end
    col_order = sortperm(eachindex(canonical_vals); by=index ->
        (canonical_cols[index], canonical_rows[index]))
    return canonical_rows, canonical_cols, canonical_vals, col_order
end

@inline function _local_mass_convert_bigfloat(
        ::Type{T},
        value::Complex{BigFloat},
        label::AbstractString,
        index) where {T}
    converted = if T <: Real
        iszero(imag(value)) || throw(InexactError(:mul!, T, value))
        convert(T, real(value))
    else
        convert(T, value)
    end
    _local_mass_finite(converted) ||
        throw(OverflowError(
            "$label is outside the representable $T range at index $index"))
    return converted
end

@noinline function _local_mass_entry_bigfloat(
        M::LocalMassMatrix{T},
        i::Int,
        j::Int) where {T<:Number}
    return setprecision(BigFloat, _LOCAL_MASS_FALLBACK_PRECISION) do
        total = zero(Complex{BigFloat})
        @inbounds for k in eachindex(M.vals)
            if M.rows[k] == i && M.cols[k] == j
                total += Complex{BigFloat}(M.vals[k])
            end
        end
        return _local_mass_convert_bigfloat(
            T, total, "LocalMassMatrix entry", (i, j))
    end
end

function Base.getindex(M::LocalMassMatrix{T}, i::Int, j::Int) where {T<:Number}
    (1 <= i <= M.n && 1 <= j <= M.n) || throw(BoundsError(M, (i, j)))
    @inbounds for k in eachindex(M.vals)
        M.rows[k] == i && M.cols[k] == j && return M.vals[k]
    end
    return zero(T)
end

@noinline function _scale_local_mass_bigfloat(
        a::Number,
        M::LocalMassMatrix,
        ::Type{T}) where {T}
    return setprecision(BigFloat, _LOCAL_MASS_FALLBACK_PRECISION) do
        rows = Int[]
        cols = Int[]
        vals = T[]
        scale = Complex{BigFloat}(a)
        order = sortperm(eachindex(M.vals); by=k -> (M.rows[k], M.cols[k]))
        position = firstindex(order)
        @inbounds while position <= lastindex(order)
            k = order[position]
            row = M.rows[k]
            col = M.cols[k]
            total = zero(Complex{BigFloat})
            while position <= lastindex(order)
                j = order[position]
                M.rows[j] == row && M.cols[j] == col || break
                total += scale * Complex{BigFloat}(M.vals[j])
                position += 1
            end
            converted = _local_mass_convert_bigfloat(
                T, total, "scaled LocalMassMatrix entry", (row, col))
            push!(rows, row)
            push!(cols, col)
            push!(vals, converted)
        end
        return LocalMassMatrix(M.n, rows, cols, vals)
    end
end

function Base.:*(a::Number, M::LocalMassMatrix)
    result_type = Base.promote_op(*, typeof(a), eltype(M))
    if _local_mass_finite(a)
        real_type = _local_mass_ieee_real_type(result_type)
        scale_is_extreme = real_type !== nothing &&
                           _local_mass_extreme_factor(a, real_type)
        @inbounds for value in M.vals
            scaled = a * value
            if !_local_mass_finite(scaled) ||
               (!iszero(a) && !iszero(value) &&
                (iszero(scaled) ||
                 (real_type !== nothing &&
                  (scale_is_extreme ||
                   _local_mass_extreme_factor(value, real_type)))))
                return _scale_local_mass_bigfloat(a, M, result_type)
            end
        end
    end
    vals = [a * v for v in M.vals]
    return LocalMassMatrix(M.n, copy(M.rows), copy(M.cols), vals)
end

Base.:*(M::LocalMassMatrix, a::Number) = a * M
Base.:-(M::LocalMassMatrix) = -one(eltype(M)) * M

function Base.Array{T,2}(M::LocalMassMatrix) where {T}
    A = zeros(T, M.n, M.n)
    @inbounds for k in eachindex(M.vals)
        A[M.rows[k], M.cols[k]] = T(M.vals[k])
    end
    return A
end

Base.Array(M::LocalMassMatrix{T}) where {T<:Number} = Array{T,2}(M)
Base.Matrix(M::LocalMassMatrix{T}) where {T<:Number} = Array{T,2}(M)

function SparseArrays.sparse(M::LocalMassMatrix)
    return sparse(M.rows, M.cols, M.vals, M.n, M.n)
end

function _local_mass_mul_needs_bigfloat(
        y::AbstractVector,
        M::LocalMassMatrix,
        x::AbstractVector,
        alpha::Number,
        beta::Number,
        adjoint_operator::Bool)
    _local_mass_finite(alpha) && _local_mass_finite(beta) || return false
    real_type = _local_mass_ieee_real_type(eltype(y))
    alpha_is_extreme = real_type !== nothing &&
                       _local_mass_extreme_factor(alpha, real_type)
    beta_is_extreme = real_type !== nothing &&
                      _local_mass_extreme_factor(beta, real_type)
    magnitude_sum = 0.0
    scan_all_previous = !iszero(beta) && beta != one(beta)
    if scan_all_previous
        @inbounds for previous in y
            _local_mass_finite(previous) || return false
            if !iszero(previous) && real_type !== nothing &&
               (beta_is_extreme ||
                _local_mass_extreme_factor(previous, real_type))
                return true
            end
            term = beta * previous
            if !_local_mass_finite(term) ||
               (!iszero(previous) && iszero(term))
                return true
            end
            magnitude_sum += _local_mass_component_scale(term)
            isfinite(magnitude_sum) || return true
        end
    end
    @inbounds for k in eachindex(M.vals)
        target_index = adjoint_operator ? M.cols[k] : M.rows[k]
        if beta == one(beta)
            previous = y[target_index]
            _local_mass_finite(previous) || return false
            if !iszero(previous) && real_type !== nothing &&
               _local_mass_extreme_factor(previous, real_type)
                return true
            end
            magnitude_sum += _local_mass_component_scale(previous)
            isfinite(magnitude_sum) || return true
        end
        source_index = adjoint_operator ? M.rows[k] : M.cols[k]
        matrix_value = adjoint_operator ? conj(M.vals[k]) : M.vals[k]
        source_value = x[source_index]
        _local_mass_finite(source_value) || return false
        if !iszero(matrix_value) && !iszero(source_value) &&
           real_type !== nothing &&
           (alpha_is_extreme ||
            _local_mass_extreme_factor(matrix_value, real_type) ||
            _local_mass_extreme_factor(source_value, real_type))
            return true
        end
        term = alpha * matrix_value * source_value
        if !_local_mass_finite(term) ||
           (!iszero(alpha) && !iszero(matrix_value) &&
            !iszero(source_value) && iszero(term))
            return true
        end
        magnitude_sum += _local_mass_component_scale(term)
        isfinite(magnitude_sum) || return true
    end
    return false
end

function _local_mass_scale_needs_bigfloat(
        y::AbstractVector,
        beta::Number)
    _local_mass_finite(beta) || return false
    real_type = _local_mass_ieee_real_type(eltype(y))
    beta_is_extreme = real_type !== nothing &&
                      _local_mass_extreme_factor(beta, real_type)
    @inbounds for previous in y
        _local_mass_finite(previous) || return false
        iszero(previous) && continue
        if real_type !== nothing &&
           (beta_is_extreme ||
            _local_mass_extreme_factor(previous, real_type))
            return true
        end
        term = beta * previous
        (!_local_mass_finite(term) || iszero(term)) && return true
    end
    return false
end

@noinline function _local_mass_scale_bigfloat!(
        y::AbstractVector{T},
        beta::Number) where {T}
    return setprecision(BigFloat, _LOCAL_MASS_FALLBACK_PRECISION) do
        scale = Complex{BigFloat}(beta)
        @inbounds for index in eachindex(y)
            total = scale * Complex{BigFloat}(y[index])
            y[index] = _local_mass_convert_bigfloat(
                T, total, "LocalMassMatrix output scaling", index)
        end
        return y
    end
end

@noinline function _local_mass_mul_bigfloat!(
        y::AbstractVector{T},
        M::LocalMassMatrix,
        x::AbstractVector,
        alpha::Number,
        beta::Number,
        adjoint_operator::Bool) where {T}
    return setprecision(BigFloat, _LOCAL_MASS_FALLBACK_PRECISION) do
        alpha_big = Complex{BigFloat}(alpha)
        beta_big = Complex{BigFloat}(beta)
        order = adjoint_operator ? M.col_order : eachindex(M.vals)
        position = firstindex(order)
        if iszero(beta) || beta == one(beta)
            iszero(beta) && fill!(y, zero(T))
            @inbounds while position <= lastindex(order)
                k = order[position]
                row = adjoint_operator ? M.cols[k] : M.rows[k]
                total = iszero(beta) ? zero(Complex{BigFloat}) :
                        Complex{BigFloat}(y[row])
                while position <= lastindex(order)
                    j = order[position]
                    target_index = adjoint_operator ? M.cols[j] : M.rows[j]
                    target_index == row || break
                    source_index = adjoint_operator ? M.rows[j] : M.cols[j]
                    matrix_value = adjoint_operator ? conj(M.vals[j]) : M.vals[j]
                    total += alpha_big * Complex{BigFloat}(matrix_value) *
                             Complex{BigFloat}(x[source_index])
                    position += 1
                end
                y[row] = _local_mass_convert_bigfloat(
                    T, total, "LocalMassMatrix product", row)
            end
            return y
        end

        @inbounds for row in 1:M.n
            total = beta_big * Complex{BigFloat}(y[row])
            while position <= lastindex(order)
                k = order[position]
                target_index = adjoint_operator ? M.cols[k] : M.rows[k]
                target_index == row || break
                source_index = adjoint_operator ? M.rows[k] : M.cols[k]
                matrix_value = adjoint_operator ? conj(M.vals[k]) : M.vals[k]
                total += alpha_big * Complex{BigFloat}(matrix_value) *
                         Complex{BigFloat}(x[source_index])
                position += 1
            end
            y[row] = _local_mass_convert_bigfloat(
                T, total, "LocalMassMatrix product", row)
        end
        return y
    end
end

function LinearAlgebra.mul!(y::AbstractVector, M::LocalMassMatrix, x::AbstractVector,
                            alpha::Number, beta::Number)
    length(x) == M.n || throw(DimensionMismatch("x length $(length(x)) != $(M.n)"))
    length(y) == M.n || throw(DimensionMismatch("y length $(length(y)) != $(M.n)"))
    if iszero(alpha)
        if iszero(beta)
            fill!(y, zero(eltype(y)))
        elseif beta != one(beta)
            if _local_mass_scale_needs_bigfloat(y, beta)
                return _local_mass_scale_bigfloat!(y, beta)
            end
            y .*= beta
        end
        return y
    end
    xread = Base.mightalias(y, x) ? copy(x) : x
    if _local_mass_mul_needs_bigfloat(
            y, M, xread, alpha, beta, false)
        return _local_mass_mul_bigfloat!(
            y, M, xread, alpha, beta, false)
    end
    if iszero(beta)
        fill!(y, zero(eltype(y)))
    elseif beta != one(beta)
        y .*= beta
    end
    @inbounds for k in eachindex(M.vals)
        y[M.rows[k]] += alpha * M.vals[k] * xread[M.cols[k]]
    end
    return y
end

LinearAlgebra.mul!(y::AbstractVector, M::LocalMassMatrix, x::AbstractVector) =
    mul!(y, M, x, one(eltype(M)), zero(eltype(y)))

function LinearAlgebra.mul!(y::AbstractVector,
                            A::Adjoint{<:Any,<:LocalMassMatrix},
                            x::AbstractVector,
                            alpha::Number, beta::Number)
    M = parent(A)
    length(x) == M.n || throw(DimensionMismatch("x length $(length(x)) != $(M.n)"))
    length(y) == M.n || throw(DimensionMismatch("y length $(length(y)) != $(M.n)"))
    if iszero(alpha)
        if iszero(beta)
            fill!(y, zero(eltype(y)))
        elseif beta != one(beta)
            if _local_mass_scale_needs_bigfloat(y, beta)
                return _local_mass_scale_bigfloat!(y, beta)
            end
            y .*= beta
        end
        return y
    end
    xread = Base.mightalias(y, x) ? copy(x) : x
    if _local_mass_mul_needs_bigfloat(
            y, M, xread, alpha, beta, true)
        return _local_mass_mul_bigfloat!(
            y, M, xread, alpha, beta, true)
    end
    if iszero(beta)
        fill!(y, zero(eltype(y)))
    elseif beta != one(beta)
        y .*= beta
    end
    @inbounds for k in eachindex(M.vals)
        y[M.cols[k]] += alpha * conj(M.vals[k]) * xread[M.rows[k]]
    end
    return y
end

LinearAlgebra.mul!(y::AbstractVector, A::Adjoint{<:Any,<:LocalMassMatrix}, x::AbstractVector) =
    mul!(y, A, x, one(eltype(parent(A))), zero(eltype(y)))

function _add_scaled_matrix!(Y::AbstractMatrix, alpha::Number, A::AbstractMatrix)
    Y .+= alpha .* A
    return Y
end

function _local_mass_add_needs_bigfloat(
        Y::AbstractMatrix,
        alpha::Number,
        A::LocalMassMatrix)
    _local_mass_finite(alpha) || return false
    real_type = _local_mass_ieee_real_type(eltype(Y))
    alpha_is_extreme = real_type !== nothing &&
                       _local_mass_extreme_factor(alpha, real_type)
    magnitude_sum = 0.0
    @inbounds for k in eachindex(A.vals)
        current = Y[A.rows[k], A.cols[k]]
        _local_mass_finite(current) || return false
        if !iszero(A.vals[k]) && real_type !== nothing &&
           (alpha_is_extreme ||
            _local_mass_extreme_factor(A.vals[k], real_type) ||
            (!iszero(current) &&
             _local_mass_extreme_factor(current, real_type)))
            return true
        end
        term = alpha * A.vals[k]
        if !_local_mass_finite(term) ||
           (!iszero(alpha) && !iszero(A.vals[k]) && iszero(term))
            return true
        end
        magnitude_sum += _local_mass_component_scale(current)
        magnitude_sum += _local_mass_component_scale(term)
        isfinite(magnitude_sum) || return true
    end
    return false
end

@noinline function _add_scaled_local_mass_bigfloat!(
        Y::AbstractMatrix{T},
        alpha::Number,
        A::LocalMassMatrix) where {T}
    return setprecision(BigFloat, _LOCAL_MASS_FALLBACK_PRECISION) do
        scale = Complex{BigFloat}(alpha)
        @inbounds for k in eachindex(A.vals)
            row = A.rows[k]
            col = A.cols[k]
            total = Complex{BigFloat}(Y[row, col]) +
                    scale * Complex{BigFloat}(A.vals[k])
            Y[row, col] = _local_mass_convert_bigfloat(
                T, total, "scaled LocalMassMatrix accumulation", (row, col))
        end
        return Y
    end
end

function _add_scaled_matrix!(Y::AbstractMatrix, alpha::Number, A::LocalMassMatrix)
    iszero(alpha) && return Y
    if _local_mass_add_needs_bigfloat(Y, alpha, A)
        return _add_scaled_local_mass_bigfloat!(Y, alpha, A)
    end
    @inbounds for k in eachindex(A.vals)
        Y[A.rows[k], A.cols[k]] += alpha * A.vals[k]
    end
    return Y
end

function _dot_left_matrix_right(left::AbstractVector, A::AbstractMatrix, right::AbstractVector)
    return dot(left, A * right)
end

function _dot_left_matrix_right(left::AbstractVector, A::StridedMatrix,
                                right::AbstractVector)
    return dot(left, A, right)
end

function _dot_left_matrix_right(left::AbstractVector,
                                A::SparseArrays.AbstractSparseMatrixCSC,
                                right::AbstractVector)
    return dot(left, A, right)
end

function _dot_left_matrix_right(left::AbstractVector, A::LocalMassMatrix, right::AbstractVector)
    length(left) == A.n || throw(DimensionMismatch("left length $(length(left)) != $(A.n)"))
    length(right) == A.n || throw(DimensionMismatch("right length $(length(right)) != $(A.n)"))
    acc = zero(promote_type(eltype(left), eltype(A), eltype(right)))
    @inbounds for k in eachindex(A.vals)
        acc += conj(left[A.rows[k]]) * A.vals[k] * right[A.cols[k]]
    end
    return acc
end

"""
Patch partition: maps each triangle to a design patch.
"""
struct PatchPartition
    tri_patch::Vector{Int}      # length Nt: patch id for each triangle
    P::Int                      # number of patches

    function PatchPartition(tri_patch::Vector{Int}, P::Int)
        P >= 0 ||
            throw(ArgumentError(
                "PatchPartition patch count must be nonnegative, got $P"))
        @inbounds for t in eachindex(tri_patch)
            1 <= tri_patch[t] <= P ||
                throw(ArgumentError(
                    "PatchPartition triangle $t has patch ID $(tri_patch[t]), " *
                    "expected an ID in 1:$P"
                ))
        end
        return new(tri_patch, P)
    end
end

"""
Spherical far-field sampling grid.
"""
struct SphGrid
    rhat::Matrix{Float64}       # (3, NΩ) unit direction vectors
    theta::Vector{Float64}      # polar angles
    phi::Vector{Float64}        # azimuthal angles
    w::Vector{Float64}          # quadrature weights
end

"""
    ScatteringResult

Result from `solve_scattering`, containing the solution and metadata.
"""
struct ScatteringResult
    I_coeffs::Vector{ComplexF64}
    method::Symbol
    N::Int
    assembly_time_s::Float64
    solve_time_s::Float64
    preconditioner_time_s::Float64
    gmres_iters::Int
    gmres_residual::Float64
    mesh_report::NamedTuple
    warnings::Vector{String}
end
