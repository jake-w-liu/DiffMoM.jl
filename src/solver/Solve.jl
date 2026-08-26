# Solve.jl — Forward and adjoint linear system solves

export solve_forward, solve_system, assemble_full_Z, assemble_full_Z!,
       make_mass_regularizer, make_left_preconditioner,
       select_preconditioner, transform_patch_matrices, prepare_conditioned_system

@inline function _validate_known_matrix_entries(
    matrix::AbstractMatrix,
    label::AbstractString,
)
    values = if matrix isa StridedMatrix
        matrix
    elseif matrix isa SparseMatrixCSC
        nonzeros(matrix)
    elseif matrix isa LocalMassMatrix
        matrix.vals
    else
        return nothing
    end
    all(isfinite, values) ||
        throw(ArgumentError("$label must contain only finite values"))
    return nothing
end

function _validate_linear_system_inputs(
    matrix::AbstractMatrix,
    rhs::AbstractVector,
    label::AbstractString,
)
    N = size(matrix, 1)
    N >= 1 ||
        throw(ArgumentError("$label matrix must be nonempty"))
    size(matrix, 2) == N ||
        throw(DimensionMismatch(
            "$label matrix must be square, got size $(size(matrix))"))
    length(rhs) == N ||
        throw(DimensionMismatch(
            "$label RHS length $(length(rhs)) must equal matrix size $N"))
    _validate_known_matrix_entries(matrix, label)
    all(isfinite, rhs) ||
        throw(ArgumentError("$label RHS must contain only finite values"))
    return N
end

@inline function _assert_finite_linear_vector(
    vector::AbstractVector,
    label::AbstractString,
)
    @inbounds for i in eachindex(vector)
        isfinite(vector[i]) ||
            error("$label contains a non-finite value at index $i: $(vector[i])")
    end
    return vector
end

@inline function _assert_finite_linear_array(
    array::AbstractArray,
    label::AbstractString,
)
    @inbounds for index in eachindex(array)
        isfinite(array[index]) ||
            error("$label contains a non-finite value at index $index: $(array[index])")
    end
    return array
end

@inline function _linear_array_is_finite(array::AbstractArray)
    @inbounds for value in array
        isfinite(value) || return false
    end
    return true
end

# A finite Float64 is an integer multiple of 2^-1074 with an integer
# coefficient of at most 2098 bits. A product therefore needs at most 4196
# bits, and summing both real products of every complex term across any
# Int-indexable vector adds fewer than 64 bits. This precision makes the
# exceptional dense product exact for Float32/Float64 inputs while retaining a
# guard margin. The ordinary BLAS/generic multiplication path is unchanged.
const _IEEE_DENSE_PRODUCT_FALLBACK_PRECISION = 4352

# With Float64 component exponents restricted to ±128, every exact real
# product is a multiple of at least 2^-360 and every addressable reduction is
# smaller than 2^322. For Float32, the corresponding ±16 bounds are 2^-78
# and 2^98. Thus a nonzero exact result cannot underflow and a finite result
# cannot overflow on the ordinary path. Inputs outside these bounds use the
# exact exceptional path so individually rounded-away terms can still combine
# into a representable result.
@inline _ieee_dense_safe_factor_exponent(::Type{Float64}) = 128
@inline _ieee_dense_safe_factor_exponent(::Type{Float32}) = 16

@inline function _ieee_dense_extreme_component(
        component::Real,
        ::Type{R}) where {R<:Union{Float32,Float64}}
    scale = abs(R(component))
    isfinite(scale) || return true
    iszero(scale) && return false
    safe_exponent = _ieee_dense_safe_factor_exponent(R)
    value_exponent = exponent(scale)
    return value_exponent < -safe_exponent ||
           value_exponent > safe_exponent
end

@inline function _ieee_dense_extreme_factor(
        value::Number,
        ::Type{R}) where {R<:Union{Float32,Float64}}
    # Inspect both components independently. A normal real component must not
    # hide a subnormal imaginary component whose individually rounded products
    # can combine into a representable result.
    return _ieee_dense_extreme_component(real(value), R) ||
           _ieee_dense_extreme_component(imag(value), R)
end

function _ieee_dense_product_requires_fallback(
        matrix::AbstractMatrix,
        vector::AbstractVector,
        ::Type{R}) where {R<:Union{Float32,Float64}}
    @inbounds for value in matrix
        _ieee_dense_extreme_factor(value, R) && return true
    end
    @inbounds for value in vector
        _ieee_dense_extreme_factor(value, R) && return true
    end
    return false
end

function _ieee_dense_product_requires_fallback(
        matrix::SparseArrays.AbstractSparseMatrixCSC,
        vector::AbstractVector,
        ::Type{R}) where {R<:Union{Float32,Float64}}
    @inbounds for value in nonzeros(matrix)
        _ieee_dense_extreme_factor(value, R) && return true
    end
    @inbounds for value in vector
        _ieee_dense_extreme_factor(value, R) && return true
    end
    return false
end

function _ieee_dense_product_requires_fallback(
        matrix::LocalMassMatrix,
        vector::AbstractVector,
        ::Type{R}) where {R<:Union{Float32,Float64}}
    @inbounds for value in matrix.vals
        _ieee_dense_extreme_factor(value, R) && return true
    end
    @inbounds for value in vector
        _ieee_dense_extreme_factor(value, R) && return true
    end
    return false
end

@inline function _ieee_product_error_factor(
        ::Type{R},
        terms_per_column::Int,
        column_count::Int,
) where {R<:Union{Float32,Float64}}
    column_count <= 0 && return zero(R)
    # This covers product rounding, the reduction, and implementation-level
    # regrouping/FMA differences.  Once the bound reaches one, the ordinary
    # reduction cannot certify any nonzero contributing row.
    return min(
        one(R), R(8 * terms_per_column) * eps(R) * R(column_count))
end

@inline function _ieee_product_component_is_suspicious(
        value::Real,
        magnitude::R,
        error_factor::R,
) where {R<:Union{Float32,Float64}}
    isfinite(value) || return true
    isfinite(magnitude) || return true
    iszero(magnitude) && return false
    return error_factor == one(R) ||
           abs(R(value)) <= error_factor * magnitude
end

@inline function _ieee_add_product_with_exactness(
        total::R,
        first::R,
        second::R,
        sign::R,
) where {R<:Union{Float32,Float64}}
    product = first * second
    product_is_exact = iszero(fma(first, second, -product))
    term = sign * product
    next_total = total + term
    # Knuth's TwoSum residual is exact when the finite addition does not
    # overflow.  The safe exponent band above guarantees that precondition.
    virtual_term = next_total - total
    virtual_total = next_total - virtual_term
    addition_error = (total - virtual_total) + (term - virtual_term)
    return next_total, product_is_exact && iszero(addition_error)
end

function _ieee_dense_row_exact_reduction(
        matrix::AbstractMatrix,
        vector::AbstractVector,
        row,
        ::Type{R},
        complex_result::Bool,
) where {R<:Union{Float32,Float64}}
    sequential_real = zero(R)
    sequential_imag = zero(R)
    real_operations_were_exact = true
    imag_operations_were_exact = true
    @inbounds for column in axes(matrix, 2)
        matrix_value = matrix[row, column]
        vector_value = vector[column]
        matrix_real = R(real(matrix_value))
        vector_real = R(real(vector_value))
        if complex_result
            matrix_imag = R(imag(matrix_value))
            vector_imag = R(imag(vector_value))
            sequential_real, first_real_exact =
                _ieee_add_product_with_exactness(
                    sequential_real, matrix_real, vector_real, one(R))
            sequential_real, second_real_exact =
                _ieee_add_product_with_exactness(
                    sequential_real, matrix_imag, vector_imag, -one(R))
            sequential_imag, first_imag_exact =
                _ieee_add_product_with_exactness(
                    sequential_imag, matrix_real, vector_imag, one(R))
            sequential_imag, second_imag_exact =
                _ieee_add_product_with_exactness(
                    sequential_imag, matrix_imag, vector_real, one(R))
            real_operations_were_exact &=
                first_real_exact && second_real_exact
            imag_operations_were_exact &=
                first_imag_exact && second_imag_exact
        else
            sequential_real, operation_was_exact =
                _ieee_add_product_with_exactness(
                    sequential_real, matrix_real, vector_real, one(R))
            real_operations_were_exact &= operation_was_exact
        end
    end
    return sequential_real, sequential_imag,
           real_operations_were_exact, imag_operations_were_exact
end

function _ieee_product_result_requires_fallback(
        matrix::AbstractMatrix,
        vector::AbstractVector,
        result::AbstractVector,
        ::Type{R}) where {R<:Union{Float32,Float64}}
    @inbounds for value in vector
        _ieee_dense_extreme_factor(value, R) && return true
    end
    complex_result = eltype(result) <: Complex
    terms_per_column = complex_result ? 2 : 1
    error_factor = _ieee_product_error_factor(
        R, terms_per_column, size(matrix, 2))
    @inbounds for row in axes(matrix, 1)
        real_magnitude = zero(R)
        imag_magnitude = zero(R)
        for column in axes(matrix, 2)
            matrix_value = matrix[row, column]
            _ieee_dense_extreme_factor(matrix_value, R) && return true
            vector_value = vector[column]
            matrix_real = R(real(matrix_value))
            vector_real = R(real(vector_value))
            if complex_result
                matrix_imag = R(imag(matrix_value))
                vector_imag = R(imag(vector_value))
                real_magnitude += abs(matrix_real * vector_real) +
                                  abs(matrix_imag * vector_imag)
                imag_magnitude += abs(matrix_real * vector_imag) +
                                  abs(matrix_imag * vector_real)
            else
                real_magnitude += abs(matrix_real * vector_real)
            end
        end
        result_value = result[row]
        real_is_suspicious = _ieee_product_component_is_suspicious(
            real(result_value), real_magnitude, error_factor)
        imag_is_suspicious = complex_result &&
            _ieee_product_component_is_suspicious(
                imag(result_value), imag_magnitude, error_factor)
        if real_is_suspicious || imag_is_suspicious
            sequential_real, sequential_imag,
            real_operations_were_exact, imag_operations_were_exact =
                _ieee_dense_row_exact_reduction(
                    matrix, vector, row, R, complex_result)
            # Exact products plus exact TwoSum-certified additions prove the
            # sequential value is the mathematical reduction.  This keeps
            # exact structural cancellation on the allocation-free path.
            real_is_suspicious &&
                !(real_operations_were_exact &&
                  R(real(result_value)) == sequential_real) && return true
            imag_is_suspicious &&
                !(imag_operations_were_exact &&
                  R(imag(result_value)) == sequential_imag) && return true
        end
    end
    return false
end

function _ieee_product_result_requires_fallback(
        matrix::SparseArrays.AbstractSparseMatrixCSC,
        vector::AbstractVector,
        result::AbstractVector,
        ::Type{R}) where {R<:Union{Float32,Float64}}
    @inbounds for value in vector
        _ieee_dense_extreme_factor(value, R) && return true
    end
    complex_result = eltype(result) <: Complex
    real_magnitudes = zeros(R, length(result))
    imag_magnitudes = complex_result ? zeros(R, length(result)) : real_magnitudes
    rows = rowvals(matrix)
    values = nonzeros(matrix)
    @inbounds for column in axes(matrix, 2)
        vector_value = vector[column]
        vector_real = R(real(vector_value))
        vector_imag = complex_result ? R(imag(vector_value)) : zero(R)
        for position in nzrange(matrix, column)
            matrix_value = values[position]
            _ieee_dense_extreme_factor(matrix_value, R) && return true
            row = rows[position]
            matrix_real = R(real(matrix_value))
            if complex_result
                matrix_imag = R(imag(matrix_value))
                real_magnitudes[row] += abs(matrix_real * vector_real) +
                                        abs(matrix_imag * vector_imag)
                imag_magnitudes[row] += abs(matrix_real * vector_imag) +
                                        abs(matrix_imag * vector_real)
            else
                real_magnitudes[row] += abs(matrix_real * vector_real)
            end
        end
    end
    terms_per_column = complex_result ? 2 : 1
    error_factor = _ieee_product_error_factor(
        R, terms_per_column, size(matrix, 2))
    @inbounds for row in eachindex(result)
        result_value = result[row]
        _ieee_product_component_is_suspicious(
            real(result_value), real_magnitudes[row], error_factor) &&
            return true
        complex_result &&
            _ieee_product_component_is_suspicious(
                imag(result_value), imag_magnitudes[row], error_factor) &&
            return true
    end
    return false
end

function _ieee_product_result_requires_fallback(
        matrix::LocalMassMatrix,
        vector::AbstractVector,
        result::AbstractVector,
        ::Type{R}) where {R<:Union{Float32,Float64}}
    @inbounds for value in vector
        _ieee_dense_extreme_factor(value, R) && return true
    end
    complex_result = eltype(result) <: Complex
    real_magnitudes = zeros(R, length(result))
    imag_magnitudes = complex_result ? zeros(R, length(result)) : real_magnitudes
    @inbounds for position in eachindex(matrix.vals)
        matrix_value = matrix.vals[position]
        _ieee_dense_extreme_factor(matrix_value, R) && return true
        vector_value = vector[matrix.cols[position]]
        row = matrix.rows[position]
        matrix_real = R(real(matrix_value))
        vector_real = R(real(vector_value))
        if complex_result
            matrix_imag = R(imag(matrix_value))
            vector_imag = R(imag(vector_value))
            real_magnitudes[row] += abs(matrix_real * vector_real) +
                                    abs(matrix_imag * vector_imag)
            imag_magnitudes[row] += abs(matrix_real * vector_imag) +
                                    abs(matrix_imag * vector_real)
        else
            real_magnitudes[row] += abs(matrix_real * vector_real)
        end
    end
    terms_per_column = complex_result ? 2 : 1
    error_factor = _ieee_product_error_factor(
        R, terms_per_column, size(matrix, 2))
    @inbounds for row in eachindex(result)
        result_value = result[row]
        _ieee_product_component_is_suspicious(
            real(result_value), real_magnitudes[row], error_factor) &&
            return true
        complex_result &&
            _ieee_product_component_is_suspicious(
                imag(result_value), imag_magnitudes[row], error_factor) &&
            return true
    end
    return false
end

@noinline function _matrix_vector_product_bigfloat!(
    result::AbstractVector{T},
    matrix::AbstractMatrix{<:Number},
    vector::AbstractVector{<:Number},
    label::AbstractString,
) where {T<:Number}
    return setprecision(
            BigFloat, _IEEE_DENSE_PRODUCT_FALLBACK_PRECISION) do
        if T <: Real
            @inbounds for row in axes(matrix, 1)
                total = zero(BigFloat)
                for column in axes(matrix, 2)
                    total += BigFloat(matrix[row, column]) *
                             BigFloat(vector[column])
                end
                converted = convert(T, total)
                isfinite(converted) ||
                    throw(OverflowError(
                        "$label is outside the representable $T range at index $row"))
                result[row] = converted
            end
        else
            @inbounds for row in axes(matrix, 1)
                total_real = zero(BigFloat)
                total_imag = zero(BigFloat)
                for column in axes(matrix, 2)
                    matrix_value = matrix[row, column]
                    vector_value = vector[column]
                    matrix_real = BigFloat(real(matrix_value))
                    matrix_imag = BigFloat(imag(matrix_value))
                    vector_real = BigFloat(real(vector_value))
                    vector_imag = BigFloat(imag(vector_value))
                    total_real += matrix_real * vector_real -
                                  matrix_imag * vector_imag
                    total_imag += matrix_real * vector_imag +
                                  matrix_imag * vector_real
                end
                converted = convert(
                    T, Complex{BigFloat}(total_real, total_imag))
                isfinite(converted) ||
                    throw(OverflowError(
                        "$label is outside the representable $T range at index $row"))
                result[row] = converted
            end
        end
        return result
    end
end

@noinline function _matrix_vector_product_bigfloat!(
    result::AbstractVector{T},
    matrix::SparseArrays.AbstractSparseMatrixCSC{<:Number},
    vector::AbstractVector{<:Number},
    label::AbstractString,
) where {T<:Number}
    # CSC storage is column-oriented.  The exceptional path transposes once
    # so each exact row reduction remains O(nnz) overall without retaining a
    # BigFloat accumulator for every output row.
    row_matrix = copy(transpose(matrix))
    columns = rowvals(row_matrix)
    values = nonzeros(row_matrix)
    return setprecision(
            BigFloat, _IEEE_DENSE_PRODUCT_FALLBACK_PRECISION) do
        if T <: Real
            @inbounds for row in axes(matrix, 1)
                total = zero(BigFloat)
                for position in nzrange(row_matrix, row)
                    column = columns[position]
                    total += BigFloat(values[position]) *
                             BigFloat(vector[column])
                end
                converted = convert(T, total)
                isfinite(converted) ||
                    throw(OverflowError(
                        "$label is outside the representable $T range at index $row"))
                result[row] = converted
            end
        else
            @inbounds for row in axes(matrix, 1)
                total_real = zero(BigFloat)
                total_imag = zero(BigFloat)
                for position in nzrange(row_matrix, row)
                    column = columns[position]
                    matrix_value = values[position]
                    vector_value = vector[column]
                    matrix_real = BigFloat(real(matrix_value))
                    matrix_imag = BigFloat(imag(matrix_value))
                    vector_real = BigFloat(real(vector_value))
                    vector_imag = BigFloat(imag(vector_value))
                    total_real += matrix_real * vector_real -
                                  matrix_imag * vector_imag
                    total_imag += matrix_real * vector_imag +
                                  matrix_imag * vector_real
                end
                converted = convert(
                    T, Complex{BigFloat}(total_real, total_imag))
                isfinite(converted) ||
                    throw(OverflowError(
                        "$label is outside the representable $T range at index $row"))
                result[row] = converted
            end
        end
        return result
    end
end

@noinline function _matrix_vector_product_bigfloat!(
    result::AbstractVector{T},
    matrix::LocalMassMatrix,
    vector::AbstractVector{<:Number},
    ::AbstractString,
) where {T<:Number}
    return _local_mass_mul_bigfloat!(
        result, matrix, vector, one(T), zero(T), false)
end

function _finite_matrix_vector_product(
    matrix::AbstractMatrix{<:Number},
    vector::AbstractVector{<:Number},
    label::AbstractString,
)
    result = matrix * vector
    scalar_type = eltype(result)
    if scalar_type <:
       Union{Float32,Float64,ComplexF32,ComplexF64}
        real_type = typeof(real(zero(scalar_type)))
        if _ieee_product_result_requires_fallback(
                matrix, vector, result, real_type)
            return _matrix_vector_product_bigfloat!(
                result, matrix, vector, label)
        end
    end
    @inbounds for index in eachindex(result)
        if !isfinite(result[index])
            scalar_type <:
                Union{Float32,Float64,ComplexF32,ComplexF64} ||
                return _assert_finite_linear_vector(result, label)
            return _matrix_vector_product_bigfloat!(
                result, matrix, vector, label)
        end
    end
    return result
end

function _finite_matrix_vector_product_status!(
    result::AbstractVector{T},
    matrix::AbstractMatrix{<:Number},
    vector::AbstractVector{<:Number},
    label::AbstractString,
) where {T<:Number}
    mul!(result, matrix, vector)
    if T <: Union{Float32,Float64,ComplexF32,ComplexF64}
        real_type = typeof(real(zero(T)))
        if _ieee_product_result_requires_fallback(
                matrix, vector, result, real_type)
            _matrix_vector_product_bigfloat!(
                result, matrix, vector, label)
            return true
        end
    end
    @inbounds for index in eachindex(result)
        if !isfinite(result[index])
            T <: Union{Float32,Float64,ComplexF32,ComplexF64} ||
                _assert_finite_linear_vector(result, label)
            _matrix_vector_product_bigfloat!(
                result, matrix, vector, label)
            return true
        end
    end
    return false
end

function _finite_matrix_vector_product!(
    result::AbstractVector{T},
    matrix::AbstractMatrix{<:Number},
    vector::AbstractVector{<:Number},
    label::AbstractString,
) where {T<:Number}
    _finite_matrix_vector_product_status!(
        result, matrix, vector, label)
    return result
end

struct _EquilibratedDenseLUPlan{T<:Number,F}
    factorization::F
    row_shifts::Vector{Int}
    column_shifts::Vector{Int}
end

@inline function _direct_component_exponent(component::Real, ::Type{R}) where {
        R<:Union{Float32,Float64}}
    converted = abs(R(component))
    iszero(converted) && return typemin(Int)
    return exponent(converted)
end

@inline function _direct_value_exponent(value::Number, ::Type{R}) where {
        R<:Union{Float32,Float64}}
    return max(
        _direct_component_exponent(real(value), R),
        _direct_component_exponent(imag(value), R),
    )
end

@inline function _direct_scaled_component(
        component::Real,
        shift::Int,
        ::Type{R},
        label::AbstractString) where {R<:Union{Float32,Float64}}
    converted = R(component)
    scaled = ldexp(converted, shift)
    isfinite(scaled) ||
        throw(OverflowError(
            "$label equilibration produced a non-finite component."))
    return scaled
end


@inline function _direct_scaled_value(
        value::Number,
        shift::Int,
        ::Type{T},
        label::AbstractString) where {T<:Union{Float32,Float64}}
    return T(_direct_scaled_component(
        real(value), shift, T, label))
end

@inline function _direct_scaled_value(
        value::Number,
        shift::Int,
        ::Type{Complex{R}},
        label::AbstractString) where {R<:Union{Float32,Float64}}
    return Complex{R}(
        _direct_scaled_component(real(value), shift, R, label),
        _direct_scaled_component(imag(value), shift, R, label),
    )
end

function _factor_equilibrated_ieee_matrix(
        matrix::AbstractMatrix{<:Number},
        ::Type{T},
        label::AbstractString) where {
        T<:Union{Float32,Float64,ComplexF32,ComplexF64}}
    N = size(matrix, 1)
    size(matrix, 2) == N ||
        throw(DimensionMismatch(
            "$label matrix must be square, got size $(size(matrix))"))
    real_type = typeof(real(zero(T)))
    row_shifts = Vector{Int}(undef, N)
    @inbounds for row in axes(matrix, 1)
        maximum_exponent = typemin(Int)
        for column in axes(matrix, 2)
            maximum_exponent = max(
                maximum_exponent,
                _direct_value_exponent(matrix[row, column], real_type),
            )
        end
        maximum_exponent == typemin(Int) &&
            throw(LinearAlgebra.SingularException(row))
        row_shifts[row] = -maximum_exponent
    end

    column_shifts = Vector{Int}(undef, N)
    @inbounds for column in axes(matrix, 2)
        maximum_exponent = typemin(Int)
        for row in axes(matrix, 1)
            entry_exponent =
                _direct_value_exponent(matrix[row, column], real_type)
            entry_exponent == typemin(Int) && continue
            maximum_exponent = max(
                maximum_exponent,
                entry_exponent + row_shifts[row],
            )
        end
        maximum_exponent == typemin(Int) &&
            throw(LinearAlgebra.SingularException(column))
        column_shifts[column] = -maximum_exponent
    end

    equilibrated = Matrix{T}(undef, N, N)
    @inbounds for column in axes(matrix, 2), row in axes(matrix, 1)
        equilibrated[row, column] = _direct_scaled_value(
            matrix[row, column],
            row_shifts[row] + column_shifts[column],
            T,
            label,
        )
    end
    factorization = lu!(equilibrated)
    issuccess(factorization) && all(isfinite, factorization.factors) ||
        throw(OverflowError(
            "$label equilibrated LU produced non-finite or unsuccessful factors"))
    return _EquilibratedDenseLUPlan{T,typeof(factorization)}(
        factorization, row_shifts, column_shifts)
end

function _direct_rhs_exponent(
        rhs::AbstractVector,
        row_shifts::Vector{Int},
        ::Type{R}) where {R<:Union{Float32,Float64}}
    maximum_exponent = typemin(Int)
    @inbounds for row in eachindex(rhs, row_shifts)
        value_exponent = _direct_value_exponent(rhs[row], R)
        value_exponent == typemin(Int) && continue
        maximum_exponent = max(
            maximum_exponent,
            value_exponent + row_shifts[row],
        )
    end
    return maximum_exponent == typemin(Int) ? 0 : maximum_exponent
end

function _solve_equilibrated_plan(
        plan::_EquilibratedDenseLUPlan{T},
        rhs::AbstractVector{<:Number},
        label::AbstractString) where {T<:Number}
    N = length(plan.row_shifts)
    length(rhs) == N ||
        throw(DimensionMismatch(
            "$label RHS length $(length(rhs)) must equal matrix size $N"))
    real_type = typeof(real(zero(T)))
    rhs_exponent = _direct_rhs_exponent(
        rhs, plan.row_shifts, real_type)
    solution = Vector{T}(undef, N)
    @inbounds for row in 1:N
        solution[row] = _direct_scaled_value(
            rhs[row],
            plan.row_shifts[row] - rhs_exponent,
            T,
            "$label RHS",
        )
    end
    ldiv!(plan.factorization, solution)
    @inbounds for column in 1:N
        solution[column] = _direct_scaled_value(
            solution[column],
            plan.column_shifts[column] + rhs_exponent,
            T,
            "$label solution",
        )
    end
    return _assert_finite_linear_vector(solution, label)
end

function _solve_equilibrated_plan(
        plan::_EquilibratedDenseLUPlan{T},
        rhs::AbstractMatrix{<:Number},
        label::AbstractString) where {T<:Number}
    N = length(plan.row_shifts)
    size(rhs, 1) == N ||
        throw(DimensionMismatch(
            "$label RHS has $(size(rhs, 1)) rows, expected $N"))
    real_type = typeof(real(zero(T)))
    rhs_exponents = Vector{Int}(undef, size(rhs, 2))
    solution = Matrix{T}(undef, size(rhs))
    @inbounds for rhs_column in axes(rhs, 2)
        rhs_vector = @view rhs[:, rhs_column]
        rhs_exponent = _direct_rhs_exponent(
            rhs_vector, plan.row_shifts, real_type)
        rhs_exponents[rhs_column] = rhs_exponent
        for row in 1:N
            solution[row, rhs_column] = _direct_scaled_value(
                rhs[row, rhs_column],
                plan.row_shifts[row] - rhs_exponent,
                T,
                "$label RHS",
            )
        end
    end
    ldiv!(plan.factorization, solution)
    @inbounds for rhs_column in axes(solution, 2), column in 1:N
        solution[column, rhs_column] = _direct_scaled_value(
            solution[column, rhs_column],
            plan.column_shifts[column] + rhs_exponents[rhs_column],
            T,
            "$label solution",
        )
    end
    return _assert_finite_linear_array(solution, label)
end

Base.size(plan::_EquilibratedDenseLUPlan) =
    size(plan.factorization)
Base.size(plan::_EquilibratedDenseLUPlan, dimension::Integer) =
    size(plan.factorization, dimension)
Base.:\(plan::_EquilibratedDenseLUPlan, rhs) =
    _solve_equilibrated_plan(plan, rhs, "equilibrated dense solve")
LinearAlgebra.issuccess(plan::_EquilibratedDenseLUPlan) =
    issuccess(plan.factorization)
function LinearAlgebra.ldiv!(plan::_EquilibratedDenseLUPlan, rhs)
    solution = _solve_equilibrated_plan(
        plan, rhs, "equilibrated dense solve")
    copyto!(rhs, solution)
    return rhs
end
function LinearAlgebra.adjoint(plan::_EquilibratedDenseLUPlan{T}) where {T}
    factorization = adjoint(plan.factorization)
    return _EquilibratedDenseLUPlan{T,typeof(factorization)}(
        factorization, plan.column_shifts, plan.row_shifts)
end

@inline function _direct_backward_error_limit(
        ::Type{R}, N::Int) where {R<:Union{Float32,Float64}}
    return min(0.1, max(64.0 * Float64(N) * eps(R), 64.0 * eps(R)))
end

function _direct_backward_error_fast(
        matrix::AbstractMatrix,
        solution::AbstractVector,
        rhs::AbstractVector,
        ::Type{T}) where {T<:Number}
    maximum_error = 0.0
    @inbounds for row in axes(matrix, 1)
        computed = zero(T)
        denominator = abs(T(rhs[row]))
        for column in axes(matrix, 2)
            matrix_value = T(matrix[row, column])
            solution_value = T(solution[column])
            computed += matrix_value * solution_value
            denominator += abs(matrix_value) * abs(solution_value)
        end
        residual = abs(computed - T(rhs[row]))
        row_error = iszero(denominator) ?
                    (iszero(residual) ? 0.0 : Inf) :
                    Float64(residual / denominator)
        maximum_error = max(maximum_error, row_error)
    end
    return maximum_error
end

@noinline function _direct_backward_error_bigfloat(
        matrix::AbstractMatrix,
        solution::AbstractVector,
        rhs::AbstractVector)
    return setprecision(BigFloat, _IEEE_DENSE_PRODUCT_FALLBACK_PRECISION) do
        maximum_error = zero(BigFloat)
        @inbounds for row in axes(matrix, 1)
            computed = zero(Complex{BigFloat})
            denominator = abs(Complex{BigFloat}(rhs[row]))
            for column in axes(matrix, 2)
                matrix_value = Complex{BigFloat}(matrix[row, column])
                solution_value = Complex{BigFloat}(solution[column])
                computed += matrix_value * solution_value
                denominator += abs(matrix_value) * abs(solution_value)
            end
            residual = abs(computed - Complex{BigFloat}(rhs[row]))
            row_error = iszero(denominator) ?
                        (iszero(residual) ? zero(BigFloat) : BigFloat(Inf)) :
                        residual / denominator
            maximum_error = max(maximum_error, row_error)
        end
        return Float64(maximum_error)
    end
end

function _direct_backward_error(
        matrix::AbstractMatrix,
        solution::AbstractVector,
        rhs::AbstractVector,
        ::Type{T}) where {
        T<:Union{Float32,Float64,ComplexF32,ComplexF64}}
    real_type = typeof(real(zero(T)))
    needs_fallback =
        _ieee_dense_product_requires_fallback(
            matrix, solution, real_type) ||
        any(value -> _ieee_dense_extreme_factor(value, real_type), rhs)
    return needs_fallback ?
           _direct_backward_error_bigfloat(matrix, solution, rhs) :
           _direct_backward_error_fast(matrix, solution, rhs, T)
end

function _direct_backward_error(
        matrix::AbstractMatrix,
        solution::AbstractMatrix,
        rhs::AbstractMatrix,
        ::Type{T}) where {
        T<:Union{Float32,Float64,ComplexF32,ComplexF64}}
    maximum_error = 0.0
    @inbounds for rhs_column in axes(rhs, 2)
        maximum_error = max(
            maximum_error,
            _direct_backward_error(
                matrix,
                @view(solution[:, rhs_column]),
                @view(rhs[:, rhs_column]),
                T,
            ),
        )
    end
    return maximum_error
end

@noinline function _direct_residual_bigfloat(
        matrix::AbstractMatrix,
        solution::AbstractVector,
        rhs::AbstractVector,
        ::Type{T},
        label::AbstractString) where {T<:Number}
    return setprecision(BigFloat, _IEEE_DENSE_PRODUCT_FALLBACK_PRECISION) do
        residual = Vector{T}(undef, length(rhs))
        @inbounds for row in axes(matrix, 1)
            if T <: Real
                total = zero(BigFloat)
                for column in axes(matrix, 2)
                    total += BigFloat(matrix[row, column]) *
                             BigFloat(solution[column])
                end
                residual[row] = T(BigFloat(rhs[row]) - total)
            else
                total = zero(Complex{BigFloat})
                for column in axes(matrix, 2)
                    total += Complex{BigFloat}(matrix[row, column]) *
                             Complex{BigFloat}(solution[column])
                end
                residual[row] = T(Complex{BigFloat}(rhs[row]) - total)
            end
            isfinite(residual[row]) ||
                throw(OverflowError(
                    "$label refinement residual is outside the " *
                    "representable $T range at row $row."))
        end
        return residual
    end
end

@noinline function _direct_residual_bigfloat(
        matrix::AbstractMatrix,
        solution::AbstractMatrix,
        rhs::AbstractMatrix,
        ::Type{T},
        label::AbstractString) where {T<:Number}
    residual = Matrix{T}(undef, size(rhs))
    @inbounds for rhs_column in axes(rhs, 2)
        residual[:, rhs_column] .= _direct_residual_bigfloat(
            matrix,
            @view(solution[:, rhs_column]),
            @view(rhs[:, rhs_column]),
            T,
            label,
        )
    end
    return residual
end

@noinline function _direct_add_correction!(
        solution::AbstractArray{T},
        correction::AbstractArray,
        label::AbstractString) where {T<:Number}
    return setprecision(BigFloat, _IEEE_DENSE_PRODUCT_FALLBACK_PRECISION) do
        @inbounds for index in eachindex(solution, correction)
            converted = if T <: Real
                T(BigFloat(solution[index]) + BigFloat(correction[index]))
            else
                T(Complex{BigFloat}(solution[index]) +
                  Complex{BigFloat}(correction[index]))
            end
            isfinite(converted) ||
                throw(OverflowError(
                    "$label refined solution is outside the " *
                    "representable $T range at index $index."))
            solution[index] = converted
        end
        return solution
    end
end

# A 4352-bit BigFloat carries roughly 0.5 KiB of significand storage. Bound
# the cold exact factorization to about one million retained matrix/RHS values
# so an adversarial ill-scaled dense solve fails clearly instead of requesting
# unbounded multi-gigabyte workspace.
const _MAX_DIRECT_BIGFLOAT_VALUES = 1_000_000

# MPFR stores 4352-bit significands in 544 bytes on 64-bit limbs. Reserve
# eight additional limbs for the BigFloat/MPFR/GC metadata associated with
# every distinct value. This is deliberately above Base.summarysize's measured
# 600 bytes per BigFloat at the configured precision.
const _DIRECT_BIGFLOAT_BYTES_PER_REAL =
    cld(_IEEE_DENSE_PRODUCT_FALLBACK_PRECISION, 8sizeof(UInt)) * sizeof(UInt) +
    8sizeof(UInt)

function _checked_exact_dense_solve_work_bytes(
        ::Type{T},
        system_size::Int,
        ieee_matrix_count::Int,
        integer_vector_count::Int,
        ieee_vector_count::Int,
        additional_payloads::Integer...;
        label::AbstractString) where {
            T<:Union{Float32,Float64,ComplexF32,ComplexF64}}
    system_size >= 0 ||
        throw(ArgumentError("$label system size must be nonnegative"))
    ieee_matrix_count >= 0 && integer_vector_count >= 0 &&
        ieee_vector_count >= 0 ||
        throw(ArgumentError("$label retained counts must be nonnegative"))
    component_count = T <: Real ? 1 : 2
    big_value_bytes =
        BigInt(component_count) * _DIRECT_BIGFLOAT_BYTES_PER_REAL
    count = BigInt(system_size)
    total = BigInt(ieee_matrix_count) * sizeof(T) * count^2
    total += big_value_bytes * count^2
    # Exact solve retains the converted RHS, a distinct BigFloat solution,
    # and the IEEE output while the exact factor remains live.
    total += 2big_value_bytes * count
    total += BigInt(ieee_vector_count) * sizeof(T) * count
    total += BigInt(integer_vector_count) * sizeof(Int) * count
    for payload in additional_payloads
        payload >= 0 ||
            throw(ArgumentError(
                "$label additional payloads must be nonnegative"))
        total += payload
    end
    total <= typemax(Int) ||
        throw(ArgumentError("$label exact raw-payload estimate overflows Int"))
    return Int(total)
end

@inline function _run_exact_fallback_check(exact_fallback_check)
    exact_fallback_check === nothing || exact_fallback_check()
    return nothing
end

struct _BigFloatDenseLUPlan{T<:Number,F}
    factorization::F
end

function _validate_bigfloat_plan_size(
        matrix_values::Int,
        rhs_values::Int,
        label::AbstractString)
    rhs_values <= _MAX_DIRECT_BIGFLOAT_VALUES &&
        matrix_values <= _MAX_DIRECT_BIGFLOAT_VALUES - rhs_values ||
        throw(ArgumentError(
            "$label exact fallback exceeds the " *
            "$_MAX_DIRECT_BIGFLOAT_VALUES-value resource limit."))
    return nothing
end


@noinline function _factor_bigfloat_ieee_matrix(
        matrix::AbstractMatrix{<:Number},
        ::Type{T},
        label::AbstractString) where {
        T<:Union{Float32,Float64,ComplexF32,ComplexF64}}
    _validate_bigfloat_plan_size(length(matrix), 0, label)
    return setprecision(
            BigFloat, _IEEE_DENSE_PRODUCT_FALLBACK_PRECISION) do
        big_type = T <: Real ? BigFloat : Complex{BigFloat}
        matrix_big = Matrix{big_type}(undef, size(matrix))
        @inbounds for index in eachindex(matrix_big, matrix)
            matrix_big[index] = big_type(matrix[index])
        end
        factorization = lu!(matrix_big)
        return _BigFloatDenseLUPlan{T,typeof(factorization)}(
            factorization)
    end
end

@noinline function _solve_bigfloat_plan(
        plan::_BigFloatDenseLUPlan{T},
        rhs::Union{AbstractVector{<:Number},AbstractMatrix{<:Number}},
        label::AbstractString) where {T<:Number}
    matrix_values = size(plan, 1) * size(plan, 2)
    _validate_bigfloat_plan_size(matrix_values, length(rhs), label)
    return setprecision(
            BigFloat, _IEEE_DENSE_PRODUCT_FALLBACK_PRECISION) do
        big_type = T <: Real ? BigFloat : Complex{BigFloat}
        rhs_big = Array{big_type}(undef, size(rhs))
        @inbounds for index in eachindex(rhs_big, rhs)
            rhs_big[index] = big_type(rhs[index])
        end
        solution_big = plan.factorization \ rhs_big
        solution = Array{T}(undef, size(rhs))
        @inbounds for index in eachindex(solution, solution_big)
            solution[index] = T(solution_big[index])
            isfinite(solution[index]) ||
                throw(OverflowError(
                    "$label exact solution is outside the representable " *
                    "$T range at index $index."))
        end
        return solution
    end
end

Base.size(plan::_BigFloatDenseLUPlan) = size(plan.factorization)
Base.size(plan::_BigFloatDenseLUPlan, dimension::Integer) =
    size(plan.factorization, dimension)

# A `_BigFloatDenseLUPlan` already represents the exact stored IEEE matrix at
# the bounded fallback precision. Rechecking its rounded Float64 solution with
# a Float64 backward error can reject the correctly rounded forward solution
# of a severely ill-conditioned system. Solve from the cached exact factors
# and convert only the completed solution.
function _solve_factored_linear_system(
    factorization::_BigFloatDenseLUPlan,
    matrix::AbstractMatrix{<:Number},
    rhs::Union{AbstractVector{<:Number},AbstractMatrix{<:Number}},
    label::AbstractString,
    ;
    exact_fallback_check=nothing,
)
    size(matrix) == size(factorization) ||
        throw(DimensionMismatch(
            "$label matrix has size $(size(matrix)), expected $(size(factorization))"))
    _run_exact_fallback_check(exact_fallback_check)
    return _solve_bigfloat_plan(factorization, rhs, label)
end

Base.:\(plan::_BigFloatDenseLUPlan, rhs) =
    _solve_bigfloat_plan(plan, rhs, "exact dense solve")
LinearAlgebra.issuccess(plan::_BigFloatDenseLUPlan) =
    issuccess(plan.factorization)
function LinearAlgebra.ldiv!(plan::_BigFloatDenseLUPlan, rhs)
    solution = _solve_bigfloat_plan(plan, rhs, "exact dense solve")
    copyto!(rhs, solution)
    return rhs
end
function LinearAlgebra.adjoint(plan::_BigFloatDenseLUPlan{T}) where {T}
    factorization = adjoint(plan.factorization)
    return _BigFloatDenseLUPlan{T,typeof(factorization)}(factorization)
end

@noinline function _solve_bigfloat_ieee_linear_system(
        matrix::AbstractMatrix{<:Number},
        rhs::Union{AbstractVector{<:Number},AbstractMatrix{<:Number}},
        ::Type{T},
        label::AbstractString) where {
        T<:Union{Float32,Float64,ComplexF32,ComplexF64}}
    plan = _factor_bigfloat_ieee_matrix(matrix, T, label)
    return _solve_bigfloat_plan(plan, rhs, label)
end

@noinline function _solve_scaled_ieee_linear_system(
    matrix::AbstractMatrix{<:Number},
    rhs::Union{AbstractVector{<:Number},AbstractMatrix{<:Number}},
    ::Type{T},
    label::AbstractString,
    ;
    exact_fallback_check=nothing,
) where {T<:Number}
    T <: Union{Float32,Float64,ComplexF32,ComplexF64} ||
        throw(ArgumentError(
            "$label equilibration requires an IEEE Float32/Float64 system."))
    real_type = typeof(real(zero(T)))
    limit = _direct_backward_error_limit(real_type, size(matrix, 1))
    plan = try
        _factor_equilibrated_ieee_matrix(matrix, T, label)
    catch err
        _recoverable_direct_solve_error(err) || rethrow()
        _run_exact_fallback_check(exact_fallback_check)
        exact_solution = _solve_bigfloat_ieee_linear_system(
            matrix, rhs, T, label)
        exact_backward_error = _direct_backward_error(
            matrix, exact_solution, rhs, T)
        isfinite(exact_backward_error) && exact_backward_error <= limit &&
            return exact_solution
        error("$label failed componentwise backward-error verification: " *
              "backward_error=$exact_backward_error, limit=$limit")
    end
    solution = _solve_equilibrated_plan(plan, rhs, label)
    for refinement in 0:2
        backward_error = _direct_backward_error(
            matrix, solution, rhs, T)
        isfinite(backward_error) && backward_error <= limit &&
            return solution
        if refinement == 2
            _run_exact_fallback_check(exact_fallback_check)
            exact_solution = _solve_bigfloat_ieee_linear_system(
                matrix, rhs, T, label)
            exact_backward_error = _direct_backward_error(
                matrix, exact_solution, rhs, T)
            isfinite(exact_backward_error) && exact_backward_error <= limit &&
                return exact_solution
            error("$label failed componentwise backward-error verification: " *
                  "backward_error=$exact_backward_error, limit=$limit")
        end
        residual = _direct_residual_bigfloat(
            matrix, solution, rhs, T, label)
        correction = _solve_equilibrated_plan(plan, residual, label)
        _direct_add_correction!(solution, correction, label)
    end
    error("$label internal refinement invariant failed")
end

@inline function _recoverable_direct_solve_error(err)
    return err isa LinearAlgebra.SingularException ||
           err isa LinearAlgebra.LAPACKException ||
           err isa LinearAlgebra.ZeroPivotException ||
           err isa OverflowError
end

function _factor_dense_linear_system(
        matrix::AbstractMatrix{<:Number},
        ::Type{T},
        label::AbstractString;
        exact_fallback_check=nothing) where {T<:Number}
    real_type = typeof(real(zero(T)))
    use_ieee_scaling = real_type <: Union{Float32,Float64} &&
                       T <: Union{Float32,Float64,ComplexF32,ComplexF64}
    raw_factor = try
        lu(matrix)
    catch err
        use_ieee_scaling && _recoverable_direct_solve_error(err) || rethrow()
        nothing
    end
    if raw_factor !== nothing && issuccess(raw_factor) &&
       all(isfinite, raw_factor.factors)
        return raw_factor
    end
    use_ieee_scaling ||
        error("$label LU produced non-finite or unsuccessful factors")
    raw_factor = nothing
    try
        return _factor_equilibrated_ieee_matrix(matrix, T, label)
    catch balanced_error
        _recoverable_direct_solve_error(balanced_error) || rethrow()
        _run_exact_fallback_check(exact_fallback_check)
        return _factor_bigfloat_ieee_matrix(matrix, T, label)
    end
end

@inline _direct_factorization_backend(factorization) = factorization

function _solve_factored_linear_system(
    factorization,
    matrix::AbstractMatrix{<:Number},
    rhs::Union{AbstractVector{<:Number},AbstractMatrix{<:Number}},
    label::AbstractString,
    ;
    exact_fallback_check=nothing,
)
    scalar_type = promote_type(eltype(matrix), eltype(rhs))
    real_type = typeof(real(zero(scalar_type)))
    use_ieee_scaling = real_type <: Union{Float32,Float64} &&
                       scalar_type <:
                           Union{Float32,Float64,ComplexF32,ComplexF64}

    solve_factorization = _direct_factorization_backend(factorization)
    solution = try
        solve_factorization \ rhs
    catch err
        use_ieee_scaling && _recoverable_direct_solve_error(err) || rethrow()
        return _solve_scaled_ieee_linear_system(
            matrix, rhs, scalar_type, label;
            exact_fallback_check=exact_fallback_check)
    end
    @inbounds for value in solution
        if !isfinite(value)
            use_ieee_scaling ||
                return _assert_finite_linear_array(solution, label)
            return _solve_scaled_ieee_linear_system(
                matrix, rhs, scalar_type, label;
                exact_fallback_check=exact_fallback_check)
        end
    end
    if use_ieee_scaling
        backward_error = _direct_backward_error(
            matrix, solution, rhs, scalar_type)
        limit = _direct_backward_error_limit(
            real_type, size(matrix, 1))
        if !isfinite(backward_error) || backward_error > limit
            return _solve_scaled_ieee_linear_system(
                matrix, rhs, scalar_type, label;
                exact_fallback_check=exact_fallback_check)
        end
    end
    return solution
end

function _solve_dense_linear_system(
        matrix::AbstractMatrix{<:Number},
        rhs::Union{AbstractVector{<:Number},AbstractMatrix{<:Number}},
        label::AbstractString)
    scalar_type = promote_type(eltype(matrix), eltype(rhs))
    factorization = _factor_dense_linear_system(
        matrix, scalar_type, label)
    return _solve_factored_linear_system(
        factorization, matrix, rhs, label)
end


# Preserve the physical conditioning matrix alongside package-created LU
# factors. Matrix(lu_factor) reconstructs the rounded L*U product rather than
# the input matrix and is therefore not a valid residual oracle in general.
struct _ConditioningFactorization{F,M<:AbstractMatrix}
    factorization::F
    matrix::M
end

Base.size(factor::_ConditioningFactorization) = size(factor.factorization)
Base.size(factor::_ConditioningFactorization, dimension::Integer) =
    size(factor.factorization, dimension)
Base.Matrix(factor::_ConditioningFactorization) = copy(factor.matrix)
Base.:\(factor::_ConditioningFactorization, rhs) =
    _solve_factored_linear_system(
        factor.factorization,
        factor.matrix,
        rhs,
        "conditioned factor solution",
    )
LinearAlgebra.issuccess(factor::_ConditioningFactorization) =
    issuccess(factor.factorization)
function LinearAlgebra.adjoint(factor::_ConditioningFactorization)
    adjoint_factorization = adjoint(factor.factorization)
    adjoint_matrix = adjoint(factor.matrix)
    return _ConditioningFactorization(
        adjoint_factorization, adjoint_matrix)
end
LinearAlgebra.ldiv!(factor::_ConditioningFactorization, rhs) =
    _solve_factored_linear_system!(
        rhs,
        factor.factorization,
        factor.matrix,
        rhs,
        "conditioned factor solution",
    )
@inline _direct_factorization_backend(
    factorization::_ConditioningFactorization) =
    factorization.factorization

function _matrix_for_verified_factor_solve(
        factorization,
        matrix::Union{Nothing,AbstractMatrix{<:Number}},
        label::AbstractString)
    matrix !== nothing && return matrix
    factorization isa _ConditioningFactorization &&
        return factorization.matrix
    # Matrix(factorization) reconstructs the rounded L*U product, not the
    # physical input matrix. It therefore cannot serve as a residual oracle,
    # even when every factor entry has an ordinary exponent.
    throw(ArgumentError(
        "$label requires preconditioner_M when using an externally " *
        "constructed preconditioner_factor."))
end

function _solve_factored_linear_system!(
    destination::Union{AbstractVector{T},AbstractMatrix{T}},
    factorization,
    matrix::Union{Nothing,AbstractMatrix{<:Number}},
    rhs::Union{AbstractVector{<:Number},AbstractMatrix{<:Number}},
    label::AbstractString,
) where {T<:Number}
    size(destination) == size(rhs) ||
        throw(DimensionMismatch(
            "$label destination has size $(size(destination)), expected $(size(rhs))"))
    fallback_matrix = _matrix_for_verified_factor_solve(
        factorization, matrix, label)
    matrix_reference = Base.mightalias(destination, fallback_matrix) ?
                       copy(fallback_matrix) : fallback_matrix
    # Keep the physical RHS available for verification and exceptional retry
    # when callers intentionally reuse its storage as the destination.
    rhs_reference = Base.mightalias(destination, rhs) ? copy(rhs) : rhs
    copyto!(destination, rhs_reference)

    solve_error = nothing
    solve_factorization = _direct_factorization_backend(factorization)
    if solve_factorization isa _BigFloatDenseLUPlan
        solution = _solve_bigfloat_plan(
            solve_factorization, rhs_reference, label)
        copyto!(destination, solution)
        return destination
    end
    solution = try
        _ldiv_reusing_input(solve_factorization, destination)
    catch err
        _recoverable_direct_solve_error(err) || rethrow()
        solve_error = err
        nothing
    end
    scalar_type = promote_type(
        eltype(matrix_reference), eltype(rhs_reference), T)
    real_type = typeof(real(zero(scalar_type)))
    use_ieee_scaling = real_type <: Union{Float32,Float64} &&
                       scalar_type <:
                           Union{Float32,Float64,ComplexF32,ComplexF64}
    if solution !== nothing && _linear_array_is_finite(solution)
        if !use_ieee_scaling
            return solution
        end
        backward_error = _direct_backward_error(
            matrix_reference, solution, rhs_reference, scalar_type)
        limit = _direct_backward_error_limit(
            real_type, size(matrix_reference, 1))
        isfinite(backward_error) && backward_error <= limit &&
            return solution
    end
    if !use_ieee_scaling
        solve_error === nothing || throw(solve_error)
        return _assert_finite_linear_array(solution, label)
    end
    fallback_solution = _solve_scaled_ieee_linear_system(
        matrix_reference, rhs_reference, scalar_type, label)
    copyto!(destination, fallback_solution)
    return destination
end


function _solve_factored_linear_system!(
    destination::Union{AbstractVector{T},AbstractMatrix{T}},
    factorization::_BigFloatDenseLUPlan,
    matrix::Union{Nothing,AbstractMatrix{<:Number}},
    rhs::Union{AbstractVector{<:Number},AbstractMatrix{<:Number}},
    label::AbstractString,
) where {T<:Number}
    size(destination) == size(rhs) ||
        throw(DimensionMismatch(
            "$label destination has size $(size(destination)), expected $(size(rhs))"))
    matrix_reference = _matrix_for_verified_factor_solve(
        factorization, matrix, label)
    size(matrix_reference) == size(factorization) ||
        throw(DimensionMismatch(
            "$label matrix has size $(size(matrix_reference)), expected $(size(factorization))"))
    # `_solve_bigfloat_plan` finishes reading `rhs` before `destination` is
    # touched, so an in-place caller does not require a second physical-RHS
    # copy on this cold exact path.
    solution = _solve_bigfloat_plan(factorization, rhs, label)
    copyto!(destination, solution)
    return destination
end

"""
    solve_forward(Z, v; solver=:direct, preconditioner=nothing, gmres_precond_side=:left, gmres_tol=1e-8, gmres_maxiter=200, gmres_memory=20, verbose_gmres=false, check_gmres_convergence=true, check_true_residual=true, true_residual_factor=100.0)

Solve Z I = v. Uses direct factorization by default, or GMRES when `solver=:gmres`.

# Arguments
- `solver`: `:direct` for LU factorization, `:gmres` for preconditioned GMRES
- `preconditioner`: a preconditioner object (e.g., `AbstractPreconditionerData`), or `nothing`
- `gmres_precond_side`: `:left` or `:right` preconditioner application side
- `gmres_tol`: relative tolerance for GMRES convergence
- `gmres_maxiter`: maximum GMRES iterations
- `gmres_memory`: Krylov restart/memory parameter
- `check_gmres_convergence`: throw an error if GMRES returns an unconverged solve
- `check_true_residual`: additionally verify `norm(Z*x-v)/norm(v)`
- `true_residual_factor`: allowed true-residual multiple of `gmres_tol`
"""
function solve_forward(Z::AbstractMatrix{<:Number}, v::AbstractVector{<:Number};
                       solver::Symbol=:direct,
                       preconditioner=nothing,
                       gmres_precond_side::Symbol=:left,
                       gmres_tol::Float64=1e-8,
                       gmres_maxiter::Int=200,
                       gmres_memory::Int=20,
                       verbose_gmres::Bool=false,
                       check_gmres_convergence::Bool=true,
                       check_true_residual::Bool=true,
                       true_residual_factor::Float64=100.0)
    solver in (:direct, :gmres) ||
        throw(ArgumentError(
            "Unknown solver: $solver (expected :direct or :gmres)"))
    if solver == :direct
        Z isa Matrix ||
            throw(ArgumentError(
                "Direct solver requires a dense Matrix; use solver=:gmres for operator-based systems."))
        _validate_linear_system_inputs(Z, v, "forward solve")
        return _solve_dense_linear_system(
            Z, v, "direct forward solution")
    else
        x, stats = solve_gmres(Z, v;
                                preconditioner=preconditioner,
                                precond_side=gmres_precond_side,
                                tol=gmres_tol, maxiter=gmres_maxiter,
                                memory=gmres_memory,
                                verbose=verbose_gmres,
                                check_gmres_convergence=check_gmres_convergence,
                                check_true_residual=check_true_residual,
                                true_residual_factor=true_residual_factor)
        return _assert_finite_linear_vector(x, "GMRES forward solution")
    end
end

"""
    solve_system(Z, rhs; solver=:direct, preconditioner=nothing, gmres_precond_side=:left, gmres_tol=1e-8, gmres_maxiter=200, gmres_memory=20, check_gmres_convergence=true, check_true_residual=true, true_residual_factor=100.0)

General linear solve Z x = rhs with solver dispatch.
"""
function solve_system(Z::AbstractMatrix{<:Number}, rhs::AbstractVector{<:Number};
                      solver::Symbol=:direct,
                      preconditioner=nothing,
                      gmres_precond_side::Symbol=:left,
                      gmres_tol::Float64=1e-8,
                      gmres_maxiter::Int=200,
                      gmres_memory::Int=20,
                      check_gmres_convergence::Bool=true,
                      check_true_residual::Bool=true,
                      true_residual_factor::Float64=100.0)
    return solve_forward(Z, rhs; solver=solver, preconditioner=preconditioner,
                          gmres_tol=gmres_tol, gmres_maxiter=gmres_maxiter,
                          gmres_precond_side=gmres_precond_side,
                          gmres_memory=gmres_memory,
                          check_gmres_convergence=check_gmres_convergence,
                          check_true_residual=check_true_residual,
                          true_residual_factor=true_residual_factor)
end

"""
    assemble_full_Z(Z_efie, Mp, theta; reactive=false,
                    max_output_bytes=2_000_000_000)

Assemble the full MoM matrix: Z(θ) = Z_efie + Z_imp(θ)

For resistive impedance (default):  Z_imp = -Σ_p θ_p M_p
For reactive impedance:             Z_imp = -Σ_p (iθ_p) M_p
"""
function assemble_full_Z(Z_efie::Matrix{<:Number},
                         Mp::Vector{<:AbstractMatrix},
                         theta::AbstractVector;
                         reactive::Bool=false,
                         max_output_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    output_bytes = _checked_array_payload_bytes(
        eltype(Z_efie), size(Z_efie)...;
        label="full impedance-loaded system matrix")
    _enforce_payload_limit(
        output_bytes, max_output_bytes,
        "full impedance-loaded system matrix", "max_output_bytes")
    Z = copy(Z_efie)
    assemble_full_Z!(Z, Z_efie, Mp, theta; reactive=reactive)
    return Z
end

"""
    assemble_full_Z!(Z, Z_efie, Mp, theta; reactive=false)

In-place variant: writes Z(θ) = Z_efie + Z_imp(θ) into pre-allocated `Z`.
"""
function assemble_full_Z!(Z::Matrix{<:Number},
                          Z_efie::Matrix{<:Number},
                          Mp::Vector{<:AbstractMatrix},
                          theta::AbstractVector;
                          reactive::Bool=false)
    size(Z) == size(Z_efie) ||
        throw(DimensionMismatch(
            "output Z has size $(size(Z)), expected $(size(Z_efie))"))
    _validate_impedance_inputs(Mp, theta, size(Z_efie))
    _accumulate_scaled_matrices!(
        Z, Z_efie, Mp,
        p -> -(reactive ? (1im * theta[p]) : theta[p]),
        "full impedance-loaded system")
    all(isfinite, Z) ||
        error("full impedance-loaded system contains non-finite entries")
    return Z
end

function _validated_mass_matrix_size(Mp::Vector{<:AbstractMatrix})
    N = first(_validate_mass_matrix_sizes(Mp))
    @inbounds for p in eachindex(Mp)
        _validate_known_matrix_entries(Mp[p], "patch mass matrix")
    end
    return N
end

function _validated_conditioning_matrix(
    matrix,
    N::Int,
    label::AbstractString,
)
    converted = try
        Matrix{ComplexF64}(matrix)
    catch err
        throw(ArgumentError(
            "$label must be convertible to a dense ComplexF64 matrix: $(sprint(showerror, err))"))
    end
    size(converted) == (N, N) ||
        throw(DimensionMismatch(
            "$label has size $(size(converted)), expected ($N, $N)"))
    all(isfinite, converted) ||
        throw(ArgumentError("$label must contain only finite values"))
    return converted
end

function _validate_conditioning_factor(
    factor,
    N::Int,
    label::AbstractString,
)
    size(factor) == (N, N) ||
        throw(DimensionMismatch(
            "$label has size $(size(factor)), expected ($N, $N)"))
    if applicable(issuccess, factor) && !issuccess(factor)
        throw(ArgumentError("$label is not a successful factorization"))
    end
    factors = if hasfield(typeof(factor), :factors)
        getfield(factor, :factors)
    elseif hasproperty(factor, :factors)
        getproperty(factor, :factors)
    else
        nothing
    end
    if factors !== nothing
        factors isa AbstractMatrix &&
            _validate_known_matrix_entries(factors, label)
    end
    return factor
end

@inline function _ldiv_reusing_input(factor, rhs)
    if applicable(ldiv!, factor, rhs)
        ldiv!(factor, rhs)
        return rhs
    end
    return factor \ rhs
end

"""
    make_mass_regularizer(Mp; max_output_bytes=2_000_000_000)

Build a Hermitian positive-semidefinite mass-based regularizer from patch
mass matrices:
  R = Σ_p M_p

Returns a dense `ComplexF64` matrix so it can be used directly in
regularized solves.
"""
function make_mass_regularizer(
        Mp::Vector{<:AbstractMatrix};
        max_output_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    N = _validated_mass_matrix_size(Mp)
    output_bytes = _checked_array_payload_bytes(
        ComplexF64, N, N; label="mass regularizer matrix")
    _enforce_payload_limit(
        output_bytes, max_output_bytes,
        "mass regularizer matrix", "max_output_bytes")
    R = zeros(ComplexF64, N, N)
    _accumulate_scaled_matrices!(
        R, nothing, Mp, _ -> one(ComplexF64),
        "mass regularizer accumulation")
    all(isfinite, R) ||
        error("mass regularizer accumulation produced non-finite values")

    # Enforce Hermitian symmetry in place, avoiding two extra N×N matrices.
    @inbounds for j in 1:N
        R[j, j] = complex(real(R[j, j]), 0.0)
        for i in 1:(j - 1)
            upper = R[i, j]
            lower = R[j, i]
            value = complex(
                _safe_midpoint_component(real(upper), real(lower)),
                _safe_midpoint_component(imag(upper), -imag(lower)),
            )
            R[i, j] = value
            R[j, i] = conj(value)
        end
    end
    all(isfinite, R) ||
        error("mass regularizer symmetrization produced non-finite values")
    return R
end

@noinline function _mass_diagonal_mean_bigfloat(R::Matrix{ComplexF64})
    return setprecision(BigFloat, _IEEE_DENSE_PRODUCT_FALLBACK_PRECISION) do
        total = zero(BigFloat)
        @inbounds for index in axes(R, 1)
            total += BigFloat(real(R[index, index]))
        end
        converted = Float64(total / BigFloat(size(R, 1)))
        isfinite(converted) ||
            throw(OverflowError(
                "mass preconditioner diagonal mean is outside the Float64 range"))
        return converted
    end
end

function _mass_diagonal_mean(R::Matrix{ComplexF64})
    diagonal_scale = 0.0
    @inbounds for index in axes(R, 1)
        diagonal_scale = max(diagonal_scale, abs(real(R[index, index])))
    end
    iszero(diagonal_scale) && return 0.0

    scaled_sum = 0.0
    compensation = 0.0
    absolute_sum = 0.0
    @inbounds for index in axes(R, 1)
        value = real(R[index, index]) / diagonal_scale
        corrected = value - compensation
        updated = scaled_sum + corrected
        compensation = (updated - scaled_sum) - corrected
        scaled_sum = updated
        absolute_sum += abs(value)
    end
    if !isfinite(scaled_sum) || !isfinite(absolute_sum) ||
       abs(scaled_sum) <=
           _SCALED_SUM_CANCELLATION_FACTOR * absolute_sum
        return _mass_diagonal_mean_bigfloat(R)
    end
    mean = diagonal_scale * (scaled_sum / size(R, 1))
    isfinite(mean) || return _mass_diagonal_mean_bigfloat(R)
    return mean
end

"""
    make_left_preconditioner(Mp; eps_rel=1e-8,
                             max_output_bytes=2_000_000_000)

Build a simple mass-based left preconditioner matrix:
  M = R + ϵ I,  R = Σ_p M_p

`eps_rel` scales the diagonal shift as
  ϵ = eps_rel * max(tr(R)/N, 1).
"""
function make_left_preconditioner(Mp::Vector{<:AbstractMatrix};
                                  eps_rel::Float64=1e-8,
                                  max_output_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    (isfinite(eps_rel) && eps_rel > 0.0) ||
        throw(ArgumentError(
            "eps_rel must be finite and positive, got $eps_rel"))
    R = make_mass_regularizer(Mp; max_output_bytes=max_output_bytes)
    N = size(R, 1)
    N >= 1 ||
        throw(ArgumentError(
            "mass preconditioner requires a positive matrix dimension"))
    scale = max(_mass_diagonal_mean(R), 1.0)
    isfinite(scale) ||
        error("mass preconditioner scale is non-finite")
    ϵ = eps_rel * scale
    isfinite(ϵ) && ϵ > 0.0 ||
        error("mass preconditioner diagonal shift is non-finite or nonpositive")
    @inbounds for i in 1:N
        R[i, i] += ϵ
    end
    all(isfinite, R) ||
        error("mass preconditioner contains non-finite values")
    return R
end

"""
    select_preconditioner(Mp;
                          mode=:off,
                          preconditioner_M=nothing,
                          n_threshold=256,
                          iterative_solver=false,
                          eps_rel=1e-6,
                          max_output_bytes=2_000_000_000)

Select the effective left preconditioner matrix used by the solver.

Modes:
- `:off`: disable preconditioning (unless `preconditioner_M` is provided).
- `:on`: always build/use a mass-based preconditioner.
- `:auto`: enable only when `iterative_solver=true` or `N >= n_threshold`.

If `preconditioner_M` is provided, it takes precedence over `mode`.

Returns `(M_eff, enabled, reason)` where:
- `M_eff` is either a dense `ComplexF64` matrix or `nothing`,
- `enabled` indicates whether preconditioning is active,
- `reason` is a short status string for logging/debugging.
"""
function select_preconditioner(Mp::Vector{<:AbstractMatrix};
                               mode::Symbol=:off,
                               preconditioner_M=nothing,
                               n_threshold::Int=256,
                               iterative_solver::Bool=false,
                               eps_rel::Float64=1e-6,
                               max_output_bytes::Integer=
                                   _DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    mode ∈ (:off, :on, :auto) ||
        throw(ArgumentError(
            "Invalid preconditioner mode: $mode (expected :off, :on, or :auto)"))
    n_threshold >= 0 ||
        throw(ArgumentError(
            "n_threshold must be nonnegative, got $n_threshold"))
    (isfinite(eps_rel) && eps_rel > 0.0) ||
        throw(ArgumentError(
            "eps_rel must be finite and positive, got $eps_rel"))
    N = _validated_mass_matrix_size(Mp)

    if preconditioner_M !== nothing
        matrix = _validated_conditioning_matrix(
            preconditioner_M, N, "preconditioner_M")
        return matrix, true, "user-provided preconditioner"
    end

    if mode == :off
        return nothing, false, "mode=:off"
    elseif mode == :on
        return make_left_preconditioner(
            Mp; eps_rel=eps_rel, max_output_bytes=max_output_bytes), true,
            "mode=:on"
    else
        if iterative_solver
            return make_left_preconditioner(
                Mp; eps_rel=eps_rel, max_output_bytes=max_output_bytes), true,
                "mode=:auto (iterative_solver=true)"
        elseif N >= n_threshold
            return make_left_preconditioner(
                Mp; eps_rel=eps_rel, max_output_bytes=max_output_bytes), true,
                "mode=:auto (N=$N >= $n_threshold)"
        else
            return nothing, false, "mode=:auto (N=$N < $n_threshold)"
        end
    end
end

"""
    transform_patch_matrices(Mp; preconditioner_M=nothing,
                             preconditioner_factor=nothing,
                             max_output_bytes=2_000_000_000)

Transform derivative blocks under left preconditioning:
  M_p_tilde = M^{-1} M_p

When `preconditioner_M === nothing`, returns `Mp` unchanged.
If `preconditioner_factor` is provided, it is reused instead of factorizing
`preconditioner_M`. A factor created by this API retains its physical matrix
and can be reused alone. Pair an externally constructed factor with its
`preconditioner_M` so residual verification uses the original matrix.

Returns `(Mp_tilde, factor)` where `factor` is `nothing` for the unpreconditioned
case. `max_output_bytes` bounds the combined raw payload of the returned dense
transformed matrices before allocation.
"""
function transform_patch_matrices(Mp::Vector{<:AbstractMatrix};
                                  preconditioner_M=nothing,
                                  preconditioner_factor=nothing,
                                  max_output_bytes::Integer=
                                      _DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    N = _validated_mass_matrix_size(Mp)
    if preconditioner_M === nothing && preconditioner_factor === nothing
        return Mp, nothing
    end

    matrix_bytes = _checked_array_payload_bytes(
        ComplexF64, N, N; label="transformed patch matrix")
    output_bytes = try
        Base.Checked.checked_mul(length(Mp), matrix_bytes)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError(
            "transformed patch-matrix payload estimate overflows Int"))
    end
    _enforce_payload_limit(
        output_bytes, max_output_bytes,
        "transformed patch matrices", "max_output_bytes")

    preconditioner_matrix =
        preconditioner_M === nothing ? nothing :
        _validated_conditioning_matrix(
            preconditioner_M, N, "preconditioner_M")
    fac = if preconditioner_factor === nothing
        _ConditioningFactorization(
            _factor_dense_linear_system(
                preconditioner_matrix,
                ComplexF64,
                "preconditioner factorization",
            ),
            preconditioner_matrix,
        )
    else
        _validate_conditioning_factor(
            preconditioner_factor, N, "preconditioner_factor")
    end

    Mp_tilde = Vector{Matrix{ComplexF64}}(undef, length(Mp))
    @inbounds for p in eachindex(Mp)
        transformed = Matrix{ComplexF64}(undef, N, N)
        Mp_tilde[p] = _solve_factored_linear_system!(
            transformed,
            fac,
            preconditioner_matrix,
            Mp[p],
            "transformed patch matrix $p",
        )
    end
    return Mp_tilde, fac
end

"""
    prepare_conditioned_system(Z_raw, rhs;
                               regularization_alpha=0.0,
                               regularization_R=nothing,
                               preconditioner_M=nothing,
                               preconditioner_factor=nothing)

Build the linear system used by forward/adjoint solves:
  Z_reg = Z_raw + αR
  Z_eff = M^{-1} Z_reg
  rhs_eff = M^{-1} rhs

If no regularization/preconditioning is requested, returns `(Z_raw, rhs, nothing)`.
Returns `(Z_eff, rhs_eff, factor)` where `factor` is the reusable verified
factorization used for preconditioning (or `nothing`). Package-created factors
retain their physical matrix and can be reused alone; externally constructed
factors must remain paired with `preconditioner_M`.
"""
function prepare_conditioned_system(Z_raw::Matrix{<:Number},
                                    rhs::Vector{<:Number};
                                    regularization_alpha::Float64=0.0,
                                    regularization_R=nothing,
                                    preconditioner_M=nothing,
                                    preconditioner_factor=nothing)
    N = _validate_linear_system_inputs(
        Z_raw, rhs, "conditioned system")
    (isfinite(regularization_alpha) && regularization_alpha >= 0.0) ||
        throw(ArgumentError(
            "regularization_alpha must be finite and nonnegative, got $regularization_alpha"))
    Z_eff = Matrix{ComplexF64}(Z_raw)
    rhs_eff = Vector{ComplexF64}(rhs)
    R = nothing

    if regularization_alpha > 0.0
        regularization_R === nothing &&
            throw(ArgumentError(
                "regularization_alpha is positive but regularization_R is nothing"))
        R = _validated_conditioning_matrix(
            regularization_R, N, "regularization_R")
        _add_scaled_matrix!(Z_eff, regularization_alpha, R)
        all(isfinite, Z_eff) ||
            error("regularized system matrix contains non-finite values")
    end

    if preconditioner_M === nothing && preconditioner_factor === nothing
        return Z_eff, rhs_eff, nothing
    end

    preconditioner_matrix =
        preconditioner_M === nothing ? nothing :
        _validated_conditioning_matrix(
            preconditioner_M, N, "preconditioner_M")
    fac = if preconditioner_factor === nothing
        preconditioner_matrix === nothing &&
            throw(ArgumentError(
                "preconditioner_M is required when preconditioner_factor is nothing"))
        _ConditioningFactorization(
            _factor_dense_linear_system(
                preconditioner_matrix,
                ComplexF64,
                "preconditioner factorization",
            ),
            preconditioner_matrix,
        )
    else
        _validate_conditioning_factor(
            preconditioner_factor, N, "preconditioner_factor")
    end
    Z_eff = _solve_factored_linear_system!(
        Z_eff,
        fac,
        preconditioner_matrix,
        Z_eff,
        "conditioned system matrix",
    )
    rhs_eff = _solve_factored_linear_system!(
        rhs_eff,
        fac,
        preconditioner_matrix,
        rhs,
        "conditioned RHS",
    )
    return Z_eff, rhs_eff, fac
end
