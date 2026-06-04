# Types.jl — Core data structures for DiffMoM

export Vec3, CVec3, TriMesh, RWGData, LocalMassMatrix, PatchPartition, SphGrid, ScatteringResult
export nvertices, ntriangles

const Vec3 = SVector{3,Float64}
const CVec3 = SVector{3,ComplexF64}

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
    function LocalMassMatrix{T}(n::Int,
                                rows::Vector{Int},
                                cols::Vector{Int},
                                vals::Vector{T}) where {T<:Number}
        n >= 0 || error("LocalMassMatrix size must be nonnegative, got $n")
        length(rows) == length(cols) == length(vals) ||
            throw(DimensionMismatch("LocalMassMatrix triplet arrays must have equal lengths."))
        return new{T}(n, rows, cols, vals)
    end
end

LocalMassMatrix(n::Int, rows::Vector{Int}, cols::Vector{Int}, vals::Vector{T}) where {T<:Number} =
    LocalMassMatrix{T}(n, rows, cols, vals)

Base.size(M::LocalMassMatrix) = (M.n, M.n)
Base.size(M::LocalMassMatrix, d::Int) = d <= 2 ? M.n : 1
Base.eltype(::Type{LocalMassMatrix{T}}) where {T<:Number} = T
Base.eltype(::LocalMassMatrix{T}) where {T<:Number} = T

function Base.getindex(M::LocalMassMatrix{T}, i::Int, j::Int) where {T<:Number}
    (1 <= i <= M.n && 1 <= j <= M.n) || throw(BoundsError(M, (i, j)))
    acc = zero(T)
    @inbounds for k in eachindex(M.vals)
        if M.rows[k] == i && M.cols[k] == j
            acc += M.vals[k]
        end
    end
    return acc
end

function Base.:*(a::Number, M::LocalMassMatrix)
    vals = [a * v for v in M.vals]
    return LocalMassMatrix(M.n, copy(M.rows), copy(M.cols), vals)
end

Base.:*(M::LocalMassMatrix, a::Number) = a * M
Base.:-(M::LocalMassMatrix) = -one(eltype(M)) * M

function Base.Array{T,2}(M::LocalMassMatrix) where {T}
    A = zeros(T, M.n, M.n)
    @inbounds for k in eachindex(M.vals)
        A[M.rows[k], M.cols[k]] += T(M.vals[k])
    end
    return A
end

Base.Array(M::LocalMassMatrix{T}) where {T<:Number} = Array{T,2}(M)
Base.Matrix(M::LocalMassMatrix{T}) where {T<:Number} = Array{T,2}(M)
SparseArrays.sparse(M::LocalMassMatrix) = sparse(M.rows, M.cols, M.vals, M.n, M.n)

function LinearAlgebra.mul!(y::AbstractVector, M::LocalMassMatrix, x::AbstractVector,
                            alpha::Number, beta::Number)
    length(x) == M.n || throw(DimensionMismatch("x length $(length(x)) != $(M.n)"))
    length(y) == M.n || throw(DimensionMismatch("y length $(length(y)) != $(M.n)"))
    if beta == 0
        fill!(y, zero(eltype(y)))
    else
        y .*= beta
    end
    @inbounds for k in eachindex(M.vals)
        y[M.rows[k]] += alpha * M.vals[k] * x[M.cols[k]]
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
    if beta == 0
        fill!(y, zero(eltype(y)))
    else
        y .*= beta
    end
    @inbounds for k in eachindex(M.vals)
        y[M.cols[k]] += alpha * conj(M.vals[k]) * x[M.rows[k]]
    end
    return y
end

LinearAlgebra.mul!(y::AbstractVector, A::Adjoint{<:Any,<:LocalMassMatrix}, x::AbstractVector) =
    mul!(y, A, x, one(eltype(parent(A))), zero(eltype(y)))

function _add_scaled_matrix!(Y::AbstractMatrix, alpha::Number, A::AbstractMatrix)
    Y .+= alpha .* A
    return Y
end

function _add_scaled_matrix!(Y::AbstractMatrix, alpha::Number, A::LocalMassMatrix)
    @inbounds for k in eachindex(A.vals)
        Y[A.rows[k], A.cols[k]] += alpha * A.vals[k]
    end
    return Y
end

function _dot_left_matrix_right(left::AbstractVector, A::AbstractMatrix, right::AbstractVector)
    return dot(left, A * right)
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
