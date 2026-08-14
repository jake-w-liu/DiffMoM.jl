# Impedance.jl — Impedance sheet term assembly and exact dZ/dθ
#
# Z_imp[m,n](θ) = -Σ_p θ_p ∫_{Γ_p} f_m · f_n dS
# ∂Z_mn/∂θ_p   = -∫_{Γ_p} f_m · f_n dS  =  -M_p[m,n]

export assemble_Z_impedance, precompute_patch_mass, assemble_dZ_dtheta

const _DEFAULT_MAX_MASS_PRECOMPUTE_WORK_BYTES = 512 * 1024 * 1024
const _DEFAULT_MAX_MASS_PRECOMPUTE_TERMS = 200_000_000

function _mass_precompute_base_bytes(
        N::Int, Nt::Int, Nq::Int, collection_count::Int,
        patch_mode::Bool)
    total = BigInt(0)
    # Degree counts, quadrature cache, areas, triangle-to-basis storage, and
    # the quadrature rule itself. Outer Vector slots are charged as pointers.
    total += BigInt(sizeof(Int)) * Nt
    total += BigInt(sizeof(Ptr{Cvoid}) + sizeof(Float64)) * Nt
    total += BigInt(sizeof(Vec3)) * Nt * Nq
    total += BigInt(sizeof(Ptr{Cvoid})) * Nt
    total += BigInt(2 * sizeof(Int)) * N
    total += BigInt(sizeof(SVector{2,Float64}) + sizeof(Float64)) * Nq
    # Triangle mode retains one result slot per triangle. Patch mode retains
    # three builder-vector slots, one result slot, and an entry count per patch.
    total += BigInt(sizeof(Ptr{Cvoid})) * collection_count *
             (patch_mode ? 4 : 1)
    patch_mode && (total += BigInt(sizeof(Int)) * collection_count)
    total <= typemax(Int) ||
        throw(ArgumentError("local mass-matrix base-work estimate overflows Int"))
    return Int(total)
end

function _mass_precompute_profile(
        rwg::RWGData, Nt::Int, Nq::Int, ::Type{Tmass},
        tri_patch::Union{Nothing,Vector{Int}}, collection_count::Int,
        max_work_bytes::Integer=typemax(Int),
        max_terms::Integer=typemax(Int)) where {Tmass}
    patch_mode = tri_patch !== nothing
    base_bytes = _mass_precompute_base_bytes(
        rwg.nedges, Nt, Nq, collection_count, patch_mode)
    work_limit = _validated_resource_limit("max_work_bytes", max_work_bytes)
    base_bytes <= work_limit ||
        throw(ArgumentError(
            "local mass-matrix base workspace requires $base_bytes raw bytes, " *
            "exceeding max_work_bytes=$work_limit"))
    term_limit = _validated_resource_limit("max_terms", max_terms)
    degrees = zeros(Int, Nt)
    @inbounds for n in 1:rwg.nedges
        tp = rwg.tplus[n]
        tm = rwg.tminus[n]
        1 <= tp <= Nt ||
            throw(ArgumentError("rwg.tplus[$n]=$tp is outside 1:$Nt"))
        1 <= tm <= Nt ||
            throw(ArgumentError("rwg.tminus[$n]=$tm is outside 1:$Nt"))
        degrees[tp] = try
            Base.Checked.checked_add(degrees[tp], 1)
        catch err
            err isa OverflowError || rethrow()
            throw(ArgumentError("triangle $tp support degree overflows Int"))
        end
        degrees[tm] = try
            Base.Checked.checked_add(degrees[tm], 1)
        catch err
            err isa OverflowError || rethrow()
            throw(ArgumentError("triangle $tm support degree overflows Int"))
        end
    end

    patch_entries = patch_mode ? zeros(Int, collection_count) : Int[]
    entry_count = 0
    bytes_per_entry = 6 * sizeof(Int) + 2 * sizeof(Tmass)
    entry_capacity = (work_limit - base_bytes) ÷ bytes_per_entry
    term_capacity = term_limit ÷ Nq
    @inbounds for t in 1:Nt
        entries_t = try
            Base.Checked.checked_mul(degrees[t], degrees[t])
        catch err
            err isa OverflowError || rethrow()
            throw(ArgumentError(
                "triangle $t local mass-entry count overflows Int"))
        end
        entry_count = try
            Base.Checked.checked_add(entry_count, entries_t)
        catch err
            err isa OverflowError || rethrow()
            throw(ArgumentError("local mass-entry count overflows Int"))
        end
        entry_count <= entry_capacity ||
            throw(ArgumentError(
                "local mass-matrix output and workspace require more than " *
                "max_work_bytes=$work_limit raw bytes"))
        entry_count <= term_capacity ||
            throw(ArgumentError(
                "local mass-matrix assembly requires more than " *
                "max_terms=$term_limit terms"))
        if patch_mode
            p = tri_patch[t]
            patch_entries[p] = try
                Base.Checked.checked_add(patch_entries[p], entries_t)
            catch err
                err isa OverflowError || rethrow()
                throw(ArgumentError(
                    "patch $p local mass-entry count overflows Int"))
            end
        end
    end

    term_count = try
        Base.Checked.checked_mul(entry_count, Nq)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError("local mass-quadrature term count overflows Int"))
    end
    # The compact constructor temporarily retains input triplets, a sort order,
    # canonical triplets, and a column order. Charging the worst case prevents
    # constructor transients from escaping the workspace ceiling.
    work_bytes = Base.Checked.checked_add(
        base_bytes, Base.Checked.checked_mul(bytes_per_entry, entry_count))
    return (
        degrees=degrees,
        patch_entries=patch_entries,
        entry_count=entry_count,
        term_count=term_count,
        work_bytes=work_bytes,
    )
end

@inline function _validate_impedance_coefficients(theta::AbstractVector)
    @inbounds for p in eachindex(theta)
        isfinite(theta[p]) ||
            throw(ArgumentError(
                "impedance coefficient theta[$p] must be finite, got $(theta[p])"))
    end
    return nothing
end

function _validate_mass_matrix_sizes(
    Mp::AbstractVector{<:AbstractMatrix},
    expected_size::Union{Nothing,Tuple{Int,Int}}=nothing,
)
    if isempty(Mp)
        expected_size === nothing &&
            throw(ArgumentError(
                "mass-matrix collection must not be empty because the output size cannot be inferred."))
        expected_size[1] == expected_size[2] ||
            throw(DimensionMismatch(
                "impedance system matrix must be square, got size $expected_size"))
        return expected_size
    end

    target_size = expected_size === nothing ? size(first(Mp)) : expected_size
    target_size[1] == target_size[2] ||
        throw(DimensionMismatch(
            "impedance system matrix must be square, got size $target_size"))
    for p in eachindex(Mp)
        size(Mp[p]) == target_size ||
            throw(DimensionMismatch(
                "Mp[$p] has size $(size(Mp[p])), expected $target_size"))
    end
    return target_size
end

function _validate_impedance_inputs(
    Mp::AbstractVector{<:AbstractMatrix},
    theta::AbstractVector,
    expected_size::Union{Nothing,Tuple{Int,Int}}=nothing,
)
    length(Mp) == length(theta) ||
        throw(DimensionMismatch(
            "Mp length $(length(Mp)) must match theta length $(length(theta))"))
    _validate_impedance_coefficients(theta)
    return _validate_mass_matrix_sizes(Mp, expected_size)
end

"""
    precompute_patch_mass(mesh, rwg, partition;
                          quad_order=3,
                          max_work_bytes=536_870_912,
                          max_terms=200_000_000)

Precompute the patch mass matrices M_p[m,n] = ∫_{Γ_p} f_m · f_n dS
for each patch p = 1:P.
Returns a vector of compact local matrices.

`max_work_bytes` bounds the raw payload of the quadrature cache, support map,
triplet builders, compact results, and constructor transients. `max_terms`
bounds local basis-pair/quadrature evaluations. Both limits are checked before
the quadrature cache and triplet builders are allocated.
"""
function precompute_patch_mass(
        mesh::TriMesh, rwg::RWGData, partition::PatchPartition;
        quad_order::Int=3,
        max_work_bytes::Integer=_DEFAULT_MAX_MASS_PRECOMPUTE_WORK_BYTES,
        max_terms::Integer=_DEFAULT_MAX_MASS_PRECOMPUTE_TERMS)
    N = rwg.nedges
    Nt = ntriangles(mesh)
    P = partition.P
    length(partition.tri_patch) == Nt ||
        throw(DimensionMismatch(
            "partition.tri_patch length $(length(partition.tri_patch)) " *
            "does not match mesh triangle count $Nt"
        ))
    P >= 0 ||
        throw(ArgumentError("partition.P must be nonnegative, got $P"))
    @inbounds for t in 1:Nt
        1 <= partition.tri_patch[t] <= P ||
            throw(ArgumentError(
                "partition.tri_patch[$t]=$(partition.tri_patch[t]) is outside 1:$P"
            ))
    end
    Tcoef = promote_type(eltype(rwg.coeff_plus), eltype(rwg.coeff_minus))
    Tmass = Tcoef <: Real ? Float64 : ComplexF64

    xi, wq = tri_quad_rule(quad_order)
    Nq = length(wq)
    profile = _mass_precompute_profile(
        rwg, Nt, Nq, Tmass, partition.tri_patch, P,
        max_work_bytes, max_terms)

    # Precompute quad points
    quad_pts = [tri_quad_points(mesh, t, xi) for t in 1:Nt]
    areas    = [triangle_area(mesh, t) for t in 1:Nt]

    # For each triangle, find which basis functions have support there
    tri_to_basis = [Int[] for _ in 1:Nt]
    @inbounds for t in 1:Nt
        sizehint!(tri_to_basis[t], profile.degrees[t])
    end
    for n in 1:N
        push!(tri_to_basis[rwg.tplus[n]], n)
        push!(tri_to_basis[rwg.tminus[n]], n)
    end

    rows_p = [Int[] for _ in 1:P]
    cols_p = [Int[] for _ in 1:P]
    vals_p = [Tmass[] for _ in 1:P]
    @inbounds for p in 1:P
        sizehint!(rows_p[p], profile.patch_entries[p])
        sizehint!(cols_p[p], profile.patch_entries[p])
        sizehint!(vals_p[p], profile.patch_entries[p])
    end

    for t in 1:Nt
        p = partition.tri_patch[t]
        A = areas[t]
        basis_on_t = tri_to_basis[t]

        for bi in eachindex(basis_on_t)
            m = basis_on_t[bi]
            for bj in eachindex(basis_on_t)
                n = basis_on_t[bj]

                # Compute ∫_t f_m · f_n dS
                val = zero(Tmass)
                for q in 1:Nq
                    rq = quad_pts[t][q]
                    fm = eval_rwg(rwg, m, rq, t)
                    fn = eval_rwg(rwg, n, rq, t)
                    val += wq[q] * dot(fm, fn)
                end
                val *= 2 * A  # reference-to-physical scaling

                if val != zero(Tmass)
                    push!(rows_p[p], m)
                    push!(cols_p[p], n)
                    push!(vals_p[p], val)
                end
            end
        end
    end

    return [LocalMassMatrix(N, rows_p[p], cols_p[p], vals_p[p]) for p in 1:P]
end

"""
    assemble_Z_impedance(Mp, theta; max_output_bytes=2_000_000_000)

Assemble the impedance contribution to the MoM matrix:
Z_imp = -Σ_p θ_p M_p
"""
function assemble_Z_impedance(
        Mp::Vector{<:AbstractMatrix}, theta::AbstractVector;
        max_output_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    N = first(_validate_impedance_inputs(Mp, theta))
    CT = eltype(theta) <: Complex ? eltype(theta) : ComplexF64
    output_bytes = _checked_array_payload_bytes(
        CT, N, N; label="impedance matrix")
    _enforce_payload_limit(
        output_bytes, max_output_bytes,
        "impedance matrix", "max_output_bytes")
    Z_imp = zeros(CT, N, N)
    for p in eachindex(theta)
        _add_scaled_matrix!(Z_imp, -theta[p], Mp[p])
    end
    all(isfinite, Z_imp) ||
        error("impedance assembly produced non-finite matrix entries")
    return Z_imp
end

"""
    assemble_dZ_dtheta(Mp, p)

Return ∂Z/∂θ_p = -M_p (exact, closed-form derivative).
"""
function assemble_dZ_dtheta(Mp::Vector{<:AbstractMatrix}, p::Int)
    _validate_mass_matrix_sizes(Mp)
    checkbounds(Bool, Mp, p) ||
        throw(ArgumentError(
            "patch index p=$p is outside 1:$(length(Mp))"))
    _validate_known_matrix_entries(Mp[p], "Mp[$p]")
    derivative = -Mp[p]
    _validate_known_matrix_entries(
        derivative, "impedance derivative for patch $p")
    return derivative
end
