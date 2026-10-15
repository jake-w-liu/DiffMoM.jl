using Test
using LinearAlgebra
using SparseArrays
using Random
using DiffMoM

struct _CollectionNearfieldFixture <: AbstractMatrix{ComplexF64}
    octree::NamedTuple{(:perm,),Tuple{Vector{Int}}}
    Z_near::SparseMatrixCSC{ComplexF64,Int}
end
Base.size(fixture::_CollectionNearfieldFixture) = size(fixture.Z_near)

@testset "MLFMA preconditioner collection preserves headroom and factor ownership" begin
    policy = DiffMoM._nearfield_can_defer_collection
    @test policy(5240, 1024, 6264)
    @test !policy(5240, 1024, 6263)
    @test !policy(5240, 1024, 0)
    @test !policy(5240, 1024, 1023)
    @test policy(typemax(Int) - 100, 100, typemax(UInt64))

    n = 64
    near = spdiagm(-1 => fill(-0.25 + 0.04im, n - 1),
        0 => fill(3.0 + 0.2im, n), 1 => fill(-0.3 + 0.1im, n - 1))
    fixture = _CollectionNearfieldFixture((perm=randperm(Xoshiro(74), n),), near)
    rhs = randn(Xoshiro(75), ComplexF64, n)
    original = copy(near)
    for kind in (:ilu, :lu)
        preconditioner = redirect_stdout(devnull) do
            build_mlfma_preconditioner(fixture; factorization=kind, ilu_tau=0.0)
        end
        # Sparse LU and ILU own their factors. Collecting any discarded
        # reordered input must not change forward or adjoint application.
        GC.gc()
        @test NearFieldOperator(preconditioner) * rhs ≈ near \ rhs rtol=1e-12
        @test NearFieldAdjointOperator(preconditioner) * rhs ≈ adjoint(near) \ rhs rtol=1e-12
        @test near == original
        @test preconditioner.perm == fixture.octree.perm
        @test preconditioner.iperm[preconditioner.perm] == collect(1:n)

        patch = LocalMassMatrix(n, collect(1:n), collect(1:n), fill(0.1, n))
        loaded = near - 0.5sparse(patch)
        loaded_preconditioner = redirect_stdout(devnull) do
            build_mlfma_preconditioner(fixture, [patch], [0.5];
                factorization=kind, ilu_tau=0.0)
        end
        GC.gc()
        @test NearFieldOperator(loaded_preconditioner) * rhs ≈ loaded \ rhs rtol=1e-12
        @test NearFieldAdjointOperator(loaded_preconditioner) * rhs ≈ adjoint(loaded) \ rhs rtol=1e-12
        @test near == original
    end
end
