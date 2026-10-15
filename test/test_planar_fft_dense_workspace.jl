using DiffMoM, Test, LinearAlgebra

include(joinpath(@__DIR__, "helpers", "dense_fft_workspace.jl"))

@testset "retained FFT assembly omits iterative modal workspace" begin
    for walls in (WALL_PEC,WALL_PMC)
        prob=_dense_workspace_fixture(walls)
        kw=(;mx=23,my=19,surface_zs=[.3+.2im,.5+.1im],
            via_sigma=5.8e7,volume_sigma=4e7,
            sheet_coupling_zs=[.01+.01im .002;.002 .02+.01im])
        workspace=DiffMoM._planar_fft_workspace(prob,8e9,Val(true);kw...)
        iterative=planar_ufft_operator(prob,8e9;kw...)
        @test workspace.n==size(iterative,1)
        @test workspace.ne==size(iterative.source_te,2)==5
        @test workspace.k_te==iterative.k_te
        @test workspace.k_tm==iterative.k_tm
        @test workspace.local_loss==iterative.local_loss
        @test !hasproperty(workspace,:source_te)
        @test !hasproperty(workspace,:source_tm)
        @test !hasproperty(workspace,:field_te)
        @test !hasproperty(workspace,:field_tm)
        @test !hasproperty(workspace,:output)
        @test !hasproperty(workspace,:forward)
        matrix=assemble_planar_z_ufft(prob,8e9;kw...)
        legacy=_legacy_fft_dense_workspace_assembly(prob,8e9;kw...)
        @test matrix==legacy
        modal=assemble_planar_z(prob.stack,prob.grid,prob.sheets,
            prob.basis,2pi*8e9;vias=prob.vias,vols=prob.vols,kw...)
        @test norm(matrix-modal)/norm(modal)<2e-12
        @test matrix≈transpose(matrix) rtol=2e-12
        # Public iterative operation remains a complete operator.
        x=ComplexF64[sin(i)+im*cos(i) for i in 1:workspace.n]
        @test norm(iterative*x-modal*x)/norm(modal*x)<2e-12
        @test_throws DimensionMismatch DiffMoM._planar_fft_dense_fill!(
            zeros(ComplexF64,workspace.n+1,workspace.n),workspace)
    end

    prob=_dense_workspace_fixture()
    kw=(;mx=128,my=96,surface_zs=[.3+.2im,.5+.1im],
        via_sigma=5.8e7,volume_sigma=4e7)
    # This existing physical problem's former route needs over15MB of
    # warmed Julia allocations. The compact route fits11.5MB raw payload.
    limited=assemble_planar_z_ufft(prob,8e9;max_bytes=11_500_000,kw...)
    @test limited==_legacy_fft_dense_workspace_assembly(prob,8e9;kw...)
    @test_throws ArgumentError planar_ufft_operator(prob,8e9;
        max_bytes=11_500_000-sizeof(limited),kw...,_fold_iterative=false)
    @test planar_ufft_operator(prob,8e9;
        max_bytes=11_500_000-sizeof(limited),kw...).folded!==nothing
    @test_throws ArgumentError assemble_planar_z_ufft(prob,8e9;max_bytes=1,kw...)
    @test_throws ArgumentError assemble_planar_z_ufft(prob,8e9;mx=typemax(Int))
    @test_throws ArgumentError assemble_planar_z_ufft(prob,8e9;block=0)
    @test_throws ArgumentError assemble_planar_z_ufft(prob,Inf;kw...)
    @test_throws ArgumentError assemble_planar_z_ufft(prob,8e9;
        via_sigma=0.,kw[(:mx,:my)]...)
    # Compare the two warmed implementations in this process, including
    # their output matrix. This is cumulative allocation, not peak memory.
    assemble_planar_z_ufft(prob,8e9;kw...)
    _legacy_fft_dense_workspace_assembly(prob,8e9;kw...)
    compact_bytes=minimum(@allocated(assemble_planar_z_ufft(prob,8e9;kw...))
        for _ in 1:3)
    legacy_bytes=minimum(@allocated(_legacy_fft_dense_workspace_assembly(prob,8e9;kw...))
        for _ in 1:3)
    @test compact_bytes<=.85legacy_bytes
    # Small and oversized requested blocks must retain exactly the same
    # modal sum; their bounded integer-index payload is now reserved too.
    small=assemble_planar_z_ufft(prob,8e9;mx=23,my=19,block=1)
    @test small==assemble_planar_z_ufft(prob,8e9;mx=23,my=19,block=typemax(Int))
    @test small==assemble_planar_z_ufft(prob,8e9;mx=23,my=19,block=7)
end
