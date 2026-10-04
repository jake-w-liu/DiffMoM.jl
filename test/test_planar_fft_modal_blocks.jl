using DiffMoM,Test,LinearAlgebra

# The adjacent dense-workspace suite defines the mixed sheet/via/volume fixture.
const _fft_modal_block_observations=NamedTuple[]
@testset "Bounded dense modal blocks preserve independent physical matrices" begin
    for walls in (WALL_PEC,WALL_PMC)
        prob=_dense_workspace_fixture(walls)
        kw=(;mx=64,my=64,surface_zs=[.3+.2im,.5+.1im],
            via_sigma=5.8e7,volume_sigma=4e7,
            sheet_coupling_zs=[.01+.01im .002;.002 .02+.01im])
        modal=assemble_planar_z(prob.stack,prob.grid,prob.sheets,
            prob.basis,2pi*8e9;vias=prob.vias,vols=prob.vols,kw...)
        retained=DiffMoM._planar_fft_workspace(prob,8e9,Val(true);kw...)
        expected=DiffMoM._planar_fft_dense_fill!(Matrix{ComplexF64}(undef,retained.n,retained.n),retained)
        kinds=copy(prob.basis.kind);indices=copy(prob.basis.ei)
        x=ComplexF64[sin(i)+im*cos(i) for i in 1:retained.n]
        for block in (1,7,512,typemax(Int))
            workspace=DiffMoM._planar_fft_workspace(prob,8e9,Val(true);kw...,_fold_dense=true,block)
            @test hasproperty(workspace,:spectra)==(block!=typemax(Int))
            if hasproperty(workspace,:spectra)
                @test size(workspace.spectra,3)==4length(workspace.families)^2
                @test !hasproperty(workspace,:k_te) && !hasproperty(workspace,:k_tm)
            end
            matrix=assemble_planar_z_ufft(prob,8e9;kw...,block)
            @test matrix==expected
            @test norm(matrix-modal)/norm(modal)<2e-12
            @test matrix*x≈modal*x rtol=2e-12
            @test matrix≈transpose(matrix) rtol=2e-12
            @test_throws DimensionMismatch DiffMoM._planar_fft_dense_fill!(
                zeros(ComplexF64,retained.n+1,retained.n),workspace)
        end
        @test prob.basis.kind==kinds && prob.basis.ei==indices
    end
end
@testset "Dense modal block resources and unchanged rejection boundaries" begin
    prob=_dense_workspace_fixture()
    kw=(;mx=128,my=96,surface_zs=[.3+.2im,.5+.1im],via_sigma=5.8e7,volume_sigma=4e7)
    render()=assemble_planar_z_ufft(prob,8e9;kw...)
    matrix=render()
    @test assemble_planar_z_ufft(prob,8e9;max_bytes=8_000_000,kw...)==matrix
    @test_throws ArgumentError DiffMoM._planar_fft_workspace(prob,8e9,Val(true);
        max_bytes=8_000_000-sizeof(matrix),kw...)
    @test_throws ArgumentError assemble_planar_z_ufft(prob,8e9;max_bytes=sizeof(matrix)-1,kw...)
    @test_throws ArgumentError assemble_planar_z_ufft(prob,8e9;block=0,kw...)
    @test_throws ArgumentError assemble_planar_z_ufft(prob,8e9;volume_sigma=0.,kw[(:mx,:my)]...)
    @test_throws ArgumentError assemble_planar_z_ufft(prob,8e9;via_sigma=0.,kw[(:mx,:my)]...)
    @test_throws ArgumentError assemble_planar_z_ufft(prob,0.;kw...)
    @test_throws ArgumentError assemble_planar_z_ufft(prob,8e9;mx=typemax(Int))
    legacy()=_legacy_fft_dense_workspace_assembly(prob,8e9;kw...)
    render();legacy()
    bytes=minimum(@allocated(render()) for _ in 1:3)
    oldbytes=minimum(@allocated(legacy()) for _ in 1:3)
    @test bytes<.5oldbytes
    push!(_fft_modal_block_observations,(cumulative_gc_bytes=bytes,legacy_operator_gc_bytes=oldbytes,
        output_matrix_bytes=sizeof(matrix),scope="warmed allocations including matrix, not peak memory"))
end
