using DiffMoM,Test,LinearAlgebra

@testset "Folded FFT operator preserves physical reactions and diagonal" begin
    for walls in (WALL_PEC,WALL_PMC)
        prob=_dense_workspace_fixture(walls)
        kw=(;mx=64,my=64,surface_zs=[.3+.2im,.5+.1im],via_sigma=5.8e7,
            volume_sigma=4e7,sheet_coupling_zs=[.01+.01im .002;.002 .02+.01im])
        Z=assemble_planar_z(prob.stack,prob.grid,prob.sheets,prob.basis,2pi*8e9;
            vias=prob.vias,vols=prob.vols,kw...)
        retained=planar_ufft_operator(prob,8e9;kw...,_fold_iterative=false)
        x=ComplexF64[sin(p)+im*cos(p) for p in 1:size(Z,1)]
        v=ComplexF64[cos(p/3)-im*sin(p/2) for p in 1:size(Z,1)]
        for block in (1,7,512)
            A=planar_ufft_operator(prob,8e9;kw...,block)
            @test A isa PlanarUFFTOperator && A.folded!==nothing
            @test isempty(A.k_te) && isempty(A.k_tm)
            @test all(isempty,(A.source_te,A.source_tm,A.field_te,A.field_tm))
            @test A*x≈Z*x rtol=2e-12
            @test A*x≈retained*x rtol=2e-12
            @test DiffMoM._planar_ufft_diagonal(A)≈diag(Z) rtol=2e-12
            @test sum(v.*(A*x))≈sum(x.*(A*v)) rtol=2e-12
            alias=copy(x);mul!(alias,A,alias)
            @test alias≈Z*x rtol=2e-12
            initial=copy(v);mul!(initial,A,x,2+.3im,-.2+.1im)
            @test initial≈(2+.3im).*(Z*x)+(-.2+.1im).*v rtol=2e-12
            initial.=NaN;mul!(initial,A,x,1.,0.)
            @test initial≈Z*x rtol=2e-12
            matrix=DiffMoM._planar_fft_dense_fill!(similar(Z),A)
            @test matrix≈Z rtol=2e-12
            @test matrix==assemble_planar_z_ufft(prob,8e9;kw...,block)
            @test DiffMoM._subdivision_retained_payload(A.folded)==
                sizeof(A.folded.spectra)+sizeof(A.folded.fields)
            @test DiffMoM._subdivision_retained_payload(A)<
                DiffMoM._subdivision_retained_payload(retained)
            out=similar(x);mul!(out,A,x)
            @test (@allocated mul!(out,A,x))==0
            @test_throws DimensionMismatch mul!(zeros(ComplexF64,length(x)+1),A,x)
        end
        # An oversized block can make folding more expensive; use the
        # smaller retained representation rather than exceed its storage.
        @test planar_ufft_operator(prob,8e9;kw...,block=typemax(Int)).folded===nothing
        # Include every wall half rooftop for both transverse directions,
        # on sheets and on the volume, rather than only the port walls.
        for sh in prob.sheets
            sh.connect_south.=true;sh.connect_north.=true
        end
        for vol in prob.vols
            vol.connect_west.=true;vol.connect_east.=true
            vol.connect_south.=true;vol.connect_north.=true
        end
        allwalls=build_planar_problem(prob.stack,prob.grid,prob.sheets,prob.ports;
            vias=prob.vias,vols=prob.vols)
        allkw=merge(kw,(;mx=128,my=96))
        allmatrix=assemble_planar_z(allwalls.stack,allwalls.grid,allwalls.sheets,
            allwalls.basis,2pi*8e9;vias=allwalls.vias,vols=allwalls.vols,allkw...)
        alloperator=planar_ufft_operator(allwalls,8e9;allkw...)
        @test alloperator.folded!==nothing
        allx=ComplexF64[sin(p)+im*cos(p) for p in 1:size(allmatrix,1)]
        @test alloperator*allx≈allmatrix*allx rtol=2e-12
        @test DiffMoM._planar_ufft_diagonal(alloperator)≈diag(allmatrix) rtol=2e-12
    end
end

@testset "Folded FFT resources and complete port solution" begin
    prob=_dense_workspace_fixture()
    kw=(;mx=128,my=96,surface_zs=[.3+.2im,.5+.1im],via_sigma=5.8e7,volume_sigma=4e7)
    construct()=planar_ufft_operator(prob,8e9;kw...)
    legacy()=planar_ufft_operator(prob,8e9;kw...,_fold_iterative=false)
    A=construct();legacy()
    @test planar_ufft_operator(prob,8e9;kw...,max_bytes=8_000_000).folded!==nothing
    @test_throws ArgumentError planar_ufft_operator(prob,8e9;kw...,
        max_bytes=8_000_000,_fold_iterative=false)
    @test_throws ArgumentError planar_ufft_operator(prob,8e9;kw...,max_bytes=1)
    @test_throws ArgumentError planar_ufft_operator(prob,8e9;kw...,block=0)
    @test_throws ArgumentError planar_ufft_operator(prob,8e9;mx=typemax(Int))
    @test_throws ArgumentError planar_ufft_operator(prob,0.;kw...)
    @test_throws ArgumentError planar_ufft_operator(prob,8e9;mx=128,my=96,via_sigma=0.)
    bytes=minimum(@allocated(construct()) for _ in 1:3)
    oldbytes=minimum(@allocated(legacy()) for _ in 1:3)
    @test bytes<.3oldbytes

    # Sheet-only stripline: independently assembled original equation,
    # unequal port references, physical currents and full complex S.
    grid=CellGrid(.004,.003,6,5)
    stack=PlanarStackup([PlanarLayer(2.0-.02im,1.,.0004),PlanarLayer(1.,1.,.0004)],
        TERM_GND,TERM_GND,grid.a,grid.b)
    sheet=sheet_level(1,6,5);rasterize_rect!(sheet,grid,0.,grid.a,.0006,.0024)
    rows=findall(sheet.mask[1,:]);sheet.connect_west[rows].=true;sheet.connect_east[rows].=true
    line=build_planar_problem(stack,grid,[sheet],[PlanarPort(1,:west,first(rows):last(rows),50.),
        PlanarPort(1,:east,first(rows):last(rows),75.)])
    linekw=(;mx=64,my=64,surface_zs=.1+.1im)
    reference=solve_planar(line,5e9;linekw...)
    result=solve_planar_ufft(line,5e9;linekw...,rtol=1e-10,memory=100)
    @test result.operator.folded!==nothing
    @test result.y≈reference.y rtol=1e-9
    @test result.s≈reference.s rtol=1e-9
    @test result.currents≈reference.currents rtol=1e-8
    @test maximum(result.relative_residuals)<1e-10
    @test_throws ErrorException solve_planar_ufft(line,5e9;linekw...,rtol=1e-13,maxiter=1,memory=1)
end
