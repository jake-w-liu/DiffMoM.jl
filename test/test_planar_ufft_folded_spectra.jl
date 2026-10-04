using DiffMoM,Test,LinearAlgebra

function _folded_result_matvec_bytes(result)
    x=view(result.currents,:,1);out=zeros(ComplexF64,length(x))
    for _ in 1:30;mul!(out,result.operator,x);end
    minimum(@allocated(mul!(out,result.operator,x)) for _ in 1:5)
end

@testset "FFT vectors reject owned workspace aliases before mutation" begin
    prob=_dense_workspace_fixture()
    for fold in (false,true)
        A=planar_ufft_operator(prob,8e9;mx=64,my=64,surface_zs=[.3+.2im,.5+.1im],
            via_sigma=5.8e7,volume_sigma=4e7,_fold_iterative=fold)
        x=ComplexF64[sin(p)+im*cos(p) for p in 1:A.n];out=similar(x)
        workspaces=fold ? (A.output,A.folded.fields,A.folded.spectra) :
            (A.output,A.source_te,A.source_tm,A.field_te,A.field_tm)
        for array in workspaces
            borrowed=view(vec(array),1:A.n);borrowed.=x
            saved=copy(borrowed)
            @test_throws ArgumentError mul!(out,A,borrowed)
            @test borrowed==saved
            @test_throws ArgumentError mul!(borrowed,A,x,2+.3im,-.2+.1im)
            @test borrowed==saved
        end
    end
    # A small sheet-only problem fits a vector inside the FFT lattice,
    # including a strided view rather than only an owning array.
    grid=CellGrid(.004,.003,4,3)
    stack=PlanarStackup([PlanarLayer(2.,1.,.0004),PlanarLayer(1.,1.,.0004)],
        TERM_GND,TERM_GND,grid.a,grid.b)
    sheet=sheet_level(1,4,3);sheet.mask[:,2].=true
    sheet.connect_west[2]=sheet.connect_east[2]=true
    line=build_planar_problem(stack,grid,[sheet],
        [PlanarPort(1,:west,2:2,50.),PlanarPort(1,:east,2:2,50.)])
    for fold in (false,true)
        A=planar_ufft_operator(line,5e9;mx=64,my=64,_fold_iterative=fold)
        borrowed=view(vec(A.lattice),1:2:2A.n);borrowed.=1+.2im
        saved=copy(A.lattice);x=ones(ComplexF64,A.n);out=similar(x)
        @test_throws ArgumentError mul!(out,A,borrowed)
        @test A.lattice==saved
        @test_throws ArgumentError mul!(borrowed,A,x)
        @test A.lattice==saved
    end
end

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
        for block in (1,7,512,typemax(Int))
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
            storage=zeros(ComplexF64,2length(x))
            strided=view(storage,1:2:length(storage));strided.=x
            mul!(strided,A,strided)
            @test strided≈Z*x rtol=2e-12
            matrix=DiffMoM._planar_fft_dense_fill!(similar(Z),A)
            @test matrix≈Z rtol=2e-12
            # Source-image grouping changes roundoff in this private
            # operator materialization. Public dense assembly still has
            # its original bit-identical summation-order regressions.
            @test matrix≈assemble_planar_z_ufft(prob,8e9;kw...,block) rtol=2e-12
            @test DiffMoM._subdivision_retained_payload(A.folded)==
                sizeof(A.folded.spectra)+sizeof(A.folded.fields)+sizeof(A.folded.images)
            @test size(A.folded.spectra,3)==length(A.families)^2
            @test DiffMoM._subdivision_retained_payload(A)<
                DiffMoM._subdivision_retained_payload(retained)
            out=similar(x);mul!(out,A,x)
            @test (@allocated mul!(out,A,x))==0
            @test_throws DimensionMismatch mul!(zeros(ComplexF64,length(x)+1),A,x)
        end
        # At low counts the retained representation is still smaller,
        # even after source images reduce each family pair to one plane.
        lowkw=merge(kw,(;mx=23,my=19))
        @test planar_ufft_operator(prob,8e9;lowkw...,block=typemax(Int)).folded===nothing
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

@testset "Source images preserve low and aliased modes at every wall half" begin
    for walls in (WALL_PEC,WALL_PMC)
        base=_dense_workspace_fixture(walls)
        for sh in base.sheets
            sh.connect_south.=true;sh.connect_north.=true
        end
        for vol in base.vols
            vol.connect_west.=true;vol.connect_east.=true
            vol.connect_south.=true;vol.connect_north.=true
        end
        prob=build_planar_problem(base.stack,base.grid,base.sheets,base.ports;
            vias=base.vias,vols=base.vols)
        for (mx,my) in ((3,2),(17,13),(65,63))
            kw=(;mx,my,surface_zs=[.3+.2im,.5+.1im],via_sigma=5.8e7,volume_sigma=4e7,
                sheet_coupling_zs=[.01+.01im .002;.002 .02+.01im])
            # Low counts normally choose retained kernels. Exercise the
            # production image folding directly as well, and compare it
            # to the independently assembled Galerkin modal equation.
            A=planar_ufft_operator(prob,8e9;kw...,_fold_iterative=false)
            mg=A.modes;nf=length(A.families);ne=size(A.source_te,2)
            images=DiffMoM._ufft_source_images(A.families,prob.grid)
            spectra=zeros(ComplexF64,size(A.lattice)...,nf*length(images))
            mlist=[m+1 for n in 0:my-1 for m in 0:mx-1]
            nlist=[n+1 for n in 0:my-1 for m in 0:mx-1]
            DiffMoM._planar_fft_fold_images_block!(spectra,A.families,images,ne,mg,
                A.k_te,A.k_tm,mlist,nlist,length(mlist))
            folded=DiffMoM._PlanarUFFTFoldedWorkspace(spectra,
                zeros(ComplexF64,length(A.lattice),nf),images)
            imageoperator=PlanarUFFTOperator(A.n,A.grid,A.modes,A.families,A.k_te,A.k_tm,
                A.source_te,A.source_tm,A.field_te,A.field_tm,A.lattice,A.forward,A.backward,
                A.local_loss,A.output,folded)
            Z=assemble_planar_z(prob.stack,prob.grid,prob.sheets,prob.basis,2pi*8e9;
                vias=prob.vias,vols=prob.vols,kw...)
            x=ComplexF64[sin(p)+im*cos(p) for p in 1:A.n]
            @test imageoperator*x≈Z*x rtol=2e-12
            @test DiffMoM._planar_ufft_diagonal(imageoperator)≈diag(Z) rtol=2e-12
            @test DiffMoM._planar_fft_dense_fill!(similar(Z),imageoperator)≈Z rtol=2e-12
        end
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
    # Check the actual reported preflight boundary, including the source
    # image descriptor payload, without duplicating its estimate formula.
    rejected=try
        planar_ufft_operator(prob,8e9;kw...,max_bytes=DiffMoM._subdivision_retained_payload(A)-1)
        nothing
    catch err
        err
    end
    @test rejected isa ArgumentError
    limit=parse(Int,only(match(r"requires (\d+) raw bytes",sprint(showerror,rejected)).captures))
    @test_throws ArgumentError planar_ufft_operator(prob,8e9;kw...,max_bytes=limit-1)
    limited=planar_ufft_operator(prob,8e9;kw...,max_bytes=limit)
    @test DiffMoM._subdivision_retained_payload(limited)<=limit
    @test limited.folded!==nothing
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
    @test _folded_result_matvec_bytes(result)==0
    legacy=PlanarUFFTResult(line,result.freq,result.omega,result.operator,result.currents,
        result.y,result.s,result.iterations,result.relative_residuals)
    @test legacy.s==result.s && legacy.z0==result.z0
    @test _folded_result_matvec_bytes(legacy)==0
    @test_throws ErrorException solve_planar_ufft(line,5e9;linekw...,rtol=1e-13,maxiter=1,memory=1)
end
