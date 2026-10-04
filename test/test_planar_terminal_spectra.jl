using DiffMoM, Test, LinearAlgebra, Random

@testset "planar FFT: translated terminal half ramps" begin
    G=DiffMoM
    for walls in (WALL_PEC,WALL_PMC)
        grid=CellGrid(6e-3,4e-3,5,4;walls)
        stack=PlanarStackup([PlanarLayer(2-.02im,1.,.5e-3),
            PlanarLayer(1.,1.,.5e-3)],TERM_GND,TERM_GND,grid.a,grid.b)
        sh=sheet_level(1,grid.nx,grid.ny)
        sh.mask[2:4,2:3].=true
        ports=[PlanarPort(1,:terminal_x,1,2:3,50.;metal_side=:positive),
            PlanarPort(1,:terminal_x,4,2:3,50.;metal_side=:negative),
            PlanarPort(1,:terminal_y,1,2:4,50.;metal_side=:positive),
            PlanarPort(1,:terminal_y,3,2:4,50.;metal_side=:negative)]
        prob=build_planar_problem(stack,grid,[sh],ports)
        basis=prob.basis;nb=planar_basis_count(basis)
        modes=planar_mode_grid(grid,33,29)
        # Independent composite Simpson quadrature on the translated
        # physical support, including the mixed sin/cos transform term.
        for p in eachindex(basis.kind)
            k=basis.kind[p]
            k in (G._BASIS_X_LO,G._BASIS_X_HI,G._BASIS_Y_LO,G._BASIS_Y_HI) || continue
            xd=G._is_xdir(k)
            positive=k in (G._BASIS_X_LO,G._BASIS_Y_LO)
            d=xd ? grid.dx : grid.dy
            edge=xd ? basis.ei[p] : basis.ej[p]
            count=xd ? modes.mx : modes.my
            actual=zeros(count)
            xd ? G._basis_fx!(actual,basis,p,modes,grid) : G._basis_fy!(actual,basis,p,modes,grid)
            for m in 0:count-1
                wave=(xd ? modes.kx : modes.ky)[m+1]
                intervals=2048
                integral=0.0
                for j in 0:intervals
                    u=j/intervals
                    x=(edge+(positive ? u : -u))*d
                    shape=1-u
                    trig=walls==WALL_PEC ? cos(wave*x) : sin(wave*x)
                    weight=j in (0,intervals) ? 1 : isodd(j) ? 4 : 2
                    integral+=weight*shape*trig
                end
                integral*=d/(3intervals)
                @test actual[m+1]≈integral atol=1e-12 rtol=1e-8
            end
        end
        spatial=ComplexF64[.1*(i+j)+.02im for i in 1:grid.nx,j in 1:grid.ny]
        loss=zeros(ComplexF64,nb,nb)
        G._add_gram!(loss,basis,grid,[spatial])
        function shape(p,x,y)
            xd=G._is_xdir(basis.kind[p]);k=basis.kind[p]
            trans=xd ? basis.ej[p] : basis.ei[p]
            cell=xd ? floor(Int,y/grid.dy)+1 : floor(Int,x/grid.dx)+1
            cell==trans || return 0.0
            u=xd ? x/grid.dx-basis.ei[p] : y/grid.dy-basis.ej[p]
            k in (G._BASIS_X_LO,G._BASIS_Y_LO) && return 0<u<1 ? 1-u : 0.0
            k in (G._BASIS_X_HI,G._BASIS_Y_HI) && return -1<u<0 ? 1+u : 0.0
            return max(1-abs(u),0.0)
        end
        quad=zeros(ComplexF64,nb,nb)
        for p in 1:nb,q in 1:nb
            G._is_xdir(basis.kind[p])==G._is_xdir(basis.kind[q]) || continue
            for j in 1:grid.ny,i in 1:grid.nx,sx in (-1,1),sy in (-1,1)
                x=(i-.5+sx/(2sqrt(3)))*grid.dx
                y=(j-.5+sy/(2sqrt(3)))*grid.dy
                quad[p,q]-=spatial[i,j]*shape(p,x,y)*shape(q,x,y)*grid.dx*grid.dy/4
            end
        end
        @test loss≈quad rtol=1e-13 atol=1e-20
        kw=(;mx=33,my=29,surface_zs=[spatial])
        dense=solve_planar(prob,7e9;kw...)
        A=planar_ufft_operator(prob,7e9;kw...)
        x=randn(MersenneTwister(462),ComplexF64,nb)
        @test A*x≈dense.z_mom*x rtol=2e-12
        @test G._planar_ufft_diagonal(A)≈diag(dense.z_mom) rtol=2e-12
        @test opnorm(dense.s)<=1+1e-10
        @test dense.s≈transpose(dense.s) rtol=1e-10
        fast=solve_planar_ufft(prob,7e9;memory=100,rtol=1e-10,kw...)
        @test fast.s≈dense.s rtol=2e-9
        # High modes select folded storage even with separate cosine/sine
        # channels for all four translated terminal-half orientations.
        highkw=merge(kw,(;mx=65,my=63))
        high=solve_planar(prob,7e9;highkw...)
        for block in (1,7,512,typemax(Int))
            images=planar_ufft_operator(prob,7e9;highkw...,block)
            # The oversized block may select retained storage; every
            # smaller block must exercise the translated image channels.
            block==typemax(Int) || @test images.folded!==nothing
            if images.folded!==nothing
                @test length(images.folded.images)>length(images.families)
                @test size(images.folded.spectra,3)==length(images.families)*length(images.folded.images)
            end
            @test images*x≈high.z_mom*x rtol=2e-12
            @test G._planar_ufft_diagonal(images)≈diag(high.z_mom) rtol=2e-12
            @test G._planar_fft_dense_fill!(similar(high.z_mom),images)≈high.z_mom rtol=2e-12
            aliased=copy(x);mul!(aliased,images,aliased)
            @test aliased≈high.z_mom*x rtol=2e-12
            out=similar(x);mul!(out,images,x)
            @test (@allocated mul!(out,images,x))==0
        end
        highfast=solve_planar_ufft(prob,7e9;memory=100,rtol=1e-10,highkw...)
        @test highfast.operator.folded!==nothing
        @test highfast.s≈high.s rtol=2e-9
        @test highfast.currents≈high.currents rtol=1e-8
        @test maximum(highfast.relative_residuals)<1e-10
        # Wall and translated halves can belong to the same kind/element
        # family. Its two source channels must also preserve the wall rows.
        mixedsheet=sheet_level(1,grid.nx,grid.ny);mixedsheet.mask.=sh.mask
        # Separate corner cells add box-wall halves without closing any
        # of the existing open terminal edges.
        mixedsheet.mask[[1,grid.nx],[1,grid.ny]].=true
        mixedsheet.connect_west[[1,grid.ny]].=true
        mixedsheet.connect_east[[1,grid.ny]].=true
        mixedsheet.connect_south[[1,grid.nx]].=true
        mixedsheet.connect_north[[1,grid.nx]].=true
        mixed=build_planar_problem(stack,grid,[mixedsheet],ports)
        mixedmatrix=assemble_planar_z(stack,grid,[mixedsheet],mixed.basis,2pi*7e9;highkw...)
        mixedoperator=planar_ufft_operator(mixed,7e9;highkw...)
        @test mixedoperator.folded!==nothing
        @test any(f->f.kind==G._BASIS_X_LO &&
            any(z->rem(z-1,2grid.nx)==0,f.lattice) &&
            any(z->rem(z-1,2grid.nx)==1,f.lattice),mixedoperator.families)
        mixedx=randn(MersenneTwister(632),ComplexF64,size(mixedmatrix,1))
        @test mixedoperator*mixedx≈mixedmatrix*mixedx rtol=2e-12
        @test G._planar_ufft_diagonal(mixedoperator)≈diag(mixedmatrix) rtol=2e-12
        @test G._planar_fft_dense_fill!(similar(mixedmatrix),mixedoperator)≈mixedmatrix rtol=2e-12
    end
end
