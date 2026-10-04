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
    end
end
