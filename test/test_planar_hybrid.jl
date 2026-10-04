using DiffMoM,Test,LinearAlgebra,Random

function hybrid_fixture(;bridge=false,ground=false,volume=false,walls=WALL_PEC)
    a,b=2e-3,2e-3;nx=ny=4;grid=CellGrid(a,b,nx,ny;walls)
    layers=bridge ? [PlanarLayer(2-.002im,1.,.4e-3),PlanarLayer(1.,1.,20e-6),PlanarLayer(1.,1.,.6e-3)] :
        [PlanarLayer(2-.002im,1.,.4e-3),PlanarLayer(1.,1.,.6e-3)]
    stack=PlanarStackup(layers,TERM_GND,TERM_GND,a,b)
    vertices=Vector{Vector{Float64}}();triangles=Vector{Vector{Int}}();levels=Int[]
    # Every via contact cell is constrained into exactly two true triangles.
    for level in (bridge ? (1,2) : (1,))
        xs=bridge ? (level==1 ? (0:2) : (2:4)) : (0:4)
        base=length(vertices);width=length(xs)
        append!(vertices,[[i*grid.dx,j*grid.dy] for j in 1:3 for i in xs])
        id(i,j)=base+i+1+width*j
        for j in 0:1,i in 0:width-2
            push!(triangles,[id(i,j),id(i+1,j),id(i,j+1)],[id(i+1,j+1),id(i,j+1),id(i+1,j)])
            append!(levels,[level,level])
        end
    end
    if bridge
        # Extend the left sheet through the contact column and right sheet
        # back through it: both faces meet the same physical via footprint.
        return hybrid_bridge_fixture(;walls)
    end
    mesh=PlanarConformalMesh(hcat(vertices...),hcat(triangles...);interfaces=levels)
    cp=PlanarConformalProblem(stack,mesh,[PlanarConformalPort(1,:west,(b/4,3b/4)),PlanarConformalPort(1,:east,(b/4,3b/4))];sidewalls=walls)
    vias=ViaLevel[];vols=VolLevel[]
    if ground
        v=via_level(1,nx,ny);v.uni[2,2:3].=true;v.tap[2,2:3].=true;push!(vias,v)
    end
    if volume
        v=vol_level(2,nx,ny);v.mask[3:4,2:3].=true;push!(vols,v)
    end
    PlanarHybridProblem(cp,grid;vias,vols)
end

function hybrid_bridge_fixture(;walls=WALL_PEC)
    a=b=2e-3;nx=ny=4;grid=CellGrid(a,b,nx,ny;walls)
    stack=PlanarStackup([PlanarLayer(1.,1.,.4e-3),PlanarLayer(1.,1.,20e-6),PlanarLayer(1.,1.,.6e-3)],TERM_GND,TERM_GND,a,b)
    vs=Vector{Vector{Float64}}();ts=Vector{Vector{Int}}();levels=Int[]
    for (level,xs) in ((1,0:2),(2,1:4))
        base=length(vs);width=length(xs);append!(vs,[[i*grid.dx,j*grid.dy] for j in 1:3 for i in xs]);id(i,j)=base+i+1+width*j
        for j in 0:1,i in 0:width-2
            push!(ts,[id(i,j),id(i+1,j),id(i,j+1)],[id(i+1,j+1),id(i,j+1),id(i+1,j)]);append!(levels,[level,level])
        end
    end
    mesh=PlanarConformalMesh(hcat(vs...),hcat(ts...);interfaces=levels)
    cp=PlanarConformalProblem(stack,mesh,[PlanarConformalPort(1,:west,(b/4,3b/4)),PlanarConformalPort(2,:east,(b/4,3b/4))];sidewalls=walls)
    v=via_level(2,nx,ny);v.uni[2,2:3].=true;v.tap[2,2:3].=true
    PlanarHybridProblem(cp,grid;vias=[v])
end

@testset "genuine triangle and bulk exact cross reactions" begin
    G=DiffMoM;rng=MersenneTwister(811)
    for walls in (WALL_PEC,WALL_PMC)
        p=hybrid_fixture(;ground=true,volume=true,walls);c=p.conformal;r=p.bulk
        nc=length(c.basis.width);nr=planar_basis_count(r.basis);f=3e9;mx,my=13,15
        Z=assemble_planar_hybrid_z(p,f;mx,my,surface_zs=.2,via_sigma=1e5,volume_sigma=1e4)
        @test Z[1:nc,1:nc]≈assemble_planar_conformal_z(c,f;mx,my,surface_zs=.2) rtol=2e-14
        @test Z[nc+1:end,nc+1:end]≈assemble_planar_z(r.stack,r.grid,r.sheets,r.basis,2pi*f;mx,my,vias=r.vias,vols=r.vols,via_sigma=1e5,volume_sigma=1e4) rtol=2e-14
        @test Z≈transpose(Z) rtol=2e-14
        # Direct mode-by-mode scalar reaction, independent of grouped BLAS
        # assembly and its element-pair lookup/scattering.
        meta=G._planar_hybrid_cross_metadata(p,mx,my);mg=meta.mg;ct,cm,scratch,vs,vols=meta.workspace
        tte=zeros(nc);ttm=zeros(nc);rt=zeros(1,nr);rm=similar(rt);vt=zeros(ComplexF64,1,length(meta.pairs));vm=similar(vt)
        oracle=zeros(ComplexF64,nc,nr)
        for nn in 1:my,mm in 1:mx
            G._planar_conformal_weights!(tte,ttm,c,mg.kx[mm],mg.ky[nn])
            G._planar_weight_block!(rt,rm,[mm],[nn],mg,meta.fxb,meta.fyb,r.basis)
            G._planar_mode_voltages!(vt,vm,ct,cm,scratch,p.stack,2pi*f,mg,[mm],[nn],meta.pairs,vs,meta.vlay,vols,meta.volay)
            for q in 1:nr,b in 1:nc
                idx=meta.pairindex[(c.basis.interfaces[b],meta.elems[q])]
                oracle[b,q]-=tte[b]*rt[1,q]*vt[1,idx]+ttm[b]*rm[1,q]*vm[1,idx]
            end
        end
        @test Z[1:nc,nc+1:end]≈oracle rtol=2e-14
        rr=solve_planar_hybrid(p,f;mx,my,surface_zs=.2,via_sigma=1e5,volume_sigma=1e4)
        @test opnorm(rr.s)<=1+1e-10
        @test maximum(rr.relative_residuals)<1e-10
        @test rr.y≈transpose(rr.y) rtol=1e-10
        maps=planar_hybrid_current_maps(rr;voltages=[1.,-1.])
        @test length(maps.triangles)==length(c.mesh.interfaces)
        @test Set(x.kind for x in maps.bulk)==Set((:via,:volume))
        @test_throws ArgumentError assemble_planar_hybrid_z(p,f;mx,my,max_bytes=1)
        @test_throws ArgumentError solve_planar_hybrid(p,f;mx,my,max_bytes=1)
        @test_throws ArgumentError planar_hybrid_current_maps(rr;max_bytes=1)
    end
end

@testset "triangular sheet physical galvanic ground and interlevel contacts" begin
    bare=solve_planar_hybrid(hybrid_fixture(),1e8;mx=24,my=24)
    short=solve_planar_hybrid(hybrid_fixture(;ground=true),1e8;mx=24,my=24)
    bridge=solve_planar_hybrid(hybrid_fixture(;bridge=true),1e8;mx=24,my=24)
    @test abs(bare.s[2,1])>.99
    @test abs(short.s[2,1])<.04
    @test abs(bridge.s[2,1])>.99
    @test opnorm(short.s)<=1+1e-10
    @test opnorm(bridge.s)<=1+1e-10
    lossy=solve_planar_hybrid(hybrid_fixture(;bridge=true),1e6;mx=24,my=24,via_sigma=100.)
    @test real(2/(lossy.y[1,1]-lossy.y[1,2]))≈20e-6/(100*2*(.5e-3)^2) rtol=2e-5
end

@testset "exact mixed conformal bulk FFT and strict solve" begin
    rng=MersenneTwister(691);G=DiffMoM
    for walls in (WALL_PEC,WALL_PMC)
        p=hybrid_fixture(;ground=true,volume=true,walls);f=3e9;mx,my=19,23
        Z=assemble_planar_hybrid_z(p,f;mx,my,surface_zs=.2,via_sigma=1e5,volume_sigma=1e4)
        A=planar_hybrid_ufft_operator(p,f;mx,my,surface_zs=.2,via_sigma=1e5,volume_sigma=1e4,block=7)
        x=randn(rng,ComplexF64,A.n);y=similar(x)
        @test A*x≈Z*x rtol=5e-13
        @test G._planar_hybrid_ufft_diagonal(A)≈diag(Z) rtol=5e-13
        mul!(y,A,x);@test (@allocated mul!(y,A,x))==0
        original=copy(y);mul!(y,A,x,.7+.2im,-.3)
        @test y≈(.7+.2im)*(Z*x)-.3original rtol=5e-13
        alias=copy(x);mul!(alias,A,alias);@test alias≈Z*x rtol=5e-13
        @test size(A,3)==1
        @test_throws ArgumentError size(A,0)
        @test_throws ArgumentError mul!(y,A,A.output)
        @test_throws ArgumentError planar_hybrid_ufft_operator(p,f;mx,my,max_bytes=1)
        @test_throws ArgumentError G._planar_hybrid_ufft_diagonal(A;max_bytes=1)
        dense=solve_planar_hybrid(p,f;mx,my,surface_zs=.2,via_sigma=1e5,volume_sigma=1e4)
        fft=solve_planar_hybrid(p,f;method=:ufft,mx,my,surface_zs=.2,via_sigma=1e5,volume_sigma=1e4,memory=100,rtol=1e-10)
        @test fft.currents≈dense.currents rtol=3e-9
        @test fft.y≈dense.y rtol=3e-9
        @test maximum(fft.relative_residuals)<=1e-10
        @test opnorm(fft.s)<=1+1e-10
        cm=planar_hybrid_current_maps(fft;port=2);dm=planar_hybrid_current_maps(dense;port=2)
        @test all(cm.triangles[t].current≈dm.triangles[t].current for t in eachindex(cm.triangles))
    end
    p=hybrid_fixture(;bridge=true)
    fft=solve_planar_hybrid(p,1e8;method=:ufft,mx=24,my=24,memory=100,rtol=1e-10)
    dense=solve_planar_hybrid(p,1e8;mx=24,my=24)
    @test fft.s≈dense.s rtol=1e-9
    @test maximum(fft.relative_residuals)<=1e-10
    @test_throws ArgumentError solve_planar_hybrid(p,1e8;method=:unknown)
    @test_throws ArgumentError solve_planar_hybrid(p,1e8;method=:ufft,rtol=NaN)
    @test_throws ArgumentError solve_planar_hybrid(p,1e8;method=:ufft,memory=0)
    @test_throws ArgumentError solve_planar_hybrid(p,1e8;method=:ufft,max_bytes=1)
    # A sheet-only hybrid retains the same physical triangular operator.
    bare=hybrid_fixture();a=solve_planar_hybrid(bare,1e9;method=:ufft,mx=19,my=23,memory=100,rtol=1e-10)
    b=solve_planar_conformal(bare.conformal,1e9;method=:ufft,nx=4,ny=4,mx=19,my=23,memory=100,rtol=1e-10)
    @test a.y≈b.y rtol=1e-11
end
