module PlanarTriangleMomentReuseTests
using DiffMoM,Test,LinearAlgebra,Random,SparseArrays
const G=DiffMoM

# The previous independent, per-vertex divided-difference formula. Keep its
# evaluation here so changing the production shared table cannot change both
# sides of the comparison. Each recursion has at most four nodes and the
# clustered expansion has the same fixed 32-term bound.
function legacy_divdiff(nodes::NTuple{N,ComplexF64}) where N
    N==1 && return exp(nodes[1])
    if abs(nodes[end]-nodes[1])<=1.0
        center=sum(nodes)/N;z=map(x->x-center,nodes)
        e1=sum(z);e2=0im;e3=0im;e4=0im
        for i in 1:N,j in i+1:N
            e2+=z[i]*z[j]
            for k in j+1:N
                e3+=z[i]*z[j]*z[k]
                for l in k+1:N;e4+=z[i]*z[j]*z[k]*z[l];end
            end
        end
        h0=1.0+0im;h1=h2=h3=0im
        factorial=Float64(prod(1:N-1;init=1));value=h0/factorial
        for m in 1:32
            h=e1*h0-e2*h1+e3*h2-e4*h3;factorial*=m+N-1
            value+=h/factorial;h3=h2;h2=h1;h1=h0;h0=h
        end
        return exp(center)*value
    end
    return (legacy_divdiff(ntuple(i->nodes[i+1],Val(N-1)))-
        legacy_divdiff(ntuple(i->nodes[i],Val(N-1))))/(nodes[end]-nodes[1])
end

function legacy_triangle_weights(z::NTuple{3,ComplexF64})
    sorted=Tuple(sort(collect(z);by=imag))
    ntuple(j->legacy_divdiff(Tuple(sort([sorted...;z[j]];by=imag))),3)
end
function legacy_pulse(v,kx,ky)
    phase=ntuple(j->ComplexF64(im*(kx*v[1,j]+ky*v[2,j])),3)
    shift=phase[1];weights=legacy_triangle_weights(map(z->z-shift,phase))
    factor=abs(G._planar_orient2d(v[1,1],v[2,1],v[1,2],v[2,2],v[1,3],v[2,3]))*exp(shift)
    constant=0im;x=0im;y=0im
    for j in 1:3
        moment=factor*weights[j]
        constant+=moment;x+=(v[1,j]-v[1,1])*moment;y+=(v[2,j]-v[2,1])*moment
    end
    (constant,x,y)
end
function legacy_affine(v,values,kx,ky)
    phase=ntuple(j->ComplexF64(im*(kx*v[1,j]+ky*v[2,j])),3)
    shift=phase[1];weights=legacy_triangle_weights(map(z->z-shift,phase))
    factor=abs(G._planar_orient2d(v[1,1],v[2,1],v[1,2],v[2,2],v[1,3],v[2,3]))*exp(shift)
    factor*sum(values[j]*weights[j] for j in 1:3)
end

function allocation_checks(v,kx,ky)
    x=(1.,-.3,2.);y=(-.7,2.,.1)
    G._planar_pulse_and_first_moments(v,kx,ky)
    G._planar_triangle_affine_fourier_pair(v,x,y,kx,ky)
    ((@allocated G._planar_pulse_and_first_moments(v,kx,ky)),
     (@allocated G._planar_triangle_affine_fourier_pair(v,x,y,kx,ky)))
end

@testset "shared triangle weights retain independent previous formula" begin
    rng=MersenneTwister(1604357)
    for scale in (1e-3,1.,1e3),trial in 1:120
        v=rand(rng,2,3).*scale
        kx,ky=randn(rng,2).*10.0^(rand(rng,(-2,0,1,2,3)))./scale
        expected=legacy_pulse(v,kx,ky);actual=G._planar_pulse_and_first_moments(v,kx,ky)
        @test norm(collect(actual).-collect(expected))<=8e-13*norm(collect(expected))
        @test allocation_checks(v,kx,ky)==(0,0)
        x=(1.,-.3,2.);y=(-.7,2.,.1)
        pair=G._planar_triangle_affine_fourier_pair(v,x,y,kx,ky)
        @test pair[1]≈legacy_affine(v,x,kx,ky) rtol=8e-13 atol=1e-20
        @test pair[2]≈legacy_affine(v,y,kx,ky) rtol=8e-13 atol=1e-20
        @test pair==(G._planar_triangle_affine_fourier(v,x,kx,ky),
                     G._planar_triangle_affine_fourier(v,y,kx,ky))
    end
    # Repeated nodes, signed zero, the exact branch boundaries and neighbors,
    # mixed gaps and large integer modal phases all retain the old formula.
    for z in ((0im,0im,0im),(-0.0im,0im,0im),(0im,0im,1im),
              (0im,.5im,prevfloat(1.)*im),(0im,.5im,1im),(0im,.5im,nextfloat(1.)*im),
              (0im,prevfloat(1.)*im,3im),(0im,1im,3im),(0im,nextfloat(1.)*im,3im),
              (0im,.1im,70im),(0im,4im,4im),(0im,128pi*im,257pi*im)),
            permutation in ((1,2,3),(3,2,1),(2,3,1))
        nodes=ntuple(j->ComplexF64(z[permutation[j]]),3)
        actual=G._planar_triangle_exp_weights3(nodes);expected=legacy_triangle_weights(nodes)
        @test norm(collect(actual).-collect(expected))<=8e-13*max(norm(collect(expected)),eps(1.))
        @test all(isfinite,actual)
    end
end

@testset "shared triangle weights independent 512-bit simplex identity" begin
    # The earlier independent exact-area/simplex Taylor oracle uses no divided
    # differences or production triangle weights. Its phases are bounded.
    shapes=([0. 1. 0.;0. 0. 1.],
        [0. 1. nextfloat(1.);0. nextfloat(1.) nextfloat(nextfloat(1.))])
    for scale in (2.0^-100,1.,2.0^100),v0 in shapes,
            permutation in ((1,2,3),(3,2,1),(2,3,1)),
            (kx,ky) in ((0.,0.),(.3,.5),(-1.2,1.2),(1.2,-1.2),(1.,0.),(.9,.90000001))
        v=v0[:,collect(permutation)].*scale
        value=G._planar_pulse_and_first_moments(v,kx/scale,ky/scale)
        expected=Main.ConformalPulseAreaTests.series_moments(v,kx/scale,ky/scale)
        @test norm(collect(value).-collect(expected))<=5e-13*norm(collect(expected))
    end
end

function fixture(;bridge=false,walls=WALL_PEC,nonuniform=true)
    a=b=2e-3;grid=CellGrid(a,b,4,4;walls)
    layers=bridge ? [PlanarLayer(1.,1.,.4e-3),PlanarLayer(1.,1.,20e-6),PlanarLayer(1.,1.,.6e-3)] :
        [PlanarLayer(2-.002im,1.,.4e-3),PlanarLayer(1.,1.,.6e-3)]
    stack=PlanarStackup(layers,TERM_GND,TERM_GND,a,b)
    vertices=Vector{Vector{Float64}}();triangles=Vector{Vector{Int}}();levels=Int[]
    for (level,xs) in (bridge ? ((1,0:2),(2,1:4)) : ((1,0:4),))
        base=length(vertices);width=length(xs)
        append!(vertices,[[i*grid.dx,j*grid.dy] for j in 1:3 for i in xs])
        id(i,j)=base+i+1+width*j
        for j in 0:1,i in 0:width-2
            append!(triangles,[[id(i,j),id(i+1,j),id(i,j+1)],
                [id(i+1,j+1),id(i,j+1),id(i+1,j)]])
            append!(levels,[level,level])
        end
    end
    # Genuine nonuniform subtriangles preserve the cell/contact boundary edges.
    v=hcat(vertices...);subtriangles=Vector{Vector{Int}}();sublevels=Int[]
    for (t,ids) in enumerate(triangles)
        if nonuniform && iseven(t)
            v=hcat(v,v[:,ids]*[.2,.3,.5]);free=size(v,2)
            append!(subtriangles,[[ids[1],ids[2],free],[ids[2],ids[3],free],[ids[3],ids[1],free]])
            append!(sublevels,fill(levels[t],3))
        else
            push!(subtriangles,ids);push!(sublevels,levels[t])
        end
    end
    c=PlanarConformalProblem(stack,PlanarConformalMesh(v,hcat(subtriangles...);interfaces=sublevels),
        [PlanarConformalPort(1,:west,(b/4,3b/4)),PlanarConformalPort(bridge ? 2 : 1,:east,(b/4,3b/4))];sidewalls=walls)
    via=via_level(bridge ? 2 : 1,4,4);via.uni[2,2:3].=true;via.tap[2,2:3].=true
    vols=VolLevel[]
    if !bridge
        volume=vol_level(2,4,4);volume.mask[3:4,2:3].=true;push!(vols,volume)
    end
    PlanarHybridProblem(c,grid;vias=[via],vols)
end

function legacy_sheet_weights!(te,tm,c,kx,ky)
    for b in eachindex(c.basis.width)
        px=py=0.
        for half in 1:2
            c.basis.triangles[half,b]==0 && continue
            t,x,y=G._planar_conformal_half_values(c,b,half)
            v=view(c.mesh.vertices,:,view(c.mesh.triangles,:,t))
            xp=legacy_affine(v,x,kx,ky);xm=legacy_affine(v,x,kx,-ky)
            yp=legacy_affine(v,y,kx,ky);ym=legacy_affine(v,y,kx,-ky)
            if c.sidewalls===WALL_PEC
                px+=(imag(xp)-imag(xm))/2;py+=(imag(yp)+imag(ym))/2
            else
                px+=(imag(xp)+imag(xm))/2;py+=(imag(yp)-imag(ym))/2
            end
        end
        te[b]=ky*px-kx*py;tm[b]=kx*px+ky*py
    end
end

# Independent scalar modal reaction. This computes every original pair from
# the per-vertex legacy integrals; it does not use the grouped assembler, FFT
# action, projected action or the newly shared weights.
function legacy_hybrid_matrix(prob,f;mx,my,surface_zs,via_sigma,volume_sigma)
    c,r=prob.conformal,prob.bulk;nc=length(c.basis.width);nr=planar_basis_count(r.basis);nb=nc+nr
    mg=planar_mode_grid(r.grid,mx,my)
    elems=[c.basis.interfaces;[G._basis_elem(r.basis,b,r.sheets,r.vias,r.vols) for b in 1:nr]]
    groups=sort!(unique(elems));pairs=[(a,b) for a in groups for b in groups]
    pairindex=Dict(pair=>i for (i,pair) in enumerate(pairs))
    vlay=sort!(unique(G._via_elem_layer(e) for e in groups if G._is_via_elem(e)))
    volay=sort!(unique(G._vol_elem_layer(e) for e in groups if G._is_vol_elem(e)))
    cte,ctm,scratch,vstates,volstates=G._planar_mode_workspace(length(prob.stack.layers),!isempty(vlay),!isempty(volay))
    fxb=zeros(mx,nr);fyb=zeros(my,nr)
    for b in 1:nr
        G._basis_fx!(view(fxb,:,b),r.basis,b,mg,r.grid);G._basis_fy!(view(fyb,:,b),r.basis,b,mg,r.grid)
    end
    te=zeros(nb);tm=similar(te);rt=zeros(1,nr);rm=similar(rt)
    vt=zeros(ComplexF64,1,length(pairs));vm=similar(vt);ml=[1];nl=[1];Z=zeros(ComplexF64,nb,nb)
    for nn in 1:my,mm in 1:mx
        ml[1]=mm;nl[1]=nn
        legacy_sheet_weights!(view(te,1:nc),view(tm,1:nc),c,mg.kx[mm],mg.ky[nn])
        G._planar_weight_block!(rt,rm,ml,nl,mg,fxb,fyb,r.basis)
        for b in 1:nr;te[nc+b]=rt[1,b];tm[nc+b]=rm[1,b];end
        G._planar_mode_voltages!(vt,vm,cte,ctm,scratch,prob.stack,2pi*f,mg,ml,nl,pairs,vstates,vlay,volstates,volay)
        for q in 1:nb,p in 1:nb
            pair=pairindex[(elems[p],elems[q])]
            Z[p,q]-=te[p]*te[q]*vt[1,pair]+tm[p]*tm[q]*vm[1,pair]
        end
    end
    G._planar_conformal_loss_entries(c,surface_zs) do p,q,value;Z[p,q]+=value;end
    vr=G._planar_bulk_resistivities(via_sigma,length(r.vias),"via_sigma")
    br=G._planar_bulk_resistivities(volume_sigma,length(r.vols),"volume_sigma")
    G._planar_bulk_loss_entries(r.basis,r.grid,r.stack,r.vias,r.vols,vr,br) do p,q,value,layer,power
        Z[nc+p,nc+q]+=value
    end
    Z
end

@testset "shared triangle weights retain mixed original modal actions" begin
    rng=MersenneTwister(16431)
    for walls in (WALL_PEC,WALL_PMC),bridge in (false,true),sigma in (Inf,1e5,1im*1e5)
        prob=fixture(;bridge,walls);f=3e9;mx,my=13,15
        oracle=legacy_hybrid_matrix(prob,f;mx,my,surface_zs=.2+.1im,via_sigma=sigma,volume_sigma=1e4)
        actual=assemble_planar_hybrid_z(prob,f;mx,my,surface_zs=.2+.1im,via_sigma=sigma,volume_sigma=1e4)
        # The exact lattice FFT deliberately rejects off-lattice vertices;
        # compare its supported geometry against a separate legacy oracle.
        lattice=fixture(;bridge,walls,nonuniform=false)
        lattice_oracle=legacy_hybrid_matrix(lattice,f;mx,my,surface_zs=.2+.1im,via_sigma=sigma,volume_sigma=1e4)
        A=planar_hybrid_ufft_operator(lattice,f;mx,my,surface_zs=.2+.1im,via_sigma=sigma,volume_sigma=1e4,block=7)
        @test actual≈oracle rtol=8e-13 atol=1e-12
        @test oracle≈transpose(oracle) rtol=4e-13 atol=1e-12
        @test G._planar_hybrid_ufft_diagonal(A)≈diag(lattice_oracle) rtol=8e-13 atol=1e-12
        for trial in 1:3
            x=randn(rng,ComplexF64,size(lattice_oracle,1));y=similar(x)
            mul!(y,A,x)
            @test y≈lattice_oracle*x rtol=8e-13 atol=1e-12
            @test (@allocated mul!(y,A,x))==0
            aliased=copy(x);mul!(aliased,A,aliased)
            @test aliased≈lattice_oracle*x rtol=8e-13 atol=1e-12
        end
        @test_throws ArgumentError planar_hybrid_ufft_operator(prob,f;mx,my,max_bytes=1)
        @test_throws ArgumentError planar_hybrid_ufft_operator(prob,f;mx=0,my)
        @test_throws ArgumentError planar_hybrid_ufft_operator(prob,f;mx,my,block=0)
        @test_throws ArgumentError mul!(zeros(ComplexF64,size(A,1)),A,A.output)
        @test_throws ArgumentError planar_hybrid_ufft_operator(prob,f;mx,my)
    end
end

@testset "shared triangle weights original physical solve currents and power" begin
    for bridge in (false,true)
        prob=fixture(;bridge);f=3e9;mx,my=16,18
        oracle=legacy_hybrid_matrix(prob,f;mx,my,surface_zs=.2+.1im,via_sigma=1e5,volume_sigma=1e4)
        r=solve_planar_hybrid(prob,f;mx,my,surface_zs=.2+.1im,via_sigma=1e5,volume_sigma=1e4)
        weights,signs,ports=G._planar_hybrid_traces(prob);scale=inv.(weights)
        rhs=zeros(ComplexF64,length(weights),2)
        for b in eachindex(weights);ports[b]==0 || (rhs[b,ports[b]]=-signs[b]*weights[b]);end
        reference=(Diagonal(scale)*oracle*Diagonal(scale))\(Diagonal(scale)*rhs)
        reference=Diagonal(scale)*reference
        @test r.currents≈reference rtol=2e-9 atol=1e-12
        for p in 1:2
            @test norm(scale.*(oracle*view(r.currents,:,p)-view(rhs,:,p)))/norm(scale.*view(rhs,:,p))<=1e-10
        end
        @test maximum(r.relative_residuals)<=1e-10
        @test r.y≈transpose(r.y) rtol=2e-9 atol=1e-12
        @test opnorm(r.s)<=1+1e-10
        voltage=ComplexF64[1+im,.3-.2im];current=r.currents*voltage
        @test .5real(dot(voltage,r.y*voltage))≈-.5real(dot(current,oracle*current)) rtol=2e-9 atol=1e-12
        waves=planar_power_waves(voltage,r.y*voltage;z0=r.z0)
        @test waves.b≈r.s*waves.a rtol=2e-9 atol=1e-12
        @test .5(sum(abs2,waves.a)-sum(abs2,waves.b))≈.5real(dot(voltage,r.y*voltage)) rtol=2e-9 atol=1e-12
        maps=planar_hybrid_current_maps(r;voltages=voltage)
        @test length(maps.triangles)==length(prob.conformal.mesh.interfaces)
        @test all(t->all(isfinite,t.current),maps.triangles)
        @test all(t->all(isfinite,t.jz),maps.bulk)
        @test_throws ArgumentError solve_planar_hybrid(prob,f;mx,my,max_bytes=1)
    end
end

@testset "shared triangle transforms keep public input and cutoff contracts" begin
    prob=fixture().conformal
    @test_throws ArgumentError G._planar_conformal_fourier(prob,0,0.,0.)
    @test_throws ArgumentError G._planar_conformal_fourier(prob,1,Inf,0.)
    @test_throws ArgumentError G._planar_conformal_fourier(prob,1,0.,NaN)
    @test_throws ArgumentError assemble_planar_conformal_z(prob,3e9;mx=0,my=4)
    @test_throws ArgumentError assemble_planar_conformal_z(prob,3e9;mx=4,my=0)
    @test_throws ArgumentError assemble_planar_conformal_z(prob,NaN;mx=4,my=4)
    for kx in (0.,pi/prob.stack.a,128pi/prob.stack.a),ky in (0.,pi/prob.stack.b,128pi/prob.stack.b)
        te=zeros(length(prob.basis.width));tm=similar(te);et=similar(te);em=similar(te)
        G._planar_conformal_weights!(te,tm,prob,kx,ky)
        legacy_sheet_weights!(et,em,prob,kx,ky)
        @test te≈et rtol=8e-13 atol=1e-12
        @test tm≈em rtol=8e-13 atol=1e-12
    end
end
end
