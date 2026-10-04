using DiffMoM,Test,LinearAlgebra

function reference_source_oracle(Y,refs)
    n=length(refs);roots=sqrt.(real.(refs));S=zeros(ComplexF64,n,n)
    for q in 1:n
        a=zeros(ComplexF64,n);a[q]=1
        # An independently imposed circuit source obeys V+Zref I=2sqrt(R)a.
        v=(Matrix{ComplexF64}(I,n,n)+Diagonal(refs)*Y)\(2roots.*a)
        current=Y*v
        S[:,q].=(v-conj.(refs).*current)./(2roots)
    end
    S
end

@testset "complex references preserve independent physical power" begin
    refs=ComplexF64[40+20im,75-30im,28+90im]
    G=[.13 .04 -.02;.04 .2 .01;-.02 .01 .09]
    B=[.04 -.015 .027;-.015 -.01 .005;.027 .005 .03]
    for Y in (G+im*B,im*B)
        S=planar_y_to_s(Y,refs)
        @test S≈reference_source_oracle(Y,refs) rtol=3e-14
        @test S≈transpose(S) rtol=4e-14
        @test opnorm(S)<=1+4e-14
        @test planar_s_to_y(S,refs)≈Y rtol=5e-14
        for a in (ComplexF64[1,0,0],ComplexF64[.4+.3im,-.7im,.2-.4im])
            b=S*a;v=(conj.(refs).*a+refs.*b)./sqrt.(real.(refs));current=Y*v
            waves=planar_power_waves(v,current;z0=refs)
            @test waves.a≈a rtol=5e-14
            @test waves.b≈b rtol=5e-14
            @test waves.accepted_power≈.5*(sum(abs2,a)-sum(abs2,b)) atol=3e-15 rtol=5e-14
            @test waves.accepted_power≈.5real(dot(v,real.(Y)*v)) atol=3e-15 rtol=5e-14
            @test planar_wave_voltages(S,a;z0=refs)≈v rtol=4e-14
        end
        iszero(real(Y[1,1])) && @test adjoint(S)*S≈Matrix{ComplexF64}(I,3,3) atol=5e-14
    end
    @test planar_y_to_s(zeros(ComplexF64,3,3),refs)==Matrix{ComplexF64}(I,3,3)
    @test norm(planar_y_to_s(Diagonal(1 ./conj.(refs)),refs))<8e-16
    @test planar_y_to_s(1e14Matrix{ComplexF64}(I,3,3),refs)≈Diagonal(-conj.(refs)./refs) atol=1e-14
    @test_throws ArgumentError planar_y_to_s(ones(ComplexF64,1,1),[0+50im])
    @test_throws ArgumentError planar_y_to_s(fill(1e308,1,1),[1e308])
    nearshort=1e12*[2. .3;.3 1.];roots=Diagonal(sqrt.([50.,75.]))
    tiny=2*((I+roots*nearshort*roots)\Matrix{Float64}(I,2,2))-I
    @test planar_y_to_s(nearshort,[50.,75.])[1,2]≈tiny[1,2] rtol=4e-15
end

function complex_reference_geometry(refs)
    a=b=1e-3;grid=CellGrid(a,b,4,4)
    stack=PlanarStackup([PlanarLayer(2-.01im,1.,.4e-3),PlanarLayer(1.,1.,.6e-3)],TERM_GND,TERM_GND,a,b)
    sheet=sheet_level(1,4,4);sheet.mask[:,2:3].=true
    sheet.connect_west[2:3].=true;sheet.connect_east[2:3].=true
    raster=build_planar_problem(stack,grid,[sheet],[PlanarPort(1,:west,2:3,refs[1]),PlanarPort(1,:east,2:3,refs[2])])
    vertices=hcat(([i*grid.dx,j*grid.dy] for j in 1:3 for i in 0:4)...)
    id(i,j)=i+1+5j;triangles=Vector{Vector{Int}}()
    for j in 0:1,i in 0:3
        push!(triangles,[id(i,j),id(i+1,j),id(i,j+1)],[id(i+1,j+1),id(i,j+1),id(i+1,j)])
    end
    mesh=PlanarConformalMesh(vertices,hcat(triangles...);interfaces=fill(1,length(triangles)))
    conformal=PlanarConformalProblem(stack,mesh,[PlanarConformalPort(1,:west,(b/4,3b/4);z0=refs[1]),
        PlanarConformalPort(1,:east,(b/4,3b/4);z0=refs[2])])
    via=via_level(1,4,4);via.uni[2,2]=via.tap[2,2]=true
    volume=vol_level(2,4,4);volume.mask[2:3,2:3].=true
    hybrid=PlanarHybridProblem(conformal,grid;vias=[via],vols=[volume])
    (;raster,conformal,hybrid)
end

function current_map_difference(a,b)
    if a isa NamedTuple
        return max(current_map_difference(a.triangles,b.triangles),current_map_difference(a.bulk,b.bulk))
    elseif isempty(a)
        return 0.
    elseif hasproperty(first(a),:current)
        return maximum(norm(a[i].current-b[i].current) for i in eachindex(a))
    else
        return maximum(max(norm(a[i].jx-b[i].jx),norm(a[i].jy-b[i].jy),norm(a[i].jz-b[i].jz)) for i in eachindex(a))
    end
end

@testset "all raw solvers retain evaluated references and physical wave currents" begin
    calls=Float64[];multiplier=Ref(1.)
    law=PlanarPortImpedance(r=f->30+f/1e9,x=20.,l=1e-9)
    provider=f->(push!(calls,f);multiplier[]*law(f))
    geometry=complex_reference_geometry([provider,80-15im])
    @test isempty(calls) # Geometry and basis construction never evaluate providers.
    f=4e9;expected=ComplexF64[law(f),80-15im];results=Any[]
    for method in (:dense,:dense_fft,:ufft)
        r=method===:ufft ? solve_planar(geometry.raster,f;method,mx=17,my=19,surface_zs=.2,rtol=1e-10,memory=64,maxiter=2000) :
            solve_planar(geometry.raster,f;method,mx=17,my=19,surface_zs=.2)
        @test r.z0==expected
        @test r.s≈reference_source_oracle(r.y,expected) rtol=2e-12
        @test opnorm(r.s)<=1+1e-10
        push!(results,r)
    end
    for p in (geometry.conformal,geometry.hybrid),method in (:dense,:ufft)
        solver=p isa PlanarHybridProblem ? solve_planar_hybrid : solve_planar_conformal
        material=p isa PlanarHybridProblem ? (;surface_zs=.2,via_sigma=1e5,volume_sigma=1e4) : (;surface_zs=.2)
        lattice=p isa PlanarHybridProblem ? (;) : (;nx=4,ny=4)
        r=method===:ufft ? solver(p,f;method,mx=17,my=19,rtol=1e-10,memory=64,maxiter=2000,material...,lattice...) :
            solver(p,f;method,mx=17,my=19,material...)
        @test r.z0==expected
        @test r.s≈reference_source_oracle(r.y,expected) rtol=2e-11
        @test maximum(r.relative_residuals)<1e-10
        @test opnorm(r.s)<=1+1e-10
        push!(results,r)
    end
    @test calls==fill(f,7)
    fixed=complex_reference_geometry([50.,75.])
    for r in results
        fixed_s=planar_y_to_s(r.y,[50.,75.])
        legacy=if r isa PlanarResult
            PlanarResult(fixed.raster,r.omega,r.freq,r.z_mom,r.lu_fact,r.currents,r.y,fixed_s)
        elseif r isa PlanarUFFTResult
            PlanarUFFTResult(fixed.raster,r.freq,r.omega,r.operator,r.currents,r.y,fixed_s,r.iterations,r.relative_residuals)
        elseif r isa PlanarConformalResult
            PlanarConformalResult(fixed.conformal,r.freq,r.omega,r.z_mom,r.lu_fact,r.basis_scale,r.currents,r.y,fixed_s,r.relative_residuals)
        elseif r isa PlanarConformalUFFTResult
            PlanarConformalUFFTResult(fixed.conformal,r.freq,r.omega,r.operator,r.currents,r.y,fixed_s,r.iterations,r.relative_residuals)
        elseif r isa PlanarHybridResult
            PlanarHybridResult(fixed.hybrid,r.freq,r.omega,r.z_mom,r.lu_fact,r.basis_scale,r.currents,r.y,fixed_s,r.relative_residuals)
        else
            PlanarHybridUFFTResult(fixed.hybrid,r.freq,r.omega,r.operator,r.currents,r.y,fixed_s,r.iterations,r.relative_residuals)
        end
        @test legacy.z0==[50.,75.]
        @test legacy.currents===r.currents
        @test legacy.s===fixed_s
    end
    multiplier[]=2.;a=ComplexF64[.4+.2im,-.3im]
    for r in results
        v=(conj.(expected).*a+expected.*(r.s*a))./sqrt.(real.(expected))
        @test current_map_difference(planar_current_maps(r;incident_waves=a),planar_current_maps(r;voltages=v))<1e-9
        @test .5real(dot(v,r.y*v))≈.5*(sum(abs2,a)-sum(abs2,r.s*a)) atol=2e-13 rtol=2e-11
    end
    @test calls==fill(f,7) # Current reconstruction uses retained refs.
    next=solve_planar(geometry.raster,7e9;mx=17,my=19,surface_zs=.2)
    @test next.z0==ComplexF64[2law(7e9),80-15im]
    @test_throws ArgumentError solve_planar(geometry.raster,f;max_bytes=1)
    @test last(calls)==7e9
    count=length(calls)
    for p in (geometry.raster,geometry.conformal,geometry.hybrid),method in (:dense,:ufft)
        solver=p isa PlanarProblem ? solve_planar : p isa PlanarHybridProblem ? solve_planar_hybrid : solve_planar_conformal
        @test_throws ArgumentError solver(p,0.;method)
        @test length(calls)==count
    end
    @test_throws ArgumentError PlanarPort(1,:west,1:2,0+20im)
    @test_throws ArgumentError PlanarConformalPort(1,:west,(0.,1.);z0=0+20im)
    # Numeric four- and six-argument raster constructors remain available.
    @test PlanarPort(1,:west,1:2,50.).z0==50+0im
    @test PlanarPort(1,:west,1:2,50.,0,1).z0==50+0im
end
