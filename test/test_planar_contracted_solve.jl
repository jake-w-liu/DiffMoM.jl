using DiffMoM,Test,LinearAlgebra

function _contracted_return_fixture()
    grid=CellGrid(4e-3,3e-3,8,6)
    stack=PlanarStackup([PlanarLayer(1.,1.,.2e-3),PlanarLayer(2.,1.,.4e-3),
        PlanarLayer(1.,1.,1e-3)],TERM_GND,TERM_GND,grid.a,grid.b)
    sheet=sheet_level(2,8,6);sheet.mask[3:6,3:4].=true
    terminals=[PlanarPort(1,:terminal_x,6,3:4,50.;metal_side=:negative),
        PlanarPort(1,:terminal_x,2,3:4,75.;metal_side=:positive)]
    return planar_terminal_returns(stack,grid,[sheet],terminals;ground_direction=:below)
end

@testset "contracted solve: FFT accuracy is independent of source amplitude" begin
    grid=CellGrid(2e-3,1e-3,8,4)
    stack=PlanarStackup([PlanarLayer(4.,1.,.2e-3),PlanarLayer(1.,1.,.8e-3)],
        TERM_GND,TERM_GND,grid.a,grid.b)
    sheet=sheet_level(1,8,4);sheet.mask[:,2:3].=true
    sheet.connect_west[2:3].=true;sheet.connect_east[2:3].=true
    prob=build_planar_problem(stack,grid,[sheet],
        [PlanarPort(1,:west,2:3,50.),PlanarPort(1,:east,2:3,50.)])
    identity=Matrix{Float64}(I,2,2)
    dense=solve_planar_contracted(prob,1e9,identity;mx=32,my=32,surface_zs=.2)
    weights=prob.basis.width
    # Independent physical wall sources: west is negative, east positive.
    B=zeros(ComplexF64,size(dense.currents))
    for b in axes(B,1)
        p=prob.basis.port[b];p==0 && continue
        B[b,p]=(p==1 ? -1 : 1)*weights[b]
    end
    for alpha in (1.,1e-16,1e-18,-1e-16),precondition in (true,false)
        fft=solve_planar_contracted(prob,1e9,alpha.*identity;
            z0=50/alpha^2,method=:ufft,mx=32,my=32,surface_zs=.2,
            rtol=1e-9,maxiter=1000,memory=50,precondition)
        @test fft.currents≈alpha.*dense.currents rtol=1e-8
        @test fft.s≈dense.s rtol=1e-8
        @test maximum(fft.raw.relative_residuals)<=1e-9
        @test all(i->1<i<=1000,fft.raw.iterations)
        for q in 1:2
            source=alpha.*view(B,:,q)
            physical=norm((dense.raw.z_mom*view(fft.currents,:,q)-source)./weights)/
                norm(source./weights)
            @test physical<=1e-9
        end
    end
    @test_throws ErrorException solve_planar_contracted(prob,1e9,1e-16.*identity;
        method=:ufft,mx=32,my=32,surface_zs=.2,rtol=1e-9,maxiter=1)
end

@testset "contracted solve: physical layer sources and voltage residuals" begin
    model=_contracted_return_fixture();prob=model.problem;C=model.contraction
    @test size(C)==(8,2)
    raw=solve_planar(prob,1e9;mx=20,my=18,surface_zs=.1)
    reduced=solve_planar_contracted(prob,1e9,C;z0=model.z0,mx=20,my=18,surface_zs=.1)
    fft=solve_planar_contracted(prob,1e9,C;z0=model.z0,mx=20,my=18,
        surface_zs=.1,method=:ufft,memory=100,rtol=1e-10)
    @test reduced isa PlanarContractedResult
    @test reduced.raw isa PlanarSourceResult
    @test size(reduced.raw.currents,2)==2
    @test reduced.raw.currents===reduced.currents
    @test reduced.y≈transpose(C)*raw.y*C rtol=2e-10
    @test reduced.currents≈raw.currents*C rtol=2e-10
    @test reduced.y≈transpose(reduced.y) rtol=2e-10
    @test opnorm(reduced.s)<=1+1e-10
    @test fft.y≈reduced.y rtol=2e-10
    @test fft.s≈reduced.s rtol=2e-10
    @test maximum(fft.raw.relative_residuals)<=1e-10
    @test maximum(fft.raw.galerkin_relative_residuals)>1e-10
    B=zeros(ComplexF64,size(reduced.currents))
    weights=[DiffMoM._planar_port_weight(prob.basis,b) for b in eachindex(prob.basis.kind)]
    for b in eachindex(prob.basis.kind)
        p=prob.basis.port[b];p==0 && continue
        B[b,:].=-DiffMoM._planar_port_sign(prob.ports[p])*weights[b].*C[p,:]
    end
    @test cond(Diagonal(inv.(weights))*raw.z_mom*Diagonal(inv.(weights)))<1e5
    for q in axes(B,2)
        residual=fft.raw.operator*fft.currents[:,q]-B[:,q]
        @test norm(residual./weights)/norm(B[:,q]./weights)≈fft.raw.relative_residuals[q] rtol=1e-14
        @test norm(residual)/norm(B[:,q])≈fft.raw.galerkin_relative_residuals[q] rtol=1e-14
        @test norm((raw.z_mom*reduced.currents[:,q]-B[:,q])./weights)/norm(B[:,q]./weights)<1e-10
    end
    waves=ComplexF64[1,.2im];V=sqrt.(model.z0).*(waves+reduced.s*waves)
    maps=planar_current_maps(reduced;incident_waves=waves)
    direct=planar_current_maps(raw;voltages=C*V)
    @test maps[1].jx≈direct[1].jx rtol=2e-10
    @test maps[end].jz≈direct[end].jz rtol=2e-10
    nested=planar_contract_ports(reduced.raw,reshape([1.,-1.],2,1);z0=100)
    @test planar_current_maps(nested)[1].jx≈
        planar_current_maps(reduced.raw;voltages=[1.,-1.])[1].jx rtol=1e-13
    lean=solve_planar_contracted(prob,1e9,C;z0=model.z0,mx=20,my=18,
        surface_zs=.1,retain_matrix=false)
    @test lean.raw.z_mom===nothing
    @test lean.s≈reduced.s rtol=1e-13
    @test_throws ArgumentError solve_planar_contracted(prob,1e9,C;max_bytes=1)
    @test_throws ArgumentError solve_planar_contracted(prob,1e9,zeros(size(C)))
    @test_throws ArgumentError solve_planar_contracted(prob,1e9,fill(big"1e1000",size(C)))
    @test_throws ArgumentError solve_planar_contracted(prob,1e9,C;method=:unknown)
    @test_throws ArgumentError solve_planar_contracted(prob,1e9,C;z0=[50.,-1.])
    @test_throws ArgumentError planar_current_maps(reduced;max_bytes=1)
    @test_throws ArgumentError planar_current_maps(reduced.raw;voltages=[NaN,1.])
end

@testset "contracted solve: nonport volume-profile trace normalization" begin
    grid=CellGrid(3e-3,2e-3,3,2)
    stack=PlanarStackup([PlanarLayer(2.,1.,.4e-3),PlanarLayer(1.,1.,.8e-3)],
        TERM_GND,TERM_GND,grid.a,grid.b)
    sheet=sheet_level(1,3,2);sheet.mask.=true
    sheet.connect_west.=true;sheet.connect_east.=true
    volume=vol_level(2,3,2);volume.mask[2,1:2].=true
    prob=build_planar_problem(stack,grid,[sheet],
        [PlanarPort(1,:west,1:2,50),PlanarPort(1,:east,1:2,50)];vols=[volume])
    C=Matrix{Float64}(I,2,2)
    raw=solve_planar(prob,5e9;mx=12,my=10,surface_zs=.1,volume_sigma=4e7)
    fft=solve_planar_contracted(prob,5e9,C;method=:ufft,mx=12,my=10,
        surface_zs=.1,volume_sigma=4e7,rtol=1e-10,memory=100)
    @test fft.s≈raw.s rtol=1e-8
    volume_basis=findall(b->DiffMoM._is_vol_kind(prob.basis.kind[b]),eachindex(prob.basis.kind))
    @test !isempty(volume_basis)
    # The implemented volume profile is 1/h, and its coefficient is a
    # sheet-equivalent current. Its integrated trace is width*h*(1/h).
    @test all(b->fft.raw.basis_scale[b]==inv(prob.basis.width[b]),volume_basis)
    @test maximum(fft.raw.relative_residuals)<=1e-10
    @test all(isfinite,planar_current_maps(fft)[2].jy)
end

function _contracted_residual_allocations!(buffer,source,weights,operator,x)
    DiffMoM._planar_source_residuals!(buffer,source,weights,operator,x)
    return @allocated DiffMoM._planar_source_residuals!(buffer,source,weights,operator,x)
end

@testset "contracted solve: weighted residual workspace has no repeated allocations" begin
    model=_contracted_return_fixture();prob=model.problem;C=model.contraction
    weights=[DiffMoM._planar_port_weight(prob.basis,b) for b in eachindex(prob.basis.kind)]
    for method in (:dense,:ufft)
        result=solve_planar_contracted(prob,1e9,C;z0=model.z0,mx=20,my=18,
            surface_zs=.1,method,memory=100,rtol=1e-10)
        source=zeros(ComplexF64,size(result.currents,1))
        DiffMoM._planar_contracted_rhs!(source,prob,C,1)
        operator=method===:dense ? result.raw.z_mom : result.raw.operator
        x=method===:dense ? view(result.currents,:,1) : copy(view(result.currents,:,1))
        saved_source=copy(source);saved_x=copy(x);saved_currents=copy(result.currents)
        @test _contracted_residual_allocations!(similar(source),source,weights,operator,x)==0
        @test source==saved_source
        @test x==saved_x
        @test result.currents==saved_currents
    end
end

function _contracted_compensated_residual_allocations!(buffer,source,Z,x,imaginary)
    DiffMoM._planar_source_compensated_residual!(buffer,source,Z,x,imaginary)
    return @allocated DiffMoM._planar_source_compensated_residual!(buffer,source,Z,x,imaginary)
end

@testset "contracted solve: compensated residual preserves complex loss and source inputs" begin
    for scale in (1e-200,1.,1e200),scenario in (:imaginary,:complex,:complex_rhs)
        imaginary=scenario!==:complex
        Z=scale.*(imaginary ? ComplexF64[1im 1im 1im;2im -1im 2im] :
            ComplexF64[.25+.5im 1.25-.75im .25+.5im;2-3im -1+2im 2-3im])
        x=ComplexF64[1e16im,1im,-1e16im]
        source=scale.*fill(scenario===:complex_rhs ? .25+.5im : .25+0im,2)
        saved=(copy(Z),copy(x),copy(source));buffer=similar(source)
        expected=setprecision(BigFloat,256) do
            Complex{BigFloat}.(Z)*Complex{BigFloat}.(x)-Complex{BigFloat}.(source)
        end
        result=DiffMoM._planar_source_compensated_residual!(buffer,source,Z,x,imaginary)
        @test result===buffer
        @test buffer≈ComplexF64.(expected) rtol=1e-14
        @test Z==saved[1] && x==saved[2] && source==saved[3]
        @test _contracted_compensated_residual_allocations!(buffer,source,Z,x,imaginary)==0
    end
end

@testset "contracted solve: retained dense refinement improves original physical residual" begin
    model=_contracted_return_fixture();prob=model.problem;C=model.contraction
    weights=[DiffMoM._planar_port_weight(prob.basis,b) for b in eachindex(prob.basis.kind)]
    for surface_zs in (0.,.1)
        reference=solve_planar_contracted(prob,1e9,C;z0=model.z0,mx=20,my=18,surface_zs)
        raw=reference.raw;Z=raw.z_mom;X=(1+1e-3).*reference.currents
        sources=zeros(ComplexF64,size(X))
        for b in axes(sources,1)
            p=prob.basis.port[b];p==0 && continue
            sources[b,:].=-DiffMoM._planar_port_sign(prob.ports[p])*weights[b].*C[p,:]
        end
        physical_norm(q)=norm((Z*view(X,:,q)-view(sources,:,q))./weights)/norm(view(sources,:,q)./weights)
        voltage=[physical_norm(q) for q in axes(X,2)]
        galerkin=[norm(Z*view(X,:,q)-view(sources,:,q))/norm(view(sources,:,q)) for q in axes(X,2)]
        initial=copy(voltage);saved_Z=copy(Z);saved_F=copy(raw.lu_fact.factors);saved_C=copy(C)
        saved_reference=copy(reference.currents)
        @test minimum(initial)>1e-4
        DiffMoM._planar_dense_source_refine!(X,zeros(ComplexF64,size(X,1)),zeros(ComplexF64,size(X,1)),
            weights,Z,raw.lu_fact,raw.basis_scale,C,prob,1e-9,voltage,galerkin)
        @test maximum(voltage)<=1e-9
        @test all(q->physical_norm(q)<=1e-9,axes(X,2))
        @test X≈reference.currents rtol=1e-10
        @test Z==saved_Z && raw.lu_fact.factors==saved_F && C==saved_C
        @test reference.currents==saved_reference
        @test all(q->isapprox(physical_norm(q),voltage[q];rtol=1e-14),axes(X,2))
    end
end
