module PlanarContractedSourceAccuracyTests
using DiffMoM, Test, LinearAlgebra, TOML

fixture=joinpath(@__DIR__,"fixtures","native_volume_sheet_selector","triangle_hollow","project.son")

Q=Rational{BigInt}
exact(A)=Complex{Q}.(A)
target=DiffMoM._PLANAR_DENSE_VOLTAGE_RTOL
model=sonnet_planar_problem(read_sonnet_project(fixture);freq=1e6,
    grid=(20,20),_materials=true,_details=true)
prob=model.problem
C=Matrix{Float64}(I,length(prob.ports),length(prob.ports))
rows=Dict{String,Any}[]

@testset "contracted source: caller targets and exact stored equations" begin
    # Existing physical target, its squared accuracy requirement, and the
    # Float64 machine resolution exercise caller choices without an oracle
    # precision or a new scientific threshold.
    for rtol in (target,target^2,eps(Float64)),method in (:dense,:dense_fft),retain in (false,true)
        result=solve_planar_contracted(prob,1e6,C;mx=40,my=40,method,
            retain_matrix=retain,rtol,surface_zs=model.sheet_zs,
            via_sigma=model.via_sigma,z0=model.z0)
        raw=result.raw
        original=method===:dense ? assemble_planar_z(prob.stack,prob.grid,prob.sheets,prob.basis,
            result.omega;vias=prob.vias,vols=prob.vols,mx=40,my=40,
            surface_zs=model.sheet_zs,via_sigma=model.via_sigma) :
            assemble_planar_z_ufft(prob,1e6;mx=40,my=40,
                surface_zs=model.sheet_zs,via_sigma=model.via_sigma)
        sources=zeros(ComplexF64,size(result.currents))
        for q in axes(sources,2)
            DiffMoM._planar_contracted_rhs!(view(sources,:,q),prob,C,q)
        end
        weights=Q.([DiffMoM._planar_port_weight(prob.basis,b) for b in axes(sources,1)])
        residual=exact(original)*exact(result.currents)-exact(sources)
        errors=[sum(abs2,residual[:,q]./weights)/sum(abs2,exact(sources[:,q])./weights)
            for q in axes(sources,2)]
        @test all(e->e<=Q(rtol)^2,errors)
        @test all(e->isfinite(e) && e<=rtol,raw.relative_residuals)
        @test all(isfinite,raw.galerkin_relative_residuals)
        @test retain ? raw.z_mom==original : raw.z_mom===nothing
        @test raw.currents===result.currents && raw.contraction==C
        if eltype(raw.lu_fact.factors)!==ComplexF64
            @test all(isone,raw.basis_scale)
        end
        push!(rows,Dict("rtol"=>rtol,"method"=>string(method),"retain"=>retain,
            "maximum_exact_relative_squared"=>Float64(maximum(errors)),
            "current_type"=>string(eltype(result.currents))))
    end
end

result=solve_planar_contracted(prob,1e6,C;mx=40,my=40,
    surface_zs=model.sheet_zs,via_sigma=model.via_sigma,z0=model.z0)
raw=result.raw
@testset "contracted source: original explicit constructor parameters" begin
    ordinary=ComplexF64.(raw.currents)
    legacy=PlanarSourceResult{typeof(raw.lu_fact),typeof(raw.operator)}(
        raw.problem,raw.freq,raw.omega,raw.contraction,raw.z0,raw.z_mom,
        raw.lu_fact,raw.operator,raw.basis_scale,ordinary,raw.y,raw.s,
        raw.iterations,raw.relative_residuals,raw.galerkin_relative_residuals)
    @test legacy isa PlanarSourceResult
    @test eltype(legacy.currents)===ComplexF64
    @test legacy.currents==ordinary
    @test legacy.lu_fact===raw.lu_fact && legacy.basis_scale===raw.basis_scale
    @test legacy.problem===raw.problem && legacy.contraction===raw.contraction
    @test legacy.z_mom===raw.z_mom
    @test legacy.operator===raw.operator
end
saved_X=deepcopy(raw.currents);saved_C=copy(raw.contraction)
saved_factor=deepcopy(raw.lu_fact.factors)
v=ComplexF64[1+im,2-im]
coefficients=DiffMoM._planar_current_product(raw.currents,v)
expected_maps=planar_current_maps(raw;voltages=v)
radiation=planar_radiation_stack(prob.stack;top=TERM_SPACE)
expected_field=planar_farfield(prob,coefficients,1e6;theta=[.31],phi=[.49],radiation_stack=radiation)

@testset "contracted source: owned precision maps radiation and nested contracts" begin
    before=(precision(BigFloat),rounding(BigFloat))
    # A Float32 caller precision intentionally loses Float64 significand
    # information; source operations must retain their owned wider scalars.
    setprecision(BigFloat,precision(Float32)) do
        setrounding(BigFloat,RoundDown) do
            maps=planar_current_maps(raw;voltages=v)
            for (actual,expected) in zip(maps,expected_maps),component in (:jx,:jy,:jz)
                @test getproperty(actual,component)==getproperty(expected,component)
            end
            field=planar_farfield(result;voltages=v,theta=[.31],phi=[.49],radiation_stack=radiation)
            # The direct API explicitly narrows wide coefficients to
            # Float64; Julia's conversion honors the caller rounding mode.
            # Compare the public wrapper with the direct path under that
            # same mode, while verifying owned source products separately.
            same_caller=planar_farfield(prob,coefficients,1e6;
                theta=[.31],phi=[.49],radiation_stack=radiation)
            @test DiffMoM._planar_current_product(raw.currents,v)==coefficients
            @test field.etheta==same_caller.etheta && field.ephi==same_caller.ephi
            nested=planar_contract_ports(raw,reshape([1.,-1.],2,1);z0=sum(real,raw.z0))
            @test nested.currents==DiffMoM._planar_current_product(raw.currents,reshape([1.,-1.],2,1))
            @test all(x->precision(real(x))==DiffMoM._planar_current_precision(raw.currents),nested.currents)
        end
    end
    @test before==(precision(BigFloat),rounding(BigFloat))
    @test raw.currents==saved_X && raw.contraction==saved_C && raw.lu_fact.factors==saved_factor
    @test_throws ArgumentError planar_current_maps(raw;voltages=v,max_bytes=1)
    @test_throws ArgumentError solve_planar_contracted(prob,1e6,C;mx=40,my=40,
        surface_zs=model.sheet_zs,via_sigma=model.via_sigma,max_bytes=1)
end

@testset "contracted source: inherited small source amplitude controls" begin
    for amplitude in (1.,1e-16,1e-18,-1e-16)
        scaled=solve_planar_contracted(prob,1e6,amplitude.*C;mx=40,my=40,
            surface_zs=model.sheet_zs,via_sigma=model.via_sigma,z0=model.z0)
        @test maximum(scaled.raw.relative_residuals)<=target
        # The original API's nearby source-amplitude tests use this bound.
        @test scaled.y≈amplitude^2 .* result.y rtol=2e-10
    end
end


using DiffMoM,LinearAlgebra,Test,TOML
triangle=fixture
coarse=joinpath(@__DIR__,"fixtures","native_box_port_attachment","cases","attachment__baseline","project.son")

@testset "contracted source: measured owned output and compact storage" begin
    # Both shapes retain their already qualified native assembly controls.
    for (name,fixture,frequency,grid,modes) in (("triangle",triangle,1e6,(20,20),(40,40)),
            ("coarse_attachment",coarse,1e9,(8,8),(8,8))),method in (:dense,:dense_fft)
        model=sonnet_planar_problem(read_sonnet_project(fixture);freq=frequency,
            grid,_materials=true,_details=true)
        prob=model.problem;C=Matrix{Float64}(I,length(prob.ports),length(prob.ports))
        solve_one(retain,budget=DiffMoM._DEFAULT_MAX_DENSE_PAYLOAD_BYTES)=solve_planar_contracted(
            prob,frequency,C;mx=modes[1],my=modes[2],method,retain_matrix=retain,
            surface_zs=model.sheet_zs,via_sigma=model.via_sigma,z0=model.z0,max_bytes=budget)
        retained=solve_one(true);compact=solve_one(false)
        @test compact.raw.z_mom===nothing
        @test maximum(retained.raw.relative_residuals)<=DiffMoM._PLANAR_DENSE_VOLTAGE_RTOL
        @test maximum(compact.raw.relative_residuals)<=DiffMoM._PLANAR_DENSE_VOLTAGE_RTOL
        @test compact.s≈retained.s rtol=2e-14
        @test compact.currents≈retained.currents rtol=2e-14
        for retain in (false,true)
            result=retain ? retained : compact
            # A budget below even the owned output cannot contain the
            # operation. Exclude the outer matrix object from this observed
            # bound; solver budgets explicitly concern owned numeric payload.
            output_bytes=Base.summarysize(result.currents)-sizeof(result.currents)
            @test output_bytes>0
            @test_throws ArgumentError solve_one(retain,output_bytes-1)
            solve_one(retain) # warm the exact call shape before measurement
            allocated=@allocated solve_one(retain)
            @test allocated>=sizeof(result.currents)
            factor=result.raw.lu_fact
            push!(rows,Dict("case"=>name,"method"=>string(method),"retain"=>retain,
                "basis_count"=>size(result.currents,1),"port_count"=>size(result.currents,2),
                "current_type"=>string(eltype(result.currents)),
                "factor_type"=>string(eltype(factor.factors)),
                "owned_current_summary_bytes"=>Base.summarysize(result.currents),
                "owned_factor_summary_bytes"=>Base.summarysize(factor),
                "warmed_total_allocated_bytes"=>allocated,
                "maximum_physical_relative"=>maximum(result.raw.relative_residuals),
                "s_real"=>vec(real.(result.s)),"s_imag"=>vec(imag.(result.s))))
        end
    end
end

end
