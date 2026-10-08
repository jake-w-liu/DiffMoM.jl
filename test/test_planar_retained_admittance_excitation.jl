module RetainedAdmittanceExcitationTests
using DiffMoM,LinearAlgebra,Random,Test
source_root = normpath(joinpath(@__DIR__,".."))
@testset "Retained admittance preserves physical far fields after S quantization" begin
project = load_planar_project(joinpath(source_root,"examples/planar_project_line.toml"))
project.data["ports"] = project.data["ports"][1:1]
frequency = 1e9
model = planar_project_layout(project;freq=frequency)
prob = model.layout.problem
nb = planar_basis_count(prob.basis)
pbs = DiffMoM._port_basis_indices(prob.basis,1)
signs = [DiffMoM._planar_port_sign(prob.ports[1])]
rhs = DiffMoM._planar_dense_source_rhs!(zeros(ComplexF64,nb,1),prob)
refs = model.z0
unit_current = zeros(ComplexF64,nb,1)
for p in pbs[1]; unit_current[p,1] = 1; end
unit_admittance = real(DiffMoM._planar_dense_port_y(prob,pbs,signs,unit_current)[1])
radiation = PlanarStackup(prob.stack.layers,TERM_SPACE,TERM_SPACE,prob.stack.a,prob.stack.b)
for response in (inv(sqrt(eps(Float64))),inv(eps(Float64)),inv(eps(Float64)^2))
    g = response/(2real(refs[1])*unit_admittance)
    Z = Matrix{ComplexF64}(I,nb,nb)
    for p in pbs[1]; Z[p,p] = rhs[p,1]/g; end
    F = lu(Z); X = F\rhs
    Y = DiffMoM._planar_dense_port_y(prob,pbs,signs,X)
    S = planar_y_to_s(Y,refs)
    em = PlanarResult(prob,complex(2pi*frequency),complex(frequency),Z,F,X,Y,S)
    result = PlanarProjectResult(project,model,em,nothing,frequency,model.port_names,refs,Y,S)
    # Independently solve the physical Kurokawa equation for one real port.
    wanted_voltage = ComplexF64[2sqrt(real(refs[1]))/(1+refs[1]*Y[1])]
    physical = planar_power_waves(wanted_voltage,Y*wanted_voltage;z0=refs)
    direct = planar_farfield(prob,X*wanted_voltage,frequency;radiation_stack=radiation,
        theta=[Float64(pi/4)],phi=[0.0],accepted_power=physical.accepted_power)
    for kind in (:em_incident,:project_incident,:project_voltage)
        pattern = if kind === :em_incident
            planar_farfield(em;incident_waves=ComplexF64[1],radiation_stack=radiation,theta=[Float64(pi/4)],phi=[0.0])
        elseif kind === :project_incident
            planar_farfield(result;incident_waves=ComplexF64[1],radiation_stack=radiation,theta=[Float64(pi/4)],phi=[0.0])
        else
            planar_farfield(result;voltages=wanted_voltage,radiation_stack=radiation,theta=[Float64(pi/4)],phi=[0.0])
        end
        @test isapprox(pattern.etheta,direct.etheta;rtol=5e-11,atol=0)
        @test isapprox(pattern.ephi,direct.ephi;rtol=5e-11,atol=0)
        @test pattern.accepted_power !== nothing && isapprox(pattern.accepted_power,physical.accepted_power)
        @test isapprox(physical.a,ComplexF64[1])
        @test norm(Z*X-rhs)/norm(rhs) <= 1e-9
    end
end

end
@testset "Project accepted power uses retained terminal V and I" begin
project = load_planar_project(joinpath(source_root,"examples/planar_project_line.toml"))
project.data["ports"] = project.data["ports"][1:1]
frequency = 1e9
model = planar_project_layout(project;freq=frequency)
prob = model.layout.problem
nb = planar_basis_count(prob.basis)
pbs = DiffMoM._port_basis_indices(prob.basis,1)
signs = [DiffMoM._planar_port_sign(prob.ports[1])]
rhs = DiffMoM._planar_dense_source_rhs!(zeros(ComplexF64,nb,1),prob)
refs = model.z0
unit_current = zeros(ComplexF64,nb,1)
for p in pbs[1]; unit_current[p,1] = 1; end
unit_admittance = real(DiffMoM._planar_dense_port_y(prob,pbs,signs,unit_current)[1])
radiation = PlanarStackup(prob.stack.layers,TERM_SPACE,TERM_SPACE,prob.stack.a,prob.stack.b)
for response in (sqrt(eps(Float64)),eps(Float64),2eps(Float64),eps(Float64)^2)
    g = response/(2real(refs[1])*unit_admittance)
    Z = Matrix{ComplexF64}(I,nb,nb)
    for p in pbs[1]; Z[p,p] = rhs[p,1]/g; end
    F = lu(Z); X = F\rhs
    Y = DiffMoM._planar_dense_port_y(prob,pbs,signs,X)
    S = planar_y_to_s(Y,refs)
    em = PlanarResult(prob,complex(2pi*frequency),complex(frequency),Z,F,X,Y,S)
    loaded_project = load_planar_project(joinpath(source_root,"examples/planar_project_line.toml"))
    loaded_project.data["ports"] = loaded_project.data["ports"][1:1]
    loaded_project.data["ports"][1]["external"] = true
    loaded_project.data["components"] = [Dict("type"=>"resistor","name"=>"positive_shunt","value"=>inv(real(Y[1])),"ports"=>[1,0])]
    loaded_model = planar_project_layout(loaded_project;freq=frequency)
    budget = DiffMoM._default_max_dense_payload_bytes()
    circuit = DiffMoM._project_circuit(loaded_model,frequency,nothing,budget)
    circuit_add_em!(circuit,loaded_model.port_terminals,em;z0=refs)
    loaded = solve_planar_circuit(circuit,frequency;floating_gauge=:auto)
    for kind in (:retained_y,:loaded_mna,:s_only)
        mdl = kind === :loaded_mna ? loaded_model : model
        prj = kind === :loaded_mna ? loaded_project : project
        circ = kind === :loaded_mna ? loaded : nothing
        external_y = kind === :loaded_mna ? loaded.y : kind === :s_only ? nothing : Y
        external_s = kind === :loaded_mna ? loaded.s : S
        result = PlanarProjectResult(prj,mdl,em,circ,frequency,mdl.port_names,refs,external_y,external_s)
        waves = ComplexF64[1]
        if circ === nothing
            source_voltage = external_y === nothing ?
                planar_wave_voltages(external_s,waves;z0=refs) :
                ComplexF64[2sqrt(real(refs[1]))/(1+refs[1]*external_y[1])]
            actual_model_power = planar_power_waves(source_voltage,Y*source_voltage;z0=refs).accepted_power
        else
            node_voltage = circ.voltages*waves
            positive,negative = mdl.port_terminals[mdl.external[1]]
            voltage = (iszero(positive) ? 0.0im : node_voltage[positive]) - (iszero(negative) ? 0.0im : node_voltage[negative])
            actual_model_power = planar_power_waves([voltage],circ.currents*waves;z0=refs).accepted_power
            source_voltage = transpose(mdl.source_incidence)*view(node_voltage,1:length(mdl.node_names))
        end
        wave_power = real(dot(waves,waves)-dot(external_s*waves,external_s*waves))/2
        expected = kind === :s_only ? wave_power : actual_model_power
        accepted = expected > 0 ? expected : nothing
        pattern = planar_farfield(result;incident_waves=waves,radiation_stack=radiation,theta=[Float64(pi/4)],phi=[0.0])
        direct = planar_farfield(prob,X*source_voltage,frequency;radiation_stack=radiation,theta=[Float64(pi/4)],phi=[0.0],accepted_power=accepted)
        residual = norm(Z*X-rhs)/norm(rhs)
        retained = accepted === nothing ? pattern.accepted_power === nothing :
            pattern.accepted_power !== nothing && isapprox(pattern.accepted_power,accepted)
        @test retained
        @test isapprox(pattern.etheta,direct.etheta;rtol=5e-11,atol=0)
        @test isapprox(pattern.ephi,direct.ephi;rtol=5e-11,atol=0)
        @test residual <= 1e-9
    end
end

end
@testset "Retained admittance excitation supports coupled complex-reference ports" begin
    rng = MersenneTwister(10326) # Preserve the original power-wave fixture.
    for n in 1:5
        r,x = randn(rng,n,n),randn(rng,n,n)
        base = .01*(r'*r+I)+im*.01*(x+x')
        refs = ComplexF64.(20 .+80rand(rng,n).+im*(60rand(rng,n).-30))
        a = randn(rng,ComplexF64,n)
        for multiplier in (1.0,inv(sqrt(eps(Float64))),inv(eps(Float64)),inv(eps(Float64)^2))
            Y = multiplier*base
            y_saved,refs_saved,a_saved = copy(Y),copy(refs),copy(a)
            reference,derivative = setprecision(BigFloat,2precision(Float64)) do
                yy,zz,aa = Complex{BigFloat}.(Y),Complex{BigFloat}.(refs),Complex{BigFloat}.(a)
                matrix = Matrix{Complex{BigFloat}}(I,n,n)+Diagonal(zz)*yy
                voltage = matrix\(2sqrt.(real.(zz)).*aa)
                slope = -(matrix\(Diagonal(zz)*yy*voltage))
                ComplexF64.(voltage),ComplexF64.(slope)
            end
            voltage = @inferred DiffMoM._planar_wave_voltage_admittance(Y,a,refs)
            incident = @inferred DiffMoM._planar_incident_admittance(Y,voltage,refs)
            @test isapprox(voltage,reference;rtol=3e-14,atol=0)
            @test isapprox(incident,a;rtol=3e-14,atol=0)
            step = cbrt(eps(Float64)) # Second-order central difference balance.
            slope = (DiffMoM._planar_wave_voltage_admittance((1+step)*Y,a,refs)-
                DiffMoM._planar_wave_voltage_admittance((1-step)*Y,a,refs))/(2step)
            @test isapprox(slope,derivative)
            @test Y == y_saved && refs == refs_saved && a == a_saved
            @test all(isfinite,voltage) && all(isfinite,incident) && all(isfinite,slope)
        end
    end
end

@testset "Retained admittance excitation validates inputs before solving" begin
    Y = ComplexF64[1;;]
    a,refs = ComplexF64[1],ComplexF64[50]
    reference = ComplexF64[2sqrt(50.0)/51]
    for impedances in (50.0,(50.0,),[50.0],refs,view(refs,:),BigFloat[50])
        @test isapprox(DiffMoM._planar_wave_voltage_admittance(Y,a,impedances),reference)
        @test isapprox(DiffMoM._planar_incident_admittance(Y,reference,impedances),a)
    end
    @test isapprox(DiffMoM._planar_wave_voltage_admittance(Y,BigFloat[1],refs),reference)
    for helper in (DiffMoM._planar_wave_voltage_admittance,DiffMoM._planar_incident_admittance)
        @test_throws ArgumentError helper(zeros(ComplexF64,0,0),ComplexF64[],ComplexF64[])
        @test_throws ArgumentError helper(zeros(ComplexF64,2,2),a,refs)
        @test_throws ArgumentError helper(ComplexF64[Inf;;],a,refs)
        @test_throws ArgumentError helper(Y,ComplexF64[Inf],refs)
        @test_throws DimensionMismatch helper(Y,a,ComplexF64[])
        @test_throws ArgumentError helper(Y,a,ComplexF64[0])
        @test_throws ArgumentError helper(Y,a,ComplexF64[Inf])
        @test_throws ArgumentError helper(Y,a,ComplexF64[im])
    end
    @test_throws ArgumentError DiffMoM._planar_wave_voltage_admittance(ComplexF64[-inv(50.0);;],a,refs)
    @test_throws ArgumentError DiffMoM._planar_wave_voltage_admittance(Y,BigFloat[big(2)^exponent(floatmax(Float64))*2],refs)
end

end
