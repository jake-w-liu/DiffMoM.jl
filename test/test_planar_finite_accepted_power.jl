module FiniteAcceptedPowerTests
using DiffMoM,LinearAlgebra,Test
@testset "Radiation rejects nonfinite accepted power before gain omission" begin
    source_root=normpath(joinpath(@__DIR__,".."))
    project=load_planar_project(joinpath(source_root,"examples/planar_project_line.toml"))
    project.data["ports"]=project.data["ports"][1:1]
    frequency=1e9
    model=planar_project_layout(project;freq=frequency);prob=model.layout.problem
    nb=planar_basis_count(prob.basis);pbs=DiffMoM._port_basis_indices(prob.basis,1)
    signs=[DiffMoM._planar_port_sign(prob.ports[1])]
    rhs=DiffMoM._planar_dense_source_rhs!(zeros(ComplexF64,nb,1),prob)
    radiation=PlanarStackup(prob.stack.layers,TERM_SPACE,TERM_SPACE,prob.stack.a,prob.stack.b)
    # Put the voltage and the resulting current below their finite range
    # limits, but their power product beyond it. Leave one significand of
    # exponent headroom for field evaluation; intensity squares the current.
    top=exponent(floatmax(Float64));bits=precision(Float64)
    voltage_exponent=top-bits
    current_exponent=top÷2-bits
    large=ldexp(1.0,voltage_exponent)
    response=ldexp(1.0,current_exponent-voltage_exponent)
    for polarity in (1,-1)
        Z=Matrix{ComplexF64}(I,nb,nb)
        for p in pbs[1];Z[p,p]=rhs[p,1]/(polarity*response);end
        F=lu(Z);X=F\rhs;Y=DiffMoM._planar_dense_port_y(prob,pbs,signs,X)
        S=planar_y_to_s(Y,model.z0)
        em=PlanarResult(prob,complex(2pi*frequency),complex(frequency),Z,F,X,Y,S)
        result=PlanarProjectResult(project,model,em,nothing,frequency,model.port_names,model.z0,Y,S)
        @test norm(Z*X-rhs)/norm(rhs)<=1e-9
        for amplitude in (1.0,large)
            voltage=ComplexF64[amplitude];current=Y*voltage
            @test all(isfinite,voltage) && all(isfinite,current)
            power=.5real(dot(voltage,current))
            if !isfinite(power)
                for network in (em,result)
                    @test_throws ArgumentError planar_farfield(network;voltages=voltage,radiation_stack=radiation,
                        theta=[Float64(pi/4)],phi=[0.0])
                end
            elseif polarity<0
                @test_throws ArgumentError planar_farfield(em;voltages=voltage,radiation_stack=radiation,
                    theta=[Float64(pi/4)],phi=[0.0])
            else
                for network in (em,result)
                    pattern=planar_farfield(network;voltages=voltage,radiation_stack=radiation,
                        theta=[Float64(pi/4)],phi=[0.0])
                    @test pattern.accepted_power!==nothing && isapprox(pattern.accepted_power,power)
                    @test all(isfinite,pattern.etheta) && all(isfinite,pattern.ephi) && all(isfinite,pattern.intensity)
                end
            end
        end
    end
end
end
