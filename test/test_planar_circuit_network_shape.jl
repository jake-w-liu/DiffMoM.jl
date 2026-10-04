using DiffMoM,Test

struct _OversizedCircuitResponse <: AbstractMatrix{Float64}
    reads::Base.RefValue{Int}
end
Base.size(::_OversizedCircuitResponse)=(1024,1024)
Base.getindex(response::_OversizedCircuitResponse,i::Int,j::Int)=(response.reads[]+=1;0.)

@testset "Circuit network dimensions precede conversion and references" begin
    reads=Ref(0);response=_OversizedCircuitResponse(reads)
    references=Ref(0);provider=f->(references[]+=1;50+10im)
    @testset "Constant malformed response is transactional" begin
        c=PlanarCircuit(2,[1,2]);circuit_add_rlc!(c,1,2;r=100.)
        original=copy(c.elements)
        for format in (:s,:y,:z)
            @test_throws ArgumentError circuit_add_network!(c,[1,2],response;format,z0=provider)
            @test reads[]==0 && references[]==0
            @test c.elements==original && c.nnodes==2
            allocation=@allocated try circuit_add_network!(c,[1,2],response;format,z0=provider) catch e
                e isa ArgumentError || rethrow()
            end
            @test allocation<10000
        end
    end
    @testset "Provider matrix dimensions precede conversion and local references" begin
        for format in (:s,:y,:z)
            calls=Ref(0);c=PlanarCircuit(2,[1,2])
            circuit_add_network!(c,[1,2],f->(calls[]+=1;response);format,z0=provider)
            @test_throws ArgumentError solve_planar_circuit(c,1e9;max_bytes=1)
            @test calls[]==0 && reads[]==0 && references[]==0
            @test_throws ArgumentError solve_planar_circuit(c,1e9)
            @test calls[]==1 && reads[]==0 && references[]==0
            allocation=@allocated try solve_planar_circuit(c,1e9) catch e
                e isa ArgumentError || rethrow()
            end
            @test allocation<100000
        end
    end
end
