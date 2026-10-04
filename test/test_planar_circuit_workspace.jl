module PlanarCircuitWorkspaceTests
using DiffMoM,Test,LinearAlgebra
const DM=DiffMoM

@testset "Circuit internal network workspace rejects before providers" begin
    n=64
    for format in (:s,:y,:z),dynamic in (false,true),gauge in (:reject,:auto)
        seen=Symbol[]
        matrix=format===:s ? zeros(ComplexF64,n,n) :
            Matrix{ComplexF64}(I,n,n)*(format===:y ? .02 : 50.)
        response=dynamic ? (f->(push!(seen,:response);copy(matrix))) : matrix
        circuit=PlanarCircuit(n,[1,2];z0=f->(push!(seen,:external_reference);50.))
        circuit_add_network!(circuit,collect(1:n),response;format,
            z0=f->(push!(seen,:local_reference);50.))
        @test DM._circuit_network_workspace_payload(circuit)==2sizeof(ComplexF64)*n*n
        @test_throws ArgumentError solve_planar_circuit(circuit,1e9;
            max_bytes=300_000,floating_gauge=gauge)
        @test isempty(seen)
        result=solve_planar_circuit(circuit,1e9;max_bytes=1024^2,floating_gauge=gauge)
        @test maximum(abs,result.s)<1e-12
        @test all(isfinite,result.voltages)
        @test seen==vcat([:external_reference],dynamic ? [:response] : Symbol[],
            format===:s ? [:local_reference] : Symbol[])
    end
    circuit=PlanarCircuit(1,[1]);circuit_add_rlc!(circuit,1,0;r=50.)
    @test DM._circuit_network_workspace_payload(circuit)==0
    @test solve_planar_circuit(circuit,1e9).s≈zeros(ComplexF64,1,1) atol=1e-14
    # Reserve the largest sequential block, rather than every block's peak.
    circuit=PlanarCircuit(3,[1,2])
    circuit_add_network!(circuit,[1],zeros(ComplexF64,1,1))
    circuit_add_network!(circuit,[2,3],zeros(ComplexF64,2,2))
    @test DM._circuit_network_workspace_payload(circuit)==2sizeof(ComplexF64)*2^2
end

@testset "Direct native equations preserve response snapshots" begin
    refs=ComplexF64[30+10im,120-7im,50+3im]
    response=ComplexF64[.1+.2im .4-.1im .1im;.3+.1im -.2+.05im .1;.02im .1 .3-.1im]
    circuit=PlanarCircuit(3,[1,2,3];z0=refs)
    circuit_add_network!(circuit,[1,2,3],response;z0=refs)
    @test solve_planar_circuit(circuit,1e9).s≈response atol=2e-14 rtol=2e-14
    # The source matrix is copied before evaluating local references.
    shared=reshape(ComplexF64[.25+.1im],1,1)
    circuit=PlanarCircuit(1,[1])
    circuit_add_network!(circuit,[1],f->shared;
        z0=f->(shared[1,1]=.75;50.))
    @test only(solve_planar_circuit(circuit,1e9).s)≈.25+.1im atol=1e-14
    @test shared[1,1]==.75
    # Constructor ownership remains distinct from the caller's matrix.
    source=reshape(ComplexF64[.25+.1im],1,1)
    circuit=PlanarCircuit(1,[1]);circuit_add_network!(circuit,[1],source)
    source[1,1]=.75
    @test only(solve_planar_circuit(circuit,1e9).s)≈.25+.1im atol=1e-14
end

function scattering_stamp_allocation(n)
    M=zeros(ComplexF64,2n,2n);H=zeros(ComplexF64,n,n)
    refs=fill(50. + 3im,n);pairs=[(i,0) for i in 1:n];ids=(n+1):2n
    DM._circuit_scattering_stamp!(M,ids,pairs,H,refs)
    return @allocated DM._circuit_scattering_stamp!(M,ids,pairs,H,refs)
end
@testset "Scattering stamps retain only a linear root buffer" begin
    for n in (8,64,128)
        @test scattering_stamp_allocation(n)<=sizeof(Float64)*n+1024
    end
end
end
