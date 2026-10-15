using DiffMoM,Test,LinearAlgebra

@testset "All SPICE instance kinds retain provenance and unique names" begin
    mktempdir() do directory
        library=joinpath(directory,"models.lib")
        write(library,".subckt line a b c d\nw1 a b 0 c d 0 n=2 l=.01 rlgcmodel=m\n"*
            ".model m w modeltype=rlgc n=2 lo=4e-7 1e-7 4e-7 co=1e-10 -2e-11 1e-10\n.ends line\n"*
            ".subckt empty a b\n.ends empty\n")
        native=planar_read_spice(library)
        databank=joinpath(directory,"block.s1p")
        planar_write_touchstone(databank,[1e9],[zeros(ComplexF64,1,1)])
        spectre=joinpath(directory,"models.scs")
        write(spectre,"simulator lang=spectre\nsubckt transformer a b c d\n"*
            "tx (a b c d) transformer ratio=2\nends transformer\n"*
            "subckt measured a b\nnp (a) nport file=\"block.s1p\"\nends measured\n")
        readable=planar_read_spectre(spectre)
        for (source,name,n) in ((native,"line",4),(native,"empty",2),
                (readable,"transformer",4),(readable,"measured",2))
            model=planar_spice_model(source,name)
            circuit=PlanarCircuit(n,collect(1:n))
            circuit_add_spice!(circuit,collect(1:n),model;name="Native")
            @test circuit.spice_instances["native"]===model
            @test DiffMoM._circuit_owned_model_payload(circuit)==model.payload
            saved=(circuit.nnodes,length(circuit.elements),copy(circuit.spice_instances))
            @test_throws ArgumentError circuit_add_spice!(circuit,collect(1:n),model;name="NATIVE")
            @test (circuit.nnodes,length(circuit.elements),circuit.spice_instances)==saved
            # This cap cannot hold two copies of the compiled model. The
            # second instance shares its model and owns only new wiring.
            @test circuit_add_spice!(circuit,collect(1:n),model;name="second",
                max_bytes=2model.payload-1)===circuit
            @test DiffMoM._circuit_owned_model_payload(circuit)==model.payload
            copied=DiffMoM._circuit_copy(circuit)
            @test copied.spice_instances==circuit.spice_instances &&
                copied.spice_instances!==circuit.spice_instances
            name=="empty" && @test solve_planar_circuit(circuit,1e9).s≈Matrix{ComplexF64}(I,n,n)
        end
    end
end
