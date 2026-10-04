using Test, DiffMoM

@testset "Port reference frequency products preserve finite impedances" begin
    for topology in (:series,:parallel),f in (0.,1.,1e308,floatmax(Float64))
        @test PlanarPortImpedance(;topology)(f)==50+0im
    end
    for topology in (:series,:parallel),f in (1e308,floatmax(Float64)),l in (1e-300,-1e-300)
        oracle=ComplexF64(BigFloat(50)+im*(2big(pi)*BigFloat(f)*BigFloat(l)))
        @test PlanarPortImpedance(;l,topology)(f)≈oracle rtol=5eps(Float64)
    end
    for topology in (:series,:parallel),f in (1e308,floatmax(Float64)),c in (1e-300,-1e-300)
        omegaC=2big(pi)*BigFloat(f)*BigFloat(c)
        oracle=ComplexF64(topology===:series ? BigFloat(50)+inv(im*omegaC) :
            inv(inv(BigFloat(50))+im*omegaC))
        @test PlanarPortImpedance(;c,topology)(f)≈oracle rtol=8eps(Float64)
    end
    @test PlanarPortImpedance(c=1e-300,topology=:parallel)(0.)==50+0im
    @test_throws ArgumentError PlanarPortImpedance(c=1e-300)(0.)
    calls=Ref(0);r=f->(calls[]+=1;50.)
    for invalid in (NaN,Inf,-1.,BigFloat("1e1000"),BigFloat("1e-1000"))
        @test_throws ArgumentError PlanarPortImpedance(;r)(invalid)
        @test calls[]==0
    end
    @test PlanarPortImpedance(;r)(1e308)==50+0im && calls[]==1
    @test_throws ArgumentError PlanarPortImpedance(l=1.)(1e308)
    fixture=joinpath(@__DIR__,"fixtures","native_sparam_model_files","native","sparameter.son")
    native=read_sonnet_project(fixture);port=deepcopy(first(native.ports))
    port.values[2:5]=["50","0","1e-291","1e-288"]
    expected=PlanarPortImpedance(l=1e-300,c=1e-300,topology=:parallel)(1e308)
    @test DiffMoM._sonnet_port_reference(native,port,1e308)≈expected rtol=8eps(Float64)
    port.values[4]="1e-320"
    @test_throws ArgumentError DiffMoM._sonnet_port_reference(native,port,1e308)
end
