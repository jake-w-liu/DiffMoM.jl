using DiffMoM,Test,LinearAlgebra

@testset "Unloaded project ports retain declared circuit wiring" begin
    data=Dict("project"=>Dict("unit"=>"mm"),
        "box"=>Dict("width"=>2.,"length"=>2.),"mesh"=>Dict("nx"=>4,"ny"=>4),
        "stackup"=>Dict("layers"=>[Dict("thickness"=>.5),Dict("thickness"=>.5)]),
        "polygons"=>[Dict("name"=>"line","level"=>1,
            "vertices"=>[[0.,.5],[2.,.5],[2.,1.5],[0.,1.5]])],
        "ports"=>[Dict("name"=>"left","number"=>1,"polygon"=>"line","edge"=>4),
                  Dict("name"=>"right","number"=>2,"polygon"=>"line","edge"=>2)])
    project=planar_project_from_dict(data)
    direct=solve_planar_project(project,1e9;mx=8,my=8,max_bytes=4*1024^2)
    @test direct.circuit===nothing
    @test direct.s==direct.em.s
    for port in project.data["ports"]
        port["nodes"]=["shared","gnd"]
    end
    saved=planar_project_dict(project)
    wired=solve_planar_project(project,1e9;mx=8,my=8,max_bytes=4*1024^2)
    # The two terminal voltages are equal. KCL sees sum(Y) in parallel
    # with the two 50-ohm external wave ports, giving this scalar oracle.
    expected=fill(2/(2+50sum(direct.y)),2,2)-Matrix{ComplexF64}(I,2,2)
    @test wired.circuit!==nothing
    @test wired.s≈expected atol=1e-12
    @test wired.s*[1.,-1.]≈[-1.,1.] atol=1e-12
    @test project.data==saved
    waves=ComplexF64[1,.2im]
    nodes=wired.circuit.voltages*waves
    mapped=planar_current_maps(wired;incident_waves=waves)
    reference=planar_current_maps(wired.em;voltages=fill(nodes[1],2))
    @test mapped[1].jx≈reference[1].jx rtol=1e-12
    @test mapped[1].jy≈reference[1].jy rtol=1e-12
end
