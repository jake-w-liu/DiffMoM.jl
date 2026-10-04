using DiffMoM,Test,LinearAlgebra,Random

@testset "congruent triangle refinement retains exact FFT families" begin
    a=b=1e-3
    mesh=PlanarConformalMesh([0. a a 0. a/2;.375b .25b .75b .625b b/2],[1 2 3 4;2 3 4 1;5 5 5 5])
    stack=PlanarStackup([PlanarLayer(1.,1.,.5e-3),PlanarLayer(1.,1.,.5e-3)],TERM_GND,TERM_GND,a,b)
    ports=[PlanarConformalPort(1,:west,(.375b,.625b)),PlanarConformalPort(1,:east,(.25b,.75b))]
    rng=MersenneTwister(977)
    @test planar_refine_conformal_uniform(mesh,0)===mesh
    for level in (2,3)
        refined=planar_refine_conformal_uniform(mesh,level)
        @test length(refined.areas)==4*4^level
        @test sum(refined.areas)≈sum(mesh.areas) rtol=1e-14
        p=PlanarConformalProblem(stack,refined,ports);nx,ny=2^(level+1),2^(level+3)
        A=planar_conformal_ufft_operator(p,10e9;nx,ny,mx=19,my=23,surface_zs=.1)
        @test length(A.families)<=24
        Z=assemble_planar_conformal_z(p,10e9;mx=19,my=23,surface_zs=.1)
        x=randn(rng,ComplexF64,A.n)
        @test A*x≈Z*x rtol=5e-13
        @test DiffMoM._planar_conformal_ufft_diagonal(A)≈diag(Z) rtol=5e-13
    end
    @test_throws ArgumentError planar_refine_conformal_uniform(mesh,32;max_triangles=typemax(Int))
    @test_throws ArgumentError planar_refine_conformal_uniform(mesh,3;max_triangles=20)
    @test_throws ArgumentError planar_refine_conformal_uniform(mesh,3;max_bytes=1)
    @test_throws ArgumentError planar_refine_conformal_uniform(mesh,-1)
    v,box=planar_normalize_polygon([(0.,.375b),(a,.25b),(a,.75b),(0.,.625b)])
    poly=PlanarPolygon("taper",1,"film","",v,box)
    layout=build_planar_conformal_layout(stack,[poly],ports;edge_size=2a,interior_size=2a,metals=Dict("film"=>f->.1))
    refined=planar_refine_conformal_uniform(layout,2)
    @test all(==(1),refined.triangle_materials)
    @test refined.material_names==["film"]
    @test refined.materials===layout.materials
    @test length(refined.triangle_materials)==16length(layout.triangle_materials)
end
