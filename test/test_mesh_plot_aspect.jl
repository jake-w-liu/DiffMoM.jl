using Test
using DiffMoM

@testset "Mesh figures preserve physical coordinate scale" begin
    mesh = make_bent_slotted_panel(; max_edge=0.05)
    plot = plot_mesh_wireframe(mesh)
    @test plot.layout[:scene][:aspectmode] == "data"
    explicit = plot_mesh_wireframe(mesh;
        xlims=(-0.25, 0.25), ylims=(-0.2, 0.2), zlims=(0.0, 0.2))
    @test explicit.layout[:scene][:aspectmode] == "data"
    @test explicit.layout[:scene][:xaxis][:range] == [-0.25, 0.25]
    @test explicit.layout[:scene][:yaxis][:range] == [-0.2, 0.2]
    @test explicit.layout[:scene][:zaxis][:range] == [0.0, 0.2]
    comparison = plot_mesh_comparison(mesh, mesh)
    @test comparison.layout[:scene][:aspectmode] == "data"
    @test comparison.layout[:scene2][:aspectmode] == "data"
end
