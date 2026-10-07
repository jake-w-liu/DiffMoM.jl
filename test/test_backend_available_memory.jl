using DiffMoM, Test, LinearAlgebra

let
 names=("_DEFAULT_MAX_OCTREE_STORAGE_BYTES","_DEFAULT_MAX_MLFMA_SETUP_BYTES","_DEFAULT_MAX_SURFACE_CACHE_BYTES_3D","_DEFAULT_MAX_ACA_STORAGE_BYTES","_DEFAULT_MAX_EFIE_CACHE_BYTES")
 @testset "Remaining backend budgets preserve explicit limits and equations" begin
  @test all(!isdefined(DiffMoM,Symbol(n)) for n in names)
  available=Sys.free_memory();@test available>0
  @test DiffMoM._default_max_dense_payload_bytes(available)==Int(min(available,typemax(Int)))
  @test (@allocated DiffMoM._default_max_dense_payload_bytes(available))==0
  mesh=make_rect_plate(.06,.04,2,2);rwg=build_rwg(mesh);k=2pi*1e9/299792458.
  dense=assemble_Z_efie(mesh,rwg,k;quad_order=3)
  explicit=assemble_Z_efie(mesh,rwg,k;quad_order=3,max_cache_bytes=typemax(Int))
  @test dense==explicit
  @test_throws ArgumentError assemble_Z_efie(mesh,rwg,k;quad_order=3,max_cache_bytes=1)
  @test_throws ArgumentError assemble_Z_efie(mesh,rwg,k;quad_order=3,max_cache_bytes=0)
  centers=rwg_centers(mesh,rwg);tree=build_octree(centers,k)
  tree_explicit=build_octree(centers,k;max_storage_bytes=typemax(Int))
  @test tree.perm==tree_explicit.perm && tree.iperm==tree_explicit.iperm && tree.nLevels==tree_explicit.nLevels
  boxes,storage=DiffMoM._octree_resource_bounds(length(centers),tree.nLevels)
  @test build_octree(centers,k;max_boxes=boxes,max_storage_bytes=storage).perm==tree.perm
  @test_throws ArgumentError build_octree(centers,k;max_boxes=boxes,max_storage_bytes=storage-1)
  @test_throws ArgumentError build_octree(centers,k;max_storage_bytes=0)
  aca=build_aca_operator(mesh,rwg,k;leaf_size=rwg.nedges,quad_order=3,max_rank=rwg.nedges)
  aca_explicit=build_aca_operator(mesh,rwg,k;leaf_size=rwg.nedges,quad_order=3,max_rank=rwg.nedges,max_storage_bytes=typemax(Int),max_cache_bytes=typemax(Int))
  @test Matrix(aca)==Matrix(aca_explicit)
  # The original registered batched ACA test permits the distinct dense
  # assembly summation order under its existing1e-12 equation criterion.
  @test norm(Matrix(aca)-dense)<=1e-12*norm(dense)
  @test_throws ArgumentError build_aca_operator(mesh,rwg,k;max_storage_bytes=1)
  @test_throws ArgumentError build_aca_operator(mesh,rwg,k;max_cache_bytes=1)
  near=assemble_mlfma_nearfield(tree,mesh,rwg,k;quad_order=3)
  near_explicit=assemble_mlfma_nearfield(tree,mesh,rwg,k;quad_order=3,max_nearfield_bytes=typemax(Int),max_cache_bytes=typemax(Int))
  @test near==near_explicit
  @test_throws ArgumentError assemble_mlfma_nearfield(tree,mesh,rwg,k;max_nearfield_bytes=1)
  @test_throws ArgumentError assemble_mlfma_nearfield(tree,mesh,rwg,k;max_cache_bytes=1)
 end
end
