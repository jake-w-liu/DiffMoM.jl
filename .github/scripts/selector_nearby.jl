using DiffMoM,Test,TOML
root,repository,output=realpath(ARGS[1]),realpath(ARGS[2]),abspath(ARGS[3])
@assert realpath(dirname(dirname(pathof(DiffMoM))))==realpath(root) && !isfile(output)
result=@testset "Mac selector preserves unchanged nearby science gates" begin
 for name in ("test_planar_retained_admittance_excitation.jl","test_planar_retained_selector_quantization.jl",
  "test_planar_retained_normalized_reference_ranges.jl","test_planar_radiation.jl")
  include(joinpath(repository,"test",name))
 end
end
counts=Test.get_test_counts(result)
open(output,"w") do io
 TOML.print(io,Dict("version"=>string(VERSION),"passes"=>counts.passes+counts.cumulative_passes,
 "fails"=>counts.fails+counts.cumulative_fails,"errors"=>counts.errors+counts.cumulative_errors))
end
