using Test
for f in ["test_planar_normal_radiation.jl","test_planar_hybrid_radiation.jl",
          "test_planar_retained_admittance_excitation.jl","test_planar_finite_accepted_power.jl",
          "test_planar_terminal_current_ranges.jl","test_planar_transfer_precision_and_power.jl",
          "test_planar_complex_references.jl","test_planar_current_types.jl"]
    println("=== ",f);include(joinpath("test",f))
end
