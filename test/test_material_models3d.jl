using Test
using LinearAlgebra

if !isdefined(Main, :DiffMoM)
    include(joinpath(@__DIR__, "..", "src", "DiffMoM.jl"))
end
using .DiffMoM

@testset "3D material model helpers" begin
    iso = IsotropicMaterial3D(2.5 - 0.1im)
    @test material_epsr_3d(iso, 1.0e9) == 2.5 - 0.1im

    diag = DiagonalAnisotropicMaterial3D((2.0 - 0.1im, 3.0 - 0.2im, 4.0 + 0.0im))
    @test collect(material_epsr_3d(diag, 3.0)) == ComplexF64[2.0 - 0.1im, 3.0 - 0.2im, 4.0 + 0.0im]

    tensor = TensorAnisotropicMaterial3D(ComplexF64[
        2.0 - 0.1im 0.0 + 0.0im 0.0 + 0.0im
        0.0 + 0.0im 3.0 - 0.2im 0.0 + 0.0im
        0.0 + 0.0im 0.0 + 0.0im 4.0 - 0.3im
    ])
    @test size(material_epsr_3d(tensor, 3.0)) == (3, 3)
    loss = (material_epsr_3d(tensor, 3.0) - adjoint(material_epsr_3d(tensor, 3.0))) / (2im)
    @test maximum(eigvals(Hermitian(Matrix(loss)))) <= 100 * eps(Float64)

    mu = IsotropicPermeability3D(1.2 - 0.05im)
    magnetic = MagneticMaterial3D(iso, mu)
    @test material_epsr_3d(magnetic, 1.0e9) == material_epsr_3d(iso, 1.0e9)
    @test material_mur_3d(magnetic, 1.0e9) == 1.2 - 0.05im

    mu_diag = DiagonalPermeability3D((1.2 - 0.01im, 1.4 - 0.02im, 1.0 + 0.0im))
    @test collect(material_mur_3d(mu_diag, 2.0)) ==
          ComplexF64[1.2 - 0.01im, 1.4 - 0.02im, 1.0 + 0.0im]

    mu_tensor = TensorPermeability3D(ComplexF64[
        1.2 - 0.01im 0.0 + 0.0im 0.0 + 0.0im
        0.0 + 0.0im 1.4 - 0.02im 0.0 + 0.0im
        0.0 + 0.0im 0.0 + 0.0im 1.0 + 0.0im
    ])
    @test size(material_mur_3d(mu_tensor, 2.0)) == (3, 3)

    C6 = Matrix{ComplexF64}(I, 6, 6)
    C6[1, 1] = 2.0 - 0.01im
    C6[4, 4] = 1.3 - 0.02im
    C6[1, 5] = 0.05 + 0.0im
    C6[5, 1] = 0.05 + 0.0im
    bianiso = BianisotropicMaterial3D(C6)
    @test material_bianisotropic_matrix_3d(bianiso, 2.0) == bianiso.C6

    @test imag(drude_epsr_3d(2.0e14; eps_inf=1.0, plasma_freq_hz=1.0e15, gamma_hz=1.0e13)) <= 0
    @test imag(lorentz_epsr_3d(1.0e14; eps_inf=1.0, strength=0.5,
                               resonance_freq_hz=2.0e14, gamma_hz=1.0e13)) <= 0
    @test imag(debye_epsr_3d(1.0e9; eps_static=4.0, eps_inf=2.0, tau_s=1.0e-10)) <= 0

    @test_throws ErrorException IsotropicMaterial3D(2.0 + 0.1im)
    @test IsotropicMaterial3D(2.0 + 0.1im; passive=false).eps_r == 2.0 + 0.1im
    @test_throws ErrorException DiagonalAnisotropicMaterial3D((2.0, 3.0))
    @test_throws ErrorException TensorAnisotropicMaterial3D(ones(ComplexF64, 2, 2))
    @test_throws ErrorException DiagonalPermeability3D((1.0, 2.0))
    @test_throws ErrorException TensorPermeability3D(ones(ComplexF64, 2, 2))
    @test_throws ErrorException BianisotropicMaterial3D(ones(ComplexF64, 5, 5))
    @test_throws ErrorException material_epsr_3d(iso, Inf)
end
