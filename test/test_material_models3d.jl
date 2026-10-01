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

    extreme_passive_3 = Matrix(Diagonal(fill(1.0 - 1.0e308im, 3)))
    @test TensorAnisotropicMaterial3D(extreme_passive_3).eps_r ==
          extreme_passive_3
    @test TensorPermeability3D(extreme_passive_3).mu_r ==
          extreme_passive_3
    extreme_passive_6 = Matrix(Diagonal(fill(1.0 - 1.0e308im, 6)))
    @test BianisotropicMaterial3D(extreme_passive_6).C6 ==
          extreme_passive_6
    extreme_rank_one_loss = fill(-1.0e307, 6, 6)
    extreme_rank_one_material = Matrix{ComplexF64}(I, 6, 6) +
                                1im .* extreme_rank_one_loss
    @test BianisotropicMaterial3D(extreme_rank_one_material).C6 ==
          extreme_rank_one_material
    moderate_rank_one_loss = fill(-1.0e10, 3, 3)
    moderate_rank_one_material = Matrix{ComplexF64}(I, 3, 3) +
                                 1im .* moderate_rank_one_loss
    @test TensorAnisotropicMaterial3D(moderate_rank_one_material).eps_r ==
          moderate_rank_one_material
    extreme_mixed_loss = copy(extreme_passive_3)
    extreme_mixed_loss[2, 2] = 1.0 + 1.0e-10im
    @test_throws ErrorException TensorAnisotropicMaterial3D(
        extreme_mixed_loss)
    passivity_boundary = 100 * eps(Float64)
    extreme_boundary_loss = copy(extreme_passive_3)
    extreme_boundary_loss[2, 2] = 1.0 + passivity_boundary * im
    @test TensorAnisotropicMaterial3D(extreme_boundary_loss).eps_r ==
          extreme_boundary_loss
    extreme_boundary_loss[2, 2] =
        1.0 + nextfloat(passivity_boundary) * im
    @test_throws ErrorException TensorAnisotropicMaterial3D(
        extreme_boundary_loss)

    @test imag(drude_epsr_3d(2.0e14; eps_inf=1.0, plasma_freq_hz=1.0e15, gamma_hz=1.0e13)) <= 0
    @test imag(lorentz_epsr_3d(1.0e14; eps_inf=1.0, strength=0.5,
                               resonance_freq_hz=2.0e14, gamma_hz=1.0e13)) <= 0
    @test imag(debye_epsr_3d(1.0e9; eps_static=4.0, eps_inf=2.0, tau_s=1.0e-10)) <= 0

    active_drude = DrudePermittivity3D(
        1.0 + 0.2im, 0.0, 1.0; passive=false)
    active_lorentz = LorentzPermittivity3D(
        1.0 + 0.2im, -0.5 + 0.1im, 2.0, 0.1; passive=false)
    active_debye = DebyePermittivity3D(
        1.0 + 0.2im, 2.0 + 0.1im, 0.1; passive=false)
    @test !active_drude.passive
    @test !active_lorentz.passive
    @test !active_debye.passive
    @test material_epsr_3d(active_drude, 1.0) ==
          drude_epsr_3d(
              1.0;
              eps_inf=1.0 + 0.2im,
              plasma_freq_hz=0.0,
              gamma_hz=1.0,
              passive=false,
          )
    @test material_epsr_3d(active_lorentz, 1.0) ==
          lorentz_epsr_3d(
              1.0;
              eps_inf=1.0 + 0.2im,
              strength=-0.5 + 0.1im,
              resonance_freq_hz=2.0,
              gamma_hz=0.1,
              passive=false,
          )
    @test material_epsr_3d(active_debye, 1.0) ==
          debye_epsr_3d(
              1.0;
              eps_static=1.0 + 0.2im,
              eps_inf=2.0 + 0.1im,
              tau_s=0.1,
              passive=false,
          )

    dispersive_mu_models = (
        DrudePermittivity3D(1.0, 0.5, 0.1),
        LorentzPermittivity3D(1.0, 0.5, 2.0, 0.1),
        DebyePermittivity3D(2.0, 1.0, 0.1),
    )
    for mu_model in dispersive_mu_models
        dispersive_magnetic = MagneticMaterial3D(iso, mu_model)
        @test material_mur_3d(dispersive_magnetic, 1.0) ==
              material_epsr_3d(mu_model, 1.0)
    end
    active_at_evaluation = LorentzPermittivity3D(
        1.0, 0.5 - 1.0im, 1.0, 0.01)
    eps_passivity_error = try
        material_epsr_3d(active_at_evaluation, 2.0)
        ""
    catch err
        sprint(showerror, err)
    end
    mu_passivity_error = try
        material_mur_3d(active_at_evaluation, 2.0)
        ""
    catch err
        sprint(showerror, err)
    end
    @test occursin("eps_r violates", eps_passivity_error)
    @test occursin("mu_r violates", mu_passivity_error)
    @test !occursin("eps_r", mu_passivity_error)

    adjacent_frequency = prevfloat(1.0)
    adjacent_drude_reference = setprecision(BigFloat, 512) do
        ComplexF64(
            1 - BigFloat(1)^2 / BigFloat(adjacent_frequency)^2)
    end
    @test drude_epsr_3d(
        adjacent_frequency;
        eps_inf=1.0,
        plasma_freq_hz=1.0,
        gamma_hz=0.0,
        passive=false,
    ) == adjacent_drude_reference

    adjacent_lorentz_reference = setprecision(BigFloat, 512) do
        frequency = BigFloat(adjacent_frequency)
        resonance = BigFloat(1.0)
        ComplexF64(
            1 + resonance^2 / (resonance^2 - frequency^2))
    end
    @test lorentz_epsr_3d(
        adjacent_frequency;
        eps_inf=1.0,
        strength=1.0,
        resonance_freq_hz=1.0,
        gamma_hz=0.0,
    ) == adjacent_lorentz_reference

    mixed_debye_reference = setprecision(BigFloat, 2304) do
        frequency_tau = 2 * BigFloat(π) * BigFloat(1.0e-290) *
                        BigFloat(1.0e-179)
        ComplexF64(
            BigFloat(1.0e205) +
            (BigFloat(1.0e44) - BigFloat(1.0e205)) /
            Complex{BigFloat}(1, frequency_tau))
    end
    @test debye_epsr_3d(
        1.0e-290;
        eps_static=1.0e44,
        eps_inf=1.0e205,
        tau_s=1.0e-179,
        passive=false,
    ) == mixed_debye_reference

    @test drude_epsr_3d(
        1.0e200;
        eps_inf=1.0,
        plasma_freq_hz=2.0e200,
        gamma_hz=0.0,
    ) == -3.0 + 0.0im
    @test drude_epsr_3d(
        1.0e308;
        eps_inf=1.0,
        plasma_freq_hz=1.0e308,
        gamma_hz=0.0,
    ) == 0.0 + 0.0im
    damped_drude = drude_epsr_3d(
        1.0e-200;
        eps_inf=1.0,
        plasma_freq_hz=1.0e100,
        gamma_hz=1.0e308,
    )
    damped_drude_reference = setprecision(BigFloat, 512) do
        denominator = BigFloat(1.0e-200)^2 -
                      Complex{BigFloat}(0, 1) *
                      BigFloat(1.0e308) * BigFloat(1.0e-200)
        ComplexF64(
            BigFloat(1.0) - BigFloat(1.0e100)^2 / denominator)
    end
    @test damped_drude == damped_drude_reference
    @test lorentz_epsr_3d(
        2.0e200;
        eps_inf=1.0,
        strength=3.0,
        resonance_freq_hz=1.0e200,
        gamma_hz=0.0,
    ) == 0.0 + 0.0im
    @test_throws ErrorException lorentz_epsr_3d(
        1.0e200;
        eps_inf=1.0,
        strength=3.0,
        resonance_freq_hz=1.0e200,
        gamma_hz=0.0,
    )
    @test debye_epsr_3d(
        1.0e308;
        eps_static=3.0,
        eps_inf=1.0,
        tau_s=1.0e308,
    ) == 1.0 + 0.0im

    drude_epsr_3d(
        2.0e14;
        eps_inf=1.0,
        plasma_freq_hz=1.0e15,
        gamma_hz=1.0e13,
    )
    @test @allocated(drude_epsr_3d(
        2.0e14;
        eps_inf=1.0,
        plasma_freq_hz=1.0e15,
        gamma_hz=1.0e13,
    )) == 0
    lorentz_epsr_3d(
        1.0e14;
        eps_inf=1.0,
        strength=0.5,
        resonance_freq_hz=2.0e14,
        gamma_hz=1.0e13,
    )
    @test @allocated(lorentz_epsr_3d(
        1.0e14;
        eps_inf=1.0,
        strength=0.5,
        resonance_freq_hz=2.0e14,
        gamma_hz=1.0e13,
    )) == 0
    debye_epsr_3d(
        1.0e9;
        eps_static=4.0,
        eps_inf=2.0,
        tau_s=1.0e-10,
    )
    @test @allocated(debye_epsr_3d(
        1.0e9;
        eps_static=4.0,
        eps_inf=2.0,
        tau_s=1.0e-10,
    )) == 0
    drude_epsr_3d(
        1.0e200;
        eps_inf=1.0,
        plasma_freq_hz=2.0e200,
        gamma_hz=0.0,
    )
    @test @allocated(drude_epsr_3d(
        1.0e200;
        eps_inf=1.0,
        plasma_freq_hz=2.0e200,
        gamma_hz=0.0,
    )) <= 20_000

    @test_throws ErrorException IsotropicMaterial3D(2.0 + 0.1im)
    @test IsotropicMaterial3D(2.0 + 0.1im; passive=false).eps_r == 2.0 + 0.1im
    @test_throws ErrorException DiagonalAnisotropicMaterial3D((2.0, 3.0))
    @test_throws ErrorException TensorAnisotropicMaterial3D(ones(ComplexF64, 2, 2))
    @test_throws ErrorException DiagonalPermeability3D((1.0, 2.0))
    @test_throws ErrorException TensorPermeability3D(ones(ComplexF64, 2, 2))
    @test_throws ErrorException BianisotropicMaterial3D(ones(ComplexF64, 5, 5))
    @test_throws ErrorException material_epsr_3d(iso, Inf)
end

@testset "wideband Debye (Djordjevic-Sarkar) permittivity" begin
    # fit reproduces (epsr, tand) at the reference frequency exactly and
    # keeps tand nearly flat across the pole span
    m = wideband_debye_3d(4.3, 0.02, 1e9)
    e_ref = material_epsr_3d(m, 1e9)
    @test real(e_ref) ≈ 4.3 rtol = 1e-12
    @test -imag(e_ref) / real(e_ref) ≈ 0.02 rtol = 1e-10
    for f in (1e8, 1e10, 1e11)
        e = material_epsr_3d(m, f)
        @test -imag(e) / real(e) ≈ 0.02 rtol = 0.08
        @test 3.9 < real(e) < 4.5        # eps' decays gently with log f
    end
    @test real(material_epsr_3d(m, 1e6)) > real(e_ref)  # causal rise at low f

    # tand = 0 degenerates to the lossless constant
    m0 = wideband_debye_3d(2.2, 0.0, 10e9)
    @test material_epsr_3d(m0, 5e9) == 2.2 + 0.0im
    @test m0.eps_inf == 2.2 + 0.0im

    # single pole == DebyePermittivity3D
    m1 = WidebandDebyePermittivity3D(1.0 + 0im, [3.0], [1e-10])
    d1 = DebyePermittivity3D(4.0, 1.0, 1e-10)
    @test material_epsr_3d(m1, 3e9) ≈ material_epsr_3d(d1, 3e9)

    # monotonicity: eps' decreases with frequency (relaxation)
    e_lo = material_epsr_3d(m, 1e7)
    e_hi = material_epsr_3d(m, 1e11)
    @test real(e_lo) > real(e_hi)

    # invalid inputs / fail-closed paths
    @test_throws ErrorException wideband_debye_3d(0.0, 0.02, 1e9)
    @test_throws ErrorException wideband_debye_3d(4.3, -0.01, 1e9)
    @test_throws ErrorException wideband_debye_3d(4.3, 0.02, -1e9)
    @test_throws ErrorException wideband_debye_3d(4.3, 0.02, 1e9; npoles=0)
    # extreme tand over a narrow span -> eps_inf <= 0 is caught
    @test_throws ErrorException wideband_debye_3d(
        4.3, 50.0, 1e9; half_decades=0.5)
    @test_throws DimensionMismatch WidebandDebyePermittivity3D(
        2.0, [1.0, 1.0], [1e-9])
    @test_throws ErrorException WidebandDebyePermittivity3D(
        2.0, Float64[], Float64[])
    @test_throws ErrorException WidebandDebyePermittivity3D(
        2.0, [-0.5], [1e-9])
    @test_throws ErrorException WidebandDebyePermittivity3D(
        2.0, [0.5], [-1e-9])
    @test_throws DimensionMismatch wideband_debye_epsr_3d(
        1e9; eps_inf=1.0, delta=[1.0, 1.0], tau=[1e-9])
    @test_throws ErrorException wideband_debye_epsr_3d(
        1e9; eps_inf=1.0, delta=[-5.0], tau=[1e-9])
    @test_throws ErrorException wideband_debye_epsr_3d(
        1e9; eps_inf=1.0, delta=[1.0], tau=[-1e-9])
end
