# MaterialModels3D.jl -- Pure material helper models for 3D volume solvers
#
# Convention: exp(+i omega t). With this convention, passive electric or
# magnetic loss has non-positive imaginary material response.

export IsotropicMaterial3D, DiagonalAnisotropicMaterial3D, TensorAnisotropicMaterial3D
export IsotropicPermeability3D, DiagonalPermeability3D, TensorPermeability3D
export MagneticMaterial3D, BianisotropicMaterial3D
export DrudePermittivity3D, LorentzPermittivity3D, DebyePermittivity3D
export material_epsr_3d, material_mur_3d
export material_bianisotropic_matrix_3d
export drude_epsr_3d, lorentz_epsr_3d, debye_epsr_3d

const _PASSIVITY_TOL_3D = 100 * eps(Float64)

function _finite_complex_3d(z, label::AbstractString)
    zc = ComplexF64(z)
    isfinite(real(zc)) && isfinite(imag(zc)) ||
        error("$label must be finite, got $zc.")
    return zc
end

function _finite_nonnegative_float_3d(x, label::AbstractString)
    xf = Float64(x)
    isfinite(xf) && xf >= 0 ||
        error("$label must be finite and nonnegative, got $x.")
    return xf
end

function _finite_positive_float_3d(x, label::AbstractString)
    xf = Float64(x)
    isfinite(xf) && xf > 0 ||
        error("$label must be finite and positive, got $x.")
    return xf
end

_validate_frequency_argument_3d(freq_hz_or_k0) =
    _finite_nonnegative_float_3d(freq_hz_or_k0, "freq_hz_or_k0")

function _validate_passive_scalar_3d(z::ComplexF64, label::AbstractString)
    imag(z) <= _PASSIVITY_TOL_3D ||
        error("$label violates exp(+i omega t) passivity: imag($label) must be <= 0 for passive loss, got $z.")
    return z
end

function _validate_passive_diagonal_3d(v::SVector{3,ComplexF64}, label::AbstractString)
    for a in 1:3
        _validate_passive_scalar_3d(v[a], "$label[$a]")
    end
    return v
end

function _validate_passive_tensor_3d(M::SMatrix{3,3,ComplexF64,9}, label::AbstractString)
    loss = (M - adjoint(M)) / (2im)
    vals = eigvals(Hermitian(Matrix(loss)))
    maximum(vals) <= _PASSIVITY_TOL_3D ||
        error("$label violates exp(+i omega t) passivity: anti-Hermitian loss matrix must be negative semidefinite.")
    return M
end

function _as_eps_vector_3d(eps_r, label::AbstractString)
    length(eps_r) == 3 || error("$label must have exactly three entries.")
    return SVector{3,ComplexF64}(_finite_complex_3d(eps_r[1], "$label[1]"),
                                 _finite_complex_3d(eps_r[2], "$label[2]"),
                                 _finite_complex_3d(eps_r[3], "$label[3]"))
end

function _as_eps_tensor_3d(eps_r, label::AbstractString)
    size(eps_r) == (3, 3) || error("$label must be a 3x3 tensor.")
    return SMatrix{3,3,ComplexF64,9}(
        _finite_complex_3d(eps_r[1, 1], "$label[1,1]"),
        _finite_complex_3d(eps_r[2, 1], "$label[2,1]"),
        _finite_complex_3d(eps_r[3, 1], "$label[3,1]"),
        _finite_complex_3d(eps_r[1, 2], "$label[1,2]"),
        _finite_complex_3d(eps_r[2, 2], "$label[2,2]"),
        _finite_complex_3d(eps_r[3, 2], "$label[3,2]"),
        _finite_complex_3d(eps_r[1, 3], "$label[1,3]"),
        _finite_complex_3d(eps_r[2, 3], "$label[2,3]"),
        _finite_complex_3d(eps_r[3, 3], "$label[3,3]"),
    )
end

function _as_material_cmat6_3d(C, label::AbstractString)
    size(C) == (6, 6) || error("$label must be a 6x6 tensor.")
    vals = ntuple(i -> _finite_complex_3d(C[i], "$label[$i]"), 36)
    return SMatrix{6,6,ComplexF64,36}(vals)
end

function _validate_passive_tensor6_3d(C::SMatrix{6,6,ComplexF64,36}, label::AbstractString)
    loss = (C - adjoint(C)) / (2im)
    vals = eigvals(Hermitian(Matrix(loss)))
    maximum(vals) <= _PASSIVITY_TOL_3D ||
        error("$label violates exp(+i omega t) passivity: anti-Hermitian loss matrix must be negative semidefinite.")
    return C
end

"""
    IsotropicMaterial3D(eps_r; passive=true)

Static isotropic relative permittivity model. For `exp(+i omega t)`, passive
loss has `imag(eps_r) <= 0`.
"""
struct IsotropicMaterial3D
    eps_r::ComplexF64
    function IsotropicMaterial3D(eps_r; passive::Bool=true)
        epsc = _finite_complex_3d(eps_r, "eps_r")
        passive && _validate_passive_scalar_3d(epsc, "eps_r")
        return new(epsc)
    end
end

"""
    DiagonalAnisotropicMaterial3D((eps_x, eps_y, eps_z); passive=true)

Static diagonal anisotropic relative permittivity model.
"""
struct DiagonalAnisotropicMaterial3D
    eps_r::SVector{3,ComplexF64}
    function DiagonalAnisotropicMaterial3D(eps_r; passive::Bool=true)
        epsv = _as_eps_vector_3d(eps_r, "eps_r")
        passive && _validate_passive_diagonal_3d(epsv, "eps_r")
        return new(epsv)
    end
end

"""
    TensorAnisotropicMaterial3D(eps_r; passive=true)

Static full 3x3 relative permittivity tensor model. Passive tensors require
`(eps - eps') / (2im)` to be negative semidefinite.
"""
struct TensorAnisotropicMaterial3D
    eps_r::SMatrix{3,3,ComplexF64,9}
    function TensorAnisotropicMaterial3D(eps_r; passive::Bool=true)
        epsm = _as_eps_tensor_3d(eps_r, "eps_r")
        passive && _validate_passive_tensor_3d(epsm, "eps_r")
        return new(epsm)
    end
end

"""
    IsotropicPermeability3D(mu_r; passive=true)

Static isotropic relative permeability model. Passive magnetic loss follows
the same `exp(+i omega t)` sign convention as permittivity.
"""
struct IsotropicPermeability3D
    mu_r::ComplexF64
    function IsotropicPermeability3D(mu_r; passive::Bool=true)
        muc = _finite_complex_3d(mu_r, "mu_r")
        passive && _validate_passive_scalar_3d(muc, "mu_r")
        return new(muc)
    end
end

"""
    DiagonalPermeability3D((mu_x, mu_y, mu_z); passive=true)

Static diagonal anisotropic relative permeability model.
"""
struct DiagonalPermeability3D
    mu_r::SVector{3,ComplexF64}
    function DiagonalPermeability3D(mu_r; passive::Bool=true)
        muv = _as_eps_vector_3d(mu_r, "mu_r")
        passive && _validate_passive_diagonal_3d(muv, "mu_r")
        return new(muv)
    end
end

"""
    TensorPermeability3D(mu_r; passive=true)

Static full 3x3 relative permeability tensor model.
"""
struct TensorPermeability3D
    mu_r::SMatrix{3,3,ComplexF64,9}
    function TensorPermeability3D(mu_r; passive::Bool=true)
        mum = _as_eps_tensor_3d(mu_r, "mu_r")
        passive && _validate_passive_tensor_3d(mum, "mu_r")
        return new(mum)
    end
end

struct MagneticMaterial3D{TE,TM}
    eps_model::TE
    mu_model::TM
end

"""
    BianisotropicMaterial3D(C6; passive=true)

Static normalized bianisotropic constitutive matrix. `C6` acts on normalized
fields `[E; eta0*H]`, so its electric and magnetic diagonal blocks are relative
permittivity and permeability for uncoupled media. The volume DDA path converts
this normalized constitutive tensor to the solver's `[E; H] -> [q; m]`
polarizability convention.
"""
struct BianisotropicMaterial3D
    C6::SMatrix{6,6,ComplexF64,36}
    function BianisotropicMaterial3D(C6; passive::Bool=true)
        C = _as_material_cmat6_3d(C6, "C6")
        passive && _validate_passive_tensor6_3d(C, "C6")
        return new(C)
    end
end

"""
    DrudePermittivity3D(eps_inf, plasma_freq_hz, gamma_hz; passive=true)

Simple Drude relative permittivity model:
`eps = eps_inf - omega_p^2 / (omega^2 - i gamma omega)`.
"""
struct DrudePermittivity3D
    eps_inf::ComplexF64
    plasma_freq_hz::Float64
    gamma_hz::Float64
    function DrudePermittivity3D(eps_inf, plasma_freq_hz, gamma_hz; passive::Bool=true)
        epsc = _finite_complex_3d(eps_inf, "eps_inf")
        passive && _validate_passive_scalar_3d(epsc, "eps_inf")
        return new(epsc,
                   _finite_nonnegative_float_3d(plasma_freq_hz, "plasma_freq_hz"),
                   _finite_nonnegative_float_3d(gamma_hz, "gamma_hz"))
    end
end

"""
    LorentzPermittivity3D(eps_inf, strength, resonance_freq_hz, gamma_hz; passive=true)

Simple Lorentz relative permittivity model:
`eps = eps_inf + strength * omega_0^2 / (omega_0^2 - omega^2 + i gamma omega)`.
"""
struct LorentzPermittivity3D
    eps_inf::ComplexF64
    strength::ComplexF64
    resonance_freq_hz::Float64
    gamma_hz::Float64
    function LorentzPermittivity3D(eps_inf, strength, resonance_freq_hz, gamma_hz;
                                   passive::Bool=true)
        epsc = _finite_complex_3d(eps_inf, "eps_inf")
        fc = _finite_complex_3d(strength, "strength")
        passive && _validate_passive_scalar_3d(epsc, "eps_inf")
        passive && _validate_passive_scalar_3d(fc, "strength")
        if passive
            real(fc) >= -_PASSIVITY_TOL_3D ||
                error("Lorentz passive oscillator requires real(strength) >= 0.")
        end
        return new(epsc, fc,
                   _finite_positive_float_3d(resonance_freq_hz, "resonance_freq_hz"),
                   _finite_nonnegative_float_3d(gamma_hz, "gamma_hz"))
    end
end

"""
    DebyePermittivity3D(eps_static, eps_inf, tau_s; passive=true)

Simple Debye relaxation model:
`eps = eps_inf + (eps_static - eps_inf) / (1 + i omega tau)`.
"""
struct DebyePermittivity3D
    eps_static::ComplexF64
    eps_inf::ComplexF64
    tau_s::Float64
    function DebyePermittivity3D(eps_static, eps_inf, tau_s; passive::Bool=true)
        epss = _finite_complex_3d(eps_static, "eps_static")
        epsi = _finite_complex_3d(eps_inf, "eps_inf")
        tau = _finite_positive_float_3d(tau_s, "tau_s")
        if passive
            _validate_passive_scalar_3d(epss, "eps_static")
            _validate_passive_scalar_3d(epsi, "eps_inf")
            real(epss - epsi) >= -_PASSIVITY_TOL_3D ||
                error("Debye passive relaxation requires real(eps_static - eps_inf) >= 0.")
        end
        return new(epss, epsi, tau)
    end
end

"""
    material_epsr_3d(model, freq_hz_or_k0)

Evaluate a 3D material permittivity helper. Static models ignore the frequency
scale except for finite nonnegative validation; dispersive models interpret the
argument as frequency in Hz. Returns `ComplexF64`, `SVector{3,ComplexF64}`, or
`SMatrix{3,3,ComplexF64,9}` depending on the model.
"""
material_epsr_3d(model::Number, freq_hz_or_k0) = begin
    _validate_frequency_argument_3d(freq_hz_or_k0)
    _finite_complex_3d(model, "eps_r")
end

material_epsr_3d(model::IsotropicMaterial3D, freq_hz_or_k0) = begin
    _validate_frequency_argument_3d(freq_hz_or_k0)
    model.eps_r
end

material_epsr_3d(model::DiagonalAnisotropicMaterial3D, freq_hz_or_k0) = begin
    _validate_frequency_argument_3d(freq_hz_or_k0)
    model.eps_r
end

material_epsr_3d(model::TensorAnisotropicMaterial3D, freq_hz_or_k0) = begin
    _validate_frequency_argument_3d(freq_hz_or_k0)
    model.eps_r
end

material_epsr_3d(model::MagneticMaterial3D, freq_hz_or_k0) =
    material_epsr_3d(model.eps_model, freq_hz_or_k0)

material_epsr_3d(model::DrudePermittivity3D, freq_hz_or_k0) =
    drude_epsr_3d(freq_hz_or_k0;
                  eps_inf=model.eps_inf,
                  plasma_freq_hz=model.plasma_freq_hz,
                  gamma_hz=model.gamma_hz)

material_epsr_3d(model::LorentzPermittivity3D, freq_hz_or_k0) =
    lorentz_epsr_3d(freq_hz_or_k0;
                    eps_inf=model.eps_inf,
                    strength=model.strength,
                    resonance_freq_hz=model.resonance_freq_hz,
                    gamma_hz=model.gamma_hz)

material_epsr_3d(model::DebyePermittivity3D, freq_hz_or_k0) =
    debye_epsr_3d(freq_hz_or_k0;
                  eps_static=model.eps_static,
                  eps_inf=model.eps_inf,
                  tau_s=model.tau_s)

"""
    material_mur_3d(model, freq_hz_or_k0)

Evaluate a 3D relative permeability helper. Static models ignore the frequency
scale except for finite nonnegative validation.
"""
material_mur_3d(model::Number, freq_hz_or_k0) = begin
    _validate_frequency_argument_3d(freq_hz_or_k0)
    _finite_complex_3d(model, "mu_r")
end

material_mur_3d(model::IsotropicPermeability3D, freq_hz_or_k0) = begin
    _validate_frequency_argument_3d(freq_hz_or_k0)
    model.mu_r
end

material_mur_3d(model::DiagonalPermeability3D, freq_hz_or_k0) = begin
    _validate_frequency_argument_3d(freq_hz_or_k0)
    model.mu_r
end

material_mur_3d(model::TensorPermeability3D, freq_hz_or_k0) = begin
    _validate_frequency_argument_3d(freq_hz_or_k0)
    model.mu_r
end

material_mur_3d(model::MagneticMaterial3D, freq_hz_or_k0) =
    material_mur_3d(model.mu_model, freq_hz_or_k0)

"""
    material_bianisotropic_matrix_3d(model, freq_hz_or_k0)

Evaluate a normalized static bianisotropic `6 x 6` material matrix. Static
models ignore the frequency scale except for finite nonnegative validation.
"""
material_bianisotropic_matrix_3d(model::BianisotropicMaterial3D, freq_hz_or_k0) = begin
    _validate_frequency_argument_3d(freq_hz_or_k0)
    model.C6
end

"""
    drude_epsr_3d(freq_hz; eps_inf=1, plasma_freq_hz, gamma_hz, passive=true)

Evaluate a Drude relative permittivity for the `exp(+i omega t)` convention:
`eps = eps_inf - omega_p^2 / (omega^2 - i gamma omega)`.
"""
function drude_epsr_3d(freq_hz; eps_inf=1.0, plasma_freq_hz, gamma_hz,
                       passive::Bool=true)
    f = _finite_positive_float_3d(freq_hz, "freq_hz")
    epsc = _finite_complex_3d(eps_inf, "eps_inf")
    fp = _finite_nonnegative_float_3d(plasma_freq_hz, "plasma_freq_hz")
    gamma = 2pi * _finite_nonnegative_float_3d(gamma_hz, "gamma_hz")
    omega = 2pi * f
    omega_p = 2pi * fp
    denom = omega^2 - 1im * gamma * omega
    abs(denom) > 0 || error("Drude denominator is singular.")
    epsr = ComplexF64(epsc - omega_p^2 / denom)
    passive && _validate_passive_scalar_3d(epsr, "eps_r")
    return epsr
end

"""
    lorentz_epsr_3d(freq_hz; eps_inf=1, strength, resonance_freq_hz, gamma_hz, passive=true)

Evaluate a Lorentz relative permittivity for the `exp(+i omega t)` convention:
`eps = eps_inf + strength * omega_0^2 / (omega_0^2 - omega^2 + i gamma omega)`.
"""
function lorentz_epsr_3d(freq_hz; eps_inf=1.0, strength, resonance_freq_hz,
                         gamma_hz, passive::Bool=true)
    f = _finite_nonnegative_float_3d(freq_hz, "freq_hz")
    epsc = _finite_complex_3d(eps_inf, "eps_inf")
    fc = _finite_complex_3d(strength, "strength")
    f0 = _finite_positive_float_3d(resonance_freq_hz, "resonance_freq_hz")
    gamma = 2pi * _finite_nonnegative_float_3d(gamma_hz, "gamma_hz")
    omega = 2pi * f
    omega0 = 2pi * f0
    denom = omega0^2 - omega^2 + 1im * gamma * omega
    abs(denom) > 0 || error("Lorentz denominator is singular.")
    epsr = ComplexF64(epsc + fc * omega0^2 / denom)
    passive && _validate_passive_scalar_3d(epsr, "eps_r")
    return epsr
end

"""
    debye_epsr_3d(freq_hz; eps_static, eps_inf=1, tau_s, passive=true)

Evaluate a Debye relative permittivity for the `exp(+i omega t)` convention:
`eps = eps_inf + (eps_static - eps_inf) / (1 + i omega tau)`.
"""
function debye_epsr_3d(freq_hz; eps_static, eps_inf=1.0, tau_s,
                       passive::Bool=true)
    f = _finite_nonnegative_float_3d(freq_hz, "freq_hz")
    epss = _finite_complex_3d(eps_static, "eps_static")
    epsi = _finite_complex_3d(eps_inf, "eps_inf")
    tau = _finite_positive_float_3d(tau_s, "tau_s")
    omega = 2pi * f
    epsr = ComplexF64(epsi + (epss - epsi) / (1 + 1im * omega * tau))
    passive && _validate_passive_scalar_3d(epsr, "eps_r")
    return epsr
end
