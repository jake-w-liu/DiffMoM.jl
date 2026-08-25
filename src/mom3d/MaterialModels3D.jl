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
const _PASSIVITY_SAFE_EXPONENT_3D = 128

@inline function _passivity_extreme_component_3d(value::Float64)
    magnitude = abs(value)
    iszero(magnitude) && return false
    value_exponent = exponent(magnitude)
    return value_exponent < -_PASSIVITY_SAFE_EXPONENT_3D ||
           value_exponent > _PASSIVITY_SAFE_EXPONENT_3D
end

function _passivity_requires_exact_3d(M)
    @inbounds for value in M
        if _passivity_extreme_component_3d(real(value)) ||
           _passivity_extreme_component_3d(imag(value))
            return true
        end
    end
    return false
end

function _passivity_shifted_is_psd_exact_3d!(
        matrix::Matrix{Complex{Rational{BigInt}}})
    dimension = size(matrix, 1)
    @inbounds for pivot_index in 1:dimension
        pivot_position = 0
        for candidate in pivot_index:dimension
            diagonal = matrix[candidate, candidate]
            iszero(imag(diagonal)) ||
                error("internal exact passivity matrix is not Hermitian")
            real(diagonal) < 0 && return false
            if real(diagonal) > 0 && iszero(pivot_position)
                pivot_position = candidate
            end
        end

        if iszero(pivot_position)
            # In a positive-semidefinite Hermitian matrix, a zero diagonal
            # forces its entire row and column to be zero.  All remaining
            # diagonals are zero here, so the trailing block must vanish.
            for column in pivot_index:dimension
                for row in pivot_index:dimension
                    iszero(matrix[row, column]) || return false
                end
            end
            return true
        end

        if pivot_position != pivot_index
            for column in 1:dimension
                matrix[pivot_index, column], matrix[pivot_position, column] =
                    matrix[pivot_position, column], matrix[pivot_index, column]
            end
            for row in 1:dimension
                matrix[row, pivot_index], matrix[row, pivot_position] =
                    matrix[row, pivot_position], matrix[row, pivot_index]
            end
        end

        pivot = real(matrix[pivot_index, pivot_index])
        for column in (pivot_index + 1):dimension
            for row in column:dimension
                updated = matrix[row, column] -
                          matrix[row, pivot_index] *
                          conj(matrix[column, pivot_index]) / pivot
                matrix[row, column] = updated
                matrix[column, row] = conj(updated)
            end
        end
    end
    return true
end

@noinline function _validate_passive_tensor_exact_3d(
        M, label::AbstractString)
    rational_type = Rational{BigInt}
    complex_type = Complex{rational_type}
    dimension = size(M, 1)
    shifted_loss = Matrix{complex_type}(undef, dimension, dimension)
    tolerance = rational_type(_PASSIVITY_TOL_3D)

    # Passivity with the documented tolerance is equivalent to positive
    # semidefiniteness of tolerance*I - (M-M')/(2i).  Float64 entries are
    # dyadic rationals, so the small (at most 6x6) exceptional problem can be
    # checked exactly without overflowing intermediate sums or eigenvalues.
    @inbounds for column in 1:dimension
        for row in 1:column
            forward = M[row, column]
            reverse = M[column, row]
            loss_real =
                (rational_type(imag(forward)) +
                 rational_type(imag(reverse))) / 2
            loss_imag =
                (rational_type(real(reverse)) -
                 rational_type(real(forward))) / 2
            shifted = complex_type(
                (row == column ? tolerance : zero(tolerance)) - loss_real,
                -loss_imag,
            )
            shifted_loss[row, column] = shifted
            shifted_loss[column, row] = conj(shifted)
        end
    end

    _passivity_shifted_is_psd_exact_3d!(shifted_loss) ||
        error("$label violates exp(+i omega t) passivity: " *
              "anti-Hermitian loss matrix must be negative semidefinite.")
    return M
end

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

function _validate_passive_tensor_ieee_3d(M, label::AbstractString)
    dimension = size(M, 1)
    loss_matrix = Matrix((M - adjoint(M)) / (2im))
    maximum_row_sum = 0.0
    @inbounds for row in 1:dimension
        row_sum = 0.0
        for column in 1:dimension
            row_sum += abs(loss_matrix[row, column])
        end
        maximum_row_sum = max(maximum_row_sum, row_sum)
    end
    eigenvalue = maximum(eigvals(Hermitian(loss_matrix)))
    uncertainty = 64 * dimension * eps(Float64) * maximum_row_sum
    if eigenvalue < _PASSIVITY_TOL_3D - uncertainty
        return M
    elseif eigenvalue > _PASSIVITY_TOL_3D + uncertainty
        error("$label violates exp(+i omega t) passivity: " *
              "anti-Hermitian loss matrix must be negative semidefinite.")
    end
    return _validate_passive_tensor_exact_3d(M, label)
end

function _validate_passive_tensor_3d(
        M::SMatrix{3,3,ComplexF64,9}, label::AbstractString)
    _passivity_requires_exact_3d(M) &&
        return _validate_passive_tensor_exact_3d(M, label)
    return _validate_passive_tensor_ieee_3d(M, label)
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
    _passivity_requires_exact_3d(C) &&
        return _validate_passive_tensor_exact_3d(C, label)
    return _validate_passive_tensor_ieee_3d(C, label)
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
    passive::Bool
    function DrudePermittivity3D(eps_inf, plasma_freq_hz, gamma_hz; passive::Bool=true)
        epsc = _finite_complex_3d(eps_inf, "eps_inf")
        passive && _validate_passive_scalar_3d(epsc, "eps_inf")
        return new(epsc,
                   _finite_nonnegative_float_3d(plasma_freq_hz, "plasma_freq_hz"),
                   _finite_nonnegative_float_3d(gamma_hz, "gamma_hz"),
                   passive)
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
    passive::Bool
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
                   _finite_nonnegative_float_3d(gamma_hz, "gamma_hz"),
                   passive)
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
    passive::Bool
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
        return new(epss, epsi, tau, passive)
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
                  gamma_hz=model.gamma_hz,
                  passive=model.passive)

material_epsr_3d(model::LorentzPermittivity3D, freq_hz_or_k0) =
    lorentz_epsr_3d(freq_hz_or_k0;
                    eps_inf=model.eps_inf,
                    strength=model.strength,
                    resonance_freq_hz=model.resonance_freq_hz,
                    gamma_hz=model.gamma_hz,
                    passive=model.passive)

material_epsr_3d(model::DebyePermittivity3D, freq_hz_or_k0) =
    debye_epsr_3d(freq_hz_or_k0;
                  eps_static=model.eps_static,
                  eps_inf=model.eps_inf,
                  tau_s=model.tau_s,
                  passive=model.passive)

"""
    material_mur_3d(model, freq_hz_or_k0)

Evaluate a 3D relative permeability helper. Static models ignore the frequency
scale except for finite nonnegative validation. Drude, Lorentz, and Debye
response models interpret the argument as frequency in Hz.
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

material_mur_3d(model::DrudePermittivity3D, freq_hz_or_k0) =
    material_epsr_3d(model, freq_hz_or_k0)

material_mur_3d(model::LorentzPermittivity3D, freq_hz_or_k0) =
    material_epsr_3d(model, freq_hz_or_k0)

material_mur_3d(model::DebyePermittivity3D, freq_hz_or_k0) =
    material_epsr_3d(model, freq_hz_or_k0)

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

const _DISPERSION_FALLBACK_PRECISION_3D = 2304
const _DISPERSION_SAFE_EXPONENT_3D = 128

@inline function _dispersion_extreme_component_3d(value::Real)
    magnitude = abs(Float64(value))
    iszero(magnitude) && return false
    value_exponent = exponent(magnitude)
    return value_exponent < -_DISPERSION_SAFE_EXPONENT_3D ||
           value_exponent > _DISPERSION_SAFE_EXPONENT_3D
end

@inline function _dispersion_extreme_value_3d(value::Number)
    return _dispersion_extreme_component_3d(real(value)) ||
           _dispersion_extreme_component_3d(imag(value))
end

@inline _dispersion_requires_fallback_3d() = false

@inline function _dispersion_requires_fallback_3d(value, remaining...)
    return _dispersion_extreme_value_3d(value) ||
           _dispersion_requires_fallback_3d(remaining...)
end

@inline function _finite_dispersion_result_3d(
    value,
    label::AbstractString,
)
    converted = ComplexF64(value)
    isfinite(converted) ||
        throw(OverflowError(
            "$label is outside the ComplexF64 range"))
    return converted
end

@inline function _dispersion_sum_requires_fallback_3d(
    result::ComplexF64,
    first::ComplexF64,
    second::ComplexF64,
)
    isfinite(result) && isfinite(first) && isfinite(second) || return true
    @inbounds for component in (real, imag)
        first_component = component(first)
        second_component = component(second)
        scale = max(abs(first_component), abs(second_component))
        iszero(scale) && continue
        abs(component(result)) <= 64 * eps(Float64) * scale && return true
    end
    return false
end

@noinline function _drude_epsr_bigfloat_3d(
    f::Float64,
    epsc::ComplexF64,
    fp::Float64,
    gamma_hz::Float64,
)
    return setprecision(BigFloat, _DISPERSION_FALLBACK_PRECISION_3D) do
        frequency = BigFloat(f)
        plasma_frequency = BigFloat(fp)
        damping = BigFloat(gamma_hz)
        # The common (2π)^2 factor cancels exactly between numerator and
        # denominator when all three inputs are expressed in hertz.
        denominator = frequency * frequency -
                      Complex{BigFloat}(0, 1) * damping * frequency
        iszero(denominator) && error("Drude denominator is singular.")
        value = Complex{BigFloat}(epsc) -
                plasma_frequency * plasma_frequency / denominator
        return _finite_dispersion_result_3d(value, "Drude permittivity")
    end
end

@noinline function _lorentz_epsr_bigfloat_3d(
    f::Float64,
    epsc::ComplexF64,
    strength::ComplexF64,
    f0::Float64,
    gamma_hz::Float64,
)
    return setprecision(BigFloat, _DISPERSION_FALLBACK_PRECISION_3D) do
        frequency = BigFloat(f)
        resonance_frequency = BigFloat(f0)
        damping = BigFloat(gamma_hz)
        resonance_squared = resonance_frequency * resonance_frequency
        # As in the Drude model, the angular-frequency (2π)^2 factor cancels.
        denominator = resonance_squared - frequency * frequency +
                      Complex{BigFloat}(0, 1) * damping * frequency
        iszero(denominator) && error("Lorentz denominator is singular.")
        value = Complex{BigFloat}(epsc) +
                Complex{BigFloat}(strength) * resonance_squared /
                denominator
        return _finite_dispersion_result_3d(value, "Lorentz permittivity")
    end
end

@noinline function _debye_epsr_bigfloat_3d(
    f::Float64,
    eps_static::ComplexF64,
    eps_inf::ComplexF64,
    tau::Float64,
)
    return setprecision(BigFloat, _DISPERSION_FALLBACK_PRECISION_3D) do
        frequency_tau = 2 * BigFloat(π) * BigFloat(f) * BigFloat(tau)
        denominator = Complex{BigFloat}(1, frequency_tau)
        value = Complex{BigFloat}(eps_inf) +
                (Complex{BigFloat}(eps_static) -
                 Complex{BigFloat}(eps_inf)) / denominator
        return _finite_dispersion_result_3d(value, "Debye permittivity")
    end
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
    gamma_frequency =
        _finite_nonnegative_float_3d(gamma_hz, "gamma_hz")
    if _dispersion_requires_fallback_3d(
            f, epsc, fp, gamma_frequency)
        epsr = _drude_epsr_bigfloat_3d(
            f, epsc, fp, gamma_frequency)
        passive && _validate_passive_scalar_3d(epsr, "eps_r")
        return epsr
    end

    # The common (2π)^2 factor cancels. Working directly in hertz avoids
    # injecting two independently rounded angular-frequency products.
    denom = f^2 - 1im * gamma_frequency * f
    iszero(denom) && error("Drude denominator is singular.")
    susceptibility = ComplexF64(fp^2 / denom)
    value = ComplexF64(epsc - susceptibility)
    epsr = _dispersion_sum_requires_fallback_3d(
        value, epsc, -susceptibility) ?
        _drude_epsr_bigfloat_3d(f, epsc, fp, gamma_frequency) : value
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
    gamma_frequency =
        _finite_nonnegative_float_3d(gamma_hz, "gamma_hz")
    if _dispersion_requires_fallback_3d(
            f, epsc, fc, f0, gamma_frequency)
        epsr = _lorentz_epsr_bigfloat_3d(
            f, epsc, fc, f0, gamma_frequency)
        passive && _validate_passive_scalar_3d(epsr, "eps_r")
        return epsr
    end

    resonance_squared = f0^2
    # Factor the nearly resonant difference before adding damping. Squaring
    # two adjacent frequencies independently can lose most of their exact
    # Float64-input separation even though the resulting response is finite.
    frequency_difference = (f0 - f) * (f0 + f)
    denom = frequency_difference + 1im * gamma_frequency * f
    iszero(denom) && error("Lorentz denominator is singular.")
    resonant_term = ComplexF64(fc * resonance_squared / denom)
    value = ComplexF64(epsc + resonant_term)
    epsr = _dispersion_sum_requires_fallback_3d(
        value, epsc, resonant_term) ?
        _lorentz_epsr_bigfloat_3d(
            f, epsc, fc, f0, gamma_frequency) : value
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
    if _dispersion_requires_fallback_3d(f, epss, epsi, tau)
        epsr = _debye_epsr_bigfloat_3d(f, epss, epsi, tau)
        passive && _validate_passive_scalar_3d(epsr, "eps_r")
        return epsr
    end

    omega = 2pi * f
    relaxation_term = ComplexF64(
        (epss - epsi) / (1 + 1im * omega * tau))
    value = ComplexF64(epsi + relaxation_term)
    epsr = _dispersion_sum_requires_fallback_3d(
        value, epsi, relaxation_term) ?
        _debye_epsr_bigfloat_3d(f, epss, epsi, tau) : value
    passive && _validate_passive_scalar_3d(epsr, "eps_r")
    return epsr
end
