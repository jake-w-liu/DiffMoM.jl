# Greens.jl — Free-space scalar Green's function and derivatives
#
# Convention: exp(+iωt), so G(r,r') = exp(-ik|r-r'|) / (4π|r-r'|)

export greens, greens_smooth, grad_greens

"""
    greens(r, rp, k)

Scalar free-space Green's function G(r,r') = exp(-ik R) / (4π R)
where R = |r - r'|.
Works with complex `k` for complex-step differentiation.
"""
function greens(r::SVector{3}, rp::SVector{3}, k)
    R_vec = r - rp
    R = sqrt(dot(R_vec, R_vec))   # use sqrt(dot) for complex-step compatibility
    if abs(R) < 1e-30
        return zero(complex(typeof(real(k))))
    end
    return exp(-im * k * R) / (4π * R)
end

"""
    greens_smooth(r, rp, k)

Smooth part of the Green's function after singularity extraction:
  G_smooth(r,r') = [exp(-ikR) - 1] / (4πR)
with limit -ik/(4π) as R → 0.

Used for self-cell integration with singularity subtraction.
"""
function greens_smooth(r::SVector{3}, rp::SVector{3}, k)
    R_vec = r - rp
    R = sqrt(dot(R_vec, R_vec))
    if abs(R) < 1e-14
        return -im * k / (4π)
    end
    return expm1(-im * k * R) / (4π * R)
end

"""
    grad_greens(r, rp, k)

Gradient of G with respect to r: ∇G = dG/dR * R̂ = [(-ik - 1/R) G] * R̂
where R̂ = (r - r') / |r - r'|.
Returns a 3-vector (SVector{3}).
"""
function grad_greens(r::SVector{3}, rp::SVector{3}, k)
    R_vec = r - rp
    R = sqrt(dot(R_vec, R_vec))
    if abs(R) < 1e-30
        return SVector{3}(zero(complex(typeof(real(k)))),
                          zero(complex(typeof(real(k)))),
                          zero(complex(typeof(real(k)))))
    end
    G = exp(-im * k * R) / (4π * R)
    dGdR = (-im * k - 1 / R) * G
    Rhat = R_vec / R
    return dGdR * Rhat
end
