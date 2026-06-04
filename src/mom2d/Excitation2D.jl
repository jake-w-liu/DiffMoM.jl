# Excitation2D.jl — Incident field generation for 2D TM scattering
#
# Convention: exp(+iωt), plane wave E_z^inc = E₀ exp(-ik₀ k̂·r)

export planewave_2d, linesource_2d

"""
    planewave_2d(mesh, k0, phi_inc; E0=1.0)

Generate incident plane wave at cell centers.
Propagation direction: k̂ = (cos(phi_inc), sin(phi_inc)).
E_z^inc(r) = E₀ exp(-ik₀ k̂·r)
"""
function planewave_2d(mesh::Mesh2D, k0::Float64, phi_inc::Float64; E0::Float64=1.0)
    khat = Vec2(cos(phi_inc), sin(phi_inc))
    E_inc = Vector{ComplexF64}(undef, mesh.ncells)
    for m in 1:mesh.ncells
        E_inc[m] = E0 * exp(-im * k0 * dot(khat, mesh.centers[m]))
    end
    return E_inc
end

"""
    linesource_2d(mesh, k0, r_src)

Generate incident field from a 2D line source at position `r_src`.
E_z^inc(r) = (-i/4) H₀⁽²⁾(k₀|r - r_src|) (unit amplitude)
"""
function linesource_2d(mesh::Mesh2D, k0::Float64, r_src::Vec2)
    E_inc = Vector{ComplexF64}(undef, mesh.ncells)
    for m in 1:mesh.ncells
        E_inc[m] = greens_2d(mesh.centers[m], r_src, k0) * 4π  # normalized
    end
    return E_inc
end
