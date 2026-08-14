# Quadrature.jl — Gaussian quadrature rules on triangles

export tri_quad_rule, tri_quad_points

"""
    tri_quad_rule(order)

Return (xi, w) for Gaussian quadrature on the reference triangle with
vertices (0,0), (1,0), (0,1). `xi` is a vector of (ξ₁,ξ₂) tuples, `w` is
the corresponding weight vector (already includes the Jacobian factor of 1/2
for unit reference triangle, so ∫f dA ≈ Σ w_q f(ξ_q) * 2A for a physical
triangle of area A, or equivalently integrate on the reference triangle
and multiply by 2A).

Supported orders: 1, 3, 4, 7.
"""
function tri_quad_rule(order::Int)
    if order == 1
        # 1-point centroid rule
        xi = [SVector(1/3, 1/3)]
        w  = [0.5]  # weight for reference triangle (area=0.5)
    elseif order == 3
        # 3-point rule (degree 2)
        xi = [SVector(1/6, 1/6),
              SVector(2/3, 1/6),
              SVector(1/6, 2/3)]
        w  = [1/6, 1/6, 1/6]
    elseif order == 4
        # 4-point rule (degree 3)
        xi = [SVector(1/3, 1/3),
              SVector(0.6, 0.2),
              SVector(0.2, 0.6),
              SVector(0.2, 0.2)]
        w  = [-27/96, 25/96, 25/96, 25/96]
    elseif order == 7
        # 7-point rule (degree 5)
        a1 = 0.059715871789770; b1 = 0.470142064105115
        a2 = 0.797426985353087; b2 = 0.101286507323456
        xi = [SVector(1/3, 1/3),
              SVector(b1, b1),
              SVector(a1, b1),
              SVector(b1, a1),
              SVector(b2, b2),
              SVector(a2, b2),
              SVector(b2, a2)]
        w0 = 0.1125
        w1 = 0.0661970763942530
        w2 = 0.0629695902724135
        w  = [w0, w1, w1, w1, w2, w2, w2]
    else
        error("Unsupported quadrature order $order. Use 1, 3, 4, or 7.")
    end
    return xi, w
end

@noinline function _triangle_affine_component_big(
    first::Float64,
    second::Float64,
    third::Float64,
    xi_first::Float64,
    xi_second::Float64,
)
    return setprecision(BigFloat, _TRIANGLE_GEOMETRY_FALLBACK_PRECISION) do
        first_weight = 1 - BigFloat(xi_first) - BigFloat(xi_second)
        value = first_weight * BigFloat(first) +
                BigFloat(xi_first) * BigFloat(second) +
                BigFloat(xi_second) * BigFloat(third)
        converted = Float64(value)
        isfinite(converted) ||
            throw(OverflowError(
                "triangle quadrature point is outside the Float64 range"))
        return converted
    end
end

@inline function _triangle_affine_component(
    first::Float64,
    second::Float64,
    third::Float64,
    xi_first::Float64,
    xi_second::Float64,
)
    first_weight = 1.0 - xi_first - xi_second
    product_first = first_weight * first
    product_second = xi_first * second
    product_third = xi_second * third
    products = (product_first, product_second, product_third)
    all(isfinite, products) ||
        return _triangle_affine_component_big(
            first, second, third, xi_first, xi_second)
    ((iszero(product_first) && !iszero(first_weight) && !iszero(first)) ||
     (iszero(product_second) && !iszero(xi_first) && !iszero(second)) ||
     (iszero(product_third) && !iszero(xi_second) && !iszero(third))) &&
        return _triangle_affine_component_big(
            first, second, third, xi_first, xi_second)

    absolute_sum = abs(product_first) +
                   abs(product_second) + abs(product_third)
    isfinite(absolute_sum) ||
        return _triangle_affine_component_big(
            first, second, third, xi_first, xi_second)
    value = muladd(
        first_weight, first,
        muladd(xi_first, second, product_third))
    if !isfinite(value) ||
       (!iszero(absolute_sum) &&
        abs(value) <= 64 * eps(Float64) * absolute_sum) ||
       (!iszero(value) && abs(value) < floatmin(Float64))
        return _triangle_affine_component_big(
            first, second, third, xi_first, xi_second)
    end
    return value
end

"""
    tri_quad_points(mesh, t, xi)

Map reference triangle quadrature points `xi` to physical coordinates on
triangle `t` of the mesh. Returns a vector of Vec3.
"""
function tri_quad_points(mesh::TriMesh, t::Int, xi::Vector{<:SVector{2}})
    v1 = _mesh_vertex(mesh, mesh.tri[1, t])
    v2 = _mesh_vertex(mesh, mesh.tri[2, t])
    v3 = _mesh_vertex(mesh, mesh.tri[3, t])

    return [Vec3(ntuple(component -> _triangle_affine_component(
        v1[component], v2[component], v3[component], ξ[1], ξ[2]), 3))
            for ξ in xi]
end
