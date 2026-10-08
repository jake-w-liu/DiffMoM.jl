module ConformalCallerRoundingTests
using DiffMoM, Test

@testset "Conformal geometry retains Float64 rounding independently of caller MPFR state" begin
    caller_bits,caller_rounding = precision(BigFloat),rounding(BigFloat)
    unit = ldexp(1.0,exponent(0.001))
    vertices,bounds = planar_normalize_polygon([(0.0,unit),(3unit,nextfloat(3unit)),(3unit,4unit),(0.0,4unit)])
    polygon = PlanarPolygon("rounding",1,"pec","",vertices,bounds)
    x = 1.5unit
    constraints = [(1,((x,0.0),(x,5unit)))]
    expected_orientation = 1.5eps(1.0)+eps(1.5)
    for mode in (RoundNearest,RoundUp,RoundDown,RoundToZero)
        setrounding(BigFloat,mode) do
            # These exact products produce Float64 midpoint ties. Ordinary
            # Float64 geometry must retain its nearest-even convention.
            @test DiffMoM._planar_orient2d(0.0,0.0,nextfloat(1.0),1.0,1.5,nextfloat(1.5)) == expected_orientation
            @test DiffMoM._planar_conformal_line_y((0.0,unit),(3unit,nextfloat(3unit)),x) == 2unit
            point = DiffMoM._planar_conformal_cross_point((0.0,unit),(3unit,nextfloat(3unit)),(x,0.0),(x,5unit))
            @test point == (x,2unit)
            mesh = planar_conformal_mesh([polygon];constraints,edge_size=3unit,interior_size=3unit,edge_band=0.0)
            candidates = [mesh.vertices[2,j] for j in axes(mesh.vertices,2) if mesh.vertices[1,j] == x]
            @test !isempty(candidates) && minimum(candidates) == 2unit
        end
    end
    @test precision(BigFloat) == caller_bits
    @test rounding(BigFloat) == caller_rounding
end
end
