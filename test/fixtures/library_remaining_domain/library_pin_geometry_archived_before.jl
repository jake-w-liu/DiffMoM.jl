module ArchivedLibrary
using DiffMoM
import DiffMoM: _circuit_stored_real,_P2,_p2_seg_len,_p2_signed_area,
    _planar_regular_ngon,_EPS0,_MU0
include(joinpath(@__DIR__,"..","..","test","fixtures","library_stored_domain","source_after.jl"))
end
using DiffMoM, Test, LinearAlgebra, SHA, TOML
const shape=ArchivedLibrary.planar_line(length=.001,width=.0001,level=1,metal="pec")
rows=Dict{String,Any}[]
for angle in (.1,.3,pi/4,1.),offset in ((0.,1e9),(0.,1e10),(0.,1e11),
        (1e11,0.),(1e11,1e11))
    placed=ArchivedLibrary.planar_transform(shape;angle,offset)
    vertices=only(placed.polygons).vertices
    for pin in placed.pins
        a=vertices[pin.edge];b=vertices[mod1(pin.edge+1,length(vertices))]
        width=hypot((b-a)...)
        edge_tangent=(b-a)/width
        middle=a+(b-a)/2
        tangent_component=abs(dot(pin.direction,edge_tangent))
        midpoint_error=hypot((pin.point-middle)...)/width
        if tangent_component>.001 || midpoint_error>.001
            push!(rows,Dict("angle_rad"=>angle,"offset_m"=>collect(offset),
                "pin"=>pin.name,"edge_start"=>collect(a),"edge_stop"=>collect(b),
                "pin_direction"=>collect(pin.direction),"pin_point"=>collect(pin.point),
                "stable_edge_midpoint"=>collect(middle),
                "direction_tangent_component"=>tangent_component,
                "relative_midpoint_error"=>midpoint_error))
        end
    end
end
@testset "Archived accepted placements retain off-edge pins and non-normal directions" begin
    @test any(row->row["direction_tangent_component"]>.05,rows)
    @test any(row->row["relative_midpoint_error"]>.05,rows)
end
open(joinpath(@__DIR__,"library_pin_geometry_archived_before.toml"),"w") do io
    TOML.print(io,Dict("source_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,
        "..","..","test","fixtures","library_stored_domain","source_after.jl")))),"rows"=>rows))
end
