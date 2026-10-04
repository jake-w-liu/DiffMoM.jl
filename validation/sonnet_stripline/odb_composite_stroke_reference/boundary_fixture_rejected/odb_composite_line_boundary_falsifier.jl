using DiffMoM,Test,TOML
include("odb_composite_line_prototype.jl")
rectangle(x0,x1,y0,y1)=DiffMoM._artwork_polygon([(x0*1e-3,y0*1e-3),
    (x1*1e-3,y0*1e-3),(x1*1e-3,y1*1e-3),(x0*1e-3,y1*1e-3)])
aperture=DiffMoM._ArtworkComposite(Tuple{Bool,DiffMoM._ArtworkShape}[
    (true,rectangle(-.1875,.1875,-.25,.25)),(false,rectangle(-.125,.125,-.125,.125)),
    (true,rectangle(-.03125,.03125,-.03125,.03125))])
shape=CP.Line(aperture,(.4375e-3,.5e-3),(.5625e-3,.5e-3))
rectangles=[(.25,.4375,.375,.625),(.5625,.75,.375,.625),
    (0.,1.,.25,.375),(0.,1.,.625,.75),(.40625,.59375,.46875,.53125)]
rows=Dict{String,Any}[]
for j in 1:16,i in 1:16
    x=(i-.5)/16;y=(j-.5)/16
    expected=any(x0<=x<=x1 && y0<=y<=y1 for (x0,x1,y0,y1) in rectangles)
    actual=CP.contains(shape,x*.001,y*.001)||(.25<=y<=.375)||(.625<=y<=.75)
    if actual!=expected
        yy=BigFloat(y*.001)-BigFloat(shape.start[2])
        radius=BigFloat(-.03125e-3)
        row=Dict("cell"=>[i,j],"physical_xy_mm"=>[x,y],"actual"=>actual,"literal_rectangle_expected"=>expected,
            "stored_exact_relative_y_m"=>string(yy),"stored_exact_island_lower_y_m"=>string(radius),
            "stored_exact_query_outside_island"=>abs(yy)>abs(radius))
        push!(rows,row);println(row)
        @test abs(yy)>abs(radius)
        @test y in (.46875,.53125)
        @test .4375<x<.5625
    end
end
@test !isempty(rows)
open(joinpath(@__DIR__,"odb_composite_line_boundary_falsifier.toml"),"w") do io
    TOML.print(io,Dict("scope"=>"original 16-cell prototype coupon uses cell centers exactly on literal island boundary; exact stored binary-coordinate subtraction lies outside that boundary; no geometry tolerance change", "mismatches"=>rows))
end
