using DiffMoM,Test,LinearAlgebra
function _test_gerber_rectangle(body)
    mktempdir() do directory
        path=joinpath(directory,"metal.gbr")
        write(path,"%FSLAX36Y36*%%MOMM*%%ADD10R,0.8X0.4*%D10*"*body*"M02*")
        read_gerber(path)
    end
end
_rectangle_member(doc,x,y)=DiffMoM._artwork_contains(only(doc.objects).shape,x*1e-3,y*1e-3)

@testset "Gerber solid rectangle retains fixed aperture orientation" begin
    line=_test_gerber_rectangle("X1000000Y1000000D02*X3000000Y2000000D01*")
    # Independent clipping of the center-line parameter against q-aperture.
    function lineoracle(x,y)
        lo=max(0.,(x-.4-1)/2,y-.2-1)
        hi=min(1.,(x+.4-1)/2,y+.2-1)
        lo<=hi
    end
    for y in (.591:.117:2.391),x in (.391:.113:3.591)
        @test _rectangle_member(line,x,y)==lineoracle(x,y)
    end
    @test _rectangle_member(line,1.35,.85)
    @test !_rectangle_member(line,.65,.75)
    vertical=_test_gerber_rectangle("X1000000Y1000000D02*X1000000Y3000000D01*")
    @test _rectangle_member(vertical,1.39,.81)
    @test !_rectangle_member(vertical,1.41,2.)
    point=_test_gerber_rectangle("X1000000Y1000000D02*X1000000Y1000000D01*")
    @test _rectangle_member(point,1.39,1.19)
    @test !_rectangle_member(point,1.41,1.)
    zeroarc=_test_gerber_rectangle("G74*X1000000Y1000000D02*G03*X1000000Y1000000I1000000J0D01*")
    @test _rectangle_member(zeroarc,1.39,1.19)
    @test !_rectangle_member(zeroarc,2.,1.)
    rotated=_test_gerber_rectangle("%LMX*%%LR90*%%LS2*%X1000000Y1000000D02*X3000000Y1000000D01*")
    @test _rectangle_member(rotated,1.,1.79)
    @test !_rectangle_member(rotated,1.,1.81)
    @test _rectangle_member(rotated,.61,1.)
    # The standard permits only a solid R and circular apertures for draws;
    # a macro that happens to look rectangular is still invalid.
    for aperture in ("R,0.8X0.4X0.1","O,0.8X0.4","P,0.8X4X0")
        mktempdir() do directory
            path=joinpath(directory,"invalid.gbr")
            write(path,"%FSLAX36Y36*%%MOMM*%%ADD10$aperture*%D10*X0Y0D02*X1000000Y0D01*M02*")
            @test_throws ArgumentError read_gerber(path)
        end
    end
end

@testset "Gerber rectangle arcs use analytic circle intersection" begin
    quarter=_test_gerber_rectangle("G75*X2000000Y0D02*G03*X0Y2000000I-2000000J0D01*")
    reversequarter=_test_gerber_rectangle("G75*X0Y2000000D02*G02*X2000000Y0I0J-2000000D01*")
    # First-quadrant circle intersects the axis-aligned translated aperture
    # iff its radius lies between the rectangle's min/max distance to origin.
    function quarteroracle(x,y)
        lo_x=max(0.,x-.4);hi_x=x+.4;lo_y=max(0.,y-.2);hi_y=y+.2
        hi_x>=lo_x && hi_y>=lo_y && hypot(lo_x,lo_y)<=2<=hypot(hi_x,hi_y)
    end
    for y in (-.393:.127:2.393),x in (-.591:.131:2.591)
        @test _rectangle_member(quarter,x,y)==quarteroracle(x,y)
        @test _rectangle_member(reversequarter,x,y)==quarteroracle(x,y)
    end
    @test _rectangle_member(quarter,2.35,-.15) # square end cap
    @test !_rectangle_member(quarter,2.41,0.)
    for rotation in (0.,27.,90.),clockwise in (false,true)
        mode=clockwise ? "G02" : "G03"
        full=_test_gerber_rectangle("%LR$rotation*%G75*X2000000Y0D02*$mode*X2000000Y0I-2000000J0D01*")
        c,s=cosd(rotation),sind(rotation)
        for y in (-2.61:.219:2.61),x in (-2.61:.217:2.61)
            u,v=c*x+s*y,-s*x+c*y
            expected=hypot(max(abs(u)-.4,0.),max(abs(v)-.2,0.))<=2<=hypot(abs(u)+.4,abs(v)+.2)
            @test _rectangle_member(full,x,y)==expected
        end
    end
end
