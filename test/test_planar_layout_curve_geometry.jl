module PlanarLayoutCurveGeometryTests
using Test,DiffMoM,LinearAlgebra

@testset "DXF arcs retain endpoints and caller sagitta" begin
    for bulge in (1e-10,-1e-10,1e-8,-1e-8,1.,-1.,2.,-2.)
        a=[0.,0.];b=[1.,0.];tolerance=abs(bulge)/128
        points=DiffMoM._dxf_arc_points(a,b,bulge,tolerance)
        @test points[:,1]==a
        @test points[:,end]==b
        @test all(isfinite,points)
        n=size(points,2)-1
        bits=precision(Float64)*2+abs(exponent(abs(bulge)))+ndigits(n;base=2)
        setprecision(BigFloat,bits) do
            q=BigFloat(bulge);radius=(1+q*q)/(4abs(q));theta=4atan(q)
            sagitta=2radius*sin(theta/(4n))^2
            @test sagitta<=BigFloat(tolerance)
            center_y=(1-q*q)/(4q)
            for j in 1:n-1
                angle=theta*j/n
                x=BigFloat(.5)-cos(angle)/2+center_y*sin(angle)
                y=center_y-sin(angle)/2-center_y*cos(angle)
                scale=max(1.,abs(Float64(x)),abs(Float64(y)))
                @test abs(points[1,j+1]-Float64(x))<=eps(scale)*precision(Float64)
                @test abs(points[2,j+1]-Float64(y))<=eps(scale)*precision(Float64)
            end
        end
        @test DiffMoM._dxf_arc_points(a,b,bulge,abs(bulge)/2)==hcat(a,b)
    end
    @test_throws ArgumentError DiffMoM._dxf_arc_points([0.,0.],[1.,0.],1.,nothing)
    @test_throws ArgumentError DiffMoM._dxf_arc_points([0.,0.],[1.,0.],NaN,1.)
    @test_throws ArgumentError DiffMoM._dxf_arc_points([0.,0.],[1.,0.],1.,0.)
    @test_throws ArgumentError DiffMoM._dxf_arc_points([0.,0.],[1.,0.],1.,eps();max_bytes=sizeof(Float64)*2*2)
    @test DiffMoM._dxf_arc_points([0.,0.],[1.,0.],1e-10,1e-8;max_bytes=32)==[0. 1.;0. 0.]
    @test_throws ArgumentError DiffMoM._dxf_arc_points([0.,0.],[1.,0.],1e-10,1e-8;max_bytes=31)
end

@testset "Circle tessellation uses sagitta and byte capacity" begin
    for tolerance in (.001,.1,1.,2.)
        vertices=DiffMoM._layout_circle([0.,0.],1.,tolerance)
        n=size(vertices,2)
        @test n>=3
        @test 2sin(pi/(2n))^2<=tolerance
        @test all(isfinite,vertices)
        @test vertices[:,1]==[1.,0.]
    end
    @test size(DiffMoM._layout_circle([0.,0.],1.,nothing;segments=3,max_bytes=48))==(2,3)
    @test_throws ArgumentError DiffMoM._layout_circle([0.,0.],1.,nothing;segments=3,max_bytes=47)
    @test_throws ArgumentError DiffMoM._layout_circle([0.,0.],1.,nothing)
    @test_throws ArgumentError DiffMoM._layout_circle([0.,0.],Inf,1.)
    @test_throws ArgumentError DiffMoM._layout_circle([0.,0.],1.,Inf)
    @test_throws ArgumentError DiffMoM._layout_circle([0.,0.],1.,eps();max_bytes=48)
end

@testset "Public DXF small arcs and explicit curve accuracy" begin
    mktempdir() do directory
        path=joinpath(directory,"curve.dxf")
        pairs=[(0,"SECTION"),(2,"ENTITIES"),(0,"LWPOLYLINE"),(90,"3"),(70,"1"),
            (10,"0"),(20,"0"),(42,"1e-10"),(10,"1"),(20,"0"),(10,"0"),(20,"1"),
            (0,"ENDSEC"),(0,"EOF")]
        write(path,join((string(code,'\n',value,'\n') for (code,value) in pairs)))
        @test_throws ArgumentError read_dxf(path;unit_m=1.)
        document=read_dxf(path;unit_m=1.,curve_tolerance=1e-8)
        @test document.polygons[1].vertices==[0. 1. 0.;0. 0. 1.]
        @test_throws ArgumentError read_dxf(path;unit_m=1.,curve_tolerance=Inf)
        @test_throws ArgumentError read_dxf(path;unit_m=1.,curve_tolerance=NaN)
        @test_throws ArgumentError read_dxf(path;unit_m=1.,max_bytes=filesize(path)-1)
    end
end
@testset "Extreme chord scale and GDS lattice-derived caps" begin
    points=DiffMoM._dxf_arc_points([0.,0.],[1e200,0.],1e-200,.001)
    @test points[:,1]==[0.,0.]
    @test points[:,end]==[1e200,0.]
    @test all(isfinite,points)
    @test minimum(points[2,:])<0
    @test minimum(points[2,:])>=-.5
    @test_throws ArgumentError DiffMoM._dxf_arc_points([0.,0.],[1.,0.],0.,nothing;max_bytes=31)
    mktempdir() do directory
        path=joinpath(directory,"round-path.gds")
        # Independently encoded two-point path, width20 DB units and round caps.
        write(path,hex2bytes(join((
            "000600020258001c01020000000000000000000000000000000000000000000000000008020654455354001403053e41",
            "89374bc6a7f03c10c6f7a0b5ed8d001c0502000000000000000000000000000000000000000000000000000806065041",
            "54480004090000060d02000100060e02000000062102000100080f0300000014001410030000000000000000000003e8",
            "00000000000411000004070000040400",
        ))))
        default=read_gdsii(path)
        explicit=read_gdsii(path;curve_tolerance=default.database_unit_m/2)
        @test length(default.polygons)==3
        @test all(i->default.polygons[i].vertices==explicit.polygons[i].vertices,eachindex(default.polygons))
        @test_throws ArgumentError read_gdsii(path;curve_tolerance=Inf)
        @test_throws ArgumentError read_gdsii(path;curve_tolerance=0.)
    end
end
end
