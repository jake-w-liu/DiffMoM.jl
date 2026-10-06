module ODBRoundedAnnuliTests
using DiffMoM,Test,Random
const DM=DiffMoM

definitions=(("donut_s10x8xr2",10.,10.,8.,8.,1.,2.),
    ("donut_s10x4xr1",10.,10.,4.,4.,3.,1.),
    ("donut_s10x0xr2",10.,10.,0.,0.,5.,2.),
    ("donut_rc10x7x1xr2",10.,7.,8.,5.,1.,2.),
    ("donut_rc8x10x0.75xr3",8.,10.,6.5,8.5,.75,3.),
    ("donut_rc10x7x1xr0",10.,7.,8.,5.,1.,0.))

# Signed-distance formulation is independent of the production rectangle
# containment shortcuts. Its negative region is the filled rounded box.
function rounded_distance(x,y,w,h,r,corners)
    quadrant=x>=0 ? (y>=0 ? '1' : '4') : (y>=0 ? '2' : '3')
    radius=quadrant in corners ? r : 0.
    qx=abs(x)-(w/2-radius);qy=abs(y)-(h/2-radius)
    hypot(max(qx,0.),max(qy,0.))+min(max(qx,qy),0.)-radius
end
function annulus_truth(x,y,w,h,iw,ih,wall,r,corners)
    rounded_distance(x,y,w,h,r,corners)<=0 &&
        ((iw==0 || ih==0) || rounded_distance(x,y,iw,ih,max(r-wall,0.),corners)>0)
end
function read_annulus(body;kwargs...)
    mktempdir() do directory
        path=joinpath(directory,"features");write(path,body)
        read_odb_features(path;kwargs...)
    end
end
function occupied(art,x,y)
    state=false
    for object in art.objects
        DM._artwork_contains(object.shape,x,y) && (state=object.dark)
    end
    state
end
@noinline repeated_contains(shape,x,y)=DM._artwork_contains(shape,x,y)
resources=Any[]
    @testset "ODB rounded annuli independent geometry and corner selection" begin
        random=MersenneTwister(20261006)
        for (name,w,h,iw,ih,wall,r) in definitions,mask in 1:15
            corners=join(string(i) for i in 1:4 if mask & (1<<(i-1))!=0)
            shape=DM._odb_standard_symbol(name*"x"*corners,1.)
            @test DM._artwork_bounds(shape)==(-w/2,w/2,-h/2,h/2)
            for _ in 1:200
                x=(rand(random)-.5)*1.3w;y=(rand(random)-.5)*1.3h
                @test DM._artwork_contains(shape,x,y)==annulus_truth(x,y,w,h,iw,ih,wall,r,corners)
            end
            @test DM._artwork_contains(shape,0.,0.)==(iw==0 || ih==0)
            @test DM._artwork_contains(shape,w*.6,0.)==false
        end
        for row in definitions
            base=DM._odb_standard_symbol(row[1],1.)
            full=DM._odb_standard_symbol(row[1]*"x1234",1.)
            for point in ((0.,0.),(4.5,0.),(4.4,4.4),(-4.4,-4.4),(.4,.9))
                @test DM._artwork_contains(base,point...)==DM._artwork_contains(full,point...)
            end
            for scale in (1e-150,1e-6,1.,1e150)
                shape=DM._odb_standard_symbol(row[1],scale)
                for (x,y) in ((0.,0.),(4.5,0.),(4.4,4.4),(-4.4,-4.4),(.4,.9))
                    @test DM._artwork_contains(shape,scale*x,scale*y)==annulus_truth(x,y,row[2:end]...,"1234")
                end
            end
        end
        for name in ("donut_s10x10xr2","donut_s10x-1xr2","donut_s10x8xr6",
                "donut_s10x8xr-1","donut_s10x8xr2x11","donut_rc10x7x0xr2",
                "donut_rc10x7x4xr2","donut_rc10x7x1xr4","donut_rc10x7x1xr-1")
            @test_throws ArgumentError DM._odb_standard_symbol(name,1.)
        end
    end
    @testset "ODB rounded annuli public units orientation and transparent holes" begin
        for (name,w,h,iw,ih,wall,r) in definitions,orientation in 0:9
            angle=orientation<8 ? 90mod(orientation,4) : 23.
            suffix=orientation>=8 ? " 23" : ""
            # Imperial symbol suffix uses mils, independently of MM coordinates.
            art=read_annulus("UNITS=MM\n\$0 $(name)x1 I\nP 7 11 0 P 0 $(orientation)$(suffix);ID=321\n")
            @test length(art.objects)==1
            @test art.objects[1].attributes["ID"]==["321"]
            for (x,y) in ((0.,0.),(4.5,0.),(4.4,4.4),(-4.4,-4.4),(.4,.9))
                u=cosd(angle)*x+sind(angle)*y;v=-sind(angle)*x+cosd(angle)*y
                orientation in (4,5,6,7,9) && (u=-u)
                @test occupied(art,.007+25.4e-6u,.011+25.4e-6v)==annulus_truth(x,y,w,h,iw,ih,wall,r,"1")
            end
        end
        body="UNITS=MM\n\$0 r1000\n\$1 donut_s6000x4000xr1000\nP 0 0 0 P 0 0\nP 0 0 1 P 0 0\nP 0 0 1 N 0 0\n"
        art=read_annulus(body)
        @test occupied(art,0.,0.)
        @test !occupied(art,.0025,0.)
        @test_throws ArgumentError read_annulus(body;max_objects=2)
        @test_throws ArgumentError read_annulus(body;max_bytes=1)
    end
    @testset "ODB rounded annuli constant storage and repeated membership" begin
        for name in ("donut_s10x8xr2","donut_rc10x7x1xr2x13")
            shape=DM._odb_standard_symbol(name,1.)
            for _ in 1:10
                repeated_contains(shape,4.4,1.2)
            end
            samples=[@allocated(repeated_contains(shape,4.4,1.2)) for _ in 1:5]
            @test all(iszero,samples)
            @test shape.parts isa Tuple && length(shape.parts)==2
            push!(resources,Dict("name"=>name,"samples"=>samples,"owned_payload"=>Base.summarysize(shape)))
        end
    end

function distance_box(x,y,w,h,r)
    qx=abs(x)-w/2+r;qy=abs(y)-h/2+r
    hypot(max(qx,0.),max(qy,0.))+min(max(qx,qy),0.)-r
end
function annulus_art(name,record)
    mktempdir() do directory
        path=joinpath(directory,"features")
        write(path,"UNITS=MM\n\$0 $(name) M\n$(record)\n")
        read_odb_features(path)
    end
end
rows=Any[]
@testset "ODB rounded annuli physical cell masks and analytic area" begin
    grid=CellGrid(.012,.012,256,256)
    for (name,w,h,iw,ih,r,ri) in (("donut_s10000x8000xr2000",.010,.010,.008,.008,.002,.001),
            ("donut_rc10000x7000x1000xr2000",.010,.007,.008,.005,.002,.001))
        art=annulus_art(name,"P 6 6 0 P 0 0")
        masks=artwork_cell_masks(art,grid);mask=only(values(masks))
        expected=falses(grid.nx,grid.ny);clear=falses(grid.nx,grid.ny)
        for i in 1:grid.nx,j in 1:grid.ny
            x=(i-.5)*grid.dx-.006;y=(j-.5)*grid.dy-.006
            outer=distance_box(x,y,w,h,r);inner=distance_box(x,y,iw,ih,ri)
            expected[i,j]=outer<=0 && inner>0
            clear[i,j]=abs(outer)>1e-12 && abs(inner)>1e-12
        end
        @test all(mask[clear].==expected[clear])
        # Independent analytic rectangle area minus four corner deficits.
        area=w*h-iw*ih-(4-pi)*(r*r-ri*ri)
        measured=count(mask)*grid.dx*grid.dy
        perimeter=2(w+h+iw+ih)
        @test abs(measured-area)<=perimeter*hypot(grid.dx,grid.dy)
        @test !mask[128,128]
        @test_throws ArgumentError artwork_cell_masks(art,grid;max_bytes=1)
        push!(rows,Dict("name"=>name,"clear_cells_checked"=>count(clear),
            "expected_area"=>area,"raster_area"=>measured,"area_error_bound"=>perimeter*hypot(grid.dx,grid.dy)))
    end
end
@testset "ODB rounded square annulus swept line independent convex geometry" begin
    grid=CellGrid(.020,.012,320,192)
    art=annulus_art("donut_s10000x8000xr2000","L 6 6 10 6 0 P 0")
    mask=only(values(artwork_cell_masks(art,grid)))
    expected=falses(grid.nx,grid.ny);clear=falses(grid.nx,grid.ny)
    # Union of translated outer convex pads, minus the intersection of
    # translated holes. Four-mm sweep gives a 14x10-mm outer rounded box
    # and a 4x8-mm inner rounded box, both centered at (8,6) mm.
    for i in 1:grid.nx,j in 1:grid.ny
        x=(i-.5)*grid.dx-.008;y=(j-.5)*grid.dy-.006
        outer=distance_box(x,y,.014,.010,.002)
        inner=distance_box(x,y,.004,.008,.001)
        expected[i,j]=outer<=0 && inner>0
        clear[i,j]=abs(outer)>1e-12 && abs(inner)>1e-12
    end
    @test all(mask[clear].==expected[clear])
    @test count(mask)>0 && !mask[128,96]
    push!(rows,Dict("name"=>"swept_donut_s10000x8000xr2000","clear_cells_checked"=>count(clear),
        "occupied_cells"=>count(mask)))
end
end
