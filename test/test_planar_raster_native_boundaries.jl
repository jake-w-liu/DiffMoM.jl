module PlanarRasterNativeBoundaryTests
using Test,DiffMoM,SHA,TOML,Random
const fixture=joinpath(@__DIR__,"fixtures/native_diagonal_boundary_raster")
module PreviousRaster
using DiffMoM
using DiffMoM: _validate_sheet_grid
include(joinpath(@__DIR__,"fixtures/native_diagonal_boundary_raster/source_before/PlanarBasis_rasterize_poly.jl"))
end

function native_support(file,nx,ny)
    rows=readlines(file);first=findfirst(row->startswith(row,"SUB "),rows)
    count=parse(Int,split(rows[first])[3]);mask=falses(nx,ny)
    @test rows[first+count+1]=="END"
    for row in rows[first+1:first+count]
        fields=split(row);@test length(fields)==8 && fields[2] in ("11","12")
        xmin,ymin,xmax,ymax=parse.(Int,fields[3:6])
        @test all(iseven,(xmin,ymin,xmax,ymax)) && xmin<xmax && ymin<ymax
        for j in max(1,ymin÷2+1):min(ny,ymax÷2),i in max(1,xmin÷2+1):min(nx,xmax÷2)
            mask[i,j]=true
        end
    end
    mask
end

@testset "Actual native diagonal support and original wrong wall contacts" begin
    hashes=TOML.parsefile(joinpath(fixture,"sha256.toml"))["sha256"]
    actual=Set(replace(relpath(joinpath(dir,name),fixture),'\\'=>'/')
        for (dir,_,names) in walkdir(fixture) for name in names if name!="sha256.toml")
    @test Set(keys(hashes))==actual
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(fixture,path))))==digest
    end
    for axis in ("XDIR","YDIR"),target in (.0625,.125)
        tag="$(axis)_dir_-1_target_$(target)_literal"
        p=read_sonnet_project(joinpath(fixture,axis,"native",tag*".son"))
        prob=sonnet_planar_problem(p;freq=1e9);grid=prob.grid
        expected=native_support(joinpath(fixture,axis,"native",tag,"current_1.sid"),grid.nx,grid.ny)
        @test only(prob.sheets).mask==expected
        old=sheet_level(1,grid.nx,grid.ny)
        PreviousRaster.rasterize_poly!(old,grid,p.polygons[1].vertices[1,:],p.polygons[1].vertices[2,:])
        rows=readlines(joinpath(fixture,axis,"public_before",tag*"_mask.txt"))
        saved=BitMatrix([rows[j][i]=='1' for i in 1:grid.nx,j in 1:grid.ny])
        @test old.mask==saved
        @test (old.mask==expected)==(target==.125)
        if target==.0625
            @test axis=="XDIR" ? !old.mask[32,20] && expected[32,20] : old.mask[20,32] && !expected[20,32]
        end
        fresh=planar_read_touchstone(joinpath(fixture,axis,"native",tag,"native_raw.s2p"))
        original=planar_read_touchstone(joinpath(@__DIR__,"fixtures/native_zero_nominal_geovar/native",tag,"native_raw.s2p"))
        @test fresh.s==original.s
    end
    causal=TOML.parsefile(joinpath(fixture,"causal/comparison.toml"))
    @test causal["source_unchanged"] && causal["cases"][1]["full_s_error"]>.9 && causal["cases"][2]["full_s_error"]>.9
    @test causal["cases"][3]["full_s_error"]<=.005 && causal["cases"][3]["original_voltage_residual"]<=1e-9
    rejected=TOML.parsefile(joinpath(fixture,"rejected_inclusive_prototype/comparison.toml"))
    @test any(row->row["full_s_error"]>.005,rejected["cases"])
    prototype=TOML.parsefile(joinpath(fixture,"vertical_ray_prototype/comparison.toml"))
    @test prototype["source_unchanged"] && length(prototype["cases"])==8
    @test all(row["full_s_error"]<=.005 && row["original_voltage_residual"]<=1e-9 for row in prototype["cases"])
end

function exact_mask(points,nx,ny)
    mask=falses(nx,ny)
    for j in 1:ny,i in 1:nx
        x=(2i-1)//(2nx);y=(2j-1)//(2ny);inside=false
        previous=last(points)
        for current in points
            if (current[1]>x)!=(previous[1]>x)
                crossing=previous[2]+(current[2]-previous[2])*(x-previous[1])/(current[1]-previous[1])
                y<crossing && (inside=!inside)
            end
            previous=current
        end
        mask[i,j]=inside
    end
    mask
end
function warmed_allocation(sheet,grid,xs,ys)
    for _ in 1:20;rasterize_poly!(sheet,grid,xs,ys);end
    @allocated rasterize_poly!(sheet,grid,xs,ys)
end

@testset "Exact rational ownership, reversed edges, scales and zero allocation" begin
    rng=MersenneTwister(5440)
    polygons=[[(0//1,0//1),(1//1,0//1),(1//1,1//1)],
        [(0//1,0//1),(1//1,1//1),(0//1,1//1)],
        [(1//8,1//8),(7//8,1//8),(1//8,7//8)],
        [(0//1,0//1),(1//1,0//1),(1//1,1//4),(1//4,1//4),
            (1//4,3//4),(1//1,3//4),(1//1,1//1),(0//1,1//1)],
        [(0//1,0//1),(1//1,0//1),(1//1,1//8),(1//4,1//8),
            (1//4,3//8),(1//1,3//8),(1//1,1//2),(1//4,1//2),
            (1//4,3//4),(1//1,3//4),(1//1,7//8),(1//4,7//8),
            (1//4,1//1),(0//1,1//1)]]
    while length(polygons)<16
        points=[(rand(rng,0:64)//64,rand(rng,0:64)//64) for _ in 1:3]
        a,b,c=points
        (b[1]-a[1])*(c[2]-a[2])!=(b[2]-a[2])*(c[1]-a[1]) && push!(polygons,points)
    end
    for points in polygons,n in (16,32),scale in (1.,1e-3,2.0^-100,2.0^100)
        grid=CellGrid(scale,scale,n,n)
        expected=exact_mask(points,n,n)
        xs=[Float64(p[1])*scale for p in points];ys=[Float64(p[2])*scale for p in points]
        sheet=sheet_level(1,n,n);rasterize_poly!(sheet,grid,xs,ys)
        @test sheet.mask==expected
        reverse_sheet=sheet_level(1,n,n);rasterize_poly!(reverse_sheet,grid,reverse(xs),reverse(ys))
        @test reverse_sheet.mask==expected
        @test !any(sheet.connect_west) && !any(sheet.connect_east) && !any(sheet.connect_south) && !any(sheet.connect_north)
    end
    for n in (9,24,32)
        grid=CellGrid(1.,1.,n,n);xs=[.25,.75,.75,.25];ys=[.25,.25,.75,.75]
        rectangle=sheet_level(1,n,n);rasterize_rect!(rectangle,grid,.25,.75,.25,.75)
        polygon=sheet_level(1,n,n);rasterize_poly!(polygon,grid,xs,ys)
        old=sheet_level(1,n,n);PreviousRaster.rasterize_poly!(old,grid,xs,ys)
        @test polygon.mask==rectangle.mask==old.mask
    end
    grid=CellGrid(1e-3,1e-3,64,64);sheet=sheet_level(1,64,64)
    @test warmed_allocation(sheet,grid,[0.,1e-3,1e-3],[0.,0.,1e-3])==0
    for points in polygons[4:5]
        xs=[Float64(p[1])*1e-3 for p in points];ys=[Float64(p[2])*1e-3 for p in points]
        @test warmed_allocation(sheet,grid,xs,ys)==0
    end
    grid=CellGrid(1.,1.,16,16);sheet=sheet_level(1,16,16)
    rasterize_poly!(sheet,grid,Real[0,1//1,1.],Real[0,0.,1//1])
    @test sheet.mask==exact_mask(first(polygons),16,16)
end
end
