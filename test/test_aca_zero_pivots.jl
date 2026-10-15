module ACAZeroPivotTests
using DiffMoM,LinearAlgebra,Test

# Reflection symmetry makes orthogonal RWGs on stacked diamond plates
# decouple. Centroid quadrature preserves that exact zero analytically;
# parallel RWGs retain a nonzero far-field interaction.
function diamond_plates(column_axes)
    vertices=zeros(3,16);triangles=zeros(Int,3,8);a=.1
    for (plate,z) in enumerate((0.,1.,4.,5.))
        offset=4(plate-1)
        vertices[:,offset+1:offset+4]=[-a 0. a 0.;0. -a 0. a;z z z z]
        axis=plate==1 ? :x : plate==2 ? :y : column_axes[plate-2]
        triangles[:,2plate-1:2plate].=(axis===:x ? [1 3;2 4;4 2] : [2 4;3 1;1 3]).+offset
    end
    mesh=TriMesh(vertices,triangles);rwg=build_rwg(mesh)
    order=sortperm(getindex.(rwg_centers(mesh,rwg),3))
    cache=DiffMoM._build_efie_cache(mesh,rwg,1.;quad_order=1)
    (;mesh,rwg,order,cache)
end

@testset "ACA restarts zero rows and disconnected residual pivots" begin
    for columns in ((:y,:y),(:x,:y))
        fixture=diamond_plates(columns);(;mesh,rwg,order,cache)=fixture
        rows,cols=order[1:2],order[3:4]
        exact=[DiffMoM._efie_entry(cache,i,j) for i in rows,j in cols]
        @test iszero(exact[1,2]) && !iszero(exact[2,2])
        for r in (rows,reverse(rows)),c in (cols,reverse(cols))
            target=[DiffMoM._efie_entry(cache,i,j) for i in r,j in c]
            U,V=aca_lowrank(cache,r,c;tol=1e-10,max_rank=2)
            @test size(U,2)==(columns==(:y,:y) ? 1 : 2)
            @test norm(U*V'-target)<=1e-13norm(target)
        end
        operator=build_aca_operator(mesh,rwg,1.;quad_order=1,leaf_size=2,aca_tol=1e-10,max_rank=2)
        dense=[DiffMoM._efie_entry(cache,i,j) for i in 1:4,j in 1:4]
        @test Matrix(operator) ≈ dense rtol=1e-13
        x=ComplexF64[1,2im,-1,.5im]
        @test operator*x ≈ dense*x rtol=1e-13
        @test operator'*x ≈ dense'*x rtol=1e-13
        @test_throws ArgumentError aca_lowrank(cache,rows,cols;max_rank=2,max_output_bytes=127)
        @test_throws BoundsError aca_lowrank(cache,[0],cols)
        @test_throws BoundsError aca_lowrank(cache,rows,[rwg.nedges+1])
        U,V=aca_lowrank(cache,rows[1:1],cols[2:2])
        @test size(U)==(1,0) && size(V)==(1,0)
    end
end
end
