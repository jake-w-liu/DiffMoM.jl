module DDAOutputCancellationTests
using DiffMoM,Test,LinearAlgebra

@testset "DDA field outputs retain coherent cancellation" begin
    grid=VoxelGrid3D((0.,3.),(0.,1.),(0.,1.),3,1,1)
    observation,direction=Vec3(1.5,1.5,.5),Vec3(0,1,0)
    for em in (false,true),phase in (1.0+0im,0.0+1im,1.0+.5im)
        op=em ? em_dda_operator_3d(grid,1.,2.,1.) : dda_operator_3d(grid,1.,2.)
        fields=CVec3[CVec3(0,0,1e16phase),CVec3(0,0,3phase),CVec3(0,0,-1e16phase)]
        zeros=fill(zero(CVec3),3)
        res=em ? EMDDAResult3D(fields,zeros,copy(fields),copy(zeros),op.alpha,op,nothing,
                    :gmres,nothing,grid,1.,false) :
            DDAResult3D(fields,copy(fields),op.eps_r,op.alpha,op,nothing,:gmres,nothing,grid,1.,false)
        # The two outer z dipoles have identical Green coefficients at this
        # observation and identical far-field phases. They cancel exactly.
        expected_near,expected_far=setprecision(BigFloat,8192) do
            moment=BigFloat(.75)*3Complex{BigFloat}(phase)
            (ComplexF64(-im*moment*exp(-im*BigFloat(1))/(4BigFloat(pi))),
             ComplexF64(moment*exp(im*BigFloat(.5))/(4BigFloat(pi))))
        end
        near=em ? scattered_fields_em_dda_3d(res,[observation])[1][1] : scattered_field_dda_3d(res,[observation])[1]
        far=em ? farfield_em_dda_3d(res,direction)[1] : farfield_dda_3d(res,direction)
        @test near[3] ≈ expected_near rtol=8eps(Float64)
        @test far[3] ≈ expected_far rtol=8eps(Float64)
        if em
            _,magnetic=farfield_em_dda_3d(res,direction)
            @test magnetic[1] ≈ expected_far/res.eta0 rtol=8eps(Float64)
        end
    end
end

@testset "Huygens cancellation retains the stored electric and magnetic moments" begin
    grid=VoxelGrid3D((-.5,.5),(-.5,.5),(-.5,.5),1,1,1)
    eta=376.730313668
    electric=CVec3(0,0,1e16)
    magnetic=CVec3(-2.654418727984993e13,0,0)
    result=solve_em_dda_3d(grid,1.,Matrix{ComplexF64}(I,6,6),[electric],[magnetic])
    expected_e,expected_h=setprecision(BigFloat,8192) do
        q,m,impedance=BigFloat(electric[3]),BigFloat(magnetic[1]),BigFloat(eta)
        (Float64((q+impedance*m)/(4BigFloat(pi))),Float64((q/impedance+m)/(4BigFloat(pi))))
    end
    e,h=farfield_em_dda_3d(result,Vec3(0,1,0))
    @test real(e[3]) ≈ expected_e rtol=8eps(Float64)
    @test real(h[1]) ≈ expected_h rtol=8eps(Float64)
    @test expected_e>0 && expected_h>0
end
end
