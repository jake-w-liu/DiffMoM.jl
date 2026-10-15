using DiffMoM,Test

@testset "Native NOR finite thin-film limit" begin
    project=SonnetProject("nor.son",Dict{String,String}(),1.,1.,String[],
        Vector{String}[],Vector{String}[],String[],String[],SonnetPolygon[],
        SonnetPortSpec[],Dict{String,String}(),Vector{SonnetRecord}[],SonnetRecord[],SonnetRecord[])
    tiny=nextfloat(0.)
    for frequency in (tiny,1e-300),cover in (false,true)
        actual=sonnet_metal_zs(project,["metal","0","NOR","1e308",".5",string(tiny)],frequency;cover)
        expected=Float64(inv(BigFloat(1e308)*BigFloat(tiny)))
        @test isfinite(actual) && real(actual)≈expected
    end
    for cover in (false,true)
        actual=sonnet_metal_zs(project,["metal","0","NOR","5.8e7",".5","1e-6"],1e9;cover)
        expected=setprecision(BigFloat,512) do
            skin=sqrt(im*BigFloat(2pi*DiffMoM._MU0)*BigFloat(1e9)/BigFloat(5.8e7))
            weight=cover ? one(BigFloat) : BigFloat(1.25)/BigFloat(2.25)
            ComplexF64(weight*skin/tanh(weight*BigFloat(5.8e7)*skin*BigFloat(1e-6)))
        end
        @test actual≈expected rtol=1e-12
    end
end
