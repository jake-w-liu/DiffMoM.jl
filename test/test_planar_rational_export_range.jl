module PlanarRationalExportRangeTests
using DiffMoM, Test

function model(poles,residues)
    PlanarRationalModel(ComplexF64.(poles),[fill(ComplexF64(r),1,1) for r in residues],
        fill(.02,1,1),zeros(1,1),[1.,2.,3.],0.,0.,false,false,:none,0.,0.,0.)
end

@testset "Rational SPICE export retains finite pole and residue coefficients" begin
    mktempdir() do directory
        path=joinpath(directory,"model.cir")
        tiny=model([-1e-309],[1e-300])
        planar_write_spice(tiny,path)
        text=read(path,String)
        @test occursin("Gdecaydm_u1",text)
        @test !occursin("Inf",text) && !occursin("NaN",text)
        compiled=planar_spice_model(planar_read_spice(path),"diffmom_nport")
        @test all(e->isfinite(e.value),compiled.elements)
        for frequency in (1e-300,1e-290,1.)
            result=planar_spice_sparams(compiled,frequency;pin_pairs=[("p1","dm_ref")])
            @test result.y≈planar_rational_eval(tiny,frequency) rtol=2e-13 atol=2e-15
        end
        for residue in (1e308+0im,0+1e308im,1e308-1e308im)
            pole=-1e308+1e307im
            paired=model([pole,conj(pole)],[residue,conj(residue)])
            planar_write_spice(paired,path)
            text=read(path,String)
            @test !occursin("Inf",text) && !occursin("NaN",text)
            compiled=planar_spice_model(planar_read_spice(path),"diffmom_nport")
            @test all(e->isfinite(e.value),compiled.elements)
            for frequency in (0.,1e9,1e100)
                result=planar_spice_sparams(compiled,frequency;pin_pairs=[("p1","dm_ref")])
                @test result.y≈planar_rational_eval(paired,frequency) rtol=3e-13 atol=2e-14
            end
        end
    end
end
end
