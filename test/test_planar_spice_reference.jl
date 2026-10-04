module PlanarActualSpiceReferenceTests
using DiffMoM,Test,LinearAlgebra,TOML
const directory=joinpath(@__DIR__,"..","validation","spice_reference","ngspice47")

@testset "Actual ngspice grounded and floating N-port references" begin
    manifest=TOML.parsefile(joinpath(directory,"comparison.toml"))
    @test occursin("ngspice-47",manifest["engine_version"])
    fs=collect(range(1e8,1e10;length=51));probe=collect(range(1e6,3e10;length=37))
    D=[.02 -.005;-.005 .03];E=[1e-12 -2e-13;-2e-13 2e-12];R=[1e8 -2e7;-2e7 8e7]
    real_model=planar_fit_rational([D+2pi*1im*f*E+R/(2pi*1im*f+2e9) for f in fs],fs;order=1)
    pole=-1e9+3e9im;residue=ComplexF64[1e7+2e6im -2e6+1e6im;-2e6+1e6im 8e6+1e6im]
    pair_model=planar_fit_rational([D+residue/(2pi*1im*f-pole)+conj.(residue)/(2pi*1im*f-conj(pole)) for f in fs],fs;order=2)
    for (base_name,model) in (("real_pole_affine_capacitance",real_model),
            ("complex_conjugate_pair",pair_model)),floating in (false,true)
        name=base_name*(floating ? "_floating_reference" : "_grounded")
        path=joinpath(directory,name)
        matrices=[zeros(ComplexF64,2,2) for _ in probe]
        for drive in 1:2
            lines=readlines(joinpath(path,"drive$(drive).dat"))
            @test length(lines)==length(probe)+1
            for k in eachindex(probe)
                fields=parse.(Float64,split(lines[k+1]))
                @test length(fields)==5
                @test fields[1]≈probe[k] rtol=1e-14
                for port in 1:2
                    matrices[k][port,drive]=-complex(fields[2port],fields[2port+1])
                end
            end
        end
        @test manifest[name]["floating_reference"]===floating
        for k in eachindex(probe)
            expected=planar_rational_eval(model,probe[k])
            @test norm(matrices[k]-expected)/norm(expected)<1e-9
        end
    end
end
end
