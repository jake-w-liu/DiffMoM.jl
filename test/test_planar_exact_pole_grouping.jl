module ExactPoleGroupingTests
using DiffMoM,LinearAlgebra,Test
@testset "Exact pole grouping preserves repeated and adjacent modes" begin
    for pairs in 1:precision(Float64)
        for rates in (fill(-1.0,pairs),[-(k==1 ? 1.0 : nextfloat(1.0)) for k in 1:pairs])
            raw=ComplexF64[complex(r,i) for r in rates for i in (1.0,-1.0)]
            sort!(raw,by=p->(real(p),imag(p)))
            saved=copy(raw)
            poles=@inferred DiffMoM._vf_sort(raw)
            @test all(k->poles[k+1]==conj(poles[k]),1:2:length(poles))
            @test sort(poles,by=p->(real(p),imag(p)))==saved
            @test raw==saved
            @test size(DiffMoM._vf_basis(ComplexF64[0,im],poles))==(2,length(poles)+2)
        end
    end
    # Use the original suite's independent SPICE stamping helper. Its
    # original module is included before this regression in the test graph.
    stamp=getfield(parentmodule(@__MODULE__),:_workflow_spice_y)
    for rates in ((-1.0,-nextfloat(1.0)),(-1.0,-1.0),(-1.0,-2.0))
        raw=ComplexF64[complex(r,i) for r in rates for i in (1.0,-1.0)]
        poles=DiffMoM._vf_sort(raw)
        residues=[ones(ComplexF64,1,1) for p in poles]
        bound=sum(inv(-real(p)) for p in poles)
        model=PlanarRationalModel(poles,residues,fill(1+bound,1,1),zeros(1,1),Float64[0,1],
            0.,0.,true,true,:uniform_bound,1.,0.,0.)
        @test planar_rational_certificate(model).certified
        mktempdir() do dir
            path=planar_write_spice(model,joinpath(dir,"repeated.cir"))
            for f in (0.0,inv(2pi),1.0)
                @test isapprox(stamp(path,f,1),planar_rational_eval(model,f);rtol=1e-9)
            end
        end
    end
end
end
