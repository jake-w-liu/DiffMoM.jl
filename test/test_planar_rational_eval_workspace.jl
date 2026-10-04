module PlanarRationalEvalWorkspaceTests
using DiffMoM, Test

function model(poles,residues,d,e)
    n=size(d,1)
    PlanarRationalModel(ComplexF64.(poles),[Matrix{ComplexF64}(r) for r in residues],Matrix{Float64}(d),
        Matrix{Float64}(e),[1.,2.,3.],0.,0.,false,false,:none,0.,0.,0.)
end

function allocation_at(m,f)
    planar_rational_eval(m,f)
    @allocated planar_rational_eval(m,f)
end

@testset "Rational evaluation keeps one dense result and physical complex arithmetic" begin
    for n in (1,16,64)
        d=[(p+q)/100 for p in 1:n,q in 1:n]
        e=[(p+2q)*1e-12 for p in 1:n,q in 1:n]
        residues=[ComplexF64[(2p+q)*1e7 for p in 1:n,q in 1:n]]
        m=model([-1e9],residues,d,e)
        for frequency in (-1e9,0.,1e9)
            # Scalar evaluation independently establishes each matrix entry.
            expected=ComplexF64[d[p,q]+2pi*im*frequency*e[p,q]+
                residues[1][p,q]/(2pi*im*frequency+1e9) for p in 1:n,q in 1:n]
            @test planar_rational_eval(m,frequency)≈expected rtol=2e-15
        end
        bytes=allocation_at(m,1e9)
        @test bytes<=16n*n+1024
    end
    # A large finite capacitor product stays complex, with its real
    # feedthrough retained independently of the imaginary dynamic term.
    m=model([],Matrix{ComplexF64}[],fill(1e308,1,1),fill(1e307,1,1))
    @test only(planar_rational_eval(m,1.))==complex(1e308,2pi*1e307)
    # Finite pole terms cancel only after physical division, as before.
    m=model([-1.,-1.],[fill(1e300+1e300im,1,1),fill(-1e300-1e300im,1,1)],
        zeros(1,1),zeros(1,1))
    @test only(planar_rational_eval(m,1.))==0
    @test only(planar_rational_eval(m,-1.))==0
    for frequency in (Inf,-Inf,NaN)
        @test_throws ArgumentError planar_rational_eval(m,frequency)
    end
end
end
