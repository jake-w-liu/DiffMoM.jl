module PlanarCircuitLineTests
using DiffMoM,Test,LinearAlgebra
const DM=DiffMoM

# Independent original voltage/current ABCD equations at 4096-bit precision.
# This retains the growing hyperbolic terms and solves a separate 4x4
# boundary problem rather than restamping production travelling waves.
function high_precision_line(zc,gl,refs)
    setprecision(BigFloat,4096) do
        z=Complex{BigFloat}(zc);g=Complex{BigFloat}(gl)
        c,s=cosh(g),sinh(g);r=Complex{BigFloat}.(refs);roots=sqrt.(real.(r))
        M=zeros(Complex{BigFloat},4,4);B=zeros(Complex{BigFloat},4,2)
        M[1,:]=[1,-c,0,z*s];M[2,:]=[0,-s/z,1,c]
        M[3,1]=1;M[3,3]=r[1];M[4,2]=1;M[4,4]=r[2]
        B[3,1]=2roots[1];B[4,2]=2roots[2]
        X=M\B
        return ComplexF64.(Matrix{Complex{BigFloat}}(I,2,2)-Diagonal(roots)*X[3:4,:])
    end
end

@testset "Circuit lines against independent high-precision boundary equations" begin
    for z in (50. + 0im,25. + 13im,500. + 0im),refs in ([50.,50.],[30. + 10im,120. - 7im]),
            alpha in (0.,1e-12,.1,1.,10.,20.,40.,400.,710.,1000.,-.1,-1.,-10.),
            phase in (0.,.37,Float64(pi),1e5)
        g=alpha+im*phase;c=PlanarCircuit(2,[1,2];z0=refs)
        circuit_add_line!(c,[1,2],z,g)
        actual=solve_planar_circuit(c,1e9);expected=high_precision_line(z,g,refs)
        @test maximum(abs,actual.s-expected)<=2e-12*max(1.,maximum(abs,expected))
        @test all(isfinite,actual.s)
        if z==50 && refs==[50.,50.] && alpha>=0
            @test abs(actual.s[1,1])<=1e-14
            @test abs(actual.s[2,2])<=1e-14
            transmission=exp(-g)
            if alpha<710
                @test abs(actual.s[2,1]-transmission)<=5e-14*abs(transmission)
                @test abs(actual.s[1,2]-transmission)<=5e-14*abs(transmission)
            end
        end
    end
    # The finite semi-infinite limit is not an ABCD overflow.
    c=PlanarCircuit(2,[1,2]);circuit_add_line!(c,[1,2],50.,1000. + .37im)
    @test maximum(abs,solve_planar_circuit(c,1e9).s)<=1e-14
    # Reciprocal cascades retain the tiny forward AND reverse transmission.
    for alpha in (0.,10.,20.,40.,400.)
        c=PlanarCircuit(3,[1,3]);g1=alpha+.2im;g2=alpha+.17im
        circuit_add_line!(c,[1,2],50.,g1);circuit_add_line!(c,[2,3],50.,g2)
        result=solve_planar_circuit(c,1e9);t=exp(-(g1+g2))
        @test maximum(abs,diag(result.s))<=1e-14
        @test abs(result.s[1,2]-t)<=5e-14*abs(t)+10nextfloat(0.)
        @test abs(result.s[2,1]-t)<=5e-14*abs(t)+10nextfloat(0.)
    end
    for z in (50.,25. + 13im),g in (0.,.1+.37im,40. + .37im)
        expected=high_precision_line(z,g,[50.,75.])
        c=PlanarCircuit(4,[(1,2),(3,4)];z0=[50.,75.])
        circuit_add_line!(c,[(1,2),(3,4)],z,g)
        actual=solve_planar_circuit(c,1e9;floating_gauge=:auto)
        @test actual.s≈expected atol=1e-12 rtol=1e-12
        @test actual.gauge_nodes==[1,3]
        reverse=PlanarCircuit(4,[(1,2),(3,4)];z0=[50.,75.])
        circuit_add_line!(reverse,[(2,1),(3,4)],z,g)
        @test solve_planar_circuit(reverse,1e9;floating_gauge=:auto).s≈
            Diagonal([-1.,1.])*expected*Diagonal([-1.,1.]) atol=1e-12 rtol=1e-12
    end
end

@testset "Circuit line stored domains and provider/resource ordering" begin
    bad=(Inf,NaN,Complex(50.,Inf),BigFloat("1e1000"),BigFloat("1e-1000"),
        Complex{BigFloat}(BigFloat("50"),BigFloat("1e-1000")))
    for field in (:zc,:gl),value in bad
        c=PlanarCircuit(2,[1,2]);circuit_add_rlc!(c,1,2;r=10.)
        previous=copy(c.elements)
        @test_throws ArgumentError circuit_add_line!(c,[1,2],field===:zc ? value : 50.,field===:gl ? value : .37im)
        @test c.elements==previous
    end
    c=PlanarCircuit(2,[1,2]);@test_throws ArgumentError circuit_add_line!(c,[1,2],0.,0.)
    @test isempty(c.elements)
    for value in (BigFloat("50"),1//3,Complex{BigFloat}(50,3))
        c=PlanarCircuit(2,[1,2]);circuit_add_line!(c,[1,2],value,BigFloat("0"))
        @test only(c.elements).zc==ComplexF64(value)
        @test only(c.elements).gl===0. + 0im
        @test solve_planar_circuit(c,1e9).s≈ComplexF64[0 1;1 0] atol=1e-12
    end
    for field in (:zc,:gl),value in (bad...,0.,[50.])
        field===:gl && value==0. && continue
        seen=Symbol[];c=PlanarCircuit(2,[1,2];z0=f->(push!(seen,:reference);50.))
        circuit_add_line!(c,[1,2],f->(push!(seen,:zc);field===:zc ? value : 50.),
            f->(push!(seen,:gl);field===:gl ? value : .37im))
        @test_throws ArgumentError solve_planar_circuit(c,1e9)
        @test seen==(field===:zc ? [:reference,:zc] : [:reference,:zc,:gl])
        empty!(seen)
        @test_throws ArgumentError solve_planar_circuit(c,1e9;max_bytes=1)
        @test isempty(seen)
    end
    types=DataType[];c=PlanarCircuit(2,[1,2])
    circuit_add_line!(c,[1,2],f->(push!(types,typeof(f));50.),f->(push!(types,typeof(f));.37im))
    @test solve_planar_circuit(c,BigFloat("1e9")).s≈ComplexF64[0 exp(-.37im);exp(-.37im) 0] atol=1e-12
    @test types==[Float64,Float64]
end

function warmed_stamp_allocation(z,g)
    M=zeros(ComplexF64,6,6);terminals=[(1,0),(2,0)]
    DM._circuit_line_stamp!(M,3,terminals,z,g)
    return @allocated DM._circuit_line_stamp!(M,3,terminals,z,g)
end
@testset "Circuit line numeric stamp has no matrix temporaries" begin
    for z in (50. + 0im,1e308+1e308im,1e-200+0im),g in (0. + 0im,1000. + .37im,-10. + .37im)
        @test warmed_stamp_allocation(z,g)==0
    end
end
end
