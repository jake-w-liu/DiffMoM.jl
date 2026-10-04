module PlanarGroupDelayRangeTests
using DiffMoM, Test

samples(phases)=[ComplexF64[0 cis(phase);cis(phase) 0] for phase in phases]

@testset "Group delay retains distinct Hz knots and stable quadratic differences" begin
    close=[1e9,nextfloat(1e9),nextfloat(nextfloat(1e9))]
    @test length(unique(2pi.*close))<length(close)
    for phase in (0.,.5,-2.),count in (2,3)
        data=PlanarNetworkData(close[1:count],samples(fill(phase,count)))
        @test planar_equation_curves(data).group_delay_s==zeros(count)
    end
    for scale in (1e-200,1.,1e200),nodes in ([1.,2.,4.],[1.,1.2,1.4,2.3,3.7])
        frequencies=scale.*nodes
        phases=0.1 .* nodes
        data=PlanarNetworkData(frequencies,samples(phases))
        expected=fill(-.1/(2pi*scale),length(nodes))
        @test planar_equation_curves(data).group_delay_s≈expected rtol=3e-14
        # Quadratic phase tests all endpoint and interior nonuniform stencils.
        data=PlanarNetworkData(frequencies,samples(0.1 .* nodes.^2))
        expected=-0.2 .* nodes./scale./(2pi)
        @test planar_equation_curves(data).group_delay_s≈expected rtol=3e-14
    end
    data=PlanarNetworkData([1.,2.,4.],samples([.1,.2,.4]))
    data.s[2][2,1]=0
    @test all(isnan,planar_equation_curves(data).group_delay_s)
    x=[1.,2.,4.];y=[.1,.2,.4]
    DiffMoM._curve_derivative(x,y,2)
    measured=@allocated DiffMoM._curve_derivative(x,y,2)
    @test measured<=32
end
end
