using Test, DiffMoM

@testset "mixed-wall box resonances survive homogeneous layer splitting" begin
    a, b, height = .01, .008, .001
    pmc = PlanarTerminator(TERM_PMC)
    for (m, n, pol) in ((1, 0, TE_POL), (1, 1, TM_POL))
        expected = DiffMoM._C0 / (2pi) *
            sqrt((m*pi/a)^2 + (n*pi/b)^2 + (pi/(2height))^2)
        for (bottom, top) in ((TERM_GND, pmc), (pmc, TERM_GND)),
                count in (1, 2, 4), samples in (16, 32)
            stack = PlanarStackup(fill(PlanarLayer(1., 1., height/count), count),
                bottom, top, a, b)
            found = planar_box_resonances(stack, .5expected, 1.5expected;
                mmax=m, nmax=n, nsamp=samples)
            target = filter(r -> r.m == m && r.n == n && r.pol == pol, found)
            @test length(target) == 1
            if length(target) == 1
                @test target[1].freq ≈ expected rtol=1e-12
                residual = DiffMoM._resonance_residual(stack,
                    2pi * target[1].freq, (m*pi/a)^2 + (n*pi/b)^2, pol, count)
                @test abs(residual) < 1e-12
            end
        end
    end

    # A scalar-component crossing alone is not a complex resonance. Loss
    # leaves a nonzero residual and must still fail the acceptance gate.
    stack = PlanarStackup(fill(PlanarLayer(1. - .02im, 1., height/4), 4),
        TERM_GND, pmc, a, b)
    expected = DiffMoM._C0 / (2pi) * sqrt((pi/a)^2 + (pi/(2height))^2)
    @test isempty(planar_box_resonances(stack, .5expected, 1.5expected;
        mmax=1, nmax=0, nsamp=32))
end
