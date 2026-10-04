using DiffMoM, Test

@testset "resonance scan: closed band and requested residual tolerance" begin
    a, b, h = .03, .04, .01
    cavity = PlanarStackup([PlanarLayer(1., 1., h)], TERM_GND, TERM_GND, a, b)
    # Independent separation-of-variables PEC cavity frequency: TE101.
    exact = DiffMoM._C0 / 2 * sqrt(inv(a)^2 + inv(h)^2)
    for band in ((exact, 1.1exact), (.9exact, exact), (.9exact, 1.1exact))
        modes = planar_box_resonances(cavity, band...;
            mmax=1, nmax=0, nsamp=257, rtol=1e-12)
        @test length(modes) == 1
        isempty(modes) && continue
        @test only(modes).freq ≈ exact rtol=2e-15
        scale = maximum(abs(DiffMoM._resonance_residual(cavity, 2pi*f,
            (pi/a)^2, TE_POL, 1)) for f in range(band...; length=257))
        @test abs(DiffMoM._resonance_residual(cavity, 2pi*only(modes).freq,
            (pi/a)^2, TE_POL, 1)) <= 1e-12scale
        @test band[1] <= only(modes).freq <= band[2]
    end
    # Material loss displaces the pole off the real-frequency axis. Its
    # nonzero real residual cannot satisfy a stricter requested tolerance.
    lossy = PlanarStackup([PlanarLayer(1-1e-9im, 1., h)], TERM_GND, TERM_GND, a, b)
    @test isempty(planar_box_resonances(lossy, .9exact, 1.1exact;
        mmax=1, nmax=0, nsamp=257, rtol=1e-12))
    # A deliberately looser criterion may report the real-axis candidate.
    @test length(planar_box_resonances(lossy, .9exact, 1.1exact;
        mmax=1, nmax=0, nsamp=257, rtol=1e-8)) == 1
end
