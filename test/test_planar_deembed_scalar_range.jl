module PlanarDeembedScalarRangeTests
using Test, DiffMoM, LinearAlgebra

@testset "Scalar ABCD conversion retains finite extreme impedances" begin
    for impedance in (1e-200, 1e200), electrical_length in (.03+.4im, .05-.7im)
        transfer=planar_line_abcd(impedance,electrical_length)
        # The coupled conversion provides an independent block-solve route.
        admittance=planar_abcd_to_y(transfer)
        @test DiffMoM._y_of_abcd(transfer)≈admittance rtol=5e-14
        @test DiffMoM._abcd_of_y(admittance)≈transfer rtol=5e-14
        @test all(isfinite,DiffMoM._y_of_abcd(transfer))
        @test all(isfinite,DiffMoM._abcd_of_y(admittance))
    end
    # Products in a determinant can overflow while the final quotient fits.
    transfer=ComplexF64[1e200 1e200;0 1e200]
    @test DiffMoM._y_of_abcd(transfer)≈ComplexF64[1 -1e200;-1e-200 1]
    admittance=ComplexF64[1e200 -1e200;1e200 1e200]
    @test DiffMoM._abcd_of_y(admittance)≈ComplexF64[-1 -1e-200;-2e200 -1]
    # In the inverse direction, tiny determinant products must survive.
    admittance=ComplexF64[1e-200 -1e-200;1e-200 1e-200]
    @test DiffMoM._abcd_of_y(admittance)≈ComplexF64[-1 -1e200;-2e-200 -1]
    # A large product can cancel a finite term back into the stored range.
    admittance=ComplexF64[1e154 -1.5e308;-1 2e154]
    transfer=ComplexF64[2e154 1;1.5e308 1e154]
    @test DiffMoM._abcd_of_y(admittance)≈ComplexF64[2e154 1;5e307 1e154] rtol=1e-14
    @test DiffMoM._y_of_abcd(transfer)≈ComplexF64[1e154 -5e307;-1 2e154] rtol=1e-14
    for invalid in (ComplexF64[0 1;0 0],ComplexF64[NaN 1;1 0])
        @test_throws ArgumentError DiffMoM._abcd_of_y(invalid)
    end
    @test_throws ArgumentError DiffMoM._y_of_abcd(ComplexF64[1 0;0 1])
    @test_throws ArgumentError DiffMoM._y_of_abcd(ComplexF64[1 nextfloat(0.);0 1])
    @test_throws ArgumentError DiffMoM._abcd_of_y(ComplexF64[0 1;nextfloat(0.) 0])
end

@testset "Shunt double-delay calibration stores a positive finite length" begin
    launch=ComplexF64[1 0;.001+.002im 1]
    propagation=planar_line_abcd(50.,.1+.5im)
    first=planar_abcd_to_y(launch*propagation*launch)
    second=planar_abcd_to_y(launch*propagation^2*launch)
    for len in (big"1e1000",big"1e-1000",Inf,NaN,0.,-1.)
        @test_throws ArgumentError deembed_double_delay_calibrate(first,second;len)
    end
    for len in (.002,Float32(.002),big".002",1//500,floatmin(Float64),floatmax(Float64))
        calibration=deembed_double_delay_calibrate(first,second;len)
        @test calibration.len==Float64(len)
        @test isfinite(calibration.len) && calibration.len>0
        @test calibration.yd≈launch[2,1] rtol=1e-12
        @test planar_line_abcd(calibration.zc,calibration.gamma_l)≈propagation rtol=1e-12
        @test deembed_double_delay_apply(first,calibration;line=false)≈
            planar_abcd_to_y(propagation) rtol=1e-12
    end
end
end
