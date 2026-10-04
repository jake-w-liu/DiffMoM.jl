module PlanarCalibrationStoredLengthsTests
using Test, DiffMoM, LinearAlgebra

const launch=planar_line_abcd(43.,.015+.04im)
const propagation=planar_line_abcd(67.,.03+.6im)
const first=planar_abcd_to_y(launch*propagation*launch)
const second=planar_abcd_to_y(launch*propagation^2*launch)

const through=ComplexF64[0 1;1 0]
const electrical_length=.03+.7im
const line=ComplexF64[0 exp(-electrical_length);exp(-electrical_length) 0]
const reflect=Diagonal(ComplexF64[.9,.9])
linecal(length)=planar_line_calibrate(through,line;delta_length=length,
    reflect_standard=reflect,reflection=.9)

@testset "Calibration lengths remain finite and positive in stored units" begin
    for length in (BigFloat("1e1000"),BigFloat("1e-1000"),NaN,Inf,0.,-1.)
        @test_throws ArgumentError planar_double_delay_calibrate(first,second;length)
        @test_throws ArgumentError planar_group_double_delay_calibrate(first,second;length)
        @test_throws ArgumentError linecal(length)
    end
    for length in (Float32(.002),.002,BigFloat(".002"),1//500)
        stored=Float64(length)
        for calibration in (planar_double_delay_calibrate(first,second;length),
                planar_group_double_delay_calibrate(first,second;length))
            @test calibration.length==stored && calibration.length>0 && isfinite(calibration.length)
            @test calibration.launch≈launch rtol=2e-11
            @test calibration.line≈propagation rtol=2e-11
            @test deembed_ports(first,[calibration.launch,calibration.launch])≈
                planar_abcd_to_y(propagation) rtol=2e-11
        end
        calibration=linecal(length)
        @test calibration.delta_length==stored && isfinite(calibration.gamma)
        @test calibration.gamma≈electrical_length/stored rtol=2e-11
        @test planar_calibration_apply(line,calibration)≈line rtol=2e-11
    end
    # A stored positive length may still make the extracted propagation
    # constant unrepresentable; it must not return an Inf-bearing model.
    @test_throws ArgumentError linecal(nextfloat(0.))
    @test_throws ArgumentError linecal(BigFloat(nextfloat(0.)))
    for length in (floatmin(Float64),floatmax(Float64))
        calibration=linecal(length)
        @test calibration.delta_length==length
        @test isfinite(calibration.gamma)
        @test calibration.gamma≈electrical_length/length rtol=2e-11
    end
    # Coupled multiconductor calibration obeys the same metadata boundary.
    Zg=ComplexF64[1+.4im .1im;.1im 2+.7im]
    Yg=ComplexF64[.001im -.0002im;-.0002im .002im]
    Zl=ComplexF64[.8+12im 2im;2im .6+15im]
    Yl=ComplexF64[.0001+.005im -.001im;-.001im .0002+.006im]
    E=exp([zeros(2,2) Zg;Yg zeros(2,2)])
    L=exp([zeros(2,2) Zl;Yl zeros(2,2)])
    a=planar_abcd_to_y(E*L*E);b=planar_abcd_to_y(E*L^2*E)
    for length in (BigFloat("1e1000"),BigFloat("1e-1000"))
        @test_throws ArgumentError planar_group_double_delay_calibrate(a,b;length)
    end
    calibration=planar_group_double_delay_calibrate(a,b;length=BigFloat(".002"))
    @test calibration.length==.002
    @test calibration.launch≈E rtol=2e-11
    @test calibration.line≈L rtol=2e-11
end
end
