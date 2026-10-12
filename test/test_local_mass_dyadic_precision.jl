module LocalMassDyadicPrecisionTests
using DiffMoM, LinearAlgebra, Test

const GAP = DiffMoM._LOCAL_MASS_FALLBACK_PRECISION + precision(Float64) + 1

# scale = 2^200 * (1 + 2^-53 + 2^-GAP): a wide BigFloat whose exact Float64
# value is strictly above the halfway point 2^200*(1+2^-53), so correct
# rounding gives 2^200*(1+2^-52). A fixed GAP-1 = 6656-bit truncation drops
# the 2^-6510 bit, lands exactly on the halfway point, and ties-to-even
# rounds down to 2^200. The two results differ observably.
const WIDE_SCALE = setprecision(BigFloat, GAP + 64) do
    BigFloat(2)^200 * (1 + BigFloat(2)^-53 + BigFloat(2)^-GAP)
end
const EXPECTED = Float64(2)^200 * (1 + Float64(2)^-52)
const TRUNCATED = Float64(2)^200

@testset "Local mass mul! retains actual dyadic input precision" begin
    gap = GAP
    x = setprecision(BigFloat, gap + 2) do
        large = ldexp(BigFloat(1), gap)
        [large + 1, -large]
    end
    original = copy(x)
    mass = LocalMassMatrix(2, [1, 1, 2, 2], [1, 2, 1, 2], ones(4))
    for mode in (RoundNearest, RoundDown, RoundUp),
        operator in (mass, adjoint(mass))
        oldprecision = precision(BigFloat)
        oldrounding = rounding(BigFloat)
        setrounding(BigFloat, mode) do
            setprecision(BigFloat, precision(Float64)) do
                y = zeros(2)
                mul!(y, operator, x)
                @test y == ones(2)
                @test precision(BigFloat) == precision(Float64)
                @test rounding(BigFloat) == mode
                y = [Inf, NaN]
                mul!(y, operator, x, 1.0, 0.0)
                @test y == ones(2)
                complex_y = zeros(ComplexF64, 2)
                mul!(complex_y, operator, x, 1 + im, 0.0)
                @test complex_y == fill(1 + im, 2)
                @test x == original
            end
        end
        @test precision(BigFloat) == oldprecision
        @test rounding(BigFloat) == oldrounding
    end
    large = big(1) << gap
    integer_x = [large + 1, -large]
    y = zeros(2)
    mul!(y, mass, integer_x)
    @test y == ones(2)
    @test integer_x == [large + 1, -large]
    ordinary_x = [2.0, -1.0]
    ordinary_y = zeros(2)
    mul!(ordinary_y, mass, ordinary_x)
    @test ordinary_y == ones(2)
    @test mass.vals == ones(4)
end

@testset "mul! halfway case: wide alpha survives rounding" begin
    # alpha wide+extreme -> fallback; exact alpha*1*1 = WIDE_SCALE.
    mass = LocalMassMatrix(1, [1], [1], [1.0])
    y = zeros(1)
    mul!(y, mass, [1.0], WIDE_SCALE, 0.0)
    @test y[1] == EXPECTED
    @test y[1] != TRUNCATED
end

@testset "scale! path retains input precision" begin
    # mul!(y, M, x, 0, beta): beta wide+extreme -> BigFloat scaling path.
    mass = LocalMassMatrix(1, [1], [1], [0.0])
    y = [1.0]
    mul!(y, mass, [0.0], 0.0, WIDE_SCALE)
    @test y[1] == EXPECTED
    @test y[1] != TRUNCATED
end

@testset "checked number product retains input precision" begin
    got = DiffMoM._checked_number_product(
        Float64, WIDE_SCALE, 1.0, "test", 1)
    @test got == EXPECTED
    @test got != TRUNCATED
end

@testset "scaled matrix add retains input precision" begin
    # site 6/7: Y += alpha * A through _add_scaled_matrix!
    A = LocalMassMatrix(1, [1], [1], [1.0])
    Y = fill(0.0, 1, 1)
    DiffMoM._add_scaled_matrix!(Y, WIDE_SCALE, A)
    @test Y[1, 1] == EXPECTED
    @test Y[1, 1] != TRUNCATED

    # generic dense variant
    Ag = fill(1.0, 1, 1)
    Yg = fill(0.0, 1, 1)
    DiffMoM._add_scaled_matrix!(Yg, WIDE_SCALE, Ag)
    @test Yg[1, 1] == EXPECTED
    @test Yg[1, 1] != TRUNCATED
end

@testset "multi-matrix sum retains input precision" begin
    A = LocalMassMatrix(1, [1], [1], [1.0])
    out = zeros(1, 1)
    DiffMoM._accumulate_scaled_matrices!(
        out, nothing, AbstractMatrix[A], i -> WIDE_SCALE, "test")
    @test out[1, 1] == EXPECTED
    @test out[1, 1] != TRUNCATED

    # generic dense variant
    outg = zeros(1, 1)
    DiffMoM._accumulate_scaled_matrices!(
        outg, nothing, AbstractMatrix[fill(1.0, 1, 1)],
        i -> WIDE_SCALE, "test")
    @test outg[1, 1] == EXPECTED
    @test outg[1, 1] != TRUNCATED

    # with a base contribution
    outb = zeros(1, 1)
    DiffMoM._accumulate_scaled_matrices!(
        outb, fill(0.0, 1, 1), AbstractMatrix[A],
        i -> WIDE_SCALE, "test")
    @test outb[1, 1] == EXPECTED

    # scale_at contract: the fallback memoizes once per matrix. A two-entry
    # output gives one ordinary-path attempt call + one fallback memoized
    # call = 2 total; the old per-populated-index scheme would produce 3.
    A2 = LocalMassMatrix(2, [1, 2], [1, 2], [1.0, 1.0])
    out2 = zeros(2, 2)
    calls = 0
    DiffMoM._accumulate_scaled_matrices!(
        out2, nothing, AbstractMatrix[A2], i -> (calls += 1; WIDE_SCALE),
        "test")
    @test out2[1, 1] == EXPECTED
    @test out2[2, 2] == EXPECTED
    @test calls == 2
end

@testset "loaded block-diagonal entry retains input precision" begin
    # The negation must run at high precision: negating WIDE_SCALE at the
    # ambient precision would itself truncate the low bit before the exact
    # path ever sees it.
    neg_scale = setprecision(BigFloat, GAP + 64) do
        -(BigFloat(2)^200 * (1 + BigFloat(2)^-53 + BigFloat(2)^-GAP))
    end
    matrices = AbstractMatrix[fill(neg_scale, 1, 1)]
    theta = ones(1)
    block = zeros(ComplexF64, 1, 1)
    work = Ref(0)
    DiffMoM._load_block_diag_matrix!(
        block, matrices, theta, false, [1], work, 100_000_000)
    # result = 0 - (1 * -WIDE_SCALE) = WIDE_SCALE -> EXPECTED
    @test block[1, 1] == ComplexF64(EXPECTED)
    @test real(block[1, 1]) != TRUNCATED
    # exact-work accounting must charge the derived precision, not the
    # legacy constant: derived span is ~GAP+64 bits here (> legacy).
    @test work[] > 2 * DiffMoM._LOCAL_MASS_FALLBACK_PRECISION
end

@testset "nondyadic inputs still take the legacy fallback" begin
    # Rational and Irrational inputs are not binary-exponent values; the
    # derived-precision scan must bail to _LOCAL_MASS_FALLBACK_PRECISION.
    @test DiffMoM._local_mass_dyadic_span((1 // 3,)) === nothing
    @test DiffMoM._local_mass_dyadic_span((pi,)) === nothing
    @test DiffMoM._local_mass_dyadic_span((1.5,)) == (-52, 0)
    @test DiffMoM._local_mass_dyadic_span((BigInt(7),)) == (0, 2)
    @test DiffMoM._local_mass_dyadic_span((big"1.5",)) ==
          (0 - precision(BigFloat) + 1, 0)
    @test DiffMoM._local_mass_dyadic_span((Inf,)) === nothing
    @test DiffMoM._local_mass_dyadic_span((NaN,)) === nothing
    @test DiffMoM._local_mass_dyadic_span(()) == (0, 0)
    @test DiffMoM._local_mass_dyadic_span((0.0,)) == (0, 0)

    # a Rational x-vector with an extreme entry still completes via the
    # bounded legacy path (documented precision contract)
    mass = LocalMassMatrix(1, [1], [1], [1.0])
    yr = zeros(1)
    mul!(yr, mass, Rational{BigInt}[big(2)^200 // 1], 1.0, 0.0)
    @test yr[1] == Float64(2)^200
end

@testset "Interval spacing uses exact rational rounding" begin
    @test DiffMoM._interval_spacing(0.0, 1.0, 3, "t") == 1 / 3
    @test DiffMoM._interval_spacing(0.0, 2.0, 4, "t") == 0.5
    # 2*(1e308/7) is correctly rounded: the division rounds once and the
    # exact power-of-two scaling commutes with round-to-nearest.
    @test DiffMoM._interval_spacing(-1e308, 1e308, 7, "t") == 2 * (1e308 / 7)
    # halfway between subnormals rounds ties-to-even:
    # (3*ulp)/2 = 1.5*ulp halfway -> even mantissa -> 2*ulp
    @test DiffMoM._interval_spacing(0.0, 3 * 5e-324, 2, "t") == 2 * 5e-324
    # exact positive subnormal result is preserved
    @test DiffMoM._interval_spacing(0.0, 5e-324, 1, "t") == 5e-324
    # result underflowing to zero still violates the positive-spacing contract
    @test_throws ArgumentError DiffMoM._interval_spacing(0.0, 5e-324, 3, "t")
    @test_throws ArgumentError DiffMoM._interval_spacing(0.0, 1.0, 0, "t")
    @test_throws ArgumentError DiffMoM._interval_spacing(1.0, 0.0, 3, "t")
    @test_throws ArgumentError DiffMoM._interval_spacing(0.0, 1.0, -2, "t")
end

@testset "derived precision respects the MPFR memory budget" begin
    # A crafted exponent span must produce a catchable ArgumentError rather
    # than asking MPFR for an unallocatable precision (which aborts the
    # process). The effective budget is bounded below by 65536 bits so that
    # every IEEE-domain span (~2100 bits) remains derivable even when
    # free_memory reports nothing useful. Sys.free_memory is re-queried
    # inside the helper, so the rejection cases use spans no machine can
    # satisfy: 2^50 bits needs ~4.5 PiB free memory, and a span beyond
    # typemax(Int) is unaddressable on any budget.
    @test_throws ArgumentError DiffMoM._local_mass_dyadic_precision(
        Int128(2)^50, Int128(0), 4)
    @test_throws ArgumentError DiffMoM._local_mass_dyadic_precision(
        Int128(typemax(Int)), Int128(0), 4)
    @test_throws ArgumentError DiffMoM._local_mass_dyadic_precision(
        Int128(10), Int128(0), 0)
    # ordinary IEEE-domain spans are always derivable
    @test DiffMoM._local_mass_dyadic_precision(1024, -1074, 8) >= 2099
    # mid-range spans still derive rather than bail out
    @test DiffMoM._local_mass_dyadic_precision(2^13, 0, 4) == 8194

    # End to end: a BigFloat whose exponent reaches MPFR's clamped maximum
    # must fail with a catchable exception — OverflowError when the derived
    # precision fits the memory budget (exact result is unrepresentable in
    # Float64) or ArgumentError where the budget is smaller — never a crash.
    wide = setprecision(BigFloat, 64) do
        ldexp(BigFloat(1), 2^29) + 1
    end
    mass = LocalMassMatrix(1, [1], [1], [1.0])
    y = zeros(1)
    @test_throws Union{ArgumentError, OverflowError} mul!(
        y, mass, [1.0], wide, 0.0)
end
end
