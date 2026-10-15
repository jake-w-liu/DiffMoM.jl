module FocusedResourceTests
using DiffMoM, Test

function payload_allocations()
    DiffMoM._checked_payload_sum("focused", 16, 32, 64)
    @allocated DiffMoM._checked_payload_sum("focused", 16, 32, 64)
end

@testset "Payload arithmetic and budget boundaries" begin
    limit = typemax(Int)
    @test DiffMoM._checked_payload_sum("focused") == 0
    @test DiffMoM._checked_payload_sum("focused", 16, 32, 64) == 112
    @test DiffMoM._checked_payload_sum("focused", UInt128(limit - 1), Int8(1)) == limit
    @test DiffMoM._checked_payload_sum("focused", BigInt(limit - 1), UInt8(1)) == limit
    @test_throws ArgumentError DiffMoM._checked_payload_sum("focused", limit, 1)
    @test_throws ArgumentError DiffMoM._checked_payload_sum("focused", BigInt(limit) + 1)
    negative = try
        DiffMoM._checked_payload_sum("focused", typemax(UInt128), -1)
    catch error
        error
    end
    @test negative isa ArgumentError && occursin("nonnegative", sprint(showerror, negative))
    @test payload_allocations() == 0

    @test_throws ArgumentError DiffMoM._default_max_dense_payload_bytes(0)
    @test_throws ArgumentError DiffMoM._default_max_dense_payload_bytes(-1)
    @test DiffMoM._default_max_dense_payload_bytes(1) === 1
    @test DiffMoM._default_max_dense_payload_bytes(typemax(UInt128)) === limit
    @test 0 < DiffMoM._default_max_dense_payload_bytes() <= limit
    @test DiffMoM._enforce_payload_limit(128, 128, "focused", "max_bytes") == 128
    @test_throws ArgumentError DiffMoM._enforce_payload_limit(128, 127, "focused", "max_bytes")
end
end
