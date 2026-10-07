module AvailableMemoryBudgetTests
using DiffMoM, Test, LinearAlgebra, SHA
function run_tests()
    root = dirname(@__DIR__)
    fixtures = joinpath(@__DIR__,"fixtures")
    hashes() = Dict(relpath(joinpath(directory,file),root)=>bytes2hex(sha256(read(joinpath(directory,file))))
        for (directory,_,files) in walkdir(joinpath(root,"src")) for file in files)
    before = hashes()
    original_settings = (precision(BigFloat),rounding(BigFloat))
    allocations = Dict{String,Int}()
    observed = Int[]
    @testset "Available process memory defaults and preserved explicit budgets" begin
        for value in (false, zero(Int), -one(Int), -BigInt(typemax(Int)))
            @test_throws ArgumentError DiffMoM._default_max_dense_payload_bytes(value)
        end
        for value in (true, one(Int), typemax(Int), UInt64(typemax(Int)),
                BigInt(typemax(Int))+1, typemax(UInt64), typemax(UInt128))
            @test DiffMoM._default_max_dense_payload_bytes(value) === Int(min(BigInt(value), BigInt(typemax(Int))))
        end
        for value in (one(Int), UInt64(typemax(Int)))
            DiffMoM._default_max_dense_payload_bytes(value)
            bytes = @allocated DiffMoM._default_max_dense_payload_bytes(value)
            allocations[string(typeof(value))] = bytes
            @test bytes == 0
        end
        for _ in 1:length((Int, UInt64))
            value = DiffMoM._default_max_dense_payload_bytes()
            @test value isa Int && 0 < value <= typemax(Int)
            push!(observed, value)
        end
        DiffMoM._default_max_dense_payload_bytes()
        allocations["os_query"] = @allocated DiffMoM._default_max_dense_payload_bytes()
        @test allocations["os_query"] == 0
        capacity = first(observed)
        @test DiffMoM._enforce_payload_limit(capacity, capacity, "control", "max_bytes") == capacity
        @test_throws ArgumentError DiffMoM._enforce_payload_limit(capacity, capacity-1, "control", "max_bytes")
        @test_throws ArgumentError DiffMoM._enforce_payload_limit(1, BigInt(typemax(Int))+1, "control", "max_bytes")

        # Reuse an existing original-suite physical model and its modes.
        grid = CellGrid(.002, .002, 4, 4)
        stack = PlanarStackup([PlanarLayer(1.,1.,.001), PlanarLayer(1.,1.,.001)],
            TERM_GND, TERM_SPACE, grid.a, grid.b)
        sheet = sheet_level(1,4,4)
        rasterize_rect!(sheet,grid,0,grid.a,0,grid.b)
        prob = build_planar_problem(stack,grid,[sheet],[PlanarPort(1,:x,2,2:2,50.)])
        saved_mask = copy(sheet.mask)
        for method in (:dense, :dense_fft)
            default = solve_planar(prob,1e9;mx=12,my=12,surface_zs=.1,method)
            explicit = solve_planar(prob,1e9;mx=12,my=12,surface_zs=.1,method,max_bytes=capacity)
            @test default.currents == explicit.currents && default.y == explicit.y && default.s == explicit.s
            @test default.relative_residuals == explicit.relative_residuals
            @test_throws ArgumentError solve_planar(prob,1e9;mx=12,my=12,method,max_bytes=sizeof(ComplexF64)-1)
            default_maps = planar_current_maps(default)
            explicit_maps = planar_current_maps(default;max_bytes=capacity)
            @test all(all(getproperty(a,k)==getproperty(b,k) for k in (:jx,:jy,:jz)) for (a,b) in zip(default_maps,explicit_maps))
            @test_throws ArgumentError planar_current_maps(default;max_bytes=sizeof(ComplexF64)-1)
            default_field = planar_farfield(default;theta=[.4],phi=[.2])
            explicit_field = planar_farfield(default;theta=[.4],phi=[.2],max_bytes=capacity)
            @test default_field.etheta == explicit_field.etheta && default_field.ephi == explicit_field.ephi && default_field.intensity == explicit_field.intensity
            @test_throws ArgumentError planar_farfield(default;theta=[.4],phi=[.2],max_bytes=sizeof(ComplexF64)-1)
            contraction = Matrix{Float64}(I,length(prob.ports),length(prob.ports))
            source_default = solve_planar_contracted(prob,1e9,contraction;mx=12,my=12,surface_zs=.1,method)
            source_explicit = solve_planar_contracted(prob,1e9,contraction;mx=12,my=12,surface_zs=.1,method,max_bytes=capacity)
            @test source_default.currents == source_explicit.currents && source_default.y == source_explicit.y && source_default.s == source_explicit.s
            @test_throws ArgumentError solve_planar_contracted(prob,1e9,contraction;mx=12,my=12,method,max_bytes=sizeof(ComplexF64)-1)
        end
        @test sheet.mask == saved_mask

        native = joinpath(fixtures,"native_box_port_attachment","cases","attachment__baseline","project.son")
        native_before = bytes2hex(sha256(read(native)))
        project = read_sonnet_project(native)
        default = solve_sonnet_project(native,1e9;raw=true,grid=(8,8),mx=8,my=8)
        explicit = solve_sonnet_project(project,1e9;raw=true,grid=(8,8),mx=8,my=8,max_bytes=capacity)
        @test default.s == explicit.s && default.y == explicit.y && default.raw.currents == explicit.raw.currents
        for input in (native, project), value in (zero(Int), -one(Int), BigInt(typemax(Int))+1, typemax(UInt128))
            @test_throws ArgumentError solve_sonnet_project(input,1e9;raw=true,max_bytes=value)
        end
        @test bytes2hex(sha256(read(native))) == native_before
        mktempdir() do directory
            path = joinpath(directory,"dc_resistor.son")
            write(path,"FTYP SONNETPRJ 19\nDIM\nFREQ HZ\nRES OH\nCAP PF\nIND NH\nLNG MM\nEND DIM\nCKT\nRES 1 0 R=50\nDEF1P 1 main R 50\nEND CKT\n")
            circuit = read_sonnet_project(path)
            a, b = solve_sonnet_project(path,0.), solve_sonnet_project(circuit,0.;max_bytes=capacity)
            @test a.s == b.s && a.y == b.y
            @test_throws ArgumentError solve_sonnet_project(path,0.;max_bytes=0)
            @test_throws ArgumentError solve_sonnet_project(joinpath(directory,"missing.son"),0.;max_bytes=0)
        end
        @test (precision(BigFloat),rounding(BigFloat)) == original_settings
        @test hashes() == before
    end
end
run_tests()
end
