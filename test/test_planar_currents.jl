using DiffMoM, Test, LinearAlgebra

function _current_map_fixture()
    grid = CellGrid(2e-3, 2e-3, 2, 2)
    stack = PlanarStackup([PlanarLayer(1.0, 1.0, 1e-3),
        PlanarLayer(1.0, 1.0, 1e-3)], TERM_GND, TERM_GND, grid.a, grid.b)
    sheet = sheet_level(1, 2, 2)
    sheet.mask .= true
    sheet.connect_west .= true
    sheet.connect_east .= true
    vol = vol_level(1, 2, 2)
    vol.mask .= true
    via = via_level(2, 2, 2)
    via.uni[1, 1] = true
    via.tap[1, 1] = true
    prob = build_planar_problem(stack, grid, [sheet],
        [PlanarPort(1, :west, 1:2, 50.0),
         PlanarPort(1, :east, 1:2, 75.0)]; vols=[vol], vias=[via])
    nb = planar_basis_count(prob.basis)
    X = zeros(ComplexF64, nb, 2)
    for p in 1:nb
        kind = prob.basis.kind[p]
        if kind == DiffMoM._BASIS_X_FULL
            X[p, 1] = 2 + 4im
        elseif kind == DiffMoM._BASIS_Y_FULL
            X[p, 1] = 6 - 2im
        elseif kind == DiffMoM._BASIS_VX_FULL
            X[p, 1] = 0.004 + 0.002im
        elseif kind == DiffMoM._BASIS_VY_FULL
            X[p, 1] = 0.006 - 0.004im
        elseif kind == DiffMoM._BASIS_VIA_U
            X[p, 1] = 8 - 4im
        elseif kind == DiffMoM._BASIS_VIA_T
            X[p, 1] = 4 + 8im
        end
    end
    X[:, 2] .= -2im .* X[:, 1]
    return PlanarResult(prob, ComplexF64(2pi * 1e9), ComplexF64(1e9),
        nothing, lu(Matrix{ComplexF64}(I, nb, nb)), X,
        zeros(ComplexF64, 2, 2), ComplexF64[0.1 0.2; 0.3 0.4])
end

@testset "planar: physical current maps" begin
    r = _current_map_fixture()
    maps = planar_current_maps(r)
    @test [m.kind for m in maps] == [:sheet, :volume, :via]
    @test all(maps[1].jx .== 1 + 2im)
    @test all(maps[1].jy .== 3 - 1im)
    @test all(iszero, maps[1].jz)
    @test all(maps[2].jx .== 2 + 1im)
    @test all(maps[2].jy .== 3 - 2im)
    @test maps[2].z == 0.5e-3
    @test maps[3].z == 1.5e-3
    @test maps[3].jz[1, 1] ≈ 10 + 0im
    @test count(!iszero, maps[3].jz) == 1
    @test planar_current_maps(r; z_fraction=0)[3].jz[1, 1] == 8 - 4im
    @test planar_current_maps(r; z_fraction=1)[3].jz[1, 1] == 12 + 4im
    other = planar_current_maps(r; port=2)
    @test other[1].jx ≈ -2im .* maps[1].jx
    v = [2 + 3im, 4 - 1im]
    combined = planar_current_maps(r; voltages=v)
    @test combined[1].jx ≈ (v[1] - 2im*v[2]) .* maps[1].jx
    waves = ComplexF64[0.2 + 0.3im, 0.1 - 0.2im]
    volts = sqrt.([50.0, 75.0]) .* (waves + r.s * waves)
    power_maps = planar_current_maps(r; incident_waves=waves)
    voltage_maps = planar_current_maps(r; voltages=volts)
    @test power_maps[1].jx ≈ voltage_maps[1].jx
    @test_throws ArgumentError planar_current_maps(r; voltages=v, incident_waves=waves)
    @test_throws DimensionMismatch planar_current_maps(r; voltages=[1.0])
    @test_throws ArgumentError planar_current_maps(r; voltages=[NaN, 1.0])
    @test_throws ArgumentError planar_current_maps(r; port=3)
    @test_throws ArgumentError planar_current_maps(r; z_fraction=1.1)
    @test_throws ArgumentError planar_current_maps(r; max_bytes=1)
    mktempdir() do dir
        path = joinpath(dir, "currents.csv")
        @test write_planar_current_csv(path, maps) == path
        rows = readlines(path)
        @test length(rows) == 1 + 4 + 4 + 1
        @test occursin("jx_re,jx_im,jy_re,jy_im,jz_re,jz_im", rows[1])
        @test split(rows[2], ',')[10] == "A/m"
        @test split(rows[end], ',')[10] == "A/m^2"
        @test_throws ArgumentError write_planar_current_csv(path, PlanarCurrentMap[])
        @test readlines(path) == rows
    end
end
