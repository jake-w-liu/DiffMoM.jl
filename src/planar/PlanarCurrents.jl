export PlanarCurrentMap, planar_current_maps, write_planar_current_csv

"""Cell-centre current samples for one sheet, volume or via level.
`jx`, `jy`, `jz` are phasors in A/m for `kind=:sheet`, and A/m² for
`kind=:volume` or `:via`. `z` is the sampled height, while `zmin:zmax`
identifies the level's extent. `level` indexes the corresponding problem
array; `mask` marks conductor cells. Each array is indexed `[i,j]`.
"""
struct PlanarCurrentMap
    kind::Symbol
    level::Int
    x::Vector{Float64}
    y::Vector{Float64}
    z::Float64
    zmin::Float64
    zmax::Float64
    mask::BitMatrix
    jx::Matrix{ComplexF64}
    jy::Matrix{ComplexF64}
    jz::Matrix{ComplexF64}
end

function _planar_excitation_voltages(result::Union{PlanarResult,PlanarUFFTResult}, voltages,
        incident_waves, port::Integer)
    n = length(result.problem.ports)
    if voltages !== nothing && incident_waves !== nothing
        throw(ArgumentError("provide voltages or incident_waves"))
    end
    supplied = voltages === nothing ? incident_waves : voltages
    if supplied === nothing
        1 <= port <= n || throw(ArgumentError("port outside 1:$n"))
        v = zeros(ComplexF64, n)
        v[port] = 1
        return v
    end
    supplied isa AbstractVector && length(supplied) == n ||
        throw(DimensionMismatch("excitation must contain $n port values"))
    all(isfinite, supplied) || throw(ArgumentError("excitation must be finite"))
    v = ComplexF64.(supplied)
    if incident_waves !== nothing
        v = _planar_wave_voltage(result.s,v,result.z0)
    end
    return v
end

function _planar_map_add_rooftop!(map::PlanarCurrentMap,
        basis::PlanarBasisSet, p::Int, coefficient::ComplexF64,
        grid::CellGrid, scale::Float64)
    k = _sheet_kind(basis.kind[p])
    i, j = basis.ei[p], basis.ej[p]
    value = 0.5 * scale * coefficient
    if k == _BASIS_X_FULL
        map.jx[i, j] += value
        map.jx[i + 1, j] += value
    elseif k == _BASIS_X_LO
        map.jx[i + 1, j] += value
    elseif k == _BASIS_X_HI
        map.jx[i, j] += value
    elseif k == _BASIS_Y_FULL
        map.jy[i, j] += value
        map.jy[i, j + 1] += value
    elseif k == _BASIS_Y_LO
        map.jy[i, j + 1] += value
    elseif k == _BASIS_Y_HI
        map.jy[i, j] += value
    end
    return nothing
end

"""
    planar_current_maps(result; port=1, voltages=nothing,
                        incident_waves=nothing, z_fraction=0.5, max_bytes=...)

Reconstruct cell-centre phasor currents on every conductor level from
the solved rooftop coefficients. The default drives `port` at one volt,
with all other gap voltages zero. `voltages` specifies all gap voltages.
Alternatively, `incident_waves` specifies Kurokawa incident power waves,
so ports without an incident wave are terminated in their reference
impedance. Complex references use the evaluated `result.z0`. For the
peak phasors used here accepted power is half the incident minus reflected
squared wave norms. Via and volume samples lie at `z_fraction` of their layer
height; tapered via currents vary linearly with this height. Sheet maps
are surface current in A/m; volume and via maps are current density in
A/m². No cell interpolation or smoothing is applied.
"""
function planar_current_maps(result::Union{PlanarResult,PlanarUFFTResult};
        port::Integer=1, voltages=nothing, incident_waves=nothing,
        z_fraction::Real=0.5,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    isfinite(z_fraction) && 0 <= z_fraction <= 1 ||
        throw(ArgumentError("z_fraction must lie in [0,1]"))
    prob = result.problem
    grid = prob.grid
    nb = planar_basis_count(prob.basis)
    nl = length(prob.sheets) + length(prob.vols) + length(prob.vias)
    payload = _checked_payload_sum("planar current maps",
        _checked_array_payload_bytes(ComplexF64, 3, nl, grid.nx, grid.ny),
        _checked_array_payload_bytes(Float64, grid.nx + grid.ny),
        _checked_array_payload_bytes(ComplexF64, nb),
        _checked_array_payload_bytes(ComplexF64, 10, length(prob.ports)),
        _checked_array_payload_bytes(Float64, length(prob.stack.layers) + 1),
        # BitArray allocates complete 64-bit chunks even for a single cell.
        _checked_array_payload_bytes(UInt64, nl,
            cld(_checked_array_payload_bytes(UInt8, grid.nx, grid.ny), 64)))
    _enforce_payload_limit(payload, max_bytes, "planar current maps", "max_bytes")
    v = _planar_excitation_voltages(result, voltages, incident_waves, port)
    coefficients = result.currents * v
    return _planar_current_maps_from_coefficients(prob,coefficients;
        z_fraction,max_bytes=max_bytes-_checked_payload_sum("current excitation",
            _checked_array_payload_bytes(ComplexF64,nb),
            _checked_array_payload_bytes(ComplexF64,10,length(prob.ports))))
end

# Shared reconstruction for solvers that drive physical voltage contracts
# directly, without constructing separate raw-source coefficient columns.
function _planar_current_maps_from_coefficients(prob::PlanarProblem,
        coefficients::AbstractVector;z_fraction::Real=.5,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    isfinite(z_fraction) && 0<=z_fraction<=1 ||
        throw(ArgumentError("z_fraction must lie in [0,1]"))
    grid=prob.grid;nb=planar_basis_count(prob.basis)
    length(coefficients)==nb && all(isfinite,coefficients) ||
        throw(ArgumentError("current coefficients must be finite and match the physical basis"))
    nl=length(prob.sheets)+length(prob.vols)+length(prob.vias)
    payload=_checked_payload_sum("planar current reconstruction",
        _checked_array_payload_bytes(ComplexF64,3,nl,grid.nx,grid.ny),
        _checked_array_payload_bytes(Float64,grid.nx+grid.ny),
        _checked_array_payload_bytes(Float64,length(prob.stack.layers)+1),
        _checked_array_payload_bytes(UInt64,nl,
            cld(_checked_array_payload_bytes(UInt8,grid.nx,grid.ny),64)))
    _enforce_payload_limit(payload,max_bytes,"planar current reconstruction","max_bytes")
    z = zeros(Float64, length(prob.stack.layers) + 1)
    @inbounds for l in eachindex(prob.stack.layers)
        z[l + 1] = z[l] + Float64(real(prob.stack.layers[l].thickness))
    end
    x = [(i - 0.5) * grid.dx for i in 1:grid.nx]
    y = [(j - 0.5) * grid.dy for j in 1:grid.ny]
    maps = PlanarCurrentMap[]
    for (lv, sheet) in enumerate(prob.sheets)
        height = z[sheet.interface + 1]
        push!(maps, PlanarCurrentMap(:sheet, lv, x, y, height, height,
            height, copy(sheet.mask), zeros(ComplexF64, grid.nx, grid.ny),
            zeros(ComplexF64, grid.nx, grid.ny), zeros(ComplexF64, grid.nx, grid.ny)))
    end
    for (kind, levels) in ((:volume, prob.vols), (:via, prob.vias))
        for (lv, level) in enumerate(levels)
            bottom, top = z[level.layer], z[level.layer + 1]
            height = bottom + z_fraction * (top - bottom)
            mask = kind === :volume ? copy(level.mask) : level.uni .| level.tap
            push!(maps, PlanarCurrentMap(kind, lv, x, y, height, bottom,
                top, mask, zeros(ComplexF64, grid.nx, grid.ny),
                zeros(ComplexF64, grid.nx, grid.ny), zeros(ComplexF64, grid.nx, grid.ny)))
        end
    end
    ns = length(prob.sheets)
    nv = length(prob.vols)
    @inbounds for p in 1:nb
        k, lv = prob.basis.kind[p], prob.basis.level[p]
        if _is_via_kind(k)
            map = maps[ns + nv + lv]
            profile = k == _BASIS_VIA_U ? 1.0 : Float64(z_fraction)
            map.jz[prob.basis.ei[p], prob.basis.ej[p]] += coefficients[p] * profile
        elseif _is_vol_kind(k)
            map = maps[ns + lv]
            _planar_map_add_rooftop!(map, prob.basis, p, coefficients[p],
                grid, 1 / (map.zmax - map.zmin))
        else
            _planar_map_add_rooftop!(maps[lv], prob.basis, p,
                coefficients[p], grid, 1.0)
        end
    end
    return maps
end

"""Write current-map conductor cells as CSV with SI coordinates, explicit
current units and real/imaginary components. Empty map collections reject
before opening the destination. Returns `path`."""
function write_planar_current_csv(path::AbstractString,
        maps::AbstractVector{PlanarCurrentMap})
    isempty(maps) && throw(ArgumentError("at least one current map is required"))
    for map in maps
        shape = (length(map.x), length(map.y))
        all(a -> size(a) == shape, (map.mask, map.jx, map.jy, map.jz)) ||
            throw(DimensionMismatch("current-map arrays must match coordinate axes"))
        map.kind in (:sheet, :via, :volume) ||
            throw(ArgumentError("unknown current map kind $(map.kind)"))
        all(isfinite, map.x) && all(isfinite, map.y) &&
            all(isfinite, (map.z, map.zmin, map.zmax)) &&
            all(isfinite, map.jx) && all(isfinite, map.jy) && all(isfinite, map.jz) ||
            throw(ArgumentError("current-map coordinates and currents must be finite"))
    end
    open(path, "w") do io
        println(io, "kind,level,i,j,x_m,y_m,z_m,zmin_m,zmax_m,current_unit," *
            "jx_re,jx_im,jy_re,jy_im,jz_re,jz_im")
        for map in maps, j in eachindex(map.y), i in eachindex(map.x)
            map.mask[i, j] || continue
            unit = map.kind === :sheet ? "A/m" : "A/m^2"
            values = (map.kind, map.level, i, j, map.x[i], map.y[j],
                map.z, map.zmin, map.zmax, unit, real(map.jx[i,j]),
                imag(map.jx[i,j]), real(map.jy[i,j]), imag(map.jy[i,j]),
                real(map.jz[i,j]), imag(map.jz[i,j]))
            println(io, join(values, ','))
        end
    end
    return path
end
