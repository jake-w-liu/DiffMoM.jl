# Ohmic Galerkin reactions for finite-conductivity volume and via bases.
# With Z_mom = E_induced and RHS = -E_applied, passive metal contributes
# -rho * integral(f_p . f_q dV), matching the sheet surface-impedance sign.

function _planar_bulk_resistivities(sigma, count::Int, what::String)
    sigma isa AbstractVector && length(sigma) != count &&
        throw(DimensionMismatch("$what count must match the conductor levels"))
    rho = Vector{ComplexF64}(undef, count)
    for lv in 1:count
        value = sigma isa AbstractVector ? sigma[lv] : sigma
        if value isa Number && real(value) == Inf && iszero(imag(value))
            rho[lv] = 0
        else
            value isa Number && isfinite(value) && real(value) >= 0 && !iszero(value) ||
                throw(ArgumentError("$what[$lv] must be finite and nonzero with Re ≥ 0, or +Inf"))
            stored = ComplexF64(value)
            isfinite(stored) && real(stored) >= 0 && !iszero(stored) || throw(ArgumentError(
                "$what[$lv] must remain finite and nonzero with Re ≥ 0 in ComplexF64"))
            resistivity = inv(stored)
            isfinite(resistivity) || throw(ArgumentError(
                "$what[$lv] has a resistivity that is not finite in ComplexF64"))
            rho[lv] = resistivity
        end
    end
    return rho
end

function _planar_bulk_loss_entries(emit,
        basis::PlanarBasisSet, grid::CellGrid, stack::PlanarStackup,
        vias::Vector{ViaLevel}, vols::Vector{VolLevel},
        via_rho::Vector{ComplexF64}, volume_rho::Vector{ComplexF64})
    all(iszero, via_rho) && all(iszero, volume_rho) && return nothing
    lookup = Dict{NTuple{4,Int},Int}()
    for p in eachindex(basis.kind)
        kind = basis.kind[p]
        (_is_via_kind(kind) || _is_vol_kind(kind)) || continue
        lookup[(basis.level[p], Int(kind), basis.ei[p], basis.ej[p])] = p
    end
    for p in eachindex(basis.kind)
        kind, lv = basis.kind[p], basis.level[p]
        if _is_via_kind(kind)
            rho = via_rho[lv]
            iszero(rho) && continue
            h = stack.layers[vias[lv].layer].thickness
            g = -rho * grid.dx * grid.dy
            emit(p, p, g * _via_overlap(kind, kind, h), vias[lv].layer, 1)
            if kind == _BASIS_VIA_U
                q = get(lookup, (lv, Int(_BASIS_VIA_T),
                    basis.ei[p], basis.ej[p]), 0)
                if q != 0
                    value = g * h / 2
                    emit(p, q, value, vias[lv].layer, 1)
                    emit(q, p, value, vias[lv].layer, 1)
                end
            end
        elseif _is_vol_kind(kind)
            rho = volume_rho[lv]
            iszero(rho) && continue
            h = stack.layers[vols[lv].layer].thickness
            # Volume profiles are normalized to integral(w_z dz)=1.
            g = -rho / h
            k = _sheet_kind(kind)
            emit(p, p, g * _gram_self(k, grid), vols[lv].layer, -1)
            e, j = basis.ei[p], basis.ej[p]
            q = if k in (_BASIS_X_FULL, _BASIS_X_LO, _BASIS_X_HI)
                e2 = k == _BASIS_X_HI ? e - 1 : e + 1
                neighbor = get(lookup, (lv, Int(_BASIS_VX_FULL), e2, j), 0)
                if neighbor == 0 && k == _BASIS_X_LO && grid.nx == 1
                    neighbor = get(lookup, (lv, Int(_BASIS_VX_HI), grid.nx, j), 0)
                end
                neighbor
            else
                j2 = k == _BASIS_Y_HI ? j - 1 : j + 1
                neighbor = get(lookup, (lv, Int(_BASIS_VY_FULL), e, j2), 0)
                if neighbor == 0 && k == _BASIS_Y_LO && grid.ny == 1
                    neighbor = get(lookup, (lv, Int(_BASIS_VY_HI), e, grid.ny), 0)
                end
                neighbor
            end
            if q != 0
                value = g * _gram_adj(grid)
                emit(p, q, value, vols[lv].layer, -1)
                emit(q, p, value, vols[lv].layer, -1)
            end
        end
    end
    return nothing
end

function _add_planar_bulk_loss!(Z::Matrix{ComplexF64},
        basis::PlanarBasisSet, grid::CellGrid, stack::PlanarStackup,
        vias::Vector{ViaLevel}, vols::Vector{VolLevel},
        via_rho::Vector{ComplexF64}, volume_rho::Vector{ComplexF64})
    _planar_bulk_loss_entries(basis, grid, stack, vias, vols,
            via_rho, volume_rho) do p, q, value, layer, power
        Z[p, q] += value
    end
    return Z
end

function _planar_bulk_loss_gradient!(g, params, G, lambda, X,
        prob, via_sigma, volume_sigma)
    via_rho = _planar_bulk_resistivities(via_sigma, length(prob.vias), "via_sigma")
    volume_rho = _planar_bulk_resistivities(volume_sigma, length(prob.vols), "volume_sigma")
    _planar_bulk_loss_entries(prob.basis, prob.grid, prob.stack,
            prob.vias, prob.vols, via_rho, volume_rho) do p, q, value, layer, power
        contraction = zero(ComplexF64)
        for b in axes(G, 2), a in axes(G, 1)
            contraction += G[a, b] * lambda[p, a] * X[q, b]
        end
        derivative = power * value / prob.stack.layers[layer].thickness
        for j in eachindex(params)
            param = params[j]
            param.index == layer && param.field === :thickness || continue
            dh = param.part === :re ? 1.0 : 1im
            g[j] -= real(contraction * derivative * dh)
        end
    end
    return g
end
