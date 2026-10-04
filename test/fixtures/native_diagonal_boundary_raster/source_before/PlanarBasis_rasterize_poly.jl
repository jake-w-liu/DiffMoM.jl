function rasterize_poly!(sheet::Union{SheetLevel,VolLevel}, grid::CellGrid,
        xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real})
    _validate_sheet_grid(sheet, grid)
    nv = length(xs)
    nv == length(ys) && nv >= 3 ||
        throw(ArgumentError("polygon needs >= 3 vertices"))
    all(isfinite, xs) && all(isfinite, ys) ||
        throw(ArgumentError("polygon vertices must be finite"))
    @inbounds for j in 1:grid.ny
        yc = (j - 0.5) * grid.dy
        for i in 1:grid.nx
            xc = (i - 0.5) * grid.dx
            inside = false
            k = nv
            for v in 1:nv
                if (ys[v] > yc) != (ys[k] > yc)
                    xint = xs[k] + (xs[v] - xs[k]) * (yc - ys[k]) /
                                   (ys[v] - ys[k])
                    xc < xint && (inside = !inside)
                end
                k = v
            end
            inside && (sheet.mask[i, j] = true)
        end
    end
    return sheet
end

