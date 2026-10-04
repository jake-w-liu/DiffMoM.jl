using DiffMoM,LinearAlgebra,SHA,TOML
const repo=normpath(joinpath(@__DIR__,".."))
# Fresh-process prototype only. The production source remains unchanged.
@eval DiffMoM function rasterize_poly!(sheet::Union{SheetLevel,VolLevel},grid::CellGrid,
        xs::AbstractVector{<:Real},ys::AbstractVector{<:Real})
    _validate_sheet_grid(sheet,grid)
    nv=length(xs)
    nv==length(ys) && nv>=3 || throw(ArgumentError("polygon needs >=3 vertices"))
    all(isfinite,xs) && all(isfinite,ys) || throw(ArgumentError("polygon vertices must be finite"))
    @inbounds for j in 1:grid.ny
        yc=(j-.5)*grid.dy
        for i in 1:grid.nx
            xc=(i-.5)*grid.dx;inside=false;k=nv
            for v in 1:nv
                if (xs[v]>xc)!=(xs[k]>xc)
                    yint=ys[k]+(ys[v]-ys[k])*(xc-xs[k])/(xs[v]-xs[k])
                    diagonal_tie=ys[v]!=ys[k] && isfinite(yint) &&
                        abs(yc-yint)<=8eps(Float64)*max(abs(yc),abs(yint),abs(ys[v]),abs(ys[k]))
                    !diagonal_tie && yc<yint && (inside=!inside)
                end
                k=v
            end
            inside && (sheet.mask[i,j]=true)
        end
    end
    sheet
end
function main()
    original=joinpath(repo,"data/sonnet_validation/geovar_zero_nominal_axes_GAt2Wu")
    output=mktempdir(joinpath(repo,"data/planar_audit");prefix="vertical_ray_raster_prototype_",cleanup=false)
    cp(@__FILE__,joinpath(output,"producer.jl"))
    paths=vcat([@__FILE__],[joinpath(dir,file) for (dir,_,files) in walkdir(joinpath(repo,"src")) for file in files])
    hashes()=Dict(path=>bytes2hex(sha256(read(path))) for path in paths)
    report=Dict{String,Any}("scope"=>"Fresh-process vertical ray half-open boundary prototype, ignoring rounded diagonal ties at upper crossings. All eight native physical gates .005/1e-9, original failed baseline and rejected inclusive-diagonal prototype retained.","source_before"=>hashes(),"cases"=>Any[])
    try
        for axis in ("XDIR","YDIR"),direction in (-1,1),target in (.0625,.125)
            tag="$(axis)_dir_$(direction)_target_$(target)"
            p=read_sonnet_project(joinpath(original,tag*"_parameter.son"))
            native=only(planar_read_touchstone(joinpath(original,tag*"_parameter/native_raw.s2p")).s)
            result=solve_sonnet_project(p,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
            raw=result.raw;rhs=zeros(ComplexF64,size(raw.currents))
            for b in eachindex(raw.problem.basis.port)
                port=raw.problem.basis.port[b];iszero(port) && continue
                rhs[b,port]=(port==1 ? -1. : 1.)*raw.problem.basis.width[b]
            end
            row=Dict("case"=>tag,"full_s_error"=>maximum(abs,result.s-native),
                "original_voltage_residual"=>norm(raw.z_mom*raw.currents-rhs)/norm(rhs),
                "occupied_cells"=>count(only(raw.problem.sheets).mask))
            push!(report["cases"],row);println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained diagonal prototype: ",output)
    end
    @assert report["source_unchanged"] && all(row["full_s_error"]<=.005 && row["original_voltage_residual"]<=1e-9 for row in report["cases"])
end
main()
