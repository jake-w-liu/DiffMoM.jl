using DiffMoM,Test,TOML,Dates
const audit_directory = normpath(joinpath(@__DIR__, "..", "..", "data", "planar_audit"))
const audit_output = isempty(ARGS) ? joinpath(audit_directory, "artwork_general_composite_allocations.toml") : abspath(only(ARGS))
mkpath(dirname(audit_output))

const DM=DiffMoM

function memberbytes(s::S) where {S<:DM._ArtworkShape}
    DM._artwork_contains(s,.0023,.0007)
    return @allocated DM._artwork_contains(s,.0023,.0007)
end

function main()
    rows=Dict{String,Any}()
    mktempdir() do directory
        file=joinpath(directory,"features")
        for name in ("donut_r6000x4000","donut_s6000x4000",
                "donut_sr6000x4000","donut_rc6000x4000x500","donut_o6000x4000x500")
            write(file,"UNITS=MM\nF 1\n\$0 $(name)\nP 0 0 0 P 0 0\n")
            doc=read_odb_features(file;layer="metal")
            grid=CellGrid(.007,.005,400,400);offset=(.0035,.0025)
            mask=only(values(artwork_cell_masks(doc,grid;offset)))
            bytes=@allocated artwork_cell_masks(doc,grid;offset)
            rows[name]=Dict("raster_bytes"=>bytes,"occupied_cells"=>count(mask),
                "member_bytes"=>memberbytes(only(doc.objects).shape),
                "shape_type"=>string(typeof(only(doc.objects).shape)),
                "owned_shape_bytes"=>Base.summarysize(only(doc.objects).shape))
            println(name," => ",rows[name])
        end
    end
    rows["scope"]="warmed cumulative Julia allocation, including returned masks; no peak process-memory claim"
    rows["utc"]=string(now(UTC));rows["julia_version"]=string(VERSION)
    open(audit_output,"w") do io
        TOML.print(io,rows)
    end
end
main()
