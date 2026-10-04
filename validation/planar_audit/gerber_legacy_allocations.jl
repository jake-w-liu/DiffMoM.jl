using DiffMoM,TOML,Dates
const audit_directory = normpath(joinpath(@__DIR__, "..", "..", "data", "planar_audit"))
const audit_output = isempty(ARGS) ? joinpath(audit_directory, "gerber_legacy_allocations.toml") : abspath(only(ARGS))
mkpath(dirname(audit_output))

function main()
    mktempdir() do dir
        file=joinpath(dir,"ellipse.gbr")
        write(file,"%FSLAX36Y36*%\n%MOMM*%\n%SFA2B.5*%\n%ADD10C,.2*%\n"*
            "D10*X1000000Y0D02*\nG75*G03X1000000Y0I-1000000J0D01*\nM02*\n")
        doc=read_gerber(file);shape=only(doc.objects).shape
        grid=CellGrid(.0046,.0016,400,400);offset=(.0023,.0008)
        artwork_cell_masks(doc,grid;offset)
        mask=only(values(artwork_cell_masks(doc,grid;offset)))
        bytes=@allocated artwork_cell_masks(doc,grid;offset)
        member=@allocated DiffMoM._artwork_contains(shape,.001,.0001)
        rows=Dict("utc"=>string(now(UTC)),"julia_version"=>string(VERSION),
            "scope"=>"warmed Julia cumulative allocation, returned BitMatrix included; not peakRSS/native runtime",
            "grid_cells"=>length(mask),"mask_payload_bytes"=>cld(length(mask),64)*8,
            "occupied_cells"=>count(mask),"raster_allocation_bytes"=>bytes,
            "membership_allocation_bytes"=>member,
            "before_type_fix_80x80_allocation_bytes"=>334597096)
        open(audit_output,"w") do io;TOML.print(io,rows);end
        println(rows)
        bytes<100_000 || error("ellipse raster allocates per cell")
        member==0 || error("ellipse membership allocates")
    end
end
main()
