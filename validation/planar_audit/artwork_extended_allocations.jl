using DiffMoM,TOML,SHA
const audit_directory = normpath(joinpath(@__DIR__, "..", "..", "data", "planar_audit"))
const audit_output = isempty(ARGS) ? joinpath(audit_directory, "artwork_extended_allocations.toml") : abspath(only(ARGS))
mkpath(dirname(audit_output))

reference=joinpath(@__DIR__,"..","..","test","fixtures","odb_compress","product.tar.Z")
compressed=read(reference)
decode()=DiffMoM._odb_uncompress(IOBuffer(compressed),devnull,32*1024^2)
decode();GC.gc();decoder_allocated=@allocated expanded=decode()
mktempdir() do directory
    path=joinpath(directory,"arc.gbr")
    write(path,"%FSLAX36Y36*%%MOMM*%%ADD10R,0.8X0.4*%D10*G75*X2000000Y0D02*G03*X2000000Y0I-2000000J0D01*M02*")
    doc=read_gerber(path);grid=CellGrid(.006,.006,400,400)
    raster()=artwork_cell_masks(doc,grid;offset=(.003,.003))
    raster();GC.gc();raster_allocated=@allocated masks=raster()
    artifact=Dict("scope"=>"warmed cumulative Julia allocation; process RSS and opaque codec/archive buffers excluded",
        "decoder"=>Dict("compressed_bytes"=>length(compressed),"expanded_bytes"=>expanded,
            "allocated_bytes"=>decoder_allocated,"destination"=>"devnull; output file bytes are bounded separately",
            "compressed_sha256"=>bytes2hex(sha256(compressed))),
        "rectangle_arc_raster"=>Dict("allocated_bytes"=>raster_allocated,"grid_cells"=>160000,
            "returned_payload_bytes"=>sizeof(only(values(masks)).chunks),"occupied_cells"=>count(only(values(masks)))))
    open(audit_output,"w") do io;TOML.print(io,artifact);end
    println(artifact)
end
