using DiffMoM,TOML
const audit_directory = normpath(joinpath(@__DIR__, "..", "..", "data", "planar_audit"))
const audit_output = isempty(ARGS) ? joinpath(audit_directory, "artwork_allocations.toml") : abspath(only(ARGS))
mkpath(dirname(audit_output))

G=DiffMoM
arc=G._ArtworkRoundArc((1.,0.),(1.,0.),(0.,0.),false,0.)
shape=G._ArtworkRegion(Union{G._ArtworkRoundLine,G._ArtworkRoundArc}[arc])
object=PlanarArtworkObject("metal",shape,true,(-1.,1.,-1.,1.),Dict{String,Vector{String}}())
document=PlanarArtwork("analytic circle",[object],Dict{String,Vector{String}}(),Set{String}(),1.)
grid=CellGrid(2.,2.,400,400)
render()=artwork_cell_masks(document,grid;offset=(1.,1.))
render();GC.gc()
allocated=@allocated render()
elapsed=@elapsed mask=render()
row=Dict("grid_cells"=>grid.nx*grid.ny,"allocated_bytes"=>allocated,
    "elapsed_seconds"=>elapsed,"occupied_cells"=>count(mask["metal"]),
    "returned_payload_bytes"=>sizeof(mask["metal"]),"scope"=>"warmed single analytic circular region raster, cumulative Julia allocations")
println(row)
open(audit_output,"w") do io;TOML.print(io,row);end
