# Solve-level library acceptance, separate from polygon construction checks.
using DiffMoM,LinearAlgebra,TOML
const audit_directory = normpath(joinpath(@__DIR__, "..", "..", "data", "planar_audit"))
const audit_output = isempty(ARGS) ? joinpath(audit_directory, "rfic_library_mim.toml") : abspath(only(ARGS))
mkpath(dirname(audit_output))


grid=CellGrid(4e-3,3e-3,16,12)
stack=PlanarStackup([PlanarLayer(1.,1.,.1e-3),PlanarLayer(3.9,1.,1e-6),
    PlanarLayer(1.,1.,.1e-3)],TERM_GND,TERM_GND,grid.a,grid.b)
shape=planar_transform(planar_mim_capacitor(width=1e-3,length=1e-3,
    upper_level=2,lower_level=1,upper_metal="film",overhang=.25e-3,
    lead_width=.5e-3,lead_length=1e-3,net_top="top",net_bottom="bottom");
    offset=(2e-3,1.5e-3))
layout=build_planar_layout(stack,grid,[shape],shape.pins;metals=Dict("film"=>.05))
reference=planar_parallel_plate_capacitance(stack,2,1,shape.meta["overlap_area"])
net=planar_connectivity(layout)
isempty(net.shorted_nets) && isempty(net.open_nets) || error("MIM geometry connectivity failed")
rows=Dict{String,Any}[]
for f in (1e6,10e6,100e6,1e9)
    result=solve_planar(layout,f;mx=48,my=36)
    model=planar_pi_model(result.y)
    c=imag(model.ys)/(2pi*f)
    row=Dict("frequency_hz"=>f,"capacitance_f"=>c,"parallel_plate_f"=>reference,
        "relative_difference"=>abs(c-reference)/reference,
        "max_s_singular_value"=>maximum(svdvals(result.s)),
        "voltage_residual"=>maximum(result.raw.relative_residuals),
        "scope"=>"raw physical-return leads included; parallel-plate comparison includes geometry overlap but no fringing")
    push!(rows,row)
    println(row)
    open(audit_output,"w") do io
        TOML.print(io,Dict("measurements"=>rows,"basis_count"=>planar_basis_count(layout.source_problem.basis)))
    end
end
minimum(row["relative_difference"] for row in rows)<=.1 || error("MIM library gate remains unmet")
