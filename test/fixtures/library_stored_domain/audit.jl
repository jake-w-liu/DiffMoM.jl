using DiffMoM,SHA,TOML
rows=Dict{String,Any}[]
function record(tag,f)
    row=Dict{String,Any}("case"=>tag)
    try
        value=f()
        row["status"]="accepted"
        row["value"]=value isa Real ? string(value) : string(value.pins[1].point)
        row["finite"]=value isa Real ? isfinite(value) :
            all(p->all(isfinite,p.point),value.pins) && all(p->all(v->all(isfinite,v),p.vertices),value.polygons)
    catch error
        row["status"]="rejected";row["diagnostic"]=sprint(showerror,error)
    end
    push!(rows,row);println(row)
end
stack=PlanarStackup([PlanarLayer(4.,1.,1e-6)],TERM_GND,TERM_GND,.001,.001)
for method in (:wheeler,:current_sheet),turns in (big"1e-500",big"1e500",BigFloat(2))
    record("$(method)_turns_$(turns)",()->planar_spiral_inductance(shape=:rectangular,turns=turns,d_out=.001,d_in=.0005,method=method))
end
for area in (big"1e-500",big"1e500",big"1e-8")
    record("capacitance_area_$(area)",()->planar_parallel_plate_capacitance(stack,1,0,area))
end
for (turns,outside,inside) in ((1e-200,1e300,5e299),(1e200,1e-300,5e-301),(1e-150,1e308,9e307))
    record("wheeler_intermediate_$(turns)_$(outside)",()->planar_spiral_inductance(shape=:rectangular,turns=turns,d_out=outside,d_in=inside))
end
shape=planar_line(length=.001,width=.0001,level=1,metal="pec")
record("representable_BigFloat_angle",()->planar_transform(shape;angle=big"0.1"))
record("overflow_BigFloat_offset",()->planar_transform(shape;offset=(big"1e500",0)))
record("underflow_BigFloat_offset",()->planar_transform(shape;offset=(big"1e-500",0)))
record("collapsed_finite_translation",()->planar_transform(shape;offset=(1e100,1e100)))
output=get(ENV,"LIBRARY_DOMAIN_OUTPUT",joinpath(@__DIR__,"library_stored_domain_before.toml"))
open(output,"w") do io
    TOML.print(io,Dict("scope"=>"public RFIC library stored scalar, finite reference and rigid placement input/output domains; ordinary representable controls included", "source_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"../../src/planar/PlanarLibrary.jl")))),"rows"=>rows))
end
