using DiffMoM,SHA,TOML
function main()
    tiny=nextfloat(0.)
    cases=(("rounded_symmetric_center_first",1.,1.,nextfloat(1.),eps(1.),1.,true),
        ("rounded_symmetric_center_second",nextfloat(1.),1.,nextfloat(1.),eps(1.),1.,true),
        ("strong_contraction",1e16,1.,1.,1.,1e-16,false),
        ("nonzero_half_rounding",3tiny,3tiny,3tiny,tiny,3tiny,true))
    rows=Any[]
    source=joinpath(@__DIR__,"../src/planar/PlanarSonnetGeometryVariables.jl")
    for (name,value,first,second,nominal,target,symmetric) in cases
        expected=setprecision(BigFloat,4608) do
            anchor=symmetric ? (BigFloat(first)+BigFloat(second))/2 : BigFloat(first)
            Float64(anchor+(BigFloat(value)-anchor)*BigFloat(target)/BigFloat(nominal))
        end
        actual=DiffMoM._sonnet_geovar_scaled_coordinate(value,first,second,nominal,target,symmetric)
        row=Dict{String,Any}("case"=>name,"value"=>value,"first"=>first,"second"=>second,"nominal"=>nominal,"target"=>target,"symmetric"=>symmetric,"observed_before"=>actual,"affine_oracle"=>expected)
        push!(rows,row);println(row)
    end
    output=joinpath(@__DIR__,"scaled_affine_failure_probe_20261004.toml")
    open(output,"w") do io;TOML.print(io,Dict("scope"=>"fresh independent 4608-bit affine oracle for represented Float64 inputs; not a native EM or continuum claim","source_sha256"=>bytes2hex(sha256(read(source))),"cases"=>rows));end
    all(row->row["observed_before"]!=row["affine_oracle"],rows) || error("candidate boundary did not fail")
end
main()
