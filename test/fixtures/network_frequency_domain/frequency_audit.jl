using DiffMoM,Test,SHA,TOML
rows=Dict{String,Any}[]
for frequency in (big"1e-500",BigFloat(0),BigFloat(1e9),big"1e500")
    calls=Float64[]
    outcome=try
        data=PlanarNetworkData([frequency],[reshape(ComplexF64[.25],1,1)];z0=f->(push!(calls,f);50.))
        Dict{String,Any}("status"=>"accepted","stored_frequency_hz"=>only(data.frequencies))
    catch e
        e isa ArgumentError || rethrow()
        Dict{String,Any}("status"=>"rejected","diagnostic"=>sprint(showerror,e))
    end
    outcome["input_frequency_hz"]=string(frequency)
    outcome["provider_calls_hz"]=calls;push!(rows,outcome);println(outcome)
end
output=get(ENV,"NETWORK_FREQUENCY_OUTPUT",joinpath(@__DIR__,"network_frequency_domain_before.toml"))
open(output,"w") do io
    TOML.print(io,Dict("scope"=>"public databank wide-frequency storage and provider argument identity; explicit DC and representable controls",
        "source_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"../../src/planar/PlanarNetworkIO.jl")))),"rows"=>rows))
end
