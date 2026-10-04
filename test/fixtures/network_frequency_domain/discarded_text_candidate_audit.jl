using DiffMoM,Test,TOML,SHA,LinearAlgebra
module FrozenNetworkBefore
using LinearAlgebra
import DiffMoM
const _DEFAULT_MAX_DENSE_PAYLOAD_BYTES=DiffMoM._DEFAULT_MAX_DENSE_PAYLOAD_BYTES
const _checked_payload_sum=DiffMoM._checked_payload_sum
const _checked_array_payload_bytes=DiffMoM._checked_array_payload_bytes
const _enforce_payload_limit=DiffMoM._enforce_payload_limit
const _validated_resource_limit=DiffMoM._validated_resource_limit
const _planar_reference_values=DiffMoM._planar_reference_values
const _planar_reference_roots=DiffMoM._planar_reference_roots
const PlanarPortImpedance=DiffMoM.PlanarPortImpedance
include("network_frequency_source_before.jl")
end
function audit()
    rows=Dict{String,Any}[]
    mktempdir() do directory
        for (tag,record) in (("frequency","1e-500 .25 0"),("matrix_real","1 1e-500 0"),
                ("matrix_imag","1 0 -1D-500"),("explicit_zero","0e-500 .25 0"))
            source="# HZ S RI R 50\n"*record*"\n"
            path=joinpath(directory,"domain.s1p");write(path,source)
            before=try
                data=FrozenNetworkBefore.planar_read_touchstone(path)
                Dict{String,Any}("status"=>"accepted","frequency_hz"=>only(data.frequencies),
                    "s_real"=>real(only(only(data.s))),"s_imag"=>imag(only(only(data.s))))
            catch e
                e isa ArgumentError || rethrow()
                Dict{String,Any}("status"=>"rejected","diagnostic"=>sprint(showerror,e))
            end
            after=try
                data=planar_read_touchstone(path)
                Dict{String,Any}("status"=>"accepted","frequency_hz"=>only(data.frequencies))
            catch e
                e isa ArgumentError || rethrow()
                Dict{String,Any}("status"=>"rejected","diagnostic"=>sprint(showerror,e))
            end
            row=Dict{String,Any}("tag"=>tag,"source_text"=>source,"before"=>before,"after"=>after)
            @test (tag=="explicit_zero")==(after["status"]=="accepted")
            @test before["status"]==after["status"]
            push!(rows,row);println(row)
        end
    end
    before=TOML.parsefile(joinpath(@__DIR__,"network_frequency_domain_before.toml"))["source_sha256"]
    @test before==bytes2hex(sha256(read(joinpath(@__DIR__,"network_frequency_source_before.jl"))))
    open(joinpath(@__DIR__,"network_text_underflow_audit.toml"),"w") do io
        TOML.print(io,Dict("scope"=>"discarded text-underflow candidate: isolated byte-exact original NetworkIO parser already rejects the nonzero fields; surrounding reference helpers current and unchanged",
            "before_source_sha256"=>before,"after_source_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"../../src/planar/PlanarNetworkIO.jl")))),"rows"=>rows))
    end
end
audit()
