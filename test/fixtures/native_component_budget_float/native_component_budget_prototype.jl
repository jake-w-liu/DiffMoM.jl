using DiffMoM,Test,LinearAlgebra,TOML,SHA

# Define only a distinct validation method in this fresh Julia process.
# Registered production files and the original methods are never modified.
helpers=raw"""
function _validation_component_base_payload(base)
    seen=Base.IdSet{Any}()
    values=(base.problem.basis,base.problem.sheets,base.problem.vias,base.problem.vols,
        base.problem.stack.layers,base.sheet_zs,base.via_sigma,base.contraction,
        base.floating_common,base.z0,base.labels)
    _checked_payload_sum("native live base geometry",4096,
        (_subdivision_retained_payload(value,seen) for value in values)...)
end
function _validation_component_post_payload(base,pins,labels)
    ninput=BigInt(length(base.problem.ports))+sum(length,pins)
    nc=BigInt(size(base.floating_common,2));nn=BigInt(length(labels))
    contacts=sum(BigInt(length(pin.terminal.cells))*length(base.problem.stack.layers)
        for group in pins for pin in group;init=BigInt(0))
    nr=BigInt(length(base.problem.ports))+2contacts
    response=sum(BigInt(length(group))^2 for group in pins;init=BigInt(0))
    _checked_payload_sum("native component post geometry workspace",
        4096+512BigInt(length(pins))+256(nn+ninput),
        _checked_array_payload_bytes(Float64,ninput,nn+nc),
        _checked_array_payload_bytes(Float64,nr,nn+nc),
        _checked_array_payload_bytes(ComplexF64,2response+12nn),
        _checked_array_payload_bytes(Tuple{Int,Int},ninput))
end
function _validation_component_retained_payload(model)
    seen=Base.IdSet{Any}()
    values=(model.problem.basis,model.problem.sheets,model.problem.vias,model.problem.vols,
        model.problem.stack.layers,model.sheet_zs,model.via_sigma,model.contraction,
        model.floating_common,model.z0,model.labels,model.external_labels)
    _checked_payload_sum("native retained component model",4096,
        model.model_files===nothing ? 0 : model.model_files.payload,
        _spice_circuit_payload(model.circuit),
        (_subdivision_retained_payload(value,seen) for value in values)...)
end
"""
include_string(DiffMoM,helpers,"validation_component_budget_helpers")
source=read(joinpath(@__DIR__,"../../src/planar/PlanarSonnetComponents.jl"),String)
start=first(findfirst("function sonnet_component_model(",source))
stop=first(findfirst("function _solve_sonnet_components(",source))
method=source[start:prevind(source,stop)]
method=replace(method,"function sonnet_component_model("=>"function _validation_component_budget_model(";count=1)
needle="    if _model_files===nothing && component_response===nothing &&"
@assert occursin(needle,method)
method=replace(method,needle=>raw"""
    variable_payload=isempty(variables) ? 0 : _checked_payload_sum("native owned variables",
        512,_checked_array_payload_bytes(UInt8,128,length(variables)),2sum(ncodeunits,keys(variables)))
    staged_payload=_model_files===nothing ? 0 : _model_files.payload
    _enforce_payload_limit(_checked_payload_sum("all native component geometry preflight",
        staged_payload,variable_payload,_sonnet_files_geometry_payload(p,grid)),max_bytes,
        "all native component geometry preflight","max_bytes")
    if _model_files===nothing && component_response===nothing &&
""";count=1)
needle="        ground_direction=ground_direction,max_bytes=max_bytes-file_payload)"
@assert occursin(needle,method)
method=replace(method,needle=>"        ground_direction=ground_direction,max_bytes=max_bytes-live_prefix-post_workspace)";count=1)
needle="    physical=planar_terminal_returns(base.problem,terminals;"
@assert occursin(needle,method)
method=replace(method,needle=>raw"""
    live_prefix=_checked_payload_sum("native component live prefix",file_payload,
        variable_payload,_validation_component_base_payload(base))
    post_workspace=_validation_component_post_payload(base,pins,labels)
    _enforce_payload_limit(_checked_payload_sum("native component provider preflight",
        live_prefix,post_workspace),max_bytes,"native component provider preflight","max_bytes")
    physical=planar_terminal_returns(base.problem,terminals;
""";count=1)
needle="    return (;problem=physical.problem,contraction,floating_common,z0=refs,labels,external_labels=base.labels,"
@assert occursin(needle,method)
method=replace(method,needle=>"    model=(;problem=physical.problem,contraction,floating_common,z0=refs,labels,external_labels=base.labels,";count=1)
needle="        pins,reference_paths=physical.paths,model_files=_model_files)\nend"
@assert occursin(needle,method)
method=replace(method,needle=>raw"""
        pins,reference_paths=physical.paths,model_files=_model_files)
    payload=_validation_component_retained_payload(model)
    _enforce_payload_limit(payload,max_bytes,"native retained component model","max_bytes")
    return merge(model,(;payload,live_prefix,post_workspace))
end
""";count=1)
include_string(DiffMoM,method,"validation_component_budget_method")

fixture=joinpath(@__DIR__,"../../test/fixtures/native_ideal_component_units/res/ohm/ohm.son")
p=read_sonnet_project(fixture)
prototype=DiffMoM._validation_component_budget_model
function record_kind(p,tokens)
    q=deepcopy(p);record=only(filter(r->first(r.tokens)=="TYPE",only(q.components)))
    empty!(record.tokens);append!(record.tokens,tokens);q
end
report=Dict{String,Any}("production_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"../../src/planar/PlanarSonnetComponents.jl")))))
@testset "Validation all-branch cumulative native geometry budget" begin
    for grid in ((32,40),(128,160)),limit in (1,20000)
        @test_throws ArgumentError prototype(p,200e6;grid,max_bytes=limit)
        allocation=@allocated try prototype(p,200e6;grid,max_bytes=limit) catch e;e isa ArgumentError || rethrow() end
        @test allocation<20000
        report[string(grid)*" "*string(limit)]=Dict("rejection_allocation"=>allocation)
    end
    for kind in (:ideal,:none,:callback,:sparam)
        project=kind===:sparam ? read_sonnet_project(joinpath(@__DIR__,"../../test/fixtures/native_sparam_model_files/native/sparameter.son")) :
            kind===:ideal ? p : record_kind(p,kind===:none ? ["TYPE","NONE"] : ["TYPE","SPROJ","child.son"])
        calls=Ref(0);callback=(args...)->(calls[]+=1;(response=ComplexF64[0 1;1 0],format=:s,z0=50.))
        kw=kind===:callback ? (;component_response=callback) : (;)
        @test_throws ArgumentError prototype(project,200e6;max_bytes=20000,kw...)
        @test calls[]==0
        actual=prototype(project,200e6;kw...)
        @test calls[]==(kind===:callback ? 1 : 0)
        original=sonnet_component_model(project,200e6;kw...)
        @test actual.problem.sheets[1].mask==original.problem.sheets[1].mask
        @test actual.problem.basis.kind==original.problem.basis.kind
        @test actual.problem.basis.ei==original.problem.basis.ei
        @test actual.problem.basis.ej==original.problem.basis.ej
        @test actual.contraction==original.contraction
        @test actual.floating_common==original.floating_common
        @test actual.labels==original.labels
        @test actual.sheet_zs==original.sheet_zs && actual.via_sigma==original.via_sigma
        @test actual.payload>20000
        @test_throws ArgumentError prototype(project,200e6;max_bytes=actual.payload-1,kw...)
        report[string(kind)]=Dict("actual_retained_payload"=>actual.payload,
            "live_prefix"=>actual.live_prefix,"post_workspace"=>actual.post_workspace)
        # Same physical EM response with independently built network load.
        raw=solve_planar(actual.problem,200e6;mx=64,my=80,surface_zs=actual.sheet_zs,via_sigma=actual.via_sigma)
        y=transpose(actual.contraction)*raw.y*actual.contraction
        a=deepcopy(actual.circuit);b=deepcopy(original.circuit)
        circuit_add_network!(a,collect(eachindex(actual.labels)),y;format=:y)
        circuit_add_network!(b,collect(eachindex(original.labels)),y;format=:y)
        network_a=solve_planar_circuit(a,200e6);network_b=solve_planar_circuit(b,200e6)
        @test maximum(abs,network_a.s-network_b.s)<1e-10
        @test network_a.voltages≈network_b.voltages rtol=1e-10
        waves=ComplexF64[.3+.2im,-.1im]
        maps_a=planar_current_maps(raw;voltages=actual.contraction*(network_a.voltages*waves))
        maps_b=planar_current_maps(raw;voltages=original.contraction*(network_b.voltages*waves))
        @test maps_a[1].jx≈maps_b[1].jx rtol=1e-10
    end
    project=read_sonnet_project(joinpath(@__DIR__,"../../test/fixtures/native_sparam_model_files/native/sparameter.son"))
    calls=Ref(0)
    files=sonnet_component_files(project,200e6;requested_reference=f->(calls[]+=1;50+10im))
    savedcalls=calls[]
    @test savedcalls>0
    @test_throws ArgumentError prototype(project,200e6;_model_files=files,max_bytes=files.payload)
    @test calls[]==savedcalls
    actual=prototype(project,200e6;_model_files=files)
    @test actual.model_files===files
    @test calls[]==savedcalls
    @test only(actual.circuit.elements).z0==ComplexF64[50+10im,50+10im]
end
open(joinpath(@__DIR__,"native_component_budget_prototype.toml"),"w") do io
    TOML.print(io,report)
end
