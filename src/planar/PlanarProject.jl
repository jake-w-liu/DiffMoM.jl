# Schema and SI-expression conventions adapted from ASCENT.jl.
# Copyright (c) 2026 jake-w-liu; MIT license: ASCENT_LIBRARY_LICENSE.
export PlanarProject,load_planar_project,write_planar_project,planar_project_from_dict
export planar_project_dict,planar_project_value,planar_material_preset

"""A declarative planar TOML project, retaining expressions and unit strings.
Its owned `data` uses named polygons, vias, ports, technology layers,
materials and component contracts. Layers/interfaces default to bottom
to top; `stackup.order="ascent"` explicitly selects ASCENT's legacy
top-to-bottom layers and level zero below the top dielectric.
Bare geometry numbers use `project.unit` (default metres); `=expr`
results and unit-suffixed variable values are SI. Expressions use a
whitelist interpreter and never Julia `eval`. `freq` [Hz] is reserved
for material, component and reference-plane fields."""
struct PlanarProject
    data::Dict{String,Any}
    source::String
    PlanarProject(data::Dict{String,Any},source::String,::Val{:owned})=new(data,source)
end

function _project_table(value,label)
    value isa AbstractDict || throw(ArgumentError("$label must be a table"))
    return value
end
function _project_keys(table,allowed,label)
    _project_table(table,label)
    unknown=setdiff(String.(collect(keys(table))),allowed)
    isempty(unknown) || throw(ArgumentError("unsupported $label fields: $(join(unknown,", "))"))
    return table
end
_project_owned(x::AbstractDict)=Dict{String,Any}(String(k)=>_project_owned(v) for (k,v) in x)
_project_owned(x::AbstractVector)=Any[_project_owned(v) for v in x]
_project_owned(x::Union{AbstractString,Real,Bool})=x
_project_owned(x)=throw(ArgumentError("project values must be TOML tables, arrays, strings or real numbers"))

const _PROJECT_UNITS=Dict(
    "m"=>(:length,1.),"mm"=>(:length,1e-3),"um"=>(:length,1e-6),
    "µm"=>(:length,1e-6),"μm"=>(:length,1e-6),"nm"=>(:length,1e-9),
    "mil"=>(:length,25.4e-6),"in"=>(:length,.0254),"cm"=>(:length,.01),
    "Hz"=>(:frequency,1.),"kHz"=>(:frequency,1e3),"MHz"=>(:frequency,1e6),
    "GHz"=>(:frequency,1e9),"THz"=>(:frequency,1e12),
    "ohm"=>(:resistance,1.),"Ω"=>(:resistance,1.),"kohm"=>(:resistance,1e3),
    "H"=>(:inductance,1.),"mH"=>(:inductance,1e-3),"uH"=>(:inductance,1e-6),
    "nH"=>(:inductance,1e-9),"pH"=>(:inductance,1e-12),
    "F"=>(:capacitance,1.),"uF"=>(:capacitance,1e-6),"nF"=>(:capacitance,1e-9),
    "pF"=>(:capacitance,1e-12),"fF"=>(:capacitance,1e-15),
    "S/m"=>(:conductivity,1.),"S/cm"=>(:conductivity,100.),
    "1/m"=>(:inverse_length,1.),"1/mm"=>(:inverse_length,1e3),
    "1/m^2"=>(:density,1.),"1/um^2"=>(:density,1e12),
    "rad"=>(:angle,1.),"deg"=>(:angle,pi/180),"1"=>(:dimensionless,1.))
const _PROJECT_CALLS=Dict{Symbol,Any}(:+ => (+),:- => (-),:* => (*),:/ => (/),:^ => (^),
    :sqrt=>sqrt,:abs=>abs,:min=>min,:max=>max,:exp=>exp,:log=>log,
    :log10=>log10,:sin=>sin,:cos=>cos,:tan=>tan,:atan=>atan,
    :sinh=>sinh,:cosh=>cosh,:tanh=>tanh,:real=>real,:imag=>imag,:complex=>complex)

function _project_ast(text)
    ncodeunits(text)<=8192 || throw(ArgumentError("project expression is too long"))
    ast=try Meta.parse(text) catch;throw(ArgumentError("invalid project expression"));end
    function check(node,depth)
        depth<=64 || throw(ArgumentError("project expression nesting exceeds 64"))
        node isa Real && return
        node isa Symbol && return
        node isa Expr && node.head===:call && !isempty(node.args) &&
            node.args[1] isa Symbol && haskey(_PROJECT_CALLS,node.args[1]) ||
            throw(ArgumentError("project expressions permit only numeric literals, variables and whitelisted functions"))
        foreach(a->check(a,depth+1),node.args[2:end])
    end
    check(ast,1);return ast
end

function _project_eval(node,resolve,freq)
    if node isa Real
        value=Float64(node)
        isfinite(value) || throw(ArgumentError("expression literal must fit finite Float64"))
        return value
    end
    if node isa Symbol
        node in (:pi,:π) && return pi
        node===:e && return exp(1.)
        node in (:im,:i) && return 1im
        if node===:freq
            freq===nothing && throw(ArgumentError("freq is allowed only in material, component and reference-plane fields"))
            return freq
        end
        return resolve(String(node))
    end
    return _PROJECT_CALLS[node.args[1]]((_project_eval(a,resolve,freq) for a in node.args[2:end])...)
end

function _project_quantity(value,unit,dimension,resolve,freq)
    result=if value isa Real
        dimension===:length ? value*_PROJECT_UNITS[unit][2] : value
    elseif value isa AbstractString
        text=strip(value)
        if startswith(text,"=")
            _project_eval(_project_ast(text[2:end]),resolve,freq)
        else
            m=match(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(\S*)$",text)
            m===nothing && throw(ArgumentError("invalid project quantity: $value"))
            number=parse(Float64,m.captures[1]);suffix=m.captures[2]
            if isempty(suffix)
                dimension===:length ? number*_PROJECT_UNITS[unit][2] : number
            else
                haskey(_PROJECT_UNITS,suffix) || throw(ArgumentError("unknown project unit $suffix"))
                d,s=_PROJECT_UNITS[suffix]
                dimension in (:any,d) || throw(ArgumentError("unit $suffix does not match $dimension"))
                number*s
            end
        end
    else
        throw(ArgumentError("project scalar must be a number, unit string or =expression"))
    end
    converted=ComplexF64(result)
    isfinite(converted) || throw(ArgumentError("project quantity must fit a finite ComplexF64 value"))
    return iszero(imag(converted)) ? real(converted) : converted
end

function _project_variables(project,overrides)
    raw=merge(get(project.data,"variables",Dict{String,Any}()),_project_owned(overrides))
    "freq" in keys(raw) && throw(ArgumentError("freq is a reserved project variable"))
    unit=get(get(project.data,"project",Dict()),"unit","m")
    cache=Dict{String,Number}();active=Set{String}()
    function resolve(name)
        haskey(cache,name) && return cache[name]
        haskey(raw,name) || throw(ArgumentError("undefined project variable $name"))
        name in active && throw(ArgumentError("cyclic project variable $name"))
        length(active)<64 || throw(ArgumentError("project variable dependency depth exceeds 64"))
        push!(active,name)
        value=try _project_quantity(raw[name],unit,:any,resolve,nothing) finally delete!(active,name) end
        cache[name]=value;return value
    end
    foreach(resolve,keys(raw))
    return resolve
end

"""Evaluate a project scalar in SI. Numeric lengths use `project.unit`;
unit strings must match `dimension`, while expression results are SI.
Supply `freq` only for fields allowed to depend on frequency. Variable
overrides preserve the same units and safe expression rules."""
function planar_project_value(project::PlanarProject,value;dimension::Symbol=:dimensionless,
        freq=nothing,variables::AbstractDict=Dict{String,Any}())
    unit=get(get(project.data,"project",Dict()),"unit","m")
    return _project_quantity(value,unit,dimension,_project_variables(project,variables),freq)
end

function _project_schema(data)
    _project_keys(data,["project","variables","box","mesh","stackup","metals",
        "materials","via_types","tech_layers","polygons","vias","ports","components","sweep"],"project root")
    header=_project_keys(get(data,"project",Dict()),["name","unit","schema_version"],"project")
    get(header,"schema_version",1)==1 || throw(ArgumentError("unsupported planar project schema version"))
    unit=get(header,"unit","m")
    haskey(_PROJECT_UNITS,unit) && _PROJECT_UNITS[unit][1]===:length || throw(ArgumentError("invalid project length unit"))
    _project_keys(get(data,"box",Dict()),["width","length","auto_margin","top_cover","bottom_cover","sidewalls"],"box")
    _project_keys(get(data,"mesh",Dict()),["nx","ny"],"mesh")
    stack=_project_keys(get(data,"stackup",Dict()),["order","layers"],"stackup")
    get(stack,"order","bottom_to_top") in ("bottom_to_top","ascent") || throw(ArgumentError("invalid stackup order"))
    haskey(stack,"layers") && stack["layers"] isa AbstractVector && !isempty(stack["layers"]) ||
        throw(ArgumentError("project needs stackup.layers"))
    for layer in stack["layers"]
        _project_keys(layer,["name","material","thickness","eps_r","eps_r_z","mu_r","mu_r_z",
            "tan_delta","tan_delta_z","conductivity","conductivity_z"],"stackup layer")
    end
    for kind in ("polygons","vias","ports","components")
        entries=get(data,kind,Any[])
        entries isa AbstractVector || throw(ArgumentError("$kind must be an array of tables"))
        all(x->x isa AbstractDict,entries) || throw(ArgumentError("$kind entries must be tables"))
    end
    !isempty(get(data,"polygons",Any[])) && !isempty(get(data,"ports",Any[])) ||
        throw(ArgumentError("project needs polygons and ports"))
    required(t,keys,label)=all(k->haskey(t,k),keys) || throw(ArgumentError("$label requires $(join(keys,", "))"))
    for (name,t) in get(data,"metals",Dict())
        _project_keys(t,["type","conductivity","thickness","mu_r","rs","xs","ls","roughness","plating",
            "direction","axial_subdivisions"],"metal $name")
        get(t,"type","lossless") in ("lossless","pec","surface_impedance","resistive","sheet","thick","bulk","volume") ||
            throw(ArgumentError("unsupported metal type"))
        for p in get(t,"plating",Any[])
            _project_keys(p,["conductivity","thickness","mu_r"],"plating");required(p,["conductivity","thickness"],"plating")
        end
        haskey(t,"roughness") && _project_keys(t["roughness"],["model","rms","rf","radius","density","loss_only"],"roughness")
    end
    for (name,t) in get(data,"materials",Dict())
        _project_keys(t,["eps_r","eps_r_z","tan_delta","tan_delta_z","mu_r","mu_r_z","conductivity","conductivity_z"],"material $name")
    end
    for (name,t) in get(data,"via_types",Dict())
        _project_keys(t,["kind","conductivity","metal","wall"],"via type $name")
    end
    for (name,t) in get(data,"tech_layers",Dict())
        _project_keys(t,["kind","level","metal","from_level","to_level","via_type","stream_layer","datatype"],"technology layer $name")
    end
    for t in data["polygons"]
        _project_keys(t,["name","level","metal","net","vertices","tech_layer"],"polygon")
        required(t,["name","vertices"],"polygon")
        haskey(t,"level") || haskey(t,"tech_layer") || throw(ArgumentError("polygon needs level or tech_layer"))
    end
    for t in get(data,"vias",Any[])
        _project_keys(t,["name","from_level","to_level","via_type","vertices","center","diameter","segments","tech_layer"],"via")
        required(t,["name"],"via")
        haskey(t,"vertices") || required(t,["center","diameter"],"circular via")
        haskey(t,"tech_layer") || required(t,["from_level","to_level"],"via")
    end
    for t in data["ports"]
        _project_keys(t,["name","number","type","polygon","edge","via","layer","cells",
            "ref_polygon","ref_edge","source_edge","bridge_metal",
            "nodes",
            "resistance","reactance","inductance","capacitance","impedance","external","refplane","polarity"],"port")
        get(t,"type","box_wall")=="via" ? required(t,["via","layer","cells"],"via port") :
            required(t,["polygon","edge"],"port")
        get(t,"type","box_wall")=="floating" && required(t,["ref_polygon","ref_edge"],"floating port")
        kind=get(t,"type","box_wall")
        common=["name","number","type","resistance","reactance","inductance","capacitance","impedance","external","refplane","polarity","nodes"]
        extra=kind=="via" ? ["via","layer","cells"] : kind=="floating" ?
            ["polygon","edge","ref_polygon","ref_edge","source_edge","bridge_metal"] :
            ["polygon","edge"]
        _project_keys(t,vcat(common,extra),"$kind port")
        if haskey(t,"impedance")
            any(k->haskey(t,k),("resistance","reactance","inductance","capacitance")) &&
                throw(ArgumentError("port impedance cannot also specify separate R/X/L/C values"))
            t["impedance"] isa AbstractDict && _project_keys(t["impedance"],["r","x","l","c","topology"],"port reference impedance")
        end
        if haskey(t,"refplane")
            _project_keys(t["refplane"],["length","zc","gamma"],"reference plane")
            required(t["refplane"],["length","zc","gamma"],"reference plane")
        end
        haskey(t,"external") && !(t["external"] isa Bool) && throw(ArgumentError("port external must be Boolean"))
    end
    for t in get(data,"components",Any[])
        required(t,["type"],"component")
        xor(haskey(t,"ports"),haskey(t,"nodes")) || throw(ArgumentError("component requires exactly one of ports or nodes"))
        common=["name","type","ports","nodes"];kind=t["type"]
        extra=kind in ("resistor","capacitor","inductor") ? ["value","topology"] :
            kind=="rlc" ? ["r","l","c","topology"] : kind=="transmission_line" ? ["zc","gamma","length"] :
            kind=="transformer" ? ["ratio"] : kind=="sparam_file" ? ["path","z0"] :
            kind=="inline_s" ? ["z0","frequencies","s_real","s_imag"] :
            kind=="subckt" ? ["path","subckt","parameters","dialect"] :
            kind in ("vendor","network") ? ["path"] : throw(ArgumentError("unknown component type $kind"))
        _project_keys(t,vcat(common,extra),"$kind component")
        kind in ("resistor","capacitor","inductor") && required(t,["value"],kind)
        kind=="rlc" && !any(k->haskey(t,k),("r","l","c")) && throw(ArgumentError("RLC component has no values"))
        kind=="transmission_line" && required(t,["zc","gamma","length"],kind)
        kind=="transformer" && required(t,["ratio"],kind)
        kind=="sparam_file" && required(t,["path"],kind)
        kind=="inline_s" && required(t,["frequencies","s_real"],kind)
        if kind=="subckt" && haskey(t,"subckt")
            get(t,"dialect","spice") in ("spice","spectre") || throw(ArgumentError("subckt dialect must be spice or spectre"))
            required(t,["path","nodes"],"linear SPICE subcircuit")
            t["nodes"] isa AbstractVector && all(n->!(n isa AbstractVector),t["nodes"]) ||
                throw(ArgumentError("SPICE nodes must be a flat array in literal subckt pin order"))
            get(t,"parameters",Dict()) isa AbstractDict || throw(ArgumentError("SPICE parameters must be a table"))
        elseif haskey(t,"parameters") || haskey(t,"dialect")
            throw(ArgumentError("linear library parameters/dialect require a named subckt"))
        end
    end
    if haskey(data,"sweep")
        _project_keys(data["sweep"],["frequencies","start","stop","points","spacing","adaptive","rel_tol","max_points","n_eval"],"sweep")
        haskey(data["sweep"],"adaptive") && !(data["sweep"]["adaptive"] isa Bool) &&
            throw(ArgumentError("sweep adaptive must be Boolean"))
    end
    for cover in ("top_cover","bottom_cover")
        value=get(get(data,"box",Dict()),cover,"pec")
        value isa AbstractDict && _project_keys(value,["type","metal","zs","eps_r","mu_r"],"box cover")
    end
    function expressions(value)
        value isa AbstractDict && (foreach(expressions,values(value));return)
        value isa AbstractVector && (foreach(expressions,value);return)
        value isa AbstractString && startswith(strip(value),"=") && _project_ast(strip(value)[2:end])
        return
    end
    expressions(data)
    return data
end

"""Construct an owned project from a TOML-compatible dictionary.
Unknown schema fields, unsafe expressions and variable cycles reject."""
function planar_project_from_dict(data::AbstractDict;source::AbstractString="")
    owned=_project_owned(data);_project_schema(owned)
    project=PlanarProject(owned,String(source),Val(:owned));_project_variables(project,Dict())
    return project
end
PlanarProject(data::AbstractDict;kw...)=planar_project_from_dict(data;kw...)
"""Return an independent TOML-compatible project dictionary, preserving
units, expressions, technology declarations and component contracts."""
planar_project_dict(project::PlanarProject)=deepcopy(project.data)

"""Load a planar TOML project. `ascent=true` explicitly maps legacy
ASCENT top-to-bottom layers and level indices. The file size is checked
before parsing; `max_bytes` bounds the input text size."""
function load_planar_project(path::AbstractString;ascent::Bool=false,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    isfile(path) || throw(ArgumentError("planar project not found: $path"))
    _enforce_payload_limit(filesize(path),max_bytes,"planar project input","max_bytes")
    data=TOML.parsefile(path)
    ascent && (data["stackup"]["order"]="ascent")
    return planar_project_from_dict(data;source=abspath(path))
end

"""Serialize a project without resolving or discarding expressions.
Schema and variable validation finish before the destination is opened."""
function write_planar_project(path::AbstractString,project::PlanarProject)
    validated=planar_project_from_dict(project.data;source=project.source)
    open(path,"w") do io
        TOML.print(io,validated.data;sorted=true)
    end
    return path
end

"""Nominal material starting points; inspect `scope` and `source` before
using them. Semiconductor entries supply static permittivity only; loss
and doping conductivity must be declared by the project. Custom material
tables override every preset field. Values are not a dispersion model."""
function planar_material_preset(name::AbstractString)
    key=lowercase(name)
    key=="air" && return (eps_r=1.,tan_delta=0.,scope="ideal air",source="")
    key in ("rogers_ro4003c","ro4003c") && return (eps_r=3.55,tan_delta=.0027,
        scope="nominal design Dk; Df at 10 GHz, not frequency dispersion",
        source="https://www.rogerscorp.com/advanced-electronics-solutions/ro4000-series-laminates/ro4003c-laminates")
    key in ("si","silicon") && return (eps_r=11.7,tan_delta=0.,scope="static permittivity; conductivity unspecified",
        source="https://www.ioffe.ru/SVA/NSM/Semicond/Si/basic.html")
    key in ("gaas","gallium_arsenide") && return (eps_r=12.9,tan_delta=0.,scope="static permittivity; conductivity unspecified",
        source="https://www.ioffe.ru/SVA/NSM/Semicond/GaAs/basic.html")
    throw(ArgumentError("unknown nominal material preset $name; supply a materials table"))
end
