# Offline Sonnet technology snapshots. EzXML handles XML syntax; semantic
# providers deliberately cover only primary-schema defaults and exact nodes.
export SonnetTechnologyNode, SonnetTechnology, SonnetTechnologyTable, SonnetLinkedProject
export read_sonnet_technology, sonnet_technology_value, sonnet_technology_material
export sonnet_technology_table, sonnet_technology_lookup, sonnet_technology_stack
export read_sonnet_linked_project, sonnet_materialize_project

"""Retained XML element, decoded attributes, direct text and ordered children.
Unknown elements and attributes remain accessible; their retention does not
authorize their use by the static material or geometry providers."""
struct SonnetTechnologyNode
    name::String
    attributes::Dict{String,String}
    text::String
    children::Vector{SonnetTechnologyNode}
end

"""Bounded, offline STF snapshot with exact source/hash and optional XSD proof.
Named dictionaries retain public variable, material, model, mesh and bias
identities. `stack` retains native top-to-bottom order. `schema_status` is
`:not_validated` unless an explicit local XSD was successfully validated;
`schema_uri` is metadata and is never fetched."""
struct SonnetTechnology
    source::String
    raw::String
    sha256::String
    schema_uri::String
    schema_sha256::Union{Nothing,String}
    schema_status::Symbol
    root::SonnetTechnologyNode
    units::Dict{String,String}
    variables::Dict{String,SonnetTechnologyNode}
    materials::Dict{Tuple{Symbol,String},SonnetTechnologyNode}
    models::Dict{String,SonnetTechnologyNode}
    meshes::Dict{String,SonnetTechnologyNode}
    biases::Dict{String,SonnetTechnologyNode}
    stack::SonnetTechnologyNode
    limits::NamedTuple
end

"""Exact STF table nodes in native units. Rows/columns and complete attributes
are retained. Calling the provider requires an exact stored Float64 key;
interpolation, extrapolation and geometry bias application are unsupported."""
struct SonnetTechnologyTable
    attributes::Dict{String,String}
    axis_names::Vector{String}
    nodes::Dict{Tuple,Vector{Float64}}
end

"""Native SON records plus one bounded, exact linked STF dependency snapshot.
The declared path, resolved path and both source hashes are retained. Intake
does not certify inherited etch, RPV, encrypted or meshing semantics. Use
`sonnet_materialize_project` for the explicitly supported static subset."""
struct SonnetLinkedProject
    source::String
    raw::String
    sha256::String
    records::Vector{SonnetRecord}
    declared_technology::String
    technology::SonnetTechnology
end

function _stf_limit(value,name)
    value isa Integer && !(value isa Bool) && 0<value<=typemax(Int) ||
        throw(ArgumentError("$name must be a positive machine integer"))
    return Int(value)
end

function _stf_read(path,max_bytes)
    source=abspath(path)
    isfile(source) || throw(ArgumentError("missing technology dependency $source"))
    source=realpath(source)
    bytes=open(source,"r") do io
        size=filesize(io)
        size<=max_bytes || throw(ArgumentError("technology source byte budget exceeded"))
        # Allocate for the actual descriptor size, never the configured
        # ceiling. No ceiling+1 arithmetic is needed even at typemax(Int).
        result=read(io,size)
        length(result)==size && eof(io) || throw(ArgumentError("technology source changed size during snapshot"))
        result
    end
    raw=String(bytes)
    isvalid(raw) || throw(ArgumentError("technology input must be UTF-8 XML"))
    occursin('\0',raw) && throw(ArgumentError("NUL/UTF-16 technology input is unsupported"))
    return source,raw
end

function _stf_xml(raw;kwargs...)
    try
        return _stf_xml_impl(raw;kwargs...)
    catch err
        err isa EzXML.XMLError && throw(ArgumentError("invalid technology XML: $(sprint(showerror,err))"))
        rethrow()
    end
end

function _stf_xml_impl(raw;max_elements,max_depth,max_storage_bytes)
    # Reject declarations before libxml2 sees any resolver opportunity. The
    # public EzXML API has no NONET option; neither parsexml nor StreamReader
    # enables DTDLOAD/NOENT. We additionally reject every DTD/custom entity,
    # XInclude and external processing instruction in this offline contract.
    occursin(r"(?i)<!\s*(DOCTYPE|ENTITY)",raw) &&
        throw(ArgumentError("STF DTD/entity declarations are forbidden"))
    ncodeunits(raw)<=typemax(Cint) || throw(ArgumentError("actual XML source exceeds libxml2 parser range"))
    reader=EzXML.StreamReader(IOBuffer(raw)); elements=0; depth=0; storage=0
    try
        for typ in reader
            # Stable xmlReaderTypes ABI: element1, entity-ref5, entity6,
            # processing-instruction7, document-type10 (libxml/xmlreader.h).
            convert(Int,typ) in (10,5,6,7) &&
                throw(ArgumentError("external/entity/processing XML records are forbidden"))
            depth=max(depth,EzXML.nodedepth(reader)+1)
            depth<=max_depth || throw(ArgumentError("technology XML depth budget exceeded"))
            if convert(Int,typ)==1
                elements+=1
                elements<=max_elements || throw(ArgumentError("technology XML element budget exceeded"))
                storage+=ncodeunits(EzXML.nodename(reader))+256
                for attr in EzXML.eachattribute(reader)
                    storage+=ncodeunits(EzXML.nodename(attr))+ncodeunits(EzXML.nodevalue(attr))+128
                end
            elseif EzXML.hasnodevalue(reader)
                storage+=ncodeunits(EzXML.nodevalue(reader))
            end
            storage<=max_storage_bytes || throw(ArgumentError("technology retained-storage budget exceeded"))
        end
    finally
        close(reader)
    end
    doc=EzXML.parsexml(raw)
    EzXML.hasdtd(doc) && throw(ArgumentError("STF DTD is forbidden"))
    EzXML.hasencoding(doc) && uppercase(EzXML.encoding(doc))!="UTF-8" &&
        throw(ArgumentError("technology XML encoding must be UTF-8"))
    return doc,(;elements,depth,estimated_storage_bytes=storage)
end

function _stf_namespace(node)
    try
        return EzXML.namespace(node)
    catch err
        err isa ArgumentError || rethrow()
        return nothing
    end
end

function _stf_node(node)
    _stf_namespace(node)!==nothing && throw(ArgumentError("namespaced STF elements are unsupported"))
    attributes=Dict{String,String}()
    for attr in EzXML.eachattribute(node)
        # Keep namespace-qualified metadata distinct from same-local-name
        # physical attributes. schemaLocation is never interpreted as a URL.
        ns=_stf_namespace(attr)
        key=ns!==nothing ? "{"*ns*"}"*EzXML.nodename(attr) : EzXML.nodename(attr)
        attributes[key]=EzXML.nodecontent(attr)
    end
    children=SonnetTechnologyNode[_stf_node(x) for x in EzXML.eachelement(node)]
    text=IOBuffer()
    for x in EzXML.eachnode(node)
        (EzXML.istext(x) || EzXML.iscdata(x)) && print(text,EzXML.nodecontent(x))
    end
    return SonnetTechnologyNode(EzXML.nodename(node),attributes,String(take!(text)),children)
end

_stf_children(node,name)=filter(x->x.name==name,node.children)
function _stf_child(node,name;required=false)
    children=_stf_children(node,name)
    length(children)<=1 || throw(ArgumentError("duplicate STF $name section"))
    isempty(children) && required && throw(ArgumentError("missing STF $name section"))
    return isempty(children) ? nothing : only(children)
end
function _stf_attr(node,name,default=nothing)
    value=get(node.attributes,name,default)
    value===nothing && throw(ArgumentError("STF $(node.name) requires $name"))
    return value
end
function _stf_named(section)
    result=Dict{String,SonnetTechnologyNode}()
    section===nothing && return result
    for node in section.children
        name=get(node.attributes,"name",nothing)
        if name===nothing
            node.name in ("var","mesh","metal_model","via_model","bias") &&
                throw(ArgumentError("STF $(node.name) requires name"))
            continue # Unknown unnamed classes remain in the retained tree.
        end
        haskey(result,name) && throw(ArgumentError("duplicate STF identity $name"))
        result[name]=node
    end
    return result
end

# Use libxml2's XSD API against a caller-provided, bounded in-memory schema.
# No schema include/import/redefine is admitted, so validation is offline.
_stf_xsd_error(::Ptr{Cvoid},::Ptr{Cvoid})=nothing
function _stf_validate_xsd(doc,schema)
    GC.@preserve schema doc begin
    ctx=ccall((:xmlSchemaNewMemParserCtxt,XML2_jll.libxml2),Ptr{Cvoid},(Cstring,Cint),schema,ncodeunits(schema))
    ctx==C_NULL && throw(ArgumentError("cannot allocate STF schema parser"))
    compiled=C_NULL; valid=C_NULL
    try
        # Context-local callbacks prevent XSD errors from contaminating
        # EzXML's global parse-error stack or changing its global handler.
        callback=@cfunction(_stf_xsd_error,Cvoid,(Ptr{Cvoid},Ptr{Cvoid}))
        ccall((:xmlSchemaSetParserStructuredErrors,XML2_jll.libxml2),Cvoid,
            (Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}),ctx,callback,C_NULL)
        compiled=ccall((:xmlSchemaParse,XML2_jll.libxml2),Ptr{Cvoid},(Ptr{Cvoid},),ctx)
        compiled==C_NULL && throw(ArgumentError("invalid STF XSD schema"))
        valid=ccall((:xmlSchemaNewValidCtxt,XML2_jll.libxml2),Ptr{Cvoid},(Ptr{Cvoid},),compiled)
        valid==C_NULL && throw(ArgumentError("cannot allocate STF schema validator"))
        ccall((:xmlSchemaSetValidStructuredErrors,XML2_jll.libxml2),Cvoid,
            (Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}),valid,callback,C_NULL)
        result=ccall((:xmlSchemaValidateDoc,XML2_jll.libxml2),Cint,(Ptr{Cvoid},Ptr{Cvoid}),valid,doc.node.ptr)
        result==0 || throw(ArgumentError("STF failed explicit local XSD validation (code $result)"))
    finally
        valid!=C_NULL && ccall((:xmlSchemaFreeValidCtxt,XML2_jll.libxml2),Cvoid,(Ptr{Cvoid},),valid)
        compiled!=C_NULL && ccall((:xmlSchemaFree,XML2_jll.libxml2),Cvoid,(Ptr{Cvoid},),compiled)
        ccall((:xmlSchemaFreeParserCtxt,XML2_jll.libxml2),Cvoid,(Ptr{Cvoid},),ctx)
    end
    end
    return nothing
end

"""Read a UTF-8 Sonnet STF snapshot offline using EzXML.
Source bytes are bounded before parsing. A streaming preflight enforces XML
element/depth/estimated-storage limits before building the retained tree.
DTD/entity declarations, processing instructions and namespaced physical
elements are rejected. Unknown unnamespaced data remain retained, and static
providers reject unknown physical fields. An optional `schema_path` validates
against that exact bounded local XSD; schema imports/includes are rejected and
no `schemaLocation` network request is made. Defaults follow matl-1.3.xsd.
This function does not implement interpolation, etch or RPV application."""
function read_sonnet_technology(path::AbstractString;schema_path=nothing,
        max_bytes=1_000_000,max_elements=10_000,max_depth=64,
        max_storage_bytes=8_000_000,max_table_nodes=4096)
    max_bytes=_stf_limit(max_bytes,"max_bytes")
    max_elements=_stf_limit(max_elements,"max_elements");max_depth=_stf_limit(max_depth,"max_depth")
    max_storage_bytes=_stf_limit(max_storage_bytes,"max_storage_bytes")
    max_table_nodes=_stf_limit(max_table_nodes,"max_table_nodes")
    source,raw=_stf_read(path,max_bytes)
    doc,stats=_stf_xml(raw;max_elements,max_depth,max_storage_bytes)
    schema_sha=nothing;schema_status=:not_validated
    if schema_path!==nothing
        _,schema=_stf_read(schema_path,max_bytes)
        schema_doc,_=_stf_xml(schema;max_elements,max_depth,max_storage_bytes)
        # Unlike STF physical elements, XSD elements necessarily have a namespace.
        for x in findall("//*",schema_doc)
            EzXML.nodename(x) in ("include","import","redefine","override") &&
                throw(ArgumentError("external XSD dependencies are forbidden"))
        end
        _stf_validate_xsd(doc,schema)
        schema_sha=bytes2hex(SHA.sha256(schema));schema_status=:validated
    end
    root=_stf_node(EzXML.root(doc))
    root.name=="technology_file" || throw(ArgumentError("not a Sonnet technology_file"))
    # Schema-defined authoring metadata is retained verbatim; it changes
    # editor permissions, not the physical stack/material definitions.
    get(root.attributes,"writeable","true") in ("true","false","1","0") ||
        throw(ArgumentError("invalid STF writeable boolean"))
    unitnode=_stf_child(root,"units";required=true)
    units=Dict(k=>_stf_attr(unitnode,k,v) for (k,v) in
        (("lunit","UM"),("cunit","SM"),("runit","OHUM"),("srunit","OHSQ"),("tempunit","C")))
    public=_stf_child(root,"public";required=true)
    materials=Dict{Tuple{Symbol,String},SonnetTechnologyNode}()
    for node in _stf_child(public,"materials";required=true).children
        name=get(node.attributes,"name",nothing)
        if name===nothing
            node.name in ("dielectric","conductor") && throw(ArgumentError("STF $(node.name) requires name"))
            continue
        end
        key=(Symbol(node.name),name)
        haskey(materials,key) && throw(ArgumentError("duplicate STF material identity $key"))
        materials[key]=node
    end
    technology=SonnetTechnology(source,raw,bytes2hex(SHA.sha256(raw)),
        get(root.attributes,"{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation",""),
        schema_sha,schema_status,root,units,_stf_named(_stf_child(public,"variables")),materials,
        _stf_named(_stf_child(public,"metal_model_defs")),_stf_named(_stf_child(public,"mesh_defs")),
        _stf_named(_stf_child(public,"bias_defs")),_stf_child(public,"stackup";required=true),
        (;stats...,max_bytes,max_elements,max_depth,max_storage_bytes,max_table_nodes))
    # Validate every recognized table now, rather than waiting for provider use.
    for bias in values(technology.biases),node in bias.children
        node.name in ("lookup_table","lookup_vector") && _stf_table(node,max_table_nodes;strict=false)
    end
    return technology
end

function _stf_literal(text)
    occursin(r"^[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?$",text) ||
        throw(ArgumentError("STF scalar is not a literal number: $text"))
    value=tryparse(Float64,text)
    value!==nothing && isfinite(value) || throw(ArgumentError("nonfinite/unrepresentable STF scalar $text"))
    iszero(value) && occursin(r"[1-9]",first(split(lowercase(text),'e'))) &&
        throw(ArgumentError("STF scalar underflows Float64: $text"))
    return value
end
function _stf_scalar(t,text,variables,active)
    occursin(r"^[A-Za-z][A-Za-z0-9_]*$",text) || return _stf_literal(text)
    length(active)<64 || throw(ArgumentError("STF variable dependency depth exceeded"))
    text in active && throw(ArgumentError("cyclic STF variable $text"))
    if haskey(variables,text)
        value=variables[text]
        value isa Real && isfinite(value) || throw(ArgumentError("invalid STF variable override $text"))
        converted=Float64(value)
        isfinite(converted) && (iszero(value) || !iszero(converted)) || throw(ArgumentError("unrepresentable STF override $text"))
        return converted
    end
    haskey(t.variables,text) || throw(ArgumentError("unknown/private STF variable $text"))
    node=t.variables[text]
    node.name=="var" || throw(ArgumentError("unsupported STF variable class $(node.name)"))
    _stf_known(node,["name","value","units"])
    push!(active,text)
    value=_stf_scalar(t,_stf_attr(node,"value"),variables,active)
    delete!(active,text)
    return value
end
function _stf_scale(t,quantity)
    quantity==:dimensionless && return 1.
    _stf_known(_stf_child(t.root,"units";required=true),["lunit","cunit","runit","srunit","tempunit"])
    maps=Dict(:length=>("lunit",Dict("UM"=>1e-6,"CM"=>1e-2,"MM"=>1e-3,"M"=>1.,"MIL"=>25.4e-6,"INCH"=>.0254)),
        :conductivity=>("cunit",Dict("SM"=>1.,"SCM"=>100.,"MSCM"=>.1,"USCM"=>1e-4)),
        # Native maintain-physical GUI exports prove OHMM=ohm-m (70000
        # OHUM -> .07 OHMM), and MOSQ=mohm/square (2 OHSQ -> 2000 MOSQ).
        :resistivity=>("runit",Dict("OHCM"=>1e-2,"OHMM"=>1.,"OHUM"=>1e-6)),
        :sheet_resistance=>("srunit",Dict("OHSQ"=>1.,"MOSQ"=>1e-3,"MOHSQ"=>1e-3)))
    haskey(maps,quantity) || throw(ArgumentError("unverified STF unit quantity $quantity"))
    key,unitmap=maps[quantity];unit=uppercase(t.units[key])
    haskey(unitmap,unit) || throw(ArgumentError("unsupported STF $key $unit"))
    return unitmap[unit]
end

"""Evaluate a literal or declared STF variable and convert the explicitly
selected `quantity` to SI (`:dimensionless`, `:length`, `:conductivity`,
`:resistivity`, `:sheet_resistance`). Overrides use native STF units. General
arithmetic expressions, encrypted values, unknown quantities and nonfinite
or unrepresentable numbers reject. Variable unit labels remain metadata."""
function sonnet_technology_value(t::SonnetTechnology,text::AbstractString;
        quantity::Symbol=:dimensionless,variables=Dict{String,Float64}())
    value=_stf_scalar(t,String(text),variables,Set{String}())
    result=value*_stf_scale(t,quantity)
    isfinite(result) && (iszero(value) || !iszero(result)) || throw(ArgumentError("STF SI value is unrepresentable"))
    return result
end

function _stf_known(node,names;children=String[])
    unknown=setdiff(collect(keys(node.attributes)),names)
    isempty(unknown) || throw(ArgumentError("unsupported STF $(node.name) attributes $(join(unknown,", "))"))
    all(x->x.name in children,node.children) || throw(ArgumentError("unsupported STF $(node.name) child semantics"))
    isempty(strip(node.text)) || throw(ArgumentError("unsupported STF $(node.name) text semantics"))
end

"""Evaluate one isotropic dielectric or scalar conductor definition with
matl-1.3 defaults. Dielectric er/mur, loss tangents and conductivity are SI;
`condspec=cond/rsvy` conductors return SI conductivity and `shres` returns
ohms per square. Other conductor classes remain retained but reject here.
This provider does not apply a technology layer's etch/rho/RPV bias table."""
function sonnet_technology_material(t::SonnetTechnology,name::AbstractString;
        kind::Symbol=:dielectric,variables=Dict{String,Float64}())
    node=get(t.materials,(kind,String(name)),nothing)
    node===nothing && throw(ArgumentError("missing STF $kind material $name"))
    val(text;q=:dimensionless)=sonnet_technology_value(t,text;quantity=q,variables)
    if kind==:dielectric
        _stf_known(node,["name","anisotropic","cond_res"];children=["params"])
        _stf_attr(node,"anisotropic","no")=="no" || throw(ArgumentError("anisotropic STF material mapping is unsupported"))
        length(node.children)==1 || throw(ArgumentError("isotropic STF dielectric requires one params record"))
        params=only(node.children)
        _stf_known(params,["axes","erel","mrel","tane","tanm","cond","rsvy"])
        _stf_attr(params,"axes","xyz")=="xyz" || throw(ArgumentError("STF dielectric axes mapping is unsupported"))
        er,mur,tane,tanm=[val(_stf_attr(params,k,d)) for (k,d) in
            (("erel","1"),("mrel","1"),("tane","0"),("tanm","0"))]
        er>0 && mur>0 && tane>=0 && tanm>=0 || throw(ArgumentError("invalid passive STF dielectric values"))
        spec=_stf_attr(node,"cond_res","cond")
        cond=val(_stf_attr(params,"cond","0");q=:conductivity)
        rho=val(_stf_attr(params,"rsvy","0");q=:resistivity)
        cond>=0 && rho>=0 || throw(ArgumentError("negative STF dielectric conduction"))
        sigma=spec=="cond" ? cond : spec=="rsvy" && rho>0 ? inv(rho) :
            throw(ArgumentError("invalid STF dielectric resistivity selection"))
        isfinite(sigma) || throw(ArgumentError("unrepresentable STF dielectric conductivity"))
        return (;kind,name=String(name),er,mur,tane,tanm,sigma)
    elseif kind==:conductor
        _stf_known(node,["name","condspec","cond","rsvy","shres","rpv","rdc","rrf","xdc","ls"])
        spec=_stf_attr(node,"condspec","cond")
        for key in ("rpv","rdc","rrf","xdc","ls")
            iszero(val(_stf_attr(node,key,"0"))) || throw(ArgumentError("inactive STF $key application is unsupported"))
        end
        if spec=="cond"
            text=_stf_attr(node,"cond","0")
            sigma=text=="INF" ? Inf : val(text;q=:conductivity)
            sigma>0 || throw(ArgumentError("STF conductor conductivity must be positive"))
            return (;kind,name=String(name),spec,sigma)
        elseif spec=="rsvy"
            rho=val(_stf_attr(node,"rsvy","0");q=:resistivity)
            rho>0 && isfinite(inv(rho)) || throw(ArgumentError("invalid STF conductor resistivity"))
            return (;kind,name=String(name),spec,sigma=inv(rho))
        elseif spec=="shres"
            resistance=val(_stf_attr(node,"shres","0");q=:sheet_resistance)
            resistance>=0 || throw(ArgumentError("negative STF sheet resistance"))
            return (;kind,name=String(name),spec,resistance)
        end
        throw(ArgumentError("STF conductor class $spec requires its physical adapter"))
    end
    throw(ArgumentError("unsupported STF material class $kind"))
end

function _stf_table(node,budget;strict=true)
    nodes=Dict{Tuple,Vector{Float64}}();attrs=copy(node.attributes)
    rowname=_stf_attr(node,"row_name");_stf_attr(node,"value_name")
    if strict
        allowed=node.name=="lookup_table" ? ["row_name","col_name","value_name","etch_type"] : ["row_name","value_name","size"]
        children=node.name=="lookup_table" ? ["column_keys","row"] : ["vector"]
        _stf_known(node,allowed;children)
        for child in node.children
            isempty(child.children) && all(k->k=="key",keys(child.attributes)) ||
                throw(ArgumentError("unknown STF table child semantics"))
        end
        haskey(attrs,"etch_type") && !(attrs["etch_type"] in ("DEFAULT","RES","CAP","TOP")) &&
            throw(ArgumentError("unknown STF etch table selector"))
    end
    function put(key,vals)
        length(nodes)<budget || throw(ArgumentError("STF table node budget exceeded"))
        haskey(nodes,key) && throw(ArgumentError("duplicate STF table node $key"))
        nodes[key]=vals
    end
    if node.name=="lookup_table"
        columns=_stf_child(node,"column_keys";required=true)
        column_keys=_stf_literal.(split(columns.text))
        !isempty(column_keys) && length(unique(column_keys))==length(column_keys) || throw(ArgumentError("missing/duplicate STF column keys"))
        rows=_stf_children(node,"row")
        length(column_keys)<=budget && length(rows)<=budget÷length(column_keys) || throw(ArgumentError("STF table node budget exceeded"))
        for row in rows
            values=_stf_literal.(split(row.text));length(values)==length(column_keys) || throw(ArgumentError("STF table row width mismatch"))
            key=_stf_literal(_stf_attr(row,"key"))
            for (col,value) in zip(column_keys,values);put((key,col),[value]);end
        end
        names=[rowname,_stf_attr(node,"col_name")]
    elseif node.name=="lookup_vector"
        size=parse(Int,_stf_attr(node,"size"))
        0<size<=budget || throw(ArgumentError("STF vector size budget exceeded"))
        rows=_stf_children(node,"vector")
        length(rows)<=budget÷size || throw(ArgumentError("STF vector storage budget exceeded"))
        for row in rows
            values=_stf_literal.(split(row.text));length(values)==size || throw(ArgumentError("STF vector width mismatch"))
            put((_stf_literal(_stf_attr(row,"key")),),values)
        end
        names=[rowname]
    else
        throw(ArgumentError("unsupported STF table class $(node.name)"))
    end
    isempty(nodes) && throw(ArgumentError("empty STF table"))
    return SonnetTechnologyTable(attrs,names,nodes)
end

"""Select one retained STF lookup table/vector by bias identity, `value_name`
and optional exact `etch_type`. Missing or ambiguous selections reject.
Only exact tabulated-node evaluation is implemented."""
function sonnet_technology_table(t::SonnetTechnology,bias::AbstractString;
        value_name::AbstractString,etch_type=nothing)
    node=get(t.biases,String(bias),nothing)
    node===nothing && throw(ArgumentError("unknown STF bias identity $bias"))
    selected=filter(x->x.name in ("lookup_table","lookup_vector") &&
        get(x.attributes,"value_name","")==value_name &&
        (etch_type===nothing || get(x.attributes,"etch_type","")==etch_type),node.children)
    length(selected)==1 || throw(ArgumentError("STF table selection is missing or ambiguous"))
    return _stf_table(only(selected),t.limits.max_table_nodes)
end

"""Evaluate one exact native-unit table node. Returns a copied value vector;
unknown/off-node/outside-table coordinates reject without tolerance, fitting,
interpolation or extrapolation. Results retain native units; geometry etch,
rho and RPV application require separately verified physical semantics."""
function sonnet_technology_lookup(table::SonnetTechnologyTable,coordinates::Real...)
    length(coordinates)==length(table.axis_names) || throw(ArgumentError("STF table coordinate dimension mismatch"))
    all(isfinite,coordinates) || throw(ArgumentError("nonfinite STF table coordinate"))
    key=Tuple(Float64.(coordinates))
    all(isfinite,key) || throw(ArgumentError("unrepresentable STF table coordinate"))
    haskey(table.nodes,key) || throw(ArgumentError("STF interpolation/extrapolation is unsupported; require an exact tabulated node"))
    return copy(table.nodes[key])
end

function _stf_static_guard(t)
    _stf_attr(t.root,"has_private","false") in ("false","0") || throw(ArgumentError("private STF physical semantics are unsupported"))
    any(x->x.name in ("private","privenc"),t.root.children) && throw(ArgumentError("private/encrypted STF physical semantics are unsupported"))
    all(x->x.name in ("source","units","comments","public"),t.root.children) || throw(ArgumentError("unknown STF root semantics"))
    all(k->k in ("version","has_private","writeable","{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation"),keys(t.root.attributes)) || throw(ArgumentError("unknown STF root attributes"))
    public=_stf_child(t.root,"public";required=true)
    isempty(public.attributes) || throw(ArgumentError("unknown STF public attributes"))
    isempty(t.stack.attributes) && isempty(strip(t.stack.text)) || throw(ArgumentError("unknown STF stack semantics"))
    all(x->x.name in ("variables","materials","mesh_defs","metal_model_defs","bias_defs","stackup"),public.children) || throw(ArgumentError("unknown STF public semantics"))
    return nothing
end

function _stf_static_rows(t,variables)
    _stf_static_guard(t)
    rows=Vector{Vector{String}}();seen=Set{String}()
    for node in t.stack.children
        node.name in ("TOP","BOTTOM","vias") && continue
        node.name=="diel" || throw(ArgumentError("unknown STF stack record $(node.name)"))
        _stf_known(node,["name","dielectric","thickness","zpart","is_conformal","measured_from","associated_cond",
            "sidewall_thickness","topwall_thickness","bottomwall_thickness"])
        name=_stf_attr(node,"name");name in seen && throw(ArgumentError("duplicate STF stack layer $name"));push!(seen,name)
        _stf_attr(node,"is_conformal","false") in ("false","0") || throw(ArgumentError("conformal STF dielectric geometry is unsupported"))
        _stf_attr(node,"measured_from","N/A")=="N/A" && _stf_attr(node,"associated_cond","N/A")=="N/A" || throw(ArgumentError("STF wall/measurement geometry is unsupported"))
        for key in ("sidewall_thickness","topwall_thickness","bottomwall_thickness")
            iszero(sonnet_technology_value(t,_stf_attr(node,key,"0");quantity=:length,variables)) || throw(ArgumentError("STF dielectric wall geometry is unsupported"))
        end
        parse(Int,_stf_attr(node,"zpart","0"))==0 || throw(ArgumentError("STF z-partition mapping is unsupported"))
        d=sonnet_technology_value(t,_stf_attr(node,"thickness");quantity=:length,variables)
        d>0 || throw(ArgumentError("STF dielectric thickness must be positive"))
        mat=sonnet_technology_material(t,_stf_attr(node,"dielectric");variables)
        push!(rows,[string.([d,mat.er,mat.mur,mat.tane,mat.tanm,mat.sigma]);name])
    end
    isempty(rows) && throw(ArgumentError("STF stack has no dielectrics"))
    vias=_stf_child(t.stack,"vias")
    vias!==nothing && !isempty(vias.children) && throw(ArgumentError("STF via technology inheritance requires its physical adapter"))
    return rows
end
function _stf_cover(t,name)
    node=_stf_child(t.stack,name;required=true)
    _stf_known(node,["material","model"])
    _stf_attr(node,"material")=="Lossless" || throw(ArgumentError("STF cover material requires its physical model adapter"))
    model=get(t.models,_stf_attr(node,"model"),nothing)
    model!==nothing && model.name=="metal_model" && _stf_attr(model,"model_type")=="Normal" || throw(ArgumentError("STF cover model requires its physical adapter"))
    _stf_known(model,["name","model_type","top_roughness","bottom_roughness","current_ratio","cross_section","num_sheets"])
    for key in ("top_roughness","bottom_roughness")
        iszero(_stf_literal(_stf_attr(model,key,"0"))) || throw(ArgumentError("STF cover roughness is unsupported"))
    end
    _stf_attr(model,"cross_section","Thin")=="Thin" || throw(ArgumentError("STF cover cross-section is unsupported"))
    return TERM_GND
end

"""Build the proven static isotropic STF dielectric stack at `freq` Hz.
`a,b` are SI box dimensions. XML order is reversed to the solver's bottom-to-
top convention; dielectric loss/conductivity are preserved. Only builtin
Lossless covers with Normal models are admitted. Applied technology layers,
wall geometry, private data, partitioning and unknown semantics reject.
This is a physical stack provider, not a certificate of native Lite linked-
STF EM equivalence (Lite rejects linked STF projects)."""
function sonnet_technology_stack(t::SonnetTechnology,freq::Real,a::Real,b::Real;
        variables=Dict{String,Float64}())
    isfinite(freq) && freq>0 || throw(ArgumentError("STF stack requires positive finite frequency"))
    rows=_stf_static_rows(t,variables)
    layers=PlanarLayer[]
    for row in reverse(rows)
        d,e,m,te,tm,sigma=parse.(Float64,row[1:6])
        push!(layers,PlanarLayer(e*(1-im*te)-im*_sonnet_dielectric_conduction(sigma,freq),m*(1-im*tm),d))
    end
    return PlanarStackup(layers,_stf_cover(t,"BOTTOM"),_stf_cover(t,"TOP"),a,b)
end

"""Read one SON geometry file and its explicitly linked STF snapshot.
Both sources are byte-bounded and retained exactly. Dependencies must resolve
inside `dependency_root` (default SON directory); absolute paths, traversal,
missing files and multiple STF records reject. No geometry is silently
materialized, and all component/technology records remain intact."""
function read_sonnet_linked_project(path::AbstractString;dependency_root=dirname(abspath(path)),
        max_project_bytes=1_000_000,kwargs...)
    max_project_bytes=_stf_limit(max_project_bytes,"max_project_bytes")
    source,raw=_stf_read(path,max_project_bytes)
    records=SonnetRecord[]
    for (line,text) in enumerate(split(raw,'\n'))
        tokens=_sonnet_tokens(text);isempty(tokens) || push!(records,SonnetRecord(line,tokens))
    end
    !isempty(records) && length(records[1].tokens)>=2 && records[1].tokens[1:2]==["FTYP","SONPROJ"] || throw(ArgumentError("linked STF intake requires a SONPROJ geometry file"))
    linked=filter(x->x.tokens[1]=="STF",_sonnet_section(records,"GEO",source))
    length(linked)==1 && length(only(linked).tokens)==2 || throw(ArgumentError("require exactly one complete STF dependency record"))
    declared=only(linked).tokens[2]
    isabspath(declared) && throw(ArgumentError("absolute STF dependency paths are forbidden"))
    any(x->x=="..",split(replace(declared,'\\'=>'/'),'/')) && throw(ArgumentError("STF dependency traversal is forbidden"))
    dep=abspath(joinpath(dirname(source),declared));root=realpath(dependency_root)
    isfile(dep) || throw(ArgumentError("missing linked STF dependency $declared"))
    resolved=realpath(dep);relative=relpath(resolved,root)
    (isabspath(relative) || any(x->x=="..",split(replace(relative,'\\'=>'/'),'/'))) && throw(ArgumentError("STF dependency is outside dependency_root"))
    technology=read_sonnet_technology(resolved;kwargs...)
    return SonnetLinkedProject(source,raw,bytes2hex(SHA.sha256(raw)),records,declared,technology)
end

"""Materialize only a static linked STF subset into an ordinary SonnetProject.
The linked stack and builtin Lossless covers replace serialized inline stack/
cover data according to the primary STF contract. This subset requires PEC
polygons without inherited technology layers; components and ports retain
their original records and identities. Applied etch/rho/RPV bias, inherited
metal/via models, encrypted data and unknown physical semantics reject before
solving. Original SON records remain in the returned project; retain the
SonnetLinkedProject alongside it for exact dependency provenance. STF overrides
use STF units; project variable evaluation still uses native DIM units."""
function sonnet_materialize_project(linked::SonnetLinkedProject;
        technology_variables=Dict{String,Float64}())
    t=linked.technology;rows=_stf_static_rows(t,technology_variables)
    _stf_cover(t,"TOP");_stf_cover(t,"BOTTOM")
    records=SonnetRecord[];geo=_sonnet_section(linked.records,"GEO",linked.source)
    boxindex=findfirst(r->r.tokens[1]=="BOX",geo)
    boxindex!==nothing || throw(ArgumentError("linked SON project lacks BOX"))
    oldcount=parse(Int,geo[boxindex].tokens[2])+1
    # Linked BOX0 has no inline layers. If serialized inline layers are
    # present, recognize only numeric seven-field dielectric rows.
    skip=Set{Int}()
    for row in geo[boxindex+1:min(length(geo),boxindex+oldcount)]
        length(row.tokens)>=7 && !(row.tokens[1] in ("TMET","BMET","MET","TECHLAY","VALVAR","GEOVAR","LORGN")) || break
        push!(skip,row.line)
    end
    dims=_sonnet_section(linked.records,"DIM",linked.source)
    lng=only(filter(x->x.tokens[1]=="LNG",dims)).tokens[2]
    ls=get(Dict("M"=>1.,"CM"=>1e-2,"MM"=>1e-3,"UM"=>1e-6,"NM"=>1e-9,"MIL"=>25.4e-6,"MILS"=>25.4e-6,"IN"=>.0254),uppercase(lng),NaN)
    isfinite(ls) || throw(ArgumentError("unsupported linked-project length units"))
    for record in linked.records
        tokens=copy(record.tokens)
        record.line in skip && continue
        tokens[1]=="STF" && continue
        if tokens[1] in ("TMET","BMET")
            tokens=[tokens[1],"Lossless","0","SUP","0","0","0","0"]
        elseif tokens[1]=="BOX"
            tokens[2]=string(length(rows)-1)
            push!(records,SonnetRecord(record.line,tokens))
            for (i,row) in enumerate(rows)
                converted=copy(row[1:6]);converted[1]=string(parse(Float64,row[1])/ls)
                append!(converted,["2",row[7]])
                push!(records,SonnetRecord(record.line,converted))
            end
            continue
        end
        push!(records,SonnetRecord(record.line,tokens))
    end
    project=_sonnet_read_records(linked.source,records)
    all(p->p.material==-1 && isempty(p.technology),project.polygons) || throw(ArgumentError("linked STF polygon material/technology inheritance requires its physical adapter"))
    isempty(project.metals) || throw(ArgumentError("linked STF metal-type replacement requires its physical adapter"))
    return SonnetProject(project.source,project.units,project.length_scale,project.frequency_scale,
        project.box,project.layers,project.metals,project.top,project.bottom,project.polygons,
        project.ports,project.variables,project.components,project.sweeps,linked.records)
end

# Verify the mutable decoded STF views against their retained XML before
# attributing a CSV-backed effective project to this captured dependency.
function _sonnet_scalar_same_technology_node(a,b)
    pending=[(a,b)]
    while !isempty(pending)
        left,right=pop!(pending)
        left.name==right.name && left.text==right.text && left.attributes==right.attributes &&
            length(left.children)==length(right.children) || return false
        for i in eachindex(left.children);push!(pending,(left.children[i],right.children[i]));end
    end
    return true
end
function _sonnet_scalar_technology_baseline(t,max_bytes)
    bytes2hex(SHA.sha256(t.raw))==t.sha256 || throw(ArgumentError("native scalar retained STF source was mutated"))
    _enforce_payload_limit(_checked_payload_sum("native scalar STF verification",
        BigInt(16)*ncodeunits(t.raw)+2BigInt(t.limits.estimated_storage_bytes)+4096),
        max_bytes,"native scalar STF verification","max_bytes")
    doc,_=_stf_xml(t.raw;max_elements=t.limits.max_elements,max_depth=t.limits.max_depth,
        max_storage_bytes=min(t.limits.max_storage_bytes,max_bytes))
    root=_stf_node(EzXML.root(doc))
    _sonnet_scalar_same_technology_node(t.root,root) ||
        throw(ArgumentError("native scalar decoded STF tree disagrees with its captured XML"))
    unitnode=_stf_child(root,"units";required=true)
    units=Dict(k=>_stf_attr(unitnode,k,v) for (k,v) in
        (("lunit","UM"),("cunit","SM"),("runit","OHUM"),("srunit","OHSQ"),("tempunit","C")))
    t.units==units || throw(ArgumentError("native scalar decoded STF units disagree with captured XML"))
    public=_stf_child(root,"public";required=true)
    materials=Dict((Symbol(n.name),n.attributes["name"])=>n for
        n in _stf_child(public,"materials";required=true).children if haskey(n.attributes,"name"))
    for (current,expected) in ((t.materials,materials),
            (t.variables,_stf_named(_stf_child(public,"variables"))),
            (t.models,_stf_named(_stf_child(public,"metal_model_defs"))),
            (t.meshes,_stf_named(_stf_child(public,"mesh_defs"))),
            (t.biases,_stf_named(_stf_child(public,"bias_defs"))))
        keys(current)==keys(expected) && all(key->_sonnet_scalar_same_technology_node(current[key],expected[key]),keys(expected)) ||
            throw(ArgumentError("native scalar decoded STF identities disagree with captured XML"))
    end
    _sonnet_scalar_same_technology_node(t.stack,_stf_child(public,"stackup";required=true)) ||
        throw(ArgumentError("native scalar decoded STF stack disagrees with captured XML"))
    return nothing
end

# Internal RF staging hook. The caller's effective p may be materialized or
# intentionally detached, but its original SON records must match these bytes.
function _sonnet_scalar_parent(linked::SonnetLinkedProject;
        max_bytes::Integer=64*1024^2)
    limit=_stf_limit(max_bytes,"max_bytes")
    _enforce_payload_limit(_checked_payload_sum("native scalar captured parent",
        BigInt(16)*(ncodeunits(linked.raw)+ncodeunits(linked.technology.raw))+
        2BigInt(linked.technology.limits.estimated_storage_bytes)+4096),
        limit,"native scalar captured parent","max_bytes")
    bytes2hex(SHA.sha256(linked.raw))==linked.sha256 ||
        throw(ArgumentError("native scalar retained SON source was mutated"))
    _sonnet_scalar_technology_baseline(linked.technology,limit)
    source=SonnetScalarSource(linked.source,Vector{UInt8}(codeunits(linked.raw)),linked.sha256)
    _sonnet_scalar_parent_records(source,linked.records)
    technology=linked.technology
    dependency=SonnetScalarSource(technology.source,Vector{UInt8}(codeunits(technology.raw)),technology.sha256)
    return (;_parent_source=source,_conversion="static-stf",_conversion_sources=[dependency])
end

"""Stage CSV scalar dependencies for the supported static linked-STF subset.
The captured SON/STF bytes and decoded identities are checked and owned;
`technology_variables` apply during explicit static materialization. CSV
domain policy and resource limits follow `sonnet_scalar_files`. Unsupported
STF inheritance, interpolation, etch and RPV still reject."""
function sonnet_scalar_files(linked::SonnetLinkedProject;
        technology_variables=Dict{String,Float64}(),max_storage::Integer=64*1024^2,kw...)
    parent=_sonnet_scalar_parent(linked;max_bytes=max_storage)
    p=sonnet_materialize_project(linked;technology_variables)
    return sonnet_scalar_files(p;max_storage,parent...,kw...)
end
