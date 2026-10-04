# Complete matrix buildup contracts for STEP-REPEAT FLIP. Primary ODB++
# Update 3 pp67-71 and148: physical buildup is separate from drill/rout/
# documentation tails, and partial axial spans use canonical row order.
const _ODB_PHYSICAL_LAYER_TYPES=Set(["SIGNAL","POWER_GROUND","MIXED","DIELECTRIC",
    "SOLDER_MASK","SOLDER_PASTE","SILK_SCREEN","COMPONENT","MASK","CONDUCTIVE_PASTE"])

# Entity paths remain strictly lowercase. The primary's matrix example
# uses uppercase references to legal lowercase entities; canonicalize only
# those logical references and keep raw matrix records for output metadata.
function _odb_reference_name(value)
    value isa AbstractString || throw(ArgumentError("ODB entity reference must be a string"))
    return _odb_name(any(c->'A'<=c<='Z',value) ? lowercase(value) : value)
end
function _odb_canonical_reference_records(records;layers=false,max_bytes)
    payload=_checked_array_payload_bytes(UInt8,16,length(records))
    fields=layers ? ("NAME","START_NAME","END_NAME") : ("NAME",)
    for record in records
        if any(field->any(c->'A'<=c<='Z',get(record,field,"")),fields)
            payload=_checked_payload_sum("ODB canonical references",payload,256,
                _checked_array_payload_bytes(UInt8,128,length(record)))
        end
    end
    _enforce_payload_limit(payload,max_bytes,"ODB canonical matrix references","max_bytes")
    result=typeof(records)()
    for record in records
        name=_odb_reference_name(get(record,"NAME",""));changed=name!=get(record,"NAME","")
        out=changed ? copy(record) : record
        if layers
            for field in ("START_NAME","END_NAME")
                value=get(record,field,"");isempty(value) && continue
                canonical=_odb_reference_name(value)
                if canonical!=value
                    out===record && (out=copy(record));out[field]=canonical
                end
            end
        end
        out===record || (out["NAME"]=name)
        push!(result,out)
    end
    return result
end

function _odb_ordered_matrix_records(records,index;max_bytes)
    _enforce_payload_limit(_checked_array_payload_bytes(UInt8,128,length(records)),max_bytes,
        "ODB matrix index validation","max_bytes")
    indices=Int[]
    for record in records
        value=tryparse(Int,get(record,index,""))
        value!==nothing && value>0 || throw(ArgumentError("ODB matrix $index must be a positive integer"))
        push!(indices,value)
    end
    length(unique(indices))==length(indices) || throw(ArgumentError("duplicate ODB matrix $index"))
    return records[sortperm(indices)]
end

function _odb_flip_layer_map(records;override=nothing,max_bytes)
    override===nothing || override isa AbstractDict || throw(ArgumentError("flip_layer_map must be a dictionary"))
    _enforce_payload_limit(_checked_payload_sum("ODB FLIP map",_checked_array_payload_bytes(UInt8,1024,length(records)),
        _checked_array_payload_bytes(UInt8,256,override===nothing ? 0 : length(override))),max_bytes,
        "ODB complete FLIP layer map","max_bytes")
    ordered=_odb_canonical_reference_records(_odb_ordered_matrix_records(records,"ROW";max_bytes);layers=true,max_bytes)
    byname=Dict(_odb_name(get(r,"NAME",""))=>r for r in ordered)
    length(byname)==length(records) || throw(ArgumentError("duplicate ODB layer names"))
    physical=[r["NAME"] for r in ordered if get(r,"CONTEXT","BOARD")=="BOARD" &&
        get(r,"TYPE","") in _ODB_PHYSICAL_LAYER_TYPES]
    isempty(physical) && throw(ArgumentError("ODB FLIP requires a physical BOARD buildup"))
    position=Dict(n=>i for (i,n) in enumerate(physical));mapping=Dict(n=>n for n in keys(byname))
    for (source,target) in zip(physical,reverse(physical))
        for field in ("TYPE","POLARITY")
            default=field=="POLARITY" ? "POSITIVE" : ""
            get(byname[source],field,default)==get(byname[target],field,default) ||
                throw(ArgumentError("ODB FLIP requires symmetric layer types and polarities"))
        end
        mapping[source]=target
    end
    function span(record)
        a,b=get(record,"START_NAME",""),get(record,"END_NAME","")
        isempty(a) && (a=first(physical));isempty(b) && (b=last(physical))
        haskey(position,a) && haskey(position,b) && position[a]<=position[b] ||
            throw(ArgumentError("invalid ODB drill/rout START_NAME/END_NAME span"))
        return (a,b)
    end
    explicit=Dict{String,String}()
    if override!==nothing
        for (source,target) in override
            source isa AbstractString && target isa AbstractString ||
                throw(ArgumentError("flip_layer_map keys and values must be layer names"))
            s,t=_odb_reference_name(source),_odb_reference_name(target)
            haskey(byname,s) && haskey(byname,t) || throw(ArgumentError("unknown ODB flip_layer_map layer"))
            haskey(explicit,s) && throw(ArgumentError("duplicate case-folded ODB flip_layer_map source"))
            explicit[s]=t
        end
    end
    for (name,record) in byname
        if get(record,"TYPE","") in ("DRILL","ROUT") && get(record,"CONTEXT","BOARD")=="BOARD"
            get(record,"POLARITY","POSITIVE")=="POSITIVE" ||
                throw(ArgumentError("ODB drill/rout layers require positive polarity"))
            a,b=span(record);expected=(mapping[b],mapping[a])
            candidates=[n for (n,r) in byname if get(r,"TYPE","")==record["TYPE"] &&
                get(r,"CONTEXT","BOARD")=="BOARD" && get(r,"ADD_TYPE","")==get(record,"ADD_TYPE","") && span(r)==expected]
            if haskey(explicit,name)
                explicit[name] in candidates || throw(ArgumentError("ODB flip_layer_map does not reflect the drill/rout span"))
                mapping[name]=explicit[name]
            else
                length(candidates)==1 || throw(ArgumentError("missing or ambiguous reflected ODB drill/rout layer; provide flip_layer_map"))
                mapping[name]=only(candidates)
            end
        elseif haskey(explicit,name)
            target=explicit[name]
            haskey(position,name) && target!=mapping[name] &&
                throw(ArgumentError("flip_layer_map cannot change the physical buildup reversal"))
            get(byname[target],"TYPE","")==get(record,"TYPE","") &&
                get(byname[target],"CONTEXT","BOARD")==get(record,"CONTEXT","BOARD") &&
                get(byname[target],"POLARITY","POSITIVE")==get(record,"POLARITY","POSITIVE") ||
                throw(ArgumentError("incompatible ODB auxiliary flip_layer_map layers"))
            mapping[name]=target
        end
    end
    all(n->mapping[mapping[n]]==n,keys(mapping)) || throw(ArgumentError("ODB flip_layer_map must be an involution"))
    return mapping
end

# JSON1 buffers complete strings before writing to a supplied IO. Metadata
# therefore uses its limited actual domain (strings, dictionaries, arrays,
# booleans/null) with an exact size pass before allocating an output buffer.
@noinline _odb_json_limit()=throw(ArgumentError("ODB JSON metadata exceeds max_bytes"))
function _odb_json_size(value,budget,depth=0)
    depth<=64 || throw(ArgumentError("ODB JSON metadata hierarchy too deep"))
    if value isa AbstractString
        budget>=2 || _odb_json_limit();count=2
        for b in codeunits(value)
            extra=b in (0x22,0x5c,0x08,0x0c,0x0a,0x0d,0x09) ? 2 : b<0x20 ? 6 : 1
            extra<=budget-count || _odb_json_limit();count+=extra
        end
        return count
    elseif value isa AbstractDict
        budget>=2 || _odb_json_limit();count=2;first=true
        for (key,child) in value
            key isa AbstractString || throw(ArgumentError("ODB JSON metadata keys must be strings"))
            extra=first ? 1 : 2 # colon and optional comma
            extra<=budget-count || _odb_json_limit();count+=extra;first=false
            count+=_odb_json_size(key,budget-count,depth+1)
            count+=_odb_json_size(child,budget-count,depth+1)
        end
        return count
    elseif value isa AbstractVector
        budget>=2 || _odb_json_limit();count=2;first=true
        for child in value
            if !first;count<budget || _odb_json_limit();count+=1;end
            first=false;count+=_odb_json_size(child,budget-count,depth+1)
        end
        return count
    elseif value===nothing || value isa Bool
        count=value===false ? 5 : 4;count<=budget || _odb_json_limit();return count
    end
    throw(ArgumentError("unsupported ODB JSON metadata value"))
end
function _odb_json_emit(io::IOBuffer,value)
    if value isa AbstractString
        write(io,0x22)
        for b in codeunits(value)
            if b in (0x22,0x5c)
                write(io,0x5c,b)
            elseif b in (0x08,0x0c,0x0a,0x0d,0x09)
                letter=b==0x08 ? 0x62 : b==0x0c ? 0x66 : b==0x0a ? 0x6e : b==0x0d ? 0x72 : 0x74
                write(io,0x5c,letter)
            elseif b<0x20
                write(io,0x5c,0x75,0x30,0x30)
                digits=codeunits("0123456789abcdef");write(io,digits[Int(b>>4)+1],digits[Int(b&0x0f)+1])
            else
                write(io,b)
            end
        end
        write(io,0x22)
    elseif value isa AbstractDict
        write(io,0x7b);first=true
        for (key,child) in value
            first || write(io,0x2c);first=false
            _odb_json_emit(io,key);write(io,0x3a);_odb_json_emit(io,child)
        end
        write(io,0x7d)
    elseif value isa AbstractVector
        write(io,0x5b);first=true
        for child in value
            first || write(io,0x2c);first=false;_odb_json_emit(io,child)
        end
        write(io,0x5d)
    else
        write(io,value===nothing ? "null" : value ? "true" : "false")
    end
end
function _odb_bounded_json(value,max_bytes)
    limit=_validated_resource_limit("max_bytes",max_bytes)
    _enforce_payload_limit(256,limit,"ODB JSON writer bookkeeping","max_bytes")
    bytes=_odb_json_size(value,(limit-256)÷3)
    io=IOBuffer(Vector{UInt8}(undef,bytes);read=false,write=true,truncate=true,maxsize=bytes)
    _odb_json_emit(io,value)
    return String(take!(io))
end
